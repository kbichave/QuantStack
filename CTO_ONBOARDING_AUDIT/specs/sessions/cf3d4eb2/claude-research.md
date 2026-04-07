# Research Findings: Phase 6 Execution Layer

---

## Part 1: Codebase Analysis

### Execution Layer Structure

```
src/quantstack/execution/
├── order_lifecycle.py           # OMS state machine (NEW → FILLED)
├── risk_gate.py                 # Hard risk controls
├── paper_broker.py              # Paper trading with slippage model
├── execution_monitor.py         # Deterministic exit rules (SL/TP/trailing/time)
├── portfolio_state.py           # PostgreSQL-backed positions
├── tick_executor.py             # Hot-path async executor
├── trade_service.py             # Business logic wrapper
├── alpaca_broker.py             # Live broker
├── broker_factory.py
├── hook_registry.py
├── kill_switch.py
├── microstructure_pipeline.py
├── price_feed.py
├── risk_state.py
├── shadow_comparator.py
├── signal_cache.py
├── strategy_breaker.py
└── adapters/etrade/             # E*Trade integration
```

### Order Lifecycle (`order_lifecycle.py`)

**State Machine:**
```
NEW → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED (terminal)
                                                   → REJECTED (terminal)
                                                   → CANCELLED (terminal)
                                                   → EXPIRED (terminal)
```

**Key Fields on `Order`:**
- `order_id`, `symbol`, `side`, `quantity`, `signal_id`
- `arrival_price` — price at signal fire time (TCA benchmark)
- `exec_algo: ExecAlgoOMS` — IMMEDIATE | TWAP | VWAP | POV
- `filled_quantity`, `fill_price`

**Algo Selection (Lines 455-478):**
```python
def _select_exec_algo(quantity, adv):
    pct_adv = quantity / adv
    if pct_adv < 0.002: return IMMEDIATE   # < 0.2% ADV
    if pct_adv < 0.01:  return TWAP        # 0.2–1% ADV
    if pct_adv < 0.05:  return VWAP        # 1–5% ADV
    return POV                              # > 5% ADV
```

**Critical Gap:** Algo is selected but never executed — everything goes through as IMMEDIATE.

**TCA Integration (Lines 644-684):** After fill, computes `implementation_shortfall_bps = (fill_price - arrival_price) / arrival_price * 10_000`. Persisted to `TCAStore` but NOT fed back.

### Risk Gate (`risk_gate.py`)

**Layered checks (Lines 293-688):**
1. Daily halt sentinel check (fast path)
2. Restricted symbol check
3. Volume validation
4. Daily loss limit (-2% halts trading for the day)
5. Liquidity check (ADV < min_daily_volume → warn only, no reject)
6. Participation cap (scales down if > 1% ADV)
7. Execution quality scalar (from DB)
8. Macro stress scalar
9. Options-specific: DTE bounds, premium-at-risk checks
10. Equities: position size, gross exposure

**Limits:**
- `max_position_pct: 0.10` (10% equity per symbol)
- `max_position_notional: $20,000`
- `max_gross_exposure_pct: 1.50`
- `daily_loss_limit_pct: 0.02`
- `min_daily_volume: 500,000`
- `max_participation_pct: 0.01`

### Paper Broker (`paper_broker.py`, 168 lines)

**Fill Price Model (Lines 243-276):**
- Half-spread: 2 bps assumption
- Square-root impact: `k=5`, so 1% ADV ≈ 5 bps impact
- `fill_price = ref_price + direction * (spread_cost + impact_cost)`
- Partial fill: capped at 2% of daily volume per order

**Key Gap:** Single instant fill — no time-slicing, no volume profile simulation.

### Execution Monitor (`execution_monitor.py`)

**Rule Evaluation Order (Lines 146-201):**
1. Exit pending check (skip if already exiting)
2. Kill switch (exit all)
3. Hard stop-loss
4. Take profit
5. Trailing stop (HWM-based, ATR-scaled)
6. Time stop (max bars × bar timeframe)
7. Intraday flatten (15:55 ET for INTRADAY positions)

**No options-specific rules** — all 7 rules are equity-centric.

**Periodic tasks:** Price poll every 5s, reconciliation loop, circuit breaker (kill switch if DB unreachable > 60s).

### Existing Cost Models

**TCA Engine (`src/quantstack/core/execution/tca_engine.py`):**
- Pre-trade forecast with Almgren-Chriss coefficients
- Components: spread_cost, market_impact, timing_cost, commission
- Default coefficients: `eta=0.142`, `gamma=0.314`, `beta=0.60`
- Recommends algo based on participation rate

**Almgren-Chriss (`src/quantstack/core/execution/almgren_chriss.py`):**
- Full cost breakdown: permanent + temporary impact + timing risk
- Optimal trajectory computation across N time slices
- **Has calibration from fills** (Lines 219-267): fits gamma/eta from historical fills via least-squares, requires ≥10 fills

**Options Slippage (`src/quantstack/core/options/slippage.py`):**
- `SpreadBasedSlippage` class with spread crossing + sqrt market impact + urgency premium

### Database Schema

**Orders table:** `order_id, symbol, side, quantity, signal_id, arrival_price, exec_algo, status, filled_quantity, fill_price, created_at, submitted_at, filled_at, expires_at`

**Fills table:** `order_id (PK), symbol, side, requested_quantity, filled_quantity, fill_price, slippage_bps, commission, partial, rejected, filled_at, session_id`
- **Critical Gap:** `order_id` is PK — only one fill record per order. Cannot store multiple partial fills.

**Positions table:** `symbol (PK), quantity, avg_cost, side, strategy_id, instrument_type, time_horizon, stop_price, target_price, trailing_stop, option fields...`

**Closed trades table:** `symbol, entry_price, exit_price, realized_pnl, strategy_id, exit_reason`
- **No tax lot tracking**, no wash sale flags, no cost basis adjustment

### Options Engine (`src/quantstack/core/options/engine.py`)

- `price_option_dispatch` (Lines 42-102): auto-selects vollib (European) or financepy (American)
- `compute_greeks_dispatch` (Lines 293-364): delta, gamma, theta, vega, rho with interpretations
- `compute_iv_dispatch` (Lines 367-400+): Newton-Raphson IV solver

### Testing Setup

**Framework:** pytest

**Test directories:**
```
tests/
├── unit/execution/          # TCA, slippage budget, algo feedback
├── unit/test_execution_monitor*.py
├── core/execution/          # TCA feedback, recalibration, quality, beta coefficients
├── quant_pod/               # Order lifecycle, paper broker, execution MCP
└── integration/             # Execution monitor integration
```

**Key fixtures:**
- `MonitoredPosition` builder with defaults (SPY, long, 100 shares, stops/targets)
- `PaperBroker` with in-memory SQLite context
- `OrderRequest` builder helper

### Signal → Fill Data Flow

```
TradeSignal (from LLM)
  → RiskGate.check()
  → OrderLifecycle.submit() [selects exec algo]
  → broker.execute(OrderRequest)
  → OrderLifecycle.record_fill() [compute IS]
  → PortfolioState.upsert_position()
  → TCAStore.save_result()
```

### Summary of Gaps

| Area | Current State | Gap |
|------|--------------|-----|
| TWAP/VWAP | Algo selected in OMS, never executed | No time-slicing, no child orders |
| TCA Feedback | IS computed post-fill, stored | Not fed back to pre-trade forecast |
| Partial Fills | Paper broker models partials | `fills` table uses order_id as PK (1 row per order) |
| Liquidity | ADV check (warn only) | No depth, spread, time-of-day |
| SEC Compliance | Zero | No wash sale, PDT, tax lots, margin |
| Audit Trail | Basic fill logging | No NBBO, venue, algo rationale |
| Options Monitoring | No options exit rules | All 7 rules equity-centric |
| Slippage Feedback | Almgren-Chriss has calibration code | Not wired to live recalibration |

---

## Part 2: TWAP/VWAP Execution Algorithms (Web Research)

### TWAP Implementation

**Core approach:** Divide total quantity evenly across N time buckets. `child_qty = total_qty / N`.

**Child order generation pattern:**
```python
pct_to_complete = elapsed_time / total_duration * 100
should_have_executed = round(pct_to_complete / 100 * total_size, 3)
execution_slice = should_have_executed - executed_so_far - exposed_size
```

**Randomization to avoid detection:**
- +/-15-30% jitter on scheduled time
- +/-10-20% variation on child sizes
- Alternate limit vs marketable limit orders
- Skip occasional slices and catch up next window

**Partial fill handling:**
- Track `exposed_size` (open unfilled children) separately from `executed_so_far`
- Minimum slice threshold (~1% of total) to avoid dust orders
- Cancel stale unfilled children before new slices

### VWAP Implementation

**Core formula:** `P_VWAP = Sum(P_j * Q_j) / Sum(Q_j)`

**Intraday volume profile (U-shaped):**
- 9:30-10:00: ~10-15% of daily volume
- 10:00-11:30: ~4-6% per 30-min bucket
- 11:30-13:30: trough at 3-5% per bucket
- 13:30-15:30: ramp to 5-8%
- 15:30-16:00: ~12-18% of daily volume

**Building the profile:** Average normalized volume distribution across prior 10-20 trading days per 5-min bucket. Exponential weighting for recency.

**Volume-weighted sizing:**
```
child_qty[i] = total_qty * (hist_volume_pct[i] / sum(remaining_pcts))
```

**Favorability check:** Buy only when `ask < VWAP * (1 - threshold_bps/10000)`. Sell only when `bid > VWAP * (1 + threshold_bps/10000)`.

### Paper Trading Simulation

Alpaca's paper trading is too optimistic (no market impact, no queue position). Build custom simulation:

**Fill against historical bars:**
- Participation cap: can't fill more than X% of bar volume (1-5%)
- Price: use bar VWAP + noise within [low, high]
- Partial fill probability increases with order_size / bar_volume

### Parent/Child State Machine

**Parent:** `PENDING → ACTIVE → COMPLETING → COMPLETED` (or `→ CANCELLED`)
**Child:** `PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED` (or `→ CANCELLED/EXPIRED/REJECTED`)

**Key invariant:** `sum(child.filled_qty) == parent.filled_qty` at all times.

**Failure handling:**
- REJECTED: check reason, reduce size 50%, retry up to max_attempts
- Timeout: cancel child, redistribute to subsequent slices
- API error: exponential backoff (1s, 2s, 4s)
- 3+ consecutive failures: pause parent, alert

**Completion criteria:**
- `filled_qty >= total_qty * 0.995` → COMPLETED
- End time reached → submit final aggressive market order or COMPLETED with shortfall

### Performance Measurement

**Implementation Shortfall (gold standard):**
```
IS_bps = ((avg_fill_price / arrival_price) - 1) * 10_000  # for buys
```

Components: explicit costs + delay cost + market impact + opportunity cost.

**VWAP Slippage:**
```
vwap_slippage_bps = ((execution_vwap - benchmark_vwap) / benchmark_vwap) * 10_000
```

**Recommended `algo_performance` table:**
```sql
CREATE TABLE algo_performance (
    id SERIAL PRIMARY KEY,
    parent_order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    algo_type TEXT NOT NULL,
    total_qty NUMERIC NOT NULL,
    filled_qty NUMERIC NOT NULL,
    arrival_price NUMERIC NOT NULL,
    avg_fill_price NUMERIC NOT NULL,
    benchmark_vwap NUMERIC,
    decision_time TIMESTAMPTZ NOT NULL,
    first_fill_time TIMESTAMPTZ,
    last_fill_time TIMESTAMPTZ,
    scheduled_end_time TIMESTAMPTZ NOT NULL,
    implementation_shortfall_bps NUMERIC,
    vwap_slippage_bps NUMERIC,
    delay_cost_bps NUMERIC,
    market_impact_bps NUMERIC,
    num_children INT NOT NULL,
    num_children_filled INT NOT NULL,
    num_children_failed INT NOT NULL,
    max_participation_rate NUMERIC,
    actual_participation_rate NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Alpaca-Specific Constraints

| Constraint | Detail |
|-----------|--------|
| No native TWAP/VWAP | Must build execution algo layer on top of basic order types |
| Order types | Market, Limit, Stop, Stop-Limit, Bracket, OCO, OTO, Trailing Stop |
| TIF options | `day`, `gtc`, `ioc`, `fok`, `opg`, `cls` |
| IOC for children | Use `ioc` for aggressive catch-up slices |
| Paper fills | 10% chance of random partial; no slippage/impact simulation |

### Cross-Validated Recommendations

1. **Start with TWAP** — simpler, no volume profile dependency. Add VWAP once child order management is proven.
2. **Use limit orders for children** at/inside NBBO. `ioc` TIF for aggressive catch-up slices.
3. **Build custom fill simulator** for paper mode — Alpaca paper is too optimistic.
4. **Parent/child state machine is the hardest part** — get this right first.
5. **Log everything to `algo_performance`** from day one.
6. **Participation rate caps:** 2% for small/mid-cap, 5% for large-cap, never exceed 10%.
7. **Randomize timing and sizing** (+/-20% jitter) to avoid predatory HFT detection.
