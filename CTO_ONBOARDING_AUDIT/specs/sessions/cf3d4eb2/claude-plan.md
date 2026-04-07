# Phase 6: Execution Layer Completion — Implementation Plan

## 1. Context and Motivation

QuantStack is an autonomous trading system built on three LangGraph StateGraphs (Research, Trading, Supervisor) running as Docker services. The execution layer lives in `src/quantstack/execution/` and currently handles order lifecycle management, risk gating, paper broker simulation, and deterministic exit monitoring.

The CTO audit rated execution architecture B+ (well-designed multi-layer enforcement) but the Quant Scientist audit rated execution reality D. The gap: the OMS selects TWAP/VWAP/POV algorithms based on ADV thresholds but executes everything as a single immediate fill. There is zero SEC compliance, no partial fill tracking, no TCA feedback, and all exit rules are equity-centric.

This plan covers 9 items that transform the execution layer from prototype to production-grade:

| # | Item | Severity | Effort |
|---|------|----------|--------|
| 6.1 | TCA EWMA Feedback Loop | CRITICAL | 2 days |
| 6.2 | Partial Fill Tracking | HIGH | 1 day |
| 6.3 | Real TWAP/VWAP Execution | CRITICAL | 5 days |
| 6.4 | Liquidity Model | CRITICAL | 3 days |
| 6.5 | SEC Compliance (PDT, Wash Sale, Tax Lots) | CRITICAL | 5-7 days |
| 6.6 | Best Execution Audit Trail | HIGH | 2 days |
| 6.7 | Options Monitoring Rules | HIGH | 2 days |
| 6.8 | Slippage Model Enhancement | HIGH | 2 days |
| 6.9 | Borrowing/Funding Cost Model | HIGH | 1 day |

**Account constraint:** Account is below $25K, making PDT enforcement critical path (FINRA 4210 violation risk).

**No equity shorts currently** — funding cost model (6.9) scoped to margin interest only.

---

## 2. Current Architecture (What Exists)

### 2.1 Order Lifecycle (`order_lifecycle.py`)

The OMS implements an explicit state machine: `NEW → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED` (plus terminal states REJECTED, CANCELLED, EXPIRED).

Key fields on `Order`: `order_id`, `symbol`, `side`, `quantity`, `arrival_price` (mid at signal time), `exec_algo` (IMMEDIATE/TWAP/VWAP/POV), `filled_quantity`, `fill_price`.

Algo selection logic at lines 455-478 chooses based on `quantity / adv`:
- < 0.2% ADV → IMMEDIATE
- 0.2-1% → TWAP
- 1-5% → VWAP
- \> 5% → POV

After fill, computes implementation shortfall in bps (lines 139-153) and persists to `TCAStore`. But the shortfall is never fed back into pre-trade forecasts.

### 2.2 Risk Gate (`risk_gate.py`)

Layered pre-trade checks (lines 293-688):
1. Daily halt sentinel (survives restarts via file)
2. Restricted symbol check
3. Volume validation
4. Daily loss limit (-2% halts all trading)
5. Liquidity check (ADV < 500K → warn only)
6. Participation cap (> 1% ADV → scale down)
7. Execution quality scalar
8. Macro stress scalar
9. Options-specific: DTE bounds (7-60), premium-at-risk (2% per position, 8% total)
10. Equity: position size (10% or $20K cap), gross exposure (150%)

**Gap:** No PDT check, no wash sale awareness, no margin calculation.

### 2.3 Paper Broker (`paper_broker.py`)

Fills market orders with half-spread (2 bps) + square-root impact (k=5, so 1% ADV ≈ 5 bps). Partial fill modeled as cap at 2% of daily volume. Single instant fill — no time-slicing.

### 2.4 Cost Models

**TCA Engine** (`core/execution/tca_engine.py`): Pre-trade forecast with Almgren-Chriss coefficients (eta=0.142, gamma=0.314, beta=0.60). Components: spread, market impact, timing cost, commission.

**Almgren-Chriss** (`core/execution/almgren_chriss.py`): Full cost breakdown with optimal trajectory computation. Has calibration-from-fills (lines 219-267) requiring ≥10 fills via least-squares.

**Options Slippage** (`core/options/slippage.py`): `SpreadBasedSlippage` with spread crossing + sqrt impact + urgency premium.

### 2.5 Execution Monitor (`execution_monitor.py`)

Deterministic exit rules evaluated on price ticks:
1. Kill switch → exit all
2. Hard stop-loss
3. Take profit
4. Trailing stop (HWM-based, ATR-scaled)
5. Time stop
6. Intraday flatten (15:55 ET)

All rules are equity-centric. No options-specific rules.

### 2.6 Database Schema

**`orders`:** Full order record with `order_id` PK, arrival_price, exec_algo, status, fill info.

**`fills`:** `order_id` as PK (one row per order). Fields: symbol, side, quantities, fill_price, slippage_bps, commission.

**`positions`:** `symbol` as PK with entry info, stops/targets, option fields.

**`closed_trades`:** Realized P&L with strategy_id, regime, exit_reason.

### 2.7 Options Engine (`core/options/engine.py`)

`compute_greeks_dispatch` (lines 293-364): delta, gamma, theta, vega, rho with interpretations and risk metrics. Backend chain: vollib → financepy → internal.

### 2.8 Testing

Framework: pytest. Tests in `tests/unit/execution/`, `tests/core/execution/`, `tests/quant_pod/`, `tests/integration/`. Key fixtures: `MonitoredPosition` builder, `PaperBroker` with in-memory SQLite, `OrderRequest` helper.

---

## 3. Implementation Order

Dependencies drive the order. SEC compliance comes before TCA optimization because the account is below $25K — every trade without PDT enforcement is regulatory risk. Audit trail ships early for IMMEDIATE orders, then extends to child fills when TWAP/VWAP lands.

```
6.2 Partial Fill Tracking (schema foundation)
  ↓
6.5 SEC Compliance (PDT is critical path — regulatory risk)
  ↓
6.6 Audit Trail (compliance value for IMMEDIATE orders now)
  ↓
6.1 TCA EWMA Feedback Loop (uses fill legs for accurate cost)
  ↓
6.3 TWAP/VWAP (largest item, produces child fill legs, extends 6.6)
  ↓
6.4 Liquidity Model (enhances risk gate, informs algo scheduling)
  ↓
6.8 Slippage Enhancement (calibrated from TCA EWMA in 6.1)
  ↓
6.7 Options Monitoring (depends on Phase 2 Greeks in risk gate)
  ↓
6.9 Funding Costs (simplest, lowest dependency)
```

---

## 4. Section: Partial Fill Tracking (6.2)

### 4.1 Problem

The `fills` table uses `order_id` as primary key — one row per order. When partial fills arrive, previous fill data is overwritten. You cannot reconstruct average fill price, execution VWAP, or fill trajectory.

### 4.2 Schema Change

Add a new `fill_legs` table alongside the existing `fills` table. The existing `fills` table remains as a summary view — backward compatible, existing queries and code that read `fills` continue to work unmodified.

```python
@dataclass
class FillLeg:
    leg_id: int          # auto-increment PK
    order_id: str        # FK to orders
    leg_sequence: int    # 1, 2, 3... per order
    quantity: int
    price: float
    timestamp: datetime
    venue: str | None    # for audit trail (6.6)
```

Table: `fill_legs` with `(order_id, leg_sequence)` as unique constraint. Index on `order_id` for fast leg lookups.

### 4.3 Fill Recording Changes

In `paper_broker.py` and `alpaca_broker.py`, after receiving a fill:
1. Insert a row into `fill_legs` with the fill details
2. Compute running VWAP from all legs for this order: `sum(qty_i * price_i) / sum(qty_i)`
3. Update the `fills` summary row with the cumulative VWAP and total filled quantity

This dual-write ensures backward compatibility. Old code reads `fills`; new code reads `fill_legs` when it needs granular data.

### 4.4 VWAP Computation Helper

Add a function that takes an `order_id` and returns the volume-weighted average price across all legs. This is used by TCA (6.1), algo performance tracking (6.3), and audit trail (6.6).

### 4.5 Migration

Schema migration adds the `fill_legs` table. No existing data migration needed — historical orders predate this feature and don't need legs retroactively.

---

## 5. Section: TCA EWMA Feedback Loop (6.1)

### 5.1 Problem

The TCA engine computes pre-trade cost forecasts using static Almgren-Chriss coefficients. Post-trade, it measures implementation shortfall. But the realized costs never feed back — if slippage is consistently 2x forecast, the next trade uses the same stale parameters.

### 5.2 EWMA Recalibration Design

After every fill completes, update the cost model parameters using exponential weighted moving average:

```
forecast_new = α * realized_cost + (1 - α) * forecast_old
```

Where α = 0.1 (spec requirement). This runs per symbol, per time-of-day bucket (morning/midday/afternoon/close).

**Conservative multiplier:** Until a symbol has accumulated 50 fills, apply a 2x multiplier to the forecast. This prevents undertrained models from underestimating costs. The multiplier linearly decays from 2.0 to 1.0 as fills accumulate from 0 to 50.

### 5.3 Storage

New `tca_parameters` table:

```python
@dataclass
class TCAParameters:
    symbol: str           # e.g., "SPY"
    time_bucket: str      # "morning" | "midday" | "afternoon" | "close"
    ewma_spread_bps: float
    ewma_impact_bps: float
    ewma_total_bps: float
    sample_count: int
    last_updated: datetime
```

Primary key: `(symbol, time_bucket)`. Upserted after every fill.

Time buckets:
- morning: 9:30–11:00
- midday: 11:00–14:00
- afternoon: 14:00–15:30
- close: 15:30–16:00

### 5.4 Integration Points

**After fill (hook in `order_lifecycle.py`):**
1. Compute realized cost components from fill legs (6.2)
2. Look up current EWMA parameters for symbol + time bucket
3. Apply EWMA update
4. Upsert to `tca_parameters`

**Before trade (in `tca_engine.pre_trade_forecast()`):**
1. Look up EWMA parameters for symbol + time bucket
2. If found and sample_count ≥ 50: use EWMA values directly
3. If found but sample_count < 50: use EWMA values × conservative multiplier
4. If not found: fall back to default A-C coefficients (existing behavior)

**Position sizing:**
The pre-trade forecast's `total_expected_bps` feeds into position sizing logic. Higher estimated costs → smaller positions. This is the automatic feedback loop — poor execution quality self-corrects by reducing future position sizes.

### 5.5 Relationship to Existing A-C Calibration

The EWMA layer is separate from the existing least-squares calibration in `almgren_chriss.py`. The EWMA provides fast, per-fill updates. The A-C calibration could run periodically (weekly/monthly) with accumulated fill data for deeper coefficient recalibration. This plan implements only the EWMA layer per the interview decision.

---

## 6. Section: SEC Compliance (6.5)

### 6.1 Overview

Four compliance domains, each with different enforcement models:

| Domain | Enforcement | Where |
|--------|------------|-------|
| PDT (FINRA 4210) | **Hard block** pre-trade | Risk gate |
| Wash Sale (§1091) | Post-trade accounting + pre-trade warning | Fill hooks + risk gate warning |
| Tax Lots (Form 8949) | Post-trade accounting | Fill hooks |
| Margin (Reg T) | Pre-trade check | Risk gate |

### 6.2 PDT Enforcement

**Critical path** — account is below $25K.

**Definition:** A "day trade" is opening and closing the same symbol on the same trading day. A "pattern day trader" has ≥4 day trades in a rolling 5-business-day window AND account equity < $25K.

**Implementation:**

New `day_trades` table:

```python
@dataclass
class DayTrade:
    id: int
    symbol: str
    open_order_id: str
    close_order_id: str
    trade_date: date        # business day
    quantity: int
    account_equity: float   # at time of close
```

**Pre-trade check in risk gate** (new check, inserted after daily loss limit check):
1. Query `day_trades` for rolling 5-business-day window
2. If count ≥ 3: this would be the 4th day-trade
3. Check if this order would create a new day trade (same symbol opened today, now closing)
4. If yes AND account_equity < $25K → REJECT with reason "PDT: 4th day trade blocked (account < $25K)"

**Post-trade recording:**
After every fill, check if this fill closes a position that was opened on the same business day. If so, insert a `day_trades` record.

**Edge cases:**
- Multiple round-trips on same symbol same day: each counts as a separate day trade
- Partial fills that close an intra-day position: counts as a day trade on the fill that brings quantity to zero or reverses
- Options day trades: opening and closing the same contract on the same day counts. Match on the full OCC contract symbol (e.g., `SPY240119C00450000`), NOT the underlying symbol. Two different SPY option contracts are not the same day trade.

### 6.3 Wash Sale Tracking

**Definition:** Selling a security at a loss and buying substantially identical security within 30 calendar days before or after the sale.

**Two-phase detection (fixes look-forward problem):**

You cannot check at time of sale whether the same symbol will be bought within the next 30 days. The correct approach is two-phase:

**Phase 1 — On every realized loss sale:**
1. Query: was same symbol bought within 30 calendar days *before* this sell? If yes → immediate wash sale (retroactive detection).
2. Insert a `pending_wash_losses` record: `(symbol, loss_amount, sell_date, sell_order_id, window_end = sell_date + 30 days)`. This flags the loss as potentially washable by a future buy.

**Phase 2 — On every buy:**
1. Query `pending_wash_losses` for same symbol where `window_end >= today`. If found → this buy triggers wash sale treatment on the pending loss.
2. Retroactively apply: disallow the loss, adjust cost basis of the new shares by the disallowed amount.
3. Mark the `pending_wash_losses` entry as resolved.

New tables:

```python
@dataclass
class PendingWashLoss:
    id: int
    symbol: str
    loss_amount: float       # the realized loss
    sell_order_id: str
    sell_date: date
    window_end: date         # sell_date + 30 calendar days
    resolved: bool           # True when matched by a buy
    resolved_by_order_id: str | None

@dataclass
class WashSaleFlag:
    id: int
    loss_trade_id: int      # FK to closed_trades
    replacement_order_id: str  # the buy that triggered wash
    disallowed_loss: float
    adjusted_cost_basis: float
    wash_window_start: date
    wash_window_end: date
    flagged_at: datetime
```

**Pre-trade warning:**
In the risk gate, when processing a buy order: check `pending_wash_losses` for the same symbol with open windows. If found, add a warning to the risk gate result (but do NOT block). The warning surfaces in the audit trail and trade logging.

### 6.4 Tax Lot Tracking

**On every buy fill:**
Create a tax lot record: `(lot_id, symbol, quantity, cost_basis, acquired_date, order_id)`

**On every sell fill:**
Match lots using FIFO (First In, First Out):
1. Query open lots for symbol, ordered by `acquired_date ASC`
2. Match sell quantity against lots, consuming oldest first
3. For each matched lot: compute gain/loss = `(sell_price - cost_basis) * matched_qty`
4. Mark consumed lots as closed with exit date and realized gain/loss
5. If wash sale applies to this lot, adjust cost basis before computing gain/loss

New `tax_lots` table:

```python
@dataclass
class TaxLot:
    lot_id: int             # auto-increment PK
    symbol: str
    quantity: int           # remaining open quantity
    original_quantity: int  # quantity at creation
    cost_basis: float       # per share, adjusted for wash sales
    acquired_date: date
    order_id: str           # the buy order that created this lot
    closed_date: date | None
    exit_price: float | None
    realized_pnl: float | None
    wash_sale_adjustment: float  # added to cost basis
    status: str             # "open" | "closed"
```

### 6.5 Margin Check (Reg T)

Pre-trade check in risk gate. Simplified model since no equity shorts:
- Long equity: requires 50% cash (Reg T initial margin)
- Options: max_loss is the margin requirement (debit spreads = premium paid, naked = broker formula)
- If proposed order would exceed available margin → REJECT

Compute: `margin_required = sum(position_margin) + proposed_order_margin`. Compare to account equity.

### 6.6 New Module Structure

Create `src/quantstack/execution/compliance/` package with separation by lifecycle:

```
execution/compliance/
├── __init__.py
├── pretrade.py      # PDTChecker, MarginCalculator — called in risk gate
├── posttrade.py     # WashSaleTracker, TaxLotManager — called on fill hooks
└── calendar.py      # Business day calendar utility (exchange_calendars)
```

**Pre-trade gates** (`pretrade.py`): `PDTChecker.check()` and `MarginCalculator.check()` are called by the risk gate synchronously before order submission.

**Post-trade hooks** (`posttrade.py`): `WashSaleTracker.on_fill()` and `TaxLotManager.on_fill()` are called after fill recording.

**Business day calendar** (`calendar.py`): Wraps `exchange_calendars` or `pandas_market_calendars` to provide business-day-aware date arithmetic. Used by PDT (5 business day window), TWAP scheduling (market hours), and wash sale (30 calendar days). Add `exchange_calendars` as a dependency.

### 6.7 Margin Calculation Detail

For long equity: 50% initial margin (Reg T).

For options, margin depends on strategy type:
- **Long calls/puts:** Premium paid (max loss = premium)
- **Debit spreads:** Net premium paid
- **Credit spreads:** (Spread width - credit received) × contracts × 100
- **Covered calls:** No additional margin (shares serve as collateral)

The margin calculator must identify strategy type from position data. Initially support long options and debit spreads only (matching current portfolio). Credit spread and naked option margin calculations deferred until those strategies are traded.

---

## 7. Section: Real TWAP/VWAP Execution (6.3)

### 7.1 Problem

The OMS selects TWAP/VWAP but the paper broker executes everything as a single fill. No time-slicing, no child order generation, no participation constraints. Paper trading shows no market impact — live trading would incur 10-50 bps impact per trade.

### 7.2 Architecture: Separate `algo_scheduler.py` (EMS)

The original interview preference was to extend `order_lifecycle.py`, but external review correctly identified that the file's own header says "explicit state machine separating OMS from EMS" — the algo scheduler IS the EMS. Keeping scheduling logic in the OMS file would violate this design principle and push the file past 1,500 lines.

**Design:** New `src/quantstack/execution/algo_scheduler.py` that imports OMS types from `order_lifecycle.py` and manages the parent/child execution lifecycle. The OMS knows that an order can have children (via `exec_algo` field); the scheduling logic — time-slicing, volume profiles, child submission, failure handling — lives in the scheduler.

**Sync/async boundary:** The existing OMS uses `threading.RLock` for thread safety. The algo scheduler runs as an async loop (like the execution monitor). Broker calls use `loop.run_in_executor()` following the same pattern as `execution_monitor.py`. The scheduler never acquires the OMS lock inside a coroutine — all OMS state updates go through the OMS's public methods which handle their own locking.

**POV fallback:** The algo selection logic has four outcomes: IMMEDIATE, TWAP, VWAP, POV. This plan implements TWAP and VWAP. POV orders fall back to VWAP with `max_participation_rate` capped at 5%. This is explicitly documented in the scheduler's dispatch logic.

New types in `algo_scheduler.py`:

```python
@dataclass
class AlgoParentOrder:
    parent_order_id: str
    symbol: str
    side: str
    total_quantity: int
    algo_type: str          # "twap" | "vwap"
    start_time: datetime
    end_time: datetime
    arrival_price: float
    max_participation_rate: float
    status: str             # "pending" | "active" | "completing" | "completed" | "cancelled"
    filled_quantity: int
    avg_fill_price: float

@dataclass
class ChildOrder:
    child_id: str
    parent_id: str
    scheduled_time: datetime
    target_quantity: int
    filled_quantity: int
    fill_price: float
    status: str             # mirrors child state machine
    attempts: int
    broker_order_id: str | None
```

### 7.3 TWAP Scheduling

When an order is submitted with `exec_algo == TWAP`:

1. Create `AlgoParentOrder` with `start_time = now`, `end_time = now + duration`
2. Divide total quantity into N equal slices (N configurable, default = duration_minutes / 5)
3. For each slice, create a `ChildOrder` with:
   - `scheduled_time` = bucket start + random jitter (+/-20% of bucket width)
   - `target_quantity` = base_qty + random variation (+/-10%)
4. An async loop wakes at each child's scheduled time and submits it to the broker
5. After each child fills, update parent `filled_quantity` and `avg_fill_price` (VWAP of all legs)

**Invariant:** `sum(child.filled_qty) == parent.filled_qty` at all times.

### 7.4 VWAP Scheduling

Same as TWAP except child sizes are weighted by historical intraday volume profile:

1. Load volume profile for symbol from historical bar data (1-min or 5-min bars, available per interview)
2. Average normalized volume distribution across prior 10-20 trading days per bucket
3. For each bucket: `child_qty = total_qty * (bucket_volume_pct / remaining_volume_pct)`
4. This concentrates execution during high-volume periods (open/close) where per-share impact is lower

**Volume profile builder:**
A function that takes a symbol and returns a dict mapping time buckets to volume percentages. Cached daily. Falls back to synthetic U-shaped curve if insufficient historical data.

**Favorability check (optional enhancement):**
Buy children only submit when `ask < running_VWAP * (1 - 3bps)`. This avoids buying into upward momentum. Configurable, off by default for initial implementation.

### 7.5 Parent/Child State Machine

**Parent states:**
```
PENDING → ACTIVE → COMPLETING → COMPLETED
                 → CANCELLING → CANCELLED
```

**Child states:**
```
PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED
                   → CANCELLED
                   → EXPIRED
                   → REJECTED
```

**State transitions:**
- Parent → ACTIVE: first child submitted
- Parent → COMPLETING: end_time reached OR filled_qty ≥ 99.5% of total
- Parent → COMPLETED: all children terminal AND filled_qty ≥ 99.5%
- If unfilled at completion: log shortfall, don't force a final aggressive order (conservative default)

**Child failure handling:**
- REJECTED: check reason. Buying power → reduce 50%, retry (max 3 attempts). Invalid params → fail child.
- Timeout (no fill within 2× bucket duration): cancel child, redistribute to next slice
- API error: exponential backoff (1s, 2s, 4s), retry
- 3 consecutive child failures: pause ALL active parents (network-level failures affect all orders), log alert

**Parent cancellation triggers:**
- Kill switch activation → cancel all active parents immediately, cancel all open child orders
- Risk gate daily halt → cancel all active parents
- Execution monitor exit signal for the parent's symbol → cancel the parent, its remaining children
- Manual cancellation via trade service

**Crash recovery on startup:**
The scheduler's `startup_recovery()` method queries for any parent orders in ACTIVE or COMPLETING state:
- If children have open broker orders: attempt to cancel via broker API
- Log all orphaned state with full context (parent_id, filled_qty, remaining_qty)
- Mark parent as CANCELLED with reason "system_restart_recovery"
- Do NOT attempt to resume mid-execution — too risky without knowing market state. Always cancel and let the trading graph re-evaluate.

### 7.6 Database Schema

New `algo_parent_orders` table with parent lifecycle and scheduling parameters.

New `algo_child_orders` table with child lifecycle, FK to parent, scheduled_time, broker_order_id.

New `algo_performance` table:

```python
@dataclass
class AlgoPerformance:
    parent_order_id: str
    symbol: str
    side: str
    algo_type: str
    total_qty: int
    filled_qty: int
    arrival_price: float
    avg_fill_price: float
    benchmark_vwap: float | None
    implementation_shortfall_bps: float
    vwap_slippage_bps: float | None
    delay_cost_bps: float
    market_impact_bps: float
    num_children: int
    num_children_filled: int
    num_children_failed: int
    max_participation_rate: float
    actual_participation_rate: float
    decision_time: datetime
    first_fill_time: datetime | None
    last_fill_time: datetime | None
    scheduled_end_time: datetime
```

### 7.7 Paper Broker Enhancement

The paper broker must simulate realistic TWAP/VWAP execution instead of instant fills:

**For each child order submitted to paper broker:**
1. Look up the historical bar covering the child's scheduled time
2. Participation cap: `max_fillable = bar_volume * max_participation_rate` (default 2%)
3. Fill quantity: `min(child_qty, max_fillable)`
4. Fill price: `bar_vwap + direction * noise` where noise is uniform within `[0, (bar_high - bar_low) * 0.3]`
5. If child_qty > max_fillable: return partial fill, remainder becomes unfilled

This replaces the current instant-fill model for TWAP/VWAP orders while keeping IMMEDIATE orders on the existing model.

### 7.8 Async Execution Loop

The `AlgoScheduler` class in `algo_scheduler.py` runs an async loop:
1. Maintains a priority queue of pending children sorted by `scheduled_time`
2. Wakes at each child's scheduled time
3. Submits the child to the broker via `loop.run_in_executor()` (broker.execute is sync)
4. Processes the fill (or failure)
5. Updates parent state via OMS public methods (which handle their own RLock)
6. Records fill leg (6.2), triggers TCA EWMA update (6.1), writes audit trail (6.6)

This loop runs alongside the existing execution monitor's price poll loop. Both use asyncio. The algo loop only runs when there are active parent orders. On startup, `startup_recovery()` runs before the scheduling loop begins.

**Position updates during execution:** Each child fill updates the position incrementally via `PortfolioState.upsert_position()`. The position reflects the running average cost from all filled children, not just the final parent fill. This ensures the execution monitor can evaluate stop-loss rules against partially filled positions.

---

## 8. Section: Liquidity Model (6.4)

### 8.1 Problem

Current check: `if daily_volume < min_daily_volume: warn`. Missing: spread check, depth estimation, time-of-day variation, stressed liquidity assessment.

### 8.2 Liquidity Assessment Components

**Spread estimation:**
- Use historical quote data to estimate typical bid-ask spread per symbol
- Store as `spread_bps` per symbol, per time-of-day bucket (reuse TCA time buckets)
- Update daily from end-of-day quote data

**Depth estimation:**
- Estimate available depth from intraday volume profile: `depth_at_time = daily_volume * bucket_volume_pct`
- This estimates how many shares can be executed in a time bucket without excessive impact

**Pre-trade check:**
- Reject or scale down if `order_size > estimated_depth * 0.10` (10% of available depth)
- This is stricter than the current ADV-only check

**Time-of-day adjustment:**
- Spreads are wider and depth is thinner at open (9:30-10:00) and close (15:30-16:00)
- Apply multipliers: open = 1.5x spread, close = 1.3x spread, midday = 1.0x

**Stressed exit scenario:**
- Portfolio-level check: if all positions exited simultaneously, what's the total estimated slippage?
- `stressed_slippage = sum(position_size_i * estimated_slippage_bps_i)`
- If stressed slippage exceeds a threshold (configurable, e.g., 100 bps of portfolio value), alert

### 8.3 Integration

Add `LiquidityModel` class. Risk gate calls `LiquidityModel.pre_trade_check()` between the existing ADV check and participation cap. The liquidity model can:
- PASS (proceed)
- SCALE_DOWN (reduce order size, like participation cap)
- REJECT (insufficient liquidity)

The stressed exit check runs in the continuous risk monitor (every 60s alongside existing position drift checks).

---

## 9. Section: Best Execution Audit Trail (6.6)

### 9.1 Problem

SEC Rule 606 and FINRA Rule 5310 require best execution demonstration. The `fills` table stores basic data but no NBBO reference, venue data, or algo selection rationale.

### 9.2 Schema

New `execution_audit` table:

```python
@dataclass
class ExecutionAudit:
    audit_id: int           # auto-increment PK
    order_id: str           # FK to orders
    fill_leg_id: int | None # FK to fill_legs (for child fills)
    nbbo_bid: float
    nbbo_ask: float
    nbbo_midpoint: float
    fill_price: float
    fill_venue: str | None  # "alpaca" | "paper"
    price_improvement_bps: float  # (nbbo_mid - fill_price) / nbbo_mid * 10000 for buys
    algo_selected: str
    algo_rationale: str     # e.g., "TWAP: 0.5% ADV, 30-min window"
    timestamp_ns: int       # nanosecond-precision timestamp
```

### 9.3 NBBO Capture

NBBO source: Alpaca IEX quotes (15-min delayed). Acceptable for paper trading audit per interview.

**For ALL order types (IMMEDIATE, TWAP/VWAP child fills):**
On every fill event:
1. Fetch current best bid/ask for symbol from Alpaca IEX
2. Compute midpoint and price improvement
3. Insert `execution_audit` row

This ships early in the implementation order (after SEC compliance, before TWAP/VWAP) so IMMEDIATE orders get audit coverage immediately. When TWAP/VWAP lands later, each child fill gets its own audit row with `fill_leg_id` linking to the `fill_legs` table.

**Rate limiting:** For TWAP/VWAP with many children, NBBO fetches are naturally spaced by the scheduling interval (typically 5+ minutes between children). No additional rate limiting needed for typical configurations.

### 9.4 Query Interface

Key audit queries:
- "Show all fills worse than NBBO midpoint" — `WHERE price_improvement_bps < 0`
- "Average price improvement by algo type" — aggregate by `algo_selected`
- "Execution quality over time" — time series of `price_improvement_bps`

---

## 10. Section: Options Monitoring Rules (6.7)

### 10.1 Dependency

Depends on Phase 2 item 2.12 (Greeks in risk gate). Greeks computation already exists in `core/options/engine.py:293-364`. This section assumes those Greeks are accessible from the execution monitor.

### 10.2 Rule Configuration

Each rule has a configurable action: `auto_exit` or `flag_only`. Default actions per the interview:

```python
@dataclass
class OptionsMonitorRule:
    name: str
    enabled: bool
    action: str  # "auto_exit" | "flag_only"

# Defaults:
RULES = {
    "theta_acceleration": OptionsMonitorRule("theta_acceleration", True, "auto_exit"),
    "pin_risk":           OptionsMonitorRule("pin_risk", True, "auto_exit"),
    "assignment_risk":    OptionsMonitorRule("assignment_risk", True, "flag_only"),
    "iv_crush":           OptionsMonitorRule("iv_crush", True, "flag_only"),
    "max_theta_loss":     OptionsMonitorRule("max_theta_loss", True, "auto_exit"),
}
```

### 10.3 Rule Definitions

**Theta Acceleration:** Trigger when DTE < 7 AND theta/premium > 5%/day. Theta decay accelerates exponentially near expiry. Compute daily theta loss as percentage of current premium.

**Pin Risk:** Trigger when DTE < 3 AND underlying price is within 1% of strike. Near-the-money options at expiry have unpredictable delta and settlement risk.

**Assignment Risk:** Trigger when short call is ITM AND ex-dividend date is within 2 trading days. Early assignment risk spikes when time value < dividend amount. Requires ex-dividend calendar data.

**IV Crush:** Trigger when earnings were within 2 trading days AND implied volatility dropped > 30% from pre-earnings level. Requires tracking IV snapshots before earnings events.

**Max Theta Loss:** Trigger when cumulative theta decay exceeds 40% of entry premium. Track `premium_at_entry` (already in positions table) vs current premium.

### 10.4 Integration into Execution Monitor

Add an `_evaluate_options_rules()` method in `execution_monitor.py` that runs AFTER the standard equity rules (which apply to all positions). This method:

1. Skips non-options positions (check `instrument_type == "option"`)
2. Fetches current Greeks for the position via `compute_greeks_dispatch()`
3. Evaluates each enabled rule
4. For `auto_exit` rules: triggers `_submit_exit()` (same as equity stops)
5. For `flag_only` rules: logs alert to DB, does NOT trigger exit

### 10.5 Data Requirements

- Greeks: from existing options engine
- Ex-dividend calendar: needs new data source or manual maintenance table
- Pre-earnings IV snapshots: need to capture IV before known earnings dates
- Earnings dates: can derive from Alpha Vantage earnings calendar

---

## 11. Section: Slippage Model Enhancement (6.8)

### 11.1 Problem

Current paper broker uses fixed constants: 2 bps half-spread, k=5 sqrt-impact. No time-of-day variation, no per-symbol calibration from realized fills.

### 11.2 Enhancement

**Per-symbol calibration from TCA EWMA (6.1):**
Replace fixed constants with EWMA-calibrated values from `tca_parameters` table. When paper broker computes fill price:
1. Look up `tca_parameters` for symbol + current time bucket
2. If found: use `ewma_spread_bps` for spread cost, `ewma_impact_bps` for impact coefficient
3. If not found: fall back to existing constants

**Time-of-day profiles:**
Build slippage profiles from historical data:
- Morning (9:30-11:00): higher spread (1.3x), higher impact
- Midday (11:00-14:00): baseline
- Afternoon (14:00-15:30): slight increase
- Close (15:30-16:00): higher spread (1.2x), volume spike reduces per-share impact

**Model accuracy tracking:**
After every fill, compare predicted slippage (from pre-trade forecast) to realized slippage (from fill vs arrival price). Store the ratio in a `slippage_accuracy` table. Track moving average of predicted/realized ratio. Alert if ratio drifts beyond 0.5x or 2.0x.

---

## 12. Section: Borrowing/Funding Cost Model (6.9)

### 12.1 Scope

No equity shorts currently (options only). Funding cost model focuses on margin interest for leveraged positions.

### 12.2 Margin Interest Calculation

For positions using margin (leveraged longs):
```
daily_interest = margin_used * annual_rate / 252
```

Where `margin_used = position_notional - cash_allocated` and `annual_rate` is the broker's margin rate (Alpaca's current rate, stored as config).

### 12.3 Integration

**Daily P&L update:** Add `funding_cost` to position P&L calculation in `portfolio_state.py`. Deduct daily interest from unrealized P&L.

**Strategy performance:** Surface cumulative funding costs in strategy performance metrics. A strategy that holds leveraged positions for weeks accumulates meaningful interest drag.

**New fields on `positions` table:** `margin_used DOUBLE PRECISION DEFAULT 0.0`, `cumulative_funding_cost DOUBLE PRECISION DEFAULT 0.0`.

---

## 13. Migration and Rollout Strategy

### 13.1 Schema Migrations

All new tables are additive — no existing tables are modified (except adding columns to `positions`). Migrations can run without downtime:

1. `fill_legs` table (6.2)
2. `tca_parameters` table (6.1)
3. `day_trades`, `wash_sale_flags`, `tax_lots` tables (6.5)
4. `algo_parent_orders`, `algo_child_orders`, `algo_performance` tables (6.3)
5. `execution_audit` table (6.6)
6. `slippage_accuracy` table (6.8)
7. Add columns to `positions`: `margin_used`, `cumulative_funding_cost` (6.9)

### 13.2 Feature Gating

Each item should be deployable independently behind feature flags where appropriate:
- TWAP/VWAP: if flag off, all orders continue as IMMEDIATE (existing behavior)
- PDT enforcement: if flag off, risk gate skips PDT check (but should default ON given account constraint)
- Options monitoring: naturally gated by Phase 2 dependency

### 13.3 Testing Strategy

Each section ships with tests. Integration tests verify cross-section behavior:
- TCA EWMA update after TWAP child fills (6.1 + 6.3)
- PDT counting after TWAP intra-day round-trips (6.5 + 6.3)
- Audit trail records per child fill (6.6 + 6.3)

---

## 14. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| TWAP/VWAP adds complexity | Isolated in `algo_scheduler.py` (EMS), separate from OMS. Start with TWAP, add VWAP after proven. IMMEDIATE remains default for small orders. |
| PDT counting edge cases (options, partial fills, spreads) | Conservative counting — when in doubt, count as day trade. Match on full OCC contract symbol for options. Multi-leg spreads: each leg is an independent trade for PDT purposes (conservative). |
| Wash sale look-forward problem | Two-phase detection: `pending_wash_losses` table flags realized losses, subsequent buys retroactively trigger wash sale treatment. |
| Paper broker VWAP simulation fidelity | Start with TWAP simulation (equal slices against bars). Add VWAP volume weighting once TWAP proves reliable. |
| Sync/async impedance (OMS uses RLock, scheduler uses asyncio) | Scheduler calls broker via `run_in_executor()`. OMS state updates go through public methods that handle their own locking. Scheduler never acquires RLock in a coroutine. |
| Crash mid-TWAP leaves orphaned state | `startup_recovery()` cancels all ACTIVE parents on restart. Never resumes — let trading graph re-evaluate. |
| Data feed failure during 30-min TWAP | Scheduler pauses child submission if price feed is stale (>30s). Resumes on reconnection. If stale >5 min, cancels parent. |
| Options monitoring requires external data (earnings dates, ex-div calendar) | Start with rules that use only Greeks and position data (theta, pin risk, max theta loss). Add assignment risk and IV crush when data sources are available. |
