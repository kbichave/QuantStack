# 03 — Execution Layer: Make The System Actually Trade

**Priority:** P1-P2 — Must Fix for Reliable Live Trading
**Timeline:** Week 2-6
**Gate:** Real TWAP/VWAP execution, Greeks in risk gate, intraday circuit breakers, SEC compliance basics.

---

## Why This Section Matters

The CTO audit assessed execution architecture — it's well-designed (multi-layer enforcement, order lifecycle state machine, execution algo selection). The Quant Scientist audit assessed execution *reality* — the algorithms are phantom, the cost model is uncalibrated, Greeks don't gate options trades, and there's zero SEC compliance. The architecture is a B+. The substance is a D.

---

## 3.1 Real Execution Algorithm Implementation

**Finding ID:** QS-E1
**Severity:** CRITICAL
**Effort:** 5 days

### The Problem

The order lifecycle selects an execution algorithm (IMMEDIATE/TWAP/VWAP/POV) based on order size vs. ADV. This looks sophisticated. But the paper broker executes everything as a single fill. There is no time-slicing, no child order generation, no participation rate enforcement.

```
What the system claims:    "Order 5% ADV → VWAP algorithm selected"
What actually happens:      Single market order, instant fill, one price
What should happen:         50 child orders over 2 hours, each ≤0.1% ADV
```

Paper trading results show no market impact. Live trading with the same sizes would incur 10-50 bps of impact cost per trade. A strategy that backtests at Sharpe 1.2 with instant fills may have Sharpe 0.4 with realistic execution.

### The Fix

| Step | Action |
|------|--------|
| 1 | Implement TWAP as child order generator: split parent into N equal slices over T minutes |
| 2 | Implement VWAP as volume-weighted slicing: heavier at open/close, lighter midday |
| 3 | Paper broker simulates fills against historical bar data with participation constraints |
| 4 | Track per-child-order fills separately (see 3.6 Partial Fill Fix) |
| 5 | Add `algo_performance` table: parent_order_id, algo_type, expected_cost, realized_cost |

### Acceptance Criteria

- [ ] TWAP generates child orders with configurable slice count and duration
- [ ] VWAP generates volume-weighted child orders
- [ ] Paper broker fills against realistic volume profiles, not instant
- [ ] Post-trade TCA compares realized VWAP vs. arrival price

---

## 3.2 Options Greeks in Risk Gate

**Finding ID:** QS-E3
**Severity:** CRITICAL
**Effort:** 3 days

### The Problem

The risk gate's options checks are limited to DTE bounds (7-60 days) and premium at risk (2% per position, 8% total). No delta/gamma/vega/theta limits. A 50-contract short straddle at 21 DTE gets the same treatment as a 50-delta call spread — but the straddle can lose $100K on a 5% move due to gamma explosion.

**The capability exists** — `core/risk/options_risk.py` (444 lines) tracks portfolio Greeks. But `risk_gate.py` never calls it.

### The Fix

| Check | Limit | Rationale |
|-------|-------|-----------|
| Portfolio delta exposure | ±$50K per 1% underlying move | Directional risk cap |
| Portfolio gamma | Max $10K P&L per 1% squared | Convexity risk cap |
| Portfolio vega | Max $5K per 1 vol point | Volatility risk cap |
| Theta budget | Acceptable daily time decay documented | Cost of carry awareness |
| Pin risk | DTE < 3 AND near strike → force exit/roll | Assignment risk prevention |

### Implementation

| Step | Action | Location |
|------|--------|----------|
| 1 | Add Greeks aggregation call to `portfolio_state.py` | `portfolio_state.py` |
| 2 | Wire `options_risk.py` Greeks checks into `risk_gate.check()` | `risk_gate.py` |
| 3 | Block any trade that would push portfolio Greeks outside limits | `risk_gate.py` |
| 4 | Add options-specific exit rules to ExecutionMonitor (see 3.3) | `execution_monitor.py` |

### Acceptance Criteria

- [ ] Pre-trade Greeks check in risk gate for all options orders
- [ ] Portfolio-level Greeks aggregated and limit-checked
- [ ] Short straddle with excessive gamma rejected pre-trade

---

## 3.3 Options Monitoring Rules in ExecutionMonitor

**Finding ID:** DO-3
**Severity:** HIGH
**Effort:** 2 days

### The Problem

All 6 exit rules in ExecutionMonitor are equity-centric. The monitor stores `option_contract` and `underlying_symbol` but evaluates options identically to equities. Missing rules:

| Rule | What It Prevents | Status |
|------|-----------------|--------|
| Theta acceleration: DTE < 7 AND theta/premium > 5%/day → TIGHTEN | Final-week decay eating premium | NOT IMPLEMENTED |
| Pin risk: DTE < 3 AND abs(underlying - strike)/strike < 1% → EXIT | Assignment at expiration | NOT IMPLEMENTED |
| Assignment risk: short call ITM + ex-div within 2 days → EXIT/ROLL | Early assignment, short stock | NOT IMPLEMENTED |
| IV crush: post-earnings + IV dropped >30% → reassess | Debit positions lose edge post-event | NOT IMPLEMENTED |
| Max theta loss: cumulative decay > 40% of premium paid → EXIT | Slow bleed on OTM options | NOT IMPLEMENTED |

### The Fix

Add options-specific exit evaluation after the standard equity rules in `execution_monitor.py`. The system already computes Greeks on demand (`core/options/engine.py:293-364`) — wire monitoring calls to open options positions.

### Acceptance Criteria

- [ ] Options positions evaluated with theta acceleration, pin risk, and IV crush rules
- [ ] DTE < 3 near-the-money positions auto-flagged for exit/roll
- [ ] Short call positions checked against ex-dividend calendar

---

## 3.4 Intraday Drawdown Circuit Breaker

**Finding ID:** QS-E5
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Daily loss limit (-2%) triggers a halt, but only after losses are realized. No real-time unrealized P&L monitoring. The system can lose 5% in 5 minutes before the daily halt triggers.

### The Fix

Add `IntraDayCircuitBreaker` to execution monitor:

| Threshold | Action |
|-----------|--------|
| -1.5% unrealized | Halt new entries. Existing positions monitored. |
| -2.5% unrealized | Begin systematic exit of weakest positions. |
| -5.0% unrealized | Emergency liquidation of all positions. |
| -1% in 5 minutes (velocity) | Halt regardless of daily level. |
| Single position -20% | Force review of that position. |

### Acceptance Criteria

- [ ] Unrealized P&L checked every tick cycle
- [ ] Graduated response: halt entries → systematic exit → emergency liquidation
- [ ] Velocity check catches fast drops even when daily level is not breached

---

## 3.5 TCA Feedback Loop

**Finding ID:** QS-E6, Loop-2
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

Pre-trade TCA (Almgren-Chriss) forecasts cost. Post-trade TCA stores realized cost. No feedback loop:
- If realized slippage is consistently 2x forecast, coefficients don't adjust
- `tca_recalibration.py` requires 50 trades per segment to fit — months away
- Next trade still uses the original stale forecast

### The Fix

Implement EWMA recalibration after every fill:

```
forecast_new = 0.9 * forecast_old + 0.1 * realized_cost
```

Until 50 trades accumulated for full calibration, use conservative multiplier (2x forecast cost). Position sizing uses the updated forecast, so higher-than-expected costs automatically reduce position sizes.

### Acceptance Criteria

- [ ] Every fill triggers EWMA update of cost model parameters
- [ ] Cost model parameters per symbol/time-of-day stored and used
- [ ] Conservative 2x multiplier applied until sufficient trade history

---

## 3.6 Partial Fill Tracking

**Finding ID:** QS-E9
**Severity:** HIGH
**Effort:** 1 day

### The Problem

When a partial fill arrives, the fill price is updated but previous partial fill prices are overwritten, not accumulated. Can't reconstruct average fill price, execution VWAP, or fill trajectory.

### The Fix

Add `fill_legs` table: `(order_id, leg_sequence, quantity, price, timestamp)`. Compute VWAP from legs for post-trade TCA.

### Acceptance Criteria

- [ ] All partial fills stored as individual legs
- [ ] Average fill price computed from leg VWAP
- [ ] Fill trajectory available for execution quality analysis

---

## 3.7 Liquidity Model

**Finding ID:** QS-E4
**Severity:** CRITICAL
**Effort:** 3 days

### The Problem

Current liquidity check: `if daily_volume < min_daily_volume: reject`. That's it. Missing: bid-ask spread check, market depth estimation, time-of-day liquidity variation, stressed liquidity modeling, exit liquidity assessment.

### The Fix

| Component | Implementation |
|-----------|---------------|
| Spread estimation | Historical spread from AV/Alpaca quote data |
| Depth estimation | Intraday volume profile → depth at each time bucket |
| Pre-trade check | `if order_size > estimated_depth * 0.1: scale down or reject` |
| Stressed liquidity | `if all_positions_exit_simultaneously: estimate total slippage cost` |
| Time-of-day adjustment | Wider spreads at open/close, tighter midday |

### Acceptance Criteria

- [ ] Pre-trade liquidity check considers depth, not just ADV
- [ ] Time-of-day liquidity variation modeled
- [ ] Stressed exit scenario estimated for portfolio-level risk

---

## 3.8 Pre-Trade Correlation and Concentration Checks

**Finding IDs:** CTO H1, CTO H3, CTO H4
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Three risk checks that exist only in post-hoc monitoring need to move pre-trade:

| Check | Current State | Required State |
|-------|--------------|----------------|
| Pairwise correlation | 60-day rolling, monitor loop only | Pre-trade: reject if corr > 0.7 with existing position |
| Portfolio heat budget | Missing entirely | Max daily notional deployed (e.g., 30% of equity/day) |
| Sector concentration | Post-hoc Herfindahl only | Pre-trade: reject if sector would exceed 40% |

### The Fix

Add all three checks to `risk_gate.check()` — evaluate before order submission, not after.

### Acceptance Criteria

- [ ] New position rejected if correlation > 0.7 with any existing position
- [ ] Daily notional deployment capped (configurable, default 30%)
- [ ] Sector concentration checked pre-trade (configurable, default 40% max)

---

## 3.9 SEC Compliance Basics

**Finding ID:** DO-8
**Severity:** CRITICAL (for live trading)
**Effort:** 5-7 days

### The Problem

Zero SEC compliance infrastructure:

| Requirement | Statute | Status |
|------------|---------|--------|
| Wash Sale tracking | 26 USC §1091 | NOT TRACKED |
| Pattern Day Trader enforcement | FINRA 4210 | NOT ENFORCED |
| Reg T Margin calculation | 12 CFR 220 | NOT CALCULATED |
| Tax Lot tracking | IRS Form 8949 | NOT IMPLEMENTED |
| Short-Term vs Long-Term distinction | 26 USC §1222 | NOT DISTINGUISHED |

### The Fix

| Check | Implementation | Priority |
|-------|---------------|----------|
| **Wash Sale** | Query trades: sold symbol at loss within 30 calendar days? If yes, flag and adjust cost basis | P0 for live |
| **PDT** | Count round-trips in rolling 5 business days. If >= 3 AND account < $25K → REJECT | P0 if account < $25K |
| **Tax Lots** | On BUY: create `tax_lot(symbol, qty, price, date, lot_id)`. On SELL: match FIFO, compute gain/loss | P0 for live |
| **Margin** | Long equity: 50% cash. Short equity: 150%. Options: max_loss. If insufficient → REJECT | P1 |
| **Form 8949 export** | Generate from tax_lots table for tax filing | P2 |

### Acceptance Criteria

- [ ] Wash sale tracking active — flagged trades have adjusted cost basis
- [ ] PDT enforcement blocks 4th day-trade if account < $25K
- [ ] Tax lots created on every buy, matched on every sell
- [ ] Margin checked pre-trade for leveraged positions

---

## 3.10 Best Execution Audit Trail

**Finding ID:** QS-E7
**Severity:** HIGH
**Effort:** 2 days

### The Problem

SEC Rule 606 and FINRA Rule 5310 require demonstration of best execution. The `fills` table stores basic fill data but no NBBO reference, no venue data, no algorithm selection rationale.

### The Fix

Add `execution_audit` table: `(order_id, nbbo_bid, nbbo_ask, fill_price, fill_venue, algo_selected, algo_rationale, timestamp_ns)`. Populate on every fill.

### Acceptance Criteria

- [ ] Every fill records NBBO at time of execution
- [ ] Execution algorithm selection rationale logged
- [ ] Query available: "show all fills worse than NBBO midpoint"

---

## 3.11 Fed Event / Macro Calendar Enforcement

**Finding ID:** DO-4
**Severity:** HIGH
**Effort:** 1 day

### The Problem

The events collector outputs `has_fomc_24h`, `has_macro_event`. Agent prompts say "reduce sizing 50% within 24h of FOMC." But this is prompt guidance, not code. No hard rule in `risk_gate.py` checks the macro calendar.

### The Fix

Add mandatory sizing multipliers to `risk_gate.py`:

| Condition | Sizing Multiplier | Restriction |
|-----------|-------------------|-------------|
| FOMC within 4 hours | 0.5x | No naked options |
| CPI/NFP within 2 hours | 0.75x | — |
| VIX > 30 | 0.7x | — |
| VIX > 50 | 0.0x (paper only) | No live orders |

### Acceptance Criteria

- [ ] Macro calendar checked in risk gate, not just agent prompts
- [ ] FOMC proximity automatically reduces position sizes
- [ ] VIX > 50 halts all live trading

---

## 3.12 Regime Flip Forced Review

**Finding ID:** CTO H5
**Severity:** HIGH
**Effort:** 1 day

### The Problem

If you entered a momentum trade in `trending_up` and regime flips to `ranging`, the system logs an alert but takes no action. The position stays open in a hostile regime.

### The Fix

Regime flip on active positions triggers mandatory review:
- Moderate mismatch (trending → ranging): tighten stops 50%
- Severe mismatch (trending_up → trending_down): auto-exit within 1 cycle

### Acceptance Criteria

- [ ] Regime flip triggers position review for affected strategies
- [ ] Severe regime mismatches trigger automatic exit
- [ ] Regime-at-entry stored per position for comparison

---

## 3.13 Correlation Check: Pre-Trade Veto, Not Just Alert

**Finding ID:** QS-E10
**Severity:** HIGH
**Effort:** 1 day

### The Problem

The CTO audit noted correlation is post-hoc (H1). The deeper problem: when correlation data is unavailable, it returns "alert" not "reject". No pre-trade veto power exists — monitoring only. No stressed correlation modeling (correlations spike to 0.95 in crashes).

### The Fix

Pre-trade correlation gate: `if adding_position_corr_with_existing > 0.7: apply_concentration_haircut(50%)`. Stressed correlation: use `min(historical_corr, 0.9)` as stress case.

### Acceptance Criteria

- [ ] Pre-trade correlation check vetoes (not just alerts) on high correlation
- [ ] Stressed correlation estimates used for risk scenarios
- [ ] Missing correlation data = reject (not alert)

---

## 3.14 Smart Order Router: Single-Venue Limitation

**Finding ID:** QS-E12
**Severity:** HIGH (low urgency — acceptable for current scale)
**Effort:** 3-5 days (when scaling past $100K)

### The Problem

Routes to Alpaca or IBKR only. No venue splitting, dark pool access, price improvement analytics, or maker/taker fee optimization. Acceptable for paper trading and small positions, but before scaling to real capital >$100K, multi-venue routing becomes important.

### The Fix

Defer until scaling past $100K real capital. When ready: add venue splitting for large orders, price improvement tracking, maker/taker optimization.

### Acceptance Criteria

- [ ] Documented as known limitation for current scale
- [ ] Trigger: revisit when single-trade notional exceeds $50K

---

## 3.15 Real-Time Trading: Latency Budget Analysis

**Finding ID:** DO-6
**Severity:** CRITICAL (for future real-time ambitions)
**Effort:** 5-10 days (architectural change)

### The Problem

| Stage | Current | Real-Time Requirement |
|-------|---------|----------------------|
| Price feed | 1-min bars + 5-min polls + Alpaca IEX (15-min delayed) | Tick-level L1/L2, real-time |
| Signal generation | 5-min cycle + 30-300s LLM reasoning | <5 seconds, mostly deterministic |
| Entry latency (signal → order) | **5-60 minutes** | <5 seconds |
| Exit latency (price cross → order) | 5-60 seconds (polling) | <1 second |

The Alpaca IEX problem: free tier quotes are 15 minutes delayed. The system trades on stale prices.

### The Fix (When Ready for Real-Time)

| Step | Action |
|------|--------|
| 1 | Event-driven architecture — WebSocket-triggered signals, not polling |
| 2 | Split LLM from execution — LLM sets daily parameters, deterministic engine executes on tick events |
| 3 | Real-time data feed — Polygon ($199/mo) or Alpaca SIP |
| 4 | Low-latency hosting — AWS us-east-1 (1-5ms to Alpaca) vs Docker on Mac (50-200ms) |

### Acceptance Criteria

- [ ] Documented as architectural roadmap item
- [ ] Data feed upgrade budgeted ($200/mo)
- [ ] Hosting migration plan documented

---

## 3.16 Broker Abstraction: Alpaca-Coupled Components

**Finding ID:** DO-10
**Severity:** LOW
**Effort:** 2-4 days equity, 4-6 days including options (when adding IBKR)

### The Problem

`BrokerInterface` ABC is well-designed. But fill polling (30s/1s), time-in-force enums, options format, kill switch `cancel_orders`, and data feed are tightly coupled to Alpaca. Adding IBKR requires 2-6 days of adaptation.

### Acceptance Criteria

- [ ] Documented as known coupling
- [ ] Effort estimate captured for IBKR migration

---

## 3.17 Market Hours Hard Gating

**Finding ID:** CTO H2
**Severity:** HIGH
**Effort:** 1 day

### The Problem

Alpaca warns but allows orders outside market hours. No enforcement. An agent could submit orders at 3 AM that execute at market open with gap risk.

### The Fix

Hard-reject orders outside configurable trading windows unless explicitly flagged as `extended_hours=True`. Add window configuration to risk gate.

### Acceptance Criteria

- [ ] Orders outside market hours rejected by default
- [ ] `extended_hours=True` flag required for pre/post-market orders
- [ ] Gap risk acknowledged in extended hours mode

---

## Summary: Execution Layer Delivery

| # | Item | Effort | Timeline | Impact |
|---|------|--------|----------|--------|
| 3.1 | Real TWAP/VWAP | 5 days | Week 4-5 | Realistic execution costs |
| 3.2 | Greeks in risk gate | 3 days | Week 2-3 | Options risk properly gated |
| 3.3 | Options monitoring rules | 2 days | Week 3-4 | Theta/pin/assignment risk managed |
| 3.4 | Intraday circuit breaker | 2 days | Week 2-3 | Prevents flash-crash wipeout |
| 3.5 | TCA feedback loop | 2 days | Week 3-4 | Cost model improves over time |
| 3.6 | Partial fill tracking | 1 day | Week 3 | Execution quality measurable |
| 3.7 | Liquidity model | 3 days | Week 4-5 | Prevent illiquid market entries |
| 3.8 | Pre-trade correlation/concentration | 2 days | Week 3-4 | Portfolio diversification enforced |
| 3.9 | SEC compliance | 5-7 days | Week 3-5 | Legal liability eliminated |
| 3.10 | Best execution audit | 2 days | Week 4-5 | Regulatory compliance |
| 3.11 | Fed event enforcement | 1 day | Week 2-3 | Macro risk managed |
| 3.12 | Regime flip forced review | 1 day | Week 3 | Hostile regime positions managed |
| 3.13 | Pre-trade correlation veto | 1 day | Week 3-4 | Not just alert — actually reject |
| 3.14 | Smart order router (deferred) | 3-5 days | When >$100K | Multi-venue routing |
| 3.15 | Real-time latency (architectural) | 5-10 days | Roadmap | Event-driven architecture |
| 3.16 | Broker abstraction docs | 0.5 day | Week 5 | IBKR migration effort captured |
| 3.17 | Market hours hard gating | 1 day | Week 2-3 | No orders outside hours |

**Total estimated effort: 33-42 engineering days (overlapping with Sections 01-02).**
