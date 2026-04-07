# Phase 6: Execution Layer Completion — Deep Plan Spec

**Timeline:** Week 4-7
**Effort:** 23-27 days
**Gate:** Real TWAP/VWAP. TCA feedback. SEC compliance basics.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 6 transforms the execution layer from a prototype that simulates trading into a system that can actually trade. The CTO audit rated execution architecture B+ (well-designed multi-layer enforcement, order lifecycle state machine). The Quant Scientist audit rated execution reality D (phantom algorithms, uncalibrated costs, no SEC compliance).

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`03_EXECUTION_LAYER.md`](../03_EXECUTION_LAYER.md)
**Note:** Items 2.12-2.14 from the roadmap (Greeks in risk gate, intraday circuit breaker, Fed event enforcement) are covered in Phase 2 spec since they shipped with that timeline.

---

## Objective

Build real execution algorithms, implement TCA feedback, track partial fills properly, model liquidity, achieve basic SEC compliance, and create an audit trail suitable for regulatory review.

---

## Items

### 6.1 TCA Feedback Loop (EWMA)

- **Findings:** QS-E6, Loop-2 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.5](../03_EXECUTION_LAYER.md), [`07_FEEDBACK_LOOPS.md` Loop 2](../07_FEEDBACK_LOOPS.md)
- **Problem:** Pre-trade TCA (Almgren-Chriss) forecasts cost. Post-trade stores realized cost. No feedback loop. If realized slippage is consistently 2x forecast, coefficients don't adjust. Next trade uses same stale forecast.
- **Fix:** EWMA recalibration after every fill: `forecast_new = 0.9 * forecast_old + 0.1 * realized_cost`. Until 50 trades for full calibration, use conservative 2x multiplier. Position sizing uses updated forecast → higher-than-expected costs automatically reduce sizes.
- **Key files:** TCA module, fill hooks, position sizer
- **Acceptance criteria:**
  - [ ] Every fill triggers EWMA update of cost model parameters
  - [ ] Cost model parameters per symbol/time-of-day stored and used
  - [ ] Conservative 2x multiplier applied until sufficient trade history

### 6.2 Partial Fill Tracking

- **Finding:** QS-E9 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.6](../03_EXECUTION_LAYER.md)
- **Problem:** Partial fill arrives → fill price updated but previous partial fill prices overwritten. Can't reconstruct average fill price, execution VWAP, or fill trajectory.
- **Fix:** Add `fill_legs` table: `(order_id, leg_sequence, quantity, price, timestamp)`. Compute VWAP from legs for post-trade TCA.
- **Key files:** Broker fill handling, database schema
- **Acceptance criteria:**
  - [ ] All partial fills stored as individual legs
  - [ ] Average fill price computed from leg VWAP
  - [ ] Fill trajectory available for execution quality analysis

### 6.3 Real TWAP/VWAP Execution

- **Finding:** QS-E1 | **Severity:** CRITICAL | **Effort:** 5 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.1](../03_EXECUTION_LAYER.md)
- **Problem:** Order lifecycle selects algo (IMMEDIATE/TWAP/VWAP/POV) based on order size vs. ADV. But paper broker executes everything as single fill. No time-slicing, no child order generation, no participation rate. Paper trading shows no market impact → live trading would incur 10-50 bps impact per trade.
- **Fix:**
  1. Implement TWAP: split parent into N equal slices over T minutes
  2. Implement VWAP: volume-weighted slicing (heavier at open/close, lighter midday)
  3. Paper broker simulates fills against historical bar data with participation constraints
  4. Track per-child-order fills separately
  5. Add `algo_performance` table: parent_order_id, algo_type, expected_cost, realized_cost
- **Key files:** Execution algorithms, paper broker, `src/quantstack/brokers/paper_broker.py`
- **Acceptance criteria:**
  - [ ] TWAP generates child orders with configurable slice count and duration
  - [ ] VWAP generates volume-weighted child orders
  - [ ] Paper broker fills against realistic volume profiles, not instant
  - [ ] Post-trade TCA compares realized VWAP vs. arrival price

### 6.4 Liquidity Model

- **Finding:** QS-E4 | **Severity:** CRITICAL | **Effort:** 3 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.7](../03_EXECUTION_LAYER.md)
- **Problem:** Current check: `if daily_volume < min_daily_volume: reject`. Missing: spread check, depth estimation, time-of-day variation, stressed liquidity, exit assessment.
- **Fix:**
  - Spread estimation from historical quote data
  - Depth estimation from intraday volume profiles
  - Pre-trade check: `if order_size > estimated_depth * 0.1: scale down or reject`
  - Stressed liquidity: estimate total slippage if all positions exit simultaneously
  - Time-of-day adjustment: wider at open/close
- **Key files:** Pre-trade liquidity checks, risk gate
- **Acceptance criteria:**
  - [ ] Pre-trade liquidity check considers depth, not just ADV
  - [ ] Time-of-day liquidity variation modeled
  - [ ] Stressed exit scenario estimated for portfolio-level risk

### 6.5 SEC Compliance (Wash Sale, PDT, Tax Lots)

- **Finding:** DO-8 | **Severity:** CRITICAL | **Effort:** 5-7 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.9](../03_EXECUTION_LAYER.md)
- **Problem:** Zero SEC compliance infrastructure:
  - Wash Sale (26 USC §1091): NOT TRACKED
  - PDT (FINRA 4210): NOT ENFORCED
  - Reg T Margin (12 CFR 220): NOT CALCULATED
  - Tax Lot (IRS Form 8949): NOT IMPLEMENTED
- **Fix:**
  1. **Wash Sale (P0):** Query trades: sold symbol at loss within 30 calendar days? Flag + adjust cost basis.
  2. **PDT (P0):** Count round-trips in rolling 5 business days. If ≥3 AND account < $25K → REJECT.
  3. **Tax Lots (P0):** On BUY: create `tax_lot(symbol, qty, price, date, lot_id)`. On SELL: match FIFO, compute gain/loss.
  4. **Margin (P1):** Long equity 50% cash, short 150%, options max_loss. If insufficient → REJECT.
  5. **Form 8949 export (P2):** Generate from tax_lots table for tax filing.
- **Key files:** New compliance module, risk gate integration, trade hooks
- **Acceptance criteria:**
  - [ ] Wash sale tracking active — flagged trades have adjusted cost basis
  - [ ] PDT enforcement blocks 4th day-trade if account < $25K
  - [ ] Tax lots created on every buy, matched on every sell
  - [ ] Margin checked pre-trade for leveraged positions

### 6.6 Best Execution Audit Trail

- **Finding:** QS-E7 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.10](../03_EXECUTION_LAYER.md)
- **Problem:** SEC Rule 606 and FINRA Rule 5310 require best execution demonstration. `fills` table stores basic data but no NBBO reference, venue data, or algo selection rationale.
- **Fix:** Add `execution_audit` table: `(order_id, nbbo_bid, nbbo_ask, fill_price, fill_venue, algo_selected, algo_rationale, timestamp_ns)`. Populate on every fill.
- **Key files:** Fill processing, database schema
- **Acceptance criteria:**
  - [ ] Every fill records NBBO at time of execution
  - [ ] Execution algorithm selection rationale logged
  - [ ] Query available: "show all fills worse than NBBO midpoint"

### 6.7 Options Monitoring Rules

- **Finding:** DO-3 | **Severity:** HIGH | **Effort:** 2 days
- **Depends on:** Phase 2 item 2.12 (Greeks in risk gate)
- **Audit section:** [`03_EXECUTION_LAYER.md` §3.3](../03_EXECUTION_LAYER.md)
- **Problem:** All 6 exit rules are equity-centric. Missing: theta acceleration (DTE<7 + theta/premium>5%/day), pin risk (DTE<3 + near strike), assignment risk (short call ITM + ex-div within 2 days), IV crush (post-earnings + IV dropped >30%), max theta loss (cumulative decay >40%).
- **Fix:** Add options-specific exit evaluation after standard equity rules in `execution_monitor.py`. Wire to existing Greeks computation in `core/options/engine.py:293-364`.
- **Key files:** `src/quantstack/execution/execution_monitor.py`, `src/quantstack/core/options/engine.py`
- **Acceptance criteria:**
  - [ ] Options positions evaluated with theta acceleration, pin risk, IV crush
  - [ ] DTE < 3 near-the-money positions auto-flagged for exit/roll
  - [ ] Short call positions checked against ex-dividend calendar

### 6.8 Slippage Model Enhancement

- **Finding:** QS-E8 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`03_EXECUTION_LAYER.md`](../03_EXECUTION_LAYER.md)
- **Problem:** Current slippage model is basic half-spread + sqrt volume impact. No time-of-day variation, no market condition adjustment, no feedback from realized slippage.
- **Fix:** Integrate with TCA feedback (6.1). Add time-of-day slippage profiles. Use realized fills to calibrate model per symbol.
- **Key files:** Slippage model, paper broker
- **Acceptance criteria:**
  - [ ] Slippage model calibrated from realized fill data
  - [ ] Time-of-day variation in slippage estimates
  - [ ] Model accuracy tracked: predicted vs. realized slippage

### 6.9 Borrowing/Funding Cost Model

- **Finding:** QS-E11 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`03_EXECUTION_LAYER.md`](../03_EXECUTION_LAYER.md)
- **Problem:** No borrowing/funding cost model for short positions or leveraged positions.
- **Fix:** Add funding cost calculation to position P&L tracking. Account for margin interest and short borrow fees.
- **Key files:** Position P&L calculation, cost model
- **Acceptance criteria:**
  - [ ] Short positions include borrow fee estimate
  - [ ] Margin interest accounted for in position P&L
  - [ ] Funding costs visible in strategy performance metrics

---

## Dependencies

- **Depends on:** Phase 1 (safety), Phase 2 partial (Greeks in risk gate for 6.7)
- **6.7 depends on Phase 2 item 2.12** (options Greeks must be in risk gate first)
- **6.1 feeds into Phase 7** (TCA feedback loop is a learning loop)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 6.3: TWAP/VWAP adds complexity to order lifecycle | Start with TWAP (simpler). Add VWAP after TWAP proven. Keep IMMEDIATE as default for small orders. |
| 6.5: SEC compliance is complex, easy to get wrong | Start with wash sale + PDT (clear rules). Consult tax professional for edge cases. |
| 6.5: PDT enforcement may block legitimate day-trading strategies | Make PDT enforcement configurable by account type (margin vs. cash vs. >$25K) |

---

## Validation Plan

1. **TCA (6.1):** Execute 20 trades → verify cost model parameters diverge from defaults based on realized fills.
2. **TWAP (6.3):** Submit 1000-share TWAP over 30 min → verify 6 child orders at 5-min intervals in paper broker.
3. **SEC (6.5):** Sell AAPL at loss → buy AAPL within 30 days → verify wash sale flag + cost basis adjustment. Execute 3 round-trips in 5 days with <$25K account → verify 4th blocked.
4. **Options monitoring (6.7):** Hold option with DTE=2 near strike → verify pin risk exit flagged.
5. **Audit trail (6.6):** Query `execution_audit` → verify NBBO recorded for all fills.
