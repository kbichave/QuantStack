# Integration Notes: Opus Review Feedback

## Integrating

### 1. Algo Scheduler Architecture — INTEGRATE
**Suggestion:** Extract algo scheduler into separate `algo_scheduler.py` instead of extending `order_lifecycle.py`.
**Why:** The reviewer correctly identified that the existing file header says "OMS separating from EMS" — the algo scheduler IS the EMS. Putting scheduling logic in the OMS file violates the system's own design principle and will push the file past 1,500 lines. The interview preference for "keep in one file" was reasonable but the reviewer's architectural argument is stronger.
**Action:** Plan will specify a separate `algo_scheduler.py` that imports OMS types and manages parent/child lifecycle. OMS knows orders can have children; scheduling logic lives elsewhere.

### 2. Sync/Async Impedance Mismatch — INTEGRATE
**Suggestion:** Address `BrokerProtocol.execute()` being sync while algo scheduler is async.
**Why:** This is a real gap. The execution monitor already solves this with `run_in_executor()`.
**Action:** Plan will specify `run_in_executor()` pattern for broker calls within the async scheduler, following the existing monitor's pattern.

### 3. Implementation Order: SEC Before TCA — INTEGRATE
**Suggestion:** Reorder to put 6.5 (SEC compliance) before 6.1 (TCA EWMA).
**Why:** Account is below $25K — PDT is regulatory risk on every trade. TCA EWMA is optimization. Regulatory compliance > optimization.
**Action:** Reorder to 6.2 → 6.5 → 6.6 → 6.1 → 6.3 → 6.4 → 6.8 → 6.7 → 6.9.

### 4. Wash Sale Look-Forward Fix — INTEGRATE
**Suggestion:** Use `pending_wash_losses` table pattern — flag losses as potentially washable, retroactively apply on subsequent buy.
**Why:** The plan incorrectly described checking future buys at time of sale. This is the standard correct implementation.
**Action:** Fix the wash sale design to use two-phase detection.

### 5. POV Algorithm Fallback — INTEGRATE
**Suggestion:** Document POV as out of scope with fallback to VWAP with participation cap.
**Why:** The plan covered TWAP/VWAP but left POV as a gap.
**Action:** Add explicit fallback: POV → VWAP with max_participation_rate capped at 5%.

### 6. Parent Order Cancellation and Recovery — INTEGRATE
**Suggestion:** Specify cancellation triggers and crash recovery for active parents.
**Why:** Without this, a kill switch or crash mid-TWAP leaves orphaned state.
**Action:** Add cancellation triggers (kill switch, risk halt, execution monitor exit) and startup recovery (query ACTIVE parents, cancel or resume).

### 7. Options PDT Counting — INTEGRATE
**Suggestion:** Match on exact OCC contract symbol, not underlying.
**Why:** This is a correctness issue — two different SPY options contracts are not the same day trade.
**Action:** Clarify PDT matching uses contract identifier.

### 8. Compliance Module Split — INTEGRATE
**Suggestion:** Split into `compliance/pretrade.py` and `compliance/posttrade.py`.
**Why:** Different lifecycles (pre-trade gates vs post-trade hooks) shouldn't be coupled.
**Action:** Use `execution/compliance/` package.

### 9. Business Day Calendar — INTEGRATE
**Suggestion:** Add a business-day calendar utility.
**Why:** PDT, TWAP scheduling, and wash sale all need business-day awareness.
**Action:** Add `exchange_calendars` or `pandas_market_calendars` dependency.

### 10. Audit Trail for IMMEDIATE Orders First — INTEGRATE
**Suggestion:** Implement audit trail for IMMEDIATE orders too, not just child fills.
**Why:** Gives immediate compliance value before TWAP/VWAP is ready.
**Action:** Move 6.6 earlier and specify it works for all order types.

## NOT Integrating

### A. Paper Broker Fill Price Noise Centering
**Suggestion:** Center noise around zero with slight adverse skew instead of always-adverse.
**Why not:** Conservative slippage estimation is intentional for paper trading. Overestimating slippage in paper mode is safer than underestimating — strategies that pass with conservative slippage are more likely to survive live. This is a feature, not a bug.

### B. PDT as Rolling Counter State Machine
**Suggestion:** Maintain a rolling counter instead of querying `day_trades` table.
**Why not:** Stateful counters are fragile — crashes, restarts, and manual trade adjustments can desync the counter. Querying the table is authoritative and with a 5-day window the query is trivial (max ~20 rows). Simplicity > micro-optimization here.

### C. Property-Based Testing (Hypothesis)
**Suggestion:** Use Hypothesis for parent/child state machine.
**Why not:** Good idea but adds a testing framework dependency. Standard pytest with explicit edge cases is sufficient for the state machine. Can add Hypothesis later if bugs appear in state transition logic.

### D. Adaptive EWMA Alpha
**Suggestion:** Make alpha adaptive based on fill frequency.
**Why not:** Over-engineering for initial implementation. Fixed alpha=0.1 is standard. If fill frequency issues appear in practice, tune later.

### E. Batching DB Writes for Child Fills
**Suggestion:** Batch non-critical writes (audit trail, TCA update) into async queue.
**Why not:** Premature optimization. With typical TWAP of 6-12 children, we're talking about ~60-80 writes over 30 minutes — trivial for PostgreSQL. Add batching if DB contention is measured.
