# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06

---

# Critical Review: Phase 6 Execution Layer Completion Plan

## 1. Completeness

### Gaps Identified

**A. Sync/Async Impedance Mismatch (Major Gap)**
The plan proposes an async execution loop for TWAP/VWAP child order scheduling (Section 7.8), but `BrokerProtocol.execute()` is a synchronous method. The `ExecutionMonitor` already solves this via `loop.run_in_executor()`, but the plan never addresses this mismatch. The algo scheduler must either run broker calls in an executor or the `BrokerProtocol` needs an async variant.

**B. `order_lifecycle.py` Uses `threading.RLock`, Not Asyncio**
The OMS is thread-safe via `RLock`. Adding an async scheduler loop inside the same file creates a footgun — acquiring an RLock inside an async coroutine blocks the event loop. Needs explicit design: either the scheduler runs in a separate thread with proper locking, or the OMS needs asyncio-native locks.

**C. POV Algorithm Unaddressed**
Algo selection has four outcomes: IMMEDIATE, TWAP, VWAP, POV. The plan covers TWAP and VWAP but says nothing about POV. Orders exceeding 5% ADV will still be selected as POV and presumably fall through to IMMEDIATE.

**D. No Cancellation Path for Active Algo Parents**
The plan describes parent states including CANCELLING/CANCELLED, but never specifies how cancellation is triggered. What happens if the risk gate triggers a halt mid-TWAP? What happens if the execution monitor fires a stop-loss while a parent order is still active?

**E. Wash Sale "Look-Forward" Problem**
You cannot check if a symbol was bought within 30 days *after* a sale at time of sale — you cannot know the future. The correct implementation: (1) at time of loss sale, flag the loss as potentially washable, and (2) at time of any subsequent buy within 30 days of a flagged loss, retroactively apply wash sale treatment.

**F. Options PDT Counting Missing Detail**
Options contracts are identified by full OCC symbol (e.g., `SPY240119C00450000`), not by underlying symbol. The PDT checker must match on the exact contract, not the underlying.

### Missing Requirements

- No rollback/cleanup strategy for partially completed algo parents on system restart
- No rate limiting for Alpaca IEX quote calls in NBBO capture for audit trail
- No mention of how `fill_legs` interacts with `PortfolioState.upsert_position()`

## 2. Correctness

**A. TCA EWMA Alpha Value Concern**
With alpha=0.1, half-life ≈ 6.6 observations. For a symbol trading once/day, EWMA is dominated by last week. Should alpha be adaptive based on fill frequency?

**B. Conservative Multiplier Math**
At 0 fills, there is no EWMA value to multiply — the fallback to default A-C coefficients applies. The decay description needs clarification.

**C. Paper Broker Fill Price Model**
Always-adverse noise will systematically overestimate slippage. More realistic: center noise around zero with slight adverse skew.

**D. Margin Calculation Oversimplification**
"Options: max_loss is the margin requirement" is only correct for long options and debit spreads. Credit spreads have different margin requirements.

## 3. Architecture

**A. Putting Everything in `order_lifecycle.py` is a Mistake**
The file is already 760 lines. The plan adds AlgoParentOrder, ChildOrder, TWAP/VWAP scheduling, async execution loop, and performance tracking — easily 1,500+ lines. The existing file header says "explicit state machine separating OMS from EMS." The algo scheduler IS the EMS. Recommend separate `algo_scheduler.py`.

**B. Compliance Module Coupling**
PDT and Margin are pre-trade gates; Wash Sale and Tax Lots are post-trade hooks. Bundling into one `compliance.py` creates unnecessary coupling. Consider `execution/compliance/` package.

**C. Six New Tables with No Migration Framework Mentioned**
The codebase uses `run_migrations`. How are these structured, versioned, and applied in Docker?

## 4. Missing Risks

- Data feed reliability during 30-minute TWAP execution
- Clock skew and scheduling drift
- PDT false negatives from multi-leg options spreads
- TCA EWMA cold start for new symbols
- Database contention under TWAP (420+ writes in 30 minutes per parent)

## 5. Implementation Order

**SEC Compliance (6.5) Should Come Before TCA EWMA (6.1)**. PDT is "critical path" because account < $25K. Every trade without PDT enforcement is regulatory risk. Reorder to: 6.2 → 6.5 → 6.6 → 6.1 → 6.3 → 6.4 → 6.8 → 6.7 → 6.9.

**Audit Trail (6.6) Could Be Implemented Earlier** for IMMEDIATE orders — gives compliance value now.

## 6. Testing

- No property-based testing for parent/child state machine invariants
- No stress/load testing for concurrent TWAP parents
- PDT testing needs business-day calendar-aware fixtures
- No regression test strategy across sections

## 7. Specific Suggestions

1. Extract algo scheduler into `algo_scheduler.py` — respects OMS/EMS separation
2. Add `BrokerProtocol.execute_async()` or standardize on `run_in_executor()` wrapping
3. Add business-day calendar utility (PDT, TWAP scheduling, wash sale)
4. Implement PDT as rolling counter state machine, not per-trade query
5. Add `system_restart_recovery()` for active parent orders
6. Batch DB writes for non-critical child fill operations
7. Define margin calculation per option strategy type explicitly
8. Add circuit breaker to algo scheduler (3+ consecutive child failures → pause all parents)
9. Maintain `pending_wash_losses` table for look-forward wash sale tracking
10. Make 99.5% completion threshold configurable with option to submit remainder as IMMEDIATE

## Summary Assessment

Solid B+ plan. Strong domain understanding. Main weaknesses: architectural (algo scheduler in OMS file), concurrency model (sync/async mismatch), implementation ordering (SEC before TCA), and recovery edge cases. All fixable.
