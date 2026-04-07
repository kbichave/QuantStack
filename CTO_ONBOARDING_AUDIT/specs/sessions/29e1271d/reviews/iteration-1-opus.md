# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06T00:00:00Z

---

## Summary Verdict

The plan is well-structured and covers all 9 spec items with reasonable technical depth. However, there are several gaps ranging from a critical state schema issue to missing migration coordination and under-specified error handling. The plan is roughly 80% production-ready -- the issues below need to be resolved before implementation begins.

---

## Critical Issues

### C1. TradingState uses `extra="forbid"` -- attribution node will crash on state merge

**Section:** 4 (Performance Attribution Node)

**Issue:** `TradingState` at `src/quantstack/graphs/state.py:83` has `model_config = ConfigDict(extra="forbid")`. The plan says the attribution node "returns updated state with attribution summary" but never specifies which new field(s) are added to `TradingState`. Any attempt to return a key not already declared on the model will raise a `ValidationError` at LangGraph merge time.

**Fix:** The plan must explicitly list the new field(s) to add to `TradingState` (e.g., `cycle_attribution: dict = {}`). This is a schema change to a Pydantic model with strict validation -- it needs to be called out as a prerequisite for Section 4, not left implicit.

---

### C2. No migration strategy for existing `loop_events` rows

**Section:** 7 (EventBus ACK Pattern)

**Issue:** The plan adds 4 new columns to `loop_events` but doesn't address what happens to the existing rows. With a 7-day TTL, old rows will have `requires_ack=NULL` (not `FALSE`). The ACK monitor query (`requires_ack=True AND acked_at IS NULL AND expected_ack_by < now()`) is safe because it checks `requires_ack=True`, but the plan should explicitly acknowledge this and confirm NULL-safety in the query. More importantly: the plan says `expected_ack_by` is set as `now + (consumer's cycle interval * 1.5)` -- but the publisher doesn't know the consumer's cycle interval. The publisher and consumer are different graphs.

**Fix:** (a) Document that NULLs on existing rows are safe. (b) The ACK timeout needs to be defined on the publisher side as a fixed duration per event type (e.g., risk events = 600s), not as "consumer's cycle interval * 1.5" which the publisher cannot know.

---

## Major Issues

### M1. `factor_exposure_history` table mentioned but never defined

**Section:** 3 (Factor Exposure Monitor)

**Issue:** Section 3.3 says "Results stored in a `factor_exposure_history` table for trend analysis" but Section 1 (Database Schema) does not define this table.

**Fix:** Add the `factor_exposure_history` table definition to Section 1.

---

### M2. `cycle_attribution` table mentioned but not in Section 1

**Section:** 4 (Performance Attribution Node)

**Issue:** Section 4.2 says "New DB table: `cycle_attribution`" but this table is not part of Section 1's schema definitions.

**Fix:** Move all table definitions to Section 1 or restructure dependency order.

---

### M3. `get_operating_mode()` doesn't specify graph routing mechanism

**Section:** 8 (Multi-Mode 24/7 Operation)

**Issue:** The plan says "The graph routing uses `get_operating_mode()` to decide which nodes to visit" but doesn't specify where this conditional logic lives. Is it a conditional edge? A router node?

**Fix:** Specify: likely a conditional edge after `data_refresh` that routes to full pipeline or truncated monitor-only subgraph.

---

### M4. Risk gate extended-hours check conflates order direction with position intent

**Section:** 8 (Multi-Mode 24/7 Operation)

**Issue:** The plan says check `order_side == 'buy'` for new entries. This is wrong for short positions: a sell can be a new short entry, a buy can be covering a short.

**Fix:** Check whether the order would *increase* or *decrease* absolute exposure, not just order side.

---

### M5. EDGAR CIK mapping is hand-waved

**Section:** 2 (Corporate Actions Monitor)

**Issue:** No concrete strategy for CIK-to-ticker mapping.

**Fix:** Use SEC's `company_tickers.json` endpoint. Cache at startup. Update weekly. Handle missing tickers gracefully.

---

### M6. No tests specified anywhere in the plan

**Section:** All

**Issue:** Zero tests specified for a financial system with strict invariants.

**Fix:** Each section needs unit tests (computation), integration tests (DB schema), regression tests (ACK edge cases).

---

## Minor Issues

### m1. `edgartools` library maturity unverified
**Section:** 2. Verify capabilities during implementation. Fallback: AV-only for v1.

### m2. LLM provider health check ping is wasteful
**Section:** 9. Track health from production calls instead of probing. Only probe if zero calls in N minutes.

### m3. Discord TODO needs a trigger condition
**Section:** 6. Use: "TODO(kbichave): Add Discord webhook when DISCORD_WEBHOOK_URL env var is set."

### m4. Fan-out semaphore (10) confused with rate limiter
**Section:** 10. Semaphore limits concurrency, not rate. Use both: semaphore for memory + existing AV rate limiter for quota.

### m5. `split_adjustments` unique constraint too narrow
**Section:** 1. Add `event_type` to `(symbol, effective_date)` constraint for correctness.

---

## Suggestions

### S1. Batch 2 parallelism is optimistic
Section 5 (system alerts) should move to Batch 1 since Sections 2, 3, 4, 7 all create system alerts.

### S2. No rollback plan for fan-out default flip
Note that `RESEARCH_FAN_OUT_ENABLED=false` env var override is the rollback mechanism.

### S3. Alpaca auto-adjusts splits -- risk of double adjustment
Plan's manual adjustment could double-adjust broker-managed positions. Need reconciliation check.

### S4. `factor_config` as key-value store is a smell
Consider typed Pydantic model with DB override instead of string parsing.

---

## Scorecard

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Completeness | 7/10 | All 9 items covered, but 2 tables and all tests missing |
| Correctness | 6/10 | State schema issue (C1), order-side conflation (M4), semaphore confusion (m4) |
| Edge cases | 5/10 | Alpaca double-adjustment, reverse split fractional, graph restart grace |
| Dependencies | 7/10 | Batch structure mostly right, Section 5 should be earlier |
| Testability | 3/10 | Zero tests for financial system |
| Risks | 7/10 | Good risk table but misses key items |
