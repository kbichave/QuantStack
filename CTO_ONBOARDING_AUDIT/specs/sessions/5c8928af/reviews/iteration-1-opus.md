# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06T00:00:00Z

---

## Executive Summary

The plan is thorough for a first pass, correctly identifies the psycopg3 migration as foundational, and gets the stop-loss enforcement layering right. However, it has three critical blind spots: (1) it underspecifies what happens between bracket entry fill and SL placement in the contingent fallback path — the window where the position is unprotected, (2) the psycopg3 migration plan ignores 15 files outside `db.py` that directly import psycopg2, and (3) the prompt sanitization approach is a blocklist (strip known bad patterns) when the threat model demands an allowlist (extract known good fields only). Several sections also conflate Phase 1 deliverables with Phase 2 aspirations, creating scope creep risk.

---

## Per-Section Findings

### Section 1: psycopg3 Migration

**Completeness: Gap.** The plan says "Grep for `import psycopg2` across the codebase" as a risk, but does not enumerate the results. There are **15 files** outside the plan docs that import psycopg2 directly: `db.py`, `pg_storage.py`, `rag/query.py`, `health/langfuse_retention.py`, 8 test files, `scripts/ewf_analyzer.py`, `scripts/heartbeat.sh`, and `stop.sh`. The plan treats this as a risk to investigate; it should be a concrete migration checklist with each file listed and its required changes scoped.

**Correctness issue.** The plan says `%s` placeholders "still work in psycopg3 (compatibility mode)." This is true, but the existing code in `trade_service.py` lines 248-254 uses `?` placeholders. The plan mentions the `?` to `%s` translation "can be removed if all callers already use `%s`" — but the callers do NOT all use `%s`. This needs a systematic audit and bulk replacement, not an "if" statement.

**Risk: `dict_row` behavioral change.** `DictCursor` returns `RealDictRow` which supports both key and index access. psycopg3's `dict_row` returns a plain `dict` — any code using integer indexing (e.g., `row[0]`) will break. The plan should mandate a grep for `row[0]`, `row[1]`, etc. patterns.

**Missing: rollback plan.** If the migration introduces subtle query behavior differences, what is the fallback? Keep psycopg2 as a pinned dependency for N days with a feature flag to revert the pool.

### Section 2: Mandatory Stop-Loss Enforcement

**Critical gap: Unprotected window.** Layer 3's contingent fallback path has an unprotected window between entry fill and SL submission. The plan does not specify: how the fill is detected (polling? WebSocket?), maximum latency, or SL submission retry policy.

**Correctness issue.** The plan says "Reject `OrderRequest` if `stop_price is None`" but `OrderRequest` does not have a `stop_price` field. `stop_price` is a separate parameter to `execute_trade()`. The validation must go on the function parameters, not the model.

**Over-engineering: E*Trade broker adapter.** System is paper-only with Alpaca. Is E*Trade actively used? If not, this is scope creep for Phase 1.

**Missing edge case: partial fills.** What if entry leg partially fills and SL leg is rejected? Need: cancel remaining entry quantity, submit standalone SL for filled quantity.

**Missing edge case: `client_order_id` collision.** Timestamp-based IDs could collide if two strategies signal same symbol within same second. Add random suffix or millisecond precision.

### Section 3: Prompt Injection Defense

**Correctness: Blocklist approach is fragile.** Stripping known patterns is trivially bypassable. The plan's Layer 4 (field-level extraction) is the right approach but is positioned as supplementary. It should be PRIMARY, with blocklist as secondary monitoring signal.

**Gap: Dual LLM separation is tool-category-level, not physical separation.** Enforced in same process by same executor function. A determined injection manipulating tool resolution could bypass it. Acknowledge limitation and state that risk gate + kill switch remain as hard code gates.

**Gap: No monitoring for injection attempts.** Sanitization should LOG when it strips something. External data source containing injection patterns should trigger anomaly alerting.

**Sequencing concern.** Migration order is backwards. Supervisor has least exposure to untrusted data. Research has MOST exposure. Research should be hardened FIRST.

### Section 4: Output Schema Validation with Retry

**CRITICAL: Safety check fallback is fail-OPEN.** `parse_json_response` for safety_check agent falls back to `{"halted": False}`, meaning parse failure bypasses safety check. This directly undermines "hard safety over soft checks" invariant. Every safety-critical fallback must fail CLOSED.

**Risk: Retry changes agent behavior.** Retried outputs should be flagged in audit trail so downstream consumers know the output came from retry path.

**Missing: Audit all fallback values.** Some fallbacks are dangerous (safety_check → `{"halted": False}`), some are safe (entry signals → `[]`). Need systematic review.

### Section 5: Non-Root Containers

**Missing: init process.** Add `init: true` in docker-compose. Without it, zombie processes accumulate with subprocess spawning.

**Missing: read-only root filesystem.** Add `read_only: true` with explicit tmpfs for `/tmp` and writable volume mounts.

### Section 6: Durable Checkpoints (PostgresSaver)

**Risk: Checkpoint data growth.** ~14,000 checkpoint rows/day across three graphs. No retention policy specified. Need pruning strategy.

**Correctness: Pool sizing discrepancy.** Spec says max_size=10, plan says max_size=6. Total connections (app pool + checkpoint pool) should be stated explicitly against PostgreSQL max_connections.

**Missing: `setup()` idempotency verification.** Verify from source whether `setup()` uses `CREATE TABLE IF NOT EXISTS`.

### Section 7: EventBus Integration

**Risk: EventBus publication from kill_switch.trigger() must be best-effort.** If PostgreSQL is down, EventBus publish fails — must not delay or prevent kill switch activation. Use try/except.

**Missing: `KILL_SWITCH_TRIGGERED` not in EventType enum.** Note as concrete code change, not conditional.

**Gap: Event polling idempotency.** Cursor must advance regardless of action taken. Halt check must be idempotent for re-processing.

### Section 8: Automated Database Backups

**Risk: Local-only backups don't protect against hardware failure.** State residual risk explicitly.

**Missing: WAL archive retention.** No size cap on WAL archive directory. Add pruning (e.g., 7-day retention).

**Missing: Backup lock.** Concurrent pg_dump jobs possible. Use flock.

### Section 9: Containerize Scheduler

**Risk: 2-hour stop_grace_period.** Deploys will wait up to 2 hours. Consider shorter grace period with resume-on-restart logic.

**Over-engineering: Flask health endpoint.** Python's `http.server` or file-age-based check is sufficient for Phase 1.

### Section 10: DB Transaction Isolation

**Risk: Lock timeout failure handling.** If execution monitor can't lock a position for trailing stop update, what's the retry policy?

**Missing: Deadlock ordering.** Two writers locking multiple rows in different order = deadlock. Specify single-row-at-a-time constraint.

**Gap: Kill switch position closer is a fifth write path.** Not listed in the plan.

### Section 11: Testing Strategy

**Gap: No load/stress testing.** Section 10 needs concurrent-writer stress test.

**Gap: No chaos/failure injection for Section 2.** Test broker API failures during contingent SL fallback path.

---

## Cross-Cutting Concerns

1. **Scope creep risk.** E*Trade adapter and parallel prompt comparison add significant Phase 2 work.
2. **Hidden dependency.** Section 2 (bracket_legs table) should use psycopg3 from day one — not truly parallelizable with Section 1.
3. **No end-to-end kill switch propagation test.** Kill switch → EventBus → all graphs halt → execution monitor stops → position closer fires.
4. **In-flight LLM calls during kill switch.** Already handled by guard() at execute_trade(), but should be explicitly called out.

---

## Top 5 Most Critical Issues

1. **P0: Safety check fallback is fail-OPEN.** Parse failure at safety_check results in `{"halted": False}`, bypassing the safety check entirely.
2. **P0: Unprotected window in contingent SL fallback.** No specification of fill detection mechanism, max latency, or retry policy between entry fill and SL placement.
3. **P1: 15 files import psycopg2 directly; none enumerated.** Migration will be incomplete or leave hybrid driver state.
4. **P1: Prompt sanitization uses blocklist.** Trivially bypassable. Field-level extraction (Layer 4) should be primary, not supplementary.
5. **P2: No checkpoint data retention policy.** ~420,000 rows/month with no pruning.
