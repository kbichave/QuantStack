# Integration Notes — Opus Review Feedback

## Suggestions INTEGRATED

### 2a. Idempotency guards before PostgresSaver migration
**Integrating:** Add `client_order_id` deduplication to `trade_service.submit_order()` as a prerequisite in Section 5. This is a real safety issue — checkpoint replay could re-execute orders.
**Change:** Add to Section 5 as a prerequisite step.

### 2b. Replace file-sentinel with PG LISTEN/NOTIFY
**Integrating:** The Opus reviewer is right — file sentinels break across Docker containers. PG LISTEN/NOTIFY is zero-dependency and already available.
**Change:** Replace file-sentinel approach in Section 16 with PG LISTEN/NOTIFY.

### 2d. Limit orders for emergency liquidation
**Integrating:** Market orders during flash crash can amplify losses. Use limit orders with wide collar for emergency liquidation.
**Change:** Update Section 15 to use limit orders (midpoint - 1% collar) for emergency path.

### 3b. Autoresearch depends on IC Tracker
**Integrating:** The dependency graph was wrong. IC tracker must be available before autoresearch can gate hypotheses.
**Change:** Update dependency graph. Section 16 IC tracker runs before or parallel with Section 13 autoresearch.

### 3d. Uncommitted files as Section 0
**Integrating:** 321 uncommitted files must be committed before anything else. CI is meaningless without this.
**Change:** Add Section 0 (pre-work: commit baseline) at the top of Phase 1.

### 5d. Alert fallback (local file)
**Integrating:** Gmail SMTP single point of failure. Add a local file fallback for critical alerts.
**Change:** Update Section 8 to include file-based fallback logging when SMTP fails.

### 5e. Timezone handling
**Integrating:** DST transitions could shift all trading windows by an hour. Must use `America/New_York` timezone-aware datetimes.
**Change:** Add explicit timezone requirement to Section 12.

### 7a. Idempotency to trade_service
**Same as 2a above — integrated.**

### 7f. Dead-man's switch for circuit breaker liquidation
**Integrating:** If liquidation orders aren't confirmed within 60s, trigger kill switch.
**Change:** Add to Section 15.

### 7g. Rollback procedures
**Integrating:** Each section should have a 1-line revert path.
**Change:** Add rollback notes to each section.

## Suggestions NOT INTEGRATED (with reasoning)

### 1b. WAL archiving in Phase 1
**Not integrating:** WAL archiving is a nice-to-have when pg_dump already provides daily backups. With $5-10K and paper trading, daily backups are sufficient. WAL adds operational complexity. Explicitly descoped to Phase 4.

### 1c. Paper validation failure iteration loop
**Not integrating in plan:** This is an operational decision, not an implementation task. If 30-day paper validation fails, the team investigates root causes and adjusts. Adding a formal "what to do if validation fails" section would be speculative — the response depends on WHY it failed.

### 2c. bedrock_groq in FALLBACK_ORDER
**Not integrating as stated:** `bedrock_groq` is a hybrid provider, not a standalone fallback target. The circuit breaker should fallback between the underlying providers (bedrock, groq, etc.), not to the hybrid. The plan's omission is correct behavior.

### 3a. Schema validation depends on temperature
**Not integrating:** These are weakly coupled, not strongly dependent. The Groq benchmark tests structured output at temperature=0.0 (execution agents). Non-zero temperature agents (hypothesis generation) are less schema-critical. Can be evaluated independently.

### 5a, 5b, 5c. Stale audit findings / re-audit
**Partially integrated:** The reviewer found that some findings (C1 stop-loss, AC1 EventBus polling, HNSW index) may already be addressed in the 321 uncommitted files. However, these files are UNCOMMITTED — they haven't been tested, reviewed, or validated. The plan should still scope the work items but add a "verify current state" step at the start of each section to avoid duplicate work. Adding this as a general instruction rather than removing sections.

### 6b. Greeks to Phase 2
**Not integrating:** With $5-10K conservative capital and max 5 positions, options exposure is naturally limited. Greeks enforcement in Phase 3 is appropriate — Phase 2 focuses on operational resilience (failover, alerting, data quality). The risk is bounded by position limits.

### 6c. Knowledge base to Phase 2
**Not integrating:** The knowledge base returning recency-based results is suboptimal but not dangerous. Agents still get relevant results (recent trades are often the most relevant). Phase 3 timing is appropriate.

### 7c. exchange_calendars dependency
**Already in pyproject.toml** (line 63: `"exchange-calendars>=4.5.0"`). The reviewer's concern was based on incomplete search.

### 7e. Budget tracker for autoresearch first
**Not integrating as separate item:** The budget tracker design in Section 14 already specifies per-experiment ceiling as the primary use case. No structural change needed.

## Summary of Plan Changes

1. **New Section 0** — Commit uncommitted files as baseline before Phase 1
2. **Section 5** — Add idempotency guards (client_order_id dedup) as prerequisite for PostgresSaver
3. **Section 8** — Add local file fallback for alerts when SMTP fails
4. **Section 12** — Specify `America/New_York` timezone-aware datetimes explicitly
5. **Section 15** — Limit orders for emergency liquidation. Dead-man's switch (60s confirm timeout → kill switch)
6. **Section 16** — PG LISTEN/NOTIFY instead of file sentinel
7. **Dependency graph** — IC tracker (Section 16) before autoresearch (Section 13)
8. **General** — Add "verify current state" instruction to each section. Add rollback notes per section.
