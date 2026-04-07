# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-07

---

## Implementation Plan Review: QuantStack 24/7 Autonomous Trading Readiness

### 1. Missing Requirements

**1a. Spec section 2.9 (Stubbed Tool Registry Cleanup) references a `tool_health` table.** The plan (Section 10) mentions splitting the registry into `ACTIVE_TOOLS`/`PLANNED_TOOLS` but never mentions the `tool_health` table from the spec. This table is needed for runtime monitoring of tool availability and failure rates, which feeds the self-healing loop.

**1b. Spec section 1.5 mentions WAL archiving for point-in-time recovery.** The plan (Section 5) explicitly defers WAL to Phase 4 but then also says "WAL archiving can be added later" as if it's optional. The spec lists it as Phase 1 scope. This should be explicitly called out as a descope with rationale, or moved back in.

**1c. The spec requires "positive paper P&L over 30-day validation period" as a go-live gate (section 9).** The plan's success criteria reproduce this but nothing in the plan describes what happens if the 30-day paper validation fails. There is no rollback plan or iteration loop. This is an 8-week plan that could easily become 12+ weeks if the paper validation fails, and the plan provides zero guidance on what to do.

**1d. The spec (section 2.10) mentions per-agent temperature configuration.** The plan (Section 10) includes this but only in a brief bullet. Temperature misconfiguration on execution agents (non-zero temperature on `fund_manager` or `risk_sizing`) is a safety issue, not a tuning preference. There should be a validation check that execution-path agents have `temperature=0.0`.

### 2. Architectural Risks

**2a. Section 5 -- PostgresSaver direct cutover with "restart if corrupted" is riskier than stated.** The plan says "paper trading only" as mitigation, but the real risk is not data loss -- it is that a corrupted checkpoint causes an infinite retry loop or a graph that silently skips nodes. LangGraph's checkpoint recovery replays from the last super-step, which means side-effecting nodes (order submission, DB writes, email sends) can re-execute. The plan never addresses idempotency of side-effecting nodes. This is a real production risk: a crash after `execute_entries` fills an order but before the checkpoint commits means the resumed graph re-executes the entry.

**Recommendation:** Add idempotency guards (client_order_id deduplication) to `trade_service.submit_order()` and `execute_bracket()` before the PostgresSaver migration. This should be a Section 5 prerequisite.

**2b. Section 16 -- File-sentinel urgency channel is fragile.** The plan proposes a file-based sentinel for sub-second Supervisor-to-Trading communication. This breaks on NFS/Docker volumes with caching, is subject to filesystem poll latency, and creates a race condition if the sentinel is deleted before the Trading Graph reads it. More importantly, the three graphs already run in separate Docker containers -- file sentinels require shared volume mounts that may not exist.

**Recommendation:** Use the existing PostgreSQL `LISTEN/NOTIFY` mechanism instead. It's zero-dependency (PG is already running), sub-second, and doesn't require shared filesystem. The `event_bus.py` already uses PG.

**2c. Section 7 -- The circuit breaker failover chain includes `bedrock_groq` in `FALLBACK_ORDER`.** From the codebase, `FALLBACK_ORDER` is `["bedrock", "anthropic", "openai", "groq", "ollama", "bedrock_groq"]`. The plan says the fallback chain is `["bedrock", "anthropic", "openai", "groq", "ollama"]` -- it omits `bedrock_groq`. This matters because `bedrock_groq` is the primary hybrid provider used for most agents. If it's not in the fallback chain correctly, the circuit breaker won't cycle through it.

**2d. Section 15 -- Emergency liquidation at -5% daily P&L sends market orders for all positions.** With $5-10K and 5 positions, this is manageable. But the plan doesn't address the scenario where the -5% trigger fires during a flash crash when market orders would get terrible fills. A market order during a liquidity gap can turn a -5% loss into -10%+. Consider using limit orders with a wide collar (e.g., midpoint minus 1%) instead of pure market orders for the emergency liquidation path.

### 3. Dependency Gaps

**3a. Section 2 (Agent Output Schema Validation) depends on Section 10 (Per-Agent Temperature).** If Groq structured output benchmarking reveals that structured output fails at non-zero temperature, the decision about which agents stay on Haiku vs. move to Groq is temperature-dependent. The plan treats these as independent but they are coupled.

**3b. Section 13 (Autoresearch) depends on Section 16 (Signal IC Tracker).** The autoresearch loop uses IC > 0.02 as its gate criterion, but IC computation is built in Section 16. The plan's dependency graph says Section 16 is independent within Phase 3, but autoresearch cannot validate hypotheses without the IC tracker. Section 16's IC tracker should be sequenced before or parallel with Section 13, not "independent."

**3c. Section 8 (Email Alerting) is listed as Phase 2, but Section 11 (Kill Switch Auto-Recovery) also in Phase 2 depends on it.** The plan's dependency graph correctly notes this (`Section 8 BEFORE Section 11`), but does not address the fact that Section 15 (Layered Circuit Breaker, Phase 3) also sends CRITICAL emails. If Phase 3 starts before email alerting is fully tested, the circuit breaker's defensive exit has no notification path.

**3d. The 321 uncommitted files are a blocking dependency for everything.** The plan acknowledges this in the Risks table but treats it as a background task ("commit in logical batches before Phase 1"). This should be Section 0. CI/CD (Section 6) is meaningless if the working tree has 321 uncommitted changes -- the first CI run will fail or produce unpredictable results. Every section in Phase 1 modifies files that may conflict with uncommitted changes.

### 4. Testing Gaps

**4a. No integration test for the full Trading Graph cycle.** The testing strategy lists per-section unit tests, but the most critical test is an end-to-end Trading Graph cycle with PostgresSaver that: starts, processes a full pipeline, crashes mid-cycle (kill -9), restarts, and verifies no duplicate orders and correct state recovery. This is conspicuously absent.

**4b. No test for the bracket order fallback path under concurrent execution.** Section 1 describes a fallback from bracket API to separate SL/TP contingent orders. The test plan says "Bracket fallback works" but doesn't cover the race condition: what happens if the entry fills, the bracket API fails, and before the fallback SL order is placed, the price moves through the intended stop level? This is a real-money edge case.

**4c. No load test for the overnight autoresearch loop.** Section 13 claims "~96 experiments/night." Each experiment involves LLM calls, data fetching, IC computation, and DB writes. With Haiku at ~$0.25/M input tokens and a 5-minute budget, there's no validation that 96 experiments actually fit within the nightly window or the LLM budget. A single experiment that hangs (LLM timeout, data fetch stall) could block the entire queue.

**4d. No test for prompt caching invalidation.** Section 3's verification is "check Langfuse traces for cache fields." But the real failure mode is silent: the cache key changes due to a non-obvious dynamic element in the "static" portion of the prompt, and you get 0% hit rate with no error. There should be an automated test that verifies cache key stability across N consecutive calls with the same agent config.

### 5. Operational Risks

**5a. Plan says stop-loss is not enforced (finding C1), but the codebase already enforces it.** At `trade_service.py` line 121, there is already a `ValueError` for entry orders without `stop_price`. Similarly, `execute_bracket()` already exists in `alpaca_broker.py` (line 158+) with the Alpaca bracket API implementation. And `_check_pretrade_correlation()` is already wired into the `check()` pipeline at line 597 of `risk_gate.py`. The plan appears to be working from audit findings that have since been addressed. **This means Phase 1 Section 1 and Phase 2 Section 9's correlation work may already be done.** If the team implements these sections without reading the current code, they will either duplicate work or introduce regressions by replacing working code.

**Recommendation:** Re-audit findings C1, C2, and H1 against the current codebase before starting Phase 1. The git status shows these files are modified but uncommitted.

**5b. The EventBus `bus.poll()` is already called in all three graphs.** The plan (Section 6) says "Trading Graph never polls EventBus" but the codebase shows `bus.poll()` in `trading/nodes.py:52`, `supervisor/nodes.py:297`, and `research/nodes.py:41`. The finding AC1 may be stale.

**5c. The HNSW index migration already exists.** The plan (Section 16) says "Add an HNSW index on the embeddings table" but `db.py` line 2558+ already has `_migrate_hnsw_index_pg()` creating this exact index. Section 16's KB fix scope should be re-evaluated.

**5d. Gmail SMTP is a single point of failure for critical alerts.** The plan uses Gmail as the sole alerting channel. If the SMTP connection fails (network issue, app password revoked, Google rate limit), all alerts are lost silently. There is no fallback. Even a simple fallback to writing alerts to a local file that can be tailed would provide a safety net.

**5e. No mention of timezone handling in the multi-mode scheduler.** Section 12 defines modes by ET time ranges, but the Docker containers may run in UTC. Daylight saving transitions (EST to EDT) shift all windows by an hour. The plan should specify that all time checks use `America/New_York` timezone-aware datetime, not naive UTC offsets.

### 6. Ordering Issues

**6a. Section 0 (uncommitted files) must come before everything.** As noted in 3d above.

**6b. Greeks integration (Section 15, Phase 3) should move to Phase 2.** The spec says the system will trade options. The risk gate currently only checks DTE and premium for options. Without Greeks limits, a paper trading system could accumulate dangerous gamma/vega exposure during Phase 2's "unattended for days" goal. If the system is trading options in paper mode during Phase 2, Greeks enforcement should be there too.

**6c. Knowledge base fix (Section 16, Phase 3) should move to Phase 2.** A broken knowledge base means agents cannot learn from past trades. This undermines the Phase 2 goal of "runs unattended for days" because agents make increasingly uninformed decisions as the knowledge base returns irrelevant results.

**6d. IC tracker (Section 16, Phase 3) partially already exists.** The `supervisor/nodes.py` already computes IC and writes to `signal_ic`. The `ic_retirement.py` module already gates on ICIR. The plan should acknowledge the existing implementation and scope Section 16's IC work as extending/fixing the existing system, not building from scratch.

### 7. Specific Suggestions

**7a. Add idempotency to `trade_service.submit_order()`.** Before PostgresSaver migration (Section 5), add `client_order_id` based deduplication. File: `src/quantstack/execution/trade_service.py`. This prevents duplicate orders on checkpoint replay.

**7b. Replace file-sentinel urgency with PG LISTEN/NOTIFY.** File: `src/quantstack/coordination/event_bus.py`. Zero new dependencies, sub-second latency, works across containers.

**7c. Add `exchange_calendars` to dependencies.** Section 12 says it's "already a dependency" but verify it's in `pyproject.toml` before relying on it.

**7d. Add a structured re-audit step before Phase 1.** Many findings (C1, C2, H1, AC1, MC0 HNSW) appear to have been partially or fully addressed in the 321 uncommitted files. A 1-day re-audit against the current working tree would prevent significant wasted effort and accidental regressions.

**7e. Section 14 (Budget Tracker) should enforce hard limits for the autoresearch loop first.** A runaway experiment during overnight compute could exhaust the entire daily LLM budget before market open. The budget tracker should be specifically designed for the autoresearch use case (fixed per-experiment ceiling) before the general per-agent budget system.

**7f. Add a "circuit breaker for the circuit breaker."** Section 15's emergency liquidation is itself a critical path. If the liquidation orders fail (broker down), the system has no fallback. Add a dead-man's switch: if liquidation orders are not confirmed filled within 60 seconds, trigger a kill switch that prevents all activity and sends a CRITICAL alert.

**7g. Define explicit rollback procedures for each phase.** The plan has no rollback section. If PostgresSaver causes issues, how do you revert to MemorySaver? If prompt caching breaks agent behavior, how do you disable it? Each section should have a 1-line "revert path" documented.

### Summary of Critical Findings

The most impactful issues, in priority order:

1. **Stale audit findings (5a, 5b, 5c):** Multiple sections propose building things that already exist. A re-audit is essential before starting.
2. **Missing idempotency for checkpoint replay (2a):** This is a real-money bug waiting to happen.
3. **Uncommitted files blocking everything (3d):** 321 files must be committed before CI or any collaborative work can proceed.
4. **Autoresearch depends on IC tracker (3b):** The dependency graph is wrong; these are not independent.
5. **File-sentinel fragility across Docker containers (2b):** Use PG LISTEN/NOTIFY instead.
6. **No end-to-end crash recovery test (4a):** The most dangerous operational scenario has no test coverage.
7. **Emergency liquidation during liquidity gaps (2d):** Market orders in a flash crash can amplify losses.
