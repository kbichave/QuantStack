# Implementation Plan: QuantStack 24/7 Autonomous Trading Readiness

## Overview

QuantStack is an autonomous trading platform built on three LangGraph StateGraphs (Trading, Research, Supervisor) with 21 specialized agents, a 16-collector Signal Engine, a multi-layer Risk Gate, and a two-layer Kill Switch. It currently operates in attended paper-trading mode during market hours only. A comprehensive CTO audit identified 164 findings (38 critical, 60 high, 66 medium).

This plan transforms the system from attended market-hours paper trading to unattended 24/7 autonomous operation, culminating in supervised live trading with $5-10K conservative capital (max 5 concurrent positions).

**Go-live gate:** All three phases must be complete before real capital. ~8 weeks total.

**Cost trajectory:** ~$40-60/day (current) → ~$12/day (Groq hybrid) → ~$5-7/day (with prompt caching) → ~$3-5/day (with compaction).

---

## Pre-Work: Section 0 — Commit Baseline

**Problem:** 321 uncommitted files in the working tree. CI/CD is meaningless without a clean baseline. Every Phase 1 modification risks conflicting with uncommitted changes.

**Approach:** Commit all uncommitted files in logical batches. Tag the resulting commit as `v0.9-baseline` for rollback reference. Categories: execution layer, graph nodes, signal engine, tools, config, scripts, docs.

**Important: Verify current state.** The CTO audit findings were generated at a specific point in time. Some findings (C1 stop-loss, AC1 EventBus polling, HNSW index) may already be partially addressed in the uncommitted files. Each section below should start by verifying the current state of the files it modifies before making changes, to avoid duplicating work or regressing existing fixes.

**Rollback:** `git revert` individual batch commits if needed.

---

## Phase 1: Safety Hardening (Week 1-2)

*Goal: No position exists without protection. No agent output silently fails. Infrastructure is durable.*

**General instruction for all sections:** Before implementing, read the current state of files being modified. Several CTO audit findings may have been partially addressed in the recently committed baseline. Adjust scope accordingly.

### Section 1: Stop-Loss Enforcement & Bracket Orders

**Problem:** `trade_service.py` allows `stop_price=None` on `OrderRequest` (finding C1). Bracket orders silently degrade to market orders when the bracket API call fails (finding C2). This means positions can exist with no downside protection.

**Approach:**

Add validation at the OMS layer in `trade_service.py` that rejects any `OrderRequest` where `stop_price is None`. This is the single most important safety invariant in the system.

For bracket orders, implement `execute_bracket()` in `alpaca_broker.py` using Alpaca's native bracket API (OTO/OCO order types). When the bracket API fails, fall back to placing stop-loss and take-profit as separate contingent orders linked by `client_order_id`. Never allow a fill without a corresponding stop order.

**Files to modify:**
- `src/quantstack/execution/trade_service.py` — Add `stop_price` validation in `submit_order()`. Reject with `RiskViolation` if missing.
- `src/quantstack/execution/alpaca_broker.py` (lines 106-148) — Implement `execute_bracket()` with Alpaca's bracket/OTO API. Add fallback path for separate SL/TP orders.
- `src/quantstack/execution/paper_broker.py` — Mirror bracket simulation. Track linked stop orders.

**Invariant to verify:** After implementation, it must be impossible for `portfolio_state` to contain a position without a corresponding stop order in `open_orders`. A startup reconciliation check should verify this on every graph cycle start.

**Edge cases:**
- Partial fills: Stop order must be placed for the filled quantity, not the original order quantity
- Broker rejection of stop price (too close to market): Widen stop to minimum allowed distance, log warning
- Extended hours: Some stop types not available — use stop-limit as fallback

### Section 2: Agent Output Schema Validation

**Problem:** `agent_executor.py` `parse_json_response()` (lines 474-521) returns `{}` on parse failure. When `daily_plan` returns `{}`, the Trading Graph trades blind. When `fund_manager` returns `[]`, rejected entries are treated as approved.

**Approach:**

Define Pydantic output models for the 5 critical agent nodes. Use LangChain's `with_structured_output()` where the provider supports it (Anthropic/Bedrock: `method="json_schema"`). For Groq/Llama models, use `method="json_mode"` with post-hoc Pydantic validation as fallback.

On first parse failure: retry with schema hint appended to the prompt. On second failure: log a warning-level event with the raw LLM output, return a schema-compliant "safe default" (not `{}`). Safe defaults are conservative — e.g., `entry_scan` returns empty candidates (no entries), `fund_manager` returns all-reject verdicts.

**Critical output schemas to define:**

| Agent Node | Key Fields | Safe Default Behavior |
|-----------|------------|----------------------|
| `daily_plan` | plan (str), candidates (list[str]), risk_level (enum) | Conservative plan, no candidates |
| `entry_scan` | list of {symbol, signal_strength, thesis} | Empty list (no entries) |
| `position_review` | list of {symbol, thesis_intact (bool), action} | All positions flagged for review |
| `fund_manager` | list of {symbol, verdict (approve/reject), reasoning} | All entries rejected |
| `risk_sizing` | list of {symbol, action, size, stop_price} | All positions minimum size |

**Groq compatibility note:** The stakeholder decided to benchmark Groq structured output before committing. During Phase 1, implement both `json_schema` and `json_mode` paths in the agent executor. Run a benchmark across all 5 schemas with 100 sample prompts on Groq Llama 3.3 70B. If error rate > 5%, keep those agents on Haiku.

**Files to modify:**
- `src/quantstack/graphs/agent_executor.py` (lines 474-521) — Replace `parse_json_response()` with schema-aware parsing. Add retry-with-hint logic.
- `src/quantstack/tools/models.py` — Add Pydantic models for each agent output schema.

### Section 3: Deterministic Tool Ordering & Prompt Caching

**Problem:** Tool definitions are injected from `TOOL_REGISTRY` in dict iteration order (finding MC1). This means tool ordering varies between runs, invalidating the entire Anthropic prompt cache on every request. Zero prompt caching is configured (finding MC0c) despite infrastructure being partially in place.

**Approach — two sequential steps:**

**Step 1: Tool ordering (prerequisite).** In `tool_binding.py`, sort the tool list alphabetically by tool name before passing to the LLM. This is a 1-line change (`sorted(tools, key=lambda t: t.name)`). It must happen first because tool ordering changes invalidate the cache prefix.

**Step 2: Prompt caching.** In `build_system_message()` in `agent_executor.py` (lines 152-178), add `cache_control: {"type": "ephemeral"}` to the `SystemMessage`'s `additional_kwargs`. The system prompt contains the agent persona + tool definitions (static, cacheable) followed by current state (dynamic, not cacheable). The cache breakpoint goes after the static content.

For Bedrock specifically: pass `anthropic_beta: ["prompt-caching-2024-07-31"]` in the model kwargs. The constant `BEDROCK_PROMPT_CACHING_BETA` already exists in `llm/config.py` — wire it into model instantiation in `provider.py`.

Prompt caching only applies to the heavy tier (Anthropic/Bedrock). Groq/Llama models don't support it. No changes needed for medium/light/bulk tiers.

**Expected impact:** At current scale (~$126/day in system prompt tokens for heavy tier), prompt caching at 80%+ hit rate saves ~$100/day. Combined with Groq hybrid (already implemented), total daily cost drops from ~$40-60/day to ~$5-7/day.

**Files to modify:**
- `src/quantstack/tools/tool_binding.py` — Sort tool list before injection
- `src/quantstack/graphs/agent_executor.py` (lines 152-178) — Add cache_control to SystemMessage
- `src/quantstack/llm/provider.py` — Wire BEDROCK_PROMPT_CACHING_BETA into Bedrock model kwargs

**Verification:** After deployment, check Langfuse traces for `cache_creation_input_tokens` and `cache_read_input_tokens` fields. Cache hit rate should be > 80% within the first hour (5-min TTL auto-refreshes on hit).

### Section 4: Prompt Injection Defense

**Problem:** `trading/nodes.py` (lines 80-92) and `research/nodes.py` (lines 219-224) inject untrusted data (DB records, API responses, market data) directly into prompts via f-strings. An adversary who can influence market data descriptions, SEC filings, or community intel content could inject prompt instructions.

**Approach:**

Create a sanitization utility module `llm/sanitize.py` with a `sanitize_for_prompt(text: str) -> str` function that:
1. Escapes XML-like tags (prevents closing/opening instruction blocks)
2. Strips known prompt injection patterns (e.g., "ignore previous instructions")
3. Truncates to a configurable max length per field

Replace f-string interpolation in node functions with structured XML-tagged templates:
```
<market_data>
{sanitized_data}
</market_data>
```

The XML tags create clear boundaries between instructions and data, making it harder for injected content to be interpreted as instructions.

**Files to modify:**
- New: `src/quantstack/llm/sanitize.py` — Sanitization utility
- `src/quantstack/graphs/trading/nodes.py` (lines 80-92) — Use structured templates
- `src/quantstack/graphs/research/nodes.py` (lines 219-224) — Use structured templates

### Section 5: Database Backups & Durable Checkpoints

**Problem:** ALL system state lives in PostgreSQL with zero backup procedure (finding OC1). All three graphs use in-process `MemorySaver` — container crash loses all graph state (finding GC1).

**Approach — two independent work items:**

**Backups:** Add a backup sidecar service to `docker-compose.yml` that runs `pg_dump` daily at 01:00 ET and retains 7 days of backups locally. Create `scripts/pg_backup.sh` for the dump logic and `scripts/pg_restore_test.sh` for monthly restore verification. WAL archiving can be added later for point-in-time recovery (Phase 4).

**PostgresSaver migration:** Direct cutover from MemorySaver (stakeholder decision — paper trading only, restart if corrupted). Use `AsyncPostgresSaver` from `langgraph-checkpoint-postgres` with `AsyncConnectionPool`. Each graph gets its own connection pool (2-5 connections). Call `checkpointer.setup()` on startup (idempotent, creates tables if missing).

The `thread_id` config parameter identifies graph runs for crash recovery. On restart, the graph resumes from the last completed super-step. Existing MemorySaver state is not migrated — new graph runs start fresh.

**Pool sizing consideration:** 3 graphs × 5 connections = 15 new PG connections. Verify `max_connections` in PostgreSQL config accommodates this plus existing connections from the application, signal engine, and scheduler.

**Prerequisite — Idempotency guards:** Before enabling PostgresSaver, add `client_order_id`-based deduplication to `trade_service.submit_order()` and `execute_bracket()`. LangGraph checkpoint recovery replays from the last super-step, which means side-effecting nodes (order submission, DB writes) can re-execute after a crash. Without idempotency, a crash after `execute_entries` fills an order but before checkpoint commit causes a duplicate order on resume. Use Alpaca's `client_order_id` parameter to detect and reject duplicate submissions.

**Files to modify:**
- `src/quantstack/execution/trade_service.py` — Add client_order_id deduplication (PREREQUISITE)
- `docker-compose.yml` — Add backup sidecar service
- New: `scripts/pg_backup.sh` — Daily pg_dump logic
- New: `scripts/pg_restore_test.sh` — Restore verification
- `src/quantstack/graphs/trading/graph.py` — Swap MemorySaver → AsyncPostgresSaver
- `src/quantstack/graphs/research/graph.py` — Same
- `src/quantstack/graphs/supervisor/graph.py` — Same

**Rollback:** Revert graph.py changes to use MemorySaver. Idempotency guards remain (they're valuable regardless).

### Section 6: EventBus Wiring & CI/CD

**Problem:** Trading Graph never polls EventBus (finding AC1). Kill switch doesn't publish events (finding AC2). CI is disabled (finding OC3). The scheduler runs in tmux, not Docker (finding OC2).

**Approach:**

**EventBus wiring:** Add `bus.poll()` call in Trading Graph's `safety_check` node for events: `IC_DECAY`, `RISK_EMERGENCY`, `KILL_SWITCH_TRIGGERED`. Add `bus.publish("KILL_SWITCH_TRIGGERED", ...)` in `kill_switch.trigger()`. These are small additions (~5-10 lines each) to existing functions.

**Containerize scheduler:** The scheduler service definition already exists in `docker-compose.yml` (port 8422). Add a health check endpoint to `scripts/scheduler.py` and configure restart policy (`restart: unless-stopped`) in docker-compose.

**Enable CI/CD:** Rename `.github/workflows/ci.yml.disabled` → `ci.yml` and `release.yml.disabled` → `release.yml`. Verify the pipeline runs tests, type checks, and builds the Docker image.

**Files to modify:**
- `src/quantstack/graphs/trading/nodes.py` — Add bus.poll() in safety_check
- `src/quantstack/execution/kill_switch.py` — Add event publish in trigger()
- `docker-compose.yml` — Add health check + restart to scheduler service
- `scripts/scheduler.py` — Add health check endpoint
- `.github/workflows/ci.yml` — Re-enable (rename from .disabled)
- `.github/workflows/release.yml` — Re-enable

---

## Phase 2: Operational Resilience (Week 3-4)

*Goal: System runs unattended for days. LLM outages, stale data, and correlated positions are handled automatically.*

### Section 7: LLM Circuit Breaker & Runtime Failover

**Problem:** Provider availability is checked at startup only (finding LH2). A mid-session 429 or 500 error crashes the entire graph cycle with no recovery.

**Approach:**

Create a `CircuitBreaker` class in `llm/circuit_breaker.py` that wraps `get_chat_model()`. It maintains per-provider health state (in-memory, no DB needed since provider health is transient).

Failover logic:
1. On retryable error (429, 500, timeout): retry same provider with exponential backoff (2 attempts, 1s → 2s)
2. On 3rd failure: switch to next provider in `FALLBACK_ORDER` (`["bedrock", "anthropic", "openai", "groq", "ollama"]`)
3. Mark failed provider as "cooling down" for 5 minutes
4. Non-retryable errors (400, 401, 403): fail immediately, don't retry

Use LangChain's `with_fallbacks()` for the provider chain, wrapped in the CircuitBreaker for cooldown/health tracking. The existing `FALLBACK_ORDER` in `provider.py` defines the priority.

**Files to modify:**
- New: `src/quantstack/llm/circuit_breaker.py` — CircuitBreaker class with per-provider health state
- `src/quantstack/llm/provider.py` — Wrap `get_chat_model()` return with fallback chain

### Section 8: Email Alerting System

**Problem:** No alerting exists. Critical events (kill switch, drawdown, system errors) go unnoticed until manual check.

**Approach:**

Create a lightweight email alerting module using Gmail SMTP (stakeholder decision). Three alert levels:
- **INFO**: Daily digest, strategy registrations, overnight research summary
- **WARNING**: Threshold approaching (80% of daily loss limit), data staleness, failed retries
- **CRITICAL**: Kill switch triggered, circuit breaker activated (daily or portfolio), system errors

```python
@dataclass
class AlertConfig:
    smtp_host: str  # smtp.gmail.com
    smtp_port: int  # 587
    sender_email: str
    app_password: str  # Gmail app password (env var)
    recipient_email: str
```

Rate limiting: Max 1 email per event type per 15 minutes to prevent alert storms. CRITICAL level bypasses rate limiting.

**Fallback:** When SMTP fails (network issue, app password revoked), write alerts to a local file (`/var/log/quantstack/alerts.log`) that can be tailed. This ensures critical events are never lost silently even if email is down.

**Files to create/modify:**
- New: `src/quantstack/alerting/email_sender.py` — SMTP wrapper with rate limiting + file fallback
- New: `src/quantstack/alerting/alert_manager.py` — Alert routing by level, deduplication
- `.env.example` — Add GMAIL_APP_PASSWORD, ALERT_RECIPIENT_EMAIL

**Rollback:** Disable email sender, fall back to file-only alerts.

### Section 9: Risk Gate Enhancements

**Problem:** Multiple risk gate gaps identified: pre-trade correlation is post-hoc only (H1), no market hours hard gating (H2), no daily notional deployment cap (H3).

**Approach — three additions to the existing `risk_gate.check()` pipeline:**

**Pre-trade correlation:** The method `_check_pretrade_correlation()` already exists but runs post-hoc. Move it into the `check()` pipeline's pre-trade sequence. If pairwise correlation with any existing position exceeds 0.7, apply a 50% concentration haircut to the proposed position size.

**Market hours gating:** Add a time check using the existing `TradingWindow` enum from `trading_window.py`. Hard-reject orders outside the configured trading window unless `extended_hours=True` is explicitly set. Default windows: equity 9:30-16:00 ET, options 9:30-16:15 ET.

**Daily notional cap:** Track cumulative new notional deployed each day. Default cap: 30% of equity. With $5-10K capital, this means ~$1.5-3K max new deployment per day. Reset at market open. Tracked in-memory with DB persistence at EOD for recovery.

**Files to modify:**
- `src/quantstack/execution/risk_gate.py` — Add all three checks to the `check()` pipeline

### Section 10: Signal & Data Quality Gates

**Problem:** Signal cache holds stale data for 1 hour while intraday refresh runs every 5 minutes (DC1). Collectors compute signals on arbitrarily stale data (DC3). 92 of 122 tools are stubbed (TC1).

**Approach:**

**Cache invalidation:** Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` at the end of each intraday refresh cycle. When fresh data arrives, the old cached signals are immediately invalidated.

**Data staleness rejection:** Add a freshness gate in `signal_engine/engine.py` before running collectors. Each collector checks `data_metadata.last_timestamp`. If data is staler than a configurable threshold (default: 2x the expected refresh interval), the collector returns an empty result with a staleness warning.

**Tool registry cleanup:** Identify the top 10-15 most impactful stubbed tools by analyzing which tools agents actually attempt to call (from Langfuse traces). Split `TOOL_REGISTRY` into `ACTIVE_TOOLS` (working) and `PLANNED_TOOLS` (stubs). Agent configs in `agents.yaml` only bind to active tools. Implement the top 10-15 stubs in Phase 2-3 based on call frequency data.

**Per-agent temperature:** Add `temperature` field to agent configs in `agents.yaml`. Hypothesis generation: 0.7, debate: 0.3-0.5, validation/execution: 0.0. Read the temperature in the agent executor and pass to model instantiation.

**Files to modify:**
- `src/quantstack/data/scheduled_refresh.py` — Add cache invalidation after refresh
- `src/quantstack/signal_engine/engine.py` — Add freshness gate before collectors
- `src/quantstack/tools/registry.py` — Split into ACTIVE_TOOLS and PLANNED_TOOLS
- `src/quantstack/graphs/*/config/agents.yaml` — Remove stubs from bindings, add temperature
- `src/quantstack/graphs/agent_executor.py` — Read temperature from agent config

### Section 11: Kill Switch Auto-Recovery & Log Aggregation

**Problem:** Kill switch requires manual `reset()` after trigger (OH3). Logs go to local Docker json-file driver only (OH2).

**Approach:**

**Kill switch recovery:** Add tiered recovery logic:
1. On trigger: send CRITICAL email alert immediately
2. Classify trigger reason (transient vs permanent)
3. Transient conditions (broker reconnected, data refreshed, brief API outage): auto-reset after 30-minute cooldown
4. Permanent conditions (daily loss limit, consecutive failures): require manual reset. Escalate email after 4 hours if not reset.

The trigger reason classification uses the `AutoTriggerMonitor`'s 4 conditions. Broker disconnect and data staleness are transient. Daily loss and consecutive failures are permanent.

**Log aggregation:** Wire the existing Fluent-bit, Loki, and Grafana services in `docker-compose.yml` (they're defined but not connected). Create a Fluent-bit config that ships container logs to Loki. Add a Grafana alert rule for ERROR rate spikes (>10 errors in 5 minutes) that sends email via the alerting module from Section 8.

**Files to modify:**
- `src/quantstack/execution/kill_switch.py` — Add tiered recovery logic, integrate email alerting
- `docker-compose.yml` — Wire Fluent-bit → Loki → Grafana
- New: `config/fluent-bit.conf` — Log shipping configuration
- New: `config/grafana/alerts/error_rate.yaml` — Alert rules

---

## Phase 3: Autonomy (Week 5-8)

*Goal: System operates in three modes, self-improves from outcomes, and manages its own compute budget.*

### Section 12: Multi-Mode Operation

**Problem:** All three graphs run the same way regardless of time. Research competes with trading for LLM budget during market hours. No overnight compute utilization.

**Approach:**

Add a `ScheduleMode` enum with three values: `MARKET_HOURS`, `EXTENDED_HOURS`, `OVERNIGHT_WEEKEND`. The graph runners check the current mode at each cycle start and route accordingly:

- **Market Hours** (9:30-16:00 ET Mon-Fri): Trading Graph runs full pipeline. Research Graph runs light (no heavy compute). Supervisor monitors all.
- **Extended Hours** (16:00-20:00, 04:00-09:30 ET): Trading Graph runs position monitoring only (no new entries). Research Graph processes earnings, EOD sync. Supervisor runs health checks.
- **Overnight/Weekend** (20:00-04:00 ET, weekends): Trading Graph idle. Research Graph runs heavy compute (ML training, autoresearch, community intel). Supervisor runs strategy lifecycle.

Mode detection uses `exchange_calendars` (already a dependency — `pyproject.toml` line 63) for market schedule awareness, including holidays and early closes. **All time checks must use `America/New_York` timezone-aware datetimes**, not naive UTC offsets. DST transitions (EST↔EDT) shift windows by one hour — naive UTC offsets will gate incorrectly twice per year.

**Files to modify:**
- `src/quantstack/graphs/trading/graph.py` — Add mode-aware conditional edges
- `src/quantstack/graphs/research/graph.py` — Add overnight routing for heavy compute
- `src/quantstack/graphs/supervisor/graph.py` — Add mode-aware behavior

### Section 13: Overnight Autoresearch & Error-Driven Iteration

**Problem:** Research is human-initiated only. No systematic alpha discovery. Losses don't inform future research (finding AC1-2).

**Approach — two complementary systems:**

**Autoresearch loop:** A new node in the Research Graph activated during overnight/weekend mode. Each experiment has a fixed 5-minute budget (wall clock). Hypothesis generation uses Haiku tier (cheap, fast). Single success metric: out-of-sample Information Coefficient (IC) on a purged holdout set. Winners (IC > 0.02) are registered as `draft` status → the morning strategy pipeline validates with a full backtest.

Budget allocation per stakeholder decision: 70% of nightly compute on new hypotheses, 30% on refining existing winning strategies (parameter tuning, expanding asset coverage).

At ~96 experiments/night, this produces 50+ validated hypotheses per week for the morning pipeline to evaluate.

**Loss analyzer:** A new node in the Supervisor Graph that runs daily at 16:30 ET. It classifies each losing trade by failure mode: `regime_shift`, `signal_failure`, `thesis_wrong`, `sizing_error`, `entry_timing`, `theta_burn`. Aggregates 30-day failure frequencies and generates prioritized research tasks targeting the top failure modes. These tasks feed the `research_queue` table, which the autoresearch loop consumes.

**Files to modify/create:**
- New: `src/quantstack/graphs/research/autoresearch_node.py` — Autoresearch loop logic
- `src/quantstack/graphs/research/graph.py` — Add overnight mode routing to autoresearch
- New: `src/quantstack/graphs/supervisor/loss_analyzer.py` — Loss classification and research task generation
- `src/quantstack/graphs/supervisor/nodes.py` — Wire loss_analyzer at 16:30 ET
- DB: New tables `autoresearch_experiments`, `loss_classifications`

### Section 14: Budget Tracking & Context Compaction

**Problem:** No per-agent budget limits. Agents can consume unlimited tokens/time. After parallel merge points, downstream agents receive bloated context from all branches.

**Approach:**

**Budget tracker:** Create `graphs/budget_tracker.py` that tracks per-cycle token count, wall-clock time, and estimated cost per agent. When budget is exhausted, the tracker signals a graceful exit at the next node boundary (not mid-generation). Agent configs in `agents.yaml` get `max_tokens_budget` and `max_wall_clock_seconds` fields.

**Context compaction:** After the two parallel merge points in the Trading Graph (`merge_parallel` and `merge_pre_execution`), add compaction nodes. These use Haiku tier to summarize branch outputs into a concise brief for downstream agents. Expected 40-60% context size reduction. The compaction prompt extracts: key decisions, action items, and risk flags from each branch, discarding verbose analysis.

**Files to modify/create:**
- New: `src/quantstack/graphs/budget_tracker.py` — Per-agent budget tracking
- `src/quantstack/graphs/agent_executor.py` — Integrate budget checking at each node
- `src/quantstack/graphs/*/config/agents.yaml` — Add budget fields
- `src/quantstack/graphs/trading/graph.py` — Add compaction nodes after merges
- `src/quantstack/graphs/trading/nodes.py` — New `compact_context()` node function

### Section 15: Layered Circuit Breaker & Greeks Risk

**Problem:** No intraday circuit breaker (QS-E5). Options risk checks only look at DTE + premium, not Greeks (QS-E3). Stakeholder wants layered protection (daily P&L + portfolio high-water mark).

**Approach:**

**Layered circuit breaker:** Two independent threshold layers in `execution_monitor.py`:

*Daily P&L layer* (resets each morning):
- -1.5% unrealized+realized → halt new entries
- -2.5% → begin systematic exit (close weakest positions first)
- -5% → emergency liquidation of all positions

*Portfolio HWM layer* (tracks from equity high-water mark):
- -3% from HWM → halt all new trading
- -5% from HWM → defensive exit: close all positions at market, trigger kill switch, send CRITICAL email

When the portfolio HWM layer triggers defensive exit, behavior per stakeholder decision: close everything, kill switch, email alert. **Important: use limit orders with a wide collar (midpoint - 1%) for emergency liquidation, not market orders.** Market orders during a flash crash / liquidity gap can amplify losses significantly (e.g., -5% trigger becomes -10% realized). If limit orders are not filled within 60 seconds, trigger a dead-man's switch: escalate to kill switch (halt all activity) and send CRITICAL email alert. This prevents the system from being stuck in a "trying to liquidate" state indefinitely.

**Greeks integration:** Wire the existing `core/risk/options_risk.py` (444 lines) into `risk_gate.py`'s options path. Add portfolio-level limits: max absolute delta exposure, gamma limit (scales with portfolio size), vega limit, daily theta budget (max theta decay acceptable per day). With $5-10K capital, start with tight limits that prevent options from dominating the portfolio.

**Files to modify:**
- `src/quantstack/execution/execution_monitor.py` — Add both circuit breaker layers
- `src/quantstack/execution/risk_gate.py` — Integrate Greeks manager, add portfolio-level limits

### Section 16: Knowledge Base, IC Tracking & Urgency Channel

**Problem:** `search_knowledge_base` queries by recency, ignoring the query parameter (MC0). No signal validation against forward returns (QS-S1). Supervisor → Trading has 5-10 min poll latency (GC2).

**Approach:**

**Knowledge base fix:** In `tools/langchain/learning_tools.py` (lines 25-31), replace the SQL recency query with a call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`. Add an HNSW index on the embeddings table for fast vector similarity search. This is functionally a one-line fix plus a DB migration.

**Signal IC tracker:** Create `signal_engine/ic_tracker.py` that computes daily Information Coefficient for all 22 signal collectors. IC = rank correlation between signal strength and forward returns (1-day, 5-day, 20-day horizons). Store in the existing `signal_ic` table. Gate: if `rolling_63d_IC < 0.02` for a collector, disable it from the synthesis step. Re-enable if IC recovers above 0.03 (hysteresis to prevent flapping).

**Urgency channel:** Add a PG `LISTEN/NOTIFY`-based urgent channel to EventBus. Supervisor sends `NOTIFY quantstack_urgent, '{event_json}'` for urgent events. Trading Graph listens on the `quantstack_urgent` channel and receives events sub-second (vs 5-10 min polling). This is zero-dependency (PG already running), works across Docker containers (no shared filesystem needed), and avoids the race conditions inherent in file-based sentinels.

**Files to modify/create:**
- `src/quantstack/tools/langchain/learning_tools.py` — Fix search_knowledge_base
- DB migration: Add HNSW index on embeddings table
- New: `src/quantstack/signal_engine/ic_tracker.py` — IC computation and gating
- `src/quantstack/signal_engine/engine.py` — Integrate IC gating
- `scripts/scheduler.py` — Add daily IC computation job
- `src/quantstack/coordination/event_bus.py` — Add file-sentinel urgent channel
- `src/quantstack/graphs/trading/nodes.py` — Check urgent channel pre-execution

---

## Phase 4: Scale & Self-Improvement (Week 9+)

*Deferred until Phases 1-3 are validated. Included for completeness.*

Phase 4 adds compounding intelligence:
- **Alpha Knowledge Graph** — PostgreSQL JSON node/edge schema populated from strategy/trade/intel data
- **Meta-Improvement Layer** — 4 weekly meta-agents (prompt optimizer, threshold tuner, tool selector, architecture critic)
- **Hierarchical Governance** — Split fund_manager into CIO + risk_officer, specialize execution agents. 74% token reduction.
- **Parallel Research Streams** — 4 concurrent streams on weekends (~2688 experiments/weekend vs ~12 today)
- **Adversarial Consensus** — 3-agent voting for positions > $5K (bull advocate, bear advocate, neutral arbiter)
- **Feature Factory** — Autonomous feature enumeration → IC screening → monitoring. 500+ candidates → 50-100 curated.
- **TCA Feedback Loop** — Daily slippage recalibration, Almgren-Chriss parameter updates per symbol/time-of-day

---

## Dependency Graph

Implementation order within each phase matters. Key dependencies:

```
Phase 1 (must be sequential where noted):
  Section 3 (tool ordering) BEFORE Section 3 (prompt caching)  ← same section, sequential steps
  Section 5 (DB backups) BEFORE Section 5 (PostgresSaver)      ← same section, backup first
  Section 1 (stop-loss) is independent — can start immediately
  Section 2 (schemas) is independent — can start immediately
  Section 4 (injection defense) is independent
  Section 6 (EventBus + CI) is independent

Phase 2 (after Phase 1):
  Section 8 (email alerting) BEFORE Section 11 (kill switch recovery uses email)
  Section 7 (circuit breaker) is independent
  Section 9 (risk gate) is independent
  Section 10 (signal/data/tools) is independent

Phase 3 (after Phase 2):
  Section 12 (multi-mode) BEFORE Section 13 (autoresearch needs overnight mode)
  Section 16 IC tracker BEFORE Section 13 (autoresearch uses IC > 0.02 as gate)
  Section 14 (budget) is independent
  Section 15 (circuit breaker + Greeks) is independent
  Section 16 KB fix and urgency channel are independent of other Phase 3 work
```

---

## Testing Strategy

Each section gets its own test file. Tests cover:

1. **Safety invariants** (Section 1): No order without stop-loss. Bracket fallback works. Partial fill handling.
2. **Schema validation** (Section 2): Parse failure retries. Safe defaults are conservative. Groq compatibility benchmark.
3. **Caching** (Section 3): Tool order determinism. Cache hit rate measurement.
4. **Sanitization** (Section 4): Known injection patterns blocked. XML escaping works.
5. **Durability** (Section 5): PostgresSaver checkpoint/resume cycle. Backup script produces valid dumps. **Critical integration test:** Full Trading Graph cycle with PostgresSaver → kill process mid-cycle after execute_entries → restart → verify no duplicate orders (idempotency) and correct state recovery.
6. **EventBus** (Section 6): Kill switch event propagates to Trading Graph. CI pipeline passes.
7. **Failover** (Section 7): LLM provider switch on 429. Cooldown and recovery.
8. **Alerting** (Section 8): Email sends on CRITICAL. Rate limiting prevents storms.
9. **Risk gate** (Section 9): Correlated entry rejected. Off-hours order rejected. Notional cap enforced.
10. **Data quality** (Section 10): Stale cache invalidated. Stale data rejected by collector.
11. **Recovery** (Section 11): Transient kill switch auto-resets. Permanent stays triggered.
12. **Multi-mode** (Section 12): Correct mode detected for each time window.
13. **Autoresearch** (Section 13): Low-IC hypotheses rejected. Loss classification correct.
14. **Budget** (Section 14): Agent exits at budget boundary. Compaction reduces context size.
15. **Circuit breaker** (Section 15): Daily and portfolio layers trigger at correct thresholds.
16. **Knowledge/IC/Urgency** (Section 16): Semantic search returns relevant results. IC gating disables weak collectors. Urgency sentinel detected sub-second.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Groq Llama structured output quality | Benchmark in Phase 1. Revert to Haiku if >5% error rate |
| PostgresSaver corrupts state | Paper trading only. Restart graph cycle. Backup exists. |
| Prompt cache misses due to dynamic content | Separate static/dynamic. Measure in Langfuse. |
| Overnight autoresearch garbage | IC > 0.02 gate. Morning validation pipeline. |
| Kill switch auto-recovery too eager | 30-min cooldown. Monitor false positive rate. |
| Gmail SMTP rate limits | 500 emails/day limit is sufficient. Rate-limit at app level. |
| 321 uncommitted files | Commit in logical batches before Phase 1. Tag baseline. |

---

## Success Criteria (Go-Live Gate)

Before deploying real capital ($5-10K), ALL must be true:

- [ ] No position exists without stop-loss protection
- [ ] Agent outputs validated with Pydantic, retry on failure, conservative defaults
- [ ] Prompt cache hit rate > 80% on heavy tier
- [ ] Daily pg_dump running, restore tested
- [ ] All 3 graphs on PostgresSaver with crash recovery verified
- [ ] LLM failover works across providers within 30s
- [ ] Layered circuit breaker active (daily P&L + portfolio HWM)
- [ ] Greeks enforced in risk gate for options
- [ ] Email alerting tested end-to-end for all CRITICAL events
- [ ] Three operating modes functional
- [ ] Overnight research producing 50+ experiments/night
- [ ] Signal IC computed daily, weak collectors gated
- [ ] 7 consecutive days of unattended paper trading without kill switch trigger
- [ ] Positive paper P&L over 30-day validation period
