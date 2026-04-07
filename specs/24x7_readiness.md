# Spec: QuantStack 24/7 Autonomous Trading Readiness

**Author:** CTO Audit + Principal Quant Scientist findings (consolidated)
**Date:** 2026-04-07
**Status:** DRAFT — input spec for `/deep-plan`
**Source:** `CTO_AUDIT_FINDINGS.md` (164 findings: 38 CRITICAL, 60 HIGH, 66 MEDIUM)

---

## Objective

Take QuantStack from attended market-hours-only paper trading to unattended 24/7 autonomous operation capable of managing real capital. The system currently scores **C-** overall (D+ on execution, D- on signal rigor, F on feedback loops). Target: **B+** within 10 weeks, sufficient for supervised live trading with position limits.

---

## Current System State

### What Works (don't break these)

| Component | Status | Location |
|-----------|--------|----------|
| Signal Engine (16 collectors, fault-tolerant) | Production-grade | `src/quantstack/signal_engine/` |
| Risk Gate (daily loss, position caps, gross exposure) | Production-grade | `src/quantstack/execution/risk_gate.py` |
| Kill Switch (two-layer: DB + sentinel file) | Production-grade | `src/quantstack/execution/kill_switch.py` |
| 21 specialized agents across 3 LangGraph graphs | Complete | `src/quantstack/graphs/*/config/agents.yaml` |
| Execution Monitor (deterministic exit rules) | Complete | `src/quantstack/execution/execution_monitor.py` |
| Strategy lifecycle (draft→backtested→forward_testing→live→retired) | Complete | Strategy pipeline |
| SPAN margin module (537 lines, 16-scenario stress) | Complete | `src/quantstack/core/risk/span_margin.py` |
| Options Greeks tracking | Complete | `src/quantstack/core/risk/options_risk.py` |
| Docker Compose (9 services, health checks) | Complete | `docker-compose.yml` |
| Data pipeline (14-phase, idempotent, rate-limited) | Complete | `src/quantstack/data/acquisition_pipeline.py` |
| LLM tier routing (heavy/medium/light/bulk) | Complete | `src/quantstack/llm/provider.py` |
| Groq hybrid provider (bedrock_groq) | Just added | `src/quantstack/llm/config.py` |

### What's Broken (the 164 findings, distilled)

**321 uncommitted files** in the working tree. No CI/CD. No automated tests on deploy.

---

## Phase 1: Safety Hardening (P0 — Week 1-2)

*Goal: System can paper trade without losing money to bugs.*

### 1.1 Mandatory Stop-Loss Enforcement

**Finding:** C1 — `trade_service.py:99, 212-220` allows `stop_price=None`
**What:** Reject any `OrderRequest` where `stop_price is None` at OMS level. If bracket order fails, place SL/TP as separate contingent orders. Never allow an unprotected position.
**Files:**
- `src/quantstack/execution/trade_service.py` — add validation
- `src/quantstack/execution/alpaca_broker.py:106-148` — implement `execute_bracket()` using Alpaca's native bracket API
- `src/quantstack/execution/paper_broker.py` — simulate bracket fills
**Tests:** Kill switch blocks execute_order. Risk gate violation blocks order. Bracket failure falls back to separate SL/TP orders.
**Invariant:** No position can exist without a stop-loss order.

### 1.2 Output Schema Validation with Retry

**Finding:** LC2 — `agent_executor.py:474-521` silently returns `{}` on parse failure
**What:** Add Pydantic output models per agent. On parse failure, retry once with schema hint. Log all fallback events as warnings. Never silently continue with empty defaults.
**Files:**
- `src/quantstack/graphs/agent_executor.py` — add schema validation + retry
- New Pydantic models per agent output (can go in existing `tools/models.py` or per-graph)
**Critical outputs that must validate:**
| Node | Expected Schema | Current Fallback | Impact of Fallback |
|------|----------------|-----------------|-------------------|
| daily_plan | `{"plan": ..., "candidates": [...]}` | `{}` | Trades blind |
| entry_scan | `[{symbol, signal_strength}]` | `[]` | Entries missed |
| position_review | `[{symbol, thesis_intact}]` | `[]` | Positions unmonitored |
| fund_manager | `[{symbol, verdict}]` | `[]` | Rejected entries treated as approved |
| risk_sizing | `[{symbol, action}]` | `[]` | Risk assessment skipped |

### 1.3 Deterministic Tool Ordering (1-Line Fix)

**Finding:** MC1 — Non-deterministic tool ordering in `tools/registry.py` breaks prompt cache
**What:** Sort tool definitions alphabetically by name before injection in `tool_binding.py`.
**Files:** `src/quantstack/tools/tool_binding.py` — add `sorted()` call
**Impact:** 30-50% reduction in prompt token costs from improved cache hits. At current scale (~$126/day in system prompt tokens), saves ~$50-60/day.

### 1.4 Prompt Caching Enablement

**Finding:** MC0c — Zero prompt caching configured despite Anthropic/Bedrock support
**What:** Add `cache_control` breakpoints to system message construction. For Bedrock: use `anthropic_beta: ["prompt-caching-2024-07-31"]` header.
**Files:**
- `src/quantstack/graphs/agent_executor.py:152-178` — add cache_control to SystemMessage
- `src/quantstack/llm/provider.py` — already has `BEDROCK_PROMPT_CACHING_BETA` constant, wire it up
**Impact:** 90% cost reduction on cached input tokens. $126/day → ~$12.60/day.

### 1.5 Automated Database Backups

**Finding:** OC1 — ALL system state lives in PostgreSQL with zero backup procedure
**What:** Daily `pg_dump` → S3 upload. WAL archiving for point-in-time recovery. Test restore monthly.
**Files:**
- `docker-compose.yml` — add backup sidecar service or cron
- New: `scripts/pg_backup.sh`
- New: `scripts/pg_restore_test.sh`

### 1.6 Prompt Injection Defense

**Finding:** LC1 — `trading/nodes.py:80-92`, `research/nodes.py:219-224` inject untrusted data via f-strings
**What:** Replace f-string interpolation with structured XML-tagged sections. Sanitize all DB/API-sourced data before prompt injection. Escape any content that could contain prompt instructions.
**Files:**
- `src/quantstack/graphs/trading/nodes.py:80-92` — structured templates
- `src/quantstack/graphs/research/nodes.py:219-224` — structured templates
- New: `src/quantstack/llm/sanitize.py` — input sanitization utilities

### 1.7 Containerize Scheduler

**Finding:** OC2 — `scripts/scheduler.py` runs in tmux. If it crashes, 13 critical jobs stop silently.
**What:** Add `scheduler` service to docker-compose.yml with health check + restart policy.
**Files:**
- `docker-compose.yml` — add scheduler service
- `scripts/scheduler.py` — add health check endpoint

### 1.8 Durable Checkpoints (MemorySaver → PostgresSaver)

**Finding:** GC1 — All three graphs use in-process `MemorySaver`. Container crash = lost state.
**What:** Switch to LangGraph's `PostgresSaver` for durable checkpointing. Enables crash recovery from exact node.
**Files:**
- `src/quantstack/graphs/trading/graph.py` — swap checkpointer
- `src/quantstack/graphs/research/graph.py` — swap checkpointer
- `src/quantstack/graphs/supervisor/graph.py` — swap checkpointer
**Dependency:** `langgraph-checkpoint-postgres` package

### 1.9 EventBus Wiring (Close Feedback Loops)

**Finding:** AC1, AC2 — Trading Graph never polls EventBus. Kill switch doesn't publish events.
**What:**
- Trading Graph `safety_check` node: add `bus.poll()` for `IC_DECAY`, `RISK_EMERGENCY`, `KILL_SWITCH_TRIGGERED`
- Kill switch `trigger()`: publish `KILL_SWITCH_TRIGGERED` to EventBus
**Files:**
- `src/quantstack/graphs/trading/nodes.py` — add bus.poll at safety_check (~5-10 lines)
- `src/quantstack/execution/kill_switch.py` — add event publish (~3 lines)
- `src/quantstack/coordination/event_bus.py` — add event type if needed

### 1.10 Enable CI/CD

**Finding:** OC3 — `.github/workflows/ci.yml.disabled`, `release.yml.disabled`
**What:** Re-enable CI pipeline. Run test suite + type check + build image on every push to main.
**Files:**
- `.github/workflows/ci.yml` — re-enable
- `.github/workflows/release.yml` — re-enable

---

## Phase 2: Operational Resilience (P1 — Week 3-4)

*Goal: System can run unattended for days without human intervention.*

### 2.1 LLM Runtime Failover (Per-Call, Not Startup-Only)

**Finding:** LH2 — Provider availability checked at startup only. Mid-session 429/500 crashes the cycle.
**What:** On 429/500/timeout, retry same provider 2x → switch to next provider in chain → cooldown failed provider 5 min. Uses existing fallback chain in `provider.py`.
**Files:**
- `src/quantstack/llm/provider.py` — wrap `get_chat_model()` with runtime retry+failover
- New: `src/quantstack/llm/circuit_breaker.py` — per-provider health state, cooldown logic

### 2.2 Signal Cache Auto-Invalidation

**Finding:** DC1 — 1hr cache TTL, but intraday refresh every 5 min. Stale signals used for decisions.
**What:** Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after each intraday refresh cycle.
**Files:**
- `src/quantstack/data/scheduled_refresh.py` — add cache invalidation after refresh
- `src/quantstack/signal_engine/cache.py` — verify invalidation method works

### 2.3 Pre-Trade Correlation Check

**Finding:** H1 — Correlation is post-hoc monitoring only, not pre-trade gate.
**What:** Add pairwise correlation check in `risk_gate.check()` against existing portfolio. If `corr > 0.7` with existing position, apply 50% concentration haircut to sizing.
**Files:**
- `src/quantstack/execution/risk_gate.py` — add pre-trade correlation gate

### 2.4 Market Hours Hard Gating

**Finding:** H2 — Alpaca warns but allows orders outside market hours.
**What:** Hard-reject orders outside configurable trading windows unless `extended_hours=True`.
**Files:**
- `src/quantstack/execution/risk_gate.py` or `order_lifecycle.py` — add time gate
- Uses existing `TradingWindow` enum from `trading_window.py`

### 2.5 Portfolio Heat Budget

**Finding:** H3 — No cap on total new notional deployed in a single day.
**What:** Add `max_daily_notional_deployed` limit (default 30% of equity/day for new positions).
**Files:**
- `src/quantstack/execution/risk_gate.py` — add daily notional tracking + gate

### 2.6 Kill Switch Auto-Recovery

**Finding:** OH3 — Once triggered, requires manual `reset()`. No auto-recovery even for transient conditions.
**What:** Tiered recovery: (1) Discord alert immediately, (2) investigate root cause, (3) if transient (broker reconnected, data refreshed), auto-reset after configurable cooldown, (4) escalate to email/SMS after 4 hours.
**Files:**
- `src/quantstack/execution/kill_switch.py` — add tiered recovery logic
- `src/quantstack/coordination/daily_digest.py` — add `urgent_alert()` for immediate Discord

### 2.7 Log Aggregation + Alerting

**Finding:** OH2 — Logs go to local Docker json-file driver only. No centralized shipping or alerting.
**What:** Add Fluent-bit sidecar → Loki (already in docker-compose.yml). Add alerts for ERROR rate spikes.
**Files:**
- `docker-compose.yml` — wire up existing Fluent-bit/Loki/Grafana services
- New: `config/fluent-bit.conf`
- New: `config/grafana/alerts/error_rate.yaml`

### 2.8 Stubbed Tool Registry Cleanup

**Finding:** TC1 — 92 of 122 tools are stubbed. Agents waste LLM calls on tools that return errors.
**What:** Split `TOOL_REGISTRY` into `ACTIVE_TOOLS` (30 working) and `PLANNED_TOOLS` (92 stubs). Agents only bind to active tools. Add `tool_health` table for monitoring.
**Files:**
- `src/quantstack/tools/registry.py` — split registry
- `src/quantstack/graphs/*/config/agents.yaml` — remove stubbed tools from agent bindings

### 2.9 Per-Agent Temperature Configuration

**Finding:** LC3 — Temperature hardcoded to 0.0 for all 21 agents.
**What:** Add `temperature` field to agent configs. Hypothesis generation=0.7, debate=0.3-0.5, validation/execution=0.0.
**Files:**
- `src/quantstack/graphs/*/config/agents.yaml` — add temperature per agent
- `src/quantstack/graphs/agent_executor.py` or graph builders — read temperature from config

### 2.10 Data Staleness Rejection

**Finding:** DC3 — Collectors compute signals on arbitrarily stale data without checking freshness.
**What:** Each collector checks `data_metadata.last_timestamp`. Return `{}` if data staler than configurable threshold.
**Files:**
- `src/quantstack/signal_engine/engine.py` — add freshness gate before running collectors
- Individual collectors in `src/quantstack/signal_engine/collectors/` — add freshness check

---

## Phase 3: Autonomy (P2 — Week 5-8)

*Goal: System operates in multiple modes (market hours, extended, overnight) and self-improves.*

### 3.1 Multi-Mode Operation

**What:** Add three operating modes to graph runners:
- **Market Hours** (9:30-16:00 ET Mon-Fri): Full trading + research pipelines
- **Extended Hours** (16:00-20:00, 04:00-09:30 ET): Position monitoring only, EOD sync, earnings processing
- **Overnight/Weekend** (20:00-04:00 ET, weekends): Full research compute, ML training, community intel, no trading
**Files:**
- `src/quantstack/graphs/trading/graph.py` — mode-aware runner
- `src/quantstack/graphs/research/graph.py` — mode-aware runner (heavy compute overnight)
- `src/quantstack/graphs/supervisor/graph.py` — mode-aware behavior

### 3.2 Overnight Autoresearch Loop (AR-1)

**What:** Fixed 5-min budget per experiment. ~96 experiments/night. Haiku for hypothesis generation. Single metric: OOS IC on purged holdout. Winners registered as draft → morning strategy pipeline validates.
**Files:**
- New: `src/quantstack/graphs/research/autoresearch_node.py`
- `src/quantstack/graphs/research/graph.py` — add overnight mode routing
- New DB table: `autoresearch_experiments`

### 3.3 Error-Driven Iteration (AR-7)

**What:** Daily loss analysis pipeline. Classify losses by failure mode taxonomy (regime_shift, signal_failure, thesis_wrong, sizing_error, entry_timing, theta_burn, etc.). Aggregate 30-day failure frequencies. Generate prioritized research tasks targeting top failure modes. Feed `research_queue`.
**Files:**
- New: `src/quantstack/graphs/supervisor/loss_analyzer.py`
- `src/quantstack/graphs/supervisor/nodes.py` — add loss_analyzer node (daily 16:30 ET)
- New DB table: `loss_classifications`

### 3.4 Experiment Budget Protocol (AR-9)

**What:** Per-cycle token/wall-clock/cost budgets per agent. When budget exhausted, graceful exit at next node boundary. Prioritization formula for research queue tasks.
**Files:**
- New: `src/quantstack/graphs/budget_tracker.py`
- `src/quantstack/graphs/agent_executor.py` — integrate budget tracking
- `src/quantstack/graphs/*/config/agents.yaml` — add `max_tokens_budget`, `max_wall_clock_seconds`

### 3.5 Haiku Compaction at Merge Points (MC2)

**What:** After parallel merge points in Trading Graph, run Haiku-tier compaction that summarizes branch outputs. 40-60% context size reduction for downstream agents.
**Files:**
- `src/quantstack/graphs/trading/graph.py` — add compaction nodes after `merge_parallel` and `merge_pre_execution`
- `src/quantstack/graphs/trading/nodes.py` — new `compact_context` node

### 3.6 Knowledge Base Fix (search_knowledge_base Bypasses RAG)

**Finding:** MC0 — `tools/langchain/learning_tools.py:25-31` queries by recency, ignores the query parameter.
**What:** Replace SQL recency query with call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`. Add HNSW index on embeddings table.
**Files:**
- `src/quantstack/tools/langchain/learning_tools.py` — one-line fix
- DB migration: add `CREATE INDEX idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops)`

### 3.7 Greeks Integration in Risk Gate

**Finding:** QS-E3 — Options risk checks only look at DTE + premium, not delta/gamma/vega/theta.
**What:** Wire `core/risk/options_risk.py` (already exists, 444 lines) into `risk_gate.py` options path. Add portfolio-level Greeks limits.
**Files:**
- `src/quantstack/execution/risk_gate.py` — integrate Greeks manager
- Add limits: max portfolio delta exposure, gamma limit, vega limit, theta budget

### 3.8 Inter-Graph Urgency Channel

**Finding:** GC2 — Supervisor → Trading communication has 5-10 min poll latency.
**What:** Add Redis pub/sub or file sentinel for sub-second supervisor → trading interrupts. Trading graph checks sentinel before each trade execution.
**Files:**
- `src/quantstack/coordination/event_bus.py` — add urgent channel
- `src/quantstack/graphs/trading/nodes.py` — check urgent channel pre-execution

### 3.9 Signal IC Computation (QS-S1)

**Finding:** QS-S1 — No signal has ever been validated against forward returns. `signal_ic` table exists but empty.
**What:** Implement `ICTracker` module. Daily IC computation for all 22 collectors. Store in `signal_ic` table. Gate: `if rolling_63d_IC < 0.02: disable collector from synthesis`.
**Files:**
- New: `src/quantstack/signal_engine/ic_tracker.py`
- `src/quantstack/signal_engine/engine.py` — integrate IC gating
- `scripts/scheduler.py` — add daily IC computation job

### 3.10 Intraday Circuit Breaker (QS-E5)

**What:** Check unrealized P&L every tick cycle. Thresholds: -1.5% unrealized → halt new entries, -2.5% → begin systematic exit, -5% → emergency liquidation.
**Files:**
- `src/quantstack/execution/execution_monitor.py` — add unrealized P&L checks
- `src/quantstack/execution/risk_gate.py` — add intraday drawdown state

---

## Phase 4: Scale & Self-Improvement (P3 — Week 9+)

*Goal: System compounds intelligence as it compounds capital.*

### 4.1 Alpha Knowledge Graph (AR-3)
Add `alpha_knowledge_graph` table (node/edge schema in PostgreSQL JSON). Populate from strategy registration, trade outcomes, and community intel. Enable gap discovery and contradiction detection.

### 4.2 Meta-Improvement Layer (AR-2)
Four meta-agents (weekly): `meta_prompt_optimizer`, `meta_threshold_tuner`, `meta_tool_selector`, `meta_architecture_critic`. Each reviews agent performance data and proposes improvements.

### 4.3 Hierarchical Governance / OrgAgent (AR-4)
Split `fund_manager` into CIO (governance, daily, heavy) + risk_officer (compliance, per-trade, deterministic). Split `trade_debater` into strategy-specific execution agents (momentum, mean_rev, options, earnings). Expected 74% token reduction.

### 4.4 Parallel Research Streams (AR-5)
Weekend mode: 4 parallel research streams (factor mining, regime research, cross-asset signals, portfolio construction). 56 weekend hours × 4 streams = ~2,688 experiments/weekend vs. ~12 today.

### 4.5 Adversarial Consensus Protocol (AR-6)
For high-stakes trades (>$5k position): 3 parallel agents (bull advocate, bear advocate, neutral arbiter). Voting: 3/3=full size, 2/3=50%, else reject.

### 4.6 Feature Factory (AR-10)
Autonomous feature enumeration → IC screening → monitoring. Auto-replace decaying features. Target: 500+ candidate features → 50-100 curated.

### 4.7 TCA Feedback Loop (QS-E6)
Daily recalibration: compare realized vs. forecast slippage. Update Almgren-Chriss parameters per symbol/time-of-day bucket. Conservative 2x multiplier until 50 trades accumulated.

---

## Architecture Reference

### Graph Pipeline Flows

**Trading Graph (16 nodes, 2 parallel branches):**
```
START → data_refresh → safety_check → market_intel → daily_plan
  ↓ (parallel)
  ├── position_review → execute_exits
  └── entry_scan → (earnings_analysis?)
  ↓ (merge)
  risk_sizing → portfolio_construction
  ↓ (parallel)
  ├── portfolio_review
  └── analyze_options
  ↓ (merge)
  execute_entries → reflect → END
```

**Research Graph (8 nodes, self-critique loop):**
```
START → context_load → domain_selection → hypothesis_generation
  ↓ (loop if confidence < 0.7, max 3)
  hypothesis_critique → signal_validation → backtest_validation
  → ml_experiment → strategy_registration → knowledge_update → END
```

**Supervisor Graph (7 nodes, linear):**
```
START → health_check → diagnose_issues → execute_recovery
  → strategy_pipeline → strategy_lifecycle → scheduled_tasks
  → eod_data_sync → END
```

### Agent → Tier → Model Mapping (with bedrock_groq hybrid)

| Agent | Tier | Model |
|-------|------|-------|
| trade_debater, fund_manager, options_analyst | heavy | Bedrock Sonnet 4.6 |
| quant_researcher, ml_scientist, strategy_rd | heavy | Bedrock Sonnet 4.6 |
| domain_researcher, execution_researcher | heavy | Bedrock Sonnet 4.6 |
| daily_planner, position_monitor, exit_evaluator | medium | Groq Llama 3.3 70B |
| earnings_analyst, market_intel, trade_reflector | medium | Groq Llama 3.3 70B |
| executor, hypothesis_critic, community_intel | medium | Groq Llama 3.3 70B |
| self_healer, portfolio_risk_monitor, strategy_promoter | medium | Groq Llama 3.3 70B |
| health_monitor | light | Groq Llama 3.1 8B |

### Key File Locations

| Subsystem | Path |
|-----------|------|
| Graph builders | `src/quantstack/graphs/{trading,research,supervisor}/graph.py` |
| Agent configs | `src/quantstack/graphs/{trading,research,supervisor}/config/agents.yaml` |
| Agent executor | `src/quantstack/graphs/agent_executor.py` |
| Graph nodes | `src/quantstack/graphs/{trading,research,supervisor}/nodes.py` |
| Risk gate | `src/quantstack/execution/risk_gate.py` |
| Kill switch | `src/quantstack/execution/kill_switch.py` |
| Trade service | `src/quantstack/execution/trade_service.py` |
| Alpaca broker | `src/quantstack/execution/alpaca_broker.py` |
| Paper broker | `src/quantstack/execution/paper_broker.py` |
| Order lifecycle | `src/quantstack/execution/order_lifecycle.py` |
| Execution monitor | `src/quantstack/execution/execution_monitor.py` |
| LLM config | `src/quantstack/llm/config.py` |
| LLM provider | `src/quantstack/llm/provider.py` |
| Signal engine | `src/quantstack/signal_engine/engine.py` |
| Signal cache | `src/quantstack/signal_engine/cache.py` |
| Collectors | `src/quantstack/signal_engine/collectors/` |
| Tool registry | `src/quantstack/tools/registry.py` |
| Tool binding | `src/quantstack/tools/tool_binding.py` |
| Learning tools | `src/quantstack/tools/langchain/learning_tools.py` |
| EventBus | `src/quantstack/coordination/event_bus.py` |
| Daily digest | `src/quantstack/coordination/daily_digest.py` |
| Data pipeline | `src/quantstack/data/acquisition_pipeline.py` |
| Scheduled refresh | `src/quantstack/data/scheduled_refresh.py` |
| DB module | `src/quantstack/db.py` |
| Scheduler | `scripts/scheduler.py` |
| AutoResearchClaw | `scripts/autoresclaw_runner.py` |
| Docker | `docker-compose.yml`, `Dockerfile` |
| CTO audit | `CTO_AUDIT_FINDINGS.md` |

### Hard Rules (NEVER violate)

1. **Risk gate is LAW.** Never weaken or bypass `risk_gate.py`. Strengthening is encouraged.
2. **Kill switch halts everything.** Check before any operation.
3. **Paper mode is default.** Live requires `USE_REAL_TRADING=true`.
4. **Audit trail is mandatory.** Every decision logged with reasoning.
5. **DB writes use `db_conn()` context managers.**
6. **Protected files** (AutoResearchClaw cannot modify): `risk_gate.py`, `kill_switch.py`, `db.py`

### Cost Estimates (bedrock_groq hybrid)

| Tier | $/MTok (blended) | Agents | Calls/day | Est. daily cost |
|------|-----------------|--------|-----------|-----------------|
| Heavy (Sonnet) | $9.00 | 7 | ~50 | ~$9.00 |
| Medium (Groq 70B) | $0.64 | 12 | ~200 | ~$2.56 |
| Light (Groq 8B) | $0.10 | 1 | ~100 | ~$0.20 |
| Bulk (Groq 8B) | $0.10 | misc | ~50 | ~$0.10 |
| **Total** | | **21** | **~400** | **~$11.86/day** |

With prompt caching enabled (Phase 1.4): ~$5-7/day.
Without Groq hybrid (all Bedrock): ~$40-60/day.

---

## Success Criteria

### Phase 1 Complete (Week 2)
- [ ] No position can exist without a stop-loss
- [ ] Agent output parse failures retry once, then log warning (never silent `{}`)
- [ ] Tool ordering is deterministic (prompt cache hit rate > 80%)
- [ ] Daily pg_dump runs and uploads to S3
- [ ] Scheduler runs inside Docker with health check
- [ ] All 3 graphs use PostgresSaver
- [ ] Trading Graph polls EventBus for supervisor events
- [ ] CI pipeline runs tests on every push

### Phase 2 Complete (Week 4)
- [ ] LLM calls survive provider outages (failover within 30s)
- [ ] Signal cache invalidates on data refresh
- [ ] Pre-trade correlation check rejects >0.7 correlated entries
- [ ] Market hours hard-gated (no accidental off-hours orders)
- [ ] Daily notional deployment capped at 30% equity
- [ ] Kill switch auto-recovers from transient failures
- [ ] Error rate alerting active in Grafana
- [ ] Agents only see working tools (92 stubs removed from bindings)

### Phase 3 Complete (Week 8)
- [ ] Three operating modes functional (market/extended/overnight)
- [ ] Overnight autoresearch produces 50+ experiments/night
- [ ] Loss analysis feeds research queue with prioritized tasks
- [ ] Per-agent budget tracking active
- [ ] Context compaction at merge points (40%+ reduction)
- [ ] Knowledge base search uses semantic similarity (not recency)
- [ ] Greeks limits enforced in risk gate
- [ ] Signal IC computed daily for all collectors, gates at IC < 0.02
- [ ] Intraday circuit breaker active (-1.5% halts entries)

### Phase 4 Complete (Week 10+)
- [ ] Knowledge graph populated from strategy + trade data
- [ ] Meta-agents run weekly, propose measurable improvements
- [ ] Token spend reduced 50%+ from OrgAgent hierarchy
- [ ] Weekend research produces 1000+ experiments
- [ ] High-stakes trades require consensus (3-agent voting)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Groq Llama tool calling quality worse than Haiku | Monitor first week on paper. Revert executor/position_monitor to Haiku if error rate >5% |
| PostgresSaver migration breaks graph state | Run parallel (MemorySaver + PostgresSaver) for 1 week before cutting over |
| Prompt caching doesn't hit due to dynamic content | Measure cache hit rate in Langfuse. Separate static (system prompt) from dynamic (user state) |
| Overnight autoresearch produces garbage | Gate: only register drafts with IC > 0.02. Morning pipeline validates with full backtest |
| Kill switch auto-recovery triggers too eagerly | Start with 30-min cooldown. Monitor false positive rate. Increase cooldown if needed |
| 321 uncommitted files create merge conflicts | Commit in logical batches before starting Phase 1. Tag current state as baseline |

---

## Dependencies

| Dependency | Status | Required By |
|-----------|--------|-------------|
| `langchain-groq>=0.2.0` | Just added to pyproject.toml | Phase 1 (already done) |
| `langgraph-checkpoint-postgres` | Not installed | Phase 1.8 |
| `GROQ_API_KEY` env var | Needs to be set | Phase 1 |
| Grafana/Loki/Fluent-bit in docker-compose | Services defined but not wired | Phase 2.7 |
| Redis (optional, for urgency channel) | Not deployed | Phase 3.8 |
