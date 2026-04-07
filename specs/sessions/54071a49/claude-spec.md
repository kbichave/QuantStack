# Complete Specification: QuantStack 24/7 Autonomous Trading Readiness

**Synthesized from:** `specs/24x7_readiness.md` (164-finding CTO audit), codebase research (10 subsystems), web research (4 implementation patterns), stakeholder interview (10 decisions)

---

## 1. Mission

Take QuantStack from attended market-hours-only paper trading to unattended 24/7 autonomous operation. Current grade: **C-** (D+ execution, D- signal rigor, F feedback loops). Target: **B+** within 8 weeks, sufficient for supervised live trading with conservative capital ($5-10K, max 5 concurrent positions).

**Go-live gate:** Phase 1 + 2 + 3 must all be complete before any real capital is deployed. This is a non-negotiable stakeholder requirement.

---

## 2. Stakeholder Decisions (from interview)

These decisions are FINAL and must be reflected throughout the implementation plan:

| Decision | Choice | Constraint |
|----------|--------|------------|
| Starting capital | $5-10K conservative | Max 5 concurrent positions, tight stops |
| Unknown state behavior | Defensive exit + alert | Close all positions at market, kill switch, email alert |
| Research priority | 70% new / 30% refine | Balanced exploration in overnight compute |
| Alerting | Email only (Gmail SMTP) | No Discord dependency. All alert code uses email. |
| PostgresSaver migration | Direct cutover | No parallel validation period needed |
| Go-live gate | Phase 1+2+3 complete | ~8 weeks before real money |
| Circuit breaker | Layered (daily P&L + portfolio HWM) | Both intraday and multi-day protection |
| Stubbed tools | Prioritize top 10-15 | Implement best, hide rest from bindings |
| Groq structured output | Benchmark first | Run compatibility test before committing |

---

## 3. Current System State

### What Works (preserve these)

| Component | Location |
|-----------|----------|
| Signal Engine (16 collectors, fault-tolerant) | `src/quantstack/signal_engine/` |
| Risk Gate (12+ sequential checks, daily loss, position caps) | `src/quantstack/execution/risk_gate.py` |
| Kill Switch (two-layer: DB + sentinel file, AutoTriggerMonitor) | `src/quantstack/execution/kill_switch.py` |
| 21 specialized agents across 3 LangGraph graphs | `src/quantstack/graphs/*/config/agents.yaml` |
| Execution Monitor (deterministic exit rules) | `src/quantstack/execution/execution_monitor.py` |
| SPAN margin module (537 lines, 16-scenario stress) | `src/quantstack/core/risk/span_margin.py` |
| Docker Compose (9+ services, health checks) | `docker-compose.yml` |
| Data pipeline (14-phase, idempotent, rate-limited) | `src/quantstack/data/acquisition_pipeline.py` |
| LLM tier routing (heavy/medium/light/bulk) | `src/quantstack/llm/provider.py` |
| Groq hybrid provider (bedrock_groq) | `src/quantstack/llm/config.py` |

### What's Broken (164 findings, distilled)

**321 uncommitted files** in working tree. No CI/CD. No automated tests on deploy.

Critical gaps by subsystem:
- **Execution**: Stop-loss optional (C1), bracket orders silently degrade (C2), no intraday circuit breaker (QS-E5)
- **LLM/Agent**: Silent JSON parse failures (LC2), prompt injection via f-strings (LC1), zero prompt caching (MC0c), non-deterministic tool ordering (MC1)
- **Operations**: No DB backups (OC1), scheduler in tmux not Docker (OC2), CI disabled (OC3)
- **Signals**: No IC computation (QS-S1), stale cache (DC1), data staleness unchecked (DC3)
- **Feedback loops**: Trading Graph doesn't poll EventBus (AC1), kill switch doesn't publish events (AC2), losses don't drive research (AC1-2)
- **Tools**: 92/122 tools stubbed (TC1), knowledge base bypasses RAG (MC0)

---

## 4. Architecture Reference

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

### Agent → Tier → Model Mapping (bedrock_groq hybrid)

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

### Key Code Locations

| Subsystem | Path | Key Modification Points |
|-----------|------|------------------------|
| Agent executor | `agent_executor.py` | `parse_json_response()` (L474-521), `build_system_message()` (L152-178) |
| Risk gate | `risk_gate.py` | `check()` method, `monitor()` method, options path |
| Graph builders | `graphs/{trading,research,supervisor}/graph.py` | `checkpointer=` param, conditional edges |
| Tool binding | `tools/tool_binding.py` | Tool list sorting, registry split |
| EventBus | `coordination/event_bus.py` | `poll()` consumer, event publishing |
| Kill switch | `execution/kill_switch.py` | `trigger()` event publish, auto-recovery |
| LLM provider | `llm/provider.py` | `_instantiate_chat_model()`, caching headers |
| Signal cache | `signal_engine/cache.py` | `invalidate()` hook from scheduled_refresh |

---

## 5. Implementation Phases

### Phase 1: Safety Hardening (P0 — Week 1-2)

*Goal: System can paper trade without losing money to bugs.*

#### 1.1 Mandatory Stop-Loss Enforcement
- Reject `OrderRequest` where `stop_price is None` at OMS level
- Implement `execute_bracket()` using Alpaca's native bracket API
- On bracket failure, fall back to separate SL/TP contingent orders
- **Invariant:** No position can exist without a stop-loss order
- **Files:** `trade_service.py`, `alpaca_broker.py:106-148`, `paper_broker.py`

#### 1.2 Output Schema Validation with Retry
- Add Pydantic output models for 5 critical agents: daily_plan, entry_scan, position_review, fund_manager, risk_sizing
- On parse failure: retry once with schema hint → log warning → return schema-compliant default (NOT `{}`)
- **Note:** Groq/Llama structured output compatibility must be benchmarked (stakeholder decision). Use `json_mode` + Pydantic validation as fallback if `json_schema` mode fails on Groq.
- **Files:** `agent_executor.py:474-521`, new Pydantic models in `tools/models.py`

#### 1.3 Deterministic Tool Ordering
- Sort tool definitions alphabetically before injection
- **Prerequisite for prompt caching** — tool order changes invalidate entire cache
- **Files:** `tools/tool_binding.py` — 1-line `sorted()` call

#### 1.4 Prompt Caching Enablement
- Add `cache_control: {"type": "ephemeral"}` to system messages in `build_system_message()`
- For Bedrock: pass `anthropic_beta: ["prompt-caching-2024-07-31"]` header
- Separate static content (persona, tools) from dynamic content (state, prices) — static goes before cache breakpoint
- Only applies to heavy tier (Anthropic/Bedrock). Groq models don't support prompt caching.
- **Impact:** 90% cost reduction on cached tokens. ~$12/day → ~$5-7/day
- **Files:** `agent_executor.py:152-178`, `llm/provider.py`

#### 1.5 Automated Database Backups
- Daily `pg_dump` → local storage (S3 optional later)
- WAL archiving for point-in-time recovery
- **Files:** `docker-compose.yml` (backup sidecar), new `scripts/pg_backup.sh`

#### 1.6 Prompt Injection Defense
- Replace f-string interpolation with XML-tagged structured sections
- Sanitize all DB/API-sourced data before prompt injection
- **Files:** `trading/nodes.py:80-92`, `research/nodes.py:219-224`, new `llm/sanitize.py`

#### 1.7 Containerize Scheduler
- Add `scheduler` service to docker-compose.yml with health check + restart policy
- Scheduler service definition already exists (port 8422) — needs health check + restart
- **Files:** `docker-compose.yml`, `scripts/scheduler.py`

#### 1.8 Durable Checkpoints (MemorySaver → PostgresSaver)
- Direct cutover (stakeholder decision — no parallel validation)
- Use `AsyncPostgresSaver` with `AsyncConnectionPool` (2-5 connections per graph)
- `setup()` is idempotent — safe to call on every startup
- Each graph gets its own pool (don't share)
- **Dependency:** `langgraph-checkpoint-postgres` (already in pyproject.toml optional deps)
- **Files:** All 3 `graph.py` files

#### 1.9 EventBus Wiring
- Trading Graph `safety_check` node: add `bus.poll()` for `IC_DECAY`, `RISK_EMERGENCY`, `KILL_SWITCH_TRIGGERED`
- Kill switch `trigger()`: publish `KILL_SWITCH_TRIGGERED` to EventBus
- **Files:** `trading/nodes.py`, `kill_switch.py`, `event_bus.py`

#### 1.10 Enable CI/CD
- Re-enable `.github/workflows/ci.yml` and `release.yml`
- Run test suite + type check + build image on every push to main
- **Files:** `.github/workflows/ci.yml`, `.github/workflows/release.yml`

### Phase 2: Operational Resilience (P1 — Week 3-4)

*Goal: System runs unattended for days without human intervention.*

#### 2.1 LLM Runtime Failover
- On 429/500/timeout: retry same provider 2x → switch to next in FALLBACK_ORDER → cooldown 5 min
- Use LangChain `with_fallbacks()` (already in ecosystem) wrapped in `CircuitBreaker` class
- Error classification: retryable (429, 500, timeout) vs non-retryable (400, 401, 403)
- Health state in-memory (transient, no DB needed)
- **Files:** `llm/provider.py`, new `llm/circuit_breaker.py`

#### 2.2 Signal Cache Auto-Invalidation
- Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after each intraday refresh
- **Files:** `data/scheduled_refresh.py`, `signal_engine/cache.py`

#### 2.3 Pre-Trade Correlation Check
- Add pairwise correlation check in `risk_gate.check()` against existing portfolio
- If `corr > 0.7`: apply 50% concentration haircut to sizing
- Risk gate already has `_check_pretrade_correlation()` — currently post-hoc only, needs pre-trade wiring
- **Files:** `execution/risk_gate.py`

#### 2.4 Market Hours Hard Gating
- Hard-reject orders outside configurable trading windows unless `extended_hours=True`
- Uses existing `TradingWindow` enum from `trading_window.py`
- **Files:** `execution/risk_gate.py`

#### 2.5 Portfolio Heat Budget
- `max_daily_notional_deployed` limit (default 30% of equity/day for new positions)
- With $5-10K capital: ~$1.5-3K max new deployment per day
- **Files:** `execution/risk_gate.py`

#### 2.6 Kill Switch Auto-Recovery
- Tiered recovery: (1) Email alert immediately, (2) investigate root cause, (3) if transient → auto-reset after 30-min cooldown, (4) escalate to email after 4 hours
- **Note:** Alerting is email-only (Gmail SMTP) per stakeholder decision
- **Files:** `execution/kill_switch.py`, new email alerting module

#### 2.7 Email Alerting System
- Gmail SMTP with app password (stakeholder decision)
- Alert levels: INFO (daily digest), WARNING (threshold approach), CRITICAL (kill switch, drawdown)
- **New files:** `src/quantstack/alerting/email_sender.py`, `src/quantstack/alerting/alert_manager.py`

#### 2.8 Log Aggregation + Alerting
- Wire up existing Fluent-bit/Loki/Grafana services in docker-compose.yml
- Add alerts for ERROR rate spikes → email notification
- **Files:** `docker-compose.yml`, new `config/fluent-bit.conf`, `config/grafana/alerts/`

#### 2.9 Stubbed Tool Registry Cleanup
- Identify top 10-15 most impactful stubs for Phase 2-3 implementation
- Split `TOOL_REGISTRY` into `ACTIVE_TOOLS` and `PLANNED_TOOLS`
- Remove remaining stubs from agent bindings
- Add `tool_health` table for monitoring
- **Files:** `tools/registry.py`, all `agents.yaml` files

#### 2.10 Per-Agent Temperature Configuration
- Add `temperature` field to agent configs
- Hypothesis generation=0.7, debate=0.3-0.5, validation/execution=0.0
- **Files:** `agents.yaml` files, `agent_executor.py`

#### 2.11 Data Staleness Rejection
- Each collector checks `data_metadata.last_timestamp`
- Return `{}` if data staler than configurable threshold
- **Files:** `signal_engine/engine.py`, individual collectors

### Phase 3: Autonomy (P2 — Week 5-8)

*Goal: System operates in multiple modes and self-improves.*

#### 3.1 Multi-Mode Operation
- **Market Hours** (9:30-16:00 ET Mon-Fri): Full trading + research
- **Extended Hours** (16:00-20:00, 04:00-09:30 ET): Position monitoring only, EOD sync
- **Overnight/Weekend** (20:00-04:00 ET, weekends): Research compute, ML training, no trading
- **Files:** All 3 `graph.py` files

#### 3.2 Overnight Autoresearch Loop
- 5-min budget per experiment, ~96 experiments/night
- 70% new hypotheses, 30% refining existing winners (stakeholder decision)
- Haiku for hypothesis generation, single metric: OOS IC on purged holdout
- Winners registered as draft → morning pipeline validates
- **Files:** New `research/autoresearch_node.py`, `research/graph.py`

#### 3.3 Error-Driven Iteration
- Daily loss analysis pipeline at 16:30 ET
- Classify losses: regime_shift, signal_failure, thesis_wrong, sizing_error, entry_timing, theta_burn
- Aggregate 30-day failure frequencies → prioritized research tasks
- **Files:** New `supervisor/loss_analyzer.py`, `supervisor/nodes.py`

#### 3.4 Experiment Budget Protocol
- Per-cycle token/wall-clock/cost budgets per agent
- Graceful exit at next node boundary when budget exhausted
- **Files:** New `graphs/budget_tracker.py`, `agent_executor.py`, `agents.yaml`

#### 3.5 Haiku Compaction at Merge Points
- After parallel merge points in Trading Graph, run compaction
- 40-60% context size reduction for downstream agents
- **Files:** `trading/graph.py`, `trading/nodes.py`

#### 3.6 Knowledge Base Fix
- Replace SQL recency query with `rag.query.search_knowledge_base(query=query, n_results=top_k)`
- Add HNSW index on embeddings table
- **Files:** `tools/langchain/learning_tools.py`

#### 3.7 Greeks Integration in Risk Gate
- Wire `core/risk/options_risk.py` (444 lines, already exists) into `risk_gate.py`
- Add portfolio-level delta, gamma, vega, theta limits
- **Files:** `execution/risk_gate.py`

#### 3.8 Layered Circuit Breaker
- **Daily P&L layer**: -1.5% halt entries, -2.5% begin exits, -5% emergency liquidation (resets daily)
- **Portfolio HWM layer**: -3% from high-water mark halts all trading, -5% triggers defensive exit + kill switch
- Defensive exit behavior (stakeholder decision): close all at market, kill switch, email alert
- **Files:** `execution/execution_monitor.py`, `execution/risk_gate.py`

#### 3.9 Inter-Graph Urgency Channel
- Add urgent channel to EventBus (file sentinel or Redis pub/sub)
- Trading graph checks sentinel before each trade execution
- **Files:** `coordination/event_bus.py`, `trading/nodes.py`

#### 3.10 Signal IC Computation
- Daily IC computation for all 22 collectors
- Gate: `if rolling_63d_IC < 0.02: disable collector`
- **Files:** New `signal_engine/ic_tracker.py`, `signal_engine/engine.py`, `scripts/scheduler.py`

### Phase 4: Scale & Self-Improvement (P3 — Week 9+)

*Goal: System compounds intelligence as it compounds capital. Deferred until Phases 1-3 validated.*

- 4.1 Alpha Knowledge Graph
- 4.2 Meta-Improvement Layer (4 meta-agents)
- 4.3 Hierarchical Governance / OrgAgent (74% token reduction)
- 4.4 Parallel Research Streams (4 streams, ~2688 experiments/weekend)
- 4.5 Adversarial Consensus Protocol (3-agent voting for >$5K positions)
- 4.6 Feature Factory (500+ candidates → 50-100 curated)
- 4.7 TCA Feedback Loop

---

## 6. Dependency Graph

```
MC1 (tool ordering) ──→ MC0c (prompt caching)
                    ──→ TC1 (tool registry split, needs deterministic ordering first)

C1 (stop-loss) ──→ C2 (bracket orders)

OC1 (DB backups) ──→ GC1 (PostgresSaver, backup must exist before migration)

GC1 (PostgresSaver) ──→ Mode-aware runners (Phase 3, needs durable state)

AC1+AC2 (EventBus wiring) ──→ GC2 (urgency channel, Phase 3)

OC3 (CI/CD) ──→ All subsequent phases (safety net for iteration)

Email alerting (2.7) ──→ Kill switch recovery (2.6)
                     ──→ Log alerting (2.8)
                     ──→ Circuit breaker alerts (3.8)
```

---

## 7. Cost Projections

| Phase | Daily LLM Cost | Notes |
|-------|---------------|-------|
| Current (all Bedrock, no caching) | ~$40-60/day | Sonnet for everything |
| After Groq hybrid (Phase 1) | ~$12/day | Sonnet heavy, Groq medium/light |
| After prompt caching (Phase 1.4) | ~$5-7/day | 90% cache hit on heavy tier |
| After compaction (Phase 3.5) | ~$3-5/day | 40-60% context reduction |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Groq Llama structured output quality | Benchmark during Phase 1. Revert to Haiku if error rate >5% |
| PostgresSaver direct cutover breaks state | Paper trading only — restart graph cycle if corrupted |
| Prompt caching doesn't hit (dynamic content) | Measure cache hit rate in Langfuse. Separate static from dynamic. |
| Overnight autoresearch produces garbage | Gate: only register drafts with IC > 0.02. Morning validates. |
| Kill switch auto-recovery too eager | Start with 30-min cooldown. Monitor false positives. |
| 321 uncommitted files create conflicts | Commit in logical batches before Phase 1. Tag baseline. |
| Gmail SMTP rate limits for alerts | Gmail allows 500 emails/day — more than sufficient. Monitor. |
| $5-10K too small for meaningful options strategies | Start with equity-only. Enable options after proving equity edge. |

---

## 9. Success Criteria

### Phase 1 Complete (Week 2)
- [ ] No position exists without a stop-loss
- [ ] Agent output parse failures retry + log warning (never silent `{}`)
- [ ] Tool ordering deterministic (prompt cache hit rate > 80%)
- [ ] Daily pg_dump running
- [ ] Scheduler in Docker with health check
- [ ] All 3 graphs use PostgresSaver
- [ ] Trading Graph polls EventBus for supervisor events
- [ ] CI pipeline runs on every push

### Phase 2 Complete (Week 4)
- [ ] LLM calls survive provider outages (failover < 30s)
- [ ] Signal cache invalidates on data refresh
- [ ] Pre-trade correlation check active
- [ ] Market hours hard-gated
- [ ] Daily notional capped at 30% equity
- [ ] Kill switch auto-recovers from transients
- [ ] Email alerting active for critical events
- [ ] Error rate alerting in Grafana
- [ ] Agents only see working tools
- [ ] Groq structured output benchmark completed

### Phase 3 Complete (Week 8) — GO-LIVE GATE
- [ ] Three operating modes functional
- [ ] Overnight research: 50+ experiments/night
- [ ] Loss analysis feeds research queue
- [ ] Per-agent budget tracking active
- [ ] Context compaction at merge points (40%+ reduction)
- [ ] Knowledge base uses semantic similarity
- [ ] Greeks limits enforced in risk gate
- [ ] Layered circuit breaker active (daily + portfolio)
- [ ] Signal IC computed daily, gates at IC < 0.02
- [ ] System runs unattended for 7 consecutive days on paper without intervention

### Go-Live Criteria (after Phase 3)
- [ ] All Phase 1-3 success criteria met
- [ ] 7-day unattended paper trading without kill switch trigger
- [ ] Positive paper P&L over 30-day validation period
- [ ] All critical alerts tested end-to-end
- [ ] Backup/restore procedure verified
