# Research Findings: Phase 10 Advanced Research

## Part 1: Codebase Analysis

### Executive Summary

QuantStack is a production-grade autonomous trading system (~185k lines) built on LangGraph agents with PostgreSQL state management. The system is architecturally solid for Phase 10, with flexible tool binding, separation of concerns, hard safety guarantees, and event-driven inter-loop coordination.

### 1. Tool Registry & Tool Layer

**Architecture (`src/quantstack/tools/registry.py`):**
- Centralized `TOOL_REGISTRY` (dict[str, BaseTool]) with 100+ LLM-facing tools
- Per-agent subsets defined in `agents.yaml`, resolved via `bind_tools_to_llm()`
- Deferred loading via BM25 keyword search (`search_deferred_tools`) — 90% cache hit rate
- Always-loaded tools per agent for prompt cache stability (signal_brief, fetch_market_data, etc.)

**Tool Categories:**
- Signal & Analysis, Data, Options, Risk, Execution, Backtesting, ML, Learning, Strategy Mgmt, Advanced (FinRL, QC, EWF)

**Shared Layer (`tools/_shared.py`):**
- Core implementations callable by both langchain tools and autonomous modules
- `register_strategy_impl()` auto-detects regime affinity, validates entry rules against `_KNOWN_INDICATORS`
- `run_backtest_impl()` full pipeline: fetch data -> generate signals -> run engine -> persist summary

**Functions Layer (`tools/functions/`):**
- Direct callable functions (not LLM-exposed) for data, execution, risk, system operations
- Key invariant: imports only from _shared and core — NO circular dependencies

### 2. AutoResearchClaw

**Purpose:** Autonomous research task execution (bug fixes, ML arch search, strategy hypotheses, RL env design) via Claude Code in sandboxed Docker.

**Scheduling:** Sunday 20:00 ET, top-3 pending tasks by priority.

**Task Types:**
1. `bug_fix` — Locate code -> reproduce -> fix -> validate (py_compile + import). Hard constraints: NEVER touch risk_gate.py, kill_switch.py, db.py. Post-fix: confidence check, syntax check, protected file check, commit, restart affected tmux loops.
2. `ml_arch_search` — Propose features/architecture -> train -> evaluate OOS Sharpe/IC/max DD
3. `rl_env_design` — Design gymnasium environment -> train 100k steps -> shadow test
4. `strategy_hypothesis` — Literature search -> propose rules -> backtest walk-forward

**Safety:** Docker required, DB connectivity check, timeout handling (task stays 'running' for retry), protected file detection, confidence-based revert.

### 3. Research Graph

**Control flow:**
```
START -> context_load -> domain_selection -> hypothesis_generation -> hypothesis_critique
  -> [confidence < 0.7: retry] -> signal_validation -> backtest_validation
  -> ml_experiment (optional) -> strategy_registration -> knowledge_update -> END
```

**Fan-out enabled:** Parallel validation per symbol + hypothesis via `fan_out_hypotheses`.

**Key Nodes:**
- `context_load`: Polls event_bus for IDEAS_DISCOVERED, research_queue for pending tasks, fetches regime
- `hypothesis_generation`: Uses quant_researcher agent (heavy LLM, 30+ tools), retry loop (max 3)
- `hypothesis_critique`: Quality gate (medium LLM, 2 tools), scores 0-1 confidence
- `backtest_validation`: Walk-forward, gates: Sharpe > 0.5, trades > 20, profit_factor > 1.2
- `strategy_registration`: Inserts as status='draft' (never auto-promotes)

**State Schema (ResearchState):** cycle_number, regime, context_summary, selected_domain/symbols, hypothesis, confidence, validation_result, backtest_id, ml_experiment_id, registered_strategy_id, hypothesis_attempts, queued_task_ids, errors (append-only), decisions (append-only).

**Agent Configs:**
- quant_researcher: heavy tier (Sonnet 4.6), 30+ tools, 300s timeout
- ml_scientist: heavy tier, 25+ tools, 300s timeout
- hypothesis_critic: medium tier (Haiku 4.5), 2 tools, 120s timeout
- domain_researcher: heavy tier, domain-specific tool sets

### 4. Execution & Risk Layer

**Risk Gate (`risk_gate.py`, ~2000 lines):**
- Position size: max 10% equity, $20K notional cap
- Liquidity: min 500K ADV, order <= 1% ADV
- Portfolio: 150% gross, 100% net exposure
- Daily loss limit: -2% halts all new entries
- Correlation/diversification check (> 0.80 warns)
- Regime-conditional sizing (relaxes in low-vol, tightens in high-vol)

**Kill Switch (`kill_switch.py`, ~400 lines):**
- Persisted to sentinel file (survives restarts)
- Triggers: manual, daily loss breach, consecutive fill failures, extreme volatility
- Actions: cancel orders, close all positions, log CRITICAL

**Trade Service:** prepare_and_submit_order -> portfolio state -> position size -> risk_gate.check() -> submit -> record outcome

### 5. Learning & Feedback Systems

**Drift Detector:** PSI-based feature drift detection. Thresholds: <0.10 NONE, 0.10-0.25 WARNING, >=0.25 CRITICAL. Pure numpy (<1ms). 5-10 day lead over IC decay.

**Outcome Tracker:** Non-parametric RLHF via regime affinity weights. Win -> affinity[regime] += step * tanh(pnl/scale). Loss -> affinity[regime] -= step. Clipped [0.1, 1.0]. Min 5 outcomes before update. DB writes best-effort (never blocks fills).

**IC Attribution:** Per-collector signal quality via rolling Spearman rank correlation. Reports best/worst collectors, degraded list, suggested weights for SignalBrief synthesis.

**Skill Tracker:** Agent prediction accuracy + signal quality (IC, ICIR). ICIR > 0.5 good, > 1.0 institutional-grade.

### 6. LLM Routing

**Tiers:** light (Haiku 4.5), heavy (Sonnet 4.6), bulk (Haiku 4.5), embedding (text-embedding-3-small)
**Providers:** bedrock (primary), anthropic, openai, vertex_ai, gemini, groq, ollama
**Resolution:** Per-tier env override -> Primary provider -> Fallback chain
**Optional LiteLLM Router** for load balancing + retry logic
**Thinking mode** available for research agents (cost/latency tradeoff)

### 7. Database Schema

**PostgreSQL as single source of truth.** psycopg3 ConnectionPool (4-20 connections).

Key tables:
- Operational: positions, strategies, signals, fills, audit_trail
- Research: research_queue, ml_experiments, bugs, drift_baselines, ic_observations, regime_affinity_log
- Event Bus: event_log (append-only), loop_cursors (per-consumer high-water marks)

**PgDataStore:** Bulk upserts via psycopg3 pipeline mode, ON CONFLICT DO UPDATE for idempotency.

### 8. Testing Setup

```
tests/
  unit/         — 240+ unit tests, conftest.py with synthetic OHLCV generators
  integration/  — Alembic migrations, event bus coordination
  regression/   — End-to-end tests
  graphs/       — Graph composition tests
  benchmarks/   — Performance tests
```

Fixtures: make_ohlcv_df, make_monotonic_uptrend/downtrend, make_flat_market, make_v_shape, make_w_shape, make_impulse_wave, make_swing_leg/point, add_atr_column.

### 9. Event Bus

**Poll-based, append-only, per-consumer cursors, 7-day TTL.**

Event types: STRATEGY_PROMOTED/RETIRED/DEMOTED, MODEL_TRAINED, DEGRADATION_DETECTED, MARKET_MOVE, IDEAS_DISCOVERED, REGIME_CHANGE, RISK_* (warning/sizing_override/entry_halt/liquidation/emergency), KILL_SWITCH_TRIGGERED, LOOP_HEARTBEAT/ERROR.

### 10. Alpha Discovery

**Nightly strategy discovery pipeline (runs without Claude Code session):**
1. Load daily OHLCV -> Detect regime -> Select parameter templates -> Bounded grid (<=200 combos)
2. Two-stage filter: IS screen -> OOS walk-forward
3. Register passing candidates as status='draft'

**HypothesisAgent:** Groq/llama-3.3-70b for rule generation, temperature 0, 20s timeout, max 5 hypotheses.
**Grammar GP:** Genetic programming for rule evolution via mutation/crossover.

### Phase 10 Foundation Readiness

**In place:** Tool infrastructure, research graph skeleton, learning feedback loops, risk enforcement, event bus, alpha discovery, DB schema, autonomous patching, testing framework.

**Phase 10 will build:** Multi-hypothesis parallel exploration, overnight autoresearch, feature factory, knowledge graph, consensus validation, metacognitive self-modification, hierarchical governance.

---

## Part 2: Web Research — Best Practices (2025-2026)

### Topic 1: LangGraph Multi-Agent Orchestration

**Parallel Execution:** LangGraph's `Send` API enables dynamic parallelism — spawn N hypothesis-validation tasks with isolated state slices, results merge via reducer functions (`Annotated[list, operator.add]`). Already aligned with QuantStack's fan-out pattern.

**Three Canonical Topologies:**
1. Multi-Agent Collaboration: shared message scratchpad, rule-based router
2. Supervisor: central orchestrator, workers with independent scratchpads
3. Hierarchical Teams: nested subgraphs with recursive decomposition

**The `Command` primitive** unifies state updates with routing decisions.

**Overnight/Batch Research:** LangGraph Platform supports background runs, cron jobs, horizontally scalable task queues, persistent state. Implement budget tracker as state field that nodes decrement; conditional edge routes to "synthesize_findings" when exhausted.

**Consensus Pattern:** No built-in primitive, but clean via `Send` + reducer + consensus_node. Anthropic recommends voting pattern for high-confidence decisions. For trade entry: 3-agent voting (technical, fundamental, risk), consensus node requires 2/3 agreement.

**Agent Communication:** Prefer deterministic workflows over autonomous agents. All communication through state graph channels. Anti-pattern: agents calling agents directly.

Sources: LangGraph Docs, Anthropic "Building Effective Agents", LangChain Blog

### Topic 2: Knowledge Graphs in PostgreSQL

**JSONB Strengths:** Decomposed binary storage, GIN indexing (@>, ?, JSONPath), jsonb_path_ops (50-70% smaller indexes), subscript updates (PG14+), full ACID.

**Recommended Schema:**
```sql
-- Nodes: strategies, factors, signals, instruments
CREATE TABLE kg_nodes (
    id UUID PRIMARY KEY,
    node_type TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Edges: relationships between nodes
CREATE TABLE kg_edges (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES kg_nodes(id),
    target_id UUID REFERENCES kg_nodes(id),
    edge_type TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ
);
```

**pgvector:** HNSW index for embedding similarity. Combine with B-tree indexes on filter columns for hybrid queries.

**Hybrid Approach:** Relational edges for structured traversal (recursive CTEs), JSONB for flexible metadata, pgvector for semantic similarity. No need for Neo4j at QuantStack's scale (thousands of nodes, tens of thousands of edges).

**Factor Correlation:** Store correlation coefficients as edge properties with valid_from/valid_to for temporal decay.

### Topic 3: Metacognitive / Self-Modifying Architectures

**Agent Symbolic Learning (arXiv:2406.18532):** Treat agent components as learnable weights. Prompts = weights, natural language evaluation = loss function, natural language critique = gradient, backpropagation through agent graph.

**Reflection Patterns (progressively sophisticated):**
1. Basic Reflection: generator + reflector loop (N iterations)
2. Reflexion: grounded in external data, evidence-based feedback
3. LATS: Monte-Carlo tree search + reflection, explores multiple trajectories

**DSPy for Prompt Optimization:** Treat LLM pipelines as compilable programs. Optimizers (BootstrapRS, MIPROv2, BootstrapFinetune) automatically improve against measured metrics. 25%+ improvement over hand-crafted prompts. ~$2 per optimization run. Practical path to "threshold tuning from outcome data."

**Safety Guardrails:**
- File-path allowlists (NEVER touch risk_gate.py, kill_switch.py)
- Parallel safety classification: safety classifier + diff reviewer + regression runner
- Maximum iteration limits for self-modification loops
- Human feedback checkpoints for critical decisions

Sources: arXiv:2406.18532, LangChain Blog (Reflection Agents), DSPy.ai, Anthropic (Building Effective Agents)

### Topic 4: LLM Token Budget Management & Cost Optimization

**FrugalGPT Cascade (arXiv:2305.05176):**
- Route to cheapest model first, escalate on low confidence
- Up to 98% cost reduction matching GPT-4 performance
- QuantStack application: Haiku (default) -> Sonnet (confidence < 0.7) -> Opus (breakthrough/disagreement)

**Anthropic Prompt Caching:**
- 90% savings on cached tokens (5-min TTL)
- Cache tool definitions (static across cycles): 80%+ savings
- Cache system prompts: 85%+ savings on multi-turn
- Minimum cacheable: Opus 4,096 tokens, Sonnet/Haiku 2,048 tokens
- Tool definition changes invalidate entire cache — keep stable

**Hierarchical Governance (Orchestrator-Workers):**
- CIO (Sonnet/Opus): daily mandate, high blast radius decisions
- Department orchestrators (Sonnet): trading decisions, research synthesis
- Worker nodes (Haiku): data gathering, signal collection, narrow tasks
- Target: 70%+ tokens through Haiku with prompt caching

**Budget Formula:**
```
cycle_budget = base_cost_per_agent * num_agents * avg_turns
effective_cost = cycle_budget * (1 - cache_hit_rate * 0.9)
experiment_priority = (P(success) * expected_alpha * capital) / (compute_cost + data_cost + opportunity_cost)
```

**Concrete Estimates:**
- Research cycle (8 agents, ~5 turns, Sonnet): ~$5/cycle without caching, ~$0.90/cycle with Haiku + caching
- Trading cycle (10 agents, ~3 turns, mixed): ~$0.50/cycle with caching + Haiku
- Target: route 70%+ tokens through Haiku + prompt caching for 70%+ cost reduction

Sources: arXiv:2305.05176 (FrugalGPT), Anthropic Prompt Caching Docs

### Cross-Cutting Recommendations

1. **Prompt caching is lowest-hanging fruit** — enable immediately, expected 50%+ bill reduction
2. **Use `Send` API for parallel research streams** — dynamic fan-out with isolated state
3. **Knowledge graph in PostgreSQL** — kg_nodes + kg_edges + pgvector, no separate graph DB needed
4. **DSPy for self-improving prompts** — define outcome metrics, ~$2 per optimization run
5. **FrugalGPT cascade** — Haiku default, escalate to Sonnet/Opus on confidence/disagreement
6. **Self-modification guardrails** — file-path allowlists, regression tests, blocks on risk/execution code
