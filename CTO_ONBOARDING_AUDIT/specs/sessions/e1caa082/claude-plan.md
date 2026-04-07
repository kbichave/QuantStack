# Implementation Plan: Phase 10 — Advanced Research

## Overview

### What We're Building

QuantStack is an autonomous trading system built on LangGraph agents with PostgreSQL state management. It currently runs three graph services (research, trading, supervisor) that discover strategies, execute trades, and monitor health. Phase 10 transforms this from a system that trades into a system that improves itself — overnight autoresearch, a knowledge graph for institutional memory, metacognitive agents that tune their own prompts, and a hierarchical governance layer that cuts LLM costs by 70%+.

### Why

The current system generates ~12 strategy hypotheses per research cycle and has no memory of what it's already tested. Research agents waste tokens re-exploring dead ends. All agents run at the same LLM tier regardless of task complexity. There's no systematic connection between trade losses and research priorities. Phase 10 closes these gaps: 16x nightly research throughput, 224x weekend throughput, zero redundant research, and a token cost reduction from ~$150-450/day to ~$10/day.

### Architecture Context

The codebase (~185k lines) has these key subsystems that Phase 10 builds on:

- **Tool Registry** (`src/quantstack/tools/registry.py`): Centralized `TOOL_REGISTRY` with 100+ tools, deferred BM25 loading, per-agent subsets via `agents.yaml`
- **Research Graph** (`src/quantstack/graphs/research/`): Full hypothesis-to-strategy pipeline with fan-out parallel validation
- **AutoResearchClaw** (`scripts/autoresclaw_runner.py`): Autonomous task runner (bug_fix, ml_arch_search, strategy_hypothesis) in Docker sandbox, currently weekly
- **Event Bus** (`src/quantstack/coordination/event_bus.py`): Poll-based append-only coordination between research/trading/supervisor loops
- **Learning Layer** (`src/quantstack/learning/`): Drift detector (PSI), outcome tracker (regime affinity weights), IC attribution, skill tracker
- **Risk Gate** (`src/quantstack/execution/risk_gate.py`): ~2000 lines of hard-coded controls — immutable, never modified by any Phase 10 work
- **LLM Routing** (`src/quantstack/llm/`): Tier-based model selection (light=Haiku, heavy=Sonnet), Bedrock primary, LiteLLM fallback

The 5-graph architecture from Phase 5 is substantially complete. Phase 10 adds governance on top.

### Constraints

- **Home machine** (always-on Mac/Linux, local Docker, limited GPU)
- **$10/night LLM budget** for autoresearch (flexible upward if needed)
- **Haiku for bulk work**, Sonnet for synthesis, Opus only for AutoResearchClaw
- **Full v4.0 vision** delivered in one implementation, strict sub-phase ordering (10A -> 10B -> 10C -> 10D)
- **Risk gate is immutable** — never weakened, only strengthened

### Delivery Order

| Sub-Phase | Duration | Key Deliverables |
|-----------|----------|------------------|
| 10A: Foundation | 3 weeks | Tool lifecycle, loss-driven research, experiment budgets |
| 10B: Research Multiplier | 5 weeks | Overnight autoresearch, feature factory, weekend parallel streams |
| 10C: Intelligence Layer | 5 weeks | Knowledge graph, consensus validation |
| 10D: Self-Improvement | 6 weeks | Metacognitive agents, hierarchical governance |

---

## Section 1: Tool Lifecycle (AR-8)

### Problem

92 of 122 tools in `TOOL_REGISTRY` are stubs that return errors when invoked. Agents waste LLM round-trips discovering tools exist, calling them, getting errors, and retrying. The deferred loading system (BM25 search) surfaces stubs alongside working tools because it matches on descriptions, not health status.

### Design

**Registry Split.** Introduce two dictionaries: `ACTIVE_TOOLS` and `PLANNED_TOOLS`. Both live in `registry.py`. The existing `TOOL_REGISTRY` becomes a computed union for backward compatibility, but all agent-facing code (`bind_tools_to_llm`, `get_tools_for_agent_with_search`) reads only from `ACTIVE_TOOLS`. A tool's classification is determined by a new `tool_status` field in the registration decorator or a `tool_manifest.yaml` that lists each tool's status.

**Health Monitoring.** New table `tool_health`:

```python
@dataclass
class ToolHealthRecord:
    tool_name: str
    invocation_count: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    last_invoked: datetime
    last_error: str | None
    status: str  # "active", "degraded", "disabled"
```

Every tool invocation (in the `@tool` wrapper or a middleware layer in `tools/_helpers.py`) increments counters and logs latency. A daily health check (part of supervisor graph) computes trailing 7-day success rate. Tools below 50% are auto-disabled: moved from `ACTIVE_TOOLS` to a `DEGRADED_TOOLS` holding pen, and a `TOOL_DISABLED` event published to event bus. Tools can be re-enabled manually or by AutoResearchClaw after a fix.

**Demand Signal Tracking.** When `search_deferred_tools` finds a match in `PLANNED_TOOLS`, log the search query, requesting agent, and matched tool name to a `tool_demand_signals` table. A weekly aggregation job ranks planned tools by demand frequency. The top 3 become synthesis tasks in `research_queue`.

**Tool Synthesis Pipeline.** Extend AutoResearchClaw with a new task type `tool_implement`. Input: tool name, description, expected I/O from `PLANNED_TOOLS` definition. AutoResearchClaw generates the implementation in `tools/langchain/` or `tools/functions/`, validates (py_compile + import + invoke with test fixture), and if passing, moves the tool from `PLANNED_TOOLS` to `ACTIVE_TOOLS`. A `TOOL_ADDED` event fires, and agents reload bindings next cycle.

**Capability Announcement.** New event types `TOOL_ADDED` and `TOOL_DISABLED` on the event bus. Research and trading graphs poll for these at cycle start (already poll for other events in `context_load`). On `TOOL_ADDED`, the graph rebuilds its tool binding. On `TOOL_DISABLED`, the graph removes the tool from its working set.

### Key Decisions

- **Manifest vs. decorator**: A `tool_manifest.yaml` is easier to audit and modify without touching Python code. The alternative (a `status` field in each `@tool` decorator) keeps everything co-located but requires Python changes to reclassify. Recommend manifest for operational flexibility.
- **Health check frequency**: Daily is sufficient. Real-time health monitoring would add per-invocation latency. The 7-day trailing window smooths transient failures.
- **Auto-disable threshold**: 50% success rate over 7 days. This is aggressive enough to catch broken tools quickly but forgiving enough to tolerate intermittent API failures.

### Files to Create/Modify

```
src/quantstack/tools/
  tool_manifest.yaml          # NEW: tool name -> status mapping
  registry.py                 # MODIFY: split into ACTIVE/PLANNED, read manifest
  _helpers.py                 # MODIFY: add health tracking middleware
  health_monitor.py           # NEW: daily health check, auto-disable logic
src/quantstack/coordination/
  event_bus.py                # MODIFY: add TOOL_ADDED, TOOL_DISABLED event types
src/quantstack/graphs/
  tool_binding.py             # MODIFY: read only ACTIVE_TOOLS
scripts/
  autoresclaw_runner.py       # MODIFY: add tool_implement task type
```

---

## Section 2: Error-Driven Research (AR-7)

### Problem

When a trade loses money, the system records the loss in `fills` and adjusts regime affinity weights via `outcome_tracker`. But there's no systematic analysis of *why* the loss happened or connection to research priorities. The same failure mode can repeat indefinitely because nothing tells the research graph to investigate it.

### Design

**Daily Loss Analysis Pipeline.** A new module `src/quantstack/learning/loss_analyzer.py` runs at 16:30 ET (after market close, before overnight research). Five stages:

1. **COLLECT**: Query `fills` and `positions` for today's closed losers (negative realized P&L). Include metadata: strategy_id, entry regime, exit regime, holding period, signal strength at entry.

2. **CLASSIFY**: Map each loss to a failure mode from the Phase 7 taxonomy. The taxonomy includes: bad_entry_timing, regime_mismatch, liquidity_trap, model_degradation, signal_decay, adverse_selection, fat_tail_event, correlation_breakdown. Classification uses a combination of deterministic rules (e.g., if entry_regime != exit_regime -> regime_mismatch) and a Haiku call for ambiguous cases.

3. **AGGREGATE**: Maintain a rolling 30-day window in a `failure_mode_stats` table. Track per-mode: frequency, cumulative P&L impact, average loss size, affected strategies.

4. **PRIORITIZE**: Rank failure modes by cumulative P&L impact (not frequency — a few large losses outweigh many small ones).

5. **GENERATE**: For the top 3 failure modes, create `research_queue` tasks with context: the failure mode, affected strategies, example losses, suggested research direction (e.g., "regime_mismatch on AAPL swing strategy — research regime transition detection improvements").

The research graph's `context_load` node already polls `research_queue` for pending tasks. These error-driven tasks will be picked up in the next research cycle alongside community intel and hypothesis tasks.

### Closed Loop Verification

To verify the loop is closed: when a research task generated from a loss analysis produces a new strategy or hedge, tag the `research_queue` entry with the resulting strategy_id. Track whether that strategy subsequently prevents the same failure mode. This is observable but not enforced — it's a monitoring metric, not a gate.

### Key Decisions

- **Haiku for ambiguous classification** vs. purely deterministic: Some losses don't fit clean categories (e.g., a liquidity trap that also had signal decay). Haiku can reason about the combination. Cost: ~$0.001/classification, negligible.
- **30-day rolling window**: Long enough to capture patterns, short enough to adapt to regime changes. Could be configurable via env var.
- **Top 3 per day**: Keeps research queue manageable. The overnight autoresearch (AR-1) can only handle ~96 experiments, so flooding the queue doesn't help.

### Files to Create/Modify

```
src/quantstack/learning/
  loss_analyzer.py            # NEW: 5-stage loss analysis pipeline
  failure_modes.py            # NEW: failure mode taxonomy + classification rules
src/quantstack/data/
  scheduled_refresh.py        # MODIFY: add 16:30 ET loss analysis trigger
```

---

## Section 3: Experiment Budget Discipline (AR-9)

### Problem

Research agents have no cost constraints. A single research cycle can consume unlimited tokens if the hypothesis generation loop retries repeatedly or the ML experiment node trains multiple models. There's no way to compare experiment value against compute cost.

### Design

**Budget State Fields.** Add to `ResearchState`:

```python
class ResearchState(BaseModel):
    # ... existing fields ...
    token_budget_remaining: int = 50_000  # per-cycle cap
    cost_budget_remaining: float = 0.50   # per-cycle $ cap
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
```

Similarly for `TradingState`: 30K tokens, $0.20 per cycle.

**Deployment Note:** Both `ResearchState` and `TradingState` use `ConfigDict(extra="forbid")`. Adding new fields requires a clean restart of all graph services with empty checkpoint state — in-flight checkpoints will be incompatible. This is a one-time migration cost per deployment.

**Scope Clarification:** Per-cycle budgets apply to the daytime interactive research graph only. The overnight autoresearch (AR-1) uses its own per-experiment budget, independent of per-cycle limits. The two budget systems do not interact.

**Budget Tracking.** After each LLM call within a graph node, update the consumed counters. LangFuse callbacks already track token usage — tap into the existing callback handler to extract token counts and model-specific costs. Add a `budget_check` conditional edge after each LLM-calling node: if remaining budget < estimated_next_node_cost, route to a `synthesize_partial_results` terminal node that summarizes what was accomplished and what was deferred.

**Experiment Prioritization.** When multiple experiments compete for budget, rank them:

```
priority = (expected_IC * regime_fit * novelty_score) / estimated_compute_cost
```

Where:
- `expected_IC`: Prior from similar strategies in knowledge graph (AR-3), or 0.03 default for novel hypotheses
- `regime_fit`: How well the hypothesis matches current regime (from RegimeDetector)
- `novelty_score`: 1.0 if not in knowledge graph, 0.1 if similar hypothesis tested before
- `estimated_compute_cost`: Token estimate based on hypothesis complexity (simple rule = 5K tokens, ML model = 30K tokens)

This formula lives in a new `experiment_prioritizer.py` module. The research graph's `context_load` node calls it to sort the experiment queue before selecting which hypotheses to validate.

**Patience Protocol.** The existing backtest validation uses a single time window. Change to 3 mandatory windows:
1. Full historical period (e.g., 2020-2025)
2. Recent period (last 12 months)
3. Stressed period (COVID crash or 2022 rate hikes, configurable)

A hypothesis is only rejected if it fails gates in ALL 3 windows. Passing 2/3 -> mark as "provisional" (lower confidence, smaller position sizing). This prevents premature rejection of strategies that work in specific regimes.

### Key Decisions

- **Budget per cycle vs. per day**: Per cycle is simpler and maps to graph state. The overnight autoresearch (AR-1) will have its own budget mechanism (nightly ceiling).
- **Synthesize vs. abort**: When budget is exhausted, synthesize partial results rather than aborting. This captures value from work already done.
- **3 windows**: The spec says "3 time windows." The specific windows above are configurable. The important thing is that the patience protocol is enforced by the graph structure, not by agent judgment.

### Files to Create/Modify

```
src/quantstack/graphs/
  state.py                    # MODIFY: add budget fields to ResearchState, TradingState
  research/nodes.py           # MODIFY: add budget_check conditional edges
  research/graph.py           # MODIFY: add synthesize_partial_results node + routing
src/quantstack/learning/
  experiment_prioritizer.py   # NEW: prioritization formula + queue sorting
src/quantstack/core/
  backtesting.py              # MODIFY: add multi-window validation (patience protocol)
```

---

## Section 4: Overnight Autoresearch (AR-1)

### Problem

Research runs only when a Claude session is active or when AutoResearchClaw fires weekly. This means ~12 hypotheses/week. The opportunity cost is enormous: 8 overnight hours * 7 nights = 56 hours/week of idle compute.

### Design

**Overnight Mode.** A new runner `src/quantstack/research/overnight_runner.py` that operates the research graph in a tight loop from 20:00-04:00 ET nightly. Each iteration:

1. Generate hypothesis via Haiku (fast, ~2 seconds, ~500 tokens)
2. Quick-screen: does it overlap with knowledge graph? (After AR-3 is built; before that, skip)
3. Run backtest with 5-minute wall-clock budget (most backtests complete in seconds since they're CPU-bound on historical data)
4. Score by OOS IC on purged holdout
5. Log to `autoresearch_experiments` table
6. If OOS IC > 0.02: mark as "winner" for morning validation

**Budget Enforcement.** Track cumulative LLM cost across all experiments in the night, **persisted to DB** (a `nightly_budget` row in `autoresearch_experiments` keyed by night_date). Halt when approaching $10 ceiling ($9.50 trigger to leave headroom for morning validation). At Haiku rates (~$0.001/experiment for LLM), the budget is not the binding constraint — wall-clock time is. The 5-minute budget per experiment is a **timeout** (kill if exceeding 5 minutes), not a sleep interval. Experiments run back-to-back. Since most backtests complete in <10 seconds, actual throughput is likely hundreds per night. The 96 figure is a conservative lower bound.

**Crash Recovery.** On crash and restart, the runner reads the last cumulative cost from DB and resumes from there. Completed experiments are already logged and not re-run (idempotent via experiment_id). The morning validator at 04:00 runs unconditionally — even if only 30 experiments completed, it validates whatever winners exist.

**Budget Scope.** Each experiment uses a per-experiment budget of ~$0.10 (Haiku hypothesis + backtest evaluation + scoring). This is independent of the per-cycle budget from AR-9, which governs daytime interactive research only.

**Morning Validation Pipeline.** At 04:00 ET, a separate job evaluates winners:
- Run full multi-window backtest (the 3-window patience protocol from AR-9)
- Use Sonnet for deeper analysis of the strategy's regime fit
- If passing: register as status='draft' in strategies table
- If failing: log failure reason in `autoresearch_experiments` for knowledge graph learning

**Scheduling Change.** Modify `scripts/scheduler.py` and `docker-compose.yml` to run the overnight loop nightly instead of AutoResearchClaw's current Sunday-only schedule. AutoResearchClaw's weekly bug_fix/ml_arch_search tasks move to a separate weekly slot.

**New Table:**

```python
@dataclass
class AutoresearchExperiment:
    experiment_id: str
    night_date: str
    hypothesis: str          # JSON: entry_rules, exit_rules, parameters
    hypothesis_source: str   # "haiku_generated", "error_driven", "feature_factory"
    oos_ic: float | None
    sharpe: float | None
    cost_tokens: int
    cost_usd: float
    duration_seconds: int
    status: str              # "tested", "winner", "validated", "rejected"
    rejection_reason: str | None
    created_at: datetime
```

### Key Decisions

- **Haiku for hypothesis generation**: At $0.25/MTok input, $1.25/MTok output, a 500-token hypothesis costs ~$0.0008. Even at 96/night, total LLM cost is ~$0.08. The budget constraint is almost irrelevant for Haiku-only experiments. The $10 ceiling matters more when Sonnet is used for morning validation.
- **5-minute experiment budget**: Most backtests complete in <10 seconds (CPU-bound on ~5 years of daily data). The 5-minute budget is a safety net for ML experiments that might involve training. If a backtest takes >5 minutes, it's likely stuck — kill and move on.
- **Purged holdout for OOS IC**: Use the last 20% of data as holdout, with a purge gap equal to the strategy's holding period. This prevents information leakage from overlapping windows.

### Files to Create/Modify

```
src/quantstack/research/
  overnight_runner.py         # NEW: 8-hour overnight research loop
  morning_validator.py        # NEW: 04:00 ET winner validation
scripts/
  scheduler.py                # MODIFY: nightly schedule, separate weekly slot
docker-compose.yml            # MODIFY: overnight service definition
```

---

## Section 5: Autonomous Feature Factory (AR-10)

### Problem

The system's feature set is manually defined in `src/quantstack/core/features.py` and `tools/functions/data_functions.py`. There's no systematic discovery of new features or detection of decaying ones beyond the per-collector IC attribution in `learning/ic_attribution.py`.

### Design

**Three-Phase Pipeline.** New module `src/quantstack/research/feature_factory.py`:

**Phase 1: Enumeration (weekly overnight, LLM-assisted).** Start with the existing base features (RSI, MACD, ADX, BB%, SMA crossovers, volume metrics). For each base feature, programmatically generate:
- Lags: 1, 2, 3, 5, 10, 21 days
- Rolling stats: mean, std, skew, zscore over 5, 10, 21, 63 windows
- Cross-interactions: ratio of feature A / feature B for all pairs
- Regime-conditional: feature value * regime_indicator (trending=1, ranging=0)

Then call Haiku with the base feature list and recent market context to hypothesize novel combinations: "Given these base features and the current trending_up regime with normal volatility, suggest 20 novel composite features that might have predictive power for 5-day forward returns." Parse Haiku's suggestions into computable feature definitions. Target: 500+ candidates total (mostly programmatic, ~50-100 from Haiku). **Hard cap: 2000 candidates.** If cross-interactions produce more (O(N^2) for N base features), prioritize by expected novelty before computing IC. This prevents runaway compute on home hardware.

**Phase 2: Screening (weekly overnight, deterministic).** For each candidate feature:
1. Compute values across all universe symbols over trailing 2 years
2. Calculate IC (Spearman rank correlation with 5-day forward returns)
3. Calculate IC stability (rolling 63-day IC standard deviation)
4. Filter: IC > 0.01 AND stability > 0.5
5. Correlation check: drop features with >0.95 Pearson correlation to any already-selected feature
6. Output: ~50-100 curated features, stored in `feature_candidates` table

**Phase 3: Monitoring (daily, deterministic).** Integrated with existing drift detector:
1. Compute PSI for each curated feature vs. its screening-time distribution
2. Track rolling IC per feature
3. If PSI > 0.25 (CRITICAL) or IC < 0.005 for 10 consecutive days: mark as decayed
4. Replace with next-best feature from screening pool (the one with highest IC that wasn't selected due to correlation overlap)
5. Fire `FEATURE_DECAYED` and `FEATURE_REPLACED` events

**New Table:**

```python
@dataclass
class FeatureCandidate:
    feature_id: str
    feature_name: str
    definition: str          # Computable expression (e.g., "rsi_14_lag_5 / bb_pct_21")
    source: str              # "programmatic", "haiku_generated"
    ic: float
    ic_stability: float
    correlation_group: str   # Which cluster this belongs to
    status: str              # "candidate", "curated", "active", "decayed", "replaced"
    screening_date: str
    decay_date: str | None
```

### Key Decisions

- **LLM-assisted enumeration**: The user specifically requested Haiku for novel feature discovery beyond templates. Cost: ~20 Haiku calls/week * ~1000 tokens each = ~$0.025/week. Negligible.
- **IC > 0.01 threshold**: Intentionally low. An IC of 0.01 is barely predictive, but the feature factory's job is to cast a wide net. The research graph and backtest validation will further filter.
- **0.95 correlation cutoff**: Standard in factor research. Prevents multicollinearity in downstream models without being so aggressive that it kills useful variants.

### Files to Create/Modify

```
src/quantstack/research/
  feature_factory.py          # NEW: 3-phase feature pipeline
  feature_enumerator.py       # NEW: programmatic + LLM-assisted enumeration
  feature_screener.py         # NEW: IC screening + correlation filtering
src/quantstack/learning/
  drift_detector.py           # MODIFY: integrate feature monitoring
```

---

## Section 6: Weekend Parallel Research (AR-5)

### Problem

Weekend is 56 hours of idle compute. Current weekend research is a single AutoResearchClaw run on Sunday evening.

### Design

**Four Parallel Research Streams.** A new runner `src/quantstack/research/weekend_runner.py` launches Friday 20:00 and runs until Monday 04:00. Each stream is an independent LangGraph subgraph spawned via the `Send` API:

1. **Factor Mining Stream**: Takes academic paper references (from knowledge graph seeds, arxiv queries via web search) -> extracts testable factor definitions -> computes IC on universe symbols -> logs results. Haiku for paper parsing, deterministic IC computation.

2. **Regime Research Stream**: Takes historical price data -> labels regimes at multiple timeframes (daily, weekly, monthly) -> tests regime-conditional allocation rules (e.g., "in trending_up, overweight momentum; in ranging, overweight mean_reversion") -> computes regime transition probabilities. Mostly deterministic with Haiku for transition hypothesis generation.

3. **Cross-Asset Signal Stream**: Takes bond yields, FX rates, commodity prices (from existing data acquisition) -> computes lead-lag relationships at various lags -> identifies cross-asset signals that predict equity returns. Purely deterministic (correlation, Granger causality).

4. **Portfolio Construction Stream**: Takes existing strategies -> tests alternative portfolio optimizers (risk parity, Black-Litterman, Hierarchical Risk Parity) -> compares portfolio Sharpe, max drawdown, turnover vs. current equal-weight allocation. Mostly deterministic.

**State Isolation.** Each stream gets its own state slice via `Send`. No cross-contamination. Results merge via a reducer into a `weekend_research_results` list in the parent state. Monday morning at 04:00, a synthesis node (Sonnet) reviews all stream results and creates prioritized research tasks.

**Budget.** ~$50/weekend (user confirmed budget flexibility). Most compute is deterministic (backtests, correlation, IC). LLM usage is primarily Haiku for hypothesis generation in streams 1 and 2.

### Key Decisions

- **4 streams, not N streams**: The spec defines exactly 4 domains. Starting with more would dilute focus. The parallel execution infrastructure (Send API + reducers) supports adding streams later.
- **No GPU for weekend research**: All ML experiments that need training are queued for the overnight loop (AR-1) which has more frequent scheduling. Weekend research focuses on breadth (many hypotheses) over depth (model training).
- **Monday synthesis via Sonnet**: Worth the cost because the synthesis across 4 streams requires reasoning about cross-domain patterns (e.g., "factor mining found momentum works in trending regime" + "regime research found we're entering trending regime" = "activate momentum strategies").

### Files to Create/Modify

```
src/quantstack/research/
  weekend_runner.py           # NEW: 56-hour parallel research coordinator
  streams/
    factor_mining.py          # NEW: academic paper -> testable factors
    regime_research.py        # NEW: regime labeling + conditional allocation
    cross_asset_signals.py    # NEW: lead-lag cross-asset analysis
    portfolio_construction.py # NEW: alternative optimizer comparison
scripts/
  scheduler.py                # MODIFY: Friday 20:00 weekend launch
```

---

## Section 7: Alpha Knowledge Graph (AR-3)

### Problem

The system has no memory of what it's tested. The research graph can (and does) generate hypotheses similar to previously rejected ones. There's no way to query "have we already tested RSI divergence on AAPL in a trending regime?" or "which of our strategies share the same underlying factors?" The existing `search_knowledge_base` tool searches a text knowledge base, not a structured graph.

### Design

**PostgreSQL-Native Graph.** Two new tables: `kg_nodes` and `kg_edges`. No separate graph database — this keeps everything in PostgreSQL with ACID guarantees and the existing `db_conn()` context managers.

**Node Types:**
- `strategy`: Links to strategies table. Properties: status, regime_affinity, Sharpe, win_rate.
- `factor`: A named predictive feature (e.g., "RSI divergence", "earnings momentum"). Properties: description, IC range, decay_rate.
- `hypothesis`: A tested idea. Properties: entry_rules, exit_rules, test_date, outcome.
- `result`: Backtest or live performance outcome. Properties: Sharpe, max_dd, IC, window.
- `instrument`: A traded symbol. Properties: sector, market_cap, ADV.
- `regime`: A market state. Properties: trend, volatility, start_date, end_date.
- `evidence`: Supporting data for a relationship. Properties: source, confidence, date.

**Edge Types:**
- `uses`: strategy -> factor (this strategy uses this factor)
- `tested_by`: hypothesis -> result (this hypothesis produced this result)
- `correlates_with`: factor -> factor (these factors are correlated, with weight)
- `contradicted_by`: hypothesis -> evidence (this evidence contradicts this hypothesis)
- `favors`: regime -> strategy (this regime favors this strategy)
- `contains`: instrument -> factor (this instrument exhibits this factor pattern)

**Embeddings.** Each node gets a vector embedding stored in a `vector(1536)` column indexed with HNSW. Use Amazon Titan Text Embeddings v2 via Bedrock (the existing primary LLM provider), falling back to local `sentence-transformers/all-MiniLM-L6-v2` if Bedrock is unavailable. Add an `embedding` tier to `src/quantstack/llm/provider.py` that resolves to Titan embeddings. This enables semantic similarity queries: "find hypotheses similar to 'RSI divergence in ranging market'."

**Query Patterns (implemented as tool functions):**

1. `check_hypothesis_novelty(hypothesis_text: str) -> NoveltyResult`: Embed the hypothesis, find top-5 similar hypotheses in kg_nodes. If any have >0.85 cosine similarity AND were tested in the same regime, return "redundant" with links to previous results.

2. `check_factor_overlap(strategy_id: str) -> OverlapResult`: Traverse strategy -> uses -> factor -> uses -> strategy. Count shared factors with existing positions. If >2 shared factors, return "crowded" with factor names and affected strategies.

3. `get_research_history(topic: str) -> list[HypothesisResult]`: Semantic search + edge traversal. "What have we learned about momentum in ranging regimes?" -> find hypothesis nodes with similar embeddings + regime edges pointing to "ranging".

4. `record_experiment(hypothesis, result, factors_used, regime)`: Create hypothesis node, result node, factor nodes (if new), and all edges. This is called by the research graph's `knowledge_update` node and by the overnight autoresearch logger.

**Population Strategy.** Backfill from existing data:
- `strategies` table -> strategy nodes + factor nodes (parsed from entry_rules) + uses edges
- `ml_experiments` table -> result nodes + tested_by edges
- `autoresearch_experiments` table (new from AR-1) -> hypothesis nodes + result nodes
- `ic_observations` -> evidence nodes supporting factor effectiveness

### Key Decisions

- **PostgreSQL over Neo4j**: At QuantStack's scale (thousands of nodes, tens of thousands of edges), PostgreSQL is sufficient. Recursive CTEs handle graph traversal for factor overlap (max 3 hops). The operational simplicity of one database far outweighs any query performance difference.
- **Embeddings for novelty detection**: Text matching would miss semantic equivalents ("RSI divergence" vs. "RSI bearish divergence from price"). Embeddings catch these with cosine similarity.
- **valid_from/valid_to on edges**: Factor correlations change over time. A factor that correlated with momentum in 2023 may not in 2025. Temporal edges prevent stale relationships from influencing decisions.

### Files to Create/Modify

```
src/quantstack/knowledge/
  graph.py                    # NEW: KnowledgeGraph class with CRUD + query methods
  models.py                   # NEW: Node, Edge, NoveltyResult, OverlapResult types
  population.py               # NEW: backfill from existing tables
  embeddings.py               # NEW: embed node text via text-embedding-3-small
src/quantstack/tools/langchain/
  knowledge_graph_tools.py    # NEW: LLM-facing tools wrapping graph queries
src/quantstack/tools/
  registry.py                 # MODIFY: register new KG tools in ACTIVE_TOOLS
src/quantstack/db.py          # MODIFY: add kg_nodes, kg_edges table creation
```

---

## Section 8: Consensus-Based Signal Validation (AR-6)

### Problem

A single research/trading agent makes entry decisions. If that agent has a blind spot (e.g., bullish bias in a topping market), there's no counterweight. The risk gate catches position sizing violations but not bad directional calls.

### Design

**Trade Decision Routing.** In the trading graph's entry scanning node, after a signal is generated, check the estimated notional: if > $5K, route to the consensus subgraph; if <= $5K, proceed directly to risk gate (current behavior).

**Consensus Subgraph.** Three agents spawned via LangGraph `Send`, each with independent state:

1. **Bull Advocate**: Given the signal, strategy, and market data, build the strongest possible case FOR entry. Use technical analysis, momentum, catalyst identification. Tools: signal_brief, fetch_market_data, search_knowledge_base.

2. **Bear Advocate**: Given the same inputs, build the strongest possible case AGAINST entry. Look for divergences, overhead resistance, adverse macro conditions, factor crowding (via knowledge graph). Tools: same as bull + compute_risk_metrics.

3. **Neutral Arbiter**: Evaluate the quality of both arguments. Score each on evidence strength (1-5), logical coherence (1-5), and data recency (1-5). Return a binary vote: ENTER or REJECT, plus confidence score.

**Deterministic Merge.** A `consensus_merge` node collects all three votes:
- 3/3 ENTER -> full position size (as computed by PositionSizer)
- 2/3 ENTER -> 50% of computed position size
- <2/3 ENTER -> reject trade entirely

No LLM in the merge — it's a simple count. The merge node writes the decision to a new `consensus_log` table for post-hoc analysis.

**Fully Automated.** No human in the loop for consensus decisions. The sizing rules are enforced in code. **Feature flag:** `CONSENSUS_ENABLED` env var (default true). When disabled, all trades bypass consensus and go directly to risk gate. This allows disabling consensus if it adds latency without improving outcomes.

**Consensus Log Table:**

```python
@dataclass
class ConsensusDecision:
    decision_id: str
    signal_id: str
    symbol: str
    notional: float
    bull_vote: str          # "ENTER" or "REJECT"
    bull_confidence: float
    bull_reasoning: str
    bear_vote: str
    bear_confidence: float
    bear_reasoning: str
    arbiter_vote: str
    arbiter_confidence: float
    arbiter_reasoning: str
    consensus_level: str    # "unanimous", "majority", "minority"
    final_sizing_pct: float # 1.0, 0.5, or 0.0
    created_at: datetime
```

### Key Decisions

- **$5K threshold**: Balances protection (most meaningful trades are >$5K) against latency (consensus adds ~30 seconds of LLM calls). The threshold should be configurable via env var.
- **No debate between agents**: The agents don't see each other's arguments. This prevents anchoring bias. They evaluate the signal independently.
- **Arbiter votes, not just scores**: The arbiter makes a binary decision because scoring without commitment leads to indecisive middle-ground. The arbiter must choose a side.

### Files to Create/Modify

```
src/quantstack/graphs/trading/
  consensus.py                # NEW: consensus subgraph (bull, bear, arbiter, merge)
  nodes.py                    # MODIFY: add consensus routing in entry scan
  graph.py                    # MODIFY: wire consensus subgraph
src/quantstack/graphs/trading/config/
  agents.yaml                 # MODIFY: add bull_advocate, bear_advocate, neutral_arbiter configs
```

---

## Section 9: Metacognitive Self-Modification (AR-2)

### Problem

Agent prompts and thresholds are static. The hypothesis_critique threshold (0.7), backtest Sharpe gate (0.5), IC gate (0.02) were chosen once and never updated based on outcomes. Prompt text was written once and not optimized for the system's evolving needs.

### Design

**Four Meta Agents.** Each operates on a different cadence and scope. All run as supervisor graph nodes.

**1. meta_prompt_optimizer (weekly):**
- Input: Agent outcomes over trailing 30 days (from skill_tracker, outcome_tracker, autoresearch_experiments)
- Analysis: For each task agent, compare outcomes before/after the most recent prompt change (if any). Identify patterns: which agents are underperforming? What types of hypotheses do they miss?
- Action: Use DSPy's MIPROv2 optimizer to search for better instructions. Define metric as strategy Sharpe improvement over 30 days. Generate 3 prompt variants. Deploy the most promising via A/B split: 50% of cycles use the new prompt, 50% use the current. After 1 week, evaluate. If improved: adopt. If not: revert.
- Output: Updated agent prompt in `agents.yaml` (or a separate `prompt_versions/` directory), committed with `meta:` prefix.

**Prerequisite: Threshold Extraction.** Before meta_threshold_tuner can operate, hardcoded thresholds in node logic (e.g., `if sharpe > 0.5` in `backtest_validation`, `if confidence >= 0.7` in `hypothesis_critique`) must be extracted to a `thresholds.yaml` config file. This is a refactor task within 10D, before the meta agents come online.

**2. meta_threshold_tuner (monthly):**
- Input: 30-day outcome data. For each gate threshold, compute: how many strategies passed that eventually succeeded? How many failed that would have succeeded?
- Analysis: If the false rejection rate (good strategies rejected) > 20%: lower threshold by 0.05. If the false acceptance rate (bad strategies passed) > 30%: raise threshold by 0.05.
- Action: Update threshold in `thresholds.yaml` (not hardcoded in node logic).
- Constraints: Thresholds have hard floors/ceilings. hypothesis_critique: [0.4, 0.9]. Sharpe gate: [0.2, 1.0]. IC gate: [0.005, 0.05]. Never adjusts beyond these bounds.

**3. meta_tool_selector (weekly):**
- Input: Tool usage data from `tool_health` table + agent outcomes.
- Analysis: For each agent, identify: tools in its binding that are never used (waste of prompt cache space). Tools NOT in its binding that it searches for via deferred loading (should be always-loaded). Tools correlated with successful outcomes (boost priority).
- Action: Update `agents.yaml` tool bindings. Add high-demand tools to always-loaded set. Remove never-used tools from binding.

**4. meta_architecture_critic (quarterly):**
- Input: Portfolio Sharpe vs. benchmark (SPY) over 90 days. Per-node latency and token consumption from Langfuse traces. Strategy win rates by domain.
- Analysis: Identify the bottleneck node (highest token cost per unit of alpha, or longest latency on critical path). Propose improvement: is the bottleneck a bad prompt (refer to meta_prompt_optimizer)? A missing tool? A structural graph issue?
- Recursive: If meta_prompt_optimizer's suggestions don't improve performance over 2 consecutive weeks, meta_architecture_critic modifies the optimizer's own prompt. This is the recursive self-modification the spec calls for.

**Guardrails:**
- Protected files: risk_gate.py, kill_switch.py, db.py, execution layer — NEVER modified by meta agents. Enforced by a file-path allowlist checked before any git commit.
- All changes committed to git with `meta:` prefix for audit trail.
- Regression test suite runs before any change goes live. Failure -> auto-revert.
- Maximum 3 prompt variants per week per agent.
- 30-day Sharpe monitoring: if Sharpe declines >10% after a meta change, auto-revert to previous version.

### Key Decisions

- **DSPy for prompt optimization**: The research confirmed DSPy's MIPROv2 as the most practical path. Cost: ~$2 per optimization run. Weekly runs = ~$8/month. Worth it for compounding improvement.
- **Fully autonomous**: User confirmed no human review gate. The guardrails (regression tests, Sharpe monitoring, protected files, auto-revert) provide safety without human bottleneck.
- **Quarterly architecture critic**: Longer cadence because architectural changes have higher blast radius and need more data to evaluate. Monthly would risk overfitting to short-term noise.
- **A/B split for prompts**: 50/50 split for 1 week gives ~168 hours of data across overnight and intraday cycles. Enough to detect meaningful differences.

### Files to Create/Modify

```
src/quantstack/meta/
  __init__.py                 # NEW
  prompt_optimizer.py         # NEW: DSPy-based prompt optimization
  threshold_tuner.py          # NEW: gate threshold adjustment
  tool_selector.py            # NEW: agent tool binding optimization
  architecture_critic.py      # NEW: quarterly bottleneck analysis
  guardrails.py               # NEW: file allowlist, auto-revert, regression runner
  config.py                   # NEW: thresholds.yaml loader, version tracking
src/quantstack/graphs/supervisor/
  nodes.py                    # MODIFY: add meta agent scheduling nodes
  graph.py                    # MODIFY: wire meta nodes into supervisor graph
```

---

## Section 10: Hierarchical Governance (AR-4)

### Problem

All agents run at the same tier (mostly Sonnet). A signal collection task that could use Haiku runs at Sonnet cost. There's no separation between strategic decisions (what to trade) and tactical execution (how to trade). Current daily token cost: $150-450. Target: ~$10/day.

### Design

**Three Governance Tiers.** Built on top of the existing Phase 5 five-graph architecture.

**Tier 1: CIO Agent (Sonnet, once/day at 09:00 ET).**
The CIO agent reviews:
- Overnight research results (autoresearch_experiments winners)
- Current market regime (from RegimeDetector)
- Portfolio state (open positions, P&L, exposure)
- Knowledge graph insights (factor crowding alerts, research memory)
- Meta agent reports (if any prompt/threshold changes were made)

Produces a `DailyMandate`:

```python
@dataclass
class DailyMandate:
    date: str
    regime_assessment: str
    allowed_sectors: list[str]
    blocked_sectors: list[str]
    max_new_positions: int
    max_daily_notional: float
    strategy_directives: dict[str, str]  # strategy_id -> "active"|"reduce"|"pause"|"exit"
    risk_overrides: dict                 # optional: tighten/relax specific risk params
    focus_areas: list[str]              # what should research prioritize today
    reasoning: str                       # audit trail
```

Mandate persisted to `daily_mandates` table. Cost: ~$0.15/day (one Sonnet call with cached system prompt).

**Mandate Failure Fallback.** If the CIO agent fails (LLM error, timeout, Bedrock outage) and no mandate exists for today by 09:30 ET, a conservative default mandate activates: `max_new_positions=0`, all existing positions in "monitor" mode (no exits unless stop-loss hit), all strategy_directives set to "pause". This degrades gracefully — the system pauses new entries without liquidating existing positions. The default mandate is generated deterministically (no LLM), ensuring the system always has valid constraints.

**Tier 2: Strategy Agents (Haiku, every 5 minutes during market hours).**
Four strategy agents, each responsible for a domain:
- Swing agent: momentum/trend following
- Investment agent: fundamental/value
- Options agent: volatility/structure trades
- Mean reversion agent: statistical arbitrage

Each agent operates within the CIO mandate:
- Only scans allowed sectors
- Respects max_new_positions and max_daily_notional
- Follows strategy_directives (if CIO says "pause AAPL_swing", agent skips AAPL)
- Uses Haiku for all LLM calls (signal collection, pattern recognition)
- Mandate adherence enforced in code: a `mandate_check` function validates every proposed action against the active mandate before reaching the risk gate

Cost: ~$3.20/day (4 agents * 6.5 hours * 78 cycles * ~$0.001/cycle).

**Tier 3: Risk Officer (deterministic, per-trade).**
The existing `risk_gate.py` — unchanged, immutable. No LLM cost. Checks position sizing, exposure limits, daily loss, liquidity, concentration. Additionally validates mandate adherence (new check: is this trade in an allowed sector? Does it exceed max_daily_notional?).

**Mandate Enforcement.** The mandate is not advisory — it's enforced by code. The `mandate_check` function in the trading graph reads the active mandate from `daily_mandates` table and rejects any proposed trade that violates it. This is a hard gate like the risk gate, not an LLM judgment call.

**Token Cost Breakdown:**
- CIO: 1 Sonnet call/day with ~4K cached system prompt: ~$0.15/day
- Strategy agents: 4 * Haiku * 78 cycles: ~$3.20/day
- Risk officer: $0/day (deterministic)
- Supervisor/monitoring: ~$1/day
- Total: ~$4.35/day operational + ~$5/day overnight research = ~$9.35/day
- Savings: $150-450/day -> ~$10/day = **93-98% reduction**

### Key Decisions

- **Daily mandate, not per-trade CIO**: A per-trade CIO call would add latency and cost. Daily is sufficient because regime doesn't change intraday (for swing/investment timeframes). The mandate can be updated intraday via `REGIME_CHANGE` event if needed.
- **Mandate enforcement in code**: If it were prompt-enforced, a hallucinating Haiku agent could ignore it. Code enforcement means the mandate is as hard as the risk gate.
- **4 strategy agents**: Maps to the 4 trading domains (swing, investment, options, mean_reversion). Each gets the tools and context relevant to its domain, keeping Haiku's limited context window focused.
- **Phase 5 as foundation**: The 5-graph architecture already separates research, trading, and supervision. AR-4 adds the CIO as a new node in the supervisor graph and restructures the trading graph's agent tier.
- **Implementation order within 10D**: Governance (AR-4) should be implemented BEFORE meta agents (AR-2) because meta agents need to target the NEW agent hierarchy (CIO + strategy agents), not the old flat model. The section numbering in this plan does not imply implementation order within a sub-phase.

### Files to Create/Modify

```
src/quantstack/governance/
  __init__.py                 # NEW
  cio_agent.py                # NEW: daily mandate generation
  mandate.py                  # NEW: DailyMandate dataclass + enforcement
  mandate_check.py            # NEW: hard gate for mandate compliance
src/quantstack/graphs/supervisor/
  nodes.py                    # MODIFY: add CIO scheduling (09:00 ET)
  graph.py                    # MODIFY: wire CIO node
src/quantstack/graphs/trading/
  nodes.py                    # MODIFY: add mandate_check before risk gate
  graph.py                    # MODIFY: restructure for Haiku-tier agents
  config/agents.yaml          # MODIFY: strategy agents to Haiku tier
src/quantstack/llm/
  provider.py                 # MODIFY: add "governance" tier mapping
```

---

## Section 11: AutoResearchClaw Upgrades

### Problem

The existing AutoResearchClaw (683 lines) has 4 gaps that block Phase 10: weekly-only scheduling, reactive-only task types, tmux-based restarts, and no functional validation.

### Design

**Nightly Schedule.** Change the scheduler from Sunday 20:00 only to nightly 20:00. The overnight autoresearch (AR-1) runs its own loop, but AutoResearchClaw still handles bug_fix, ml_arch_search, and the new tool_implement tasks on a separate schedule.

**Proactive Task Types.** Add two new task types to the existing 4:
- `tool_implement`: From AR-8 demand signals. Input: planned tool definition. Output: working tool in tools/langchain/ or tools/functions/.
- `gap_detection`: From AR-7 error-driven research. Input: failure mode analysis results. Output: research tasks targeting the gap.

These are fed by the loss analyzer (AR-7) and tool demand tracker (AR-8), not just by reactive bug reports.

**Docker Compose Restarts.** Replace `tmux send-keys` restart mechanism with `docker compose restart <service>`. This is more reliable (tmux session might not exist) and integrates with the Docker-based deployment.

**Functional Validation.** After a bug_fix or tool_implement patch, run the tool's test fixture (not just py_compile + import). The test fixture is a simple invocation with known inputs and expected output shape. If the invocation fails, the patch is reverted. This is stored in `tool_manifest.yaml` alongside each tool's status.

### Files to Create/Modify

```
scripts/
  autoresclaw_runner.py       # MODIFY: add task types, Docker restart, functional validation
  scheduler.py                # MODIFY: nightly schedule
src/quantstack/tools/
  tool_manifest.yaml          # MODIFY: add test_fixture field per tool
```

---

## Section 12: Event Bus Extensions

### Problem

The event bus needs new event types to support Phase 10's inter-component coordination.

### Design

Add the following event types to the `EventType` enum in `event_bus.py`:

- `TOOL_ADDED` — payload: tool_name, source (synthesis/manual)
- `TOOL_DISABLED` — payload: tool_name, reason, success_rate
- `EXPERIMENT_COMPLETED` — payload: experiment_id, status (winner/rejected), oos_ic
- `FEATURE_DECAYED` — payload: feature_id, psi, ic_current
- `FEATURE_REPLACED` — payload: old_feature_id, new_feature_id
- `MANDATE_ISSUED` — payload: mandate_id, key directives summary
- `META_OPTIMIZATION_APPLIED` — payload: agent_id, change_type (prompt/threshold/tool), change_summary
- `CONSENSUS_REQUIRED` — payload: signal_id, symbol, notional
- `CONSENSUS_REACHED` — payload: decision_id, consensus_level, final_sizing

Each event follows the existing pattern: append-only to event_log, polled by consumers via loop_cursors. No architectural changes to the event bus itself — just new enum values and documentation of payload schemas.

### Files to Create/Modify

```
src/quantstack/coordination/
  event_bus.py                # MODIFY: add new EventType values
  event_schemas.py            # NEW: payload schema documentation/validation
```

---

## Section 13: Database Migrations

### Problem

Phase 10 introduces 8+ new tables. These need idempotent migrations consistent with the existing pattern in `db.py`.

### Design

New tables (all created via `CREATE TABLE IF NOT EXISTS` in `db.py`'s `ensure_schema()` function):

1. `tool_health` — per-tool invocation metrics
2. `tool_demand_signals` — search queries matching planned tools
3. `autoresearch_experiments` — overnight experiment log
4. `feature_candidates` — enumerated/screened features
5. `failure_mode_stats` — rolling failure mode aggregation
6. `kg_nodes` — knowledge graph nodes
7. `kg_edges` — knowledge graph edges
8. `consensus_log` — 3-agent consensus decisions
9. `daily_mandates` — CIO daily directives
10. `meta_optimizations` — prompt/threshold/tool changes with before/after metrics

All follow existing patterns: UUID primary keys, TIMESTAMPTZ for timestamps, JSONB for flexible properties, ON CONFLICT DO UPDATE for idempotent writes.

The `kg_nodes` table additionally requires `pgvector` extension (`CREATE EXTENSION IF NOT EXISTS vector`) and an HNSW index on the embedding column.

**Docker Image Change.** The standard `postgres:16` Docker image does NOT include pgvector. Switch to `pgvector/pgvector:pg16` in `docker-compose.yml`. This is a drop-in replacement that adds the vector extension.

### Files to Create/Modify

```
src/quantstack/db.py          # MODIFY: add all new table definitions to ensure_schema()
docker-compose.yml            # MODIFY: switch PostgreSQL image to pgvector/pgvector:pg16
```

---

## Section 14: Prompt Caching

### Problem

The $10/day token cost target requires prompt caching. Without it, Haiku strategy agents running 78 cycles/day will exceed the budget on system prompt tokens alone.

### Design

**Enable Anthropic prompt caching** in `src/quantstack/llm/provider.py` for all Bedrock/Anthropic API calls.

**What to cache (in priority order):**
1. **Tool definitions** (`TOOL_REGISTRY` schemas): Static across cycles. These are the largest token consumers in agent prompts. Cache with `cache_control={"type": "ephemeral"}`. Expected savings: 80%+ on input tokens.
2. **System prompts** (per-agent from `agents.yaml`): Static unless meta_prompt_optimizer modifies them. Cache the system message. Expected savings: 85%+ on multi-turn conversations.
3. **Strategy context documents** loaded during research cycles: Cache the frequently-loaded strategy registry and knowledge base summaries.

**Implementation:**
- Add `cache_control` parameter support to `get_chat_model()` in `provider.py`
- For Bedrock: use the `anthropic_beta` header with `prompt-caching-2024-07-31`
- Minimum cacheable sizes: Opus 4,096 tokens, Sonnet 2,048, Haiku 2,048 — all agent system prompts exceed these
- **Keep tool definitions stable**: Any change to tool definitions invalidates the entire cache. The tool_manifest.yaml from AR-8 should be versioned; tool additions only happen via event bus notifications, not mid-cycle

**Expected Savings:**
- Without caching: ~$0.015/cycle (Haiku, 78 cycles = ~$1.17/day just for strategy agents)
- With caching (80% hit rate): ~$0.003/cycle (~$0.23/day for strategy agents)
- Total daily savings: ~$1-3/day on strategy agents alone, more on research cycles

### Files to Create/Modify

```
src/quantstack/llm/
  provider.py                 # MODIFY: add cache_control support
  config.py                   # MODIFY: add caching configuration
```

---

## Integration & Testing Strategy

### Unit Tests

Each new module gets unit tests following existing patterns:
- Feature factory: test enumeration produces expected count, screening filters correctly, monitoring detects decay
- Knowledge graph: test node/edge CRUD, novelty detection with mock embeddings, factor overlap traversal
- Consensus: test merge logic (3/3, 2/3, <2/3), threshold routing ($5K boundary)
- Meta agents: test threshold bounds enforcement, protected file allowlist, auto-revert logic
- Governance: test mandate generation, mandate enforcement (rejection on violation), tier routing
- Loss analyzer: test failure mode classification, aggregation, research task generation
- Budget discipline: test budget tracking, exhaustion routing, patience protocol

### Integration Tests

- Event bus: verify new event types publish/poll correctly
- Knowledge graph population: backfill from test strategies table, verify graph queries
- Overnight runner: simulate a short run (3 experiments), verify experiment log and morning validator
- Mandate flow: CIO produces mandate -> strategy agent reads it -> mandate_check enforces it

### Regression Tests

- Risk gate: unchanged behavior after Phase 10. Run existing risk gate tests plus new tests that verify mandate_check doesn't weaken risk gate.
- Kill switch: unchanged behavior. Verify kill switch overrides mandate (mandate says "trade" but kill switch is active -> no trade).
- Existing research graph: verify backward compatibility when budget fields have default values.

### Performance Tests

- Feature factory enumeration: 500+ candidates in < 10 minutes on home machine
- Knowledge graph query: factor overlap query < 100ms for 10K nodes
- Overnight experiment loop: verify 96+ experiments in 8 hours (5-min timeout each, back-to-back)

### Validation Milestones (per Sub-Phase)

These map to the spec's acceptance gates:

- **10A (week 3):** Active vs. stub tool count (target: 50+ active). Loss -> research -> task pipeline verified end-to-end. Budget exhaustion correctly routes to synthesis.
- **10B (month 2):** Overnight experiment count (target: 2000+/month). Winner rate 5-10% pass morning validation. Budget adherence verified ($10/night). Feature factory produces 500+ candidates.
- **10C (month 3):** Knowledge graph redundancy detection prevents re-testing a known hypothesis. Factor overlap query returns correct results for a known-crowded scenario. Consensus sizing correct for 3/3, 2/3, <2/3 cases.
- **10D (month 5):** Task agent win rates before/after meta-optimization show 5-10% improvement. Token cost reduced 70%+ vs. pre-governance baseline (measured over 7-day window).
