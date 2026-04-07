# Phase 10: Advanced Research — Complete Specification

## Mission

Transform QuantStack from a trading system into a self-improving trading company. The system doesn't just make money — it gets better at making money, and gets better at getting better. Deliver the complete v4.0 vision in one implementation across 4 sub-phases (~20 weeks, ~100 days).

## Delivery Model

Full vision delivery following strict spec order: 10A -> 10B -> 10C -> 10D. Each sub-phase depends on the previous. No items pulled forward or deferred.

## Infrastructure Constraints

- **Runtime:** Home machine (always-on Mac/Linux), local Docker, limited GPU
- **Nightly LLM budget:** $10/night ceiling (flexible upward if needed)
- **Primary LLM for bulk work:** Haiku for hypothesis generation and enumeration
- **Existing infrastructure:** Phase 5 graph restructuring substantially complete — governance layer (AR-4) builds on existing 5-graph architecture

## Existing System Foundation

### What's Already Built
- **Tool Registry:** 100+ LLM-facing tools in centralized `TOOL_REGISTRY` with deferred loading (BM25 search), 90% cache hit rate. Per-agent subsets via `agents.yaml`.
- **AutoResearchClaw:** Autonomous task runner (bug_fix, ml_arch_search, rl_env_design, strategy_hypothesis). Runs weekly Sunday 20:00 in Docker sandbox. Protected file guardrails (never touches risk_gate.py, kill_switch.py, db.py).
- **Research Graph:** Full pipeline: context_load -> domain_selection -> hypothesis_generation -> hypothesis_critique -> signal_validation -> backtest_validation -> ml_experiment -> strategy_registration -> knowledge_update. Fan-out parallel validation supported.
- **Risk Gate:** ~2000 lines of hard-coded controls: 10% position cap, $20K notional, 500K ADV minimum, -2% daily loss halt, correlation check, regime-conditional sizing.
- **Kill Switch:** Sentinel file persistence, survives restarts, closes all positions on trigger.
- **Learning Systems:** Drift detector (PSI), outcome tracker (regime affinity weights), IC attribution (per-collector quality), skill tracker (agent accuracy + ICIR).
- **Event Bus:** Poll-based, append-only, per-consumer cursors, 7-day TTL. 20+ event types covering strategy lifecycle, model lifecycle, risk, control plane.
- **Alpha Discovery:** Nightly grid search + GP + Groq hypothesis generation. Two-stage IS/OOS filtering. Registers as status='draft'.
- **LLM Routing:** Tier-based (light=Haiku, heavy=Sonnet, bulk=Haiku). Bedrock primary, multi-provider fallback. LiteLLM router optional.
- **Database:** PostgreSQL as single source of truth. psycopg3 pool (4-20 connections). Tables: positions, strategies, signals, fills, audit_trail, research_queue, ml_experiments, bugs, event_log, loop_cursors.
- **Testing:** 240+ unit tests, integration tests, regression tests. Synthetic OHLCV generators. pytest framework.

---

## Sub-Phase 10A: Foundation (Week 10-11, ~3 weeks)

### AR-8: Dynamic Tool Lifecycle

**Problem:** 92 of 122 tools are stubs. Agents offered tools that return errors, wasting LLM round-trips.

**Scope:** Full lifecycle — split registry + synthesis pipeline + health monitoring + auto-disable.

**Implementation:**
1. Split `TOOL_REGISTRY` into `ACTIVE_TOOLS` (~30 working) and `PLANNED_TOOLS` (~92 stubs). Agents only see active tools.
2. Health monitoring: Track invocation count, success rate, latency per tool in a `tool_health` table. Auto-disable tools with <50% success rate over trailing 7 days.
3. Demand signal tracking: Log tool search queries that match planned tools. Prioritize implementation of most-requested stubs.
4. Tool synthesis pipeline: AutoResearchClaw implements highest-priority planned tools based on demand signals. After implementation, tool moves from PLANNED to ACTIVE.
5. Capability announcement: `TOOL_ADDED` event via event bus -> agents reload bindings next cycle.

**Acceptance:**
- Agents only see active, working tools
- Tool health monitoring populating `tool_health` table
- Auto-disable triggers on <50% success rate
- Demand signals tracked and prioritized
- AutoResearchClaw synthesis pipeline operational (implements stubs from queue)

### AR-7: Error-Driven Research Iteration

**Problem:** No systematic connection from losses to research priorities.

**Depends on:** Phase 7 failure mode taxonomy.

**Implementation:**
Daily loss analysis pipeline (16:30 ET):
1. COLLECT: Query fills + positions for daily losers
2. CLASSIFY: Map each loss to failure mode taxonomy (bad entry timing, regime mismatch, liquidity trap, model degradation, etc.)
3. AGGREGATE: Rolling 30-day failure mode frequency + P&L impact
4. PRIORITIZE: Rank by cumulative P&L impact
5. GENERATE: Create research_queue tasks targeting top 3 failure modes
6. FEED: Tasks picked up by research graph next cycle

**Acceptance:**
- Daily pipeline runs automatically at 16:30 ET
- Research tasks generated targeting top failure modes
- Closed loop demonstrated: loss -> failure classification -> research task -> new strategy/hedge -> prevented future loss

### AR-9: Fixed-Budget Experiment Discipline

**Problem:** No experiment budgets. Runaway research agents consume unlimited tokens.

**Implementation:**
- Per-cycle budgets: 50K tokens (research), 30K (trading)
- Per-cycle cost ceiling: $0.50 (research), $0.20 (trading)
- Budget tracked as state field in graph; conditional edge checks remaining budget before routing to next node
- Experiment prioritization formula: `priority = (expected_IC * regime_fit * novelty_score) / estimated_compute_cost`
- Patience protocol: Never discard after 1 backtest — run 3 time windows minimum. Only reject if all 3 fail gates.
- Budget exhaustion: Route to "synthesize_findings" terminal node, summarize partial results

**Acceptance:**
- Per-cycle token and cost budgets enforced (observable in Langfuse traces)
- Experiment prioritization formula implemented and used for queue ordering
- 3-window patience protocol prevents premature rejection

---

## Sub-Phase 10B: Research Multiplier (Week 11-15, ~5 weeks)

### AR-1: Overnight Autoresearch Loop

**Impact:** 16x research throughput.

**Constraints:** $10/night LLM budget, home machine, Haiku for hypothesis generation.

**Implementation:**
1. Add `overnight_mode` to research graph runner (20:00-04:00 ET, 8 hours)
2. Create `autoresearch_node.py` with fixed 5-minute experiment budget per hypothesis
3. Haiku for hypothesis generation (speed > depth, fits budget)
4. Single metric: OOS IC on purged holdout
5. Budget tracking: sum token costs across experiments, halt when approaching $10 ceiling
6. Store log in `autoresearch_experiments` table (experiment_id, hypothesis, IC, cost, duration, status)
7. Morning pipeline (04:00): evaluate winners with full backtest (Sonnet for validation)
8. Nightly scheduling: change AutoResearchClaw from Sunday-only to nightly 20:00

**Experiments per night estimate:** ~96 at Haiku rates within $10 budget (each experiment: hypothesis gen ~500 tokens + backtest evaluation ~1000 tokens ~ $0.001/experiment for LLM + CPU time for backtest).

**Acceptance:**
- Overnight mode runs 8 hours producing ~96 experiments
- Winners automatically queued for morning full validation
- Experiment log queryable for research quality tracking
- Stays within $10/night budget

### AR-10: Autonomous Feature Factory

**Impact:** 10x feature coverage.

**Implementation:**
Three phases:
1. **Enumeration (weekly overnight, LLM-assisted):** Use Haiku to hypothesize novel feature combinations beyond programmatic templates. Base features -> lags, rolling stats, cross-interactions, regime-conditional variants, residualized features. Target: 500+ candidates. LLM suggests creative combinations (e.g., "RSI momentum divergence from sector ETF RSI").
2. **Screening (weekly overnight):** Compute IC for all candidates. Filter: IC > 0.01 AND stability > 0.5. Drop >0.95 correlation with existing features. Output: ~50-100 curated features.
3. **Monitoring (daily):** PSI drift per feature (reuse existing drift_detector). IC decay tracking per feature. Auto-replace decaying features with next-best from screening pool.

**Acceptance:**
- Feature enumeration produces 500+ candidates weekly
- IC screening filters to 50-100 curated features
- Daily monitoring auto-replaces decaying features
- LLM-assisted enumeration discovers features not in programmatic templates

### AR-5: Long-Horizon Parallel Research

**Impact:** 224x weekend research throughput.

**Implementation:**
4 parallel research streams (Friday 20:00 -> Monday 04:00, 56 hours):
1. **Factor Mining:** Academic paper references -> testable factors -> IC screening
2. **Regime Research:** Historical regime labeling -> regime-conditional allocation optimization
3. **Cross-Asset Signals:** Bond-equity-FX-commodity lead-lag relationships
4. **Portfolio Construction:** Alternative optimizers (risk parity, Black-Litterman, HRP)

Use LangGraph `Send` API for dynamic fan-out. Each stream runs as independent subgraph with isolated state. Results merge via reducer into shared weekend_research_results.

Budget: ~$50/weekend (56 hours * 4 streams, mostly Haiku with Sonnet for synthesis). User confirmed budget flexibility.

**Acceptance:**
- 4 parallel weekend research streams operational
- ~2,688 experiments per weekend (vs. ~12 today)
- Monday morning: curated results ready for validation

---

## Sub-Phase 10C: Intelligence Layer (Week 15-18, ~5 weeks)

### AR-3: Alpha Knowledge Graph

**Impact:** Full research memory, factor crowding prevention.

**Primary use case:** Full research memory — track all hypotheses tested, results, contradictions, prevent redundant research. Factor overlap detection is a secondary but important query pattern.

**Implementation:**
PostgreSQL-native (no separate graph DB):

```sql
CREATE TABLE kg_nodes (
    id UUID PRIMARY KEY,
    node_type TEXT NOT NULL,  -- 'strategy', 'factor', 'signal', 'hypothesis', 'result', 'instrument', 'regime'
    properties JSONB NOT NULL DEFAULT '{}',
    embedding vector(1536),   -- pgvector for semantic similarity
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE kg_edges (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES kg_nodes(id),
    target_id UUID REFERENCES kg_nodes(id),
    edge_type TEXT NOT NULL,  -- 'uses', 'tested_by', 'correlates_with', 'contradicted_by', 'favors'
    properties JSONB NOT NULL DEFAULT '{}',  -- weight, confidence, timeframe
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ
);
```

Query patterns:
- **Redundancy detection:** "Has this hypothesis been tested before?" -> semantic similarity on hypothesis embeddings + structured edge traversal
- **Factor overlap:** "Does this strategy share >2 factors with existing positions?" -> recursive CTE from strategy -> uses -> factor -> uses -> strategy
- **Contradiction tracking:** "What evidence contradicts this factor?" -> contradicted_by edges
- **Research memory:** "What have we learned about momentum in ranging regimes?" -> node_type + regime filter + edge traversal

Populate from:
- Existing strategies table -> strategy nodes + factor nodes + uses edges
- ml_experiments table -> result nodes + tested_by edges
- autoresearch_experiments table -> hypothesis nodes + result nodes
- IC observations -> evidence nodes + supports/contradicts edges

**Acceptance:**
- Knowledge graph populated with strategies, factors, results, hypotheses
- Redundancy detection prevents re-testing known hypotheses
- Factor-level correlation available for pre-trade checks
- Research memory queryable by regime, factor, outcome

### AR-6: Consensus-Based Signal Validation

**Impact:** Fewer catastrophic trades.

**Handling:** Fully automated sizing, no human in the loop.

**Implementation:**
For high-stakes decisions (>$5K notional):
1. Spawn 3 independent agents via LangGraph `Send`:
   - Bull advocate: finds reasons to enter
   - Bear advocate: finds reasons against
   - Neutral arbiter: evaluates evidence quality
2. Each agent has independent state (no cross-contamination)
3. Deterministic merge: 3/3 agree -> full size; 2/3 -> 50% size; <2/3 -> reject
4. Low-stakes (<$5K): single agent (current behavior, no latency penalty)

Store consensus decisions in `consensus_log` table for post-hoc analysis of disagreement patterns.

**Acceptance:**
- High-stakes trades require consensus from 3 agents
- Consensus level determines position sizing automatically
- Low-stakes trades bypass (no latency penalty)
- Consensus log populated for analysis

---

## Sub-Phase 10D: Self-Improvement (Week 18-24, ~6 weeks)

### AR-2: Metacognitive Self-Modification

**Impact:** Compounding quality improvement.

**Safety model:** Fully autonomous with guardrails. Regression tests + protected file allowlist. No human review gate for prompt changes.

**Implementation:**
4 meta agents:
1. **meta_prompt_optimizer** (weekly): Analyze agent outcomes -> generate prompt variants -> A/B test via split traffic -> adopt winner. Use DSPy MIPROv2 optimizer for systematic prompt search. Metric: strategy Sharpe ratio improvement over 30-day window.
2. **meta_threshold_tuner** (monthly): Adjust confidence thresholds by 0.05 increments based on outcome data. Tracks: hypothesis_critique threshold (currently 0.7), backtest Sharpe gate (0.5), IC gate (0.02). Widens if too restrictive (rejecting eventual winners), tightens if too permissive (passing eventual losers).
3. **meta_tool_selector** (weekly): Track tool usage per agent -> inject unused-but-needed tools -> remove always-failing tools. Cross-reference tool_health table with agent outcomes.
4. **meta_architecture_critic** (quarterly): Compare portfolio Sharpe to benchmark -> identify bottleneck node in research/trading pipeline -> propose improvement. If meta_prompt_optimizer suggestions don't improve performance, modify the optimizer's own prompt (recursive).

**Guardrails:**
- Protected file allowlist: NEVER touch risk_gate.py, kill_switch.py, db.py, execution layer
- All prompt changes committed to git with `meta:` prefix for audit trail
- Regression test suite must pass before any change goes live
- Maximum 3 prompt variants tested per week per agent (prevent runaway experimentation)
- Rollback: if 30-day Sharpe declines after prompt change, auto-revert

**Acceptance:**
- All 4 meta agents operational
- Measurable improvement in task agent performance over 30-day windows
- Recursive self-modification demonstrated (optimizer improving its own prompt)
- All changes auditable via git history

### AR-4: Hierarchical Governance

**Impact:** 74% token reduction (~$50K/year savings).

**Context:** Phase 5 graph restructuring substantially complete. AR-4 adds CIO/hierarchy layer on top of existing 5-graph architecture.

**Implementation:**
Three governance tiers:
1. **CIO Agent** (Sonnet, once/day at 09:00 ET):
   - Reviews overnight research results, market regime, portfolio state
   - Produces daily mandate: allowed sectors, max new positions, regime assessment, strategy activation/deactivation
   - Mandate persisted to `daily_mandates` table, consumed by strategy agents
   - Cost: ~$0.15/day

2. **Strategy Agents** (Haiku, every 5 minutes during market hours):
   - Operate within daily mandate constraints
   - Signal collection, entry scanning, position monitoring
   - Cannot override CIO mandate (enforced in code, not prompts)
   - Cost: ~$3.20/day (4 agents * ~$0.80/day)

3. **Risk Officer** (deterministic, per-trade):
   - Same protection as existing risk_gate.py (immutable)
   - No LLM cost — pure code execution
   - Compliance checks: mandate adherence, position limits, concentration

Total: ~$10/day vs. current $150-450/day.

**Mandate schema:**
```python
@dataclass
class DailyMandate:
    date: str
    regime_assessment: str
    allowed_sectors: list[str]
    max_new_positions: int
    strategy_directives: dict[str, str]  # strategy_id -> "active"|"reduce"|"pause"
    risk_overrides: dict  # optional CIO risk adjustments
    reasoning: str  # audit trail
```

**Acceptance:**
- CIO Agent produces daily mandate constraining trading decisions
- Strategy agents operate within mandate using Haiku
- Compliance layer is immutable (same protection as risk_gate.py)
- Token cost reduced 70%+ vs. current architecture
- Mandate adherence enforced in code

---

## AutoResearchClaw Upgrades

4 upgrades to support AR-1, AR-7, AR-8:

| Gap | Current | Required | Fix |
|-----|---------|----------|-----|
| Weekly schedule | Sunday 20:00 only | Nightly 20:00 | Change scheduler cron |
| Reactive only | Bug_fix and drift only | Proactive gap detection + hypothesis | Add gap detection feeds from AR-7 |
| tmux restarts | `tmux send-keys` | Docker Compose restart | Use `docker compose restart` |
| No functional validation | py_compile + import only | Tool invocation test after fix | Add tool test after patch |

---

## Cross-Cutting Concerns

### Prompt Caching
Enable Anthropic prompt caching on all API calls immediately. Cache tool definitions (static across cycles) and system prompts. Expected savings: 70-90% on input tokens.

### Event Bus Extensions
New event types needed:
- `TOOL_ADDED`, `TOOL_DISABLED` (AR-8)
- `EXPERIMENT_COMPLETED` (AR-1)
- `FEATURE_DECAYED`, `FEATURE_REPLACED` (AR-10)
- `MANDATE_ISSUED` (AR-4)
- `META_OPTIMIZATION_APPLIED` (AR-2)
- `CONSENSUS_REQUIRED`, `CONSENSUS_REACHED` (AR-6)

### Database Tables (New)
- `tool_health` — per-tool invocation count, success rate, latency
- `tool_demand_signals` — search queries matching planned tools
- `autoresearch_experiments` — experiment log with hypothesis, IC, cost, duration
- `feature_candidates` — enumerated features with IC, stability, correlation
- `kg_nodes`, `kg_edges` — knowledge graph
- `consensus_log` — 3-agent consensus decisions
- `daily_mandates` — CIO daily directives
- `meta_optimizations` — prompt changes, threshold adjustments, tool selection changes

### Testing Strategy
- Unit tests for each new module following existing patterns (synthetic OHLCV generators)
- Integration tests for event bus new event types
- Regression tests for risk gate (must remain unchanged)
- End-to-end tests for overnight autoresearch loop
- Property-based tests for knowledge graph traversal
- Benchmark tests for feature factory (500+ candidates < 10 min)

---

## Dependencies

- **Hard prerequisite:** Phases 1-7 substantially complete
- **10A depends on Phase 7** (failure mode taxonomy for AR-7)
- **10B depends on 10A** (budget discipline and tool registry cleanup)
- **10C depends on 10B** (research output to populate knowledge graph)
- **10D depends on 10C** (performance data for meta-optimization)
- **AR-4 depends on Phase 5** (existing 5-graph architecture as foundation)

## Validation Plan

1. **10A (week 3):** Active vs. stub tool count (target: 50+ active). Loss -> research -> task pipeline verified.
2. **10B (month 2):** Overnight experiments count (target: 2000+/month). Winner rate 5-10%. Budget adherence ($10/night).
3. **10C (month 3):** Knowledge graph redundancy detection demonstrated. Factor overlap query returns correct results.
4. **10D (month 5):** Task agent win rates before/after meta-optimization. 5-10% improvement expected. Token cost reduced 70%+.
