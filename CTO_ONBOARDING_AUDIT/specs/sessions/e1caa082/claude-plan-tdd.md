# TDD Plan: Phase 10 — Advanced Research

Testing framework: **pytest** with existing fixtures in `tests/unit/conftest.py`. Synthetic OHLCV generators, mock_settings, patch_get_settings. Test file convention: `tests/unit/test_<module>.py`. Integration tests in `tests/integration/`.

---

## Section 1: Tool Lifecycle (AR-8)

### Unit Tests (`tests/unit/test_tool_lifecycle.py`)

```python
# Test: ACTIVE_TOOLS contains only tools with status="active" in manifest
# Test: PLANNED_TOOLS contains only tools with status="planned" in manifest
# Test: bind_tools_to_llm resolves only from ACTIVE_TOOLS (never PLANNED_TOOLS)
# Test: search_deferred_tools excludes PLANNED_TOOLS from results
# Test: tool_health_check auto-disables tool with success_rate < 50% over 7 days
# Test: tool_health_check does NOT disable tool with success_rate = 51%
# Test: tool_health_check moves disabled tool to DEGRADED_TOOLS
# Test: health tracking middleware increments counters on success
# Test: health tracking middleware increments failure_count and records last_error on exception
# Test: demand_signal_tracker logs search query when planned tool matches
# Test: demand_signal_weekly_aggregation ranks by frequency correctly
# Test: tool_implement task type generates valid tool from planned definition
# Test: TOOL_ADDED event published when tool moves from PLANNED to ACTIVE
# Test: TOOL_DISABLED event published when tool auto-disabled
```

### Integration Tests (`tests/integration/test_tool_lifecycle.py`)

```python
# Test: agent receives updated tool set after TOOL_ADDED event (poll + rebind cycle)
# Test: full lifecycle: planned tool → demand signal → synthesis → active → agent sees it
```

---

## Section 2: Error-Driven Research (AR-7)

### Unit Tests (`tests/unit/test_loss_analyzer.py`)

```python
# Test: collect_daily_losers returns only negative P&L fills for today
# Test: classify_loss maps regime_mismatch correctly (entry_regime != exit_regime)
# Test: classify_loss maps liquidity_trap correctly (slippage > threshold)
# Test: classify_loss maps model_degradation correctly (PSI > 0.25 at entry time)
# Test: classify_loss returns "unknown" for unclassifiable losses
# Test: aggregate_failure_modes maintains rolling 30-day window (drops day 31)
# Test: aggregate_failure_modes ranks by cumulative P&L impact (not frequency)
# Test: prioritize selects top 3 failure modes by P&L impact
# Test: generate_research_tasks creates research_queue entries with correct context
# Test: pipeline runs all 5 stages in sequence
# Test: pipeline handles zero losers gracefully (no tasks generated)
# Test: pipeline handles single loser correctly
```

### Unit Tests (`tests/unit/test_failure_modes.py`)

```python
# Test: each failure mode in taxonomy has at least one deterministic rule
# Test: deterministic rules are mutually exclusive (no loss matches 2+ rules)
# Test: Haiku classification called only when no deterministic rule matches
# Test: Haiku classification returns valid failure mode from taxonomy
```

---

## Section 3: Experiment Budget Discipline (AR-9)

### Unit Tests (`tests/unit/test_budget_discipline.py`)

```python
# Test: ResearchState initializes with correct default budget (50K tokens, $0.50)
# Test: TradingState initializes with correct default budget (30K tokens, $0.20)
# Test: budget_check returns "continue" when remaining > estimated_next_cost
# Test: budget_check returns "synthesize" when remaining < estimated_next_cost
# Test: budget_check returns "synthesize" when remaining == 0
# Test: synthesize_partial_results node produces summary from partial state
# Test: budget fields compatible with extra="forbid" (no ValidationError)
```

### Unit Tests (`tests/unit/test_experiment_prioritizer.py`)

```python
# Test: prioritization formula ranks high-IC cheap experiments above low-IC expensive ones
# Test: novelty_score = 1.0 for novel hypotheses (not in KG)
# Test: novelty_score = 0.1 for known hypotheses (similar in KG)
# Test: cold-start behavior: all novel hypotheses ranked by regime_fit / compute_cost
# Test: prioritizer handles empty experiment queue gracefully
```

### Unit Tests (`tests/unit/test_patience_protocol.py`)

```python
# Test: hypothesis rejected only when ALL 3 windows fail gates
# Test: hypothesis marked "provisional" when 2/3 windows pass
# Test: hypothesis fully accepted when 3/3 windows pass
# Test: 3 windows are: full historical, recent 12 months, stressed period
# Test: stressed period is configurable (default: 2020-03 to 2020-06)
```

---

## Section 4: Overnight Autoresearch (AR-1)

### Unit Tests (`tests/unit/test_overnight_runner.py`)

```python
# Test: runner starts at 20:00 ET and stops at 04:00 ET
# Test: runner halts when cumulative cost reaches $9.50 (leaves headroom)
# Test: budget tracking persists to DB (survives simulated crash)
# Test: runner resumes from last cumulative cost after restart
# Test: experiment with OOS IC > 0.02 marked as "winner"
# Test: experiment with OOS IC <= 0.02 marked as "tested" (not winner)
# Test: experiment exceeding 5-minute timeout is killed and logged
# Test: experiments run back-to-back (no artificial sleep between)
# Test: experiment_id is unique (no duplicates on restart)
```

### Unit Tests (`tests/unit/test_morning_validator.py`)

```python
# Test: morning validator runs at 04:00 unconditionally
# Test: morning validator processes all winners from overnight
# Test: morning validator uses 3-window patience protocol
# Test: passing winner registered as status='draft' in strategies
# Test: failing winner logged with rejection reason
# Test: morning validator handles zero winners gracefully
```

---

## Section 5: Autonomous Feature Factory (AR-10)

### Unit Tests (`tests/unit/test_feature_factory.py`)

```python
# Test: programmatic enumeration from 10 base features produces >100 candidates
# Test: enumeration respects 2000 candidate hard cap
# Test: LLM-assisted enumeration adds novel features not in programmatic set
# Test: LLM failure falls back to programmatic-only (no crash)
# Test: IC screening filters candidates below IC 0.01
# Test: IC screening filters candidates below stability 0.5
# Test: correlation check drops features with >0.95 Pearson to already-selected
# Test: screening output is 50-100 features from 500+ input
# Test: daily monitoring detects PSI > 0.25 as CRITICAL decay
# Test: daily monitoring detects IC < 0.005 for 10 days as decay
# Test: auto-replacement selects next-best feature from screening pool
# Test: FEATURE_DECAYED and FEATURE_REPLACED events published
```

---

## Section 6: Weekend Parallel Research (AR-5)

### Unit Tests (`tests/unit/test_weekend_runner.py`)

```python
# Test: weekend runner spawns exactly 4 parallel streams
# Test: each stream has isolated state (no cross-contamination)
# Test: results merge via reducer into weekend_research_results list
# Test: synthesis node produces prioritized research tasks from 4 stream results
# Test: runner operates from Friday 20:00 to Monday 04:00
# Test: individual stream failure doesn't crash other streams
```

### Unit Tests (per stream)

```python
# Test: factor_mining stream extracts testable factor from mock paper
# Test: regime_research stream labels regimes from historical data
# Test: cross_asset_signals stream computes lead-lag correlation
# Test: portfolio_construction stream compares risk parity vs. equal weight
```

---

## Section 7: Alpha Knowledge Graph (AR-3)

### Unit Tests (`tests/unit/test_knowledge_graph.py`)

```python
# Test: create_node inserts node with correct type and properties
# Test: create_edge inserts edge between existing nodes
# Test: create_edge fails gracefully with non-existent node IDs
# Test: check_hypothesis_novelty returns "redundant" for >0.85 cosine similarity in same regime
# Test: check_hypothesis_novelty returns "novel" for <0.85 similarity
# Test: check_hypothesis_novelty returns "novel" for same hypothesis in different regime
# Test: check_factor_overlap returns "crowded" when >2 shared factors with existing positions
# Test: check_factor_overlap returns "clear" when <=2 shared factors
# Test: get_research_history returns matching hypotheses by semantic search + regime filter
# Test: record_experiment creates hypothesis node, result node, factor nodes, and all edges
# Test: temporal edges (valid_from/valid_to) filter correctly on date queries
# Test: embedding generation calls Bedrock Titan (not OpenAI)
# Test: embedding fallback uses local sentence-transformers when Bedrock unavailable
```

### Integration Tests (`tests/integration/test_knowledge_graph.py`)

```python
# Test: population backfill from strategies table creates correct graph structure
# Test: population backfill from ml_experiments creates result nodes with tested_by edges
# Test: factor overlap query with recursive CTE returns correct results for 3-hop traversal
# Test: full lifecycle: record experiment → check novelty → detect redundancy on re-test
```

### Performance Tests (`tests/benchmarks/test_knowledge_graph_perf.py`)

```python
# Test: factor overlap query < 100ms with 10K nodes and 50K edges
# Test: novelty detection < 50ms with 10K hypothesis nodes
```

---

## Section 8: Consensus-Based Signal Validation (AR-6)

### Unit Tests (`tests/unit/test_consensus.py`)

```python
# Test: trades > $5K route to consensus subgraph
# Test: trades <= $5K bypass consensus (go directly to risk gate)
# Test: $5K threshold configurable via CONSENSUS_THRESHOLD env var
# Test: consensus_merge with 3/3 ENTER returns full position size (1.0)
# Test: consensus_merge with 2/3 ENTER returns half position size (0.5)
# Test: consensus_merge with 1/3 ENTER returns reject (0.0)
# Test: consensus_merge with 0/3 ENTER returns reject (0.0)
# Test: consensus decision logged to consensus_log table
# Test: CONSENSUS_ENABLED=false bypasses consensus entirely
# Test: bull, bear, arbiter agents have independent state (no shared context)
# Test: arbiter vote is binary (ENTER or REJECT), not a score
```

---

## Section 9: Metacognitive Self-Modification (AR-2)

### Unit Tests (`tests/unit/test_meta_agents.py`)

```python
# Test: meta_prompt_optimizer generates <= 3 prompt variants per week per agent
# Test: A/B split applies new prompt to 50% of cycles
# Test: auto-revert triggers when 30-day Sharpe declines >10% after prompt change
# Test: protected file allowlist blocks modification of risk_gate.py
# Test: protected file allowlist blocks modification of kill_switch.py
# Test: protected file allowlist blocks modification of db.py
# Test: all meta changes committed with "meta:" prefix
# Test: regression test failure triggers auto-revert
```

### Unit Tests (`tests/unit/test_threshold_tuner.py`)

```python
# Test: threshold lowered by 0.05 when false rejection rate > 20%
# Test: threshold raised by 0.05 when false acceptance rate > 30%
# Test: threshold never drops below floor (e.g., hypothesis_critique >= 0.4)
# Test: threshold never exceeds ceiling (e.g., hypothesis_critique <= 0.9)
# Test: threshold unchanged when both rates within acceptable range
# Test: thresholds read from thresholds.yaml, not hardcoded
```

### Unit Tests (`tests/unit/test_tool_selector.py`)

```python
# Test: never-used tools removed from agent binding
# Test: frequently-searched deferred tools promoted to always-loaded
# Test: tool changes written to agents.yaml
```

### Unit Tests (`tests/unit/test_architecture_critic.py`)

```python
# Test: bottleneck identified as node with highest token_cost / alpha ratio
# Test: recursive modification: critic modifies optimizer prompt when optimizer underperforms
# Test: quarterly cadence enforced (no run within 90 days of last)
```

---

## Section 10: Hierarchical Governance (AR-4)

### Unit Tests (`tests/unit/test_governance.py`)

```python
# Test: CIO agent produces valid DailyMandate with all required fields
# Test: mandate_check rejects trade in blocked sector
# Test: mandate_check rejects trade when max_new_positions reached
# Test: mandate_check rejects trade when max_daily_notional exceeded
# Test: mandate_check approves trade within all mandate constraints
# Test: mandate_check is a hard gate (not advisory)
# Test: strategy agents use Haiku tier (not Sonnet)
# Test: CIO agent uses Sonnet tier
# Test: conservative default mandate activates when no mandate by 09:30
# Test: default mandate has max_new_positions=0
# Test: default mandate does not liquidate existing positions
# Test: kill switch overrides mandate (kill active + mandate says "trade" = no trade)
# Test: mandate_check runs before risk gate (not after)
```

### Integration Tests (`tests/integration/test_governance.py`)

```python
# Test: full flow: CIO produces mandate → strategy agent reads → mandate_check enforces
# Test: mandate published via MANDATE_ISSUED event
# Test: token cost with governance < 50% of token cost without (over simulated day)
```

---

## Section 11: AutoResearchClaw Upgrades

### Unit Tests (`tests/unit/test_autoresclaw_upgrades.py`)

```python
# Test: tool_implement task type accepted and processed
# Test: gap_detection task type accepted and processed
# Test: functional validation invokes tool test fixture (not just py_compile)
# Test: functional validation reverts patch on test fixture failure
# Test: Docker Compose restart called instead of tmux send-keys
# Test: nightly schedule triggers (not just Sunday)
```

---

## Section 12: Event Bus Extensions

### Unit Tests (`tests/unit/test_event_bus_extensions.py`)

```python
# Test: each new EventType value is a valid enum member
# Test: TOOL_ADDED event publishes and polls correctly
# Test: TOOL_DISABLED event publishes and polls correctly
# Test: EXPERIMENT_COMPLETED event publishes and polls correctly
# Test: MANDATE_ISSUED event publishes and polls correctly
# Test: META_OPTIMIZATION_APPLIED event publishes and polls correctly
# Test: CONSENSUS_REQUIRED and CONSENSUS_REACHED events publish and poll correctly
# Test: FEATURE_DECAYED and FEATURE_REPLACED events publish and poll correctly
```

---

## Section 13: Database Migrations

### Unit Tests (`tests/unit/test_db_migrations.py`)

```python
# Test: ensure_schema creates all 10 new tables (idempotent — run twice, no error)
# Test: kg_nodes table has vector(1536) column
# Test: kg_nodes table has HNSW index on embedding column
# Test: pgvector extension created (CREATE EXTENSION IF NOT EXISTS vector)
# Test: all tables have UUID primary keys
# Test: all tables have TIMESTAMPTZ timestamps
# Test: ON CONFLICT DO UPDATE works for each table (idempotent writes)
```

---

## Section 14: Prompt Caching

### Unit Tests (`tests/unit/test_prompt_caching.py`)

```python
# Test: cache_control parameter added to Bedrock API calls
# Test: system prompt includes cache_control ephemeral marker
# Test: tool definitions include cache_control ephemeral marker
# Test: cache invalidation occurs when tool definitions change (version bump)
# Test: caching disabled when PROMPT_CACHING_ENABLED=false
```

---

## Regression Tests (Cross-Cutting)

```python
# Test: risk gate behavior unchanged (run all existing risk_gate tests)
# Test: kill switch behavior unchanged (run all existing kill_switch tests)
# Test: kill switch overrides mandate (mandate active + kill switch = no trade)
# Test: existing research graph works with default budget values (backward compatible)
# Test: existing tool bindings unaffected by registry split (ACTIVE_TOOLS is superset check)
```
