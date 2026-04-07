<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-db-migrations
section-02-tool-lifecycle
section-03-error-driven-research
section-04-budget-discipline
section-05-event-bus-extensions
section-06-prompt-caching
section-07-overnight-autoresearch
section-08-feature-factory
section-09-weekend-parallel
section-10-knowledge-graph
section-11-consensus-validation
section-12-governance
section-13-meta-agents
section-14-autoresclaw-upgrades
END_MANIFEST -->

# Implementation Sections Index — Phase 10 Advanced Research

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-db-migrations | - | 02, 03, 04, 07, 08, 10, 11, 12, 13 | Yes |
| section-02-tool-lifecycle | 01 | 07, 14 | Yes |
| section-03-error-driven-research | 01 | 07 | Yes |
| section-04-budget-discipline | 01 | 07, 08, 09 | Yes |
| section-05-event-bus-extensions | - | 02, 03, 07, 08, 10, 11, 12, 13 | Yes |
| section-06-prompt-caching | - | 12 | Yes |
| section-07-overnight-autoresearch | 01, 02, 03, 04, 05 | 10 | No |
| section-08-feature-factory | 01, 04, 05 | 10 | Yes |
| section-09-weekend-parallel | 04 | 10 | Yes |
| section-10-knowledge-graph | 01, 05, 07, 08, 09 | 11, 13 | No |
| section-11-consensus-validation | 01, 05, 10 | 13 | Yes |
| section-12-governance | 01, 05, 06 | 13 | No |
| section-13-meta-agents | 10, 11, 12 | - | No |
| section-14-autoresclaw-upgrades | 02 | - | Yes |

## Execution Order (Batched)

1. **Batch 1 (parallel):** section-01-db-migrations, section-05-event-bus-extensions, section-06-prompt-caching
2. **Batch 2 (parallel):** section-02-tool-lifecycle, section-03-error-driven-research, section-04-budget-discipline
3. **Batch 3 (parallel):** section-07-overnight-autoresearch, section-08-feature-factory, section-09-weekend-parallel, section-14-autoresclaw-upgrades
4. **Batch 4:** section-10-knowledge-graph
5. **Batch 5 (parallel):** section-11-consensus-validation, section-12-governance
6. **Batch 6:** section-13-meta-agents

## Section Summaries

### section-01-db-migrations
All 10 new database tables (tool_health, autoresearch_experiments, kg_nodes, kg_edges, etc.), pgvector extension, docker-compose.yml image change. Foundation for all other sections.

### section-02-tool-lifecycle
Split TOOL_REGISTRY into ACTIVE/PLANNED, tool_manifest.yaml, health monitoring middleware, demand signal tracking, tool synthesis pipeline, TOOL_ADDED/TOOL_DISABLED events. Corresponds to plan Section 1 (AR-8).

### section-03-error-driven-research
Daily loss analysis pipeline (16:30 ET): collect losers, classify by failure mode, aggregate 30-day rolling window, prioritize by P&L impact, generate research_queue tasks. Corresponds to plan Section 2 (AR-7).

### section-04-budget-discipline
Budget state fields on ResearchState/TradingState, budget_check conditional edges, synthesize_partial_results node, experiment prioritization formula, 3-window patience protocol. Corresponds to plan Section 3 (AR-9).

### section-05-event-bus-extensions
New EventType enum values (TOOL_ADDED, TOOL_DISABLED, EXPERIMENT_COMPLETED, FEATURE_DECAYED, MANDATE_ISSUED, META_OPTIMIZATION_APPLIED, CONSENSUS_REQUIRED, CONSENSUS_REACHED). Corresponds to plan Section 12.

### section-06-prompt-caching
Enable Anthropic prompt caching in LLM provider layer, cache_control for system prompts and tool definitions. Corresponds to plan Section 14.

### section-07-overnight-autoresearch
overnight_runner.py (20:00-04:00 ET nightly loop), morning_validator.py (04:00 winner validation), DB-persisted budget tracking, crash recovery, scheduler changes. Corresponds to plan Section 4 (AR-1).

### section-08-feature-factory
3-phase pipeline: LLM-assisted enumeration (2000 cap), IC screening, daily drift monitoring with auto-replacement. Corresponds to plan Section 5 (AR-10).

### section-09-weekend-parallel
4 parallel research streams (factor mining, regime research, cross-asset signals, portfolio construction) via LangGraph Send API, Friday-Monday runner. Corresponds to plan Section 6 (AR-5).

### section-10-knowledge-graph
kg_nodes + kg_edges PostgreSQL tables, embedding generation (Bedrock Titan), 4 query tools (check_hypothesis_novelty, check_factor_overlap, get_research_history, record_experiment), population backfill. Corresponds to plan Section 7 (AR-3).

### section-11-consensus-validation
3-agent consensus subgraph (bull, bear, arbiter), deterministic merge, $5K threshold routing, CONSENSUS_ENABLED feature flag, consensus_log. Corresponds to plan Section 8 (AR-6).

### section-12-governance
CIO Agent (Sonnet, daily mandate), 4 strategy agents (Haiku, 5-min cycles), mandate_check hard gate, conservative default mandate fallback, DailyMandate schema. Corresponds to plan Section 10 (AR-4). Must implement BEFORE section-13.

### section-13-meta-agents
4 meta agents (prompt_optimizer via DSPy, threshold_tuner, tool_selector, architecture_critic), threshold extraction to thresholds.yaml, protected file allowlist, auto-revert guardrails. Corresponds to plan Section 9 (AR-2).

### section-14-autoresclaw-upgrades
Nightly schedule, tool_implement + gap_detection task types, Docker Compose restarts, functional validation with test fixtures. Corresponds to plan Section 11.
