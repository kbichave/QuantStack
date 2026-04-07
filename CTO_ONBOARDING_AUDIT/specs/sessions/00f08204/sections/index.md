<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-consolidate-llm-configs
section-02-context-compaction
section-03-cost-tracking
section-04-tier-reclassification
section-05-temperature-config
section-06-ewf-dedup
section-07-hardcoded-strings
section-08-litellm-proxy
section-09-memory-decay
END_MANIFEST -->

# Phase 5 Cost Optimization — Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-consolidate-llm-configs | - | 04, 07, 08 | Yes |
| section-02-context-compaction | - | - | Yes |
| section-03-cost-tracking | - | 04 | Yes |
| section-04-tier-reclassification | 01, 03 | - | No |
| section-05-temperature-config | - | - | Yes |
| section-06-ewf-dedup | - | - | Yes |
| section-07-hardcoded-strings | 01 | 08 | No |
| section-08-litellm-proxy | 01, 07 | - | No |
| section-09-memory-decay | - | - | Yes |

## Execution Order

1. **Batch 1** (no dependencies): section-01-consolidate-llm-configs, section-02-context-compaction, section-03-cost-tracking, section-05-temperature-config, section-06-ewf-dedup, section-09-memory-decay
2. **Batch 2** (after 01, 03): section-04-tier-reclassification, section-07-hardcoded-strings
3. **Batch 3** (after 01, 07): section-08-litellm-proxy

## Section Summaries

### section-01-consolidate-llm-configs
**Plan ref: 1A (Item 5.0)** — Audit and consolidate `llm_config.py` (IC/Pod/Assistant/Decoder/Workshop) into `llm/provider.py` (heavy/medium/light). Migrate all callers, add backward-compatible env var aliases, deprecate and remove legacy file.

### section-02-context-compaction
**Plan ref: 2 (Item 5.1)** — Replace no-op merge nodes in Trading graph with deterministic compaction nodes. Define Pydantic brief schemas (ParallelMergeBrief, PreExecutionBrief). Update downstream agent prompts to consume briefs. Measure 40%+ context reduction.

### section-03-cost-tracking
**Plan ref: 3 (Item 5.2)** — Enrich Langfuse metadata with agent_name/graph_name/cycle_id. Build aggregation queries for per-agent cost rollup. Add max_tokens_budget to AgentConfig. Implement graceful budget enforcement in agent_executor.py. Add cost alerting to Supervisor health monitor.

### section-04-tier-reclassification
**Plan ref: 4 (Item 5.3)** — Audit actual model usage via Langfuse. Remove naming-convention fallback in get_chat_model(). Ensure every agent has explicit llm_tier. Verify and correct tier assignments. Measure cost impact.

### section-05-temperature-config
**Plan ref: 5 (Item 5.4)** — Add temperature field to AgentConfig. Change get_chat_model() signature to accept temperature parameter (backward compatible). Configure per-agent temperatures in all agents.yaml files.

### section-06-ewf-dedup
**Plan ref: 6 (Item 5.5)** — Add module-level EWF cache to ewf_tools.py. Populate cache in data_refresh node. Modify get_ewf_analysis and get_ewf_blue_box_setups to check cache first. Clear cache at cycle end. Apply same pattern to Research graph.

### section-07-hardcoded-strings
**Plan ref: 7 (Item 5.6)** — Grep for all hardcoded model strings. Replace with get_chat_model() calls. Add bulk tier for OPRO/TextGrad. Handle special cases (mem0_client, hypothesis_agent). Add pytest regression test.

### section-08-litellm-proxy
**Plan ref: 8 (Item 5.7)** — Add LiteLLM service to docker-compose.yml with Zscaler cert. Create litellm_config.yaml with provider priority chains and circuit breaker. Refactor get_chat_model() to thin wrapper. Gradual rollout (Supervisor → Trading → Research). Integration tests.

### section-09-memory-decay
**Plan ref: 9 (Item 5.8)** — Add last_accessed_at and archived_at columns to agent_memory. Create archive table. Implement SQL-based temporal decay weighting. Weekly pruning job in Supervisor graph. Enhance compact-memory skill with TTL enforcement for .claude/memory/ files.
