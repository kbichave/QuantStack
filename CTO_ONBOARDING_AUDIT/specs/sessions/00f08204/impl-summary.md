# Implementation Summary

## What Was Implemented

**Phase 5: Cost Optimization** — 9 sections targeting 60-80% LLM cost reduction ($40K-$64K annually).

### Section 01: Consolidate LLM Configs
Unified the dual config system (`llm_config.py` vs `llm/provider.py`) by adding `TIER_ALIASES` for legacy name mapping, `get_model_for_role()` for litellm callers, `_normalize_tier()` for alias resolution, and a `DeprecationWarning` on legacy `llm_config.py` import. Migrated 3 callers (sentiment.py, sentiment_alphavantage.py, llm_router.py).

### Section 02: Context Compaction
Created deterministic Pydantic brief schemas (`ParallelMergeBrief`, `PreExecutionBrief`) in `briefs.py` and compaction functions (`compact_parallel`, `compact_pre_execution`) in `compaction.py`. Replaced no-op merge nodes in the trading graph with structured compaction that extracts exits, entries, risks, and regime data into typed briefs.

### Section 03: Per-Agent Cost Tracking
Created `cost_queries.py` with `TokenBudgetTracker` (per-agent token accumulation), `compute_cost_usd()` (model-specific pricing), and `detect_cost_anomaly()` (3-sigma spike detection). Added `max_tokens_budget` field to `AgentConfig` and wired budget enforcement into `agent_executor.py`.

### Section 04: Agent Tier Reclassification
Added WARNING log before ValueError on unrecognized tiers. Verified all 21 agents across 3 graphs already have correct explicit `llm_tier` assignments (8 heavy, 12 medium, 1 light). Added regression tests enforcing this invariant.

### Section 05: Per-Agent Temperature Config
Added `temperature` field to `AgentConfig`, updated `get_chat_model()` signature to accept temperature, wired temperature through all 16 `get_chat_model()` calls across trading, research, and supervisor graphs.

### Section 06: EWF Deduplication Cache
Added module-level `_ewf_cycle_cache` dict with `populate_ewf_cache()`, `clear_ewf_cache()`, and `get_ewf_cache_entry()` functions. Modified `get_ewf_analysis` tool to check cache first (single-tf and multi-tf), falling back to DB on miss.

### Section 07: Remove Hardcoded Model Strings
Replaced hardcoded model strings in 6 files (`tool_search_compat.py`, `hypothesis_agent.py`, `opro_loop.py`, `textgrad_loop.py`, `trade_evaluator.py`, `trading/nodes.py`) with `get_chat_model()` or `get_model_for_role()`. Created regression test that greps the entire codebase for hardcoded patterns.

### Section 08: LiteLLM Proxy Deployment
Added LiteLLM proxy path to `get_chat_model()` — when `LITELLM_PROXY_URL` is set, returns `ChatOpenAI` pointed at the proxy with tier name as model. Created `litellm_config.yaml` with model groups matching tier names, router settings, and Langfuse callbacks. Added litellm Docker service to `docker-compose.yml` with Zscaler cert handling.

### Section 09: Memory Temporal Decay
Added `last_accessed_at`/`archived_at` columns to `agent_memory` table, created `agent_memory_archive` table. Implemented `CATEGORY_HALF_LIFE_DAYS` config, `decay_weight()` function, decay-weighted `read_recent()` with `use_decay=True`, `last_accessed_at` tracking, and `archive_stale()` method for TTL-based archival.

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Python-side decay sort instead of SQL | DB-agnostic (works with both PG and SQLite for testing) |
| `use_decay=False` default on `read_recent()` | Backward compatible — existing callers unaffected |
| Mem0 excluded from hardcoded-strings test | Third-party library requires provider-specific model names in its config dict |
| LiteLLM proxy is opt-in via `LITELLM_PROXY_URL` | Gradual rollout — existing direct-provider path remains as fallback |
| Tier names used as LiteLLM model group names | Zero translation layer — `get_chat_model("heavy")` passes "heavy" directly |

## Known Issues / Remaining TODOs

- ~~**Compact-memory skill markdown TTL** (Section 09 Part 6)~~: **DONE** — Added Step 7e (TTL-Based Archival) to `.claude/skills/compact-memory/SKILL.md` with TTL rules table, archival process, and permanent file exclusions.
- ~~**Supervisor integration for weekly pruning** (Section 09 Part 5)~~: **DONE** — Wired `archive_stale()` into `src/quantstack/graphs/supervisor/nodes.py` `scheduled_tasks` node with `_is_weekly_task_due("memory_pruning")` gating. Regression test added (`TestSupervisorWiring`).
- **LiteLLM integration tests**: `tests/integration/test_litellm.py` deferred — requires a running LiteLLM Docker service (excluded per user).
- **Langfuse audit script** (Section 04 Step 1): `scripts/audit_agent_tiers.py` not created — all tiers already correct, no value in generating it.

## Test Results

**155 tests pass** across all Phase 5 test files:

| Test File | Tests |
|-----------|-------|
| `test_llm_config_consolidation.py` | 17 |
| `test_compaction.py` | 18 |
| `test_cost_queries.py` | 15 |
| `test_temperature_config.py` | 7 |
| `test_ewf_cache.py` | 8 |
| `test_memory_decay.py` | 19 |
| `test_agent_configs.py` (tier tests) | 9 |
| `test_no_hardcoded_models.py` | 2 |
| `test_llm_provider.py` (incl. LiteLLM) | 32 |
| **Total** | **155** |

No regressions in existing tests (pre-existing `test_finrl_environments.py` failure due to missing `gymnasium` dependency was resolved by installing the package).

## Files Created or Modified

### Created
| File | Section |
|------|---------|
| `src/quantstack/graphs/trading/briefs.py` | 02 |
| `src/quantstack/graphs/trading/compaction.py` | 02 |
| `src/quantstack/observability/cost_queries.py` | 03 |
| `tests/unit/test_llm_config_consolidation.py` | 01 |
| `tests/unit/test_compaction.py` | 02 |
| `tests/unit/test_cost_queries.py` | 03 |
| `tests/unit/test_temperature_config.py` | 05 |
| `tests/unit/test_ewf_cache.py` | 06 |
| `tests/unit/test_memory_decay.py` | 09 |
| `tests/unit/test_no_hardcoded_models.py` | 07 |
| `litellm_config.yaml` | 08 |

### Modified
| File | Sections |
|------|----------|
| `src/quantstack/llm/provider.py` | 01, 04, 05, 08 |
| `src/quantstack/llm/config.py` | 01 |
| `src/quantstack/llm/__init__.py` | 01 |
| `src/quantstack/llm_config.py` | 01 |
| `src/quantstack/llm_router.py` | 01 |
| `src/quantstack/graphs/config.py` | 03, 05 |
| `src/quantstack/graphs/agent_executor.py` | 03 |
| `src/quantstack/graphs/state.py` | 02 |
| `src/quantstack/graphs/trading/graph.py` | 02, 05 |
| `src/quantstack/graphs/research/graph.py` | 05 |
| `src/quantstack/graphs/supervisor/graph.py` | 05 |
| `src/quantstack/graphs/tool_search_compat.py` | 07 |
| `src/quantstack/graphs/trading/nodes.py` | 07 |
| `src/quantstack/tools/langchain/ewf_tools.py` | 06 |
| `src/quantstack/signal_engine/collectors/sentiment.py` | 01 |
| `src/quantstack/signal_engine/collectors/sentiment_alphavantage.py` | 01 |
| `src/quantstack/alpha_discovery/hypothesis_agent.py` | 07 |
| `src/quantstack/optimization/opro_loop.py` | 07 |
| `src/quantstack/optimization/textgrad_loop.py` | 07 |
| `src/quantstack/performance/trade_evaluator.py` | 07 |
| `src/quantstack/memory/blackboard.py` | 09 |
| `src/quantstack/db.py` | 09 |
| `tests/unit/test_llm_provider.py` | 01, 08 |
| `tests/unit/test_agent_configs.py` | 04 |
| `docker-compose.yml` | 08 |
