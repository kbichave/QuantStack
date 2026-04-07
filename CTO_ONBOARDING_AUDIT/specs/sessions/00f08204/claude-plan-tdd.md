# Phase 5: Cost Optimization — TDD Plan

**Testing framework:** pytest (existing)
**Test locations:** `tests/unit/`, `tests/integration/`
**Conventions:** Markers `@slow`, `@integration`, `@requires_api`; fixtures in `conftest.py` and `tests/_fixtures/`
**Run command:** `pytest tests/unit/` (default)

---

## 1A. Item 5.0: Consolidate Dual LLM Config Systems

### Tests to write BEFORE implementation

```python
# tests/unit/test_llm_config_consolidation.py

# Test: all callers of llm_config.py resolve to the same model as the equivalent llm/provider.py tier
# Test: IC tier maps to "light" and returns same model string
# Test: Pod tier maps to "heavy" and returns same model string
# Test: Workshop tier maps to "heavy" and returns same model string
# Test: env var override LLM_MODEL_IC still works after migration (backward compat)
# Test: importing from quantstack.llm_config raises DeprecationWarning after migration
# Test: no file in src/quantstack/ imports from llm_config after migration (grep-based)
```

---

## 2. Item 5.1: Context Compaction at Merge Points

### Tests to write BEFORE implementation

```python
# tests/unit/test_compaction.py

# --- Brief schema validation ---
# Test: ParallelMergeBrief validates with complete fields
# Test: ParallelMergeBrief rejects signal_strength outside 0.0-1.0
# Test: ParallelMergeBrief accepts empty lists (no exits, no entries)
# Test: PreExecutionBrief validates with complete fields
# Test: PreExecutionBrief rejects missing required fields

# --- Compaction node logic ---
# Test: compact_parallel extracts exits from execute_exits state key
# Test: compact_parallel extracts entries from entry_scan state key
# Test: compact_parallel includes earnings_flags when earnings_analysis ran
# Test: compact_parallel produces empty earnings_flags when earnings skipped
# Test: compact_pre_execution extracts approved/rejected from portfolio_review
# Test: compact_pre_execution extracts options_specs from analyze_options
# Test: compact_pre_execution includes risk_checks dict

# --- Fallback behavior ---
# Test: compaction with malformed state produces degraded brief (not crash)
# Test: degraded brief has compaction_degraded flag set
# Test: compaction with empty state produces valid brief with empty lists

# --- Context size reduction ---
# Test: brief serialization is <20% of raw state serialization for typical cycle data
```

---

## 3. Item 5.2: Per-Agent Cost Tracking

### Tests to write BEFORE implementation

```python
# tests/unit/test_agent_executor.py (extend existing)

# --- Budget enforcement ---
# Test: agent with max_tokens_budget=1000 stops after exceeding budget
# Test: agent with max_tokens_budget=None runs without budget limit
# Test: budget-exceeded agent returns partial result with budget_exceeded flag
# Test: budget accumulates across multiple tool-calling rounds
# Test: budget tracks prompt_tokens + completion_tokens from LLM response

# tests/unit/test_cost_queries.py (new)

# --- Langfuse metadata ---
# Test: LLM call metadata includes agent_name, graph_name, cycle_id, llm_tier
# Test: cost_by_agent_day returns aggregated costs grouped by agent and date
# Test: cost_anomaly_detection flags agents exceeding 3x 7-day average
```

---

## 4. Item 5.3: Agent Tier Reclassification

### Tests to write BEFORE implementation

```python
# tests/unit/test_llm_provider.py (extend existing)

# Test: get_chat_model with unrecognized tier logs WARNING (not silent default)
# Test: get_chat_model never returns a model via naming-convention fallback
# Test: every agent in all 3 agents.yaml has explicit llm_tier field

# tests/unit/test_agent_configs.py (extend existing)

# Test: no agent config has llm_tier=None or missing llm_tier
# Test: all tier values are in {"heavy", "medium", "light"}
```

---

## 5. Item 5.4: Per-Agent Temperature Config

### Tests to write BEFORE implementation

```python
# tests/unit/test_agent_executor.py (extend existing)

# Test: agent with temperature=0.7 passes temperature to get_chat_model
# Test: agent with temperature=None uses default (0.0)
# Test: get_chat_model(tier, temperature=0.5) creates LLM with temperature=0.5
# Test: get_chat_model(tier) without temperature kwarg still works (backward compat)

# tests/unit/test_agent_configs.py (extend existing)

# Test: AgentConfig accepts temperature field from YAML
# Test: research agents have temperature > 0.0
# Test: executor/fund_manager have temperature == 0.0
```

---

## 6. Item 5.5: EWF Deduplication

### Tests to write BEFORE implementation

```python
# tests/unit/test_ewf_cache.py (new)

# --- Cache population ---
# Test: populate_ewf_cache stores results for all symbol:timeframe combos
# Test: populate_ewf_cache handles empty symbol list gracefully
# Test: clear_ewf_cache empties the module-level cache

# --- Cache lookup ---
# Test: get_ewf_analysis returns cached result when cache hit
# Test: get_ewf_analysis queries DB when cache miss
# Test: get_ewf_blue_box_setups returns cached result when cache hit
# Test: cache is keyed by symbol:timeframe (different timeframes = different entries)

# --- Cycle lifecycle ---
# Test: cache is empty before populate_ewf_cache called
# Test: cache is empty after clear_ewf_cache called
```

---

## 7. Item 5.6: Remove Hardcoded Model Strings

### Tests to write BEFORE implementation

```python
# tests/unit/test_no_hardcoded_models.py (new)

# Test: grep src/quantstack/ for hardcoded model patterns returns zero hits
#       (excluding llm/config.py and litellm_config.yaml)
#       Patterns: claude-sonnet, claude-haiku, gpt-4o, llama-3,
#                 anthropic/, bedrock/, openai/, groq/, gemini/
# Test: every LLM instantiation goes through get_chat_model() or LiteLLM proxy
```

---

## 8. Item 5.7: LiteLLM Proxy Deployment

### Tests to write BEFORE implementation

```python
# tests/unit/test_llm_provider.py (extend existing)

# Test: get_chat_model with LITELLM_PROXY_URL returns ChatOpenAI pointed at proxy
# Test: get_chat_model without LITELLM_PROXY_URL falls back to direct provider
# Test: tier names map directly to LiteLLM model names (no translation)

# tests/integration/test_litellm.py (new, @integration marker)

# Test: LiteLLM health endpoint responds (requires running Docker service)
# Test: request through LiteLLM returns valid LLM response
# Test: simulated 429 triggers fallback to next provider
# Test: provider cooldown prevents requests to cooled-down provider
# Test: Zscaler cert is mounted and provider connections succeed
```

---

## 9. Item 5.8: Memory Temporal Decay

### Tests to write BEFORE implementation

```python
# tests/unit/test_memory_decay.py (new)

# --- Temporal decay weighting ---
# Test: entry created today has weight ~1.0
# Test: entry created 14 days ago with half_life=14 has weight ~0.5
# Test: entry created 28 days ago with half_life=14 has weight ~0.25
# Test: different categories use different half-lives
# Test: decay weighting changes result ordering (recent != most relevant)

# --- Archival ---
# Test: entries older than 3x half_life are archived
# Test: archived entries moved to agent_memory_archive, not deleted
# Test: archived entries have archived_at timestamp set
# Test: active entries (within TTL) are not archived

# --- last_accessed_at ---
# Test: read_recent updates last_accessed_at for returned rows
# Test: read_as_context updates last_accessed_at for returned rows
# Test: entries not accessed in 60+ days are flagged for archival

# --- SQL query ---
# Test: decay weight computed in SQL matches Python formula
# Test: LIMIT applied after decay weighting (not before)

# --- Markdown memory ---
# Test: compact-memory skill identifies entries past TTL
# Test: expired entries archived to .archive.md file
# Test: permanent entries (workshop_lessons) never archived
```

---

## Testing Order

Match the implementation order (spec order). For each item:
1. Write test stubs first (red)
2. Implement the feature
3. Make tests pass (green)
4. Refactor if needed

Items 5.2 and 5.4 share AgentConfig schema changes — write tests for both before implementing either.
