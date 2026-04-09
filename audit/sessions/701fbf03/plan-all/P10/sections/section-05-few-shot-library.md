# Section 05: Few-Shot Example Library

**depends_on:** section-01-agent-quality-tracking

## Objective

Build a curated library of high-quality agent input/output examples that can be injected into agent prompts at runtime. Examples are auto-extracted from top-scoring agent outputs and filtered by context relevance (regime, strategy type). This improves agent decision quality by providing concrete examples of good reasoning.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/db.py` | Modify | Add `few_shot_examples` table to `ensure_tables()` |
| `src/quantstack/learning/few_shot_library.py` | Create | Example curation, retrieval, and prompt injection |

## Implementation Details

### 1. Database Schema

Add to `ensure_tables()` in `db.py`:

```sql
CREATE TABLE IF NOT EXISTS few_shot_examples (
    id              SERIAL PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    context_type    TEXT NOT NULL,       -- 'regime_trending_up', 'strategy_momentum', etc.
    example_input   JSONB NOT NULL,
    example_output  TEXT NOT NULL,       -- the agent's actual text output
    quality_score   REAL NOT NULL,       -- from agent_quality_scores
    source_cycle_id TEXT,               -- traceability to the originating cycle
    is_gold         BOOLEAN DEFAULT FALSE,  -- manually curated gold-standard
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retired_at      TIMESTAMPTZ,         -- soft-delete for stale examples
    UNIQUE(agent_name, context_type, source_cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_fse_agent_context
    ON few_shot_examples (agent_name, context_type, retired_at);
```

### 2. Few-Shot Library Module (`learning/few_shot_library.py`)

**Auto-Curation:**

**`curate_examples(agent_name: str, top_percentile: float = 0.10, max_per_context: int = 3)`**
- Query `agent_quality_scores` joined with cycle context for the given agent
- Select outputs in the top `top_percentile` by quality_score
- For each context_type, keep only the best `max_per_context` examples
- Insert into `few_shot_examples` (skip duplicates via unique constraint)
- Retire existing non-gold examples that fall below the new quality threshold
- Return count of examples added/retired

**Context Type Derivation:**

```python
def derive_context_type(input_context: dict) -> str:
    """Derive a context type string from agent input context.
    
    Combines regime + strategy_type for targeted example retrieval.
    Examples: 'regime_trending_up', 'strategy_momentum', 
              'regime_trending_up__strategy_momentum'
    """
    parts = []
    if regime := input_context.get("regime"):
        parts.append(f"regime_{regime}")
    if strategy := input_context.get("strategy_type"):
        parts.append(f"strategy_{strategy}")
    return "__".join(parts) if parts else "general"
```

**Manual Curation:**

**`mark_gold_example(example_id: int)`**
- Set `is_gold = True` for the given example
- Gold examples are never auto-retired

**`retire_example(example_id: int)`**
- Set `retired_at = NOW()`
- Prevents the example from being used in future prompts

**Retrieval:**

**`get_examples(agent_name: str, context: dict, max_examples: int = 3) -> list[dict]`**
- Derive context_type from the current context
- Query `few_shot_examples` where `agent_name` matches, `retired_at IS NULL`, and context_type matches (with fallback)
- Retrieval priority:
  1. Exact context_type match, gold examples first
  2. Exact context_type match, highest quality_score
  3. Partial match (regime only or strategy only)
  4. "general" context_type as last resort
- Return up to `max_examples` results as `[{"input": dict, "output": str, "quality": float, "is_gold": bool}]`

**Prompt Injection:**

**`format_examples_for_prompt(examples: list[dict], agent_name: str) -> str`**
- Format retrieved examples into a prompt-injectable string:
  ```
  ## Reference Examples
  
  The following are examples of high-quality {agent_name} outputs in similar conditions:
  
  ### Example 1 (quality: 0.92)
  **Context:** {formatted input}
  **Output:** {example output}
  
  ### Example 2 (quality: 0.88)
  ...
  ```
- Return empty string if no examples available
- Keep formatting concise to minimize token usage

**Maintenance:**

**`cleanup_stale_examples(max_age_days: int = 90)`**
- Retire non-gold examples older than `max_age_days`
- Prevents example staleness as market conditions evolve

**`get_library_stats() -> dict`**
- Return: `{"total_examples": int, "gold_examples": int, "by_agent": {agent: count}, "by_context": {context: count}}`

### 3. Integration Points

- Auto-curation runs as a post-cycle hook (wired in section-06) -- after quality scores are computed
- Example retrieval happens in the agent prompt builder before each agent execution
- The formatted examples are appended to the agent's system prompt or user message

## Test Requirements

File: `tests/unit/learning/test_few_shot_library.py`

1. **test_derive_context_type_full** -- regime + strategy present, verify combined context
2. **test_derive_context_type_regime_only** -- only regime, verify regime-only context
3. **test_derive_context_type_empty** -- no context, verify "general"
4. **test_curate_examples_top_percentile** -- 100 scored outputs, verify only top 10 are curated
5. **test_curate_max_per_context** -- 20 examples for same context, verify only 3 kept
6. **test_gold_examples_not_retired** -- mark example as gold, run curation, verify it persists
7. **test_get_examples_exact_match** -- examples exist for exact context, verify they are returned
8. **test_get_examples_fallback** -- no exact match, verify partial match returned
9. **test_get_examples_general_fallback** -- no match at all, verify "general" examples returned
10. **test_format_examples_empty** -- no examples, verify empty string
11. **test_format_examples_content** -- 2 examples, verify formatted string contains both
12. **test_cleanup_stale** -- examples older than 90 days, verify they are retired
13. **test_library_stats** -- mixed examples, verify stats are accurate

## Acceptance Criteria

- [ ] `few_shot_examples` table is created by `ensure_tables()`
- [ ] Auto-curation selects from top percentile of quality-scored outputs
- [ ] Max 3 examples per agent per context type (configurable)
- [ ] Gold examples are protected from auto-retirement
- [ ] Retrieval uses tiered fallback: exact match -> partial match -> general
- [ ] Formatted prompt injection is concise and includes quality scores
- [ ] Stale example cleanup prevents drift
- [ ] All 13 unit tests pass
