# Section 02: Prompt A/B Testing

**depends_on:** section-01-agent-quality-tracking

## Objective

Build a prompt variant management system that shadow-runs alternative prompts alongside production, records both outputs, and promotes variants when they demonstrate statistically significant quality improvement. This does NOT auto-rewrite prompts -- it A/B tests human-curated variants.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/db.py` | Modify | Add `prompt_variants` and `prompt_variant_results` tables to `ensure_tables()` |
| `src/quantstack/learning/prompt_ab.py` | Create | Variant lifecycle: create, activate, shadow-run, evaluate, promote/retire |

## Implementation Details

### 1. Database Schema

Add to `ensure_tables()` in `db.py`:

```sql
CREATE TABLE IF NOT EXISTS prompt_variants (
    id              SERIAL PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    variant_prompt  TEXT NOT NULL,
    description     TEXT,              -- human-readable description of what changed
    status          TEXT NOT NULL DEFAULT 'draft',  -- draft, active, promoted, retired
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ,
    UNIQUE(agent_name, variant_id)
);

CREATE TABLE IF NOT EXISTS prompt_variant_results (
    id              SERIAL PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    cycle_id        TEXT NOT NULL,
    production_output JSONB,
    variant_output    JSONB,
    production_quality_score REAL,
    variant_quality_score    REAL,
    input_context   JSONB,             -- regime, symbols, etc. for reproducibility
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_name, variant_id, cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_pvr_agent_variant
    ON prompt_variant_results (agent_name, variant_id, recorded_at DESC);
```

### 2. Prompt A/B Module (`learning/prompt_ab.py`)

**`create_variant(agent_name, variant_id, variant_prompt, description=None)`**
- Insert into `prompt_variants` with status='draft'
- Validate agent_name exists in known agents (soft check -- log warning if unknown)

**`activate_variant(agent_name, variant_id)`**
- Set status='active'. Only one variant per agent can be active at a time
- If another variant is already active for this agent, retire it first

**`get_active_variant(agent_name) -> dict | None`**
- Return the active variant for an agent, or None if no active variant
- Used by the agent executor to decide whether to shadow-run

**`record_shadow_result(agent_name, variant_id, cycle_id, production_output, variant_output, production_quality, variant_quality, input_context)`**
- Insert into `prompt_variant_results`
- Upsert on unique constraint

**`evaluate_variant(agent_name, variant_id, min_samples=28, significance_level=0.05) -> dict`**
- Query all results for this variant
- If fewer than `min_samples` results: return `{"status": "insufficient_data", "sample_count": N}`
- Compute mean quality scores for production vs. variant
- Run a one-sided Welch's t-test (scipy.stats.ttest_ind with equal_var=False) testing whether variant > production
- Return:
  ```python
  {
      "status": "significant_improvement" | "no_improvement" | "insufficient_data",
      "production_mean": float,
      "variant_mean": float,
      "p_value": float,
      "sample_count": int,
      "recommendation": "promote" | "continue_testing" | "retire",
  }
  ```

**`promote_variant(agent_name, variant_id)`**
- Set status='promoted', set promoted_at
- Log the promotion event
- Does NOT auto-update the production prompt -- returns the variant_prompt text for human/supervisor review

**`retire_variant(agent_name, variant_id)`**
- Set status='retired'

**`get_variant_summary() -> list[dict]`**
- Summary of all variants: agent, variant_id, status, sample_count, mean_quality_delta

### 3. Target Agents for Initial Variants

The plan specifies initial prompt variant targets:
- `trade_debater` -- strongest quality signal (directly impacts trade decisions)
- `daily_planner` -- high-frequency, easy to measure (regime/plan accuracy)
- `fund_manager` -- allocation decisions with clear outcome measurement

These are configuration targets, not hard-coded. Any agent can have variants.

### 4. Shadow Execution Pattern

The integration point (wired in section-06) follows this pattern in the agent executor:

```python
variant = get_active_variant(agent_name)
if variant:
    # Run production prompt (normal path)
    production_output = run_agent(production_prompt, input_context)
    # Shadow-run variant (same input, variant prompt)
    variant_output = run_agent(variant["variant_prompt"], input_context)
    # Score both using agent_quality scoring from section-01
    record_shadow_result(...)
```

Shadow execution must not block or slow the production path. If the variant call fails, log and continue with production output only.

## Test Requirements

File: `tests/unit/learning/test_prompt_ab.py`

1. **test_create_variant** -- create a variant, verify it exists with status='draft'
2. **test_activate_variant** -- activate a draft variant, verify status='active'
3. **test_only_one_active_per_agent** -- activate variant A, then activate variant B, verify A is retired
4. **test_get_active_variant_none** -- no active variant, verify returns None
5. **test_record_shadow_result** -- record a result, verify it persists
6. **test_evaluate_insufficient_data** -- fewer than 28 samples, verify insufficient_data status
7. **test_evaluate_significant_improvement** -- variant scores consistently higher, verify p_value < 0.05 and recommendation=promote
8. **test_evaluate_no_improvement** -- variant scores similar, verify no_improvement status
9. **test_promote_variant** -- promote variant, verify status and promoted_at
10. **test_variant_summary** -- create multiple variants across agents, verify summary returns all

## Acceptance Criteria

- [ ] `prompt_variants` and `prompt_variant_results` tables are created by `ensure_tables()`
- [ ] Full variant lifecycle works: create -> activate -> shadow-run -> evaluate -> promote/retire
- [ ] Only one active variant per agent at a time
- [ ] Statistical evaluation uses Welch's t-test with configurable significance level
- [ ] `evaluate_variant` returns actionable recommendation (promote/continue/retire)
- [ ] All 10 unit tests pass
- [ ] Shadow execution failure does not impact production path (error isolation)
