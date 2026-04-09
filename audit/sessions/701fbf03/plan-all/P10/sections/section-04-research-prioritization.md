# Section 04: Research Prioritization

## Objective

Replace the FIFO research queue with a priority-scored queue. Each research task gets a composite priority score based on expected alpha uplift, portfolio gap, failure frequency, and staleness. The research graph processes highest-priority tasks first.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/learning/experiment_prioritizer.py` | Modify | Extend with multi-factor priority scoring (alpha uplift, portfolio gap, failure frequency, staleness) |
| `src/quantstack/db.py` | Modify | Add `priority_score` column to `research_tasks` table if not present |
| `src/quantstack/graphs/research/nodes.py` | Modify | Queue consumer sorts by priority instead of FIFO |

## Implementation Details

### 1. Extended Priority Scoring

The existing `experiment_prioritizer.py` uses a formula: `(expected_IC * regime_fit * novelty_score) / estimated_compute_cost`. Extend this with additional factors while preserving the existing API.

**New scoring factors:**

```python
@dataclass
class ResearchPriority:
    task_id: str
    expected_alpha_uplift: float    # 0.0-1.0, domain heuristic
    portfolio_gap: float            # 1.0 if underexplored asset/strategy, 0.0 if saturated
    failure_frequency: float        # normalized count from loss_analyzer
    staleness: float                # time-decay: 1.0 for fresh, decays toward 0
    regime_fit: float               # from existing prioritizer
    novelty_score: float            # from existing prioritizer
    composite_score: float          # weighted sum
```

**`compute_research_priority(task: dict, active_strategies: list[str], loss_stats: dict, current_regime: str) -> ResearchPriority`**

Factor computation:
- **expected_alpha_uplift**: heuristic based on task type. `"new_strategy"` = 0.8, `"strategy_improvement"` = 0.5, `"bug_fix"` = 0.3, `"agent_prompt_investigation"` = 0.4. Override with explicit value if provided.
- **portfolio_gap**: count active strategies per asset class and strategy type. Gap = 1.0 - (count_in_category / max_category_count). If an asset class has 0 strategies, gap = 1.0. If saturated (>= 3), gap = 0.1.
- **failure_frequency**: query `loss_analyzer` for recent losses in the target symbol/strategy. Normalize: `min(1.0, loss_count / 10)`. Higher frequency = higher priority to investigate.
- **staleness**: `exp(-0.05 * days_since_last_research)` where days_since_last_research is computed from `research_tasks` table. Fresh research (< 7 days) gets low staleness priority; stale topics (> 30 days) get high priority.
- **composite_score**: `w1 * alpha_uplift + w2 * portfolio_gap + w3 * failure_frequency + w4 * staleness + w5 * regime_fit + w6 * novelty_score`
  - Default weights: `{alpha_uplift: 0.30, portfolio_gap: 0.25, failure_frequency: 0.15, staleness: 0.10, regime_fit: 0.10, novelty_score: 0.10}`
  - Weights configurable via function parameter

### 2. Database Changes

Add to `research_tasks` table (if column doesn't exist):

```sql
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS priority_score REAL DEFAULT 0.0;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS priority_factors JSONB;
```

Or ensure the column exists in `ensure_tables()` if the table is created there.

### 3. Queue Integration

**`prioritize_research_queue(tasks: list[dict], active_strategies: list[str], loss_stats: dict, current_regime: str) -> list[dict]`**
- Score each task using `compute_research_priority`
- Sort descending by composite_score
- Update `priority_score` and `priority_factors` in the database
- Return the sorted list

### 4. Research Graph Integration

In `src/quantstack/graphs/research/nodes.py`, the queue consumer node currently fetches tasks in insertion order. Change to:

```python
# Before: SELECT * FROM research_tasks WHERE status='pending' ORDER BY created_at ASC
# After:  SELECT * FROM research_tasks WHERE status='pending' ORDER BY priority_score DESC NULLS LAST, created_at ASC
```

This is a minimal, backwards-compatible change. Tasks without priority scores fall back to FIFO.

### 5. Backward Compatibility

- The existing `prioritize_experiments()` function must continue to work unchanged
- New `prioritize_research_queue()` is an additional function, not a replacement
- The existing `compute_priority()` function stays as-is for experiment-level prioritization

## Test Requirements

File: `tests/unit/learning/test_research_prioritization.py`

1. **test_alpha_uplift_by_task_type** -- new_strategy gets 0.8, bug_fix gets 0.3
2. **test_portfolio_gap_empty_category** -- no strategies in an asset class, verify gap=1.0
3. **test_portfolio_gap_saturated** -- 3+ strategies in category, verify gap=0.1
4. **test_failure_frequency_normalization** -- 15 losses -> 1.0, 5 losses -> 0.5
5. **test_staleness_decay** -- 0 days -> low staleness priority, 60 days -> high staleness
6. **test_composite_score_ordering** -- high-alpha + high-gap task ranks above low-alpha + low-gap
7. **test_custom_weights** -- override default weights, verify scoring changes accordingly
8. **test_fifo_fallback** -- tasks without priority_score processed in created_at order
9. **test_prioritize_research_queue_returns_sorted** -- pass 5 tasks, verify returned in descending priority order
10. **test_existing_prioritize_experiments_unchanged** -- call existing function, verify same behavior as before

## Acceptance Criteria

- [ ] `compute_research_priority` correctly computes all 6 factors
- [ ] Composite score uses configurable weights with sensible defaults
- [ ] `prioritize_research_queue` sorts tasks by composite score
- [ ] Research graph queue consumer respects priority ordering
- [ ] Tasks without priority scores fall back to FIFO (backward compatible)
- [ ] Existing `prioritize_experiments()` and `compute_priority()` remain unchanged
- [ ] All 10 unit tests pass
