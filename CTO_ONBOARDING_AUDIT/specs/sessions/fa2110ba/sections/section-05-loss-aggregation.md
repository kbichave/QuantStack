# Section 05: Loss Aggregation in Supervisor

## Purpose

Losses are recorded in `strategy_outcomes` but never aggregated or analyzed downstream. The nightly supervisor batch runs IC computation and signal scoring, but nothing reads losses to find patterns. This section adds a new supervisor batch node, `run_loss_aggregation()`, that groups classified losses (from the failure taxonomy in Section 04), ranks them by P&L impact, and auto-generates targeted research tasks for the top failure patterns. A new `loss_aggregation` table stores daily snapshots for trend analysis.

## Dependencies

- **Section 04 (Failure Taxonomy):** The `failure_mode` column on `strategy_outcomes` must exist and be populated. Without classified failures, aggregation still works but everything groups under `unclassified`.
- **Existing supervisor batch infrastructure:** The nightly batch loop in `src/quantstack/graphs/supervisor/nodes.py` (around line 1520) already calls `run_signal_scoring()`, `run_ic_computation()`, and `run_ic_retirement_sweep()`. The new node follows the same pattern.

## Database Table

Create a `loss_aggregation` table. Use the existing `CREATE TABLE IF NOT EXISTS` pattern from `src/quantstack/db.py`.

Columns:
- `id SERIAL PRIMARY KEY`
- `date DATE NOT NULL` -- aggregation date
- `failure_mode TEXT NOT NULL` -- matches FailureMode enum values
- `strategy_id TEXT NOT NULL`
- `symbol TEXT` -- nullable; some aggregations may be strategy-level
- `trade_count INTEGER NOT NULL`
- `cumulative_pnl FLOAT NOT NULL` -- total P&L for this group (negative for losses)
- `avg_loss_pct FLOAT NOT NULL` -- average loss percentage per trade in this group
- `rank INTEGER` -- rank by absolute cumulative P&L impact within this date's aggregation
- `created_at TIMESTAMPTZ DEFAULT NOW()`

Add a unique constraint on `(date, failure_mode, strategy_id, symbol)` to prevent duplicate aggregation runs on the same day.

**File to modify:** `src/quantstack/db.py` -- add the `CREATE TABLE IF NOT EXISTS` statement to the schema initialization section.

## Core Function: `run_loss_aggregation()`

**File to create:** `src/quantstack/learning/loss_aggregation.py`

This is a standalone async function, following the same signature pattern as `run_signal_scoring()` and `run_ic_computation()` in the supervisor nodes. It returns a summary dict for logging.

### Algorithm

```python
async def run_loss_aggregation() -> dict[str, Any]:
    """
    Aggregate losses from the trailing 30 days, grouped by failure mode,
    strategy, and symbol. Rank by absolute P&L impact. Auto-generate
    targeted research tasks for the top 3 failure patterns.

    Returns a summary dict with keys: groups_found, tasks_created, top_patterns.
    """
```

Steps:

1. **Query losses:** Read from `strategy_outcomes` where `pnl < 0` and `closed_at >= NOW() - INTERVAL '30 days'`. Join with the `failure_mode` column. If `failure_mode` is NULL, treat as `'unclassified'`.

2. **Group and aggregate:** Group by `(failure_mode, strategy_id, symbol)`. For each group, compute:
   - `trade_count`: number of losing trades
   - `cumulative_pnl`: sum of pnl
   - `avg_loss_pct`: average of `pnl_pct` (or `pnl / entry_value` if `pnl_pct` not stored)

3. **Rank:** Sort all groups by `abs(cumulative_pnl)` descending. Assign rank 1, 2, 3, etc.

4. **Store snapshot:** Insert each group into the `loss_aggregation` table with today's date. Use `ON CONFLICT (date, failure_mode, strategy_id, symbol) DO UPDATE` to handle re-runs.

5. **Auto-generate research tasks (top 3 only):** For the top 3 groups by absolute P&L impact:
   - Create a research task using the existing research queue mechanism (the same pattern used in `trade_hooks.py` for queuing `bug_fix` tasks)
   - Set `task_type` to the failure mode (e.g., `regime_mismatch`, `timing_error`) instead of generic `bug_fix`
   - Set context to include the affected strategy ID, symbol, trade count, and cumulative loss
   - Compute priority as `min(9, int(abs(cumulative_pnl) / 100))` -- higher loss = higher priority, capped at 9

6. **Handle empty case:** If zero losses in 30 days, return `{"groups_found": 0, "tasks_created": 0, "top_patterns": []}` and skip table inserts.

7. **Return summary:** `{"groups_found": N, "tasks_created": M, "top_patterns": [list of top 3 failure mode summaries]}`

All DB operations use `db_conn()` context manager. Wrap the entire function in try/except, logging failures with full context.

## Supervisor Integration

**File to modify:** `src/quantstack/graphs/supervisor/nodes.py`

Add the `run_loss_aggregation()` call in the nightly batch section (after IC computation, around line 1550). Follow the exact same try/except/logging/heartbeat pattern used by the existing nightly functions.

**Schedule:** The plan specifies 16:30 ET. Since the existing nightly batch already runs after market close on trading days (gated by `_is_nightly_functions_due()`), the simplest integration is to add loss aggregation to that same nightly block. It naturally runs after IC computation and retirement sweeps, which is the correct ordering -- IC data is fresh when loss aggregation references it.

If a separate schedule is needed later, extract into its own `_is_loss_aggregation_due()` check. For initial implementation, piggyback on the existing nightly schedule.

Integration sketch:

```python
# In the nightly_due block, after ic_retirement_sweep:
loss_agg_result: dict[str, Any] = {}
try:
    loss_agg_result = await run_loss_aggregation()
    logger.info(
        "[nightly] Loss aggregation: %d groups, %d tasks created",
        loss_agg_result.get("groups_found", 0),
        loss_agg_result.get("tasks_created", 0),
    )
except Exception as loss_agg_exc:
    logger.error("[nightly] Loss aggregation failed: %s", loss_agg_exc)
    loss_agg_result = {"error": str(loss_agg_exc)}

nightly_result["loss_aggregation"] = loss_agg_result
```

## Rollback Path

Disable the `run_loss_aggregation()` call in the supervisor nightly batch (comment out or feature-flag). The `loss_aggregation` table goes stale but causes no harm. No other system depends on this table -- it is a leaf node in the dependency graph.

## Cold-Start Behavior

With zero losses in the trailing 30 days (new system, paper-only period), the function returns early with an empty summary. No research tasks are created. No table writes occur. This is safe and expected.

---

## Tests

**File:** `tests/unit/test_loss_aggregation.py`

All tests use mocked DB connections (via `mock_settings` fixture). No real DB calls.

### Test: groups losses by failure_mode, strategy, symbol over 30 days

Given a set of mock `strategy_outcomes` rows with various failure modes, strategies, and symbols within the last 30 days, verify that `run_loss_aggregation()` produces the correct number of groups with accurate `trade_count`, `cumulative_pnl`, and `avg_loss_pct` per group.

### Test: top 3 patterns ranked by absolute P&L impact

Given 5+ groups with different cumulative losses, verify the returned `top_patterns` list contains exactly 3 entries, ordered by descending absolute P&L impact.

### Test: auto-generates research tasks with failure_mode as type

For each of the top 3 patterns, verify that a research task is queued with `task_type` matching the failure mode string (not `bug_fix`). Verify task context includes strategy_id, symbol, trade_count, and cumulative_pnl.

### Test: aggregation stored in loss_aggregation table

Verify that after `run_loss_aggregation()` completes, the mock DB received INSERT calls to the `loss_aggregation` table with correct date, rank, and aggregate values.

### Test: handles empty losses (no trades in 30 days) gracefully

With no loss rows in `strategy_outcomes`, verify the function returns `{"groups_found": 0, "tasks_created": 0, "top_patterns": []}` and makes no DB writes to `loss_aggregation`.

### Test: UNCLASSIFIED losses still appear in aggregation

Given losses where `failure_mode` is NULL or `'unclassified'`, verify they are grouped under `'unclassified'` and included in ranking and research task generation like any other failure mode.

### Test: re-run on same day upserts (no duplicates)

Run `run_loss_aggregation()` twice on the same mocked date. Verify the second run uses `ON CONFLICT ... DO UPDATE` and does not create duplicate rows in `loss_aggregation`.

### Test: priority computation scales with loss magnitude

Verify that a group with $500 cumulative loss gets priority `min(9, 5) = 5`, while a group with $1200 cumulative loss gets priority `min(9, 12) = 9`.

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/db.py` | Modify | Add `loss_aggregation` table DDL |
| `src/quantstack/learning/loss_aggregation.py` | Create | `run_loss_aggregation()` function |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Wire `run_loss_aggregation()` into nightly batch |
| `tests/unit/test_loss_aggregation.py` | Create | Unit tests |
