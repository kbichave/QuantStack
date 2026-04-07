# Section 03: Error-Driven Research (AR-7)

## Overview

When a trade loses money, the system currently records the loss in `fills` and adjusts regime affinity weights via `OutcomeTracker` (`src/quantstack/learning/outcome_tracker.py`). But there is no systematic analysis of *why* the loss happened and no connection to research priorities. The same failure mode can repeat indefinitely because nothing tells the research graph to investigate it.

This section builds a **daily loss analysis pipeline** that runs at 16:30 ET (after market close, before overnight research). It collects losing trades, classifies them by failure mode, aggregates patterns over a rolling 30-day window, and generates prioritized research tasks so the research graph can address the root causes.

**Key insight:** The codebase already has partial implementations of several pieces. `src/quantstack/learning/failure_taxonomy.py` defines a `FailureMode` enum and a `classify_failure()` function. `src/quantstack/learning/loss_aggregation.py` contains `run_loss_aggregation()` which groups losses, ranks by P&L impact, and creates research queue tasks. This section extends and integrates these into a complete 5-stage pipeline, adds the `failure_mode_stats` table, wires the trigger into the scheduled refresh, and closes the loop with verification tracking.

## Dependencies

- **section-01-db-migrations**: The `failure_mode_stats` table must exist before this pipeline can aggregate data. The `research_queue` table already exists (created in `db.py::_migrate_research_queue_pg`).
- **section-05-event-bus-extensions**: Not strictly required, but the pipeline should publish events for downstream consumers once event bus extensions are in place.

## Existing Code Inventory

Before writing new code, understand what already exists:

| File | What it does | What's missing |
|------|-------------|----------------|
| `src/quantstack/learning/failure_taxonomy.py` | `FailureMode` enum (7 modes), `classify_failure()` with rule-based heuristics, `compute_research_priority()` | Missing: `liquidity_trap`, `signal_decay`, `adverse_selection`, `correlation_breakdown` modes from the Phase 7 taxonomy. Missing: Haiku fallback for ambiguous cases. Missing: `model_degradation` mode. |
| `src/quantstack/learning/loss_aggregation.py` | `run_loss_aggregation()` — groups losses by (failure_mode, strategy_id, symbol), ranks by cumulative P&L, inserts top 3 into `research_queue` | Missing: rolling 30-day window tracking in `failure_mode_stats`. Missing: per-mode frequency/avg-loss tracking. Currently reads from `strategy_outcomes` table which may not include all fill-level data. |
| `src/quantstack/learning/outcome_tracker.py` | Records entry/exit with regime, adjusts regime_affinity weights | No connection to loss analysis pipeline. |
| `src/quantstack/data/scheduled_refresh.py` | `run_eod_refresh()` runs at 16:30 ET | No loss analysis trigger. |

## Tests First

All tests go in `tests/unit/test_loss_analyzer.py` and `tests/unit/test_failure_modes.py`.

### `tests/unit/test_loss_analyzer.py`

```python
"""Tests for the daily loss analysis pipeline."""

import pytest
from datetime import date, datetime, timezone

# Test: collect_daily_losers returns only negative P&L fills for today
def test_collect_daily_losers_filters_negative_pnl():
    """collect_daily_losers should return only fills with realized_pnl < 0,
    closed on the given date. Positive P&L fills are excluded."""
    ...

# Test: collect_daily_losers includes strategy_id, entry/exit regime, holding period, signal strength
def test_collect_daily_losers_returns_required_metadata():
    """Each loser record must include: strategy_id, entry_regime, exit_regime,
    holding_period_days, signal_strength_at_entry, symbol, realized_pnl."""
    ...

# Test: classify_loss maps regime_mismatch correctly (entry_regime != exit_regime)
def test_classify_regime_mismatch():
    """When entry_regime != exit_regime, classification should be REGIME_MISMATCH."""
    ...

# Test: classify_loss maps liquidity_trap correctly (slippage > threshold)
def test_classify_liquidity_trap():
    """When slippage exceeds 2% of expected fill price, classify as LIQUIDITY_TRAP."""
    ...

# Test: classify_loss maps model_degradation correctly (PSI > 0.25 at entry time)
def test_classify_model_degradation():
    """When the strategy's PSI at entry time exceeded 0.25, classify as MODEL_DEGRADATION."""
    ...

# Test: classify_loss returns UNCLASSIFIED for unclassifiable losses
def test_classify_unclassified_fallback():
    """When no deterministic rule matches and Haiku is unavailable,
    return UNCLASSIFIED."""
    ...

# Test: aggregate_failure_modes maintains rolling 30-day window (drops day 31)
def test_aggregate_rolling_window_drops_old_data():
    """failure_mode_stats should only contain data from the last 30 days.
    Data from day 31 should be pruned."""
    ...

# Test: aggregate_failure_modes ranks by cumulative P&L impact (not frequency)
def test_aggregate_ranks_by_pnl_not_frequency():
    """A mode with 2 losses totaling -$5000 should rank above a mode
    with 10 losses totaling -$500."""
    ...

# Test: prioritize selects top 3 failure modes by P&L impact
def test_prioritize_selects_top_3():
    """Only the 3 highest-impact failure modes should generate research tasks."""
    ...

# Test: generate_research_tasks creates research_queue entries with correct context
def test_generate_research_tasks_context():
    """Each generated task should include: failure_mode, affected strategies,
    example losses, and a suggested research direction string."""
    ...

# Test: pipeline runs all 5 stages in sequence
def test_pipeline_runs_all_stages():
    """run_daily_loss_analysis() should call collect, classify, aggregate,
    prioritize, and generate in order."""
    ...

# Test: pipeline handles zero losers gracefully (no tasks generated)
def test_pipeline_zero_losers():
    """When there are no losing trades today, pipeline completes without error
    and creates zero research tasks."""
    ...

# Test: pipeline handles single loser correctly
def test_pipeline_single_loser():
    """A single losing trade should still produce a valid classification
    and potentially a research task."""
    ...
```

### `tests/unit/test_failure_modes.py`

```python
"""Tests for failure mode taxonomy and classification rules."""

import pytest

# Test: each failure mode in taxonomy has at least one deterministic rule
def test_all_modes_have_deterministic_rule():
    """Every FailureMode enum value (except UNCLASSIFIED) should be reachable
    via at least one deterministic rule without requiring LLM classification."""
    ...

# Test: deterministic rules have a priority order (first match wins)
def test_classification_priority_order():
    """When a loss matches multiple rules (e.g., regime_mismatch AND liquidity_trap),
    the higher-priority rule wins. Priority: regime_mismatch > data_stale >
    black_swan > timing_error."""
    ...

# Test: Haiku classification called only when no deterministic rule matches
def test_haiku_called_only_for_ambiguous():
    """The LLM classifier should only be invoked when classify_failure()
    returns UNCLASSIFIED."""
    ...

# Test: Haiku classification returns valid failure mode from taxonomy
def test_haiku_returns_valid_mode():
    """When Haiku is called, it must return a value from the FailureMode enum.
    Invalid responses should fall back to UNCLASSIFIED."""
    ...
```

## Implementation Details

### Stage 1: COLLECT — `collect_daily_losers()`

Query `fills` and `positions` tables for today's closed losers (negative realized P&L). Join with strategy metadata to include:

- `strategy_id`
- `symbol`
- `entry_regime` (from `strategy_outcomes` or `positions` metadata)
- `exit_regime` (current regime at close)
- `holding_period_days`
- `signal_strength_at_entry` (from the signal that triggered the trade)
- `realized_pnl` and `realized_pnl_pct`
- `slippage_pct` (difference between expected and actual fill price)

This is a pure DB query function. No LLM involvement.

### Stage 2: CLASSIFY — `classify_losses()`

For each collected loser, run through the existing `classify_failure()` in `failure_taxonomy.py`. The existing function already handles: `REGIME_MISMATCH`, `DATA_STALE`, `BLACK_SWAN`, `TIMING_ERROR`, and `UNCLASSIFIED`.

**Extend `failure_taxonomy.py`** with these additional modes and rules:

| Mode | Deterministic Rule |
|------|-------------------|
| `LIQUIDITY_TRAP` | `slippage_pct > 0.02` (2% slippage) |
| `MODEL_DEGRADATION` | Strategy's PSI > 0.25 at entry time (query drift_detector results) |
| `SIGNAL_DECAY` | Signal IC for the triggering factor dropped below 0.005 in the 10 days prior to entry |
| `ADVERSE_SELECTION` | Loss occurred within 30 minutes of entry (immediate adverse move) |
| `CORRELATION_BREAKDOWN` | Cross-asset correlation that the strategy depends on diverged > 2 sigma from trailing mean |
| `FACTOR_CROWDING` | Already exists in the enum but has no rule — add: when the factor's IC decayed and multiple strategies share it |

For losses that remain `UNCLASSIFIED` after all deterministic rules, call Haiku with a structured prompt containing the loss metadata and ask for classification. Cost is negligible (~$0.001/call). If Haiku is unavailable, keep `UNCLASSIFIED`.

**Key rule for Haiku integration:** The Haiku call uses `get_chat_model("light")` from `src/quantstack/llm/provider.py`. The prompt should include the full FailureMode enum with descriptions, the loss metadata, and a strict instruction to return only a valid enum value.

### Stage 3: AGGREGATE — `aggregate_failure_modes()`

This extends the existing `run_loss_aggregation()` logic. The existing function already groups by `(failure_mode, strategy_id, symbol)` and ranks by cumulative P&L. The extension:

1. Write to the `failure_mode_stats` table (created in section-01-db-migrations) instead of only `loss_aggregation`.
2. Maintain a **rolling 30-day window**: on each run, delete rows older than 30 days from `failure_mode_stats`, then upsert today's aggregates.
3. Track per-mode: `frequency` (count of losses), `cumulative_pnl_impact` (sum of P&L), `avg_loss_size`, `affected_strategies` (JSON array of strategy_ids).

The `failure_mode_stats` table schema (defined in section-01-db-migrations):

```sql
CREATE TABLE IF NOT EXISTS failure_mode_stats (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    date            DATE NOT NULL,
    failure_mode    TEXT NOT NULL,
    frequency       INTEGER NOT NULL DEFAULT 0,
    cumulative_pnl  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    avg_loss_size   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    affected_strategies JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, failure_mode)
);
```

### Stage 4: PRIORITIZE — `prioritize_failure_modes()`

Query `failure_mode_stats` for the trailing 30-day window. Rank failure modes by **cumulative P&L impact** (absolute value), not frequency. A few large losses outweigh many small ones. Return the top 3.

This is purely deterministic: a SQL query with `ORDER BY ABS(cumulative_pnl) DESC LIMIT 3`, aggregated across dates within the window.

### Stage 5: GENERATE — `generate_research_tasks()`

For each of the top 3 failure modes, create a `research_queue` entry. The `research_queue` table already exists with columns: `task_id`, `task_type`, `priority`, `context_json`, `source`, `status`, `topic`.

Each task includes:
- `task_type`: the failure mode string (e.g., `"regime_mismatch"`)
- `priority`: computed from cumulative P&L impact using `compute_research_priority()` from `failure_taxonomy.py`
- `context_json`: JSON with failure mode, affected strategy IDs, example losses (top 3 by size), and a suggested research direction
- `source`: `"loss_analysis"`
- `topic`: the failure mode name for deduplication

**Suggested research direction examples:**
- `regime_mismatch` on AAPL swing: "Research regime transition detection improvements — AAPL swing strategy entered trending_up but regime shifted to ranging within holding period"
- `signal_decay` on momentum factor: "Investigate IC decay in momentum factor — IC dropped below 0.005 over 10 consecutive days, affecting 3 active strategies"

The research graph's `context_load` node already polls `research_queue` for pending tasks. These error-driven tasks will be picked up in the next research cycle alongside community intel and hypothesis tasks.

### Closed Loop Verification

When a research task generated from loss analysis produces a new strategy or hedge, tag the `research_queue` entry with the resulting `strategy_id` (update the `context_json` to include `"resulting_strategy_id"`). Track whether that strategy subsequently prevents the same failure mode. This is a monitoring metric stored in `failure_mode_stats` (add a `mitigated_count` column), not an enforcement gate.

### Orchestrator: `run_daily_loss_analysis()`

A single async entry point that runs the 5 stages in sequence:

```python
async def run_daily_loss_analysis() -> LossAnalysisReport:
    """Run the full 5-stage loss analysis pipeline.

    Called at 16:30 ET by scheduled_refresh after EOD data sync.

    Returns:
        LossAnalysisReport with stage results and any errors.
    """
    ...
```

The return type `LossAnalysisReport` is a dataclass summarizing: losers collected, classifications made, modes aggregated, tasks generated, and any errors.

### Wiring: Trigger from `scheduled_refresh.py`

Modify `src/quantstack/data/scheduled_refresh.py` to call the loss analysis pipeline after `run_eod_refresh()` completes. The EOD refresh already runs at 16:30 ET. Add a call to `run_daily_loss_analysis()` at the end of the EOD refresh flow (or as a separate step triggered by the supervisor graph's scheduled tasks node immediately after EOD refresh).

The modification is a single function call addition, not a restructuring. The loss analysis pipeline is independent of data refresh — it reads from `fills`/`positions`/`strategy_outcomes` which are populated by the trading graph throughout the day.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/learning/loss_analyzer.py` | **CREATE** | New 5-stage pipeline: `collect_daily_losers()`, `classify_losses()`, `aggregate_failure_modes()`, `prioritize_failure_modes()`, `generate_research_tasks()`, `run_daily_loss_analysis()` |
| `src/quantstack/learning/failure_taxonomy.py` | **MODIFY** | Add missing failure modes to `FailureMode` enum: `LIQUIDITY_TRAP`, `MODEL_DEGRADATION`, `SIGNAL_DECAY`, `ADVERSE_SELECTION`, `CORRELATION_BREAKDOWN`. Add deterministic rules for each. Add Haiku fallback for `UNCLASSIFIED`. |
| `src/quantstack/data/scheduled_refresh.py` | **MODIFY** | Add 16:30 ET loss analysis trigger after EOD refresh |
| `tests/unit/test_loss_analyzer.py` | **CREATE** | Unit tests for the 5-stage pipeline |
| `tests/unit/test_failure_modes.py` | **CREATE** | Unit tests for failure mode classification rules |

## Key Design Decisions

1. **Haiku for ambiguous classification vs. purely deterministic**: Some losses don't fit clean categories (e.g., a liquidity trap that also had signal decay). Haiku can reason about the combination. Cost: ~$0.001/classification, negligible. If Haiku is unavailable, the loss stays `UNCLASSIFIED` and is still counted in aggregation.

2. **30-day rolling window**: Long enough to capture patterns, short enough to adapt to regime changes. The window is enforced by a `DELETE` + `UPSERT` pattern in `aggregate_failure_modes()`, not by querying with a date filter each time. This keeps the table small.

3. **Top 3 per day**: Keeps research queue manageable. The overnight autoresearch (section-07) can only handle a limited number of experiments per night, so flooding the queue with 10+ failure-driven tasks doesn't help. The top 3 by P&L impact ensures the most costly problems get research attention first.

4. **Extend existing modules vs. rewrite**: The existing `failure_taxonomy.py` and `loss_aggregation.py` are solid foundations. The loss analyzer orchestrates them rather than replacing them. The `loss_aggregation.py` module continues to work independently for its current consumers; the new `loss_analyzer.py` calls into it for the aggregation step.

5. **Priority by P&L impact, not frequency**: A single catastrophic loss from a correlation breakdown is more important to research than 20 small timing errors. The prioritization formula in `compute_research_priority()` already accounts for cumulative loss and recency.
