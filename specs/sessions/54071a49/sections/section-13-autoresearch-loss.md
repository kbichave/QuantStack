# Section 13: Overnight Autoresearch & Error-Driven Iteration

## Overview

This section wires two complementary systems into the graph pipelines:

1. **Autoresearch loop** -- a Research Graph node activated during overnight/weekend mode that generates, backtests, and scores trading hypotheses autonomously. Each experiment has a 5-minute wall-clock budget. Winners (OOS IC > 0.02) are registered as draft strategies for morning validation.

2. **Loss analyzer integration** -- wiring the existing `src/quantstack/learning/loss_analyzer.py` pipeline into the Supervisor Graph as a daily 16:30 ET node. It classifies losing trades by failure mode, aggregates 30-day failure frequencies, and generates prioritized research tasks that feed the autoresearch loop's hypothesis queue.

Together, these create a closed loop: losses inform research priorities, research produces new hypotheses, hypotheses that survive validation become strategies, and those strategies' outcomes feed back into loss analysis.

## Dependencies

- **Section 12 (Multi-Mode Operation):** The autoresearch node is only activated during `OVERNIGHT_WEEKEND` mode. The `ScheduleMode` enum and mode-aware routing in the Research Graph must exist before this section's routing logic can work.
- **Section 16 (IC Tracker):** The IC > 0.02 gate for experiment winners depends on IC computation infrastructure from section 16. The `signal_ic` table and `run_ic_computation()` in supervisor nodes must be operational.

## Current State of the Codebase

Several key components already exist and should NOT be reimplemented:

| Component | Location | Status |
|-----------|----------|--------|
| Overnight runner | `src/quantstack/research/overnight_runner.py` | Fully implemented: budget ceiling ($9.50/night), 5-min timeout, crash recovery via cumulative DB reads, `autoresearch_experiments` table logging, event publishing |
| Morning validator | `src/quantstack/research/morning_validator.py` | Implemented: 3-window patience protocol, draft strategy registration. Backtest engine integration is a stub (`_run_patience_windows` raises `NotImplementedError`) |
| Loss analyzer | `src/quantstack/learning/loss_analyzer.py` | Fully implemented: 5-stage pipeline (collect, classify, aggregate, prioritize, generate research tasks). Uses `failure_taxonomy.py` for rule-based classification with LLM fallback |
| Loss aggregation | `src/quantstack/learning/loss_aggregation.py` | Implemented: 30-day rolling aggregation by failure_mode/strategy/symbol |
| Failure taxonomy | `src/quantstack/learning/failure_taxonomy.py` | Enum with 12 failure modes including `regime_mismatch`, `signal_decay`, `thesis_wrong`, `sizing_error`, `entry_timing`, `theta_burn`, etc. |
| DB tables | `src/quantstack/db.py` | `autoresearch_experiments`, `failure_mode_stats`, `research_queue` tables all exist with migrations |
| Event types | `src/quantstack/coordination/event_bus.py` | `EXPERIMENT_COMPLETED` event type exists |

**What is missing** (and what this section must build):

1. A Research Graph node (`autoresearch_node`) that calls `run_overnight_loop()` with mode-aware gating
2. Wiring that node into the Research Graph's overnight routing
3. A Supervisor Graph node (`loss_analyzer_node`) that calls `run_daily_loss_analysis()` at 16:30 ET
4. Wiring that node into the Supervisor Graph
5. The 70/30 budget split logic (new hypotheses vs. refinement of existing winners)
6. Research queue consumption: the autoresearch loop currently only generates hypotheses via LLM; it does not consume tasks from `research_queue` (where the loss analyzer deposits prioritized tasks)

## Tests

All tests go in the files listed below. Tests are stubs with docstrings describing expected behavior.

```python
# tests/graphs/test_autoresearch.py

import pytest


class TestAutoresearchNode:
    """Tests for the Research Graph autoresearch node."""

    def test_experiment_respects_five_minute_budget(self):
        """An experiment that exceeds 5 minutes wall-clock is terminated.
        
        Mock time.monotonic to simulate passage of 301 seconds.
        Verify the experiment is logged with status='timeout' in the DB.
        The overnight_runner.EXPERIMENT_TIMEOUT_SECONDS constant is 300.
        """

    def test_winner_registered_as_draft_when_ic_above_threshold(self):
        """An experiment with OOS IC > 0.02 gets status='winner'.
        
        Mock _run_backtest to return {'oos_ic': 0.05, 'sharpe': 1.2}.
        Verify score_experiment returns status='winner'.
        Verify the experiment row in autoresearch_experiments has status='winner'.
        """

    def test_experiment_rejected_when_ic_below_threshold(self):
        """An experiment with OOS IC <= 0.02 gets status='tested' (not registered).
        
        Mock _run_backtest to return {'oos_ic': 0.01, 'sharpe': 0.5}.
        Verify score_experiment returns status='tested'.
        """

    def test_budget_split_seventy_thirty(self):
        """70% of experiments are new hypotheses, 30% are refinements.
        
        Over a batch of 10 experiments, verify approximately 7 call
        generate_hypothesis() and 3 call refine_existing_winner().
        The split should be deterministic (round-robin or modular).
        """

    def test_research_queue_consumed_for_hypothesis_generation(self):
        """When research_queue has pending tasks, autoresearch consumes them.
        
        Insert a task into research_queue with source='loss_analyzer'.
        Run the autoresearch loop for one iteration.
        Verify the task is dequeued (status changes from 'pending' to 'in_progress').
        Verify the generated hypothesis incorporates the task's context.
        """

    def test_hung_experiment_terminated_at_budget_boundary(self):
        """An experiment exceeding the timeout is terminated cleanly.
        
        asyncio.wait_for wraps run_single_experiment with 300s timeout.
        Verify TimeoutError is caught and logged, not propagated.
        """

    def test_experiment_results_written_to_db(self):
        """Every experiment (success, failure, timeout) is logged to autoresearch_experiments.
        
        Run 3 experiments with different outcomes.
        Verify 3 rows exist in autoresearch_experiments with correct night_date.
        """

    def test_overnight_loop_stops_at_budget_ceiling(self):
        """Loop stops when cumulative cost >= BUDGET_CEILING_USD ($9.50).
        
        Mock experiments to each cost $5.00.
        Verify loop runs exactly 2 experiments (total $10.00 >= $9.50).
        """

    def test_overnight_loop_stops_outside_operating_window(self):
        """Loop stops when _is_within_operating_window() returns False.
        
        Mock _now_et to return 04:01 ET on second iteration.
        Verify loop runs exactly 1 experiment.
        """


class TestAutoresearchGraphWiring:
    """Tests for Research Graph integration."""

    def test_autoresearch_node_only_runs_in_overnight_mode(self):
        """The autoresearch node is gated by ScheduleMode.OVERNIGHT_WEEKEND.
        
        In MARKET_HOURS mode, the node should be a no-op.
        In OVERNIGHT_WEEKEND mode, the node should call run_overnight_loop().
        """

    def test_autoresearch_node_registered_in_research_graph(self):
        """build_research_graph includes an 'autoresearch' node.
        
        Verify the compiled graph contains a node named 'autoresearch'.
        """
```

```python
# tests/graphs/test_loss_analyzer.py

import pytest
from datetime import date


class TestLossAnalyzerNode:
    """Tests for the Supervisor Graph loss analyzer node."""

    def test_losing_trade_classified_as_regime_mismatch(self):
        """A trade where entry_regime != exit_regime is classified as regime_mismatch.
        
        Insert a closed_trade with entry_regime='trending_up', exit_regime='ranging'.
        Run classify_losses.
        Verify failure_mode == 'regime_mismatch'.
        """

    def test_losing_trade_classified_as_signal_failure(self):
        """A trade where signal strength was below threshold is classified as signal_failure.
        
        The failure_taxonomy.py rules check signal_strength_at_entry.
        Verify the classification matches.
        """

    def test_losing_trade_classified_as_sizing_error(self):
        """A trade with outsized loss relative to position size is classified as sizing_error.
        
        The failure_taxonomy.py rules check realized_pnl_pct magnitude.
        Verify the classification matches.
        """

    def test_thirty_day_failure_frequency_aggregation(self):
        """aggregate_failure_modes correctly groups by failure_mode over 30 days.
        
        Insert 5 classified losses across 3 failure modes.
        Run aggregate_failure_modes.
        Verify failure_mode_stats has 3 rows with correct frequencies.
        """

    def test_research_tasks_generated_for_top_failure_modes(self):
        """generate_research_tasks inserts tasks into research_queue for top 3 modes.
        
        Set up failure_mode_stats with 5 failure modes.
        Run prioritize_failure_modes + generate_research_tasks.
        Verify exactly 3 rows in research_queue with source='loss_analyzer'.
        """

    def test_research_tasks_feed_research_queue(self):
        """Tasks generated by loss analyzer are consumable by autoresearch.
        
        Run the full pipeline: collect -> classify -> aggregate -> prioritize -> generate.
        Verify research_queue has rows with task_type='strategy_hypothesis'
        and topic starting with 'loss_pattern:'.
        """

    def test_loss_analyzer_node_runs_at_scheduled_time(self):
        """The loss_analyzer node in Supervisor Graph fires daily on trading days.
        
        The node should check _is_trading_day_today() before running.
        On weekends, it should be a no-op.
        """

    def test_loss_analyzer_handles_no_losers_gracefully(self):
        """When there are no losing trades, the pipeline returns early with zero counts.
        
        Call run_daily_loss_analysis with no data in closed_trades.
        Verify summary shows losers_found=0, no errors raised.
        """
```

## Implementation Details

### 1. Research Graph: Autoresearch Node

Create `src/quantstack/graphs/research/autoresearch_node.py` with a `make_autoresearch()` factory function that returns an async node function.

The node function must:

- **Check mode gate:** Only run if the current `ScheduleMode` is `OVERNIGHT_WEEKEND`. If not, return immediately with `{"autoresearch_summary": {"skipped": True, "reason": "not_overnight"}}`.
- **Delegate to existing runner:** Call `run_overnight_loop()` from `src/quantstack/research/overnight_runner.py`. This already handles budget ceiling, 5-min per-experiment timeout, crash recovery, and DB logging.
- **Return results** into the Research Graph state as `autoresearch_summary`.

The key new logic this node adds (beyond what `overnight_runner.py` already does) is the **70/30 budget split** and **research queue consumption**:

- Maintain a counter of experiments run. For every 10 experiments, 7 should generate new hypotheses via LLM (current behavior of `generate_hypothesis()`), and 3 should consume tasks from `research_queue` and generate hypotheses informed by the task's context (failure mode, affected strategies, etc.).
- To consume from the queue: query `SELECT * FROM research_queue WHERE status='pending' ORDER BY priority DESC, created_at LIMIT 1`, update status to `in_progress`, and pass the task's `context_json` to a modified hypothesis generation prompt that incorporates the failure mode context.

Signature for the factory:

```python
def make_autoresearch():
    """Create the autoresearch node for overnight Research Graph execution.
    
    Returns an async function: (ResearchState) -> dict
    """
```

### 2. Research Graph Wiring

In `src/quantstack/graphs/research/graph.py`, add the autoresearch node to the overnight routing path:

- Import `make_autoresearch` from `.autoresearch_node`
- Add a node: `graph.add_node("autoresearch", make_autoresearch())`
- Add a conditional edge from the mode-checking entry point (provided by section 12) that routes to `"autoresearch"` when mode is `OVERNIGHT_WEEKEND`, bypassing the normal research pipeline
- Add an edge from `"autoresearch"` to `END`

This means during overnight mode, the Research Graph runs ONLY the autoresearch loop (not the normal hypothesis -> critique -> validation pipeline). During market/extended hours, it runs the existing pipeline.

### 3. Supervisor Graph: Loss Analyzer Node

Create a node factory in `src/quantstack/graphs/supervisor/loss_analyzer.py`:

```python
def make_loss_analyzer_node():
    """Create the loss_analyzer node for daily Supervisor Graph execution.
    
    Runs the 5-stage loss analysis pipeline from learning/loss_analyzer.py
    daily at EOD (after market close). Gated by _is_trading_day_today().
    
    Returns an async function: (SupervisorState) -> dict
    """
```

The node function must:

- Check if today is a trading day (reuse `_is_trading_day_today()` pattern from supervisor/nodes.py)
- Check if loss analysis has already run today (query `loop_heartbeats` for `loss_analyzer` with today's date)
- If due: call `run_daily_loss_analysis()` from `src/quantstack/learning/loss_analyzer.py`
- Record a heartbeat on completion
- Return `{"loss_analysis_summary": summary}` into the Supervisor state

### 4. Supervisor Graph Wiring

In `src/quantstack/graphs/supervisor/nodes.py` (or `graph.py`), wire the loss analyzer:

- The loss analyzer should run as part of the `scheduled_tasks` node OR as its own node in the supervisor graph, executed after the nightly functions block
- The simplest integration is to add it as another scheduled task within `make_scheduled_tasks()`, following the same pattern as attribution, nightly_functions, and regime_detection
- Gate: `_is_trading_day_today() and not _already_ran_today("loss_analyzer")`

### 5. DB Tables (Already Exist)

The following tables already exist in `src/quantstack/db.py` migrations and do NOT need to be created:

- `autoresearch_experiments` -- experiment log with experiment_id, night_date, hypothesis, oos_ic, sharpe, cost_usd, status, etc.
- `failure_mode_stats` -- rolling 30-day failure mode aggregation
- `research_queue` -- task inbox with task_id, task_type, priority, topic, context_json, status, source

No new tables are needed. The existing `loss_classifications` concept is covered by the `failure_mode_stats` table plus the classified loss records in `closed_trades`.

### 6. Research Queue Consumption

The existing `overnight_runner.py` generates hypotheses only via `generate_hypothesis()` (pure LLM generation). The 70/30 split requires a new function that generates hypotheses from research queue tasks:

```python
def generate_hypothesis_from_task(task: dict) -> dict:
    """Generate a hypothesis informed by a research queue task.
    
    The task's context_json contains failure_mode, affected strategies,
    and P&L impact data from the loss analyzer. The LLM prompt incorporates
    this context to generate a hypothesis that specifically addresses the
    identified failure pattern.
    
    Parameters
    ----------
    task : dict
        A row from research_queue with keys: task_id, topic, context_json.
    
    Returns
    -------
    dict with entry_rules, exit_rules, parameters, rationale.
    """
```

This function should live in `src/quantstack/research/overnight_runner.py` alongside the existing `generate_hypothesis()`.

### 7. Integration Flow Summary

The complete data flow after implementation:

```
16:30 ET daily (Supervisor Graph):
  loss_analyzer_node
    -> collect_daily_losers (closed_trades WHERE pnl < 0)
    -> classify_losses (failure_taxonomy rules + LLM fallback)
    -> aggregate_failure_modes (upsert failure_mode_stats)
    -> prioritize_failure_modes (rank by cumulative P&L impact)
    -> generate_research_tasks (insert into research_queue)

20:00-04:00 ET overnight (Research Graph):
  autoresearch_node
    -> check mode == OVERNIGHT_WEEKEND
    -> run_overnight_loop:
       for each experiment slot:
         if slot % 10 < 7:  # 70% new
           hypothesis = generate_hypothesis()  # pure LLM
         else:              # 30% from queue
           task = dequeue from research_queue
           hypothesis = generate_hypothesis_from_task(task)
         run_single_experiment(hypothesis)
         if OOS IC > 0.02: status = 'winner'

04:00 ET morning (can be a separate node or part of extended hours):
  morning_validator (already exists)
    -> fetch winners from autoresearch_experiments
    -> validate each with 3-window patience protocol
    -> register passing experiments as draft strategies
```

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/graphs/research/autoresearch_node.py` | Research Graph node factory wrapping `run_overnight_loop()` with mode gating and 70/30 split |
| `src/quantstack/graphs/supervisor/loss_analyzer.py` | Supervisor Graph node factory wrapping `run_daily_loss_analysis()` with schedule gating |
| `tests/graphs/test_autoresearch.py` | Tests for autoresearch node and graph wiring |
| `tests/graphs/test_loss_analyzer.py` | Tests for loss analyzer node and Supervisor integration |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/graphs/research/graph.py` | Import and add `autoresearch` node; add overnight conditional edge |
| `src/quantstack/graphs/supervisor/nodes.py` | Add loss analyzer as a scheduled task in `make_scheduled_tasks()` (or import and wire `make_loss_analyzer_node`) |
| `src/quantstack/research/overnight_runner.py` | Add `generate_hypothesis_from_task()` function; modify `run_overnight_loop()` to implement 70/30 split with research queue consumption |
| `src/quantstack/graphs/state.py` | Add `autoresearch_summary` and `loss_analysis_summary` fields to `ResearchState` and `SupervisorState` respectively (if not already present) |

## Edge Cases and Failure Modes

- **Empty research queue:** When the 30% refinement slot finds no pending tasks in `research_queue`, fall back to a new hypothesis via `generate_hypothesis()`. Do not block or error.
- **Loss analyzer finds no losers:** This is expected on good days. The pipeline returns early with `losers_found=0`. No research tasks are generated. This is correct behavior.
- **Backtest engine not wired:** `overnight_runner._run_backtest()` currently raises `NotImplementedError`. The autoresearch node should catch this and log it as an error status, not crash the graph. Similarly, `morning_validator._run_patience_windows()` is a stub. These are known tech debt, not bugs introduced by this section.
- **Concurrent overnight runs:** If the overnight runner crashes and restarts mid-night, `get_nightly_budget_state()` reads cumulative cost from DB, ensuring no budget overrun. This crash-recovery mechanism already exists.
- **LLM failure during hypothesis generation:** `run_overnight_loop()` already catches exceptions from `generate_hypothesis()`, logs the error, increments the error counter, sleeps 30 seconds, and continues. No additional error handling needed.
- **Research queue task dequeue race condition:** Use `UPDATE research_queue SET status='in_progress' WHERE task_id = (SELECT task_id FROM research_queue WHERE status='pending' ORDER BY priority DESC, created_at LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING *` to prevent two concurrent consumers from dequeuing the same task.
