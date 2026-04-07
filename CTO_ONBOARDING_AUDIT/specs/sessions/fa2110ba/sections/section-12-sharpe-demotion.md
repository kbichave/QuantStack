# Section 12: Live vs. Backtest Sharpe Demotion

## Problem

Strategies can paper trade (or live trade) indefinitely with poor performance because the current circuit breakers only trigger on absolute drawdown (5%) or consecutive losses (3). A strategy whose backtest Sharpe was 1.5 but live Sharpe is 0.2 can trade forever as long as it avoids 3 losses in a row. There is no mechanism to detect sustained underperformance relative to expectations and automatically reduce exposure.

## Dependencies

- **section-06-eventbus-extension** must be completed first. This section publishes a `STRATEGY_DEMOTED` event, which requires the EventBus to support that event type. The `STRATEGY_DEMOTED` type should be added to the `EventType` enum in `src/quantstack/coordination/event_bus.py` as part of section-06 or as a prerequisite step here.

## Scope

1. A function to compute rolling 21-day live Sharpe ratio from realized daily returns.
2. A daily supervisor batch check (`run_sharpe_demotion_check()`) that compares each active strategy's live Sharpe to its stored backtest Sharpe.
3. Auto-demotion logic: if live Sharpe < 50% of backtest Sharpe for 21+ consecutive trading days, demote the strategy.
4. Demotion actions: set strategy status to `forward_testing`, apply 0.25x sizing via `StrategyBreaker.force_scale()`, publish `STRATEGY_DEMOTED` event, queue a research task.
5. Schema change: add `backtest_sharpe FLOAT` column to the `strategies` table if not present.
6. Kill-switch config flag: `FEEDBACK_SHARPE_DEMOTION` (default `false`).
7. Cold-start behavior: fewer than 21 trading days of live returns skips the check entirely.

## Files to Create or Modify

- **New file:** `src/quantstack/learning/sharpe_demotion.py` -- contains the live Sharpe computation function and the demotion gate logic.
- **Modify:** `src/quantstack/graphs/supervisor/nodes.py` -- add `run_sharpe_demotion_check()` as a daily batch node, scheduled after market close.
- **Modify:** `src/quantstack/db.py` -- add `backtest_sharpe` column to `strategies` table via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS backtest_sharpe FLOAT`.
- **Modify:** `src/quantstack/coordination/event_bus.py` -- add `STRATEGY_DEMOTED` to `EventType` enum (if not already added by section-06).
- **New test file:** `tests/unit/test_sharpe_demotion.py`

## Tests (Write First)

All tests go in `tests/unit/test_sharpe_demotion.py`. Use the project's existing pytest conventions: class-based tests, `mock_settings` fixture for DB mocking, no real DB calls.

### Live Sharpe computation

- **Test: rolling 21-day Sharpe from known returns.** Given a list of 21 daily returns with known mean and standard deviation, verify the computed Sharpe matches the expected value (annualized: `mean / std * sqrt(252)`). Use a deterministic input like `[0.01, -0.005, 0.008, ...]` where the answer can be hand-computed.
- **Test: handles missing return days.** If only 15 of the last 21 calendar days have return data (weekends, holidays), the function should compute Sharpe from the 15 available trading days without error. It should NOT pad missing days with zeros.

### Demotion gate

- **Test: live Sharpe < 50% of backtest for 21 days triggers auto-demote.** Given a strategy with `backtest_sharpe=1.5` and a mock `strategy_outcomes` table returning daily returns that produce a live Sharpe of 0.6 (40% of backtest) for 21 consecutive trading days, verify: (a) strategy status is set to `forward_testing`, (b) `force_scale(strategy_id, 0.25)` is called on StrategyBreaker, (c) a `STRATEGY_DEMOTED` event is published with the correct payload, (d) a research task is queued for degradation investigation.
- **Test: live Sharpe < 50% of backtest for 20 days does NOT trigger demotion.** Same setup but with only 20 consecutive days of underperformance. Verify no demotion action is taken. The 21-day threshold is strict.
- **Test: STRATEGY_DEMOTED event published with correct payload.** Verify the event payload includes `strategy_id`, `live_sharpe`, `backtest_sharpe`, `ratio`, and `consecutive_days`.

### Config flag

- **Test: `FEEDBACK_SHARPE_DEMOTION=false` skips all demotion checks.** When the flag is false, `run_sharpe_demotion_check()` returns immediately without querying any data or publishing any events. Verify no DB queries are made.

### Cold-start

- **Test: fewer than 21 days of live data skips the check.** A strategy with only 15 days of return data should not be evaluated for demotion, regardless of how poor those returns are. Verify no demotion action is taken and the function returns cleanly.

## Implementation Details

### Live Sharpe computation function

In `src/quantstack/learning/sharpe_demotion.py`, implement a function with this signature:

```python
def compute_live_sharpe(strategy_id: str, lookback_days: int = 21) -> float | None:
    """
    Compute annualized Sharpe ratio from realized daily returns.

    Queries strategy_outcomes for the given strategy, groups by date,
    sums daily P&L, and computes Sharpe = (mean_daily_return / std_daily_return) * sqrt(252).

    Returns None if fewer than lookback_days trading days of data exist.
    """
```

Data source: the `strategy_outcomes` table, filtered by `strategy_id`, ordered by `closed_at`. Group returns by trading day (date of `closed_at`). Each day's return is the sum of `pnl_pct` for all trades closed that day. If a day has no closed trades, it is excluded (not zero-filled) -- the Sharpe is computed from days the strategy was active.

Edge case: if `std_daily_return` is zero (all returns identical), return `float('inf')` if mean is positive, `float('-inf')` if negative, or `0.0` if mean is also zero. This prevents division-by-zero without masking information.

### Demotion gate logic

```python
def check_sharpe_demotion(strategy_id: str, backtest_sharpe: float) -> dict | None:
    """
    Compare live Sharpe to backtest Sharpe. Return demotion info if triggered, else None.

    Demotion triggers when live_sharpe < 0.5 * backtest_sharpe for 21+ consecutive
    trading days.
    """
```

The "21 consecutive days" tracking requires state. Two approaches, prefer the simpler one:

**Approach (recommended):** Query the last 42 trading days of daily returns. Compute a rolling 21-day Sharpe for each possible window. If every window in the last 21 days shows live Sharpe < 50% of backtest, the condition is met. This avoids storing a separate counter -- the data itself is the state.

**Why 42 days:** To confirm 21 consecutive days of underperformance, you need at least 21 days of windows, each requiring 21 days of data. The earliest window starts at day -42.

### Supervisor batch node

Add `run_sharpe_demotion_check()` to the supervisor's daily batch schedule (after market close). The function:

1. Checks `FEEDBACK_SHARPE_DEMOTION` env var. If `false`, return immediately.
2. Queries all strategies with status `active` or `paper_trading` that have a non-null `backtest_sharpe`.
3. For each strategy, calls `check_sharpe_demotion()`.
4. If demotion triggered:
   - Updates strategy status to `forward_testing` in the `strategies` table.
   - Calls `strategy_breaker.force_scale(strategy_id, 0.25)` to apply 75% sizing reduction.
   - Publishes `STRATEGY_DEMOTED` event via EventBus with payload: `{"strategy_id": ..., "live_sharpe": ..., "backtest_sharpe": ..., "ratio": ..., "consecutive_days": 21}`.
   - Queues a research task of type `sharpe_degradation` with the strategy details.
5. Logs a summary: how many strategies checked, how many demoted.

### Schema change

In `src/quantstack/db.py`, within the schema initialization section, add:

```sql
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS backtest_sharpe FLOAT;
```

This column should be populated at strategy registration time. Existing strategies without a backtest Sharpe are skipped by the demotion check (the non-null filter in step 2 above handles this).

### EventBus integration

The `STRATEGY_DEMOTED` event type must exist in the `EventType` enum. If section-06 has already added it, no action needed. If not, add it:

```python
STRATEGY_DEMOTED = "strategy_demoted"
```

### Kill-switch behavior

The `FEEDBACK_SHARPE_DEMOTION` environment variable defaults to `false`. When false:
- `run_sharpe_demotion_check()` is a no-op (returns immediately).
- No DB queries, no events, no demotion actions.
- Data continues accumulating in `strategy_outcomes` so the check can be enabled later without a cold-start delay.

When true:
- Full demotion logic runs daily.
- Strategies are actively monitored and demoted if underperforming.

### Cold-start behavior

If a strategy has fewer than 21 trading days of return data in `strategy_outcomes`, `compute_live_sharpe()` returns `None`, and `check_sharpe_demotion()` skips the strategy. No demotion action is taken. This prevents false demotions during the initial paper-trading phase when the Sharpe estimate is noisy.

### Interaction with other sizing adjustments

The 0.25x Sharpe demotion multiplier stacks with other multiplicative adjustments in `risk_sizing`:

```
final_size = kelly_size * breaker_factor * transition_factor * sharpe_demotion_factor
```

Where `sharpe_demotion_factor` is 0.25 for demoted strategies, 1.0 otherwise. The minimum tradeable size floor ($100 or 1 share, defined in section-15-regime-transitions) applies after all multiplicative factors. If the compounded result is below the floor, the trade is skipped entirely.

### Rollback

Set `FEEDBACK_SHARPE_DEMOTION=false`. The check stops running. Already-demoted strategies remain demoted (their status was changed in the DB), so manual intervention is needed to re-promote them if the demotion was incorrect. To fully revert: set the flag to false AND update affected strategies' status back to `active` and call `force_scale(strategy_id, 1.0)` to restore normal sizing.

## Checklist

1. Write all tests in `tests/unit/test_sharpe_demotion.py` (they should fail initially).
2. Add `backtest_sharpe` column to `strategies` table in `db.py`.
3. Add `STRATEGY_DEMOTED` to `EventType` enum if not present.
4. Implement `compute_live_sharpe()` in `src/quantstack/learning/sharpe_demotion.py`.
5. Implement `check_sharpe_demotion()` in the same file.
6. Implement `run_sharpe_demotion_check()` in `src/quantstack/graphs/supervisor/nodes.py`.
7. Wire the supervisor batch node into the daily schedule.
8. Verify all tests pass.
9. Verify the kill-switch flag disables the check completely.
