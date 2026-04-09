# Section 07: Scheduler Jobs

## Objective

Register three new scheduled tasks in `scripts/scheduler.py` to run the P05 batch computations at appropriate intervals:
1. **Weekly IC weight precompute** -- Sunday 2:00 AM ET
2. **Quarterly conviction calibration** -- 1st of each quarter, 3:00 AM ET
3. **Weekly ensemble A/B evaluation** -- Sunday 2:30 AM ET

## Dependencies

- **Section 02**: `compute_and_store_ic_weights()` must be implemented
- **Section 04**: `calibrate_conviction_factors()` must be implemented
- **Section 05**: `evaluate_ensemble_ab()` must be implemented

## File to Modify

**`scripts/scheduler.py`**

## Current State

The scheduler has 30+ jobs defined in the `JOBS` list (line 991). Each job is a dict with `trigger` (cron params), `func` (callable), and `label` (string ID). Job functions follow a consistent pattern: log timestamp, check dry_run, execute in try/except, log result.

The `--run-now` CLI supports manually triggering any registered job.

Existing Sunday schedule for reference:
- 02:00 Sun -- `langfuse_retention_cleanup`
- 17:00 Sun -- `memory_compaction`
- 17:30 Sun -- `correlation_analysis`
- 18:00 Sun -- `strategy_lifecycle_weekly`
- 19:00 Sun -- `community_intel_weekly`
- 20:00 daily -- `autoresclaw_nightly`

The new jobs at 02:00 and 02:30 fit into the early-morning maintenance window.

## Implementation

### Job 1: `run_weekly_ic_weight_precompute()`

Add this function after the existing `run_weekly_correlation_analysis()` function (line 988):

```python
def run_weekly_ic_weight_precompute(dry_run: bool = False) -> None:
    """Precompute IC-driven synthesis weights per regime (Sunday 2:00 AM ET).

    Reads 63 days of ic_attribution_data, computes per-collector Spearman IC
    conditioned on regime, applies gates and penalties, and upserts into
    precomputed_ic_weights. Results are consumed by synthesis.py when
    FEEDBACK_IC_DRIVEN_WEIGHTS=true.
    """
    label = "weekly_ic_weight_precompute"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would precompute IC weights at {timestamp}")
        return

    try:
        from quantstack.learning.ic_attribution import compute_and_store_ic_weights
        results = compute_and_store_ic_weights()
        regimes_computed = len(results)
        total_collectors = sum(len(v) for v in results.values())
        logger.info(
            f"'{label}' completed: {regimes_computed} regimes, "
            f"{total_collectors} collector weights computed"
        )
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")
```

### Job 2: `run_quarterly_conviction_calibration()`

```python
def run_quarterly_conviction_calibration(dry_run: bool = False) -> None:
    """Calibrate conviction multiplicative factors (1st of quarter, 3:00 AM ET).

    Correlates conviction factor values at signal time with realized trade P&L.
    Stores optimized parameters in conviction_calibration_params for use by
    the synthesis conviction scaling logic.
    """
    label = "quarterly_conviction_calibration"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would calibrate conviction factors at {timestamp}")
        return

    try:
        from quantstack.learning.ic_attribution import calibrate_conviction_factors
        results = calibrate_conviction_factors()
        factors_calibrated = len(results)
        logger.info(
            f"'{label}' completed: {factors_calibrated} factors calibrated"
        )
        if not results:
            logger.info(f"'{label}' note: no factors had sufficient data for calibration")
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")
```

### Job 3: `run_weekly_ensemble_ab_evaluate()`

```python
def run_weekly_ensemble_ab_evaluate(dry_run: bool = False) -> None:
    """Evaluate ensemble A/B test and promote winner (Sunday 2:30 AM ET).

    Compares IC across ensemble aggregation methods (weighted_avg, weighted_median,
    trimmed_mean). If a challenger method beats the baseline with p < 0.05 over
    30+ days, it becomes the active method in ensemble_config.
    """
    label = "weekly_ensemble_ab_evaluate"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"[{timestamp}] Triggering {label}")

    if dry_run:
        print(f"\n[DRY RUN] Would evaluate ensemble A/B at {timestamp}")
        return

    try:
        from quantstack.learning.ic_attribution import evaluate_ensemble_ab
        result = evaluate_ensemble_ab()
        logger.info(
            f"'{label}' completed: winner={result.get('winner', 'unknown')} "
            f"promoted={result.get('promoted', False)} "
            f"ics={result.get('method_ics', {})}"
        )
    except Exception as exc:
        logger.error(f"'{label}' failed: {exc}")
```

### Register jobs in the JOBS list

Add these entries to the `JOBS` list. Place them in the "Weekly" section (after the `langfuse_retention_cleanup` entry at line 1037, before the `listing_status_check` entry):

```python
    # P05: Weekly IC weight precompute — batch compute regime-conditioned synthesis weights.
    {"trigger": {"hour": 2, "minute": 0, "day_of_week": "sun"}, "func": run_weekly_ic_weight_precompute, "label": "ic_weight_precompute_sun02:00"},
    # P05: Weekly ensemble A/B evaluation — compare aggregation methods, promote winner.
    {"trigger": {"hour": 2, "minute": 30, "day_of_week": "sun"}, "func": run_weekly_ensemble_ab_evaluate, "label": "ensemble_ab_evaluate_sun02:30"},
```

Add the quarterly job in the "Monthly" section (after the `strategy_lifecycle_monthly` entry at line 1074):

```python
    # P05: Quarterly conviction calibration — correlate factor values with realized P&L.
    {"trigger": {"hour": 3, "minute": 0, "day": "1", "month": "1,4,7,10"}, "func": run_quarterly_conviction_calibration, "label": "conviction_calibration_quarterly"},
```

**Note on quarterly trigger**: APScheduler's CronTrigger supports `month="1,4,7,10"` combined with `day="1"` to run on the 1st of January, April, July, October.

### Register in `--run-now` CLI

Add to the `func_map` dict in the `main()` function (around line 1213):

```python
            "weekly_ic_weight_precompute": run_weekly_ic_weight_precompute,
            "quarterly_conviction_calibration": run_quarterly_conviction_calibration,
            "weekly_ensemble_ab_evaluate": run_weekly_ensemble_ab_evaluate,
```

### Update the startup banner

Add these lines to the `start_scheduler()` banner (around line 1149):

```
        f"  02:00 Sun          -- IC weight precompute (regime-conditioned synthesis weights)\n"
        f"  02:30 Sun          -- Ensemble A/B evaluation (promote winning method)\n"
        f"  03:00 1st/quarter  -- Conviction factor calibration (empirical factor tuning)\n"
```

## Schedule Conflict Check

The 02:00 Sunday slot is shared with `langfuse_retention_cleanup`. No conflict:
- `langfuse_retention_cleanup` is a stub (logs only, ~0s runtime)
- `run_weekly_ic_weight_precompute` runs at 02:00 (expected runtime: 5-30s)
- `run_weekly_ensemble_ab_evaluate` runs at 02:30 (expected runtime: 5-60s)
- Both complete well before the 17:00 memory compaction window

## Edge Cases

1. **Import failure**: If `quantstack.learning.ic_attribution` is broken, the try/except logs the error and moves on. Other scheduler jobs are unaffected.
2. **DB not available**: Each job's inner try/except catches DB connection failures. The job logs an error and retries next week.
3. **Quarter detection**: APScheduler's `month="1,4,7,10"` with `day="1"` is tested. If the 1st falls on a weekend, the job still runs (APScheduler is not weekday-aware unless `day_of_week` is specified).
4. **Misfire grace time**: All jobs in the JOBS list use `misfire_grace_time=300` (5 minutes). If the scheduler is restarted and a job was missed by <5 minutes, it runs immediately.
5. **Dry run**: All three jobs respect the `dry_run` parameter, consistent with existing jobs.

## Tests

The scheduler job functions are thin wrappers around the batch functions tested in Sections 02, 04, and 05. Test the wrappers with:

```python
def test_run_weekly_ic_weight_precompute_dry_run(capsys):
    """Dry run prints message without executing."""
    run_weekly_ic_weight_precompute(dry_run=True)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out

def test_run_quarterly_conviction_calibration_dry_run(capsys):
    run_quarterly_conviction_calibration(dry_run=True)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out

def test_run_weekly_ensemble_ab_evaluate_dry_run(capsys):
    run_weekly_ensemble_ab_evaluate(dry_run=True)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out

def test_jobs_list_includes_p05_entries():
    """P05 jobs are registered in the JOBS list."""
    labels = [j["label"] for j in JOBS]
    assert "ic_weight_precompute_sun02:00" in labels
    assert "ensemble_ab_evaluate_sun02:30" in labels
    assert "conviction_calibration_quarterly" in labels
```
