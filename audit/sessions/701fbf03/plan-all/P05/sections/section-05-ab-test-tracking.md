# Section 05: Ensemble A/B Test Tracking

## Objective

Track which ensemble aggregation method (weighted_avg, weighted_median, trimmed_mean) produces the best forward-return IC, and auto-promote the winner. Currently, ensemble method selection is either fixed (weighted_avg) or hash-based random per symbol (when `FEEDBACK_ENSEMBLE_AB_TEST=true`). This section adds:
1. Recording of each method's output per symbol per day into `ensemble_ab_results`
2. A weekly evaluation job that compares methods by IC and promotes the winner
3. Synthesis reads the active method from `ensemble_config` instead of always defaulting

## Dependencies

- **Section 01** (schema migrations): `ensemble_ab_results` and `ensemble_config` tables must exist

## Files to Modify

1. **`src/quantstack/signal_engine/synthesis.py`** -- record ensemble outputs, read active method
2. **`src/quantstack/learning/ic_attribution.py`** -- add `evaluate_ensemble_ab()` function

## Current State

- Three ensemble methods defined at lines 178-227 of synthesis.py: `_ensemble_weighted_avg`, `_ensemble_weighted_median`, `_ensemble_trimmed_mean`
- `_ENSEMBLE_METHODS` list at line 227
- At lines 581-587, when `ensemble_ab_test_enabled()`, the method is selected by `hash(symbol) % len(_ENSEMBLE_METHODS)` (deterministic per symbol but random across symbols). Otherwise, always `_ensemble_weighted_avg`.
- `ensemble_config` table (Section 01) stores the active method with a single-row constraint.

## Implementation

### Step 1: Record all ensemble method outputs

File: `src/quantstack/signal_engine/synthesis.py`, in `_compute_bias_and_conviction()`.

After the ensemble method selection and score computation (line 587: `score = ensemble_fn(scores, weights)`), add recording logic:

```python
        # --- P05 §5.4: Record ensemble method outputs for A/B evaluation ---
        if ensemble_ab_test_enabled() and symbol:
            try:
                from datetime import date as _date_mod
                method_name = ensemble_fn.__name__.replace("_ensemble_", "")
                # Record the chosen method's output
                with _synth_db() as _ab_conn:
                    _ab_conn.execute(
                        "INSERT INTO ensemble_ab_results "
                        "(symbol, signal_date, method_name, signal_value, recorded_at) "
                        "VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT DO NOTHING",
                        [symbol, _date_mod.today(), method_name, score],
                    )
                    # Also record what the other methods WOULD have produced
                    for alt_fn in _ENSEMBLE_METHODS:
                        if alt_fn is ensemble_fn:
                            continue
                        alt_name = alt_fn.__name__.replace("_ensemble_", "")
                        alt_score = alt_fn(scores, weights)
                        _ab_conn.execute(
                            "INSERT INTO ensemble_ab_results "
                            "(symbol, signal_date, method_name, signal_value, recorded_at) "
                            "VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT DO NOTHING",
                            [symbol, _date_mod.today(), alt_name, alt_score],
                        )
            except Exception:
                pass  # Fire-and-forget -- never blocks synthesis
```

**Important**: We record ALL methods' outputs for every symbol, not just the chosen one. This enables fair comparison -- every method has the same number of observations. The `ON CONFLICT DO NOTHING` prevents duplicates if synthesis runs multiple times per day per symbol.

**Note on import**: `_synth_db` is already imported inline at line 319 as `from quantstack.db import db_conn as _synth_db`. Reuse the same alias in the new block (it's within the same method scope). If placing this code in a different method, import `db_conn` similarly.

### Step 2: Read active method from ensemble_config

File: `src/quantstack/signal_engine/synthesis.py`, replace the method selection block (lines 581-587).

**Before:**
```python
        # --- P05 §5.4: Ensemble method selection ---
        from quantstack.config.feedback_flags import ensemble_ab_test_enabled
        if ensemble_ab_test_enabled() and symbol:
            ensemble_fn = _ENSEMBLE_METHODS[hash(symbol) % len(_ENSEMBLE_METHODS)]
        else:
            ensemble_fn = _ensemble_weighted_avg
```

**After:**
```python
        # --- P05 §5.4: Ensemble method selection ---
        from quantstack.config.feedback_flags import ensemble_ab_test_enabled
        ensemble_fn = _ensemble_weighted_avg  # default
        if ensemble_ab_test_enabled() and symbol:
            # Read promoted method from ensemble_config (if any)
            active_method = _get_active_ensemble_method()
            ensemble_fn = _ENSEMBLE_METHOD_MAP.get(active_method, _ensemble_weighted_avg)
```

Add a module-level method map and cached reader:

```python
_ENSEMBLE_METHOD_MAP: dict[str, callable] = {
    "weighted_avg": _ensemble_weighted_avg,
    "weighted_median": _ensemble_weighted_median,
    "trimmed_mean": _ensemble_trimmed_mean,
}

_active_ensemble_cache: tuple[str, float] = ("weighted_avg", 0.0)


def _get_active_ensemble_method() -> str:
    """Read active ensemble method from DB with 1-hour cache."""
    import time as _time_mod
    global _active_ensemble_cache

    method, loaded_at = _active_ensemble_cache
    if (_time_mod.time() - loaded_at) < 3600:
        return method

    try:
        from quantstack.db import db_conn as _cfg_db
        with _cfg_db() as conn:
            conn.execute("SELECT active_method FROM ensemble_config WHERE id = 1")
            row = conn.fetchone()
            if row:
                method = row["active_method"]
                _active_ensemble_cache = (method, _time_mod.time())
    except Exception:
        pass  # Return cached/default on failure

    return method
```

### Step 3: Add `evaluate_ensemble_ab()`

File: `src/quantstack/learning/ic_attribution.py`, add as a module-level function.

```python
_AB_MIN_DAYS = 30
_AB_SIGNIFICANCE = 0.05


def evaluate_ensemble_ab() -> dict[str, Any]:
    """Weekly evaluation of ensemble A/B test results.

    1. Query ensemble_ab_results joined with OHLCV for 5-day forward returns
    2. Group by method_name, compute IC per method
    3. Paired t-test: each non-default method vs weighted_avg
    4. If p < 0.05 improvement over 30+ days: promote winner

    Returns:
        {"winner": str, "promoted": bool, "method_ics": dict, "pvalues": dict}
    """
    from scipy.stats import ttest_rel as _ttest_rel

    result = {
        "winner": "weighted_avg",
        "promoted": False,
        "method_ics": {},
        "pvalues": {},
    }

    try:
        with db_conn() as conn:
            # Get ensemble results with forward returns from OHLCV
            conn.execute("""
                SELECT
                    ab.method_name,
                    ab.symbol,
                    ab.signal_date,
                    ab.signal_value,
                    (fwd.close - cur.close) / NULLIF(cur.close, 0) AS forward_return_5d
                FROM ensemble_ab_results ab
                JOIN ohlcv cur
                    ON cur.symbol = ab.symbol
                    AND cur.timeframe IN ('1D', '1d', 'daily', 'D1')
                    AND cur.timestamp::date = ab.signal_date
                JOIN ohlcv fwd
                    ON fwd.symbol = ab.symbol
                    AND fwd.timeframe IN ('1D', '1d', 'daily', 'D1')
                    AND fwd.timestamp::date = ab.signal_date + 5
                WHERE ab.signal_date > CURRENT_DATE - INTERVAL '90 days'
                  AND ab.signal_date < CURRENT_DATE - 5
                ORDER BY ab.signal_date, ab.symbol
            """)
            rows = conn.fetchall()
    except Exception as exc:
        logger.warning("[EnsembleAB] Failed to query results: %s", exc)
        return result

    if not rows:
        logger.info("[EnsembleAB] No A/B data with forward returns")
        return result

    # Group by method
    from collections import defaultdict
    method_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        method = row["method_name"]
        sig = row["signal_value"]
        fwd = row["forward_return_5d"]
        if sig is not None and fwd is not None and math.isfinite(sig) and math.isfinite(fwd):
            method_pairs[method].append((sig, fwd))

    # Compute IC per method
    method_ics: dict[str, float] = {}
    for method, pairs in method_pairs.items():
        if len(pairs) < 20:
            continue
        signals = [p[0] for p in pairs]
        returns = [p[1] for p in pairs]
        corr, _ = _spearmanr(signals, returns)
        if math.isfinite(corr):
            method_ics[method] = corr

    result["method_ics"] = method_ics

    if "weighted_avg" not in method_ics:
        logger.info("[EnsembleAB] Insufficient data for baseline method")
        return result

    # Check days of coverage
    unique_dates = set()
    for row in rows:
        unique_dates.add(row["signal_date"])
    if len(unique_dates) < _AB_MIN_DAYS:
        logger.info(
            "[EnsembleAB] Only %d unique dates (need %d)",
            len(unique_dates), _AB_MIN_DAYS,
        )
        return result

    # Paired t-test: compare each method's daily IC against baseline
    baseline_pairs = method_pairs["weighted_avg"]
    baseline_signals = [p[0] for p in baseline_pairs]
    baseline_returns = [p[1] for p in baseline_pairs]

    best_method = "weighted_avg"
    best_ic = method_ics.get("weighted_avg", 0.0)

    for method, ic in method_ics.items():
        if method == "weighted_avg":
            continue
        if ic <= best_ic:
            continue

        # Paired test requires same observations -- use date-aligned pairs
        challenger_pairs = method_pairs[method]
        if len(challenger_pairs) != len(baseline_pairs):
            # Can't do paired test with different N -- skip
            continue

        # Per-observation IC proxy: signal * return (rank-direction product)
        baseline_products = [s * r for s, r in baseline_pairs]
        challenger_products = [s * r for s, r in challenger_pairs]

        try:
            t_stat, pvalue = _ttest_rel(challenger_products, baseline_products)
            result["pvalues"][method] = round(pvalue, 4)

            if pvalue < _AB_SIGNIFICANCE and t_stat > 0:
                best_method = method
                best_ic = ic
        except Exception:
            continue

    result["winner"] = best_method

    # Promote if winner is not the current default
    if best_method != "weighted_avg":
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    UPDATE ensemble_config
                    SET active_method = %s,
                        promoted_at = NOW(),
                        evidence_ic = %s,
                        evidence_pvalue = %s
                    WHERE id = 1
                    """,
                    [best_method, best_ic, result["pvalues"].get(best_method, 1.0)],
                )
            result["promoted"] = True
            logger.info(
                "[EnsembleAB] Promoted method=%s ic=%.4f pvalue=%.4f",
                best_method, best_ic, result["pvalues"].get(best_method, 1.0),
            )
        except Exception as exc:
            logger.warning("[EnsembleAB] Promotion failed: %s", exc)

    return result
```

## Edge Cases

1. **No forward return data**: OHLCV may not have prices 5 days forward for recent signals. The query filters `signal_date < CURRENT_DATE - 5` to avoid this.
2. **Missing OHLCV join**: Symbols with no daily OHLCV data are excluded by the INNER JOIN. This is correct -- no return data means no IC computation.
3. **Unequal observation counts across methods**: Since we record ALL methods for every symbol (Step 1), observation counts should be equal. The `len(challenger_pairs) != len(baseline_pairs)` guard handles any edge cases.
4. **Method name mapping**: The `__name__.replace("_ensemble_", "")` produces `"weighted_avg"`, `"weighted_median"`, `"trimmed_mean"`. The `_ENSEMBLE_METHOD_MAP` uses these same keys.
5. **Cache freshness**: The 1-hour cache in `_get_active_ensemble_method()` means a promotion takes up to 1 hour to take effect. Acceptable for a weekly evaluation cadence.
6. **Multiple promotions in one evaluation**: Only the single best method is promoted. If two challengers both beat the baseline, the one with higher IC wins.
7. **Regression**: If the current promoted method regresses (IC drops below weighted_avg), the next evaluation will revert to weighted_avg since it becomes the best performer.

## Tests

File: `tests/unit/signal_engine/test_ensemble_ab.py`

```python
"""Tests for ensemble A/B test tracking and evaluation."""

def test_all_methods_recorded_on_synthesis(monkeypatch):
    """When AB test enabled, all 3 methods' outputs are recorded."""
    # Mock db_conn, enable FEEDBACK_ENSEMBLE_AB_TEST
    # Run synthesis for one symbol
    # Assert: 3 INSERT calls to ensemble_ab_results (one per method)

def test_method_name_mapping():
    """Method function names map correctly to string keys."""
    assert _ensemble_weighted_avg.__name__ == "_ensemble_weighted_avg"
    assert "weighted_avg" == _ensemble_weighted_avg.__name__.replace("_ensemble_", "")

def test_evaluate_with_clear_winner(monkeypatch):
    """Evaluation promotes a method with significantly better IC."""
    # Mock ensemble_ab_results + OHLCV with data where weighted_median
    # has IC=0.15 vs weighted_avg IC=0.05 (p < 0.05)
    # Assert: result["winner"] == "weighted_median"
    # Assert: result["promoted"] is True
    # Assert: UPDATE executed on ensemble_config

def test_evaluate_no_promotion_when_insufficient_days(monkeypatch):
    """Evaluation does not promote with <30 unique dates."""
    # Mock data with only 20 dates
    # Assert: result["promoted"] is False

def test_evaluate_no_promotion_when_pvalue_high(monkeypatch):
    """Evaluation does not promote when p-value >= 0.05."""
    # Mock data where methods have similar IC (p > 0.3)
    # Assert: result["promoted"] is False

def test_evaluate_handles_no_data(monkeypatch):
    """Evaluation returns defaults when no AB data exists."""
    # Mock empty query result
    # Assert: result == default dict

def test_get_active_ensemble_method_caches(monkeypatch):
    """Active method is cached for 1 hour."""
    # Call twice within 1 hour
    # Assert: only 1 DB query executed

def test_get_active_ensemble_method_returns_default_on_error(monkeypatch):
    """Returns 'weighted_avg' when DB query fails."""
    # Mock db_conn to raise
    # Assert: returns "weighted_avg"

def test_synthesis_uses_promoted_method(monkeypatch):
    """Synthesis uses the method from ensemble_config."""
    # Set ensemble_config.active_method = "trimmed_mean"
    # Enable FEEDBACK_ENSEMBLE_AB_TEST
    # Run synthesis
    # Assert: _ensemble_trimmed_mean was used (check via mock or output)
```
