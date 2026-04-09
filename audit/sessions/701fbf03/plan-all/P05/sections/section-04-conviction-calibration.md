# Section 04: Conviction Factor Calibration

## Objective

Enable empirical calibration of the 6 conviction multiplicative factors (adx, stability, timeframe, regime_agreement, ml_confirmation, data_quality). Currently these factors use hardcoded thresholds and scaling constants. This section:
1. Enriches signal metadata with the conviction factor breakdown
2. Adds a quarterly batch calibration job that correlates factor values with realized P&L
3. Provides a cache layer so the hot synthesis path can read calibrated params without DB queries per call

## Dependencies

- **Section 01** (schema migrations): `conviction_calibration_params` table must exist
- The existing `conviction_calibration` table (raw observations) already exists from the current P05 migration

## Files to Modify

1. **`src/quantstack/signal_engine/synthesis.py`** -- enrich signals metadata with conviction_factors
2. **`src/quantstack/learning/ic_attribution.py`** -- add `calibrate_conviction_factors()` function
3. **`src/quantstack/signal_engine/synthesis.py`** -- read calibrated params in `_conviction_multiplicative()`

## Current State

- `_conviction_multiplicative()` (lines 648-730 of synthesis.py) computes 6 multiplicative factors with hardcoded thresholds (e.g., ADX threshold 15, ramp to 1.15 at ADX 50).
- The signals INSERT (lines 317-336) stores `metadata = {"votes": vote_scores, "weights": final_weights}`. The conviction factor breakdown is computed but only logged and returned in the SymbolBrief -- it is not persisted.
- The `conviction_calibration` table stores raw (factor_name, factor_value, forward_return, regime) observations. The new `conviction_calibration_params` table (Section 01) stores calibrated parameters.

## Implementation

### Step 1: Enrich signals metadata with conviction_factors

File: `src/quantstack/signal_engine/synthesis.py`, in the `synthesize()` method.

Change the metadata dict in the signals INSERT (around line 332):

**Before:**
```python
                    _json_synth.dumps({"votes": vote_scores, "weights": final_weights}),
```

**After:**
```python
                    _json_synth.dumps({
                        "votes": vote_scores,
                        "weights": final_weights,
                        "conviction_factors": conviction_factor_breakdown,
                    }),
```

The `conviction_factor_breakdown` variable is already computed by `_compute_bias_and_conviction()` and returned as the 7th element of the tuple (line 303). It is available at the INSERT site.

### Step 2: Add `calibrate_conviction_factors()`

File: `src/quantstack/learning/ic_attribution.py`, add as a module-level function.

```python
_CALIBRATION_FACTORS = (
    "adx", "stability", "timeframe", "regime_agreement",
    "ml_confirmation", "data_quality",
)
_MIN_CALIBRATION_TRADES = 100
_MIN_CORRELATION = 0.01


def calibrate_conviction_factors() -> dict[str, dict[str, float]]:
    """Quarterly calibration of conviction multiplicative factors.

    Joins signals.metadata conviction_factors with closed_trades to correlate
    each factor's value at signal time with the realized P&L.

    For each factor:
      1. Extract (factor_value, realized_pnl_pct) pairs
      2. Compute Pearson correlation
      3. If significant (>100 trades, |corr| > 0.01): derive optimized scaling
      4. Store in conviction_calibration_params table

    Returns:
        {factor_name: {param_name: param_value}} for logging.
    """
    results: dict[str, dict[str, float]] = {}

    try:
        with db_conn() as conn:
            # Join signals with closed_trades: match on symbol and date proximity
            conn.execute("""
                SELECT
                    s.metadata,
                    ct.realized_pnl_pct
                FROM signals s
                JOIN closed_trades ct
                    ON ct.symbol = s.symbol
                    AND ct.entry_date BETWEEN s.signal_date - 1 AND s.signal_date + 1
                WHERE s.metadata IS NOT NULL
                  AND s.metadata::text LIKE '%%conviction_factors%%'
                  AND ct.realized_pnl_pct IS NOT NULL
                ORDER BY s.signal_date DESC
                LIMIT 5000
            """)
            rows = conn.fetchall()
    except Exception as exc:
        logger.warning("[ConvictionCalibration] Failed to query data: %s", exc)
        return results

    if len(rows) < _MIN_CALIBRATION_TRADES:
        logger.info(
            "[ConvictionCalibration] Only %d trade-signal pairs (need %d), skipping",
            len(rows), _MIN_CALIBRATION_TRADES,
        )
        return results

    # Parse conviction_factors from metadata JSON
    import json
    factor_data: dict[str, list[tuple[float, float]]] = {f: [] for f in _CALIBRATION_FACTORS}

    for row in rows:
        meta_raw = row["metadata"]
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        except (json.JSONDecodeError, TypeError):
            continue

        factors = meta.get("conviction_factors")
        if not isinstance(factors, dict):
            continue

        pnl = float(row["realized_pnl_pct"])
        for factor_name in _CALIBRATION_FACTORS:
            val = factors.get(factor_name)
            if val is not None:
                factor_data[factor_name].append((float(val), pnl))

    # Compute correlations and store calibrated params
    from scipy.stats import pearsonr as _pearsonr

    for factor_name, pairs in factor_data.items():
        if len(pairs) < _MIN_CALIBRATION_TRADES:
            logger.debug(
                "[ConvictionCalibration] %s: %d pairs (need %d), skipping",
                factor_name, len(pairs), _MIN_CALIBRATION_TRADES,
            )
            continue

        factor_vals = [p[0] for p in pairs]
        pnl_vals = [p[1] for p in pairs]

        try:
            corr, pvalue = _pearsonr(factor_vals, pnl_vals)
        except Exception:
            continue

        if not math.isfinite(corr) or abs(corr) < _MIN_CORRELATION:
            logger.debug(
                "[ConvictionCalibration] %s: corr=%.4f (below threshold), skipping",
                factor_name, corr if math.isfinite(corr) else float("nan"),
            )
            continue

        # Derive optimized scaling: simple linear fit slope as adjustment
        mean_factor = sum(factor_vals) / len(factor_vals)
        mean_pnl = sum(pnl_vals) / len(pnl_vals)
        cov = sum((f - mean_factor) * (p - mean_pnl) for f, p in pairs) / len(pairs)
        var_f = sum((f - mean_factor) ** 2 for f in factor_vals) / len(factor_vals)
        slope = cov / var_f if var_f > 0 else 0.0

        # R-squared
        r_squared = corr ** 2

        params = {
            "correlation": round(corr, 4),
            "slope": round(slope, 6),
            "r_squared": round(r_squared, 4),
        }

        # Upsert into conviction_calibration_params
        try:
            with db_conn() as conn:
                for param_name, param_value in params.items():
                    conn.execute(
                        """
                        INSERT INTO conviction_calibration_params
                            (factor_name, param_name, param_value, calibrated_at, sample_size, r_squared)
                        VALUES (%s, %s, %s, NOW(), %s, %s)
                        ON CONFLICT (factor_name, param_name)
                        DO UPDATE SET param_value = EXCLUDED.param_value,
                                      calibrated_at = EXCLUDED.calibrated_at,
                                      sample_size = EXCLUDED.sample_size,
                                      r_squared = EXCLUDED.r_squared
                        """,
                        [factor_name, param_name, param_value, len(pairs), r_squared],
                    )
        except Exception as exc:
            logger.warning("[ConvictionCalibration] Upsert failed for %s: %s", factor_name, exc)
            continue

        results[factor_name] = params
        logger.info(
            "[ConvictionCalibration] %s: corr=%.4f slope=%.6f r2=%.4f n=%d",
            factor_name, corr, slope, r_squared, len(pairs),
        )

    return results
```

### Step 3: Module-level cache for calibrated params

File: `src/quantstack/learning/ic_attribution.py`, add near the top (after imports):

```python
import time as _time

_CALIBRATED_PARAMS: dict[str, dict[str, float]] = {}
_CALIBRATED_PARAMS_LOADED_AT: float = 0.0
_CALIBRATION_CACHE_TTL = 7 * 24 * 3600  # 1 week in seconds


def get_calibrated_conviction_params() -> dict[str, dict[str, float]]:
    """Read calibrated conviction params with weekly cache.

    Returns:
        {factor_name: {param_name: param_value}} or empty dict.
    """
    global _CALIBRATED_PARAMS, _CALIBRATED_PARAMS_LOADED_AT

    now = _time.time()
    if _CALIBRATED_PARAMS and (now - _CALIBRATED_PARAMS_LOADED_AT) < _CALIBRATION_CACHE_TTL:
        return _CALIBRATED_PARAMS

    try:
        with db_conn() as conn:
            conn.execute(
                "SELECT factor_name, param_name, param_value "
                "FROM conviction_calibration_params"
            )
            rows = conn.fetchall()

        result: dict[str, dict[str, float]] = {}
        for row in rows:
            result.setdefault(row["factor_name"], {})[row["param_name"]] = row["param_value"]

        _CALIBRATED_PARAMS = result
        _CALIBRATED_PARAMS_LOADED_AT = now
        return result
    except Exception as exc:
        logger.warning("[ConvictionCalibration] Cache refresh failed: %s", exc)
        return _CALIBRATED_PARAMS  # Return stale cache on failure
```

### Step 4: Read calibrated params in `_conviction_multiplicative()`

File: `src/quantstack/signal_engine/synthesis.py`, in `_conviction_multiplicative()` (line 648).

Add at the top of the method, before the factor computations:

```python
        # P05 §5.3: Load empirically calibrated factor params (if available)
        calibrated = {}
        try:
            from quantstack.learning.ic_attribution import get_calibrated_conviction_params
            calibrated = get_calibrated_conviction_params()
        except Exception:
            pass  # Fall back to hardcoded defaults

        # Factor 1: ADX strength (calibrated or default)
        adx_params = calibrated.get("adx", {})
        adx_threshold = adx_params.get("threshold", 15)
        adx_scale = adx_params.get("scale_factor", 0.15)
        ...
```

**Important**: For the initial implementation, only read the `correlation` and `slope` values for logging/diagnostics. Do NOT change the actual factor computations yet. The calibration needs to accumulate several quarters of data before the params are trustworthy enough to replace hardcoded defaults.

Add a comment in the code:

```python
        # NOTE: Calibrated params are logged but not yet used to replace hardcoded
        # factor computations. Requires 2+ quarters of data accumulation before
        # the calibrated values are stable enough to trust. Enable replacement
        # via FEEDBACK_CONVICTION_CALIBRATION flag when ready.
```

## Edge Cases

1. **No closed_trades data**: The JOIN returns no rows. Calibration skips with an INFO log.
2. **signals.metadata missing conviction_factors key**: Old signals before this enrichment won't have the key. The `meta.get("conviction_factors")` returns None, row is skipped.
3. **Pearson correlation with constant values**: Returns NaN. The `math.isfinite()` check handles this.
4. **Cache stampede**: Single-threaded Python GIL prevents true stampede. The stale cache is returned while refresh happens.
5. **Date proximity JOIN**: `entry_date BETWEEN signal_date - 1 AND signal_date + 1` accounts for signals generated after market close being matched to next-day entries.
6. **Very large result set**: LIMIT 5000 bounds the query. This is ~1 year of daily trading across 20 symbols.

## Tests

File: `tests/unit/signal_engine/test_conviction_calibration.py`

```python
"""Tests for conviction factor calibration."""

def test_metadata_enrichment_includes_conviction_factors(monkeypatch):
    """Signals INSERT metadata contains conviction_factors dict."""
    # Run RuleBasedSynthesizer.synthesize() with mocked DB
    # Capture the INSERT params
    # Assert: metadata JSON contains "conviction_factors" key with 6 factor values

def test_calibrate_with_sufficient_data(monkeypatch):
    """Calibration produces params when >100 trade-signal pairs exist."""
    # Mock db_conn to return 200 rows with known factor values and PnL
    # Assert: calibrate_conviction_factors() returns non-empty dict
    # Assert: upsert SQL called for each factor with significant correlation

def test_calibrate_skips_with_insufficient_data(monkeypatch):
    """Calibration is a no-op when <100 trade-signal pairs."""
    # Mock db_conn to return 50 rows
    # Assert: returns empty dict, no upsert called

def test_calibrate_skips_low_correlation_factors(monkeypatch):
    """Factors with |corr| < 0.01 are not stored."""
    # Mock data where adx factor is random noise (near-zero correlation)
    # Assert: adx not in returned dict

def test_cache_returns_stale_on_failure(monkeypatch):
    """Cache returns stale data when DB read fails."""
    # Pre-populate _CALIBRATED_PARAMS
    # Mock db_conn to raise exception
    # Assert: get_calibrated_conviction_params() returns stale data

def test_cache_refreshes_after_ttl(monkeypatch):
    """Cache refreshes from DB after TTL expires."""
    # Set _CALIBRATED_PARAMS_LOADED_AT to old timestamp
    # Mock db_conn to return new data
    # Assert: returned dict matches new DB data

def test_conviction_multiplicative_falls_back_to_hardcoded():
    """Factor computation uses hardcoded defaults when no calibrated params."""
    # Ensure _CALIBRATED_PARAMS is empty
    # Call _conviction_multiplicative with known inputs
    # Assert: output matches hardcoded factor logic exactly
```
