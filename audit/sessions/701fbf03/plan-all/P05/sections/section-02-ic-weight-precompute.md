# Section 02: IC Weight Precomputation

## Objective

Replace the expensive per-synthesis-call `ICAttributionTracker().get_weights_for_regime()` with a precomputed lookup table. A weekly batch job computes IC-driven weights for each regime, stores them in `precomputed_ic_weights`, and synthesis reads from the table in constant time.

## Dependencies

- **Section 01** (schema migrations): `precomputed_ic_weights` table must exist.

## Files to Modify

1. **`src/quantstack/learning/ic_attribution.py`** -- add two new functions
2. **`src/quantstack/signal_engine/cross_sectional_ic.py`** -- import for correlation penalty (already exists, used as dependency)

## Current State

- `ICAttributionTracker.get_weights_for_regime()` (line 308-347 of `ic_attribution.py`) loads all observations in-memory, filters by regime, computes Spearman IC per collector, normalizes positive-IC collectors to sum 1.0. This runs on every synthesis call when `FEEDBACK_IC_DRIVEN_WEIGHTS=true`.
- The `ic_attribution_data` table has a `regime` column (added by P05 migration).
- `CrossSectionalICTracker.compute_pairwise_correlation()` exists and returns per-collector correlation penalties.

## Implementation

### Function 1: `compute_and_store_ic_weights()`

Add to `src/quantstack/learning/ic_attribution.py` as a module-level function (not a method on `ICAttributionTracker` -- this is a batch job, not an instance method).

```python
# At top of file, add these imports (some already exist):
from quantstack.db import db_conn
from scipy.stats import spearmanr as _spearmanr

_BASE_REGIMES = ("trending_up", "trending_down", "ranging", "unknown")
_IC_FLOOR = 0.02       # Drop collectors with IC below this
_ICIR_THRESHOLD = 0.1  # ICIR penalty threshold
_ICIR_PENALTY = 0.7    # Multiply weight by this if ICIR < threshold
_WEIGHT_FLOOR = 0.1    # If total positive-IC weight < this, skip (static fallback)


def compute_and_store_ic_weights() -> dict[str, dict[str, float]]:
    """Batch-compute IC-driven weights per regime and store in precomputed_ic_weights.

    For each base regime:
      1. Query ic_attribution_data for the last 63 days, filtered by regime
      2. Compute per-collector Spearman IC
      3. Apply IC gate: drop collectors with IC < 0.02
      4. Apply ICIR penalty: multiply by 0.7 if IC/std(IC) < 0.1
      5. Apply correlation penalty from CrossSectionalICTracker
      6. Check weight floor: if total positive weight < 0.1, skip regime
      7. Normalize to sum=1.0
      8. Upsert into precomputed_ic_weights

    Returns:
        {regime: {collector: weight}} for logging/diagnostics.
    """
    results: dict[str, dict[str, float]] = {}

    for regime in _BASE_REGIMES:
        # Step 1: Query observations for this regime, last 63 days
        with db_conn() as conn:
            conn.execute(
                """
                SELECT collector, signal_value, forward_return
                FROM ic_attribution_data
                WHERE regime = %s
                  AND recorded_at > NOW() - INTERVAL '63 days'
                ORDER BY collector, recorded_at ASC
                """,
                [regime],
            )
            rows = conn.fetchall()

        if not rows:
            logger.info("[ICPrecompute] No data for regime=%s, skipping", regime)
            continue

        # Group by collector
        collector_data: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            collector = row["collector"]
            collector_data.setdefault(collector, []).append(
                (row["signal_value"], row["forward_return"])
            )

        # Step 2: Compute per-collector Spearman IC
        collector_ic: dict[str, float] = {}
        collector_ic_std: dict[str, float] = {}
        for coll, pairs in collector_data.items():
            if len(pairs) < 20:
                continue
            signals = [p[0] for p in pairs]
            returns = [p[1] for p in pairs]
            corr, _ = _spearmanr(signals, returns)
            if not math.isfinite(corr):
                continue
            collector_ic[coll] = corr

            # Compute rolling IC std for ICIR (use 21-day sub-windows)
            sub_ics = _rolling_sub_ics(signals, returns, window=21)
            if sub_ics:
                std_ic = _std(sub_ics)
                collector_ic_std[coll] = std_ic

        # Step 3: IC gate -- drop collectors below floor
        gated = {k: v for k, v in collector_ic.items() if v >= _IC_FLOOR}

        # Step 4: ICIR penalty
        for coll in list(gated):
            std = collector_ic_std.get(coll, 0.0)
            if std > 0:
                icir = gated[coll] / std
                if icir < _ICIR_THRESHOLD:
                    gated[coll] *= _ICIR_PENALTY

        # Step 5: Correlation penalty
        try:
            from quantstack.signal_engine.cross_sectional_ic import (
                CrossSectionalICTracker,
            )
            penalties = CrossSectionalICTracker().compute_pairwise_correlation()
            if penalties:
                for coll in gated:
                    if coll in penalties:
                        gated[coll] *= penalties[coll]
        except Exception as exc:
            logger.warning("[ICPrecompute] Correlation penalty failed: %s", exc)

        # Step 6: Weight floor check
        total_positive = sum(v for v in gated.values() if v > 0)
        if total_positive < _WEIGHT_FLOOR:
            logger.info(
                "[ICPrecompute] regime=%s total_positive=%.4f < floor=%.2f, "
                "skipping (static fallback will be used)",
                regime, total_positive, _WEIGHT_FLOOR,
            )
            continue

        # Step 7: Normalize to sum=1.0
        positive = {k: v for k, v in gated.items() if v > 0}
        total = sum(positive.values())
        weights = {k: round(v / total, 4) for k, v in positive.items()}

        # Step 8: Upsert into precomputed_ic_weights
        with db_conn() as conn:
            for coll, weight in weights.items():
                ic_val = collector_ic.get(coll, 0.0)
                conn.execute(
                    """
                    INSERT INTO precomputed_ic_weights (regime, collector, weight, ic_value, computed_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (regime, collector)
                    DO UPDATE SET weight = EXCLUDED.weight,
                                  ic_value = EXCLUDED.ic_value,
                                  computed_at = EXCLUDED.computed_at
                    """,
                    [regime, coll, weight, ic_val],
                )

        results[regime] = weights
        logger.info(
            "[ICPrecompute] regime=%s collectors=%d weights=%s",
            regime, len(weights), weights,
        )

    return results
```

### Helper functions (add in the same file, above the main function):

```python
def _rolling_sub_ics(
    signals: list[float], returns: list[float], window: int = 21,
) -> list[float]:
    """Compute IC over rolling sub-windows for ICIR calculation."""
    sub_ics: list[float] = []
    for i in range(0, len(signals) - window + 1, window):
        s = signals[i : i + window]
        r = returns[i : i + window]
        if len(s) < 10:
            continue
        corr, _ = _spearmanr(s, r)
        if math.isfinite(corr):
            sub_ics.append(corr)
    return sub_ics


def _std(values: list[float]) -> float:
    """Standard deviation (population) of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5
```

### Function 2: `get_precomputed_weights()`

Add to `src/quantstack/learning/ic_attribution.py` as a module-level function.

```python
def get_precomputed_weights(regime: str) -> dict[str, float] | None:
    """Read precomputed IC-driven weights for a regime.

    Returns None if:
      - No rows exist for this regime
      - Data is stale (computed_at > 7 days ago)

    When None is returned, the caller should fall back to static weight profiles.
    """
    try:
        with db_conn() as conn:
            conn.execute(
                """
                SELECT collector, weight
                FROM precomputed_ic_weights
                WHERE regime = %s
                  AND computed_at > NOW() - INTERVAL '7 days'
                """,
                [regime],
            )
            rows = conn.fetchall()

        if not rows:
            return None

        return {row["collector"]: row["weight"] for row in rows}
    except Exception as exc:
        logger.warning("[ICPrecompute] Failed to read precomputed weights: %s", exc)
        return None
```

## Edge Cases

1. **No IC data for a regime**: Skip that regime entirely. Log at INFO level. The synthesis will use static weight profiles.
2. **All collectors have IC < 0.02**: The gating step removes all collectors, `total_positive` is 0 (below floor), regime is skipped.
3. **Single collector survives gating**: Valid -- it gets weight 1.0. The synthesis still applies the static fallback vote-redistribution for inactive voters.
4. **Stale precomputed data (>7 days)**: `get_precomputed_weights()` returns None. This handles the case where the weekly job fails to run.
5. **Concurrent reads during write**: PostgreSQL MVCC handles this. Readers see the old weights until the upsert transaction commits.
6. **ICIR with zero std**: If all sub-window ICs are identical, std=0, ICIR is undefined. The `if std > 0` guard prevents division by zero; the penalty is not applied.
7. **`_spearmanr` with constant input**: Returns NaN. The `math.isfinite()` check handles this.

## Tests

File: `tests/unit/signal_engine/test_ic_weight_precompute.py`

```python
"""Tests for IC weight precomputation batch job."""

def test_compute_and_store_ic_weights_with_valid_data(monkeypatch):
    """Weights computed correctly from mocked IC data."""
    # Mock db_conn to return synthetic ic_attribution_data rows
    # 3 collectors, trending_up regime, 63 days of data
    # Collector A: IC=0.10, Collector B: IC=0.05, Collector C: IC=0.01 (below gate)
    # Assert: C excluded, A and B normalized to sum=1.0
    # Assert: upsert SQL called with correct values

def test_compute_skips_regime_with_no_data(monkeypatch):
    """Regime with no observations is skipped (no upsert)."""
    # Mock db_conn to return empty rows for "unknown" regime
    # Assert: result dict does not contain "unknown"

def test_compute_skips_regime_below_weight_floor(monkeypatch):
    """Regime where all ICs are near zero is skipped."""
    # All collectors have IC < 0.02 after gating
    # Assert: regime skipped, static fallback used

def test_icir_penalty_applied(monkeypatch):
    """Collectors with low ICIR get 0.7x weight penalty."""
    # Collector with high IC but high IC variance (ICIR < 0.1)
    # Assert: weight reduced by 0.7x before normalization

def test_correlation_penalty_applied(monkeypatch):
    """Redundant collectors get correlation penalty."""
    # Mock CrossSectionalICTracker().compute_pairwise_correlation()
    # to return penalty for one collector
    # Assert: that collector's weight is reduced

def test_get_precomputed_weights_returns_none_when_stale(monkeypatch):
    """Stale data (>7 days) returns None."""
    # Mock db_conn to return rows with old computed_at
    # Assert: returns None

def test_get_precomputed_weights_returns_none_when_empty(monkeypatch):
    """No rows for regime returns None."""
    # Mock db_conn to return empty result
    # Assert: returns None

def test_get_precomputed_weights_returns_dict_when_fresh(monkeypatch):
    """Fresh data returns {collector: weight} dict."""
    # Mock db_conn to return 3 rows within 7 days
    # Assert: returns dict with 3 entries

def test_rolling_sub_ics_produces_values():
    """Rolling sub-IC computation returns list of floats."""
    # 63 data points, window=21 -> 3 sub-ICs
    # Assert: len(result) == 3, all finite

def test_std_edge_cases():
    """Std of empty/single-element list returns 0.0."""
    assert _std([]) == 0.0
    assert _std([5.0]) == 0.0
```
