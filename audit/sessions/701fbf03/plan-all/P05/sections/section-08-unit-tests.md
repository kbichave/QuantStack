# Section 08: Unit Tests

## Objective

Write comprehensive unit tests for all P05 features. Tests use `monkeypatch` for DB isolation and `pytest.parametrize` for edge cases. All tests run without a real database.

## Dependencies

- **Sections 01-07** must be implemented (these tests validate that code)

## Files to Create

All tests go in `tests/unit/signal_engine/`:

1. `test_ic_weight_precompute.py`
2. `test_transition_zone.py`
3. `test_conviction_calibration.py`
4. `test_ensemble_ab.py`

If `tests/unit/signal_engine/__init__.py` does not exist, create it (empty file).

## Test Infrastructure

### DB Mocking Pattern

All P05 code uses `db_conn()` as a context manager. The standard monkeypatch pattern:

```python
from unittest.mock import MagicMock, patch
from contextlib import contextmanager


class FakeConn:
    """Mock PgConnection that returns configurable rows."""

    def __init__(self, rows=None):
        self._rows = rows or []
        self._execute_calls = []

    def execute(self, query, params=None):
        self._execute_calls.append((query, params))
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


@contextmanager
def fake_db_conn(rows=None):
    conn = FakeConn(rows)
    yield conn


def mock_db(monkeypatch, rows=None):
    """Monkeypatch db_conn to return a FakeConn."""
    conn = FakeConn(rows)

    @contextmanager
    def _fake():
        yield conn

    monkeypatch.setattr("quantstack.learning.ic_attribution.db_conn", _fake)
    return conn
```

### Common Fixtures

```python
import pytest
from datetime import date, datetime, timezone


@pytest.fixture
def trending_up_ic_data():
    """63 days of IC attribution data for trending_up regime, 3 collectors."""
    rows = []
    for i in range(63):
        for coll, sig, ret in [("trend", 0.8, 0.01), ("rsi", 0.3, 0.005), ("ml", 0.6, 0.008)]:
            rows.append({
                "collector": coll,
                "signal_value": sig + (i * 0.001),  # slight variation
                "forward_return": ret + (i * 0.0001),
            })
    return rows
```

## File 1: `test_ic_weight_precompute.py`

```python
"""Tests for IC weight precomputation (Section 02)."""

import math
import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager


class TestComputeAndStoreIcWeights:
    """Tests for compute_and_store_ic_weights()."""

    def test_computes_weights_for_regime_with_data(self, monkeypatch):
        """Weights are computed and stored for regimes with sufficient data."""
        # Setup: 63 rows for trending_up with 3 collectors
        # Collector A: strong positive IC, B: weak positive, C: negative IC
        # Assert: A and B have weights, C is gated out
        # Assert: weights sum to 1.0
        # Assert: upsert SQL executed for A and B

    def test_skips_regime_with_no_data(self, monkeypatch):
        """Regime with zero observations is skipped."""
        # Mock DB returns empty for "unknown" regime
        # Assert: "unknown" not in results dict

    def test_skips_regime_below_weight_floor(self, monkeypatch):
        """Regime where all collectors have IC < 0.02 is skipped."""
        # All collectors return near-zero IC
        # Assert: regime skipped, result dict empty for that regime

    def test_icir_penalty_reduces_noisy_collector(self, monkeypatch):
        """Collector with high IC variance gets 0.7x penalty."""
        # Collector with IC=0.05 but std(IC)=1.0 (ICIR=0.05)
        # Assert: weight reduced by 0.7x before normalization

    def test_correlation_penalty_applied(self, monkeypatch):
        """Redundant collector pair gets correlation penalty."""
        # Mock CrossSectionalICTracker.compute_pairwise_correlation()
        # to penalize one collector
        # Assert: penalized collector has lower weight

    def test_idempotent_upsert(self, monkeypatch):
        """Running twice produces same weights (ON CONFLICT DO UPDATE)."""
        # Run compute_and_store_ic_weights() twice
        # Assert: second run's upsert SQL uses same values

    @pytest.mark.parametrize("n_observations", [0, 5, 19])
    def test_insufficient_observations_skipped(self, monkeypatch, n_observations):
        """Collectors with <20 observations are excluded."""
        # Mock data with only n_observations for one collector
        # Assert: that collector not in weights


class TestGetPrecomputedWeights:
    """Tests for get_precomputed_weights()."""

    def test_returns_dict_when_fresh(self, monkeypatch):
        """Fresh data (within 7 days) returns {collector: weight}."""
        # Mock DB returns 3 rows
        # Assert: returns dict with 3 entries

    def test_returns_none_when_stale(self, monkeypatch):
        """Stale data (>7 days) returns None."""
        # Mock DB returns 0 rows (stale filter)
        # Assert: returns None

    def test_returns_none_when_empty(self, monkeypatch):
        """No rows for regime returns None."""
        # Mock DB returns empty
        # Assert: returns None

    def test_returns_none_on_db_error(self, monkeypatch):
        """DB connection failure returns None (no exception raised)."""
        # Mock db_conn to raise
        # Assert: returns None


class TestHelpers:
    """Tests for _rolling_sub_ics and _std."""

    def test_rolling_sub_ics_correct_count(self):
        """63 data points with window=21 produces 3 sub-ICs."""
        from quantstack.learning.ic_attribution import _rolling_sub_ics
        signals = [float(i) for i in range(63)]
        returns = [float(i) * 0.01 for i in range(63)]
        result = _rolling_sub_ics(signals, returns, window=21)
        assert len(result) == 3

    def test_std_empty_list(self):
        from quantstack.learning.ic_attribution import _std
        assert _std([]) == 0.0

    def test_std_single_element(self):
        from quantstack.learning.ic_attribution import _std
        assert _std([5.0]) == 0.0

    def test_std_known_values(self):
        from quantstack.learning.ic_attribution import _std
        result = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.0) < 0.01  # population std = 2.0
```

## File 2: `test_transition_zone.py`

```python
"""Tests for transition zone propagation (Section 03)."""

import pytest


class TestSymbolBriefTransitionZone:
    """Tests for the transition_zone field on SymbolBrief."""

    def test_defaults_to_false(self):
        from quantstack.shared.schemas import SymbolBrief
        brief = SymbolBrief(
            symbol="AAPL", market_summary="test",
            consensus_bias="neutral", pod_agreement="mixed",
        )
        assert brief.transition_zone is False

    def test_can_be_set_true(self):
        from quantstack.shared.schemas import SymbolBrief
        brief = SymbolBrief(
            symbol="AAPL", market_summary="test",
            consensus_bias="neutral", pod_agreement="mixed",
            transition_zone=True,
        )
        assert brief.transition_zone is True

    def test_serialization_roundtrip(self):
        from quantstack.shared.schemas import SymbolBrief
        brief = SymbolBrief(
            symbol="AAPL", market_summary="test",
            consensus_bias="neutral", pod_agreement="mixed",
            transition_zone=True,
        )
        data = brief.model_dump()
        assert data["transition_zone"] is True
        restored = SymbolBrief(**data)
        assert restored.transition_zone is True


class TestSynthesisTransitionZone:
    """Tests for synthesis setting transition_zone based on regime."""

    def test_set_when_probability_above_threshold(self, monkeypatch):
        """transition_zone=True when transition_probability > 0.3."""
        # Mock DB to prevent INSERT
        # Call synthesize() with regime={"trend_regime": "trending_up", "transition_probability": 0.5}
        # Assert: brief.transition_zone is True

    def test_not_set_when_probability_below_threshold(self, monkeypatch):
        """transition_zone=False when transition_probability <= 0.3."""
        # Call synthesize() with regime={"trend_regime": "trending_up", "transition_probability": 0.2}
        # Assert: brief.transition_zone is False

    def test_not_set_when_probability_missing(self, monkeypatch):
        """transition_zone=False when transition_probability absent."""
        # Call synthesize() with regime={"trend_regime": "trending_up"}
        # Assert: brief.transition_zone is False

    @pytest.mark.parametrize("prob", [0.0, 0.1, 0.3])
    def test_boundary_values_not_in_zone(self, monkeypatch, prob):
        """Boundary: exactly 0.3 is NOT in transition zone (exclusive)."""


class TestTransitionPositionSizingFlag:
    """Tests for the feedback flag."""

    def test_default_is_false(self):
        from quantstack.config.feedback_flags import transition_position_sizing_enabled
        assert transition_position_sizing_enabled() is False

    def test_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("FEEDBACK_TRANSITION_POSITION_SIZING", "true")
        from quantstack.config.feedback_flags import transition_position_sizing_enabled
        assert transition_position_sizing_enabled() is True
```

## File 3: `test_conviction_calibration.py`

```python
"""Tests for conviction factor calibration (Section 04)."""

import json
import math
import pytest
from unittest.mock import MagicMock


class TestMetadataEnrichment:
    """Tests for conviction_factors in signals metadata."""

    def test_signals_insert_includes_conviction_factors(self, monkeypatch):
        """The signals INSERT metadata contains conviction_factors dict."""
        # Capture INSERT params by mocking db_conn
        # Run synthesize() with full inputs
        # Parse the JSON metadata param
        # Assert: "conviction_factors" key exists
        # Assert: contains adx, stability, timeframe, regime_agreement, ml_confirmation, data_quality


class TestCalibrateConvictionFactors:
    """Tests for calibrate_conviction_factors() batch function."""

    def test_calibrates_with_sufficient_data(self, monkeypatch):
        """Returns calibrated params when >100 trade-signal pairs."""
        # Mock DB to return 200 rows with:
        #   metadata containing conviction_factors
        #   realized_pnl_pct correlated with adx factor
        # Assert: "adx" in results with correlation, slope, r_squared

    def test_skips_with_insufficient_data(self, monkeypatch):
        """Returns empty dict when <100 pairs."""
        # Mock DB with 50 rows
        # Assert: returns {}

    def test_skips_low_correlation_factors(self, monkeypatch):
        """Factors with |correlation| < 0.01 excluded."""
        # Mock data with random noise correlation
        # Assert: factor not in results

    def test_handles_missing_conviction_factors_key(self, monkeypatch):
        """Old signals without conviction_factors are silently skipped."""
        # Mix of rows: some with conviction_factors, some without
        # Assert: no error, only rows with conviction_factors used

    def test_handles_malformed_json(self, monkeypatch):
        """Malformed metadata JSON is skipped without error."""
        # Include rows with invalid JSON in metadata
        # Assert: no exception raised


class TestCalibrationCache:
    """Tests for get_calibrated_conviction_params() cache."""

    def test_returns_cached_within_ttl(self, monkeypatch):
        """Cache hit: returns stored params without DB query."""
        import quantstack.learning.ic_attribution as mod
        mod._CALIBRATED_PARAMS = {"adx": {"correlation": 0.05}}
        mod._CALIBRATED_PARAMS_LOADED_AT = __import__("time").time()
        # Mock db_conn to track if called
        # Call get_calibrated_conviction_params()
        # Assert: DB not called, returns cached value

    def test_refreshes_after_ttl(self, monkeypatch):
        """Cache miss after TTL: queries DB for fresh data."""
        import quantstack.learning.ic_attribution as mod
        mod._CALIBRATED_PARAMS = {"old": {}}
        mod._CALIBRATED_PARAMS_LOADED_AT = 0.0  # expired
        # Mock db_conn to return new data
        # Assert: returns new data

    def test_returns_stale_on_db_error(self, monkeypatch):
        """Returns stale cache when DB query fails."""
        import quantstack.learning.ic_attribution as mod
        mod._CALIBRATED_PARAMS = {"stale": {"param": 1.0}}
        mod._CALIBRATED_PARAMS_LOADED_AT = 0.0
        # Mock db_conn to raise
        # Assert: returns stale dict
```

## File 4: `test_ensemble_ab.py`

```python
"""Tests for ensemble A/B test tracking and evaluation (Section 05)."""

import math
import pytest
from unittest.mock import MagicMock


class TestEnsembleMethods:
    """Verify ensemble method outputs are deterministic."""

    def test_weighted_avg(self):
        from quantstack.signal_engine.synthesis import _ensemble_weighted_avg
        scores = {"trend": 1.0, "rsi": -0.5}
        weights = {"trend": 0.6, "rsi": 0.4}
        result = _ensemble_weighted_avg(scores, weights)
        assert abs(result - 0.4) < 0.001

    def test_weighted_median(self):
        from quantstack.signal_engine.synthesis import _ensemble_weighted_median
        scores = {"a": 0.0, "b": 0.5, "c": 1.0}
        weights = {"a": 0.2, "b": 0.3, "c": 0.5}
        result = _ensemble_weighted_median(scores, weights)
        assert result == 1.0  # cumulative 0.5 >= 0.5 at c

    def test_trimmed_mean(self):
        from quantstack.signal_engine.synthesis import _ensemble_trimmed_mean
        scores = {"a": -1.0, "b": 0.0, "c": 0.5, "d": 1.0}
        weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        result = _ensemble_trimmed_mean(scores, weights)
        # Drops a (-1.0) and d (1.0), avg of b (0.0) and c (0.5) = 0.25
        assert abs(result - 0.25) < 0.001


class TestABRecording:
    """Tests for recording ensemble outputs during synthesis."""

    def test_all_methods_recorded(self, monkeypatch):
        """When AB test enabled, all 3 method outputs are recorded."""
        # Enable FEEDBACK_ENSEMBLE_AB_TEST
        # Mock db_conn, capture INSERT calls
        # Run _compute_bias_and_conviction()
        # Assert: 3 INSERT calls to ensemble_ab_results

    def test_no_recording_when_flag_disabled(self, monkeypatch):
        """No recording when FEEDBACK_ENSEMBLE_AB_TEST is not set."""
        # Do not set the flag
        # Assert: no INSERT calls to ensemble_ab_results


class TestEvaluateEnsembleAB:
    """Tests for evaluate_ensemble_ab() weekly evaluation."""

    def test_promotes_clear_winner(self, monkeypatch):
        """Promotes method with significantly better IC (p < 0.05)."""
        # Mock ensemble_ab_results + OHLCV data where:
        #   weighted_median has IC=0.15, weighted_avg has IC=0.05
        #   40 unique dates
        # Assert: result["winner"] == "weighted_median"
        # Assert: result["promoted"] is True

    def test_no_promotion_insufficient_days(self, monkeypatch):
        """Does not promote with <30 unique dates."""
        # Mock data with 15 dates
        # Assert: result["promoted"] is False

    def test_no_promotion_high_pvalue(self, monkeypatch):
        """Does not promote when p-value >= 0.05."""
        # Mock data where all methods have similar IC
        # Assert: result["promoted"] is False

    def test_handles_empty_data(self, monkeypatch):
        """Returns defaults when no data exists."""
        # Mock empty query
        # Assert: result["winner"] == "weighted_avg"
        # Assert: result["promoted"] is False

    def test_handles_db_error(self, monkeypatch):
        """Returns defaults on DB query failure."""
        # Mock db_conn to raise
        # Assert: no exception, returns default dict

    @pytest.mark.parametrize("current_method", ["weighted_avg", "weighted_median", "trimmed_mean"])
    def test_regression_reverts_to_baseline(self, monkeypatch, current_method):
        """If current promoted method regresses, baseline wins."""
        # Mock data where weighted_avg is now best
        # Assert: weighted_avg is winner


class TestGetActiveEnsembleMethod:
    """Tests for _get_active_ensemble_method() cache."""

    def test_returns_default_on_fresh_start(self, monkeypatch):
        """Returns 'weighted_avg' when cache is empty."""
        import quantstack.signal_engine.synthesis as mod
        mod._active_ensemble_cache = ("weighted_avg", 0.0)
        # Mock db_conn to return row with active_method='weighted_avg'
        result = mod._get_active_ensemble_method()
        assert result == "weighted_avg"

    def test_caches_for_one_hour(self, monkeypatch):
        """Does not query DB within 1 hour of last load."""
        import time
        import quantstack.signal_engine.synthesis as mod
        mod._active_ensemble_cache = ("trimmed_mean", time.time())
        # Do NOT mock db_conn (it should not be called)
        result = mod._get_active_ensemble_method()
        assert result == "trimmed_mean"
```

## Running Tests

```bash
# Run all P05 unit tests
pytest tests/unit/signal_engine/test_ic_weight_precompute.py \
       tests/unit/signal_engine/test_transition_zone.py \
       tests/unit/signal_engine/test_conviction_calibration.py \
       tests/unit/signal_engine/test_ensemble_ab.py \
       -v

# Run with coverage
pytest tests/unit/signal_engine/ -v --cov=quantstack.learning.ic_attribution \
       --cov=quantstack.signal_engine.synthesis --cov=quantstack.shared.schemas \
       --cov=quantstack.config.feedback_flags
```

## Quality Criteria

- All tests pass with `pytest -x` (fail-fast)
- No test requires a running PostgreSQL instance
- No test imports from `quantstack.db` without mocking
- Edge cases covered: empty data, stale data, DB errors, boundary values
- `pytest.parametrize` used for threshold boundary testing
