"""Tests for enhanced drift detection (dynamic features, calibrated thresholds, rolling IC)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantstack.learning.drift_detector import (
    DEFAULT_TRACKED_FEATURES,
    DriftDetector,
    compute_psi,
)


@pytest.fixture
def tmp_baseline_dir(tmp_path):
    """Create a temporary baseline directory."""
    return tmp_path / "drift_baselines"


@pytest.fixture
def detector(tmp_baseline_dir):
    """Create a DriftDetector with temp directory."""
    return DriftDetector(baseline_dir=tmp_baseline_dir)


class TestDynamicTrackedFeatures:
    def test_dynamic_features_from_baseline(self, detector):
        """Features should be loaded dynamically from baseline file."""
        features = {
            "rsi_14": np.random.randn(100),
            "custom_feature_1": np.random.randn(100),
            "custom_feature_2": np.random.randn(100),
        }
        detector.set_baseline("strat_abc", features, calibrate_thresholds=False)

        tracked = detector._get_tracked_features("strat_abc")
        assert "custom_feature_1" in tracked
        assert "custom_feature_2" in tracked
        assert "rsi_14" in tracked

    def test_fallback_to_defaults(self, detector):
        """Without baseline, should fall back to DEFAULT_TRACKED_FEATURES."""
        tracked = detector._get_tracked_features("nonexistent")
        assert tracked == DEFAULT_TRACKED_FEATURES

    def test_check_drift_uses_all_baseline_features(self, detector):
        """check_drift should check all features in the baseline."""
        baseline = {
            "rsi_14": np.random.randn(100),
            "my_custom": np.random.randn(100),
        }
        detector.set_baseline("strat_x", baseline, calibrate_thresholds=False)

        # Current features with a shifted distribution for my_custom
        current = {
            "rsi_14": np.random.randn(50),
            "my_custom": np.random.randn(50) + 10,  # massive shift
        }
        report = detector.check_drift("strat_x", current)

        assert "my_custom" in report.feature_psis
        assert report.feature_psis["my_custom"] > 0


class TestPerFeaturePSIThresholds:
    def test_calibrated_thresholds_stored(self, detector):
        """set_baseline with calibrate_thresholds=True stores per-feature thresholds."""
        rng = np.random.default_rng(42)
        features = {
            "feat_a": rng.standard_normal(200),
            "feat_b": rng.standard_normal(200),
        }
        detector.set_baseline("strat_cal", features, calibrate_thresholds=True)

        # Check thresholds were computed
        assert "strat_cal" in detector._threshold_cache
        thresholds = detector._threshold_cache["strat_cal"]
        assert "feat_a" in thresholds
        assert "feat_b" in thresholds

        # Thresholds should be floored at global defaults
        from quantstack.learning.drift_detector import PSI_CRITICAL, PSI_WARNING

        for feat, (warn, crit) in thresholds.items():
            assert warn >= PSI_WARNING
            assert crit >= PSI_CRITICAL

    def test_per_feature_thresholds_used_in_check(self, detector):
        """check_drift should use per-feature thresholds when available."""
        rng = np.random.default_rng(42)
        features = {"feat_a": rng.standard_normal(200)}
        detector.set_baseline("strat_pf", features, calibrate_thresholds=True)

        # A small shift that might be below calibrated threshold
        current = {"feat_a": rng.standard_normal(50) + 0.1}
        report = detector.check_drift("strat_pf", current)

        # Should not crash; result should be a valid DriftReport
        assert report.severity in ("NONE", "WARNING", "CRITICAL")

    def test_small_feature_skips_calibration(self, detector):
        """Features with < 20 samples should skip threshold calibration."""
        features = {
            "small_feat": np.array([1.0, 2.0, 3.0]),  # only 3 samples
        }
        detector.set_baseline("strat_small", features, calibrate_thresholds=True)

        # Should not have calibrated thresholds for small_feat
        thresholds = detector._threshold_cache.get("strat_small", {})
        assert "small_feat" not in thresholds


class TestRollingICHistory:
    def test_record_and_load(self, detector):
        """Rolling IC values should be recorded and loadable."""
        # Record 70 days of IC data
        for day in range(70):
            detector.record_rolling_ic("strat_ic", {
                "rsi_14": 0.05 + np.random.randn() * 0.01,
                "adx_14": 0.03 + np.random.randn() * 0.01,
            })

        baseline = detector.load_ic_baseline("strat_ic", warmup_entries=60)
        assert "rsi_14" in baseline
        assert "adx_14" in baseline

        # Each entry should be (mean, std)
        mean, std = baseline["rsi_14"]
        assert abs(mean - 0.05) < 0.05  # roughly centered
        assert std > 0

    def test_insufficient_history_returns_empty(self, detector):
        """With < warmup entries, load_ic_baseline returns empty."""
        for day in range(10):
            detector.record_rolling_ic("strat_short", {"rsi_14": 0.05})

        baseline = detector.load_ic_baseline("strat_short", warmup_entries=60)
        assert baseline == {}

    def test_no_history_returns_empty(self, detector):
        """No history file returns empty baseline."""
        baseline = detector.load_ic_baseline("nonexistent")
        assert baseline == {}


class TestBaselineFormat:
    def test_new_format_required(self, detector, tmp_baseline_dir):
        """Baseline files must use the _features/_thresholds format."""
        tmp_baseline_dir.mkdir(parents=True, exist_ok=True)
        # Old format (no _features key) should NOT load
        old_data = {
            "rsi_14": list(np.random.randn(100)),
        }
        path = tmp_baseline_dir / "strat_old.json"
        path.write_text(json.dumps(old_data))

        baseline = detector._load_baseline("strat_old")
        assert baseline is None  # Should fail to load

    def test_proper_format_loads(self, detector, tmp_baseline_dir):
        """Baseline with _features key loads correctly."""
        tmp_baseline_dir.mkdir(parents=True, exist_ok=True)
        proper_data = {
            "_features": {"rsi_14": list(np.random.randn(100))},
            "_thresholds": {},
        }
        path = tmp_baseline_dir / "strat_proper.json"
        path.write_text(json.dumps(proper_data))

        baseline = detector._load_baseline("strat_proper")
        assert baseline is not None
        assert "rsi_14" in baseline

    def test_compute_psi_unchanged(self):
        """PSI computation should be unchanged."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100) + 5  # massive shift

        psi = compute_psi(a, b)
        assert psi > 0.1  # should detect shift
