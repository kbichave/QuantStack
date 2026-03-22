"""Tests for proactive drift detection via PSI."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantstack.learning.drift_detector import (
    TRACKED_FEATURES,
    DriftDetector,
    DriftReport,
    compute_psi,
)


# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------


class TestComputePSI:
    """Core PSI computation."""

    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"Identical distributions should have PSI ~0, got {psi:.4f}"

    def test_similar_distributions_low_psi(self):
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(0.05, 1, 1000)  # tiny mean shift
        psi = compute_psi(expected, actual)
        assert (
            psi < 0.10
        ), f"Similar distributions should have PSI < 0.10, got {psi:.4f}"

    def test_shifted_distribution_moderate_psi(self):
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(0.5, 1, 1000)  # moderate mean shift
        psi = compute_psi(expected, actual)
        assert psi > 0.01, f"Shifted distribution should have PSI > 0.01, got {psi:.4f}"

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(3, 0.5, 1000)  # large shift + different spread
        psi = compute_psi(expected, actual)
        assert (
            psi > 0.25
        ), f"Very different distributions should have PSI > 0.25, got {psi:.4f}"

    def test_psi_is_nonnegative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            expected = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 500)
            actual = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 500)
            assert compute_psi(expected, actual) >= 0

    def test_empty_arrays_return_zero(self):
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert compute_psi(np.array([1, 2, 3]), np.array([])) == 0.0

    def test_single_element_returns_zero(self):
        assert compute_psi(np.array([1.0]), np.array([2.0])) == 0.0

    def test_nan_values_handled(self):
        expected = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        actual = np.array([1.5, 2.5, 3.5, 4.5, np.nan, 5.5])
        psi = compute_psi(expected, actual)
        assert np.isfinite(psi)

    def test_psi_increases_with_shift_magnitude(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 1000)
        psi_small = compute_psi(baseline, rng.normal(0.2, 1, 1000))
        psi_large = compute_psi(baseline, rng.normal(1.0, 1, 1000))
        assert psi_large > psi_small


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class TestDriftDetector:
    """DriftDetector baseline management and drift checking."""

    @pytest.fixture
    def tmp_baseline_dir(self, tmp_path):
        return tmp_path / "drift_baselines"

    @pytest.fixture
    def detector(self, tmp_baseline_dir):
        return DriftDetector(baseline_dir=tmp_baseline_dir)

    @pytest.fixture
    def sample_baseline(self):
        rng = np.random.default_rng(42)
        return {
            "rsi_14": rng.uniform(20, 80, 252),
            "atr_pct": rng.uniform(0.01, 0.05, 252),
            "adx_14": rng.uniform(10, 50, 252),
            "bb_pct": rng.uniform(-0.5, 1.5, 252),
            "volume_ratio": rng.uniform(0.5, 2.0, 252),
            "regime_confidence": rng.uniform(0.3, 0.95, 252),
        }

    def test_no_baseline_returns_none_severity(self, detector):
        report = detector.check_drift("nonexistent", {"rsi_14": np.array([50.0])})
        assert report.severity == "NONE"
        assert report.overall_psi == 0.0

    def test_set_and_has_baseline(self, detector, sample_baseline):
        assert not detector.has_baseline("strat_test")
        detector.set_baseline("strat_test", sample_baseline)
        assert detector.has_baseline("strat_test")

    def test_baseline_persists_to_disk(
        self, detector, tmp_baseline_dir, sample_baseline
    ):
        detector.set_baseline("strat_disk", sample_baseline)
        path = tmp_baseline_dir / "strat_disk.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "rsi_14" in data
        assert len(data["rsi_14"]) == 252

    def test_baseline_loaded_from_disk(self, tmp_baseline_dir, sample_baseline):
        # Write with one detector instance
        det1 = DriftDetector(baseline_dir=tmp_baseline_dir)
        det1.set_baseline("strat_load", sample_baseline)

        # Read with a new instance (empty cache)
        det2 = DriftDetector(baseline_dir=tmp_baseline_dir)
        assert det2.has_baseline("strat_load")

    def test_identical_features_no_drift(self, detector, sample_baseline):
        detector.set_baseline("strat_ok", sample_baseline)
        report = detector.check_drift("strat_ok", sample_baseline)
        assert report.severity == "NONE"
        assert report.overall_psi < 0.05

    def test_shifted_features_trigger_warning(self, detector, sample_baseline):
        detector.set_baseline("strat_warn", sample_baseline)

        # Shift RSI distribution significantly
        shifted = dict(sample_baseline)
        shifted["rsi_14"] = sample_baseline["rsi_14"] + 15  # shift mean by 15

        report = detector.check_drift("strat_warn", shifted)
        assert report.severity in ("WARNING", "CRITICAL")
        assert "rsi_14" in report.drifted_features

    def test_very_shifted_features_trigger_critical(self, detector):
        rng = np.random.default_rng(42)
        baseline = {"rsi_14": rng.uniform(20, 40, 252)}  # low RSI regime
        detector.set_baseline("strat_crit", baseline)

        current = {"rsi_14": rng.uniform(60, 80, 252)}  # completely different regime
        report = detector.check_drift("strat_crit", current)
        assert report.severity == "CRITICAL"
        assert report.overall_psi >= 0.25

    def test_report_has_correct_fields(self, detector, sample_baseline):
        detector.set_baseline("strat_fields", sample_baseline)
        report = detector.check_drift("strat_fields", sample_baseline)

        assert isinstance(report, DriftReport)
        assert report.strategy_id == "strat_fields"
        assert isinstance(report.feature_psis, dict)
        assert isinstance(report.drifted_features, list)
        assert report.checked_at is not None

    def test_report_to_dict(self, detector, sample_baseline):
        detector.set_baseline("strat_dict", sample_baseline)
        report = detector.check_drift("strat_dict", sample_baseline)
        d = report.to_dict()
        assert "strategy_id" in d
        assert "overall_psi" in d
        assert "severity" in d

    def test_partial_features_handled(self, detector, sample_baseline):
        """Only some features present in current data — should still work."""
        detector.set_baseline("strat_partial", sample_baseline)
        partial = {"rsi_14": sample_baseline["rsi_14"]}  # only RSI
        report = detector.check_drift("strat_partial", partial)
        assert report.severity == "NONE"
        assert "rsi_14" in report.feature_psis

    def test_nan_in_baseline_filtered(self, detector):
        baseline = {"rsi_14": np.array([30.0, np.nan, 40.0, 50.0, 60.0, 70.0] * 50)}
        detector.set_baseline("strat_nan", baseline)
        assert detector.has_baseline("strat_nan")

        # Verify baseline was cleaned
        loaded = detector._load_baseline("strat_nan")
        assert not np.any(np.isnan(loaded["rsi_14"]))


# ---------------------------------------------------------------------------
# check_drift_from_brief
# ---------------------------------------------------------------------------


class TestCheckDriftFromBrief:
    """Feature extraction from SignalBrief dicts."""

    @pytest.fixture
    def detector(self, tmp_path):
        det = DriftDetector(baseline_dir=tmp_path / "baselines")
        rng = np.random.default_rng(42)
        det.set_baseline(
            "strat_brief",
            {
                "rsi_14": rng.uniform(25, 75, 252),
                "atr_pct": rng.uniform(0.01, 0.04, 252),
                "adx_14": rng.uniform(15, 45, 252),
            },
        )
        return det

    def test_flat_brief_format(self, detector):
        brief = {"rsi_14": 45.0, "atr_pct": 0.025, "adx_14": 30.0}
        report = detector.check_drift_from_brief("strat_brief", brief)
        assert isinstance(report, DriftReport)

    def test_nested_brief_format(self, detector):
        brief = {
            "symbol_briefs": [
                {
                    "raw_collectors": {
                        "technical": {
                            "rsi_14": 45.0,
                            "atr_pct": 0.025,
                            "adx_14": 30.0,
                        }
                    }
                }
            ]
        }
        report = detector.check_drift_from_brief("strat_brief", brief)
        assert isinstance(report, DriftReport)

    def test_empty_brief_returns_none_severity(self, detector):
        report = detector.check_drift_from_brief("strat_brief", {})
        assert report.severity == "NONE"

    def test_missing_strategy_returns_none_severity(self, detector):
        report = detector.check_drift_from_brief("nonexistent", {"rsi_14": 50.0})
        assert report.severity == "NONE"
