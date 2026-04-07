# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 13: Concept Drift Detection.

Covers IC-based drift, label drift (KS test), interaction drift (adversarial
validation), auto-retrain decision tree, and FEEDBACK_DRIFT_DETECTION config flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from quantstack.learning.drift_detector import (
    DriftDetector,
    ICDriftReport,
    InteractionDriftReport,
    LabelDriftReport,
    RetrainDecision,
    evaluate_retrain_decision,
)


# ===========================================================================
# Layer 1: IC-based Concept Drift
# ===========================================================================


class TestICDrift:
    """IC-based concept drift detection (daily check)."""

    def test_ic_drop_triggers_alert(self):
        """IC drops > 2 std below baseline -> drift alert for that feature."""
        detector = DriftDetector()
        current_ic = {"rsi_14": 0.02, "adx_14": 0.04}
        baseline_ic = {
            "rsi_14": (0.04, 0.005),  # mean=0.04, std=0.005
            "adx_14": (0.04, 0.005),
        }
        report = detector.check_ic_drift(current_ic, baseline_ic)
        assert isinstance(report, ICDriftReport)
        assert "rsi_14" in report.drifted_features
        # rsi_14 z = (0.04 - 0.02) / 0.005 = 4.0
        assert report.z_scores["rsi_14"] == pytest.approx(4.0, abs=0.01)
        # adx_14 z = 0.0 -> not drifted
        assert "adx_14" not in report.drifted_features

    def test_stable_ic_no_alert(self):
        """IC within 2 std of baseline -> no alert."""
        detector = DriftDetector()
        current_ic = {"rsi_14": 0.038}
        baseline_ic = {"rsi_14": (0.04, 0.005)}
        report = detector.check_ic_drift(current_ic, baseline_ic)
        assert len(report.drifted_features) == 0
        # z = (0.04 - 0.038) / 0.005 = 0.4
        assert report.z_scores["rsi_14"] == pytest.approx(0.4, abs=0.01)

    def test_detection_latency_within_5_days(self):
        """IC step change detected within 5 days using rolling window."""
        detector = DriftDetector()
        # Simulate 30-day rolling IC with step change at day 20
        ic_series = [0.04] * 20 + [0.01] * 10  # drops at day 20
        # Rolling window of 5 days: by day 25 the rolling mean is 0.01
        rolling_window = 5
        for day in range(20, min(25, len(ic_series))):
            window = ic_series[max(0, day - rolling_window + 1):day + 1]
            rolling_ic = np.mean(window)
            report = detector.check_ic_drift(
                {"rsi_14": rolling_ic},
                {"rsi_14": (0.04, 0.005)},
            )
            if report.drifted_features:
                detection_delay = day - 20
                assert detection_delay <= 5
                return
        # Should have detected by day 24
        pytest.fail("IC drift not detected within 5 days")


# ===========================================================================
# Layer 2: Label Drift (KS test)
# ===========================================================================


class TestLabelDrift:
    """Label drift via pure-numpy KS test (weekly check)."""

    def test_shifted_return_distribution_detected(self):
        """Shifted mean/std -> KS p < 0.01 -> label drift alert."""
        detector = DriftDetector()
        rng = np.random.default_rng(42)
        # Use larger shift and more samples for statistical power
        training_returns = rng.normal(0.001, 0.02, size=500)
        recent_returns = rng.normal(-0.01, 0.06, size=500)
        report = detector.check_label_drift(training_returns, recent_returns)
        assert isinstance(report, LabelDriftReport)
        assert report.is_drifted is True
        assert report.p_value < 0.01

    def test_stable_returns_no_alert(self):
        """Same distribution -> p >> 0.01 -> no alert."""
        detector = DriftDetector()
        rng = np.random.default_rng(42)
        training_returns = rng.normal(0.001, 0.02, size=252)
        recent_returns = rng.normal(0.001, 0.02, size=252)
        report = detector.check_label_drift(training_returns, recent_returns)
        assert report.is_drifted is False
        assert report.p_value > 0.01


# ===========================================================================
# Layer 3: Interaction Drift (Adversarial Validation)
# ===========================================================================


class TestInteractionDrift:
    """Adversarial validation for joint distribution shift (monthly)."""

    def test_shifted_joint_distribution_flagged(self):
        """Clearly different joint distributions -> AUC > 0.60 -> flagged."""
        detector = DriftDetector()
        rng = np.random.default_rng(42)
        # Training: features from N(0,1), returns near 0
        train_features = rng.normal(0, 1, size=(300, 3))
        train_returns = rng.normal(0.0, 0.02, size=300)
        training_data = np.column_stack([train_features, train_returns])
        # Recent: features shifted mean, returns shifted
        recent_features = rng.normal(1.0, 1, size=(300, 3))
        recent_returns = rng.normal(-0.05, 0.05, size=300)
        recent_data = np.column_stack([recent_features, recent_returns])

        report = detector.check_interaction_drift(training_data, recent_data)
        assert isinstance(report, InteractionDriftReport)
        assert report.is_drifted is True
        assert report.auc > 0.60

    def test_stable_joint_distribution_not_flagged(self):
        """Same distribution -> AUC ~ 0.50 -> not flagged."""
        detector = DriftDetector()
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, size=(400, 3))
        returns = rng.normal(0, 0.02, size=400)
        data = np.column_stack([features, returns])
        training_data = data[:200]
        recent_data = data[200:]

        report = detector.check_interaction_drift(training_data, recent_data)
        assert report.is_drifted is False
        assert report.auc < 0.60


# ===========================================================================
# Auto-Retrain Decision Tree
# ===========================================================================


class TestRetrainDecision:
    """Auto-retrain decision logic with cooldown."""

    def test_gradual_ic_decline_triggers_retrain(self):
        """IC declining over 60+ days with current IC < 0.01 -> retrain."""
        # Linearly declining IC over 80 days
        ic_history = [0.04 - i * 0.0005 for i in range(80)]
        current_ic = ic_history[-1]  # ~0.0
        decision = evaluate_retrain_decision(
            current_ic=current_ic,
            ic_history=ic_history,
            last_retrain_date=date.today() - timedelta(days=90),
            ic_drift_report=None,
        )
        assert isinstance(decision, RetrainDecision)
        assert decision.should_retrain is True
        assert decision.reason == "gradual_ic_decline"
        assert decision.data_window == 252

    def test_abrupt_ic_drop_no_retrain_publishes_event(self):
        """Abrupt IC step change -> no retrain, publish MODEL_DEGRADATION."""
        # Stable IC then sudden drop
        ic_history = [0.04] * 55 + [0.005] * 5
        current_ic = 0.005
        decision = evaluate_retrain_decision(
            current_ic=current_ic,
            ic_history=ic_history,
            last_retrain_date=date.today() - timedelta(days=90),
            ic_drift_report=None,
        )
        assert decision.should_retrain is False
        assert decision.reason == "abrupt_shift"
        assert decision.publish_event is True

    def test_cooldown_blocks_retrain_within_20_days(self):
        """Retrain requested within 20-day cooldown -> blocked."""
        ic_history = [0.04 - i * 0.0005 for i in range(80)]
        current_ic = ic_history[-1]
        # Last retrain 15 days ago
        decision = evaluate_retrain_decision(
            current_ic=current_ic,
            ic_history=ic_history,
            last_retrain_date=date.today() - timedelta(days=15),
            ic_drift_report=None,
        )
        assert decision.should_retrain is False
        assert decision.reason == "cooldown"

    def test_cooldown_expires_after_20_days(self):
        """After 20+ days, retrain allowed again."""
        ic_history = [0.04 - i * 0.0005 for i in range(80)]
        current_ic = ic_history[-1]
        decision = evaluate_retrain_decision(
            current_ic=current_ic,
            ic_history=ic_history,
            last_retrain_date=date.today() - timedelta(days=21),
            ic_drift_report=None,
        )
        assert decision.should_retrain is True

    def test_benign_covariate_shift(self):
        """Feature drift but IC still healthy -> no retrain."""
        ic_history = [0.04] * 60
        current_ic = 0.035  # healthy
        # Simulate IC drift report with drifted features
        ic_drift = ICDriftReport(
            z_scores={"rsi_14": 2.5},
            drifted_features=["rsi_14"],
        )
        decision = evaluate_retrain_decision(
            current_ic=current_ic,
            ic_history=ic_history,
            last_retrain_date=date.today() - timedelta(days=90),
            ic_drift_report=ic_drift,
        )
        assert decision.should_retrain is False
        assert decision.reason == "benign_covariate_shift"


# ===========================================================================
# Config Flag
# ===========================================================================


class TestDriftDetectionConfigFlag:
    """FEEDBACK_DRIFT_DETECTION kill switch."""

    def test_flag_false_skips_all_checks(self):
        """When flag is false, all drift layers return no-op results."""
        detector = DriftDetector()
        with patch.dict(os.environ, {"FEEDBACK_DRIFT_DETECTION": "false"}):
            ic_report = detector.check_ic_drift_gated(
                {"rsi_14": 0.01}, {"rsi_14": (0.04, 0.005)}
            )
            label_report = detector.check_label_drift_gated(
                np.array([0.01] * 100), np.array([-0.01] * 100)
            )
        assert len(ic_report.drifted_features) == 0
        assert label_report.is_drifted is False

    def test_flag_true_runs_checks(self):
        """When flag is true, checks run normally."""
        detector = DriftDetector()
        rng = np.random.default_rng(42)
        training = rng.normal(0.001, 0.02, size=500)
        recent = rng.normal(-0.01, 0.06, size=500)
        with patch.dict(os.environ, {"FEEDBACK_DRIFT_DETECTION": "true"}):
            label_report = detector.check_label_drift_gated(training, recent)
        assert label_report.is_drifted is True
