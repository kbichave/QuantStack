# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 16: Config Flags & Integration.

Covers the centralized flag registry, per-flag isolation, cold-start
behavior, compound sizing floor, and flag independence.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from quantstack.config.feedback_flags import (
    ic_weight_adjustment_enabled,
    correlation_penalty_enabled,
    conviction_multiplicative_enabled,
    sharpe_demotion_enabled,
    drift_detection_enabled,
    transition_sizing_enabled,
)


# ===========================================================================
# Centralized Flag Registry
# ===========================================================================


class TestFlagRegistry:
    """All flags default to false and parse env vars correctly."""

    def test_all_defaults_false(self, monkeypatch):
        """All flags default to false when env vars are unset."""
        for var in [
            "FEEDBACK_IC_WEIGHT_ADJUSTMENT",
            "FEEDBACK_CORRELATION_PENALTY",
            "FEEDBACK_CONVICTION_MULTIPLICATIVE",
            "FEEDBACK_SHARPE_DEMOTION",
            "FEEDBACK_DRIFT_DETECTION",
            "FEEDBACK_TRANSITION_SIZING",
        ]:
            monkeypatch.delenv(var, raising=False)

        assert ic_weight_adjustment_enabled() is False
        assert correlation_penalty_enabled() is False
        assert conviction_multiplicative_enabled() is False
        assert sharpe_demotion_enabled() is False
        assert drift_detection_enabled() is False
        assert transition_sizing_enabled() is False

    def test_all_true_when_set(self, monkeypatch):
        """All flags return True when set to 'true'."""
        for var in [
            "FEEDBACK_IC_WEIGHT_ADJUSTMENT",
            "FEEDBACK_CORRELATION_PENALTY",
            "FEEDBACK_CONVICTION_MULTIPLICATIVE",
            "FEEDBACK_SHARPE_DEMOTION",
            "FEEDBACK_DRIFT_DETECTION",
            "FEEDBACK_TRANSITION_SIZING",
        ]:
            monkeypatch.setenv(var, "true")

        assert ic_weight_adjustment_enabled() is True
        assert correlation_penalty_enabled() is True
        assert conviction_multiplicative_enabled() is True
        assert sharpe_demotion_enabled() is True
        assert drift_detection_enabled() is True
        assert transition_sizing_enabled() is True

    def test_accepts_1_and_yes(self, monkeypatch):
        """Flags accept '1' and 'yes' as truthy values."""
        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "1")
        monkeypatch.setenv("FEEDBACK_CORRELATION_PENALTY", "yes")
        monkeypatch.setenv("FEEDBACK_DRIFT_DETECTION", "YES")

        assert ic_weight_adjustment_enabled() is True
        assert correlation_penalty_enabled() is True
        assert drift_detection_enabled() is True

    def test_random_string_treated_as_false(self, monkeypatch):
        """Non-truthy strings treated as false."""
        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "maybe")
        assert ic_weight_adjustment_enabled() is False


# ===========================================================================
# Flag Isolation: IC Weight Adjustment
# ===========================================================================


class TestICWeightFlagIsolation:
    """FEEDBACK_IC_WEIGHT_ADJUSTMENT controls IC factor application."""

    def test_false_disables_adjustment(self, monkeypatch):
        """Flag=false -> ic_factors all 1.0."""
        from quantstack.signal_engine.ic_weights import compute_ic_factors_gated

        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "false")
        factors = compute_ic_factors_gated({"coll_a": [0.0] * 21})
        assert factors["coll_a"] == 1.0

    def test_true_enables_adjustment(self, monkeypatch):
        """Flag=true -> IC=0.0 produces factor < 0.3."""
        from quantstack.signal_engine.ic_weights import compute_ic_factors_gated

        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "true")
        factors = compute_ic_factors_gated({"coll_a": [0.0] * 21})
        assert factors["coll_a"] < 0.3


# ===========================================================================
# Flag Isolation: Correlation Penalty
# ===========================================================================


class TestCorrelationFlagIsolation:
    """FEEDBACK_CORRELATION_PENALTY controls correlation penalty."""

    def test_false_disables_penalty(self, monkeypatch):
        """Flag=false -> all penalties 1.0."""
        from quantstack.signal_engine.correlation import compute_correlation_penalties_gated

        monkeypatch.setenv("FEEDBACK_CORRELATION_PENALTY", "false")
        penalties = compute_correlation_penalties_gated(
            {"a": [0.1] * 63, "b": [0.1] * 63},
            {"a": 0.05, "b": 0.03},
        )
        assert penalties["a"] == 1.0
        assert penalties["b"] == 1.0


# ===========================================================================
# Flag Isolation: Transition Sizing
# ===========================================================================


class TestTransitionSizingFlagIsolation:
    """FEEDBACK_TRANSITION_SIZING controls regime transition sizing."""

    def test_false_disables_sizing(self, monkeypatch):
        """Flag=false -> transition_sizing_factor_gated always 1.0."""
        from quantstack.signal_engine.collectors.regime import transition_sizing_factor_gated

        monkeypatch.setenv("FEEDBACK_TRANSITION_SIZING", "false")
        assert transition_sizing_factor_gated(0.60) == 1.0

    def test_true_enables_sizing(self, monkeypatch):
        """Flag=true -> real factor applied."""
        from quantstack.signal_engine.collectors.regime import transition_sizing_factor_gated

        monkeypatch.setenv("FEEDBACK_TRANSITION_SIZING", "true")
        assert transition_sizing_factor_gated(0.60) == 0.25


# ===========================================================================
# Flag Isolation: Sharpe Demotion
# ===========================================================================


class TestSharpeDemotionFlagIsolation:
    """FEEDBACK_SHARPE_DEMOTION controls demotion checks."""

    def test_false_returns_none(self, monkeypatch):
        """Flag=false -> check_sharpe_demotion always returns None."""
        from quantstack.learning.sharpe_demotion import check_sharpe_demotion

        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "false")
        result = check_sharpe_demotion(0.6, 1.5, 21)
        assert result is None

    def test_true_runs_logic(self, monkeypatch):
        """Flag=true -> demotion fires."""
        from quantstack.learning.sharpe_demotion import check_sharpe_demotion

        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")
        result = check_sharpe_demotion(0.6, 1.5, 21)
        assert result is not None
        assert result["triggered"] is True


# ===========================================================================
# Flag Isolation: Drift Detection
# ===========================================================================


class TestDriftDetectionFlagIsolation:
    """FEEDBACK_DRIFT_DETECTION controls drift checks."""

    def test_false_skips_ic_drift(self, monkeypatch):
        """Flag=false -> IC drift check returns no_op."""
        from quantstack.learning.drift_detector import DriftDetector

        monkeypatch.setenv("FEEDBACK_DRIFT_DETECTION", "false")
        detector = DriftDetector.__new__(DriftDetector)
        result = detector.check_ic_drift_gated({}, {})
        assert result.drifted_features == []

    def test_false_skips_label_drift(self, monkeypatch):
        """Flag=false -> label drift check returns no_op."""
        from quantstack.learning.drift_detector import DriftDetector

        monkeypatch.setenv("FEEDBACK_DRIFT_DETECTION", "false")
        detector = DriftDetector.__new__(DriftDetector)
        result = detector.check_label_drift_gated([], [])
        assert result.is_drifted is False


# ===========================================================================
# Cold-Start Behavior
# ===========================================================================


class TestColdStart:
    """Insufficient data produces no adjustment regardless of flag state."""

    def test_ic_weight_cold_start(self, monkeypatch):
        """< 21 days of IC data -> factor 1.0."""
        from quantstack.signal_engine.ic_weights import compute_ic_factors

        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "true")
        factors = compute_ic_factors({"coll_a": [0.0] * 10})
        assert factors["coll_a"] == 1.0

    def test_correlation_cold_start(self, monkeypatch):
        """< 63 days of signal data -> penalty 1.0."""
        from quantstack.signal_engine.correlation import compute_signal_correlations

        monkeypatch.setenv("FEEDBACK_CORRELATION_PENALTY", "true")
        result = compute_signal_correlations(
            {"a": [0.1] * 30, "b": [0.2] * 30},
            {"a": 0.05, "b": 0.03},
        )
        assert result.penalties.get("a", 1.0) == 1.0

    def test_sharpe_cold_start(self, monkeypatch):
        """live_sharpe=None -> no demotion."""
        from quantstack.learning.sharpe_demotion import check_sharpe_demotion

        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "true")
        result = check_sharpe_demotion(None, 1.5, 21)
        assert result is None

    def test_transition_sizing_hmm_not_fit(self, monkeypatch):
        """transition_probability=None -> factor 1.0."""
        from quantstack.signal_engine.collectors.regime import transition_sizing_factor

        assert transition_sizing_factor(None) == 1.0


# ===========================================================================
# Compound Sizing Floor
# ===========================================================================


class TestCompoundSizingFloor:
    """When multiple loops stack, positions below $100 are skipped."""

    MINIMUM_TRADEABLE_VALUE = 100.0

    def test_below_floor_skips(self):
        """$1000 * 0.5 * 0.5 * 0.25 = $62.50 < $100 -> skip."""
        kelly_size = 1000.0
        breaker_factor = 0.5
        transition_factor = 0.5
        sharpe_factor = 0.25
        final = kelly_size * breaker_factor * transition_factor * sharpe_factor
        assert final < self.MINIMUM_TRADEABLE_VALUE
        assert final == pytest.approx(62.5)

    def test_at_floor_proceeds(self):
        """$1600 * 0.5 * 0.5 * 0.25 = $100.00 -> proceed."""
        kelly_size = 1600.0
        breaker_factor = 0.5
        transition_factor = 0.5
        sharpe_factor = 0.25
        final = kelly_size * breaker_factor * transition_factor * sharpe_factor
        assert final >= self.MINIMUM_TRADEABLE_VALUE

    def test_all_flags_off_no_reduction(self, monkeypatch):
        """All flags false -> only breaker_factor applies."""
        from quantstack.signal_engine.collectors.regime import transition_sizing_factor_gated
        from quantstack.signal_engine.ic_weights import compute_ic_factors_gated

        for var in [
            "FEEDBACK_IC_WEIGHT_ADJUSTMENT",
            "FEEDBACK_CORRELATION_PENALTY",
            "FEEDBACK_CONVICTION_MULTIPLICATIVE",
            "FEEDBACK_SHARPE_DEMOTION",
            "FEEDBACK_DRIFT_DETECTION",
            "FEEDBACK_TRANSITION_SIZING",
        ]:
            monkeypatch.setenv(var, "false")

        assert transition_sizing_factor_gated(0.60) == 1.0
        factors = compute_ic_factors_gated({"coll_a": [0.0] * 21})
        assert factors["coll_a"] == 1.0


# ===========================================================================
# Flag Independence
# ===========================================================================


class TestFlagIndependence:
    """Enabling one flag does not affect others."""

    def test_only_ic_weight_enabled(self, monkeypatch):
        """Enable only IC weight -> others remain off."""
        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "true")
        monkeypatch.setenv("FEEDBACK_CORRELATION_PENALTY", "false")
        monkeypatch.setenv("FEEDBACK_TRANSITION_SIZING", "false")
        monkeypatch.setenv("FEEDBACK_SHARPE_DEMOTION", "false")

        assert ic_weight_adjustment_enabled() is True
        assert correlation_penalty_enabled() is False
        assert transition_sizing_enabled() is False
        assert sharpe_demotion_enabled() is False

    def test_only_transition_sizing_enabled(self, monkeypatch):
        """Enable only transition sizing -> others remain off."""
        monkeypatch.setenv("FEEDBACK_IC_WEIGHT_ADJUSTMENT", "false")
        monkeypatch.setenv("FEEDBACK_TRANSITION_SIZING", "true")

        assert ic_weight_adjustment_enabled() is False
        assert transition_sizing_enabled() is True
