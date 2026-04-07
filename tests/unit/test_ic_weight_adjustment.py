# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 07: IC Degradation to Weight Adjustment.

Covers sigmoid IC factor, IC_IR penalty, weight floor safety check,
config flag, and cold-start behavior.
"""

from __future__ import annotations

import math
import os
from unittest.mock import patch

import pytest

from quantstack.signal_engine.ic_weights import (
    ic_factor,
    compute_ic_factors,
    check_weight_floor,
)


# ===========================================================================
# Sigmoid IC Factor Function
# ===========================================================================


class TestICFactorFunction:
    """Continuous sigmoid IC factor: 1 / (1 + exp(-50 * (ic - 0.02)))"""

    def test_healthy_ic_high_weight(self):
        """IC=0.05 should produce factor > 0.8 (sigmoid at +1.5 std)."""
        f = ic_factor(0.05)
        assert f > 0.8, f"Expected > 0.8, got {f}"

    def test_very_healthy_ic_full_weight(self):
        """IC=0.10 should produce factor > 0.98."""
        f = ic_factor(0.10)
        assert f > 0.98, f"Expected ~1.0, got {f}"

    def test_threshold_ic_half_weight(self):
        """IC=0.02 (sigmoid center) should produce factor approximately 0.5."""
        f = ic_factor(0.02)
        assert abs(f - 0.5) < 0.01, f"Expected ~0.5, got {f}"

    def test_zero_ic_near_zero_weight(self):
        """IC=0.00 should produce factor near 0.0."""
        f = ic_factor(0.00)
        assert f < 0.3, f"Expected near 0.0, got {f}"

    def test_negative_ic_very_low_weight(self):
        """IC=-0.02 should produce factor < 0.15."""
        f = ic_factor(-0.02)
        assert f < 0.15, f"Expected low factor, got {f}"

    def test_deeply_negative_ic_near_zero(self):
        """IC=-0.05 should produce factor < 0.03."""
        f = ic_factor(-0.05)
        assert f < 0.03, f"Expected ~0.0, got {f}"

    def test_smooth_transition_no_discrete_jumps(self):
        """Verify monotonic increase across IC range with no jumps > 0.1."""
        prev = ic_factor(-0.01)
        for ic_val in [i * 0.001 for i in range(-9, 51)]:
            curr = ic_factor(ic_val)
            assert curr >= prev - 1e-9, f"Non-monotonic at IC={ic_val}"
            assert curr - prev < 0.1, f"Jump > 0.1 at IC={ic_val}: {prev:.4f} -> {curr:.4f}"
            prev = curr


# ===========================================================================
# IC_IR Penalty
# ===========================================================================


class TestICIRPenalty:
    """When IC_IR < 0.1, apply 0.7x penalty for inconsistency."""

    def test_low_icir_applies_penalty(self):
        """IC_IR < 0.1 should multiply ic_factor by 0.7."""
        # Use 21 values with mean=0.03 and std=0.5 exactly.
        # 20 alternating pairs: +0.53, -0.47 (mean=0.03 per pair, std=0.5)
        # Plus one final 0.03 to hit 21 values.
        noisy = []
        for _ in range(10):
            noisy.extend([0.53, -0.47])
        noisy.append(0.03)
        # mean = (10*0.53 + 10*(-0.47) + 0.03) / 21 = (5.3 - 4.7 + 0.03) / 21 = 0.63/21 = 0.03
        # std ~ 0.5 -> IC_IR = 0.03 / 0.5 = 0.06 < 0.1
        mean_val = sum(noisy) / len(noisy)
        factors = compute_ic_factors({"noisy": noisy})
        base = ic_factor(mean_val)
        # Should be base * 0.7 (penalty applied)
        assert factors["noisy"] == pytest.approx(base * 0.7, abs=0.02)

    def test_healthy_icir_no_penalty(self):
        """IC_IR >= 0.1 should not apply penalty."""
        # Consistent IC=0.05 with low std
        consistent = [0.05] * 21
        factors = compute_ic_factors({"stable_coll": consistent})
        base = ic_factor(0.05)
        # With zero std, IC_IR is handled as inf or capped — no penalty
        assert factors["stable_coll"] == pytest.approx(base, abs=0.01)


# ===========================================================================
# Weight Floor Check
# ===========================================================================


class TestWeightFloorCheck:
    """Total effective weight < 0.1 triggers fallback to static weights."""

    def test_all_collectors_near_zero_triggers_fallback(self):
        """When every collector has IC near zero, fall back to static weights."""
        static_weights = {"coll_a": 0.3, "coll_b": 0.3, "coll_c": 0.4}
        ic_factors = {"coll_a": 0.01, "coll_b": 0.02, "coll_c": 0.01}
        result = check_weight_floor(static_weights, ic_factors)
        assert result["floor_triggered"] is True
        # Effective weights should equal static weights (fallback)
        assert result["effective_weights"] == static_weights

    def test_partial_degradation_no_fallback(self):
        """Some healthy collectors -> no fallback."""
        static_weights = {"coll_a": 0.3, "coll_b": 0.3, "coll_c": 0.4}
        ic_factors = {"coll_a": 0.95, "coll_b": 0.02, "coll_c": 0.90}
        result = check_weight_floor(static_weights, ic_factors)
        assert result["floor_triggered"] is False
        # coll_b should have reduced weight
        assert result["effective_weights"]["coll_b"] < static_weights["coll_b"]


# ===========================================================================
# Config Flag
# ===========================================================================


class TestICWeightConfigFlag:
    """FEEDBACK_IC_WEIGHT_ADJUSTMENT env var controls IC factor application."""

    def test_flag_false_ic_factor_always_one(self):
        """With flag=false, compute_ic_factors_gated returns all 1.0."""
        from quantstack.signal_engine.ic_weights import compute_ic_factors_gated

        with patch.dict(os.environ, {"FEEDBACK_IC_WEIGHT_ADJUSTMENT": "false"}):
            factors = compute_ic_factors_gated({"coll_a": [0.0] * 21})
        assert factors["coll_a"] == 1.0

    def test_flag_true_ic_factors_applied(self):
        """With flag=true, compute_ic_factors_gated returns computed values."""
        from quantstack.signal_engine.ic_weights import compute_ic_factors_gated

        with patch.dict(os.environ, {"FEEDBACK_IC_WEIGHT_ADJUSTMENT": "true"}):
            factors = compute_ic_factors_gated({"coll_a": [0.0] * 21})
        # IC=0.0 -> factor near 0
        assert factors["coll_a"] < 0.3


# ===========================================================================
# Cold-Start
# ===========================================================================


class TestICWeightColdStart:
    """Behavior when insufficient IC data exists."""

    def test_fewer_than_21_days_returns_one(self):
        """With < 21 days of IC data, factor defaults to 1.0."""
        factors = compute_ic_factors({"coll_a": [0.0] * 10})
        assert factors["coll_a"] == 1.0
