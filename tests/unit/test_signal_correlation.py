# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 08: Signal Correlation Tracking.

Covers pairwise Spearman correlation, continuous penalty formula,
weaker-signal-gets-penalty logic, effective signal count, config flag,
and cold-start behavior.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from quantstack.signal_engine.correlation import (
    compute_signal_correlations,
    correlation_penalty,
    compute_correlation_penalties_gated,
)


# ===========================================================================
# Correlation Matrix
# ===========================================================================


class TestCorrelationMatrix:
    """Pairwise Spearman correlation across collectors."""

    def test_pairwise_spearman_computed(self):
        """Known correlated signals produce high correlation."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(63)
        signal_data = {
            "coll_a": base.tolist(),
            "coll_b": (base + rng.normal(0, 0.1, 63)).tolist(),  # highly correlated
            "coll_c": rng.standard_normal(63).tolist(),  # independent
        }
        ic_data = {"coll_a": 0.05, "coll_b": 0.03, "coll_c": 0.04}
        result = compute_signal_correlations(signal_data, ic_data)
        # a-b should be highly correlated (> 0.8)
        assert result.correlation_matrix[("coll_a", "coll_b")] > 0.8
        # a-c should be low correlation
        assert abs(result.correlation_matrix[("coll_a", "coll_c")]) < 0.5

    def test_effective_independent_signal_count(self):
        """Two groups of perfectly correlated signals -> 2 independent signals."""
        # Group 1: a and b are identical
        # Group 2: c is independent
        rng = np.random.default_rng(42)
        base1 = rng.standard_normal(63).tolist()
        base2 = rng.standard_normal(63).tolist()
        signal_data = {
            "coll_a": base1,
            "coll_b": base1,  # identical to a
            "coll_c": base2,  # independent
        }
        ic_data = {"coll_a": 0.05, "coll_b": 0.03, "coll_c": 0.04}
        result = compute_signal_correlations(signal_data, ic_data)
        # Effective count should be ~2 (not 3)
        assert result.effective_signal_count <= 2

    def test_insufficient_data_identity(self):
        """< 63 days -> no correlations detected."""
        signal_data = {
            "coll_a": [0.1] * 30,  # too few
            "coll_b": [0.2] * 30,
        }
        ic_data = {"coll_a": 0.05, "coll_b": 0.03}
        result = compute_signal_correlations(signal_data, ic_data)
        # No penalties applied
        assert result.penalties.get("coll_a", 1.0) == 1.0
        assert result.penalties.get("coll_b", 1.0) == 1.0


# ===========================================================================
# Continuous Correlation Penalty
# ===========================================================================


class TestCorrelationPenalty:
    """penalty = max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)"""

    def test_low_correlation_no_penalty(self):
        """corr=0.4 -> penalty = 1.0."""
        assert correlation_penalty(0.4) == 1.0

    def test_moderate_correlation_partial_penalty(self):
        """corr=0.6 -> penalty = 0.8."""
        assert correlation_penalty(0.6) == pytest.approx(0.8, abs=0.001)

    def test_high_correlation_strong_penalty(self):
        """corr=0.8 -> penalty = 0.4."""
        assert correlation_penalty(0.8) == pytest.approx(0.4, abs=0.001)

    def test_very_high_correlation_hits_floor(self):
        """corr=0.95 -> penalty = 0.2 (floor)."""
        assert correlation_penalty(0.95) == pytest.approx(0.2, abs=0.001)

    def test_weaker_signal_gets_penalty(self):
        """Weaker IC collector gets penalized, stronger keeps 1.0."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(63)
        signal_data = {
            "strong": base.tolist(),
            "weak": (base + rng.normal(0, 0.1, 63)).tolist(),
        }
        # strong has higher IC
        ic_data = {"strong": 0.05, "weak": 0.02}
        result = compute_signal_correlations(signal_data, ic_data)
        # weak should be penalized, strong should not
        assert result.penalties.get("strong", 1.0) == 1.0
        assert result.penalties.get("weak", 1.0) < 1.0

    def test_penalty_symmetric_for_equal_ic(self):
        """Equal IC -> both get penalized."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(63)
        signal_data = {
            "a": base.tolist(),
            "b": (base + rng.normal(0, 0.1, 63)).tolist(),
        }
        ic_data = {"a": 0.05, "b": 0.05}  # equal IC
        result = compute_signal_correlations(signal_data, ic_data)
        # Both should be penalized
        assert result.penalties.get("a", 1.0) < 1.0
        assert result.penalties.get("b", 1.0) < 1.0


# ===========================================================================
# Config Flag
# ===========================================================================


class TestCorrelationConfigFlag:
    """FEEDBACK_CORRELATION_PENALTY env var."""

    def test_flag_false_penalty_always_one(self):
        """Flag=false -> all penalties 1.0."""
        with patch.dict(os.environ, {"FEEDBACK_CORRELATION_PENALTY": "false"}):
            penalties = compute_correlation_penalties_gated(
                {"a": [0.1] * 63, "b": [0.1] * 63},
                {"a": 0.05, "b": 0.03},
            )
        assert penalties.get("a", 1.0) == 1.0
        assert penalties.get("b", 1.0) == 1.0

    def test_flag_true_penalties_applied(self):
        """Flag=true -> computed penalties flow through."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(63)
        with patch.dict(os.environ, {"FEEDBACK_CORRELATION_PENALTY": "true"}):
            penalties = compute_correlation_penalties_gated(
                {"a": base.tolist(), "b": (base + rng.normal(0, 0.1, 63)).tolist()},
                {"a": 0.05, "b": 0.02},
            )
        # b (weaker) should be penalized
        assert penalties.get("b", 1.0) < 1.0
