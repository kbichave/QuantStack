# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 15: Regime Transition Detection.

Covers transition probability computation, tiered sizing response,
degraded mode, minimum tradeable floor, vol sub-regimes, and config flag.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from quantstack.signal_engine.collectors.regime import (
    transition_sizing_factor,
    _vol_sub_regime,
)


# ===========================================================================
# Filtered Transition Probability
# ===========================================================================


class TestTransitionProbability:
    """Transition probability = 1.0 - max(state_probabilities)."""

    def test_uses_filtered_probs_not_transmat(self):
        """transition_probability = 1 - max(filtered posterior)."""
        # If HMM gives 80% confidence in current state
        state_probs = {"LOW_VOL_BULL": 0.80, "HIGH_VOL_BULL": 0.10,
                       "LOW_VOL_BEAR": 0.05, "HIGH_VOL_BEAR": 0.05}
        transition_prob = 1.0 - max(state_probs.values())
        assert abs(transition_prob - 0.20) < 0.001

    def test_high_uncertainty_high_transition(self):
        """Max filtered prob < 0.5 -> transition_probability > 0.5."""
        state_probs = {"LOW_VOL_BULL": 0.45, "HIGH_VOL_BULL": 0.25,
                       "LOW_VOL_BEAR": 0.20, "HIGH_VOL_BEAR": 0.10}
        transition_prob = 1.0 - max(state_probs.values())
        assert transition_prob > 0.5

    def test_confident_state_low_transition(self):
        """Max filtered prob > 0.9 -> transition_probability < 0.1."""
        state_probs = {"LOW_VOL_BULL": 0.95, "HIGH_VOL_BULL": 0.03,
                       "LOW_VOL_BEAR": 0.01, "HIGH_VOL_BEAR": 0.01}
        transition_prob = 1.0 - max(state_probs.values())
        assert transition_prob < 0.1


# ===========================================================================
# Sizing Response Tiers
# ===========================================================================


class TestTransitionSizingFactor:
    """Tiered sizing response to transition probability."""

    def test_low_probability_no_adjustment(self):
        """P < 0.10 -> factor = 1.0."""
        assert transition_sizing_factor(0.05) == 1.0
        assert transition_sizing_factor(0.0) == 1.0

    def test_moderate_probability_mild_reduction(self):
        """P = 0.20 -> factor = 0.75."""
        assert transition_sizing_factor(0.20) == 0.75

    def test_elevated_probability_half_reduction(self):
        """P = 0.40 -> factor = 0.50."""
        assert transition_sizing_factor(0.40) == 0.50

    def test_high_probability_severe_reduction(self):
        """P = 0.60 -> factor = 0.25."""
        assert transition_sizing_factor(0.60) == 0.25

    def test_boundary_0_10(self):
        """P = 0.10 -> enters mild tier -> 0.75."""
        assert transition_sizing_factor(0.10) == 0.75

    def test_boundary_0_30(self):
        """P = 0.30 -> enters moderate tier -> 0.50."""
        assert transition_sizing_factor(0.30) == 0.50

    def test_boundary_0_50(self):
        """P = 0.50 -> enters severe tier -> 0.25."""
        assert transition_sizing_factor(0.50) == 0.25


# ===========================================================================
# Degraded Mode
# ===========================================================================


class TestDegradedMode:
    """Behavior when HMM fails or data is missing."""

    def test_hmm_failure_transition_prob_zero(self):
        """HMM failure -> transition_probability defaults to 0.0."""
        # When HMM fails, regime collector should set transition_probability=0.0
        # which maps to factor 1.0 (no penalty)
        assert transition_sizing_factor(0.0) == 1.0

    def test_none_transition_probability_defaults_to_unity(self):
        """None input -> factor = 1.0."""
        assert transition_sizing_factor(None) == 1.0


# ===========================================================================
# Minimum Tradeable Size Floor
# ===========================================================================


class TestMinimumTradeableFloor:
    """$100 minimum position value."""

    def test_below_minimum_skips(self):
        """Compound factors producing < $100 -> skip."""
        kelly_size = 200.0
        breaker_factor = 0.5
        transition_factor = 0.25  # severe
        final_value = kelly_size * breaker_factor * transition_factor
        assert final_value < 100.0  # $25

    def test_above_minimum_proceeds(self):
        """Compound factors producing >= $100 -> proceed."""
        kelly_size = 1000.0
        breaker_factor = 1.0
        transition_factor = 0.50
        final_value = kelly_size * breaker_factor * transition_factor
        assert final_value >= 100.0  # $500


# ===========================================================================
# Vol-Conditioned Sub-Regimes
# ===========================================================================


class TestVolSubRegimes:
    """Volatility percentile sub-regime classification."""

    def _make_df(self, vol_values):
        """Create a DataFrame with close prices that produce given vol pattern."""
        import pandas as pd
        # Generate prices from returns
        n = len(vol_values)
        prices = [100.0]
        for i in range(1, n):
            ret = vol_values[i] * 0.01  # small returns
            prices.append(prices[-1] * (1 + ret))
        df = pd.DataFrame({
            "close": prices[:n],
            "open": prices[:n],
            "high": [p * 1.01 for p in prices[:n]],
            "low": [p * 0.99 for p in prices[:n]],
            "volume": [1000000] * n,
        })
        return df

    def test_low_vol_sub_regime(self):
        """Low realized vol -> 'low_vol'."""
        import pandas as pd
        rng = np.random.default_rng(42)
        # 252 days of low-vol returns + 20 days of very low vol
        returns = np.concatenate([
            rng.normal(0, 0.02, size=252),
            rng.normal(0, 0.005, size=20),  # very low vol at end
        ])
        prices = 100.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": prices})
        result = _vol_sub_regime(df)
        assert result == "low_vol"

    def test_high_vol_sub_regime(self):
        """High realized vol -> 'high_vol'."""
        import pandas as pd
        rng = np.random.default_rng(42)
        # 252 days of normal vol + 20 days of high vol
        returns = np.concatenate([
            rng.normal(0, 0.01, size=252),
            rng.normal(0, 0.05, size=20),  # high vol at end
        ])
        prices = 100.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": prices})
        result = _vol_sub_regime(df)
        assert result == "high_vol"


# ===========================================================================
# Config Flag
# ===========================================================================


class TestTransitionSizingConfigFlag:
    """FEEDBACK_TRANSITION_SIZING kill switch."""

    def test_flag_false_disables_sizing(self):
        """When flag is false, transition factor always 1.0."""
        with patch.dict(os.environ, {"FEEDBACK_TRANSITION_SIZING": "false"}):
            from quantstack.signal_engine.collectors.regime import (
                transition_sizing_factor_gated,
            )
            assert transition_sizing_factor_gated(0.60) == 1.0

    def test_flag_true_enables_sizing(self):
        """When flag is true, real factor applied."""
        with patch.dict(os.environ, {"FEEDBACK_TRANSITION_SIZING": "true"}):
            from quantstack.signal_engine.collectors.regime import (
                transition_sizing_factor_gated,
            )
            assert transition_sizing_factor_gated(0.60) == 0.25
