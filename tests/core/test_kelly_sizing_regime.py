"""Tests for regime-conditional Kelly sizing (section-02)."""

import pytest

from quantstack.core.kelly_sizing import (
    CONFIDENCE_FULL_REGIME_THRESHOLD,
    REGIME_KELLY_TABLE,
    UNKNOWN_KELLY,
    regime_kelly_fraction,
)


class TestRegimeKellyFraction:
    """Tests for the regime_kelly_fraction() lookup + interpolation."""

    # --- Full confidence (1.0) returns exact table values ---

    def test_trending_up_normal(self):
        assert regime_kelly_fraction("trending_up", "normal", 1.0) == 0.50

    def test_trending_up_high(self):
        assert regime_kelly_fraction("trending_up", "high", 1.0) == 0.35

    def test_trending_down_normal(self):
        assert regime_kelly_fraction("trending_down", "normal", 1.0) == 0.20

    def test_trending_down_high(self):
        assert regime_kelly_fraction("trending_down", "high", 1.0) == 0.20

    def test_ranging_normal(self):
        assert regime_kelly_fraction("ranging", "normal", 1.0) == 0.35

    def test_ranging_high(self):
        assert regime_kelly_fraction("ranging", "high", 1.0) == 0.20

    def test_unknown_normal(self):
        assert regime_kelly_fraction("unknown", "normal", 1.0) == 0.15

    def test_unknown_high(self):
        assert regime_kelly_fraction("unknown", "high", 1.0) == 0.15

    # --- Confidence interpolation ---

    def test_low_confidence_interpolates_toward_unknown(self):
        """At confidence 0.6, trending_up+normal is between UNKNOWN_KELLY and 0.50."""
        result = regime_kelly_fraction("trending_up", "normal", 0.6)
        assert UNKNOWN_KELLY < result < 0.50

    def test_zero_confidence_returns_unknown(self):
        """At confidence 0.0, all regimes return UNKNOWN_KELLY."""
        for regime in ("trending_up", "trending_down", "ranging", "unknown"):
            for vol in ("normal", "high"):
                assert regime_kelly_fraction(regime, vol, 0.0) == UNKNOWN_KELLY

    def test_threshold_confidence_returns_full_multiplier(self):
        """At exactly CONFIDENCE_FULL_REGIME_THRESHOLD, no interpolation occurs."""
        assert regime_kelly_fraction("trending_up", "normal", 0.8) == 0.50
        assert regime_kelly_fraction("ranging", "high", 0.8) == 0.20

    def test_interpolation_is_linear(self):
        """Midpoint confidence produces midpoint between UNKNOWN_KELLY and table value."""
        # confidence = 0.4 → t = 0.4/0.8 = 0.5 → halfway between 0.15 and 0.50 = 0.325
        result = regime_kelly_fraction("trending_up", "normal", 0.4)
        expected = UNKNOWN_KELLY + (0.50 - UNKNOWN_KELLY) * 0.5
        assert abs(result - expected) < 1e-10

    # --- Unknown/unrecognized regime fallback ---

    def test_unrecognized_regime_returns_unknown_kelly(self):
        """Unrecognized regime string falls back to unknown, not an exception."""
        assert regime_kelly_fraction("sideways", "normal", 1.0) == UNKNOWN_KELLY
        assert regime_kelly_fraction("", "normal", 1.0) == UNKNOWN_KELLY

    def test_unrecognized_vol_state_returns_unknown_kelly(self):
        """Unrecognized vol_state falls back to UNKNOWN_KELLY."""
        result = regime_kelly_fraction("trending_up", "extreme", 1.0)
        assert result == UNKNOWN_KELLY

    # --- Property: output always in [UNKNOWN_KELLY, 0.50] ---

    @pytest.mark.parametrize("regime", ["trending_up", "trending_down", "ranging", "unknown"])
    @pytest.mark.parametrize("vol", ["normal", "high"])
    @pytest.mark.parametrize("conf", [0.0, 0.3, 0.5, 0.79, 0.8, 0.9, 1.0])
    def test_output_bounded(self, regime, vol, conf):
        result = regime_kelly_fraction(regime, vol, conf)
        assert UNKNOWN_KELLY <= result <= 0.50
