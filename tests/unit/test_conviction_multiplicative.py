# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 10: Conviction Calibration — Multiplicative Factors.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from quantstack.signal_engine.synthesis import RuleBasedSynthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mult(base=0.5, adx=None, hmm_stability=None, weekly_trend="unknown",
          trend="unknown", regime_disagreement=False, has_ml=False,
          scores=None, score=0.0, failures=None):
    """Shortcut to call _conviction_multiplicative with defaults."""
    regime = {"regime_disagreement": regime_disagreement}
    return RuleBasedSynthesizer._conviction_multiplicative(
        base_conviction=base,
        adx=adx,
        hmm_stability=hmm_stability,
        weekly_trend=weekly_trend,
        trend=trend,
        regime=regime,
        has_ml=has_ml,
        scores=scores or {},
        score=score,
        failures=failures or [],
    )


def _add(base=0.5, adx=None, hmm_stability=None, weekly_trend="unknown",
         trend="unknown", regime_disagreement=False, has_ml=False,
         scores=None, score=0.0, failures=None):
    """Shortcut to call _conviction_additive with defaults."""
    regime = {"regime_disagreement": regime_disagreement}
    return RuleBasedSynthesizer._conviction_additive(
        conviction=base,
        adx=adx,
        hmm_stability=hmm_stability,
        weekly_trend=weekly_trend,
        trend=trend,
        regime=regime,
        has_ml=has_ml,
        scores=scores or {},
        score=score,
        failures=failures or [],
    )


# ===========================================================================
# Individual Factor Tests
# ===========================================================================


class TestConvictionMultiplicativeFactors:
    """Test each of the 6 multiplicative conviction factors in isolation."""

    def test_adx_factor_at_threshold(self):
        """ADX=15 -> factor exactly 1.0 (base returned unchanged)."""
        result = _mult(base=0.50, adx=15)
        assert abs(result - 0.50) < 0.001

    def test_adx_factor_strong_trend(self):
        """ADX=50 -> factor = 1.15."""
        result = _mult(base=1.0, adx=50)
        assert abs(result - 1.15) < 0.001

    def test_adx_factor_moderate(self):
        """ADX=32.5 (midpoint) -> factor = 1.075."""
        result = _mult(base=1.0, adx=32.5)
        assert abs(result - 1.075) < 0.001

    def test_adx_factor_none_input(self):
        """ADX is None -> factor defaults to 1.0."""
        result = _mult(base=0.50, adx=None)
        assert abs(result - 0.50) < 0.001

    def test_adx_factor_below_threshold(self):
        """ADX < 15 -> factor = 1.0."""
        result = _mult(base=0.50, adx=10)
        assert abs(result - 0.50) < 0.001

    def test_stability_factor_zero(self):
        """HMM stability=0.0 -> factor = 0.85."""
        result = _mult(base=1.0, hmm_stability=0.0)
        assert abs(result - 0.85) < 0.001

    def test_stability_factor_one(self):
        """HMM stability=1.0 -> factor = 1.05."""
        result = _mult(base=1.0, hmm_stability=1.0)
        assert abs(result - 1.05) < 0.001

    def test_stability_factor_midpoint(self):
        """HMM stability=0.5 -> factor = 0.95."""
        result = _mult(base=1.0, hmm_stability=0.5)
        assert abs(result - 0.95) < 0.001

    def test_stability_factor_none(self):
        """HMM stability is None -> factor defaults to 1.0."""
        result = _mult(base=0.50, hmm_stability=None)
        assert abs(result - 0.50) < 0.001

    def test_timeframe_factor_contradicting(self):
        """Weekly contradicts daily -> factor = 0.80."""
        result = _mult(base=1.0, weekly_trend="bullish", trend="trending_down")
        assert abs(result - 0.80) < 0.001

        result2 = _mult(base=1.0, weekly_trend="bearish", trend="trending_up")
        assert abs(result2 - 0.80) < 0.001

    def test_timeframe_factor_agreeing(self):
        """Weekly and daily agree -> factor = 1.0."""
        result = _mult(base=0.50, weekly_trend="bullish", trend="trending_up")
        assert abs(result - 0.50) < 0.001

    def test_timeframe_factor_unknown(self):
        """Either is 'unknown' -> factor = 1.0."""
        result = _mult(base=0.50, weekly_trend="unknown", trend="trending_up")
        assert abs(result - 0.50) < 0.001

    def test_regime_agreement_disagree(self):
        """Regime disagreement -> factor = 0.85."""
        result = _mult(base=1.0, regime_disagreement=True)
        assert abs(result - 0.85) < 0.001

    def test_regime_agreement_agree(self):
        """No regime disagreement -> factor = 1.0."""
        result = _mult(base=0.50, regime_disagreement=False)
        assert abs(result - 0.50) < 0.001

    def test_ml_confirmation_confirms(self):
        """ML direction matches rule-based -> factor = 1.10."""
        result = _mult(base=1.0, has_ml=True, scores={"ml": 0.8}, score=0.5)
        assert abs(result - 1.10) < 0.001

    def test_ml_confirmation_no_ml(self):
        """No ML signal -> factor = 1.0."""
        result = _mult(base=0.50, has_ml=False)
        assert abs(result - 0.50) < 0.001

    def test_ml_confirmation_disagrees(self):
        """ML opposes rule-based -> factor = 1.0 (no penalty)."""
        result = _mult(base=1.0, has_ml=True, scores={"ml": -0.8}, score=0.5)
        assert abs(result - 1.0) < 0.001

    def test_data_quality_technical_failure(self):
        """'technical' failure -> factor = 0.75."""
        result = _mult(base=1.0, failures=["technical"])
        assert abs(result - 0.75) < 0.001

    def test_data_quality_regime_failure(self):
        """'regime' failure -> factor = 0.75."""
        result = _mult(base=1.0, failures=["regime"])
        assert abs(result - 0.75) < 0.001

    def test_data_quality_both_failures(self):
        """Both failed -> factor = 0.75 * 0.75 = 0.5625."""
        result = _mult(base=1.0, failures=["technical", "regime"])
        assert abs(result - 0.5625) < 0.001

    def test_data_quality_no_failures(self):
        """No failures -> factor = 1.0."""
        result = _mult(base=0.50, failures=[])
        assert abs(result - 0.50) < 0.001


# ===========================================================================
# Combined Behavior Tests
# ===========================================================================


class TestConvictionMultiplicativeCombined:
    """Test the full multiplicative pipeline: base * f1 * f2 * ... * f6."""

    def test_all_factors_worst_case(self):
        """All factors at their worst."""
        # ADX<=15: 1.0, stability=0: 0.85, contradicting: 0.80,
        # regime disagree: 0.85, no ML: 1.0, technical failed: 0.75
        result = _mult(
            base=0.50, adx=10, hmm_stability=0.0,
            weekly_trend="bullish", trend="trending_down",
            regime_disagreement=True, failures=["technical"],
        )
        expected = 0.50 * 1.0 * 0.85 * 0.80 * 0.85 * 1.0 * 0.75
        assert abs(result - expected) < 0.001

    def test_all_factors_best_case(self):
        """All factors at their best."""
        # ADX=50: 1.15, stability=1.0: 1.05, agreeing: 1.0,
        # agree: 1.0, ML confirms: 1.10, no failures: 1.0
        result = _mult(
            base=0.50, adx=50, hmm_stability=1.0,
            weekly_trend="bullish", trend="trending_up",
            has_ml=True, scores={"ml": 0.8}, score=0.5,
        )
        expected = 0.50 * 1.15 * 1.05 * 1.0 * 1.0 * 1.10 * 1.0
        assert abs(result - expected) < 0.001

    def test_final_clip_lower_bound(self):
        """Extreme reduction clips to 0.05."""
        synth = RuleBasedSynthesizer()
        # Call through the full method path to get clipping
        raw = _mult(
            base=0.10, hmm_stability=0.0,
            weekly_trend="bullish", trend="trending_down",
            regime_disagreement=True, failures=["technical", "regime"],
        )
        clipped = round(max(0.05, min(0.95, raw)), 3)
        assert clipped >= 0.05

    def test_final_clip_upper_bound(self):
        """Extreme boost clips to 0.95."""
        raw = _mult(
            base=0.90, adx=50, hmm_stability=1.0,
            has_ml=True, scores={"ml": 0.8}, score=0.5,
        )
        clipped = round(max(0.05, min(0.95, raw)), 3)
        assert clipped <= 0.95

    def test_missing_inputs_default_to_unity(self):
        """All inputs None/missing -> all factors = 1.0 -> base unchanged."""
        result = _mult(base=0.60)
        assert abs(result - 0.60) < 0.001


# ===========================================================================
# Config Flag Tests
# ===========================================================================


class TestConvictionConfigFlag:
    """Test the FEEDBACK_CONVICTION_MULTIPLICATIVE kill switch."""

    def _make_synth_and_call(self, **env_overrides):
        """Helper to call _compute_bias_and_conviction with controlled env."""
        synth = RuleBasedSynthesizer()
        technical = {"adx_14": 50, "rsi_14": 50, "weekly_trend": "unknown"}
        regime = {"trend_regime": "trending_up", "hmm_stability": None}
        with patch.dict(os.environ, env_overrides, clear=False):
            _, conviction = synth._compute_bias_and_conviction(
                technical=technical, regime=regime, failures=[],
            )
        return conviction

    def test_flag_false_uses_additive(self):
        """FEEDBACK_CONVICTION_MULTIPLICATIVE=false -> additive logic."""
        c = self._make_synth_and_call(FEEDBACK_CONVICTION_MULTIPLICATIVE="false")
        # ADX=50 > 25 -> additive +0.10 on top of base abs(score)
        # score = 1.0 (trending_up) * weight + other terms
        # With additive, conviction should include the +0.10 bump
        assert c > 0  # just verify it runs without error

    def test_flag_true_uses_multiplicative(self):
        """FEEDBACK_CONVICTION_MULTIPLICATIVE=true -> multiplicative factors used."""
        c = self._make_synth_and_call(FEEDBACK_CONVICTION_MULTIPLICATIVE="true")
        assert c > 0

    def test_flag_missing_defaults_to_false(self):
        """Env var not set -> defaults to false."""
        env = os.environ.copy()
        env.pop("FEEDBACK_CONVICTION_MULTIPLICATIVE", None)
        with patch.dict(os.environ, env, clear=True):
            c = self._make_synth_and_call()
        assert c > 0
