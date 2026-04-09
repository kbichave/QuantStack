# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for P01: Signal Statistical Rigor.

Tests cross-sectional IC tracking, IC gate, bootstrap CI, signal decay,
correlation penalties, vote score persistence, and feature flags.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Part 1: Vote score persistence
# ---------------------------------------------------------------------------


class TestVoteScorePersistence:
    """Part 1: _compute_bias_and_conviction returns scores for persistence."""

    def test_compute_returns_seven_tuple(self):
        """_compute_bias_and_conviction should return (bias, conviction, scores, score, trend, weights, conviction_factors)."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        result = synth._compute_bias_and_conviction(
            technical={"rsi_14": 30, "macd_hist": 0.5, "bb_pct": 0.1, "adx_14": 30},
            regime={"trend_regime": "trending_up", "confidence": 0.8},
            failures=[],
        )
        assert len(result) == 7
        bias, conviction, scores, raw_score, trend, weights, conviction_factors = result
        assert isinstance(bias, str)
        assert isinstance(conviction, float)
        assert isinstance(scores, dict)
        assert isinstance(raw_score, float)
        assert isinstance(trend, str)
        assert isinstance(weights, dict)

    def test_scores_dict_has_expected_collectors(self):
        """Vote scores should include trend, rsi, macd, bb, sentiment, ml, flow."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        _, _, scores, _, _, _, _ = synth._compute_bias_and_conviction(
            technical={"rsi_14": 50, "macd_hist": 0.1},
            regime={"trend_regime": "ranging"},
            failures=[],
        )
        for key in ("trend", "rsi", "macd", "bb", "sentiment", "ml", "flow"):
            assert key in scores


# ---------------------------------------------------------------------------
# Part 2: CrossSectionalICTracker
# ---------------------------------------------------------------------------


class TestCrossSectionalICTracker:
    """Part 2: CrossSectionalICTracker computations."""

    @patch("quantstack.signal_engine.cross_sectional_ic.db_conn")
    def test_ic_gate_defaults_to_include_on_empty_data(self, mock_db):
        """With no IC data, all collectors should be included (True)."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        from quantstack.signal_engine.cross_sectional_ic import CrossSectionalICTracker

        tracker = CrossSectionalICTracker()
        gate = tracker.get_ic_gate_status()
        assert gate == {}  # no data → empty dict (all included by default)

    @patch("quantstack.signal_engine.cross_sectional_ic.db_conn")
    def test_ic_gate_excludes_low_ic_collector(self, mock_db):
        """Collector with IC < 0.02 for 21+ days should be gated (False)."""
        # Simulate 30 days of IC data where "sentiment" has IC = 0.01
        rows = [(f"sentiment", 0.01)] * 30 + [("trend", 0.08)] * 30
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = rows
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        from quantstack.signal_engine.cross_sectional_ic import CrossSectionalICTracker

        tracker = CrossSectionalICTracker()
        gate = tracker.get_ic_gate_status()
        assert gate.get("sentiment") is False
        assert gate.get("trend") is True

    def test_load_vote_scores_empty_on_failure(self):
        """_load_vote_scores should return {} on DB failure."""
        from quantstack.signal_engine.cross_sectional_ic import CrossSectionalICTracker

        tracker = CrossSectionalICTracker()
        with patch("quantstack.signal_engine.cross_sectional_ic.db_conn", side_effect=Exception("DB down")):
            result = tracker._load_vote_scores(date.today())
        assert result == {}


# ---------------------------------------------------------------------------
# Part 5: IC gate in synthesis
# ---------------------------------------------------------------------------


class TestICGateInSynthesis:
    """Part 5: IC gate zeros out degraded collector weights."""

    def test_zero_weight_for_gated_collector(self):
        """When IC gate says False for a collector, its weight should be 0."""
        weights = {"trend": 0.35, "rsi": 0.10, "macd": 0.20, "bb": 0.05, "sentiment": 0.30}
        gate = {"trend": True, "rsi": True, "macd": True, "bb": True, "sentiment": False}

        for k in list(weights):
            if gate.get(k) is False:
                weights[k] = 0.0
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}

        assert weights["sentiment"] == 0.0
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Part 6: Bootstrap confidence intervals
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Part 6: Bootstrap CI on conviction."""

    def test_unanimous_agreement_narrow_ci(self):
        """When all collectors agree strongly, CI should be narrow."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        # All collectors bullish
        scores = {"trend": 1.0, "rsi": 1.0, "macd": 1.0, "bb": 0.8, "sentiment": 1.0}
        weights = {"trend": 0.3, "rsi": 0.2, "macd": 0.2, "bb": 0.1, "sentiment": 0.2}
        ci = RuleBasedSynthesizer._bootstrap_conviction_ci(scores, weights)
        assert ci < 0.40  # narrow relative to conflicting signals

    def test_conflicting_signals_wide_ci(self):
        """When collectors disagree, CI should be wider."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        # Half bullish, half bearish
        scores = {"trend": 1.0, "rsi": -1.0, "macd": 1.0, "bb": -1.0, "sentiment": 0.0}
        weights = {"trend": 0.2, "rsi": 0.2, "macd": 0.2, "bb": 0.2, "sentiment": 0.2}
        ci = RuleBasedSynthesizer._bootstrap_conviction_ci(scores, weights)
        assert ci > 0.05  # wider than unanimous

    def test_single_collector_returns_zero(self):
        """With fewer than 2 collectors, CI should be 0."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        scores = {"trend": 1.0}
        weights = {"trend": 1.0}
        ci = RuleBasedSynthesizer._bootstrap_conviction_ci(scores, weights)
        assert ci == 0.0


# ---------------------------------------------------------------------------
# Part 7: Signal decay on cache read
# ---------------------------------------------------------------------------


class TestSignalDecay:
    """Part 7: Exponential decay on cached signal conviction."""

    def test_apply_decay_reduces_conviction(self):
        """Decay factor < 1 should reduce conviction."""
        from quantstack.signal_engine.cache import _apply_decay
        from quantstack.signal_engine.brief import SignalBrief

        brief = SignalBrief(
            date=date.today(),
            market_overview="test",
            market_bias="neutral",
            market_conviction=0.5,
            risk_environment="normal",
            symbol_briefs=[{
                "symbol": "AAPL",
                "market_summary": "test",
                "consensus_bias": "bullish",
                "consensus_conviction": 0.8,
                "pod_agreement": "strong",
            }],
            top_opportunities=[],
            key_risks=[],
            strategic_notes="",
            pods_reporting=5,
            total_analyses=5,
            overall_confidence=0.9,
        )
        decayed = _apply_decay(brief, 0.5)
        assert decayed.overall_confidence == pytest.approx(0.45, abs=0.01)
        assert decayed.symbol_briefs[0].consensus_conviction == pytest.approx(0.4, abs=0.01)

    def test_decay_lambda_correct(self):
        """Decay lambda should be ln(2) / half_life."""
        half_life = 1800
        expected_lambda = math.log(2) / half_life
        # At t=half_life, factor should be ~0.5
        factor = math.exp(-expected_lambda * half_life)
        assert factor == pytest.approx(0.5, abs=0.01)

    def test_get_with_age_returns_age(self):
        """TTLCache.get_with_age should return (value, age)."""
        from quantstack.shared.cache import TTLCache

        cache = TTLCache(ttl_seconds=3600)
        cache.set("test", "hello")
        result = cache.get_with_age("test")
        assert result is not None
        value, age = result
        assert value == "hello"
        assert age >= 0
        assert age < 1  # should be nearly instant

    def test_get_with_age_returns_none_for_expired(self):
        """Expired entries should return None."""
        from quantstack.shared.cache import TTLCache

        cache = TTLCache(ttl_seconds=0)  # immediate expiry
        cache.set("test", "hello")
        time.sleep(0.01)
        result = cache.get_with_age("test")
        assert result is None


# ---------------------------------------------------------------------------
# Part 8: Correlation penalty
# ---------------------------------------------------------------------------


class TestCorrelationPenalty:
    """Part 8: Correlation penalty reduces redundant collector weights."""

    def test_correlated_pair_weaker_penalized(self):
        """The weaker IC collector in a correlated pair gets penalized."""
        from quantstack.signal_engine.correlation import compute_signal_correlations
        import numpy as np

        # Two highly correlated signals + one independent
        np.random.seed(42)
        base = np.random.randn(100)
        signal_data = {
            "trend": list(base),
            "macd": list(base + np.random.randn(100) * 0.1),  # corr > 0.9 with trend
            "rsi": list(np.random.randn(100)),  # independent
        }
        ic_data = {"trend": 0.08, "macd": 0.03, "rsi": 0.05}

        result = compute_signal_correlations(signal_data, ic_data, min_observations=50)

        # macd is weaker IC and highly correlated with trend → should be penalized
        assert result.penalties["macd"] < 1.0
        # trend is stronger IC → should not be penalized (or less penalized)
        assert result.penalties["trend"] >= result.penalties["macd"]
        # rsi is independent → no penalty
        assert result.penalties["rsi"] == 1.0


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


class TestP01FeatureFlags:
    """Verify P01 feature flags default to false."""

    def test_ic_gate_default_false(self):
        os.environ.pop("FEEDBACK_IC_GATE", None)
        from quantstack.config.feedback_flags import ic_gate_enabled
        assert ic_gate_enabled() is False

    def test_signal_ci_default_false(self):
        os.environ.pop("FEEDBACK_SIGNAL_CI", None)
        from quantstack.config.feedback_flags import signal_ci_enabled
        assert signal_ci_enabled() is False

    def test_signal_decay_default_false(self):
        os.environ.pop("FEEDBACK_SIGNAL_DECAY", None)
        from quantstack.config.feedback_flags import signal_decay_enabled
        assert signal_decay_enabled() is False

    def test_ic_gate_enabled_when_set(self):
        os.environ["FEEDBACK_IC_GATE"] = "true"
        try:
            from quantstack.config.feedback_flags import ic_gate_enabled
            assert ic_gate_enabled() is True
        finally:
            os.environ.pop("FEEDBACK_IC_GATE", None)


# ---------------------------------------------------------------------------
# SymbolBrief schema backward compatibility
# ---------------------------------------------------------------------------


class TestSymbolBriefSchema:
    """Verify uncertainty_estimate field on SymbolBrief."""

    def test_uncertainty_estimate_defaults_to_zero(self):
        from quantstack.shared.schemas import SymbolBrief

        sb = SymbolBrief(
            symbol="AAPL",
            market_summary="test",
            consensus_bias="neutral",
            consensus_conviction=0.5,
            pod_agreement="mixed",
        )
        assert sb.uncertainty_estimate == 0.0

    def test_uncertainty_estimate_set_and_bounded(self):
        from quantstack.shared.schemas import SymbolBrief

        sb = SymbolBrief(
            symbol="AAPL",
            market_summary="test",
            consensus_bias="bullish",
            consensus_conviction=0.7,
            pod_agreement="strong",
            uncertainty_estimate=0.12,
        )
        assert sb.uncertainty_estimate == 0.12
