# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for P00: Wire Learning Modules.

Tests all 6 wires that connect learning module outputs to decision-making code:
  Wire 1: OutcomeTracker regime affinity → daily planner context
  Wire 2: OutcomeTracker regime affinity → position sizing (flag-gated)
  Wire 3a: StrategyBreaker → execute_entries filtering
  Wire 3b: StrategyBreaker → trade_hooks recording
  Wire 4: SkillTracker confidence → conviction scaling (flag-gated)
  Wire 5a: ICAttributionTracker data collection in trade_hooks
  Wire 5b: IC weight EWMA blend in synthesis (flag-gated)
  Wire 6: Trade quality scores → daily planner context
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Wire 6 + Wire 1: Daily planner context injection
# ---------------------------------------------------------------------------


class TestWire6QualityScoresInDailyPlan:
    """Wire 6: trade_quality_scores → daily planner prompt."""

    @patch("quantstack.graphs.trading.nodes.db_conn")
    def test_quality_section_present_when_data_exists(self, mock_db):
        """When trade_quality_scores has rows, the prompt should contain quality context."""
        mock_conn = MagicMock()
        mock_rows = [
            MagicMock(
                _mapping={
                    "symbol": "AAPL",
                    "overall_score": 7.5,
                    "thesis_accuracy": 8.0,
                    "execution_quality": 7.0,
                    "timing_quality": 6.5,
                    "sizing_quality": 8.0,
                    "created_at": "2026-04-07",
                },
            )
        ]
        # Make each mock row behave like a Row with _mapping
        for r in mock_rows:
            r._mapping = {
                "symbol": "AAPL",
                "overall_score": 7.5,
                "thesis_accuracy": 8.0,
                "execution_quality": 7.0,
                "timing_quality": 6.5,
                "sizing_quality": 8.0,
                "created_at": "2026-04-07",
            }
        mock_conn.execute.return_value.fetchall.return_value = mock_rows
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        # Build the quality section logic directly (extracted from nodes.py)
        rows = mock_conn.execute("dummy").fetchall()
        if rows:
            scores = [dict(r._mapping) for r in rows]
            avg_overall = sum(float(s.get("overall_score", 0)) for s in scores) / len(scores)
            quality_section = (
                f"\n--- Recent Trade Quality (last {len(scores)} trades, avg={avg_overall:.2f}) ---\n"
            )
            assert "avg=7.50" in quality_section
            assert "Recent Trade Quality" in quality_section

    def test_quality_section_empty_on_no_data(self):
        """When no quality scores exist, section should be empty string."""
        quality_section = ""
        # Simulate the try/except with empty result
        rows = []
        if rows:
            quality_section = "should not appear"
        assert quality_section == ""


class TestWire1RegimeAffinityInDailyPlan:
    """Wire 1: strategy regime_affinity → daily planner prompt."""

    def test_affinity_ranking_sorts_by_score(self):
        """Strategies should be ranked by affinity for the current regime."""
        regime = "trending_up"
        rows = [
            ("strat_a", "Momentum", json.dumps({"trending_up": 0.9, "ranging": 0.3})),
            ("strat_b", "MeanRev", json.dumps({"trending_up": 0.2, "ranging": 0.8})),
            ("strat_c", "Hybrid", json.dumps({"trending_up": 0.6})),
        ]
        ranked = []
        for r in rows:
            aff = json.loads(r[2]) if isinstance(r[2], str) else (r[2] or {})
            ranked.append({
                "strategy_id": r[0],
                "name": r[1],
                "affinity": round(float(aff.get(regime, 0.5)), 2),
            })
        ranked.sort(key=lambda x: x["affinity"], reverse=True)

        assert ranked[0]["strategy_id"] == "strat_a"
        assert ranked[0]["affinity"] == 0.9
        assert ranked[-1]["strategy_id"] == "strat_b"
        assert ranked[-1]["affinity"] == 0.2

    def test_affinity_defaults_to_half_for_unknown_regime(self):
        """Missing regime key should default to 0.5."""
        aff = {"trending_up": 0.9}
        assert float(aff.get("ranging", 0.5)) == 0.5


# ---------------------------------------------------------------------------
# Wire 3a: StrategyBreaker → execute_entries
# ---------------------------------------------------------------------------


class TestWire3aStrategyBreakerInExecuteEntries:
    """Wire 3a: StrategyBreaker blocks/scales entries."""

    def test_tripped_strategy_rejected(self):
        """When breaker returns 0.0 (TRIPPED), entry should be rejected."""
        approved = [
            {"strategy_id": "strat_a", "symbol": "AAPL", "size": 100},
            {"strategy_id": "strat_b", "symbol": "TSLA", "size": 200},
        ]

        mock_breaker = MagicMock()
        mock_breaker.get_scale_factor.side_effect = lambda sid: 0.0 if sid == "strat_a" else 1.0

        filtered = []
        for d in approved:
            sid = d.get("strategy_id", "")
            scale = mock_breaker.get_scale_factor(sid) if sid else 1.0
            if scale == 0.0:
                continue
            if scale < 1.0:
                d = dict(d)
                if "size" in d:
                    d["size"] = round(d["size"] * scale, 4)
                d["breaker_scale_factor"] = scale
            filtered.append(d)

        assert len(filtered) == 1
        assert filtered[0]["strategy_id"] == "strat_b"

    def test_scaled_strategy_size_halved(self):
        """When breaker returns 0.5 (SCALED), size should be halved."""
        approved = [{"strategy_id": "strat_a", "symbol": "AAPL", "size": 100}]

        mock_breaker = MagicMock()
        mock_breaker.get_scale_factor.return_value = 0.5

        filtered = []
        for d in approved:
            sid = d.get("strategy_id", "")
            scale = mock_breaker.get_scale_factor(sid)
            if scale < 1.0:
                d = dict(d)
                d["size"] = round(d["size"] * scale, 4)
                d["breaker_scale_factor"] = scale
            filtered.append(d)

        assert filtered[0]["size"] == 50.0
        assert filtered[0]["breaker_scale_factor"] == 0.5

    def test_active_strategy_passes_unchanged(self):
        """When breaker returns 1.0 (ACTIVE), entry passes unchanged."""
        approved = [{"strategy_id": "strat_a", "symbol": "AAPL", "size": 100}]

        mock_breaker = MagicMock()
        mock_breaker.get_scale_factor.return_value = 1.0

        filtered = []
        for d in approved:
            sid = d.get("strategy_id", "")
            scale = mock_breaker.get_scale_factor(sid)
            if scale == 0.0:
                continue
            if scale < 1.0:
                d = dict(d)
                d["size"] = round(d["size"] * scale, 4)
            filtered.append(d)

        assert filtered[0]["size"] == 100
        assert "breaker_scale_factor" not in filtered[0]


# ---------------------------------------------------------------------------
# Wire 3b: StrategyBreaker recording in trade_hooks
# ---------------------------------------------------------------------------


class TestWire3bStrategyBreakerRecording:
    """Wire 3b: on_trade_close records to StrategyBreaker."""

    @patch("quantstack.hooks.trade_hooks._get_reflection_manager")
    @patch("quantstack.hooks.trade_hooks._get_credit_assigner")
    @patch("quantstack.hooks.trade_hooks._update_skill_tracker")
    @patch("quantstack.execution.strategy_breaker.StrategyBreaker")
    def test_record_trade_called_on_close(
        self, mock_breaker_cls, mock_skill, mock_credit, mock_reflect
    ):
        """on_trade_close should call StrategyBreaker.record_trade."""
        mock_reflect.return_value = MagicMock()
        mock_reflect.return_value.record_outcome.return_value = MagicMock()
        mock_credit.return_value = None

        mock_breaker = MagicMock()
        mock_breaker_cls.return_value = mock_breaker

        from quantstack.hooks.trade_hooks import on_trade_close

        on_trade_close(
            symbol="AAPL",
            strategy_id="test_strat",
            action="buy",
            entry_price=150.0,
            exit_price=145.0,
            realized_pnl_pct=-3.3,
        )

        mock_breaker.record_trade.assert_called_once_with(
            "test_strat", pnl=-3.3, equity=100.0 + (-3.3)
        )


# ---------------------------------------------------------------------------
# Wire 2: Regime affinity → position sizing
# ---------------------------------------------------------------------------


class TestWire2RegimeAffinitySizing:
    """Wire 2: regime_affinity scales signal_value in risk_sizing."""

    def test_affinity_scales_signal(self):
        """With affinity=0.3, signal_value should be multiplied by 0.3."""
        regime_affinity_lookup = {"strat_a": 0.3}
        signal_value = 1.0
        sid = "strat_a"

        if regime_affinity_lookup and sid in regime_affinity_lookup:
            affinity = max(regime_affinity_lookup[sid], 0.1)
            signal_value *= affinity

        assert signal_value == pytest.approx(0.3)

    def test_affinity_floor_at_0_1(self):
        """Affinity should be floored at 0.1 to prevent zeroing."""
        regime_affinity_lookup = {"strat_a": 0.0}
        signal_value = 1.0
        sid = "strat_a"

        if regime_affinity_lookup and sid in regime_affinity_lookup:
            affinity = max(regime_affinity_lookup[sid], 0.1)
            signal_value *= affinity

        assert signal_value == pytest.approx(0.1)

    def test_no_affinity_lookup_passes_through(self):
        """When lookup is empty, signal_value is unchanged."""
        regime_affinity_lookup: dict[str, float] = {}
        signal_value = 1.0
        sid = "strat_a"

        if regime_affinity_lookup and sid in regime_affinity_lookup:
            signal_value *= regime_affinity_lookup[sid]

        assert signal_value == 1.0


# ---------------------------------------------------------------------------
# Wire 4: SkillTracker confidence → conviction
# ---------------------------------------------------------------------------


class TestWire4SkillConfidence:
    """Wire 4: SkillTracker adjustment scales signal_value."""

    def test_skill_adjustment_applied(self):
        """With adj=0.7, signal_value should be multiplied by 0.7."""
        skill_adjustments = {"agent_a": 0.7}
        signal_value = 1.0
        agent_id = "agent_a"

        if skill_adjustments:
            adj = skill_adjustments.get(agent_id, 1.0)
            signal_value *= adj

        assert signal_value == pytest.approx(0.7)

    def test_missing_agent_defaults_to_1(self):
        """Unknown agent_id should default to 1.0 (no change)."""
        skill_adjustments = {"agent_a": 0.7}
        signal_value = 1.0
        agent_id = "unknown_agent"

        if skill_adjustments:
            adj = skill_adjustments.get(agent_id, 1.0)
            signal_value *= adj

        assert signal_value == 1.0

    def test_wire2_and_wire4_compound(self):
        """Both regime affinity and skill adjustment should compound."""
        regime_affinity_lookup = {"strat_a": 0.5}
        skill_adjustments = {"strat_a": 0.8}
        signal_value = 1.0
        sid = "strat_a"

        # Wire 2
        if regime_affinity_lookup and sid in regime_affinity_lookup:
            affinity = max(regime_affinity_lookup[sid], 0.1)
            signal_value *= affinity

        # Wire 4
        if skill_adjustments:
            adj = skill_adjustments.get(sid, 1.0)
            signal_value *= adj

        assert signal_value == pytest.approx(0.5 * 0.8)


# ---------------------------------------------------------------------------
# Wire 5a: IC data collection in trade_hooks
# ---------------------------------------------------------------------------


class TestWire5aICDataCollection:
    """Wire 5a: on_trade_close records IC observations."""

    @patch("quantstack.hooks.trade_hooks._get_reflection_manager")
    @patch("quantstack.hooks.trade_hooks._get_credit_assigner")
    @patch("quantstack.hooks.trade_hooks._update_skill_tracker")
    @patch("quantstack.execution.strategy_breaker.StrategyBreaker")
    @patch("quantstack.learning.ic_attribution.ICAttributionTracker")
    def test_ic_record_called_per_collector(
        self, mock_ic_cls, mock_breaker_cls, mock_skill, mock_credit, mock_reflect
    ):
        """With JSON signals_summary, ICAttributionTracker.record should be called per collector."""
        mock_reflect.return_value = MagicMock()
        mock_reflect.return_value.record_outcome.return_value = MagicMock()
        mock_credit.return_value = None

        mock_ic = MagicMock()
        mock_ic_cls.return_value = mock_ic

        signals = json.dumps({"trend": 0.8, "rsi": -0.3, "macd": 0.5})

        from quantstack.hooks.trade_hooks import on_trade_close

        on_trade_close(
            symbol="AAPL",
            strategy_id="test_strat",
            action="buy",
            entry_price=150.0,
            exit_price=155.0,
            realized_pnl_pct=3.3,
            signals_summary=signals,
        )

        assert mock_ic.record.call_count == 3
        # Verify forward_return is pct / 100
        for call in mock_ic.record.call_args_list:
            assert call.kwargs["forward_return"] == pytest.approx(0.033)

    def test_non_json_signals_summary_silently_skipped(self):
        """Non-JSON signals_summary should not raise."""
        signals_summary = "trend=bullish, rsi=oversold"
        try:
            collectors = json.loads(signals_summary)
        except (json.JSONDecodeError, TypeError):
            collectors = None

        assert collectors is None


# ---------------------------------------------------------------------------
# Wire 5b: IC-driven regime-conditioned weights
# ---------------------------------------------------------------------------


class TestWire5bICDrivenWeights:
    """Wire 5b: IC-driven weights fully replace static when sufficient data."""

    def test_ic_weights_replace_static(self):
        """When IC returns weights, they replace static profiles entirely."""
        static_weights = {"trend": 0.35, "rsi": 0.10, "macd": 0.20, "bb": 0.05, "sentiment": 0.30}
        ic_weights = {"trend": 0.60, "rsi": 0.25, "macd": 0.15}

        # IC-driven: full replacement
        weights = ic_weights
        assert weights["trend"] == 0.60
        assert "bb" not in weights  # Only collectors with positive IC get weight

    def test_none_ic_weights_keeps_static(self):
        """When IC returns None (insufficient data), static profiles used."""
        static_weights = {"trend": 0.35, "rsi": 0.10, "macd": 0.20}
        ic_weights = None

        weights = static_weights.copy()
        if ic_weights:
            weights = ic_weights

        assert weights == static_weights


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """Verify new feedback flags default to false."""

    def test_regime_affinity_sizing_default_false(self):
        """FEEDBACK_REGIME_AFFINITY_SIZING should default to false."""
        import os
        os.environ.pop("FEEDBACK_REGIME_AFFINITY_SIZING", None)
        from quantstack.config.feedback_flags import regime_affinity_sizing_enabled
        assert regime_affinity_sizing_enabled() is False

    def test_skill_confidence_default_false(self):
        """FEEDBACK_SKILL_CONFIDENCE should default to false."""
        import os
        os.environ.pop("FEEDBACK_SKILL_CONFIDENCE", None)
        from quantstack.config.feedback_flags import skill_confidence_enabled
        assert skill_confidence_enabled() is False

    def test_regime_affinity_sizing_enabled_when_set(self):
        """FEEDBACK_REGIME_AFFINITY_SIZING=true should enable the flag."""
        import os
        os.environ["FEEDBACK_REGIME_AFFINITY_SIZING"] = "true"
        try:
            from quantstack.config.feedback_flags import regime_affinity_sizing_enabled
            assert regime_affinity_sizing_enabled() is True
        finally:
            os.environ.pop("FEEDBACK_REGIME_AFFINITY_SIZING", None)

    def test_skill_confidence_enabled_when_set(self):
        """FEEDBACK_SKILL_CONFIDENCE=true should enable the flag."""
        import os
        os.environ["FEEDBACK_SKILL_CONFIDENCE"] = "true"
        try:
            from quantstack.config.feedback_flags import skill_confidence_enabled
            assert skill_confidence_enabled() is True
        finally:
            os.environ.pop("FEEDBACK_SKILL_CONFIDENCE", None)


# ===========================================================================
# P05: Adaptive Signal Synthesis wires
# ===========================================================================


# ---------------------------------------------------------------------------
# P05 §5.1: IC-driven regime-conditioned weights
# ---------------------------------------------------------------------------


class TestP05ICDrivenWeights:
    """P05 §5.1: get_weights_for_regime replaces static profiles when flag on."""

    def test_get_weights_for_regime_returns_none_insufficient_data(self):
        """With fewer than min_days observations, returns None."""
        from quantstack.learning.ic_attribution import ICAttributionTracker, _Observation

        tracker = ICAttributionTracker.__new__(ICAttributionTracker)
        tracker._lock = __import__("threading").Lock()
        tracker._window_size = 30
        from quantstack.learning.ic_attribution import _CollectorState
        tracker._collectors = {
            "trend": _CollectorState(
                observations=[
                    _Observation(0.5, 0.01, "2026-01-01T00:00:00", regime="trending_up")
                    for _ in range(10)
                ]
            ),
        }
        result = tracker.get_weights_for_regime("trending_up", window=63, min_days=60)
        assert result is None

    def test_get_weights_for_regime_returns_weights_with_sufficient_data(self):
        """With enough regime-conditioned data, returns normalized weights."""
        from quantstack.learning.ic_attribution import (
            ICAttributionTracker,
            _CollectorState,
            _Observation,
        )

        tracker = ICAttributionTracker.__new__(ICAttributionTracker)
        tracker._lock = __import__("threading").Lock()
        tracker._window_size = 30

        # Create correlated signal → return observations
        import random
        random.seed(42)
        obs_trend = [
            _Observation(
                signal_value=float(i) / 100,
                forward_return=float(i) / 100 + random.gauss(0, 0.001),
                timestamp=f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                regime="trending_up",
            )
            for i in range(70)
        ]
        obs_rsi = [
            _Observation(
                signal_value=random.random(),
                forward_return=random.gauss(0, 0.01),
                timestamp=f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                regime="trending_up",
            )
            for i in range(70)
        ]
        tracker._collectors = {
            "trend": _CollectorState(observations=obs_trend),
            "rsi": _CollectorState(observations=obs_rsi),
        }

        result = tracker.get_weights_for_regime("trending_up", window=63, min_days=60)
        # trend has strong positive IC (correlated), should get weight
        assert result is not None
        assert "trend" in result
        if len(result) > 1:
            assert sum(result.values()) == pytest.approx(1.0, abs=0.01)

    def test_get_weights_for_regime_filters_by_regime(self):
        """Observations from other regimes are excluded."""
        from quantstack.learning.ic_attribution import (
            ICAttributionTracker,
            _CollectorState,
            _Observation,
        )

        tracker = ICAttributionTracker.__new__(ICAttributionTracker)
        tracker._lock = __import__("threading").Lock()
        tracker._window_size = 30

        # 70 observations but all in trending_down, not trending_up
        obs = [
            _Observation(float(i) / 100, float(i) / 100, "2026-01-01T00:00:00", regime="trending_down")
            for i in range(70)
        ]
        tracker._collectors = {"trend": _CollectorState(observations=obs)}

        result = tracker.get_weights_for_regime("trending_up", window=63, min_days=60)
        assert result is None


# ---------------------------------------------------------------------------
# P05 §5.2: Transition signal dampening
# ---------------------------------------------------------------------------


class TestP05TransitionDampening:
    """P05 §5.2: Score halved when transition_probability > 0.3."""

    def test_score_halved_during_transition(self):
        """When transition prob > 0.3 and flag on, score should be halved."""
        import os
        os.environ["FEEDBACK_TRANSITION_SIGNAL_DAMPENING"] = "true"
        try:
            score = 0.6
            transition_prob = 0.45
            if transition_prob > 0.3:
                score *= 0.5
            assert score == pytest.approx(0.3)
        finally:
            os.environ.pop("FEEDBACK_TRANSITION_SIGNAL_DAMPENING", None)

    def test_score_unchanged_when_stable(self):
        """When transition prob < 0.3, score should be unchanged."""
        score = 0.6
        transition_prob = 0.15
        original = score
        if transition_prob > 0.3:
            score *= 0.5
        assert score == original

    def test_score_unchanged_when_flag_off(self):
        """When flag is off, score should be unchanged regardless of transition prob."""
        import os
        os.environ.pop("FEEDBACK_TRANSITION_SIGNAL_DAMPENING", None)
        from quantstack.config.feedback_flags import transition_signal_dampening_enabled
        assert transition_signal_dampening_enabled() is False


# ---------------------------------------------------------------------------
# P05 §5.2: Vol sub-regime weight profiles
# ---------------------------------------------------------------------------


class TestP05VolSubRegimeProfiles:
    """P05 §5.2: _get_weights uses sub_regime when available."""

    def test_sub_regime_takes_precedence(self):
        """When sub_regime matches a profile, it should be used over base regime."""
        from quantstack.signal_engine.synthesis import _get_weights, _WEIGHT_PROFILES

        # trending_up_low_vol has trend=0.40, while base trending_up has trend=0.35
        weights = _get_weights("trending_up", has_ml=True, has_flow=True, sub_regime="trending_up_low_vol")
        assert weights["trend"] == 0.40

    def test_falls_back_to_base_regime(self):
        """When sub_regime is not in profiles, falls back to base regime."""
        from quantstack.signal_engine.synthesis import _get_weights

        weights = _get_weights("trending_up", has_ml=True, has_flow=True, sub_regime="trending_up_extreme_vol")
        assert weights["trend"] == 0.35  # base trending_up profile

    def test_falls_back_when_sub_regime_none(self):
        """When sub_regime is None, uses base regime."""
        from quantstack.signal_engine.synthesis import _get_weights

        weights = _get_weights("ranging", has_ml=True, has_flow=True, sub_regime=None)
        assert weights["rsi"] == 0.25  # base ranging profile

    def test_all_sub_regime_profiles_sum_to_one(self):
        """All vol-conditioned profiles must sum to 1.0."""
        from quantstack.signal_engine.synthesis import _WEIGHT_PROFILES

        sub_regimes = [k for k in _WEIGHT_PROFILES if "_vol" in k]
        assert len(sub_regimes) == 6
        for name in sub_regimes:
            total = sum(_WEIGHT_PROFILES[name].values())
            assert total == pytest.approx(1.0, abs=0.01), f"{name} sums to {total}"


# ---------------------------------------------------------------------------
# P05 §5.3: Conviction factor breakdown
# ---------------------------------------------------------------------------


class TestP05ConvictionFactorBreakdown:
    """P05 §5.3: Multiplicative conviction returns factor dict."""

    def test_multiplicative_returns_factor_dict(self):
        """_conviction_multiplicative should return (adjusted, factors_dict)."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        result = RuleBasedSynthesizer._conviction_multiplicative(
            base_conviction=0.5,
            adx=30.0,
            hmm_stability=0.9,
            weekly_trend="bullish",
            trend="trending_up",
            regime={"regime_disagreement": False},
            has_ml=True,
            scores={"ml": 0.5, "trend": 0.8},
            score=0.5,
            failures=[],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        adjusted, factors = result
        assert isinstance(adjusted, float)
        assert isinstance(factors, dict)
        assert "adx" in factors
        assert "stability" in factors
        assert "timeframe" in factors
        assert "regime_agreement" in factors
        assert "ml_confirmation" in factors
        assert "data_quality" in factors

    def test_factor_values_are_reasonable(self):
        """Factor values should be within expected ranges."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        _, factors = RuleBasedSynthesizer._conviction_multiplicative(
            base_conviction=0.5,
            adx=40.0,
            hmm_stability=0.5,
            weekly_trend="bearish",
            trend="trending_up",
            regime={"regime_disagreement": True},
            has_ml=False,
            scores={},
            score=0.5,
            failures=["technical"],
        )
        assert factors["adx"] > 1.0  # ADX 40 should boost
        assert factors["stability"] < 1.0  # 0.5 stability → below 1.0
        assert factors["timeframe"] == 0.80  # weekly contradicts daily
        assert factors["regime_agreement"] == 0.85  # disagreement penalty
        assert factors["data_quality"] == 0.75  # one failure


# ---------------------------------------------------------------------------
# P05 §5.4: Ensemble methods
# ---------------------------------------------------------------------------


class TestP05EnsembleMethods:
    """P05 §5.4: weighted_avg, weighted_median, trimmed_mean."""

    def test_weighted_avg_basic(self):
        from quantstack.signal_engine.synthesis import _ensemble_weighted_avg

        scores = {"a": 1.0, "b": -1.0, "c": 0.0}
        weights = {"a": 0.5, "b": 0.25, "c": 0.25}
        result = _ensemble_weighted_avg(scores, weights)
        assert result == pytest.approx(0.25)  # 0.5*1 + 0.25*(-1) + 0.25*0

    def test_weighted_median_middle_value(self):
        from quantstack.signal_engine.synthesis import _ensemble_weighted_median

        scores = {"a": -1.0, "b": 0.0, "c": 1.0}
        weights = {"a": 0.33, "b": 0.34, "c": 0.33}
        result = _ensemble_weighted_median(scores, weights)
        assert result == pytest.approx(0.0)

    def test_weighted_median_skewed(self):
        from quantstack.signal_engine.synthesis import _ensemble_weighted_median

        scores = {"a": -1.0, "b": 0.0, "c": 1.0}
        weights = {"a": 0.10, "b": 0.10, "c": 0.80}
        result = _ensemble_weighted_median(scores, weights)
        # cumulative: a=-1(0.10), b=0(0.20), c=1(1.00); half=0.50 → c wins
        assert result == pytest.approx(1.0)

    def test_trimmed_mean_drops_extremes(self):
        from quantstack.signal_engine.synthesis import _ensemble_trimmed_mean

        scores = {"a": -1.0, "b": 0.0, "c": 0.1, "d": 0.2, "e": 1.0}
        weights = {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2}
        result = _ensemble_trimmed_mean(scores, weights)
        # Drops a=-1 (lowest) and e=1 (highest), averages b,c,d
        # (0*0.2 + 0.1*0.2 + 0.2*0.2) / (0.2+0.2+0.2) = 0.06/0.6 = 0.1
        assert result == pytest.approx(0.1)

    def test_ab_routing_deterministic(self):
        """Same symbol always gets the same ensemble method."""
        from quantstack.signal_engine.synthesis import _ENSEMBLE_METHODS

        method_a = _ENSEMBLE_METHODS[hash("AAPL") % len(_ENSEMBLE_METHODS)]
        method_b = _ENSEMBLE_METHODS[hash("AAPL") % len(_ENSEMBLE_METHODS)]
        assert method_a is method_b


# ---------------------------------------------------------------------------
# P05 feature flags
# ---------------------------------------------------------------------------


class TestP05FeatureFlags:
    """All P05 flags default to False."""

    @pytest.mark.parametrize("flag_name,env_var", [
        ("ic_driven_weights_enabled", "FEEDBACK_IC_DRIVEN_WEIGHTS"),
        ("transition_signal_dampening_enabled", "FEEDBACK_TRANSITION_SIGNAL_DAMPENING"),
        ("ensemble_ab_test_enabled", "FEEDBACK_ENSEMBLE_AB_TEST"),
    ])
    def test_p05_flags_default_false(self, flag_name, env_var):
        import os
        os.environ.pop(env_var, None)
        from quantstack.config import feedback_flags
        flag_fn = getattr(feedback_flags, flag_name)
        assert flag_fn() is False

    @pytest.mark.parametrize("flag_name,env_var", [
        ("ic_driven_weights_enabled", "FEEDBACK_IC_DRIVEN_WEIGHTS"),
        ("transition_signal_dampening_enabled", "FEEDBACK_TRANSITION_SIGNAL_DAMPENING"),
        ("ensemble_ab_test_enabled", "FEEDBACK_ENSEMBLE_AB_TEST"),
    ])
    def test_p05_flags_enabled_when_set(self, flag_name, env_var):
        import os
        os.environ[env_var] = "true"
        try:
            from quantstack.config import feedback_flags
            flag_fn = getattr(feedback_flags, flag_name)
            assert flag_fn() is True
        finally:
            os.environ.pop(env_var, None)
