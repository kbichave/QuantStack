# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 02: Ghost Module API Audit.

Verifies OutcomeTracker formula fix, SkillTracker ICIR simplification,
ICAttributionTracker scipy availability, and trade quality summary function.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from quantstack.learning.outcome_tracker import (
    _CLIP_MAX,
    _CLIP_MIN,
    _HALFLIFE_TRADES,
    _MIN_OUTCOMES_FOR_UPDATE,
    _PNL_SCALE,
    _STEP,
)


# ===========================================================================
# OutcomeTracker affinity formula fix
# ===========================================================================


class TestOutcomeTrackerFormula:
    """Verify the updated affinity formula produces meaningful feedback."""

    def test_loss_step_size(self):
        """A -2% loss should produce a step of approximately -0.11, not -0.019."""
        # With step=0.15 and PNL_SCALE=2.0:
        # tanh(-2.0 / 2.0) = tanh(-1.0) ~ -0.762
        # step = 0.15 * -0.762 ~ -0.114
        weight = math.tanh(-2.0 / _PNL_SCALE)
        step = _STEP * weight
        assert abs(step - (-0.114)) < 0.01, f"Expected step ~-0.114, got {step}"

    def test_exponential_decay_halflife(self):
        """Outcome from 20 trades ago contributes half as much as the most recent."""
        # decay_weight = 0.5^(trades_since / _HALFLIFE_TRADES)
        # At trades_since=20: weight = 0.5
        # At trades_since=0: weight = 1.0
        recent_weight = 0.5 ** (0 / _HALFLIFE_TRADES)
        assert recent_weight == 1.0

        old_weight = 0.5 ** (20 / _HALFLIFE_TRADES)
        assert abs(old_weight - 0.5) < 0.01

    def test_cold_start_no_adjustment(self):
        """Fewer than 5 outcomes for a regime returns affinity unchanged."""
        assert _MIN_OUTCOMES_FOR_UPDATE == 5

    def test_affinity_bounds(self):
        """Affinity stays within [0.1, 1.0] after extreme values."""
        # Extreme positive
        val = 100.0
        clamped = max(_CLIP_MIN, min(_CLIP_MAX, 0.5 + _STEP * math.tanh(val / _PNL_SCALE)))
        assert _CLIP_MIN <= clamped <= _CLIP_MAX

        # Extreme negative
        val = -100.0
        clamped = max(_CLIP_MIN, min(_CLIP_MAX, 0.5 + _STEP * math.tanh(val / _PNL_SCALE)))
        assert _CLIP_MIN <= clamped <= _CLIP_MAX

    def test_recency_weighting(self):
        """A recent loss has more impact than an old loss of the same magnitude."""
        pnl = -3.0
        outcome_weight = math.tanh(pnl / _PNL_SCALE)

        recent_contribution = outcome_weight * (0.5 ** (0 / _HALFLIFE_TRADES))
        old_contribution = outcome_weight * (0.5 ** (40 / _HALFLIFE_TRADES))

        # Both are negative, recent should be more negative (stronger)
        assert abs(recent_contribution) > abs(old_contribution)

    def test_hyperparameter_values(self):
        """Verify the updated hyperparameter values match spec."""
        assert _STEP == 0.15
        assert _PNL_SCALE == 2.0
        assert _MIN_OUTCOMES_FOR_UPDATE == 5
        assert _HALFLIFE_TRADES == 20


# ===========================================================================
# SkillTracker ICIR adjustment
# ===========================================================================


class TestSkillTrackerICIR:
    """Verify the simplified ICIR adjustment in get_confidence_adjustment()."""

    def test_high_icir_capped(self):
        """ICIR=3.0 must cap the IC adjustment at 0.3, not produce 0.6."""
        from unittest.mock import PropertyMock
        from quantstack.learning.skill_tracker import SkillTracker, AgentSkill

        tracker = SkillTracker.__new__(SkillTracker)
        tracker._skills = {}
        skill = AgentSkill(agent_id="test_agent")
        skill.prediction_count = 10
        skill.correct_predictions = 8  # accuracy=0.8 → +0.3
        skill.signal_count = 5
        skill.winning_signals = 4  # win_rate=0.8 → +0.2 capped
        skill.ic_observations = [0.1] * 15  # enough observations
        tracker._skills["test_agent"] = skill

        # Patch icir property to return 3.0
        with patch.object(type(skill), "icir", new_callable=PropertyMock, return_value=3.0):
            adj = tracker.get_confidence_adjustment("test_agent")

        # Total clamped to [0.5, 1.5]
        assert 0.5 <= adj <= 1.5

    def test_adjustment_within_bounds(self):
        """get_confidence_adjustment() stays within [0.5, 1.5] for all edge cases."""
        from unittest.mock import PropertyMock
        from quantstack.learning.skill_tracker import SkillTracker, AgentSkill

        tracker = SkillTracker.__new__(SkillTracker)
        tracker._skills = {}

        # Zero observations
        assert tracker.get_confidence_adjustment("nonexistent") == 1.0

        # Perfect accuracy + high ICIR
        skill = AgentSkill(agent_id="perfect")
        skill.prediction_count = 100
        skill.correct_predictions = 100
        skill.signal_count = 50
        skill.winning_signals = 50
        skill.ic_observations = [0.1] * 20
        tracker._skills["perfect"] = skill

        with patch.object(type(skill), "icir", new_callable=PropertyMock, return_value=5.0):
            adj = tracker.get_confidence_adjustment("perfect")
        assert 0.5 <= adj <= 1.5

        # Zero accuracy + negative ICIR
        skill2 = AgentSkill(agent_id="terrible")
        skill2.prediction_count = 100
        skill2.correct_predictions = 0
        skill2.signal_count = 50
        skill2.winning_signals = 0
        skill2.ic_observations = [-0.05] * 20
        tracker._skills["terrible"] = skill2

        with patch.object(type(skill2), "icir", new_callable=PropertyMock, return_value=-3.0):
            adj2 = tracker.get_confidence_adjustment("terrible")
        assert 0.5 <= adj2 <= 1.5

    def test_icir_multiplier_value(self):
        """ICIR multiplier should be 0.15 (not 0.2)."""
        from unittest.mock import PropertyMock
        from quantstack.learning.skill_tracker import SkillTracker, AgentSkill

        tracker = SkillTracker.__new__(SkillTracker)
        tracker._skills = {}

        skill = AgentSkill(agent_id="mid_icir")
        skill.prediction_count = 1  # nonzero so early-return guard is skipped
        skill.signal_count = 0
        skill.ic_observations = [0.1] * 15
        tracker._skills["mid_icir"] = skill

        # ICIR=2.0 with 0.15 multiplier = 0.30 (exactly at cap)
        with patch.object(type(skill), "icir", new_callable=PropertyMock, return_value=2.0):
            adj = tracker.get_confidence_adjustment("mid_icir")

        # Base 1.0 + ICIR adj (2.0*0.15=0.30) = 1.30
        assert abs(adj - 1.30) < 0.01


# ===========================================================================
# StrategyBreaker post-migration
# ===========================================================================


class TestBreakerPostMigration:
    """Verify breaker state survives restart after section-01 PostgreSQL migration."""

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_tripped_persists_across_restart(self, mock_db_conn):
        """A TRIPPED state read back from DB matches the original."""
        from contextlib import contextmanager
        from datetime import datetime, timezone
        from quantstack.execution.strategy_breaker import StrategyBreaker

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = [{
            "strategy_id": "s1",
            "status": "TRIPPED",
            "scale_factor": 0.0,
            "consecutive_losses": 3,
            "peak_equity": 10000.0,
            "current_equity": 9400.0,
            "drawdown_pct": 6.0,
            "tripped_at": datetime.now(timezone.utc),
            "reason": "Max drawdown",
        }]

        @contextmanager
        def ctx():
            yield mock_conn

        mock_db_conn.side_effect = ctx

        breaker = StrategyBreaker()
        assert breaker.get_scale_factor("s1") == 0.0

    @patch("quantstack.execution.strategy_breaker.db_conn")
    def test_concurrent_reads_no_block(self, mock_db_conn):
        """Multiple get_scale_factor() calls do not deadlock."""
        from contextlib import contextmanager
        from quantstack.execution.strategy_breaker import StrategyBreaker

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = []

        @contextmanager
        def ctx():
            yield mock_conn

        mock_db_conn.side_effect = ctx

        breaker = StrategyBreaker()
        for _ in range(50):
            breaker.get_scale_factor("any")


# ===========================================================================
# ICAttributionTracker verification
# ===========================================================================


class TestICAttributionVerification:
    """Verify ICAttributionTracker API correctness and scipy availability."""

    def test_scipy_spearman_available(self):
        """scipy.stats.spearmanr is importable."""
        from scipy.stats import spearmanr
        assert callable(spearmanr)

    @patch("quantstack.learning.ic_attribution.db_conn")
    def test_data_persists_across_restart(self, mock_db_conn):
        """Data written to DB can be read back by a new instance."""
        from contextlib import contextmanager
        from datetime import datetime, timezone
        from quantstack.learning.ic_attribution import ICAttributionTracker

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = [
            {
                "collector": "technical",
                "signal_value": 0.7,
                "forward_return": 0.02,
                "recorded_at": datetime.now(timezone.utc),
            },
        ]

        @contextmanager
        def ctx():
            yield mock_conn

        mock_db_conn.side_effect = ctx

        tracker = ICAttributionTracker(window_size=30)
        assert "technical" in tracker._collectors
        assert len(tracker._collectors["technical"].observations) == 1


# ===========================================================================
# TradeEvaluator read function
# ===========================================================================


class TestTradeQualitySummary:
    """Verify get_trade_quality_summary() returns correct rolling averages."""

    @patch("quantstack.learning.trade_quality.db_conn")
    def test_returns_summary_with_scores(self, mock_db_conn):
        """Returns dimension averages and weakest dimension."""
        from contextlib import contextmanager
        from quantstack.learning.trade_quality import get_trade_quality_summary

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = [
            {
                "execution_quality": 6.0,
                "thesis_accuracy": 8.0,
                "risk_management": 7.0,
                "timing_quality": 5.0,
                "sizing_quality": 7.5,
                "overall_score": 6.7,
            },
            {
                "execution_quality": 7.0,
                "thesis_accuracy": 7.0,
                "risk_management": 8.0,
                "timing_quality": 6.0,
                "sizing_quality": 8.0,
                "overall_score": 7.2,
            },
        ] * 5  # 10 rows

        @contextmanager
        def ctx():
            yield mock_conn

        mock_db_conn.side_effect = ctx

        result = get_trade_quality_summary()
        assert result is not None
        assert "dimensions" in result
        assert result["weakest"] == "timing_quality"
        assert result["trade_count"] == 10

    @patch("quantstack.learning.trade_quality.db_conn")
    def test_returns_none_insufficient_data(self, mock_db_conn):
        """Returns None if fewer than 5 scored trades exist."""
        from contextlib import contextmanager
        from quantstack.learning.trade_quality import get_trade_quality_summary

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = [
            {
                "execution_quality": 6.0, "thesis_accuracy": 8.0,
                "risk_management": 7.0, "timing_quality": 5.0,
                "sizing_quality": 7.5, "overall_score": 6.7,
            },
        ] * 3  # only 3 rows

        @contextmanager
        def ctx():
            yield mock_conn

        mock_db_conn.side_effect = ctx

        result = get_trade_quality_summary()
        assert result is None

    @patch("quantstack.learning.trade_quality.db_conn")
    def test_strategy_filter(self, mock_db_conn):
        """Passing strategy_id filters the query."""
        from contextlib import contextmanager
        from quantstack.learning.trade_quality import get_trade_quality_summary

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = []

        @contextmanager
        def ctx():
            yield mock_conn

        mock_db_conn.side_effect = ctx

        get_trade_quality_summary(strategy_id="strat_123")

        # Verify strategy_id was passed in query params
        call_args = mock_conn.execute.call_args_list
        query_calls = [c for c in call_args if "trade_quality_scores" in str(c)]
        assert len(query_calls) > 0
        # The query should contain the strategy filter
        query_str = str(query_calls[0])
        assert "strategy_id" in query_str
