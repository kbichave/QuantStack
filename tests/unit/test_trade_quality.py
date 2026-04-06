"""Unit tests for trade quality scoring (WI-1).

Covers: TradeQualityScore schema, evaluator factory, reflection node
integration, DB persistence, and WeightLearner training data filter.
"""

from __future__ import annotations

import os
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantstack.performance.models import TradeQualityScore


# ---------------------------------------------------------------------------
# 2.1  TradeQualityScore Schema
# ---------------------------------------------------------------------------


class TestTradeQualityScoreSchema:
    def test_has_all_fields(self):
        hints = get_type_hints(TradeQualityScore)
        expected = {
            "execution_quality",
            "thesis_accuracy",
            "risk_management",
            "timing_quality",
            "sizing_quality",
            "overall_score",
            "justification",
        }
        assert expected == set(hints.keys())

    def test_float_fields_accept_zero_to_one(self):
        score: TradeQualityScore = {
            "execution_quality": 0.0,
            "thesis_accuracy": 0.5,
            "risk_management": 1.0,
            "timing_quality": 0.3,
            "sizing_quality": 0.8,
            "overall_score": 0.6,
            "justification": "test",
        }
        for key in ("execution_quality", "thesis_accuracy", "risk_management",
                     "timing_quality", "sizing_quality", "overall_score"):
            assert isinstance(score[key], float)
        assert isinstance(score["justification"], str)


# ---------------------------------------------------------------------------
# 2.2  Evaluator Factory
# ---------------------------------------------------------------------------


try:
    import openevals  # noqa: F401
    _has_openevals = True
except ImportError:
    _has_openevals = False


@pytest.mark.skipif(not _has_openevals, reason="openevals not installed")
class TestEvaluatorFactory:
    def test_create_trade_evaluator_returns_callable(self):
        from quantstack.performance.trade_evaluator import create_trade_evaluator
        evaluator = create_trade_evaluator()
        assert callable(evaluator)

    @patch("quantstack.performance.trade_evaluator.create_llm_as_judge")
    def test_evaluator_produces_all_dimensions(self, mock_judge_factory):
        mock_judge = MagicMock(return_value={
            "execution_quality": 0.8,
            "thesis_accuracy": 0.7,
            "risk_management": 0.9,
            "timing_quality": 0.6,
            "sizing_quality": 0.75,
            "overall_score": 0.75,
            "justification": "Solid execution with minor timing delay.",
        })
        mock_judge_factory.return_value = mock_judge

        from quantstack.performance.trade_evaluator import create_trade_evaluator
        evaluator = create_trade_evaluator()

        result = evaluator(
            inputs={"entry_thesis": "Bullish breakout", "signals": {"trend": 0.8}},
            outputs={"realized_pnl": 150.0, "exit_reason": "target_hit"},
        )
        assert "execution_quality" in result
        assert "justification" in result
        for key in ("execution_quality", "thesis_accuracy", "risk_management",
                     "timing_quality", "sizing_quality", "overall_score"):
            assert 0.0 <= result[key] <= 1.0

    @patch("quantstack.performance.trade_evaluator.create_llm_as_judge")
    def test_evaluator_handles_empty_trade_context(self, mock_judge_factory):
        mock_judge = MagicMock(return_value={
            "execution_quality": 0.5,
            "thesis_accuracy": 0.3,
            "risk_management": 0.5,
            "timing_quality": 0.5,
            "sizing_quality": 0.5,
            "overall_score": 0.4,
            "justification": "Limited context available.",
        })
        mock_judge_factory.return_value = mock_judge

        from quantstack.performance.trade_evaluator import create_trade_evaluator
        evaluator = create_trade_evaluator()

        result = evaluator(
            inputs={"entry_thesis": "unavailable", "signals": {}},
            outputs={"realized_pnl": -20.0, "exit_reason": "stop_loss"},
        )
        assert isinstance(result["justification"], str)
        assert len(result["justification"]) > 0

    @patch("quantstack.performance.trade_evaluator.create_llm_as_judge")
    def test_evaluator_scores_are_valid(self, mock_judge_factory):
        mock_judge = MagicMock(return_value={
            "execution_quality": 0.9,
            "thesis_accuracy": 0.85,
            "risk_management": 0.95,
            "timing_quality": 0.7,
            "sizing_quality": 0.8,
            "overall_score": 0.84,
            "justification": "Very clean execution.",
        })
        mock_judge_factory.return_value = mock_judge

        from quantstack.performance.trade_evaluator import create_trade_evaluator
        evaluator = create_trade_evaluator()
        result = evaluator(
            inputs={"entry_thesis": "test"},
            outputs={"realized_pnl": 100.0},
        )
        for key in ("execution_quality", "thesis_accuracy", "risk_management",
                     "timing_quality", "sizing_quality", "overall_score"):
            val = result[key]
            assert isinstance(val, (int, float)), f"{key} is not numeric"
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"
        assert isinstance(result["justification"], str)
        assert len(result["justification"]) > 0


# ---------------------------------------------------------------------------
# 2.3  Reflection Node Changes
# ---------------------------------------------------------------------------


class TestReflectionNodeQuality:
    """Test make_reflection integration with trade quality scoring."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        return llm

    @pytest.fixture
    def mock_config(self):
        return MagicMock(system_prompt="Reflect.", name="reflection")

    @pytest.mark.asyncio
    @patch("quantstack.graphs.trading.nodes.run_agent", new_callable=AsyncMock)
    @patch("quantstack.graphs.trading.nodes.create_trade_evaluator")
    async def test_reflection_produces_quality_scores(
        self, mock_eval_factory, mock_run_agent, mock_llm, mock_config
    ):
        mock_run_agent.return_value = '{"reflection": "good cycle", "lessons": []}'
        mock_evaluator = MagicMock(return_value={
            "execution_quality": 0.8, "thesis_accuracy": 0.7,
            "risk_management": 0.9, "timing_quality": 0.6,
            "sizing_quality": 0.75, "overall_score": 0.75,
            "justification": "Solid.",
        })
        mock_eval_factory.return_value = mock_evaluator

        from quantstack.graphs.trading.nodes import make_reflection
        node = make_reflection(mock_llm, mock_config)

        state = {
            "exit_orders": [{"symbol": "AAPL", "realized_pnl": 100.0, "exit_reason": "target"}],
            "entry_orders": [{"symbol": "AAPL"}],
            "decisions": [{"node": "entry_scan", "symbol": "AAPL", "thesis": "Breakout"}],
            "cycle_number": 5,
        }
        with patch("quantstack.graphs.trading.nodes.db_conn"):
            result = await node(state)

        assert "trade_quality_scores" in result
        assert len(result["trade_quality_scores"]) == 1
        assert result["trade_quality_scores"][0]["execution_quality"] == 0.8

    @pytest.mark.asyncio
    @patch("quantstack.graphs.trading.nodes.run_agent", new_callable=AsyncMock)
    async def test_empty_exits_yields_empty_scores(
        self, mock_run_agent, mock_llm, mock_config
    ):
        mock_run_agent.return_value = '{"reflection": "quiet cycle", "lessons": []}'

        from quantstack.graphs.trading.nodes import make_reflection
        node = make_reflection(mock_llm, mock_config)

        state = {
            "exit_orders": [],
            "entry_orders": [],
            "decisions": [],
            "cycle_number": 3,
        }
        result = await node(state)

        assert result.get("trade_quality_scores", []) == []

    @pytest.mark.asyncio
    @patch("quantstack.graphs.trading.nodes.run_agent", new_callable=AsyncMock)
    @patch("quantstack.graphs.trading.nodes.create_trade_evaluator")
    async def test_evaluator_failure_degrades_gracefully(
        self, mock_eval_factory, mock_run_agent, mock_llm, mock_config
    ):
        mock_run_agent.return_value = '{"reflection": "cycle done", "lessons": []}'
        mock_evaluator = MagicMock(side_effect=Exception("LLM timeout"))
        mock_eval_factory.return_value = mock_evaluator

        from quantstack.graphs.trading.nodes import make_reflection
        node = make_reflection(mock_llm, mock_config)

        state = {
            "exit_orders": [{"symbol": "TSLA", "realized_pnl": -50.0}],
            "entry_orders": [],
            "decisions": [],
            "cycle_number": 7,
        }
        result = await node(state)

        # Text reflection still succeeds
        assert "reflection" in result
        assert "cycle done" in result["reflection"]
        # Quality scores empty due to failure
        assert result.get("trade_quality_scores", []) == []

    @pytest.mark.asyncio
    @patch("quantstack.graphs.trading.nodes.run_agent", new_callable=AsyncMock)
    @patch("quantstack.graphs.trading.nodes.create_trade_evaluator")
    async def test_entry_thesis_extraction(
        self, mock_eval_factory, mock_run_agent, mock_llm, mock_config
    ):
        mock_run_agent.return_value = '{"reflection": "reviewed", "lessons": []}'
        captured_inputs = {}

        def capture_evaluator(inputs, outputs):
            captured_inputs.update(inputs)
            return {
                "execution_quality": 0.7, "thesis_accuracy": 0.6,
                "risk_management": 0.8, "timing_quality": 0.5,
                "sizing_quality": 0.7, "overall_score": 0.65,
                "justification": "OK.",
            }

        mock_eval_factory.return_value = capture_evaluator

        from quantstack.graphs.trading.nodes import make_reflection
        node = make_reflection(mock_llm, mock_config)

        state = {
            "exit_orders": [{"symbol": "MSFT", "realized_pnl": 200.0}],
            "entry_orders": [],
            "decisions": [
                {"node": "entry_scan", "symbol": "MSFT", "thesis": "Earnings beat"},
                {"node": "risk_sizing", "symbol": "MSFT", "action": "approved"},
            ],
            "cycle_number": 10,
        }
        with patch("quantstack.graphs.trading.nodes.db_conn"):
            result = await node(state)

        assert "entry_thesis" in captured_inputs
        assert "Earnings beat" in captured_inputs["entry_thesis"]


# ---------------------------------------------------------------------------
# 2.5  WeightLearner Training Data Filter
# ---------------------------------------------------------------------------


class TestWeightLearnerFilter:
    """Test quality score filter in _load_trades_with_signals."""

    def _make_learner_with_mock_conn(self, rows):
        """Create a WeightLearner with a mock connection returning given rows."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn.execute.return_value = mock_cursor

        from quantstack.performance.weight_learner import WeightLearner
        learner = WeightLearner.__new__(WeightLearner)
        learner._conn = mock_conn
        return learner

    def test_excludes_low_execution_quality(self):
        """Trades with execution_quality < threshold are excluded."""
        from quantstack.performance.weight_learner import WeightLearner

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # The SQL should filter at the DB level, so we verify the query includes the filter
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        learner = WeightLearner.__new__(WeightLearner)
        learner._conn = mock_conn

        from datetime import date
        with patch.dict(os.environ, {"QUALITY_SCORE_FILTER_THRESHOLD": "0.3"}):
            learner._load_trades_with_signals(date(2025, 1, 1), date(2025, 12, 31))

        sql = mock_conn.execute.call_args[0][0]
        # SQL must LEFT JOIN trade_quality_scores
        assert "trade_quality_scores" in sql
        # SQL must have filter condition for execution_quality
        assert "execution_quality" in sql

    def test_threshold_zero_disables_filtering(self):
        """When threshold is 0.0, no quality-based filtering occurs."""
        from quantstack.performance.weight_learner import WeightLearner

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        learner = WeightLearner.__new__(WeightLearner)
        learner._conn = mock_conn

        from datetime import date
        with patch.dict(os.environ, {"QUALITY_SCORE_FILTER_THRESHOLD": "0.0"}):
            learner._load_trades_with_signals(date(2025, 1, 1), date(2025, 12, 31))

        sql = mock_conn.execute.call_args[0][0]
        # With threshold 0.0, the filter should still be in the SQL
        # but any execution_quality >= 0.0 passes (which is all)
        assert "trade_quality_scores" in sql

    def test_null_scores_always_included(self):
        """Trades without quality scores (NULL) pass the filter."""
        from quantstack.performance.weight_learner import WeightLearner

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        learner = WeightLearner.__new__(WeightLearner)
        learner._conn = mock_conn

        from datetime import date
        learner._load_trades_with_signals(date(2025, 1, 1), date(2025, 12, 31))

        sql = mock_conn.execute.call_args[0][0]
        # Must handle NULL (trades without scores) — use IS NULL OR >= threshold
        assert "IS NULL" in sql or "COALESCE" in sql

    def test_vote_keys_unchanged(self):
        """_VOTE_KEYS remains exactly 7 pre-trade signal features."""
        from quantstack.performance.weight_learner import _VOTE_KEYS
        assert _VOTE_KEYS == ["trend", "rsi", "macd", "bb", "sentiment", "ml", "flow"]
        assert len(_VOTE_KEYS) == 7
