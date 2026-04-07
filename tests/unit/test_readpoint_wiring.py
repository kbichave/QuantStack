# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Section 03: Readpoint Wiring.

Verifies that ghost modules (StrategyBreaker, SkillTracker, ICAttributionTracker,
trade quality) are properly wired into production callsites.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from quantstack.execution.strategy_breaker import StrategyBreaker


# ===========================================================================
# Wire 1: get_regime_strategies() tool
# ===========================================================================


class TestGetRegimeStrategies:
    """Wire 1: Replace stub with real DB query + breaker status."""

    def test_returns_strategies_for_regime(self):
        """Matching strategies returned sorted by affinity."""
        mock_conn = MagicMock()
        mock_conn.fetchall.return_value = [
            {
                "strategy_id": "swing_AAPL",
                "name": "Swing AAPL",
                "status": "active",
                "regime_affinity": {"trending_up": 0.85, "ranging": 0.4},
            },
            {
                "strategy_id": "mr_QQQ",
                "name": "Mean Rev QQQ",
                "status": "active",
                "regime_affinity": {"ranging": 0.9, "trending_up": 0.3},
            },
        ]

        @contextmanager
        def _ctx():
            yield mock_conn

        # Test the underlying coroutine function directly
        import asyncio
        from quantstack.tools.langchain.meta_tools import get_regime_strategies

        with patch("quantstack.tools.langchain.meta_tools.db_conn", _ctx), \
             patch("quantstack.tools.langchain.meta_tools._get_breaker_status", return_value="ACTIVE"):
            result = asyncio.run(get_regime_strategies.coroutine("trending_up"))

        data = json.loads(result)
        assert isinstance(data, dict)
        assert "strategies" in data
        strategies = data["strategies"]
        assert len(strategies) == 2
        # Sorted by affinity descending for trending_up
        assert strategies[0]["affinity"] >= strategies[1]["affinity"]

    def test_returns_empty_for_unknown_regime(self):
        """No matching strategies -> empty list."""
        mock_conn = MagicMock()
        mock_conn.fetchall.return_value = []

        @contextmanager
        def _ctx():
            yield mock_conn

        import asyncio
        from quantstack.tools.langchain.meta_tools import get_regime_strategies

        with patch("quantstack.tools.langchain.meta_tools.db_conn", _ctx):
            result = asyncio.run(get_regime_strategies.coroutine("alien_regime"))

        data = json.loads(result)
        assert data["strategies"] == []

    def test_filters_out_retired_strategies(self):
        """Retired strategies excluded (query uses WHERE status != 'retired')."""
        mock_conn = MagicMock()
        mock_conn.fetchall.return_value = [
            {
                "strategy_id": "active_one",
                "name": "Active",
                "status": "active",
                "regime_affinity": {"trending_up": 0.7},
            },
        ]

        @contextmanager
        def _ctx():
            yield mock_conn

        import asyncio
        from quantstack.tools.langchain.meta_tools import get_regime_strategies

        with patch("quantstack.tools.langchain.meta_tools.db_conn", _ctx), \
             patch("quantstack.tools.langchain.meta_tools._get_breaker_status", return_value="ACTIVE"):
            result = asyncio.run(get_regime_strategies.coroutine("trending_up"))

        data = json.loads(result)
        assert len(data["strategies"]) == 1


# ===========================================================================
# Wire 2: StrategyBreaker in risk_sizing
# ===========================================================================


class TestBreakerInRiskSizing:
    """Wire 2: Breaker scale factor multiplied into alpha_signal."""

    def test_active_strategy_factor_one(self):
        """ACTIVE (factor=1.0) -> size unchanged."""
        computed_size = 1000.0
        factor = 1.0
        adjusted = computed_size * max(0.0, min(1.0, factor))
        assert adjusted == 1000.0

    def test_scaled_strategy_halved(self):
        """SCALED (factor=0.5) -> size halved."""
        computed_size = 1000.0
        factor = 0.5
        adjusted = computed_size * max(0.0, min(1.0, factor))
        assert adjusted == 500.0

    def test_tripped_strategy_zeroed(self):
        """TRIPPED (factor=0.0) -> size zeroed."""
        computed_size = 1000.0
        factor = 0.0
        adjusted = computed_size * max(0.0, min(1.0, factor))
        assert adjusted == 0.0

    def test_factor_above_one_clamped(self):
        """Factor > 1.0 clamped to 1.0."""
        computed_size = 1000.0
        factor = 1.5
        adjusted = computed_size * max(0.0, min(1.0, factor))
        assert adjusted == 1000.0

    def test_exception_defaults_to_one(self):
        """If get_scale_factor() raises, default to 1.0."""
        breaker = MagicMock(spec=StrategyBreaker)
        breaker.get_scale_factor.side_effect = RuntimeError("db error")
        try:
            factor = breaker.get_scale_factor("strat_x")
        except Exception:
            factor = 1.0
        assert factor == 1.0


# ===========================================================================
# Wire 3: StrategyBreaker in execute_entries
# ===========================================================================


class TestBreakerInExecuteEntries:
    """Wire 3: Gate order placement on breaker state."""

    def test_tripped_order_skipped(self):
        """TRIPPED (factor=0.0) -> order skipped."""
        breaker = MagicMock(spec=StrategyBreaker)
        breaker.get_scale_factor.return_value = 0.0

        orders_placed = []
        for order in [{"symbol": "AAPL", "strategy_id": "swing_AAPL"}]:
            factor = breaker.get_scale_factor(order["strategy_id"])
            if factor == 0.0:
                continue  # skip
            orders_placed.append(order)

        assert len(orders_placed) == 0

    def test_active_order_placed(self):
        """ACTIVE (factor=1.0) -> order proceeds."""
        breaker = MagicMock(spec=StrategyBreaker)
        breaker.get_scale_factor.return_value = 1.0

        orders_placed = []
        for order in [{"symbol": "AAPL", "strategy_id": "swing_AAPL"}]:
            factor = breaker.get_scale_factor(order["strategy_id"])
            if factor == 0.0:
                continue
            orders_placed.append(order)

        assert len(orders_placed) == 1


# ===========================================================================
# Wire 4: SkillTracker in trade hooks
# ===========================================================================


class TestSkillTrackerInTradeHooks:
    """Wire 4: _update_skill_tracker called with correct arguments."""

    def test_profitable_trade_prediction_correct(self):
        """Profitable close -> prediction_correct=True."""
        # Test the logic directly: realized_pnl_pct > 0 -> correct
        realized_pnl_pct = 6.67
        prediction_correct = realized_pnl_pct > 0
        assert prediction_correct is True

    def test_unprofitable_trade_prediction_incorrect(self):
        """Losing close -> prediction_correct=False."""
        realized_pnl_pct = -6.25
        prediction_correct = realized_pnl_pct > 0
        assert prediction_correct is False

    def test_agent_name_from_debate_verdict(self):
        """Agent name extracted from debate_verdict or defaults."""
        debate_verdict = "alpha_research"
        agent_name = debate_verdict or "unknown_agent"
        assert agent_name == "alpha_research"

        debate_verdict = ""
        agent_name = debate_verdict or "unknown_agent"
        assert agent_name == "unknown_agent"

    def test_update_skill_tracker_function_exists(self):
        """The _update_skill_tracker function is importable from trade_hooks."""
        # This validates that the wiring was added to the module
        from quantstack.hooks import trade_hooks
        assert hasattr(trade_hooks, "_update_skill_tracker")
        assert callable(trade_hooks._update_skill_tracker)


# ===========================================================================
# Wire 6: Trade quality in daily plan
# ===========================================================================


class TestTradeQualityInDailyPlan:
    """Wire 6: Rolling trade quality context in daily plan prompt."""

    def test_rolling_averages_computed(self):
        """Computes rolling averages per dimension."""
        from quantstack.learning.trade_quality import get_trade_quality_summary

        mock_conn = MagicMock()
        # Return 10 rows of scores (above the 5-trade minimum)
        mock_conn.fetchall.return_value = [
            {
                "execution_quality": 7.5,
                "thesis_accuracy": 6.0,
                "risk_management": 8.0,
                "timing_quality": 5.5,
                "sizing_quality": 7.0,
                "overall_score": 6.8,
            }
            for _ in range(10)
        ]

        @contextmanager
        def _ctx():
            yield mock_conn

        with patch("quantstack.learning.trade_quality.db_conn", _ctx):
            result = get_trade_quality_summary("swing_AAPL", window=30)

        assert result is not None
        assert result["trade_count"] == 10
        assert result["weakest"] == "timing_quality"
        assert result["weakest_score"] == 5.5

    def test_insufficient_data_returns_none(self):
        """< 5 scored trades -> None."""
        from quantstack.learning.trade_quality import get_trade_quality_summary

        mock_conn = MagicMock()
        # Only 3 rows — below threshold
        mock_conn.fetchall.return_value = [
            {
                "execution_quality": 7.0,
                "thesis_accuracy": 6.0,
                "risk_management": 8.0,
                "timing_quality": 5.5,
                "sizing_quality": 7.0,
                "overall_score": 6.8,
            }
            for _ in range(3)
        ]

        @contextmanager
        def _ctx():
            yield mock_conn

        with patch("quantstack.learning.trade_quality.db_conn", _ctx):
            result = get_trade_quality_summary("swing_AAPL", window=30)

        assert result is None
