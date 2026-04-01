# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for feedback.py (get_fill_quality, get_position_monitor)
and attribution.py (get_daily_equity, get_strategy_pnl).

Mocks pg_conn() and live_db_or_error() at data boundaries.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.quant_pod.mcp.conftest import _fn, assert_standard_response, synthetic_ohlcv


# ---------------------------------------------------------------------------
# feedback.py — get_fill_quality
# ---------------------------------------------------------------------------

class TestGetFillQuality:

    @pytest.mark.asyncio
    async def test_fill_found_returns_quality(self):
        """Happy path: fill exists in DB, VWAP comparison attempted."""
        from quantstack.mcp.tools.feedback import get_fill_quality

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (
            "order-123", "SPY", "buy", 450.25, 10,
            3.5, 0.65, "2024-01-15 10:30:00",
        )

        mock_store = MagicMock()
        mock_store.load_ohlcv.return_value = None  # no VWAP data
        mock_store.close = MagicMock()

        with (
            patch("quantstack.mcp.tools.feedback.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.feedback.pg_conn") as mock_pg,
            patch("quantstack.mcp.tools.feedback._get_reader", return_value=mock_store),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_fill_quality)(order_id="order-123")

        assert result["success"] is True
        assert result["order_id"] == "order-123"
        assert result["symbol"] == "SPY"
        assert result["fill_price"] == 450.25
        assert result["slippage_bps"] == 3.5
        assert "quality_note" in result

    @pytest.mark.asyncio
    async def test_fill_not_found(self):
        """Fill not in DB returns error."""
        from quantstack.mcp.tools.feedback import get_fill_quality

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None

        with (
            patch("quantstack.mcp.tools.feedback.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.feedback.pg_conn") as mock_pg,
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_fill_quality)(order_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fill_quality_no_ctx(self):
        """Without initialized context, returns error."""
        from quantstack.mcp.tools.feedback import get_fill_quality

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.feedback.live_db_or_error", return_value=(None, err)):
            result = await _fn(get_fill_quality)(order_id="any")

        assert result["success"] is False


# ---------------------------------------------------------------------------
# feedback.py — get_position_monitor
# ---------------------------------------------------------------------------

class TestGetPositionMonitor:

    @pytest.mark.asyncio
    async def test_no_position(self):
        """When no position exists, returns has_position=False."""
        from quantstack.mcp.tools.feedback import get_position_monitor

        mock_ctx = MagicMock()
        mock_ctx.portfolio.get_position.return_value = None

        with patch("quantstack.mcp.tools.feedback.live_db_or_error", return_value=(mock_ctx, None)):
            result = await _fn(get_position_monitor)(symbol="NOPOS")

        assert result["success"] is True
        assert result["has_position"] is False

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        """Without initialized context, returns error."""
        from quantstack.mcp.tools.feedback import get_position_monitor

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.feedback.live_db_or_error", return_value=(None, err)):
            result = await _fn(get_position_monitor)(symbol="SPY")

        assert result["success"] is False


# ---------------------------------------------------------------------------
# attribution.py — get_daily_equity
# ---------------------------------------------------------------------------

class TestGetDailyEquity:

    @pytest.mark.asyncio
    async def test_returns_equity_curve(self):
        """Happy path: EquityTracker returns data."""
        from quantstack.mcp.tools.attribution import get_daily_equity

        mock_tracker = MagicMock()
        mock_tracker.get_equity_curve.return_value = [
            {"date": date(2024, 1, 1), "nav": 100000, "daily_return": 0.0, "drawdown": 0.0},
            {"date": date(2024, 1, 2), "nav": 100500, "daily_return": 0.005, "drawdown": 0.0},
        ]
        mock_tracker.get_summary.return_value = {"total_return": 0.5, "sharpe": 1.2}

        mock_conn = MagicMock()

        with (
            patch("quantstack.mcp.tools.attribution.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.attribution.pg_conn") as mock_pg,
            # Patch at the source module — the deferred import resolves from here
            patch("quantstack.performance.equity_tracker.EquityTracker", return_value=mock_tracker),
            patch("quantstack.performance.benchmark.BenchmarkTracker") as mock_bench,
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)
            mock_bench.return_value.get_comparison.return_value = []

            result = await _fn(get_daily_equity)()

        assert result["success"] is True
        assert result["count"] == 2
        # Dates should be serialized to strings
        assert isinstance(result["equity_curve"][0]["date"], str)

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.attribution import get_daily_equity

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.attribution.live_db_or_error", return_value=(None, err)):
            result = await _fn(get_daily_equity)()

        assert result["success"] is False


# ---------------------------------------------------------------------------
# attribution.py — get_strategy_pnl
# ---------------------------------------------------------------------------

class TestGetStrategyPnl:

    @pytest.mark.asyncio
    async def test_returns_strategy_aggregates(self):
        """Happy path: EquityTracker returns per-strategy data."""
        from quantstack.mcp.tools.attribution import get_strategy_pnl

        mock_tracker = MagicMock()
        mock_tracker.get_strategy_pnl.return_value = [
            {
                "strategy_id": "strat_1",
                "date": date(2024, 1, 1),
                "realized_pnl": 500.0,
                "num_trades": 3,
                "win_count": 2,
                "loss_count": 1,
            },
            {
                "strategy_id": "strat_1",
                "date": date(2024, 1, 2),
                "realized_pnl": -200.0,
                "num_trades": 2,
                "win_count": 0,
                "loss_count": 2,
            },
        ]

        mock_conn = MagicMock()

        with (
            patch("quantstack.mcp.tools.attribution.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.attribution.pg_conn") as mock_pg,
            patch("quantstack.performance.equity_tracker.EquityTracker", return_value=mock_tracker),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_strategy_pnl)()

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["strategy_aggregates"]) == 1
        agg = result["strategy_aggregates"][0]
        assert agg["strategy_id"] == "strat_1"
        assert agg["total_realized_pnl"] == 300.0
        assert agg["total_trades"] == 5
        assert agg["total_wins"] == 2

    @pytest.mark.asyncio
    async def test_empty_pnl(self):
        """No trades returns empty list."""
        from quantstack.mcp.tools.attribution import get_strategy_pnl

        mock_tracker = MagicMock()
        mock_tracker.get_strategy_pnl.return_value = []
        mock_conn = MagicMock()

        with (
            patch("quantstack.mcp.tools.attribution.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.attribution.pg_conn") as mock_pg,
            patch("quantstack.performance.equity_tracker.EquityTracker", return_value=mock_tracker),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_strategy_pnl)()

        assert result["success"] is True
        assert result["count"] == 0
        assert result["strategy_aggregates"] == []

    @pytest.mark.asyncio
    async def test_win_rate_zero_trades(self):
        """win_rate should be 0.0 when total_trades is 0 (no division by zero)."""
        from quantstack.mcp.tools.attribution import get_strategy_pnl

        mock_tracker = MagicMock()
        mock_tracker.get_strategy_pnl.return_value = [
            {
                "strategy_id": "empty_strat",
                "date": date(2024, 1, 1),
                "realized_pnl": 0.0,
                "num_trades": 0,
                "win_count": 0,
                "loss_count": 0,
            },
        ]
        mock_conn = MagicMock()

        with (
            patch("quantstack.mcp.tools.attribution.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.attribution.pg_conn") as mock_pg,
            patch("quantstack.performance.equity_tracker.EquityTracker", return_value=mock_tracker),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_strategy_pnl)()

        assert result["success"] is True
        agg = result["strategy_aggregates"][0]
        assert agg["win_rate"] == 0.0  # no ZeroDivisionError
