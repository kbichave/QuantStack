"""Tests for ExecutionMonitor crash recovery and reconciliation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from quantstack.execution.execution_monitor import ExecutionMonitor, MonitoredPosition
from quantstack.execution.portfolio_state import Position

ET = pytz.timezone("US/Eastern")


def _make_db_position(symbol: str, quantity: int = 100) -> Position:
    """Create a Position as it would come from PortfolioState."""
    return Position(
        symbol=symbol,
        quantity=quantity,
        avg_cost=100.0,
        side="long",
        opened_at=datetime(2026, 4, 1, 10, 0, tzinfo=ET),
        time_horizon="short_swing",
        stop_price=95.0,
        target_price=110.0,
        entry_atr=2.0,
        strategy_id=f"{symbol}_strat",
    )


@pytest.fixture()
def mock_broker():
    return MagicMock()


@pytest.fixture()
def mock_feed():
    feed = AsyncMock()
    feed.subscribe = AsyncMock()
    feed.unsubscribe = AsyncMock()
    feed.start = AsyncMock()
    feed.stop = AsyncMock()
    return feed


def _make_monitor(portfolio, mock_broker, mock_feed):
    """Create an ExecutionMonitor without starting background tasks."""
    monitor = ExecutionMonitor(
        broker=mock_broker,
        price_feed=mock_feed,
        portfolio_state=portfolio,
    )
    return monitor


class TestCrashRecovery:

    @pytest.mark.asyncio
    async def test_start_loads_positions_from_db(self, mock_broker, mock_feed):
        """start() reads positions from DB and builds MonitoredPosition objects."""
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [
            _make_db_position("AAPL"),
            _make_db_position("NVDA"),
            _make_db_position("TSLA"),
        ]

        monitor = _make_monitor(portfolio, mock_broker, mock_feed)
        await monitor.start()

        assert len(monitor._positions) == 3
        assert "AAPL" in monitor._positions
        assert "NVDA" in monitor._positions
        assert "TSLA" in monitor._positions
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_start_subscribes_to_feeds(self, mock_broker, mock_feed):
        """start() subscribes to price feeds for all monitored symbols."""
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [
            _make_db_position("AAPL"),
            _make_db_position("NVDA"),
        ]

        monitor = _make_monitor(portfolio, mock_broker, mock_feed)
        await monitor.start()

        mock_feed.subscribe.assert_called_once()
        subscribed_symbols = set(mock_feed.subscribe.call_args[0][0])
        assert subscribed_symbols == {"AAPL", "NVDA"}
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_start_empty_db_runs_idle(self, mock_broker, mock_feed):
        """start() with no open positions → subscribes to nothing."""
        portfolio = MagicMock()
        portfolio.get_positions.return_value = []

        monitor = _make_monitor(portfolio, mock_broker, mock_feed)
        await monitor.start()

        assert len(monitor._positions) == 0
        mock_feed.subscribe.assert_not_called()
        await monitor.stop()


class TestPollPositions:

    @pytest.mark.asyncio
    async def test_adds_new_positions(self, mock_broker, mock_feed):
        """poll detects new position in DB → adds to cache."""
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [_make_db_position("AAPL")]

        monitor = _make_monitor(portfolio, mock_broker, mock_feed)
        await monitor.start()
        assert len(monitor._positions) == 1

        # DB now has 2 positions
        portfolio.get_positions.return_value = [
            _make_db_position("AAPL"),
            _make_db_position("NVDA"),
        ]
        await monitor._poll_positions()

        assert len(monitor._positions) == 2
        assert "NVDA" in monitor._positions
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_removes_closed_positions(self, mock_broker, mock_feed):
        """poll detects position gone from DB → removes from cache."""
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [
            _make_db_position("AAPL"),
            _make_db_position("NVDA"),
        ]

        monitor = _make_monitor(portfolio, mock_broker, mock_feed)
        await monitor.start()
        assert len(monitor._positions) == 2

        # NVDA closed
        portfolio.get_positions.return_value = [_make_db_position("AAPL")]
        await monitor._poll_positions()

        assert len(monitor._positions) == 1
        assert "NVDA" not in monitor._positions
        await monitor.stop()
