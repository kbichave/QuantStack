"""Integration tests for ExecutionMonitor wired to price feed and broker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from quantstack.execution.execution_monitor import ExecutionMonitor, MonitoredPosition
from quantstack.execution.portfolio_state import Position
from quantstack.execution.price_feed import PaperPriceFeed

ET = pytz.timezone("US/Eastern")


def _make_position(**overrides) -> Position:
    defaults = dict(
        symbol="TEST",
        quantity=10,
        avg_cost=100.0,
        side="long",
        opened_at=datetime(2026, 4, 6, 10, 0, tzinfo=ET),
        stop_price=95.0,
        target_price=110.0,
        time_horizon="swing",
        strategy_id="test_strategy",
        instrument_type="equity",
        entry_atr=2.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def _make_portfolio_state(positions: list[Position]) -> MagicMock:
    state = MagicMock()
    state.get_positions.return_value = positions
    state.get_position.side_effect = lambda sym: next(
        (p for p in positions if p.symbol == sym), None
    )
    state.update_monitor_state.return_value = True
    return state


def _make_broker() -> MagicMock:
    broker = MagicMock()
    fill = MagicMock()
    fill.rejected = False
    fill.filled_quantity = 10
    fill.fill_price = 94.0
    fill.reject_reason = None
    broker.execute.return_value = fill
    return broker


@pytest.mark.asyncio
async def test_monitor_full_exit_flow():
    """Monitor detects stop-loss breach and submits exit order."""
    pos = _make_position(stop_price=95.0)
    portfolio = _make_portfolio_state([pos])
    broker = _make_broker()

    # Price feed delivers price below stop
    events = [
        ("TEST", 94.0, datetime(2026, 4, 6, 10, 5, tzinfo=ET)),
    ]
    feed = PaperPriceFeed(events=events, replay_speed=0.0)

    monitor = ExecutionMonitor(
        broker=broker,
        price_feed=feed,
        portfolio_state=portfolio,
        poll_interval=100.0,  # won't fire during test
        reconcile_interval=100.0,
    )
    await monitor.start()

    # Wait for the replay to complete
    await asyncio.sleep(0.2)
    await monitor.stop()

    # Broker should have been called for the exit
    assert broker.execute.called
    req = broker.execute.call_args[0][0]
    assert req.symbol == "TEST"
    assert req.side == "sell"  # closing a long


@pytest.mark.asyncio
async def test_shadow_mode_no_execution():
    """Shadow mode logs but does not execute exits."""
    pos = _make_position(stop_price=95.0)
    portfolio = _make_portfolio_state([pos])
    broker = _make_broker()

    events = [
        ("TEST", 94.0, datetime(2026, 4, 6, 10, 5, tzinfo=ET)),
    ]
    feed = PaperPriceFeed(events=events, replay_speed=0.0)

    monitor = ExecutionMonitor(
        broker=broker,
        price_feed=feed,
        portfolio_state=portfolio,
        shadow_mode=True,
        poll_interval=100.0,
        reconcile_interval=100.0,
    )
    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    # Broker should NOT have been called
    broker.execute.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_crash_recovery():
    """Monitor reconstructs state from DB on restart."""
    pos = _make_position(stop_price=95.0)
    portfolio = _make_portfolio_state([pos])
    broker = _make_broker()
    feed = PaperPriceFeed(events=[], replay_speed=0.0)

    monitor = ExecutionMonitor(
        broker=broker,
        price_feed=feed,
        portfolio_state=portfolio,
        poll_interval=100.0,
        reconcile_interval=100.0,
    )

    # Start → loads positions
    await monitor.start()
    assert "TEST" in monitor._positions
    await monitor.stop()

    # "Crash" — positions cleared
    assert len(monitor._positions) == 0

    # Restart — should reload from DB
    await monitor.start()
    assert "TEST" in monitor._positions
    await monitor.stop()


@pytest.mark.asyncio
async def test_take_profit_exit():
    """Monitor detects take-profit and submits exit."""
    pos = _make_position(target_price=110.0)
    portfolio = _make_portfolio_state([pos])
    broker = _make_broker()

    events = [
        ("TEST", 111.0, datetime(2026, 4, 6, 10, 5, tzinfo=ET)),
    ]
    feed = PaperPriceFeed(events=events, replay_speed=0.0)

    monitor = ExecutionMonitor(
        broker=broker,
        price_feed=feed,
        portfolio_state=portfolio,
        poll_interval=100.0,
        reconcile_interval=100.0,
    )
    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    assert broker.execute.called
