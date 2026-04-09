"""Tests for unrealized P&L circuit breaker in ExecutionMonitor (QS-E5).

Verifies:
  - No action when unrealized P&L above all thresholds
  - Halt triggered at -1.5% (writes sentinel, blocks new entries)
  - Systematic exit triggered at -2.5%
  - Kill switch triggered at -5.0%
  - Velocity detection triggers exit on fast drawdown
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from quantstack.execution.execution_monitor import ExecutionMonitor, MonitoredPosition

ET = pytz.timezone("US/Eastern")


@dataclass
class _FakePosition:
    symbol: str
    unrealized_pnl: float
    quantity: int = 100
    side: str = "long"
    current_price: float = 100.0


@dataclass
class _FakeSnapshot:
    total_equity: float = 100_000.0


def _make_monitor(
    positions: list[_FakePosition] | None = None,
    equity: float = 100_000.0,
    sentinel_path: str | None = None,
) -> ExecutionMonitor:
    """Create an ExecutionMonitor with mocked dependencies."""
    broker = MagicMock()
    feed = AsyncMock()
    portfolio = MagicMock()
    portfolio.get_positions.return_value = positions or []
    portfolio.get_snapshot.return_value = _FakeSnapshot(total_equity=equity)

    monitor = ExecutionMonitor(
        broker=broker,
        price_feed=feed,
        portfolio_state=portfolio,
    )
    monitor._pnl_halt_pct = -0.015
    monitor._pnl_exit_pct = -0.025
    monitor._pnl_emergency_pct = -0.05
    monitor._pnl_velocity_pct = -0.01
    monitor._pnl_velocity_window = 60

    if sentinel_path:
        monitor._pnl_sentinel = Path(sentinel_path)

    return monitor


class TestNoAction:
    """When P&L is above all thresholds, no action is taken."""

    @pytest.mark.asyncio
    async def test_no_action_when_pnl_above_thresholds(self, tmp_path):
        positions = [_FakePosition("AAPL", unrealized_pnl=-500.0)]  # -0.5%
        monitor = _make_monitor(
            positions=positions,
            sentinel_path=str(tmp_path / "halt"),
        )

        await monitor._check_unrealized_pnl(datetime.now(ET))

        assert not monitor._pnl_halted
        assert not (tmp_path / "halt").exists()

    @pytest.mark.asyncio
    async def test_no_action_when_no_positions(self, tmp_path):
        monitor = _make_monitor(
            positions=[],
            sentinel_path=str(tmp_path / "halt"),
        )

        await monitor._check_unrealized_pnl(datetime.now(ET))
        assert not monitor._pnl_halted


class TestPnlHalt:
    """Halt triggered at -1.5% unrealized."""

    @pytest.mark.asyncio
    async def test_halt_at_threshold(self, tmp_path):
        # -1600 on 100k equity = -1.6% → breaches -1.5%
        positions = [_FakePosition("AAPL", unrealized_pnl=-1600.0)]
        monitor = _make_monitor(
            positions=positions,
            sentinel_path=str(tmp_path / "halt"),
        )

        await monitor._check_unrealized_pnl(datetime.now(ET))

        assert monitor._pnl_halted
        assert (tmp_path / "halt").exists()

    @pytest.mark.asyncio
    async def test_halt_only_fires_once(self, tmp_path):
        positions = [_FakePosition("AAPL", unrealized_pnl=-1600.0)]
        monitor = _make_monitor(
            positions=positions,
            sentinel_path=str(tmp_path / "halt"),
        )

        await monitor._check_unrealized_pnl(datetime.now(ET))
        assert monitor._pnl_halted

        # Second call should not re-log or re-write
        sentinel_mtime = (tmp_path / "halt").stat().st_mtime
        await monitor._check_unrealized_pnl(
            datetime.now(ET) + timedelta(seconds=5)
        )
        assert (tmp_path / "halt").stat().st_mtime == sentinel_mtime


class TestSystematicExit:
    """Systematic exit triggered at -2.5% unrealized."""

    @pytest.mark.asyncio
    async def test_systematic_exit_at_threshold(self, tmp_path):
        # -2600 on 100k = -2.6%
        positions = [_FakePosition("AAPL", unrealized_pnl=-2600.0)]
        monitor = _make_monitor(
            positions=positions,
            sentinel_path=str(tmp_path / "halt"),
        )
        # Populate position cache
        mp = MagicMock(spec=MonitoredPosition)
        mp.symbol = "AAPL"
        mp.exit_pending = False
        mp.current_price = 97.4
        monitor._positions = {"AAPL": mp}

        with patch.object(
            monitor, "_submit_exit", new_callable=AsyncMock
        ) as mock_exit:
            await monitor._check_unrealized_pnl(datetime.now(ET))

            mock_exit.assert_called_once()
            assert monitor._pnl_halted
            assert (tmp_path / "halt").exists()


class TestEmergencyLiquidation:
    """Kill switch triggered at -5.0% unrealized."""

    @pytest.mark.asyncio
    async def test_emergency_at_threshold(self, tmp_path):
        # -5100 on 100k = -5.1%
        positions = [_FakePosition("AAPL", unrealized_pnl=-5100.0)]
        monitor = _make_monitor(
            positions=positions,
            sentinel_path=str(tmp_path / "halt"),
        )

        with patch(
            "quantstack.execution.kill_switch.get_kill_switch"
        ) as mock_ks_fn:
            mock_ks = MagicMock()
            mock_ks_fn.return_value = mock_ks

            await monitor._check_unrealized_pnl(datetime.now(ET))

            mock_ks.trigger.assert_called_once()
            trigger_reason = mock_ks.trigger.call_args[0][0]
            assert "emergency" in trigger_reason


class TestVelocityDetection:
    """Velocity breach triggers systematic exit even above absolute thresholds."""

    @pytest.mark.asyncio
    async def test_velocity_breach_triggers_exit(self, tmp_path):
        # Current P&L at -1.5% but history shows 0% 30s ago → delta = -1.5% > -1% threshold
        monitor = _make_monitor(
            positions=[_FakePosition("AAPL", unrealized_pnl=-1500.0)],
            sentinel_path=str(tmp_path / "halt"),
        )

        # Seed history: was at 0% 30 seconds ago
        t0 = datetime.now(ET) - timedelta(seconds=30)
        monitor._pnl_history = [(t0, 0.0)]

        mp = MagicMock(spec=MonitoredPosition)
        mp.symbol = "AAPL"
        mp.exit_pending = False
        mp.current_price = 98.5
        monitor._positions = {"AAPL": mp}

        with patch.object(
            monitor, "_submit_exit", new_callable=AsyncMock
        ) as mock_exit:
            await monitor._check_unrealized_pnl(datetime.now(ET))

            mock_exit.assert_called_once()
            assert monitor._pnl_halted
