"""Shadow mode tests — ExecutionMonitor observe-only mode and ShadowComparator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from quantstack.execution.shadow_comparator import (
    ShadowComparator,
    ShadowExitRecord,
)
from quantstack.execution.execution_monitor import ExecutionMonitor
from quantstack.execution.portfolio_state import Position
from quantstack.execution.price_feed import PaperPriceFeed
from quantstack.holding_period import HoldingType

ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Shadow mode flag
# ---------------------------------------------------------------------------


class TestShadowModeFlag:

    def _make_position(self, **kw) -> Position:
        defaults = dict(
            symbol="TEST", quantity=10, avg_cost=100.0, side="long",
            opened_at=datetime(2026, 4, 6, 10, 0, tzinfo=ET),
            stop_price=95.0, target_price=110.0, time_horizon="swing",
            entry_atr=2.0, strategy_id="test_strat",
        )
        defaults.update(kw)
        return Position(**defaults)

    @pytest.mark.asyncio
    async def test_shadow_does_not_call_broker(self):
        """Shadow mode: broker.execute() never called on SL breach."""
        pos = self._make_position(stop_price=95.0)
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [pos]
        portfolio.update_monitor_state.return_value = True
        broker = MagicMock()
        events = [("TEST", 94.0, datetime(2026, 4, 6, 10, 5, tzinfo=ET))]
        feed = PaperPriceFeed(events=events, replay_speed=0.0)

        monitor = ExecutionMonitor(
            broker=broker, price_feed=feed, portfolio_state=portfolio,
            shadow_mode=True, poll_interval=100.0, reconcile_interval=100.0,
        )
        await monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()

        broker.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_shadow_calls_broker(self):
        """Non-shadow mode: broker.execute() IS called on SL breach."""
        pos = self._make_position(stop_price=95.0)
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [pos]
        portfolio.update_monitor_state.return_value = True
        broker = MagicMock()
        fill = MagicMock(rejected=False, filled_quantity=10, fill_price=94.0)
        broker.execute.return_value = fill
        events = [("TEST", 94.0, datetime(2026, 4, 6, 10, 5, tzinfo=ET))]
        feed = PaperPriceFeed(events=events, replay_speed=0.0)

        monitor = ExecutionMonitor(
            broker=broker, price_feed=feed, portfolio_state=portfolio,
            shadow_mode=False, poll_interval=100.0, reconcile_interval=100.0,
        )
        await monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()

        broker.execute.assert_called()

    @pytest.mark.asyncio
    async def test_shadow_does_not_set_exit_pending(self):
        """Shadow mode should NOT set exit_pending on MonitoredPosition."""
        pos = self._make_position(stop_price=95.0)
        portfolio = MagicMock()
        portfolio.get_positions.return_value = [pos]
        portfolio.update_monitor_state.return_value = True
        broker = MagicMock()
        events = [("TEST", 94.0, datetime(2026, 4, 6, 10, 5, tzinfo=ET))]
        feed = PaperPriceFeed(events=events, replay_speed=0.0)

        monitor = ExecutionMonitor(
            broker=broker, price_feed=feed, portfolio_state=portfolio,
            shadow_mode=True, poll_interval=100.0, reconcile_interval=100.0,
        )
        await monitor.start()
        await asyncio.sleep(0.2)

        test_pos = monitor._positions.get("TEST")
        assert test_pos is not None
        assert test_pos.exit_pending is False

        await monitor.stop()


# ---------------------------------------------------------------------------
# ShadowComparator
# ---------------------------------------------------------------------------


class TestShadowComparator:

    def _rec(self, source: str, **kw) -> ShadowExitRecord:
        defaults = dict(
            symbol="TEST", source=source, reason="stop_loss",
            trigger_price=94.0, entry_price=100.0, unrealized_pnl=-60.0,
            holding_type=HoldingType.SWING, strategy_id="test",
            timestamp=datetime(2026, 4, 6, 10, 5, tzinfo=ET),
        )
        defaults.update(kw)
        return ShadowExitRecord(**defaults)

    def test_record_both_sources(self):
        """ShadowComparator.record() accepts entries from both sources."""
        comp = ShadowComparator()
        comp.record(self._rec("intraday_pm"))
        comp.record(self._rec("execution_monitor"))
        # Should have auto-compared
        assert len(comp._comparisons) == 1

    def test_matching_decisions_agreement(self):
        """Same symbol, same reason, within window → match=True."""
        comp = ShadowComparator(comparison_window_seconds=5.0)
        t1 = datetime(2026, 4, 6, 10, 5, 0, tzinfo=ET)
        t2 = datetime(2026, 4, 6, 10, 5, 2, tzinfo=ET)  # 2s later
        comp.record(self._rec("intraday_pm", timestamp=t1))
        comp.record(self._rec("execution_monitor", timestamp=t2))

        assert len(comp._comparisons) == 1
        assert comp._comparisons[0].match is True

    def test_monitor_only_divergence(self):
        """Monitor fires, PM doesn't → divergence_type='monitor_only'."""
        comp = ShadowComparator()
        comp.record(self._rec("execution_monitor"))
        result = comp.compare("TEST")
        assert result is not None
        assert result.match is False
        assert result.divergence_type == "monitor_only"

    def test_pm_only_divergence(self):
        """PM fires, monitor doesn't → divergence_type='pm_only'."""
        comp = ShadowComparator()
        comp.record(self._rec("intraday_pm"))
        result = comp.compare("TEST")
        assert result is not None
        assert result.match is False
        assert result.divergence_type == "pm_only"

    def test_reason_mismatch(self):
        """Same symbol, different reasons → divergence_type='reason_mismatch'."""
        comp = ShadowComparator(comparison_window_seconds=5.0)
        t = datetime(2026, 4, 6, 10, 5, 0, tzinfo=ET)
        comp.record(self._rec("intraday_pm", reason="trailing_stop", timestamp=t))
        comp.record(self._rec("execution_monitor", reason="stop_loss", timestamp=t))

        assert comp._comparisons[0].match is False
        assert comp._comparisons[0].divergence_type == "reason_mismatch"

    def test_price_delta_captured(self):
        """Matching decisions with different prices → price_delta recorded."""
        comp = ShadowComparator(comparison_window_seconds=5.0)
        t = datetime(2026, 4, 6, 10, 5, 0, tzinfo=ET)
        comp.record(self._rec("intraday_pm", trigger_price=94.5, timestamp=t))
        comp.record(self._rec("execution_monitor", trigger_price=94.0, timestamp=t))

        assert comp._comparisons[0].price_delta == pytest.approx(0.5)

    def test_summary_metrics(self):
        """summary() returns correct aggregate metrics."""
        comp = ShadowComparator(comparison_window_seconds=5.0)
        t = datetime(2026, 4, 6, 10, 5, 0, tzinfo=ET)

        # Agreement
        comp.record(self._rec("intraday_pm", symbol="SPY", timestamp=t))
        comp.record(self._rec("execution_monitor", symbol="SPY", timestamp=t))

        # Divergence
        comp.record(self._rec("intraday_pm", symbol="AAPL", reason="trailing_stop", timestamp=t))
        comp.record(self._rec("execution_monitor", symbol="AAPL", reason="stop_loss", timestamp=t))

        stats = comp.summary()
        assert stats["total_exits"] == 2
        assert stats["agreements"] == 1
        assert stats["divergences"] == 1
        assert stats["agreement_rate"] == pytest.approx(50.0)

    def test_flush_clears_state(self):
        """flush_to_log() clears comparisons and records."""
        comp = ShadowComparator()
        t = datetime(2026, 4, 6, 10, 5, 0, tzinfo=ET)
        comp.record(self._rec("intraday_pm", timestamp=t))
        comp.record(self._rec("execution_monitor", timestamp=t))
        comp.flush_to_log()

        assert len(comp._comparisons) == 0
        assert len(comp._records) == 0
