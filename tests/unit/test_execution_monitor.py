"""Tests for ExecutionMonitor — rule evaluation, exit submission, exit pending."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytz

from quantstack.execution.execution_monitor import (
    MonitoredPosition,
    _time_horizon_to_holding_type,
)
from quantstack.holding_period import HOLDING_CONFIGS, HoldingType

ET = pytz.timezone("US/Eastern")


def _make_position(
    symbol: str = "SPY",
    side: str = "long",
    quantity: int = 100,
    holding_type: HoldingType = HoldingType.SHORT_SWING,
    entry_price: float = 100.0,
    stop_price: float | None = 95.0,
    target_price: float | None = 110.0,
    trailing_atr_mult: float = 0.0,
    entry_atr: float = 2.0,
    high_water_mark: float | None = None,
    exit_deadline: datetime | None = None,
    instrument_type: str = "equity",
    exit_pending: bool = False,
    entry_time: datetime | None = None,
) -> MonitoredPosition:
    """Helper to build a MonitoredPosition with sensible defaults."""
    return MonitoredPosition(
        symbol=symbol,
        side=side,
        quantity=quantity,
        holding_type=holding_type,
        entry_price=entry_price,
        entry_time=entry_time or datetime.now(ET),
        stop_price=stop_price,
        target_price=target_price,
        trailing_atr_mult=trailing_atr_mult,
        entry_atr=entry_atr,
        high_water_mark=high_water_mark or entry_price,
        exit_deadline=exit_deadline,
        instrument_type=instrument_type,
        exit_pending=exit_pending,
        strategy_id="test_strat",
    )


def _now_et() -> datetime:
    return datetime.now(ET)


# ---------------------------------------------------------------------------
# MonitoredPosition construction
# ---------------------------------------------------------------------------


class TestMonitoredPositionConstruction:

    def test_from_portfolio_position(self):
        """from_portfolio_position maps all fields correctly."""
        from quantstack.execution.portfolio_state import Position

        pos = Position(
            symbol="AAPL",
            quantity=50,
            avg_cost=175.0,
            side="long",
            opened_at=datetime(2026, 4, 1, 10, 0, tzinfo=ET),
            time_horizon="short_swing",
            stop_price=170.0,
            target_price=185.0,
            trailing_stop=1.5,
            entry_atr=3.0,
            instrument_type="equity",
            strategy_id="aapl_swing_1",
        )
        mp = MonitoredPosition.from_portfolio_position(pos)
        assert mp.symbol == "AAPL"
        assert mp.side == "long"
        assert mp.quantity == 50
        assert mp.holding_type == HoldingType.SHORT_SWING
        assert mp.entry_price == 175.0
        assert mp.stop_price == 170.0
        assert mp.target_price == 185.0
        assert mp.entry_atr == 3.0
        assert mp.strategy_id == "aapl_swing_1"
        assert mp.instrument_type == "equity"

    def test_exit_deadline_computed_correctly(self):
        """Exit deadline uses max_bars * bar_timeframe_hours."""
        from quantstack.execution.portfolio_state import Position

        entry = datetime(2026, 4, 1, 10, 0, tzinfo=ET)
        pos = Position(
            symbol="SPY",
            quantity=100,
            avg_cost=450.0,
            side="long",
            opened_at=entry,
            time_horizon="short_swing",
            entry_atr=5.0,
        )
        mp = MonitoredPosition.from_portfolio_position(pos)
        # SHORT_SWING: max_bars=30, bar=4h → 120h = 5 days
        expected = entry + timedelta(hours=30 * 4)
        assert mp.exit_deadline == expected

    def test_time_horizon_mapping(self):
        """time_horizon strings map to correct HoldingType."""
        assert _time_horizon_to_holding_type("intraday") == HoldingType.INTRADAY
        assert _time_horizon_to_holding_type("short_swing") == HoldingType.SHORT_SWING
        assert _time_horizon_to_holding_type("swing") == HoldingType.SWING
        assert _time_horizon_to_holding_type("position") == HoldingType.POSITION
        assert _time_horizon_to_holding_type("investment") == HoldingType.POSITION
        # Unknown defaults to SWING
        assert _time_horizon_to_holding_type("unknown") == HoldingType.SWING


# ---------------------------------------------------------------------------
# Rule evaluation: Stop Loss
# ---------------------------------------------------------------------------


class TestStopLoss:

    def test_long_price_below_stop(self):
        """Long position, price drops below stop → exit."""
        pos = _make_position(stop_price=95.0, side="long")
        should_exit, reason = pos.evaluate_rules(94.50, _now_et())
        assert should_exit
        assert reason == "stop_loss"

    def test_short_price_above_stop(self):
        """Short position, price rises above stop → exit."""
        pos = _make_position(stop_price=105.0, side="short")
        should_exit, reason = pos.evaluate_rules(106.0, _now_et())
        assert should_exit
        assert reason == "stop_loss"

    def test_price_at_exactly_stop(self):
        """Price at exactly stop_price → exit (boundary)."""
        pos = _make_position(stop_price=95.0, side="long")
        should_exit, reason = pos.evaluate_rules(95.0, _now_et())
        assert should_exit
        assert reason == "stop_loss"

    def test_price_above_stop_no_exit(self):
        """Price above stop → no exit."""
        pos = _make_position(stop_price=95.0, side="long")
        should_exit, reason = pos.evaluate_rules(96.0, _now_et())
        assert not should_exit


# ---------------------------------------------------------------------------
# Rule evaluation: Take Profit
# ---------------------------------------------------------------------------


class TestTakeProfit:

    def test_long_price_above_target(self):
        """Long position, price above target → exit."""
        pos = _make_position(target_price=110.0, side="long")
        should_exit, reason = pos.evaluate_rules(111.0, _now_et())
        assert should_exit
        assert reason == "take_profit"

    def test_short_price_below_target(self):
        """Short position, price below target → exit."""
        pos = _make_position(target_price=90.0, side="short", stop_price=110.0)
        should_exit, reason = pos.evaluate_rules(89.0, _now_et())
        assert should_exit
        assert reason == "take_profit"


# ---------------------------------------------------------------------------
# Rule evaluation: Trailing Stop
# ---------------------------------------------------------------------------


class TestTrailingStop:

    def test_price_rises_hwm_updated(self):
        """Price rises → HWM updated, no exit."""
        pos = _make_position(
            trailing_atr_mult=1.0, entry_atr=2.0,
            stop_price=None, target_price=None,
            high_water_mark=100.0,
        )
        should_exit, reason = pos.evaluate_rules(105.0, _now_et())
        assert not should_exit
        assert pos.high_water_mark == 105.0

    def test_drop_exceeds_trailing_distance(self):
        """Price drops from HWM by more than ATR * mult → exit."""
        pos = _make_position(
            trailing_atr_mult=1.0, entry_atr=2.0,
            stop_price=None, target_price=None,
            high_water_mark=105.0,
        )
        # 105 - 102.5 = 2.5 > 2.0 → exit
        should_exit, reason = pos.evaluate_rules(102.5, _now_et())
        assert should_exit
        assert reason == "trailing_stop"

    def test_drop_within_trailing_distance(self):
        """Price drops from HWM by less than ATR * mult → no exit."""
        pos = _make_position(
            trailing_atr_mult=1.0, entry_atr=2.0,
            stop_price=None, target_price=None,
            high_water_mark=105.0,
        )
        # 105 - 104 = 1.0 < 2.0 → no exit
        should_exit, reason = pos.evaluate_rules(104.0, _now_et())
        assert not should_exit
        assert pos.high_water_mark == 105.0

    def test_intraday_no_trailing(self):
        """INTRADAY config has trailing_stop=False → no trailing behavior."""
        # INTRADAY config has trailing_stop=False, so trailing_atr_mult should be 0
        pos = _make_position(
            holding_type=HoldingType.INTRADAY,
            trailing_atr_mult=0.0,
            entry_atr=2.0,
            stop_price=None, target_price=None,
            high_water_mark=100.0,
        )
        should_exit, reason = pos.evaluate_rules(95.0, _now_et())
        assert not should_exit  # No trailing stop active


# ---------------------------------------------------------------------------
# Rule evaluation: Time Stop
# ---------------------------------------------------------------------------


class TestTimeStop:

    def test_before_deadline_no_exit(self):
        """Current time before exit_deadline → no exit."""
        deadline = _now_et() + timedelta(hours=10)
        pos = _make_position(
            exit_deadline=deadline, stop_price=None, target_price=None,
        )
        should_exit, reason = pos.evaluate_rules(100.0, _now_et())
        assert not should_exit

    def test_at_deadline_exit(self):
        """Current time at exit_deadline → exit."""
        deadline = _now_et()
        pos = _make_position(
            exit_deadline=deadline, stop_price=None, target_price=None,
        )
        should_exit, reason = pos.evaluate_rules(100.0, deadline)
        assert should_exit
        assert reason == "time_stop"

    def test_after_deadline_exit(self):
        """Current time after exit_deadline → exit."""
        deadline = _now_et() - timedelta(hours=1)
        pos = _make_position(
            exit_deadline=deadline, stop_price=None, target_price=None,
        )
        should_exit, reason = pos.evaluate_rules(100.0, _now_et())
        assert should_exit
        assert reason == "time_stop"


# ---------------------------------------------------------------------------
# Rule evaluation: Intraday Flatten
# ---------------------------------------------------------------------------


class TestIntradayFlatten:

    def test_intraday_at_1556_et(self):
        """INTRADAY position at 15:56 ET → exit."""
        t = ET.localize(datetime(2026, 4, 6, 15, 56))
        pos = _make_position(
            holding_type=HoldingType.INTRADAY,
            stop_price=None, target_price=None,
            exit_deadline=None,
        )
        should_exit, reason = pos.evaluate_rules(100.0, t)
        assert should_exit
        assert reason == "intraday_flatten"

    def test_short_swing_at_1556_no_exit(self):
        """SHORT_SWING at 15:56 ET → NO exit."""
        t = ET.localize(datetime(2026, 4, 6, 15, 56))
        pos = _make_position(
            holding_type=HoldingType.SHORT_SWING,
            stop_price=None, target_price=None,
            exit_deadline=None,
        )
        should_exit, reason = pos.evaluate_rules(100.0, t)
        assert not should_exit

    def test_intraday_at_1500_no_exit(self):
        """INTRADAY at 15:00 ET → no exit (before 15:55)."""
        t = ET.localize(datetime(2026, 4, 6, 15, 0))
        pos = _make_position(
            holding_type=HoldingType.INTRADAY,
            stop_price=None, target_price=None,
            exit_deadline=None,
        )
        should_exit, reason = pos.evaluate_rules(100.0, t)
        assert not should_exit


# ---------------------------------------------------------------------------
# Rule evaluation: Kill Switch
# ---------------------------------------------------------------------------


class TestKillSwitch:

    def test_kill_switch_exits_all(self):
        """Kill switch active → exit regardless of type."""
        for ht in HoldingType:
            pos = _make_position(
                holding_type=ht,
                stop_price=None, target_price=None,
                exit_deadline=None,
            )
            should_exit, reason = pos.evaluate_rules(
                100.0, _now_et(), kill_switch_active=True
            )
            assert should_exit
            assert reason == "kill_switch"


# ---------------------------------------------------------------------------
# Rule evaluation: Priority Order
# ---------------------------------------------------------------------------


class TestPriorityOrder:

    def test_sl_takes_priority_over_tp(self):
        """Position hits both SL and TP simultaneously → SL wins."""
        # Price at 95 — below stop (95) AND below target (90 for short)
        # Make a long position where stop=100, target=90 (weird but tests priority)
        pos = _make_position(
            side="long", entry_price=100.0,
            stop_price=96.0, target_price=95.0,
        )
        should_exit, reason = pos.evaluate_rules(95.0, _now_et())
        assert should_exit
        assert reason == "stop_loss"


# ---------------------------------------------------------------------------
# Exit Pending Flag
# ---------------------------------------------------------------------------


class TestExitPending:

    def test_exit_pending_prevents_evaluation(self):
        """exit_pending=True → skips evaluation entirely."""
        pos = _make_position(exit_pending=True, stop_price=95.0)
        should_exit, reason = pos.evaluate_rules(50.0, _now_et())
        assert not should_exit
        assert reason == ""

    def test_exit_pending_default_false(self):
        """exit_pending defaults to False."""
        pos = _make_position()
        assert pos.exit_pending is False
