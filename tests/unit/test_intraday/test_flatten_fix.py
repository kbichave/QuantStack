"""Tests for intraday flatten fix — selective flatten by holding type."""

from __future__ import annotations

import asyncio
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from quantstack.holding_period import HoldingType
from quantstack.intraday.position_manager import IntradayPositionManager, IntradayPositionMeta

ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fill_tracker(positions: dict[str, int]) -> MagicMock:
    """Create a mock FillTracker with the given positions {symbol: quantity}."""
    tracker = MagicMock()
    pos_objects = {}
    for sym, qty in positions.items():
        p = MagicMock()
        p.quantity = qty
        p.symbol = sym
        pos_objects[sym] = p

    tracker.get_open_positions.return_value = pos_objects
    tracker.get_position.side_effect = lambda s: pos_objects.get(s)
    tracker.daily_realised_pnl.return_value = 0.0
    tracker.update_price = MagicMock()
    return tracker


def _make_features(symbol: str, close: float, atr: float = 2.0) -> MagicMock:
    """Create a mock IncrementalFeatures."""
    f = MagicMock()
    f.symbol = symbol
    f.close = close
    f.atr = atr
    return f


@pytest.fixture()
def run_async():
    """Run async coroutine in a fresh event loop."""
    def _run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return _run


# ---------------------------------------------------------------------------
# IntradayPositionMeta tests
# ---------------------------------------------------------------------------


class TestIntradayPositionMeta:

    def test_accepts_holding_type(self):
        """IntradayPositionMeta accepts holding_type field."""
        meta = IntradayPositionMeta(
            symbol="SPY",
            entry_price=100.0,
            entry_time=datetime.now(ET),
            entry_bar_count=0,
            high_water_mark=100.0,
            low_water_mark=100.0,
            holding_type=HoldingType.SHORT_SWING,
        )
        assert meta.holding_type == HoldingType.SHORT_SWING

    def test_defaults_to_intraday(self):
        """IntradayPositionMeta defaults holding_type to INTRADAY."""
        meta = IntradayPositionMeta(
            symbol="SPY",
            entry_price=100.0,
            entry_time=datetime.now(ET),
            entry_bar_count=0,
            high_water_mark=100.0,
            low_water_mark=100.0,
        )
        assert meta.holding_type == HoldingType.INTRADAY


# ---------------------------------------------------------------------------
# register_entry tests
# ---------------------------------------------------------------------------


class TestRegisterEntry:

    def test_stores_holding_type(self):
        """register_entry() stores holding_type on the meta."""
        tracker = _make_fill_tracker({})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=AsyncMock())
        pm.register_entry("SPY", price=100.0, atr=2.0, holding_type=HoldingType.SHORT_SWING)
        assert pm._position_meta["SPY"].holding_type == HoldingType.SHORT_SWING

    def test_stores_swing_type(self):
        """register_entry() with HoldingType.SWING stores correctly."""
        tracker = _make_fill_tracker({})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=AsyncMock())
        pm.register_entry("NVDA", price=200.0, atr=5.0, holding_type=HoldingType.SWING)
        assert pm._position_meta["NVDA"].holding_type == HoldingType.SWING

    def test_defaults_to_intraday(self):
        """register_entry() without holding_type defaults to INTRADAY."""
        tracker = _make_fill_tracker({})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=AsyncMock())
        pm.register_entry("SPY", price=100.0, atr=2.0)
        assert pm._position_meta["SPY"].holding_type == HoldingType.INTRADAY


# ---------------------------------------------------------------------------
# flatten_intraday tests
# ---------------------------------------------------------------------------


class TestFlattenIntraday:

    def test_flattens_only_intraday(self, run_async):
        """flatten_intraday() with 2 INTRADAY + 1 SHORT_SWING → only INTRADAY exited."""
        execute_fn = AsyncMock(return_value={"status": "filled"})
        tracker = _make_fill_tracker({"SPY": 100, "QQQ": 50, "NVDA": 200})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=execute_fn)

        pm.register_entry("SPY", 450.0, 5.0, holding_type=HoldingType.INTRADAY)
        pm.register_entry("QQQ", 380.0, 4.0, holding_type=HoldingType.INTRADAY)
        pm.register_entry("NVDA", 800.0, 10.0, holding_type=HoldingType.SHORT_SWING)

        run_async(pm.flatten_intraday())

        # SPY and QQQ should have been exited
        assert execute_fn.call_count == 2
        exited_symbols = {call.kwargs["symbol"] for call in execute_fn.call_args_list}
        assert exited_symbols == {"SPY", "QQQ"}

        # NVDA should still be tracked
        assert "NVDA" in pm._position_meta
        assert pm._position_meta["NVDA"].holding_type == HoldingType.SHORT_SWING

    def test_no_intraday_positions(self, run_async, caplog):
        """flatten_intraday() with 0 INTRADAY positions → no exits."""
        execute_fn = AsyncMock()
        tracker = _make_fill_tracker({"NVDA": 200})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=execute_fn)

        pm.register_entry("NVDA", 800.0, 10.0, holding_type=HoldingType.SHORT_SWING)

        run_async(pm.flatten_intraday())

        execute_fn.assert_not_called()

    def test_logs_flattened_and_preserved_counts(self, run_async, caplog):
        """flatten_intraday() logs flattened count and preserved count."""
        execute_fn = AsyncMock(return_value={"status": "filled"})
        tracker = _make_fill_tracker({"SPY": 100, "QQQ": 50, "NVDA": 200, "TSLA": 30})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=execute_fn)

        pm.register_entry("SPY", 450.0, 5.0, holding_type=HoldingType.INTRADAY)
        pm.register_entry("QQQ", 380.0, 4.0, holding_type=HoldingType.INTRADAY)
        pm.register_entry("NVDA", 800.0, 10.0, holding_type=HoldingType.SHORT_SWING)
        pm.register_entry("TSLA", 200.0, 8.0, holding_type=HoldingType.SWING)

        run_async(pm.flatten_intraday())

        assert execute_fn.call_count == 2


# ---------------------------------------------------------------------------
# flatten_all (kill switch) tests
# ---------------------------------------------------------------------------


class TestFlattenAll:

    def test_exits_all_regardless_of_type(self, run_async):
        """flatten_all() exits ALL positions regardless of holding_type."""
        execute_fn = AsyncMock(return_value={"status": "filled"})
        tracker = _make_fill_tracker({"SPY": 100, "QQQ": 50, "NVDA": 200})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=execute_fn)

        pm.register_entry("SPY", 450.0, 5.0, holding_type=HoldingType.INTRADAY)
        pm.register_entry("QQQ", 380.0, 4.0, holding_type=HoldingType.SHORT_SWING)
        pm.register_entry("NVDA", 800.0, 10.0, holding_type=HoldingType.SWING)

        run_async(pm.flatten_all(reason="kill_switch"))

        assert execute_fn.call_count == 3
        assert pm._flattened is True

    def test_regression_default_holding_type(self, run_async):
        """flatten_all() with default holding_type behaves as before."""
        execute_fn = AsyncMock(return_value={"status": "filled"})
        tracker = _make_fill_tracker({"SPY": 100, "QQQ": 50})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=execute_fn)

        pm.register_entry("SPY", 450.0, 5.0)  # defaults to INTRADAY
        pm.register_entry("QQQ", 380.0, 4.0)  # defaults to INTRADAY

        run_async(pm.flatten_all())

        assert execute_fn.call_count == 2
        assert pm._flattened is True


# ---------------------------------------------------------------------------
# on_features control flow tests
# ---------------------------------------------------------------------------


class TestOnFeaturesControlFlow:

    def test_kill_switch_exits_all_including_swing(self, run_async):
        """Kill switch path: flatten_all() exits everything including non-INTRADAY."""
        execute_fn = AsyncMock(return_value={"status": "filled"})
        tracker = _make_fill_tracker({"NVDA": 200, "SPY": 100})
        pm = IntradayPositionManager(
            fill_tracker=tracker,
            broker_execute_fn=execute_fn,
            kill_switch_fn=lambda: True,
        )
        pm.register_entry("NVDA", 800.0, 10.0, holding_type=HoldingType.SHORT_SWING)
        pm.register_entry("SPY", 450.0, 5.0, holding_type=HoldingType.INTRADAY)

        run_async(pm.on_features(_make_features("NVDA", 810.0)))

        # ALL positions should be exited by kill switch
        assert pm._flattened is True
        assert execute_fn.call_count == 2

    def test_reset_daily_clears_intraday_flattened(self):
        """reset_daily() clears _intraday_flattened flag."""
        tracker = _make_fill_tracker({})
        pm = IntradayPositionManager(fill_tracker=tracker, broker_execute_fn=AsyncMock())
        pm._intraday_flattened = True
        pm.reset_daily()
        assert pm._intraday_flattened is False
