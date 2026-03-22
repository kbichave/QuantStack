# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for IntradayPositionManager."""

from __future__ import annotations

import asyncio
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantstack.config.timeframes import Timeframe
from quantstack.data.streaming.incremental_features import IncrementalFeatures
from quantstack.core.execution.fill_tracker import FillEvent, FillTracker, LivePosition

from quantstack.intraday.position_manager import IntradayPositionManager


def _make_features(
    symbol: str = "SPY",
    close: float = 450.0,
    atr: float = 2.0,
) -> IncrementalFeatures:
    return IncrementalFeatures(
        symbol=symbol,
        timestamp=datetime(2026, 3, 17, 10, 30),
        timeframe=Timeframe.M1,
        close=close,
        ema_fast=450.0,
        ema_slow=449.0,
        ema_cross=1.0,
        rsi=50.0,
        roc=0.001,
        atr=atr,
        atr_pct=atr / close,
        bb_upper=455.0,
        bb_lower=445.0,
        bb_pct_b=0.5,
        volume_ratio=1.0,
        price_to_ema=0.0,
        vwap_deviation=0.0,
        is_warm=True,
    )


def _make_pm(
    flatten_time: str = "15:55",
    trailing_atr: float = 2.0,
    max_hold_bars: int = 0,
    loss_stop_pct: float = 0.02,
) -> tuple[IntradayPositionManager, FillTracker, AsyncMock]:
    tracker = FillTracker()
    execute_fn = AsyncMock(return_value={"fill": True})
    pm = IntradayPositionManager(
        fill_tracker=tracker,
        broker_execute_fn=execute_fn,
        flatten_time_et=flatten_time,
        trailing_stop_atr_mult=trailing_atr,
        max_hold_bars=max_hold_bars,
        intraday_loss_stop_pct=loss_stop_pct,
    )
    return pm, tracker, execute_fn


class TestMarkToMarket:
    @pytest.mark.asyncio
    async def test_updates_price_on_bar(self):
        pm, tracker, _ = _make_pm(flatten_time="23:59")
        # Add a position
        tracker.update_fill(
            FillEvent(
                order_id="1",
                symbol="SPY",
                side="buy",
                filled_qty=100,
                avg_fill_price=450.0,
            )
        )
        assert tracker.get_position("SPY").current_price == 450.0

        await pm.on_features(_make_features(close=455.0))
        assert tracker.get_position("SPY").current_price == 455.0


class TestTrailingStop:
    @pytest.mark.asyncio
    async def test_trailing_stop_triggers_on_drop(self):
        pm, tracker, execute_fn = _make_pm(trailing_atr=2.0, flatten_time="23:59")

        # Open long position
        tracker.update_fill(
            FillEvent(
                order_id="1",
                symbol="SPY",
                side="buy",
                filled_qty=100,
                avg_fill_price=450.0,
            )
        )
        pm.register_entry("SPY", price=450.0, atr=2.0)

        # Price goes up (updates high water mark)
        await pm.on_features(_make_features(close=456.0))
        assert pm._position_meta["SPY"].high_water_mark == 456.0
        execute_fn.assert_not_called()

        # Price drops > 2 * ATR (4.0) from HWM 456 → trigger at < 452
        await pm.on_features(_make_features(close=451.0))
        execute_fn.assert_called_once()
        assert "trailing_stop" in execute_fn.call_args.kwargs.get("reason", "")


class TestTimeStop:
    @pytest.mark.asyncio
    async def test_time_stop_after_n_bars(self):
        pm, tracker, execute_fn = _make_pm(
            max_hold_bars=5, trailing_atr=0, flatten_time="23:59"
        )

        tracker.update_fill(
            FillEvent(
                order_id="1",
                symbol="SPY",
                side="buy",
                filled_qty=100,
                avg_fill_price=450.0,
            )
        )
        pm.register_entry("SPY", price=450.0, atr=2.0)

        # Process 4 bars — no exit
        for _ in range(4):
            await pm.on_features(_make_features(close=450.0))
        execute_fn.assert_not_called()

        # 5th bar — time stop
        await pm.on_features(_make_features(close=450.0))
        execute_fn.assert_called_once()
        assert "time_stop" in execute_fn.call_args.kwargs.get("reason", "")


class TestLossStop:
    @pytest.mark.asyncio
    async def test_loss_stop_on_2pct_drop(self):
        pm, tracker, execute_fn = _make_pm(
            loss_stop_pct=0.02, trailing_atr=0, flatten_time="23:59"
        )

        tracker.update_fill(
            FillEvent(
                order_id="1",
                symbol="SPY",
                side="buy",
                filled_qty=100,
                avg_fill_price=450.0,
            )
        )
        pm.register_entry("SPY", price=450.0, atr=2.0)

        # 2.1% drop from entry: 450 * 0.021 = 9.45 → price = 440.55
        await pm.on_features(_make_features(close=440.0))
        execute_fn.assert_called_once()
        assert "loss_stop" in execute_fn.call_args.kwargs.get("reason", "")


class TestFlattenAll:
    @pytest.mark.asyncio
    async def test_flatten_closes_all_positions(self):
        pm, tracker, execute_fn = _make_pm()

        tracker.update_fill(
            FillEvent(
                order_id="1",
                symbol="SPY",
                side="buy",
                filled_qty=100,
                avg_fill_price=450.0,
            )
        )
        tracker.update_fill(
            FillEvent(
                order_id="2",
                symbol="QQQ",
                side="buy",
                filled_qty=50,
                avg_fill_price=380.0,
            )
        )

        pm.register_entry("SPY", price=450.0, atr=2.0)
        pm.register_entry("QQQ", price=380.0, atr=1.5)

        exits = await pm.flatten_all(reason="flatten_at_close")

        assert pm.is_flattened
        assert execute_fn.call_count == 2


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_kill_switch_triggers_flatten(self):
        tracker = FillTracker()
        execute_fn = AsyncMock(return_value={"fill": True})
        pm = IntradayPositionManager(
            fill_tracker=tracker,
            broker_execute_fn=execute_fn,
            kill_switch_fn=lambda: True,  # Always active
        )

        tracker.update_fill(
            FillEvent(
                order_id="1",
                symbol="SPY",
                side="buy",
                filled_qty=100,
                avg_fill_price=450.0,
            )
        )
        pm.register_entry("SPY", price=450.0, atr=2.0)

        await pm.on_features(_make_features(close=450.0))
        assert pm.is_flattened
        execute_fn.assert_called_once()


class TestTradeCount:
    def test_trades_today_increments(self):
        pm, _, _ = _make_pm()
        assert pm.trades_today == 0
        pm.register_entry("SPY", price=450.0, atr=2.0)
        assert pm.trades_today == 1
        pm.register_entry("QQQ", price=380.0, atr=1.5)
        assert pm.trades_today == 2

    def test_reset_daily(self):
        pm, _, _ = _make_pm()
        pm.register_entry("SPY", price=450.0, atr=2.0)
        pm.reset_daily()
        assert pm.trades_today == 0
        assert not pm.is_flattened
