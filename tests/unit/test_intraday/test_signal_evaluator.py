# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for IntradaySignalEvaluator."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from quantstack.config.timeframes import Timeframe
from quantstack.data.streaming.incremental_features import IncrementalFeatures
from quantstack.core.execution.fill_tracker import FillTracker

from quantstack.intraday.position_manager import IntradayPositionManager
from quantstack.intraday.signal_evaluator import (
    IntradaySignalEvaluator,
    _evaluate_scalar_rule,
)


def _make_features(
    symbol: str = "SPY",
    rsi: float = 50.0,
    ema_cross: float = 0.0,
    close: float = 450.0,
    atr: float = 2.0,
    is_warm: bool = True,
    **kwargs,
) -> IncrementalFeatures:
    """Build an IncrementalFeatures with sensible defaults."""
    return IncrementalFeatures(
        symbol=symbol,
        timestamp=datetime(2026, 3, 17, 10, 30),
        timeframe=Timeframe.M1,
        close=close,
        ema_fast=450.0,
        ema_slow=449.0,
        ema_cross=ema_cross,
        rsi=rsi,
        roc=0.001,
        atr=atr,
        atr_pct=atr / close,
        bb_upper=455.0,
        bb_lower=445.0,
        bb_pct_b=0.5,
        volume_ratio=1.0,
        price_to_ema=0.0,
        vwap_deviation=0.0,
        is_warm=is_warm,
    )


def _make_evaluator(
    strategies=None,
    entry_cutoff="15:30",
    max_trades=50,
) -> tuple[IntradaySignalEvaluator, IntradayPositionManager]:
    tracker = FillTracker()
    pm = IntradayPositionManager(
        fill_tracker=tracker,
        broker_execute_fn=MagicMock(),
    )
    strategies = strategies or [
        {
            "name": "test_rsi",
            "entry_rules": [{"indicator": "rsi", "condition": "below", "value": 30}],
            "exit_rules": [{"indicator": "rsi", "condition": "above", "value": 70}],
            "parameters": {"direction": "buy"},
            "risk_params": {"quantity": 100},
        }
    ]
    evaluator = IntradaySignalEvaluator(
        strategies=strategies,
        position_manager=pm,
        entry_cutoff_et=entry_cutoff,
        max_trades_per_day=max_trades,
    )
    return evaluator, pm


class TestScalarRuleEvaluation:
    def test_above(self):
        assert _evaluate_scalar_rule(
            {"indicator": "rsi", "condition": "above", "value": 70},
            {"rsi": 75},
            {},
        )

    def test_below(self):
        assert _evaluate_scalar_rule(
            {"indicator": "rsi", "condition": "below", "value": 30},
            {"rsi": 25},
            {},
        )

    def test_crosses_above(self):
        assert _evaluate_scalar_rule(
            {"indicator": "ema_cross", "condition": "crosses_above", "value": 0},
            {"ema_cross": 0.5},
            {"ema_cross": -0.3},
        )

    def test_crosses_above_no_prev(self):
        assert not _evaluate_scalar_rule(
            {"indicator": "ema_cross", "condition": "crosses_above", "value": 0},
            {"ema_cross": 0.5},
            {},
        )

    def test_between(self):
        assert _evaluate_scalar_rule(
            {"indicator": "rsi", "condition": "between", "value": [40, 60]},
            {"rsi": 50},
            {},
        )

    def test_missing_indicator(self):
        assert not _evaluate_scalar_rule(
            {"indicator": "nonexistent", "condition": "above", "value": 0},
            {"rsi": 50},
            {},
        )


class TestSignalEvaluator:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_warm(self):
        evaluator, _ = _make_evaluator()
        result = await evaluator(_make_features(is_warm=False))
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_buy_on_entry_signal(self):
        evaluator, _ = _make_evaluator(entry_cutoff="23:59")
        result = await evaluator(_make_features(rsi=25))
        assert result is not None
        assert result.side == "buy"
        assert result.quantity == 100

    @pytest.mark.asyncio
    async def test_returns_none_when_no_signal(self):
        evaluator, _ = _make_evaluator(entry_cutoff="23:59")
        result = await evaluator(_make_features(rsi=50))
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_max_trades(self):
        evaluator, pm = _make_evaluator(max_trades=0)
        result = await evaluator(_make_features(rsi=25))
        assert result is None

    @pytest.mark.asyncio
    @patch("quantstack.intraday.signal_evaluator.datetime")
    async def test_respects_entry_cutoff(self, mock_dt):
        mock_now = MagicMock()
        mock_now.time.return_value = datetime(2026, 3, 17, 15, 45).time()
        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        evaluator, _ = _make_evaluator(entry_cutoff="15:30")
        result = await evaluator(_make_features(rsi=25))
        assert result is None

    @pytest.mark.asyncio
    async def test_no_entry_when_already_in_position(self):
        evaluator, pm = _make_evaluator(entry_cutoff="23:59")
        # Simulate existing position
        pm._tracker._positions["SPY"] = MagicMock(quantity=100)
        result = await evaluator(_make_features(rsi=25))
        # Should not enter again (might exit if exit rules hit)
        # RSI=25 doesn't trigger exit rule (RSI > 70), so None
        assert result is None
