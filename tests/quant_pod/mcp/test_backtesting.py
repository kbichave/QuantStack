# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the backtest and walk-forward MCP tools.

Uses synthetic price data to avoid needing real market data.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from quantstack.context import create_trading_context
from quantstack.mcp.server import get_strategy, register_strategy, run_backtest, run_walkforward
from quantstack.strategies.signal_generator import generate_signals_from_rules as _generate_signals_from_rules
import quantstack.mcp._state as _mcp_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:", session_id=str(uuid.uuid4()))
    yield context
    context.db.close()


@pytest.fixture
def _inject_ctx(ctx):
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    yield ctx
    _mcp_state._ctx = original


def _fn(tool_obj):
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


def _make_trending_ohlcv(n_bars: int = 600, start_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV with a trend + mean reversion pattern."""
    np.random.seed(42)
    # Create a trending series with pullbacks (good for momentum/RSI strategies)
    t = np.arange(n_bars)
    trend = start_price + t * 0.05  # gradual uptrend
    noise = np.random.randn(n_bars) * 0.8
    oscillation = 3 * np.sin(t * 2 * np.pi / 40)  # 40-bar cycle
    close = trend + noise + oscillation

    high = close + np.abs(np.random.randn(n_bars)) * 0.5
    low = close - np.abs(np.random.randn(n_bars)) * 0.5
    open_ = close + np.random.randn(n_bars) * 0.2
    volume = np.random.randint(100_000, 1_000_000, n_bars)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2020-01-01", periods=n_bars, freq="1D"),
    )


@pytest.fixture
def synthetic_data():
    return _make_trending_ohlcv()


@pytest.fixture
async def registered_strategy(_inject_ctx):
    """Register a simple RSI mean-reversion strategy and return its ID."""
    result = await _fn(register_strategy)(
        name=f"test_rsi_mr_{uuid.uuid4().hex[:6]}",
        description="RSI mean reversion for testing",
        parameters={"rsi_period": 14, "sma_fast": 10, "sma_slow": 50},
        entry_rules=[
            {
                "indicator": "rsi",
                "condition": "crosses_below",
                "value": 35,
                "direction": "long",
            },
        ],
        exit_rules=[
            {"indicator": "rsi", "condition": "crosses_above", "value": 65},
        ],
        risk_params={"stop_loss_atr": 2.0},
        regime_affinity={"ranging": 0.8, "trending_up": 0.5},
    )
    assert result["success"]
    return result["strategy_id"]


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------


class TestSignalGeneration:
    def test_generates_signals_from_rsi_rules(self, synthetic_data):
        entry_rules = [
            {
                "indicator": "rsi",
                "condition": "crosses_below",
                "value": 35,
                "direction": "long",
            },
        ]
        exit_rules = [
            {"indicator": "rsi", "condition": "crosses_above", "value": 65},
        ]
        parameters = {"rsi_period": 14, "sma_fast": 10, "sma_slow": 50}

        signals = _generate_signals_from_rules(
            synthetic_data, entry_rules, exit_rules, parameters
        )

        assert "signal" in signals.columns
        assert "signal_direction" in signals.columns
        assert len(signals) == len(synthetic_data)
        # Should have at least some signals in 600 bars of oscillating data
        assert signals["signal"].sum() > 0

    def test_sma_crossover_generates_signals(self, synthetic_data):
        entry_rules = [
            {
                "indicator": "sma_crossover",
                "condition": "crosses_above",
                "direction": "long",
            },
        ]
        exit_rules = [
            {"indicator": "sma_crossover", "condition": "crosses_below"},
        ]
        parameters = {"sma_fast": 10, "sma_slow": 50}

        signals = _generate_signals_from_rules(
            synthetic_data, entry_rules, exit_rules, parameters
        )
        assert signals["signal"].sum() > 0

    def test_breakout_generates_signals(self, synthetic_data):
        entry_rules = [
            {"indicator": "breakout", "condition": "above", "direction": "long"},
        ]
        exit_rules = [
            {"indicator": "rsi", "condition": "crosses_above", "value": 75},
        ]
        parameters = {
            "breakout_period": 20,
            "rsi_period": 14,
            "sma_fast": 10,
            "sma_slow": 50,
        }

        signals = _generate_signals_from_rules(
            synthetic_data, entry_rules, exit_rules, parameters
        )
        long_entries = (signals["signal_direction"] == "LONG").sum()
        assert long_entries > 0


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    @pytest.mark.asyncio
    async def test_backtest_returns_metrics(
        self, _inject_ctx, registered_strategy, synthetic_data
    ):
        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data",
            return_value=synthetic_data,
        ):
            result = await _fn(run_backtest)(
                strategy_id=registered_strategy,
                symbol="TEST",
            )

        assert result["success"] is True
        assert result["strategy_id"] == registered_strategy
        assert result["total_trades"] > 0
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "calmar_ratio" in result
        assert result["bars_tested"] == len(synthetic_data)

    @pytest.mark.asyncio
    async def test_backtest_updates_strategy_record(
        self, _inject_ctx, registered_strategy, synthetic_data
    ):
        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data",
            return_value=synthetic_data,
        ):
            await _fn(run_backtest)(strategy_id=registered_strategy, symbol="TEST")

        strat = await _fn(get_strategy)(strategy_id=registered_strategy)
        assert strat["strategy"]["backtest_summary"] is not None
        assert strat["strategy"]["status"] in ("backtested", "draft")

    @pytest.mark.asyncio
    async def test_backtest_no_data_fails(self, _inject_ctx, registered_strategy):
        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data", return_value=None
        ):
            result = await _fn(run_backtest)(
                strategy_id=registered_strategy, symbol="NODATA"
            )
        assert result["success"] is False
        assert "No price data" in result["error"]

    @pytest.mark.asyncio
    async def test_backtest_bad_strategy_id(self, _inject_ctx):
        result = await _fn(run_backtest)(strategy_id="nonexistent", symbol="TEST")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_backtest_no_entry_rules(self, _inject_ctx, synthetic_data):
        r = await _fn(register_strategy)(
            name="empty_rules",
            parameters={},
            entry_rules=[],
            exit_rules=[],
        )

        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data",
            return_value=synthetic_data,
        ):
            result = await _fn(run_backtest)(
                strategy_id=r["strategy_id"], symbol="TEST"
            )
        assert result["success"] is False
        assert "no entry_rules" in result["error"]


# ---------------------------------------------------------------------------
# run_walkforward
# ---------------------------------------------------------------------------


class TestRunWalkforward:
    @pytest.mark.asyncio
    async def test_walkforward_returns_fold_results(
        self, _inject_ctx, registered_strategy
    ):
        # Need enough data: min_train_size(504) + n_splits(3) * test_size(60) = 684
        big_data = _make_trending_ohlcv(n_bars=800)

        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data", return_value=big_data
        ):
            result = await _fn(run_walkforward)(
                strategy_id=registered_strategy,
                symbol="TEST",
                n_splits=3,
                test_size=60,
                min_train_size=504,
            )

        assert result["success"] is True
        assert result["n_folds"] == 3
        assert len(result["fold_results"]) == 3
        assert "is_sharpe_mean" in result
        assert "oos_sharpe_mean" in result
        assert "overfit_ratio" in result
        assert "oos_positive_folds" in result

    @pytest.mark.asyncio
    async def test_walkforward_insufficient_data(
        self, _inject_ctx, registered_strategy
    ):
        small_data = _make_trending_ohlcv(n_bars=100)

        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data", return_value=small_data
        ):
            result = await _fn(run_walkforward)(
                strategy_id=registered_strategy,
                symbol="TEST",
                n_splits=5,
                test_size=252,
                min_train_size=504,
            )
        assert result["success"] is False
        assert "Insufficient data" in result["error"]

    @pytest.mark.asyncio
    async def test_walkforward_updates_strategy_record(
        self, _inject_ctx, registered_strategy
    ):
        big_data = _make_trending_ohlcv(n_bars=800)

        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data", return_value=big_data
        ):
            await _fn(run_walkforward)(
                strategy_id=registered_strategy,
                symbol="TEST",
                n_splits=3,
                test_size=60,
                min_train_size=504,
            )

        strat = await _fn(get_strategy)(strategy_id=registered_strategy)
        assert strat["strategy"]["walkforward_summary"] is not None

    @pytest.mark.asyncio
    async def test_walkforward_fold_metrics_have_expected_keys(
        self, _inject_ctx, registered_strategy
    ):
        big_data = _make_trending_ohlcv(n_bars=800)

        with patch(
            "quantstack.mcp.tools.backtesting._fetch_price_data", return_value=big_data
        ):
            result = await _fn(run_walkforward)(
                strategy_id=registered_strategy,
                symbol="TEST",
                n_splits=3,
                test_size=60,
                min_train_size=504,
            )

        fold = result["fold_results"][0]
        expected_keys = {
            "fold",
            "train_bars",
            "test_bars",
            "is_sharpe",
            "oos_sharpe",
            "is_return_pct",
            "oos_return_pct",
            "is_trades",
            "oos_trades",
            "oos_max_dd",
        }
        assert expected_keys.issubset(fold.keys())
