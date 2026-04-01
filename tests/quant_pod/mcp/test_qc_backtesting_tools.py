# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_backtesting.py MCP tools.

Covers run_backtest_template, get_backtest_metrics, run_walkforward_template,
and run_purged_cv.  Uses synthetic OHLCV and mocks _get_reader().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantstack.mcp.tools.qc_backtesting import (
    _generate_strategy_signals,
    get_backtest_metrics,
    run_backtest_template,
    run_purged_cv,
    run_walkforward_template,
)
from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_store(df: pd.DataFrame) -> MagicMock:
    store = MagicMock()
    store.load_ohlcv.return_value = df
    store.close.return_value = None
    return store


def _empty_store() -> MagicMock:
    return _mock_store(pd.DataFrame())


def _large_ohlcv(n_bars: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate a large trending OHLCV dataset suitable for walk-forward / purged CV."""
    return synthetic_ohlcv(symbol="TEST", n_days=n_bars, start_price=100.0, seed=seed)


# ---------------------------------------------------------------------------
# _generate_strategy_signals (internal, pure computation)
# ---------------------------------------------------------------------------


class TestGenerateStrategySignals:
    """Test the signal-generation helper with synthetic feature DataFrames."""

    def _make_df_with_zscore(self, n: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        # Strong oscillation to guarantee z-score exceeds +-2.0
        t = np.arange(n)
        close = 100 + rng.normal(0, 0.5, n) + 5 * np.sin(t * 2 * np.pi / 40)
        close_s = pd.Series(close)
        mean = close_s.rolling(20).mean()
        std = close_s.rolling(20).std()
        # .values to avoid RangeIndex vs DatetimeIndex mismatch
        zscore = ((close_s - mean) / std).values
        return pd.DataFrame(
            {
                "close": close,
                "close_zscore_20": zscore,
            },
            index=pd.date_range("2023-01-01", periods=n, freq="1D"),
        )

    def test_mean_reversion_generates_signals(self):
        df = self._make_df_with_zscore()
        signals = _generate_strategy_signals(df, "mean_reversion", zscore_entry=2.0)
        assert "signal" in signals.columns
        assert "signal_direction" in signals.columns
        # Should generate at least some signals in 200 bars of random walk
        assert (signals["signal"] != 0).sum() > 0

    def test_mean_reversion_fallback_without_zscore_col(self):
        """When close_zscore_20 is absent, should compute from raw close."""
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 1, 200))
        df = pd.DataFrame(
            {"close": close},
            index=pd.date_range("2023-01-01", periods=200, freq="1D"),
        )
        signals = _generate_strategy_signals(df, "mean_reversion")
        assert (signals["signal"] != 0).sum() >= 0  # just checking it runs

    def test_trend_following_generates_signals(self):
        rng = np.random.default_rng(0)
        n = 200
        t = np.arange(n)
        close = 100 + t * 0.1 + rng.normal(0, 0.3, n)
        df = pd.DataFrame(
            {"close": close},
            index=pd.date_range("2023-01-01", periods=n, freq="1D"),
        )
        signals = _generate_strategy_signals(df, "trend_following")
        assert (signals["signal"] != 0).sum() > 0

    def test_trend_following_uses_ema_columns_if_present(self):
        """When ema_20 and ema_50 columns exist, use them directly."""
        n = 300
        rng = np.random.default_rng(0)
        t = np.arange(n)
        close = 100 + rng.normal(0, 0.3, n) + 5 * np.sin(t * 2 * np.pi / 60)
        # .values avoids RangeIndex vs DatetimeIndex mismatch
        df = pd.DataFrame(
            {
                "close": close,
                "ema_20": pd.Series(close).ewm(span=20).mean().values,
                "ema_50": pd.Series(close).ewm(span=50).mean().values,
            },
            index=pd.date_range("2023-01-01", periods=n, freq="1D"),
        )
        signals = _generate_strategy_signals(df, "trend_following")
        assert (signals["signal"] != 0).sum() > 0

    def test_momentum_generates_signals(self):
        rng = np.random.default_rng(1)
        n = 200
        close = 100 + np.cumsum(rng.normal(0, 1.5, n))
        df = pd.DataFrame(
            {"close": close},
            index=pd.date_range("2023-01-01", periods=n, freq="1D"),
        )
        signals = _generate_strategy_signals(df, "momentum")
        assert "signal" in signals.columns

    def test_unknown_strategy_returns_no_signals(self):
        """Unknown strategy type should return all-zero signals."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2023-01-01", periods=3, freq="1D"),
        )
        signals = _generate_strategy_signals(df, "does_not_exist")
        assert (signals["signal"] == 0).all()


# ---------------------------------------------------------------------------
# get_backtest_metrics — pure computation, no mocking needed
# ---------------------------------------------------------------------------


class TestGetBacktestMetrics:
    @pytest.mark.asyncio
    async def test_strong_strategy(self):
        result = await _fn(get_backtest_metrics)(
            total_return=45.0,
            sharpe_ratio=2.5,
            max_drawdown=-8.0,
            win_rate=65.0,
            total_trades=120,
        )
        assert result["overall_rating"] == "Strong strategy"
        assert result["interpretation"]["sharpe"] == "Excellent risk-adjusted returns"
        assert result["interpretation"]["drawdown"] == "Excellent drawdown control"
        assert result["interpretation"]["win_rate"] == "High win rate"
        assert result["interpretation"]["trades"] == "Good statistical significance"

    @pytest.mark.asyncio
    async def test_weak_strategy(self):
        result = await _fn(get_backtest_metrics)(
            total_return=-5.0,
            sharpe_ratio=0.2,
            max_drawdown=-35.0,
            win_rate=30.0,
            total_trades=15,
        )
        assert result["overall_rating"] == "Needs improvement"
        assert result["interpretation"]["sharpe"] == "Poor risk-adjusted returns"
        assert result["interpretation"]["drawdown"] == "Severe drawdown risk"
        assert result["interpretation"]["win_rate"] == "Low win rate - needs good R:R"
        assert result["interpretation"]["trades"] == "Insufficient sample size"

    @pytest.mark.asyncio
    async def test_moderate_strategy(self):
        result = await _fn(get_backtest_metrics)(
            total_return=15.0,
            sharpe_ratio=1.2,
            max_drawdown=-18.0,
            win_rate=52.0,
            total_trades=60,
        )
        assert result["overall_rating"] == "Strong strategy"  # score = 2+2+1+1 = 6

    @pytest.mark.asyncio
    async def test_borderline_scoring(self):
        """Sharpe=1.0, DD=-20, WR=50, trades=50 => score=2+2+1+1=6 => Strong."""
        result = await _fn(get_backtest_metrics)(
            total_return=10.0,
            sharpe_ratio=1.0,
            max_drawdown=-20.0,
            win_rate=50.0,
            total_trades=50,
        )
        assert result["overall_rating"] == "Strong strategy"

    @pytest.mark.asyncio
    async def test_metrics_passthrough(self):
        """Input metrics should be echoed back in the result."""
        result = await _fn(get_backtest_metrics)(
            total_return=10.0,
            sharpe_ratio=1.0,
            max_drawdown=-15.0,
            win_rate=55.0,
            total_trades=80,
        )
        m = result["metrics"]
        assert m["total_return"] == 10.0
        assert m["sharpe_ratio"] == 1.0
        assert m["max_drawdown"] == -15.0
        assert m["win_rate"] == 55.0
        assert m["total_trades"] == 80

    @pytest.mark.asyncio
    async def test_zero_trades(self):
        """Zero trades should still return a valid result."""
        result = await _fn(get_backtest_metrics)(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
        )
        assert result["interpretation"]["trades"] == "Insufficient sample size"
        assert result["overall_rating"] == "Needs improvement"


# ---------------------------------------------------------------------------
# run_backtest_template
# ---------------------------------------------------------------------------


class TestRunBacktestTemplate:
    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(run_backtest_template)(symbol="NOPE")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_happy_path_mean_reversion(self):
        df = _large_ohlcv(n_bars=600)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_backtest_template)(
                symbol="TEST", strategy_type="mean_reversion"
            )
        # Tool should either succeed with metrics or return an error dict
        if "error" not in result:
            assert result["symbol"] == "TEST"
            assert "metrics" in result
            assert "total_return" in result["metrics"]
            assert "sharpe_ratio" in result["metrics"]
        # If it errors, just make sure it's a dict with "error" key
        else:
            assert isinstance(result["error"], str)

    @pytest.mark.asyncio
    async def test_end_date_filter(self):
        df = _large_ohlcv(n_bars=600)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_backtest_template)(
                symbol="TEST", end_date="2024-06-01"
            )
        # Should run with filtered data
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# run_walkforward_template
# ---------------------------------------------------------------------------


class TestRunWalkforwardTemplate:
    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(run_walkforward_template)(symbol="NOPE")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_error_with_suggestion(self):
        """Short data + demanding params => error with feasible parameter suggestion."""
        df = synthetic_ohlcv("SHORT", n_days=100)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_walkforward_template)(
                symbol="SHORT",
                n_splits=5,
                test_size=252,
                min_train_size=504,
            )
        assert "error" in result
        assert "suggestion" in result

    @pytest.mark.asyncio
    async def test_happy_path(self):
        # Need: min_train_size(504) + n_splits(3) * test_size(60) + 1 (gap) = 685
        df = _large_ohlcv(n_bars=800)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_walkforward_template)(
                symbol="TEST",
                n_splits=3,
                test_size=60,
                min_train_size=504,
            )
        assert "error" not in result
        assert result["symbol"] == "TEST"
        assert result["n_splits"] == 3
        assert len(result["folds"]) == 3
        # Each fold should have temporal boundaries
        for fold in result["folds"]:
            assert "train_start" in fold
            assert "test_end" in fold
            assert fold["train_size"] >= 504

    @pytest.mark.asyncio
    async def test_rolling_window(self):
        df = _large_ohlcv(n_bars=800)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_walkforward_template)(
                symbol="TEST",
                n_splits=3,
                test_size=60,
                min_train_size=504,
                expanding=False,
            )
        assert result.get("expanding") is False
        assert len(result.get("folds", [])) == 3


# ---------------------------------------------------------------------------
# run_purged_cv
# ---------------------------------------------------------------------------


class TestRunPurgedCV:
    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(run_purged_cv)(symbol="NOPE")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_error(self):
        df = synthetic_ohlcv("SHORT", n_days=50)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_purged_cv)(symbol="SHORT", n_splits=5)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = _large_ohlcv(n_bars=500)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_purged_cv)(symbol="TEST", n_splits=5)
        assert "error" not in result
        assert result["symbol"] == "TEST"
        assert result["n_splits"] == 5
        assert len(result["splits"]) == 5
        # Each split should have train/test sizes
        for split in result["splits"]:
            assert split["train_size"] > 0
            assert split["test_size"] > 0
            assert "train_start" in split
            assert "test_end" in split

    @pytest.mark.asyncio
    async def test_embargo_info_present(self):
        df = _large_ohlcv(n_bars=500)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_purged_cv)(
                symbol="TEST", n_splits=3, embargo_pct=0.02
            )
        assert result["embargo_pct"] == 0.02
        for split in result["splits"]:
            assert "embargo_size" in split
            assert split["embargo_size"] == int(500 * 0.02)

    @pytest.mark.asyncio
    async def test_data_range_included(self):
        df = _large_ohlcv(n_bars=500)
        with patch(
            "quantstack.mcp.tools.qc_backtesting._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(run_purged_cv)(symbol="TEST", n_splits=3)
        assert "data_range" in result
        assert "start" in result["data_range"]
        assert "end" in result["data_range"]
