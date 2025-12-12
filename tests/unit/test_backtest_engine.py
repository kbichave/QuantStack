# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.backtesting.engine module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.backtesting.engine import (
    calculate_metrics,
    run_backtest_with_params,
)


class TestCalculateMetrics:
    """Test calculate_metrics function."""

    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        initial = 100000
        final = 110000
        trades = [
            {"pnl": 5000},
            {"pnl": 3000},
            {"pnl": -2000},
            {"pnl": 4000},
        ]
        equity = [100000, 102000, 105000, 103000, 110000]

        metrics = calculate_metrics(final, initial, trades, equity)

        assert metrics["initial_capital"] == initial
        assert metrics["final_capital"] == final
        assert metrics["total_pnl"] == 10000
        assert metrics["total_return_pct"] == pytest.approx(10.0)
        assert metrics["total_trades"] == 4
        assert metrics["profitable_trades"] == 3
        assert metrics["win_rate"] == pytest.approx(75.0)

    def test_no_trades(self):
        """Test metrics with no trades."""
        metrics = calculate_metrics(100000, 100000, [], [100000])

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["total_pnl"] == 0

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        equity = [100000 + i * 100 for i in range(100)]  # Steady growth
        metrics = calculate_metrics(109900, 100000, [], equity)

        # Positive returns should give positive Sharpe
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Peak at 110, then drop to 90
        equity = [100000, 105000, 110000, 100000, 95000, 105000]
        metrics = calculate_metrics(105000, 100000, [], equity)

        # Max DD should be (110000 - 95000) / 110000 ≈ 13.6%
        assert metrics["max_drawdown"] > 10
        assert metrics["max_drawdown"] < 20


class TestRunBacktestWithParams:
    """Test run_backtest_with_params function."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample spread data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create mean-reverting spread
        spread = np.cumsum(np.random.randn(100) * 0.5)
        spread_zscore = (spread - spread.mean()) / spread.std()

        return pd.DataFrame(
            {
                "spread": spread,
                "spread_zscore": spread_zscore,
            },
            index=dates,
        )

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = pd.DataFrame({"spread": [1, 2], "spread_zscore": [0, 0.5]})
        results = run_backtest_with_params(
            data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        assert results["total_trades"] == 0
        assert results["final_capital"] == 100000

    def test_basic_backtest(self, sample_data):
        """Test basic backtest execution."""
        results = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        assert "final_capital" in results
        assert "total_trades" in results
        assert "win_rate" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results

    def test_with_stop_loss(self, sample_data):
        """Test backtest with stop loss."""
        results = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=3.0,
        )

        assert "final_capital" in results
        assert results["final_capital"] != 0

    def test_returns_trades_list(self, sample_data):
        """Test returning trades list."""
        results = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=1.5,  # Lower threshold to get more trades
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
            return_trades_list=True,
        )

        assert "trades_list" in results
        assert "equity_curve" in results

    def test_equity_curve_length(self, sample_data):
        """Test equity curve has correct length."""
        results = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
            return_trades_list=True,
        )

        # Equity curve should have one entry per bar
        assert len(results["equity_curve"]) == len(sample_data)

    def test_position_size_impact(self, sample_data):
        """Test position size affects returns."""
        results_small = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=100,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        results_large = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=10000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        # Larger position should have more extreme PnL
        pnl_small = abs(results_small["total_pnl"])
        pnl_large = abs(results_large["total_pnl"])

        # If there are trades, larger position should have larger absolute PnL
        if results_small["total_trades"] > 0 and results_large["total_trades"] > 0:
            assert pnl_large > pnl_small

    def test_different_entry_thresholds(self, sample_data):
        """Test different entry thresholds produce different results."""
        results_tight = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=1.0,  # Tight threshold - more trades
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        results_wide = run_backtest_with_params(
            sample_data,
            initial_capital=100000,
            entry_zscore=3.0,  # Wide threshold - fewer trades
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        # Tighter threshold should result in more trades
        assert results_tight["total_trades"] >= results_wide["total_trades"]


class TestBacktestEdgeCases:
    """Test backtest edge cases."""

    def test_all_nan_zscore(self):
        """Test with all NaN z-scores."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "spread": np.random.randn(50),
                "spread_zscore": [np.nan] * 50,
            },
            index=dates,
        )

        results = run_backtest_with_params(
            data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        # Should complete without error
        assert results["final_capital"] == 100000

    def test_extreme_zscores(self):
        """Test with extreme z-scores."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "spread": np.linspace(-10, 10, 50),
                "spread_zscore": np.linspace(-5, 5, 50),  # Very extreme
            },
            index=dates,
        )

        results = run_backtest_with_params(
            data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        assert results["total_trades"] > 0  # Should have trades

    def test_constant_spread(self):
        """Test with constant spread."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "spread": [1.0] * 50,
                "spread_zscore": [0.0] * 50,
            },
            index=dates,
        )

        results = run_backtest_with_params(
            data,
            initial_capital=100000,
            entry_zscore=2.0,
            exit_zscore=0.0,
            position_size=1000,
            spread_cost=0.05,
            stop_loss_zscore=None,
        )

        # No trades should occur with constant spread
        assert results["total_trades"] == 0
