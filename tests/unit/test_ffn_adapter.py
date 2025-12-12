# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ffn adapter."""

import pytest
import numpy as np
import pandas as pd


class TestFFNAdapter:
    """Test suite for ffn adapter functions."""

    def test_compute_portfolio_stats_basic(self):
        """Test basic portfolio stats computation."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        # Create simple equity curve
        equity = pd.Series([100, 101, 102, 101, 103, 105, 104, 106])

        stats = compute_portfolio_stats_ffn(equity)

        assert "total_return" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown" in stats
        assert stats["total_return"] > 0

    def test_portfolio_stats_with_returns(self):
        """Test stats with known returns."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        # 10% total return over period
        equity = pd.Series([100, 102, 104, 106, 108, 110])

        stats = compute_portfolio_stats_ffn(equity, periods_per_year=252)

        # Total return should be 10%
        assert abs(stats["total_return"] - 0.10) < 0.01

    def test_max_drawdown_calculation(self):
        """Test max drawdown is calculated correctly."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        # Clear 20% drawdown in the middle
        equity = pd.Series([100, 110, 100, 88, 90, 95, 100, 105])

        stats = compute_portfolio_stats_ffn(equity)

        # Max DD from 110 to 88 = -20%
        assert stats["max_drawdown"] < -0.15

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio for consistently positive returns."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        # Consistently positive returns -> high Sharpe
        np.random.seed(42)
        returns = 0.001 + np.random.randn(100) * 0.005  # Positive drift, low vol
        equity = pd.Series(100 * np.cumprod(1 + returns))

        stats = compute_portfolio_stats_ffn(equity, periods_per_year=252)

        # Should have positive Sharpe
        assert stats["sharpe_ratio"] > 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        np.random.seed(42)
        equity = pd.Series(100 * np.cumprod(1 + np.random.randn(100) * 0.01))

        stats = compute_portfolio_stats_ffn(equity)

        assert "sortino_ratio" in stats

    def test_calmar_ratio(self):
        """Test Calmar ratio (return / max DD)."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        equity = pd.Series([100, 105, 110, 100, 105, 115, 120])

        stats = compute_portfolio_stats_ffn(equity, periods_per_year=252)

        assert "calmar_ratio" in stats

    def test_distribution_metrics(self):
        """Test distribution metrics."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        np.random.seed(42)
        equity = pd.Series(100 * np.cumprod(1 + np.random.randn(100) * 0.01))

        stats = compute_portfolio_stats_ffn(equity)

        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "var_95" in stats
        assert "cvar_95" in stats

    def test_factor_stats(self):
        """Test factor/benchmark relative stats."""
        from quantcore.analytics.adapters.ffn_adapter import compute_factor_stats_ffn

        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        benchmark = pd.Series(np.random.randn(100) * 0.008)

        stats = compute_factor_stats_ffn(returns, benchmark)

        assert "alpha" in stats
        assert "beta" in stats
        assert "information_ratio" in stats
        assert "tracking_error" in stats

    def test_up_down_capture(self):
        """Test up/down capture ratios."""
        from quantcore.analytics.adapters.ffn_adapter import compute_factor_stats_ffn

        # Strategy that captures more upside
        np.random.seed(42)
        benchmark = pd.Series(np.random.randn(100) * 0.01)
        returns = benchmark * 1.2  # 20% more exposure

        stats = compute_factor_stats_ffn(returns, benchmark)

        assert "up_capture" in stats
        assert "down_capture" in stats

    def test_generate_tearsheet_data(self):
        """Test tearsheet data generation."""
        from quantcore.analytics.adapters.ffn_adapter import generate_tearsheet_data

        np.random.seed(42)
        equity = pd.Series(
            100 * np.cumprod(1 + np.random.randn(100) * 0.01),
            index=pd.date_range("2023-01-01", periods=100, freq="D"),
        )

        tearsheet = generate_tearsheet_data(equity)

        assert "summary_stats" in tearsheet
        assert "drawdown_periods" in tearsheet
        assert "equity_curve" in tearsheet

    def test_empty_equity_curve(self):
        """Test handling of insufficient data."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        equity = pd.Series([100])

        result = compute_portfolio_stats_ffn(equity)

        assert "error" in result

    def test_list_input(self):
        """Test handling of list input."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        equity_list = [100, 101, 102, 103, 104, 105]

        stats = compute_portfolio_stats_ffn(equity_list)

        assert "total_return" in stats


class TestFFNFallback:
    """Test internal fallback when ffn not available."""

    def test_internal_stats(self):
        """Test internal stats calculation."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        equity = pd.Series([100, 102, 104, 103, 105, 107, 106, 108])

        # Should work regardless of ffn availability
        stats = compute_portfolio_stats_ffn(equity)

        assert "total_return" in stats
        assert "max_drawdown" in stats
