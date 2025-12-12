# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quant research metrics module."""

import pytest
import numpy as np
import pandas as pd

from quantcore.research.quant_metrics import (
    QuantResearchReport,
    run_signal_diagnostics,
    run_alpha_decay_analysis,
    compute_cost_adjusted_returns,
    compare_strategies,
)


@pytest.fixture
def sample_signal():
    """Create sample signal series."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    # Generate signal with some predictive power
    signal = pd.Series(np.random.randn(500).cumsum() / 10, index=dates)
    return signal.clip(-1, 1)


@pytest.fixture
def sample_returns(sample_signal):
    """Create sample returns correlated with signal."""
    np.random.seed(42)
    # Returns with some correlation to signal
    noise = np.random.randn(500) * 0.02
    returns = sample_signal.shift(1).fillna(0) * 0.001 + noise
    return pd.Series(returns, index=sample_signal.index)


class TestQuantResearchReport:
    """Tests for QuantResearchReport dataclass."""

    def test_default_values(self):
        """Test default report values."""
        report = QuantResearchReport()

        assert report.ic == 0.0
        assert report.sharpe_ratio == 0.0
        assert report.annual_turnover == 0.0
        assert report.is_stationary == False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = QuantResearchReport(
            ic=0.05,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
        )

        result = report.to_dict()

        assert "signal_quality" in result
        assert "performance_gross" in result
        assert "turnover" in result
        assert result["signal_quality"]["ic"] == 0.05


class TestRunSignalDiagnostics:
    """Tests for run_signal_diagnostics function."""

    def test_basic_diagnostics(self, sample_signal, sample_returns):
        """Test basic signal diagnostics."""
        report = run_signal_diagnostics(sample_signal, sample_returns, cost_bps=5.0)

        assert isinstance(report, QuantResearchReport)
        assert isinstance(report.ic, float)
        assert isinstance(report.sharpe_ratio, float)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        signal = pd.Series([1, -1, 1])
        returns = pd.Series([0.01, -0.01, 0.01])

        report = run_signal_diagnostics(signal, returns)

        # Should return empty report
        assert report.ic == 0.0

    def test_ic_calculation(self, sample_signal, sample_returns):
        """Test IC calculation produces reasonable values."""
        report = run_signal_diagnostics(sample_signal, sample_returns)

        # IC should be between -1 and 1
        assert -1 <= report.ic <= 1

    def test_turnover_calculation(self, sample_signal, sample_returns):
        """Test turnover calculation."""
        report = run_signal_diagnostics(sample_signal, sample_returns)

        # Turnover should be non-negative
        assert report.annual_turnover >= 0


class TestAlphaDecayAnalysis:
    """Tests for alpha decay analysis."""

    def test_decay_curve(self, sample_signal, sample_returns):
        """Test alpha decay curve generation."""
        result = run_alpha_decay_analysis(sample_signal, sample_returns, max_lag=10)

        assert "decay_curve" in result
        assert "half_life" in result
        assert len(result["decay_curve"]) == 10

    def test_half_life_calculation(self, sample_signal, sample_returns):
        """Test half-life calculation."""
        result = run_alpha_decay_analysis(sample_signal, sample_returns)

        assert result["half_life"] >= 1
        assert result["half_life"] <= 20


class TestCostAdjustedReturns:
    """Tests for cost-adjusted returns calculation."""

    def test_cost_adjustment(self, sample_signal, sample_returns):
        """Test cost adjustment calculation."""
        result = compute_cost_adjusted_returns(
            sample_signal,
            sample_returns,
            commission_bps=1.0,
            spread_bps=2.0,
            impact_bps=2.0,
        )

        assert "gross_returns" in result
        assert "net_returns" in result
        assert "costs" in result
        assert "cumulative_gross" in result
        assert "cumulative_net" in result

    def test_net_less_than_gross(self, sample_signal, sample_returns):
        """Test that net returns are less than gross due to costs."""
        result = compute_cost_adjusted_returns(
            sample_signal,
            sample_returns,
            commission_bps=5.0,
        )

        # Net cumulative should be less than or equal to gross
        assert (
            result["cumulative_net"].iloc[-1]
            <= result["cumulative_gross"].iloc[-1] + 0.01
        )


class TestCompareStrategies:
    """Tests for strategy comparison."""

    def test_comparison_output(self, sample_signal, sample_returns):
        """Test strategy comparison output."""
        # Create two different signals
        signal1 = sample_signal
        signal2 = -sample_signal  # Opposite signal

        strategies = {
            "momentum": signal1,
            "contrarian": signal2,
        }

        result = compare_strategies(strategies, sample_returns, cost_bps=5.0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # strategy is the index, not a column
        assert result.index.name == "strategy"
        assert "sharpe_gross" in result.columns

    def test_empty_strategies(self, sample_returns):
        """Test with empty strategies."""
        result = compare_strategies({}, sample_returns)

        assert isinstance(result, pd.DataFrame)
        # Empty DataFrame will have no columns and no index name
        assert len(result) == 0 or result.empty
