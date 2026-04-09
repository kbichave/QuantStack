"""Tests for 4.5 — Monte Carlo Validation Gate (QS-B3)."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.analysis.bootstrap_mc import BootstrapResult, bootstrap_sharpe_ci


class TestBootstrapSharpeCi:
    def test_strongly_positive_returns(self):
        """Strongly positive returns should have high 5th percentile Sharpe."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.005, 252))  # ~0.1% daily
        result = bootstrap_sharpe_ci(returns, n_simulations=500)

        assert isinstance(result, BootstrapResult)
        assert result.mean_sharpe > 1.0
        assert result.ci_5 > 0.3  # passes the gate
        assert result.prob_negative < 0.05

    def test_noise_returns_rejected(self):
        """Pure noise (mean ~0) should fail the 5th percentile gate."""
        rng = np.random.default_rng(123)
        # Use exactly zero mean to ensure noise-only returns
        raw = rng.normal(0.0, 0.01, 252)
        raw -= raw.mean()  # force exact zero mean
        returns = pd.Series(raw)
        result = bootstrap_sharpe_ci(returns, n_simulations=500)

        # 5th percentile should be well below the 0.3 gate
        assert result.ci_5 < 0.3

    def test_negative_returns(self):
        """Negative mean returns should have mostly negative Sharpes."""
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(-0.001, 0.005, 252))
        result = bootstrap_sharpe_ci(returns, n_simulations=500)

        assert result.mean_sharpe < 0
        assert result.prob_negative > 0.9

    def test_too_few_returns_raises(self):
        """Fewer than 30 returns should raise ValueError."""
        with pytest.raises(ValueError, match="requires >= 30"):
            bootstrap_sharpe_ci(pd.Series([0.01] * 20))

    def test_ci_ordering(self):
        """CI percentiles should be ordered: 5 <= 25 <= 75 <= 95."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        result = bootstrap_sharpe_ci(returns, n_simulations=500)

        assert result.ci_5 <= result.ci_25
        assert result.ci_25 <= result.ci_75
        assert result.ci_75 <= result.ci_95

    def test_block_size_auto(self):
        """Auto block size should be max(5, n//20)."""
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
        result = bootstrap_sharpe_ci(returns, n_simulations=100)
        assert result.block_size == max(5, 200 // 20)

    def test_reproducibility(self):
        """Same seed produces same result."""
        returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 252))
        r1 = bootstrap_sharpe_ci(returns, seed=42, n_simulations=100)
        r2 = bootstrap_sharpe_ci(returns, seed=42, n_simulations=100)
        assert r1.mean_sharpe == r2.mean_sharpe
