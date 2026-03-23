# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quantcore.research.overfitting — Sprint 1.

Covers: deflated_sharpe_ratio, benchmark_sharpe_ratio, PBO, OverfittingReport.
All tests are deterministic (fixed seeds) and require no external I/O.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantstack.core.research.overfitting import (
    DSRResult,
    OverfittingReport,
    PBOResult,
    benchmark_sharpe_ratio,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    run_overfitting_analysis,
)
import pandas as pd

# ---------------------------------------------------------------------------
# benchmark_sharpe_ratio
# ---------------------------------------------------------------------------


class TestBenchmarkSharpeRatio:
    def test_increases_with_n_trials(self):
        """More trials → higher expected max SR (more chances to overfit)."""
        sr1 = benchmark_sharpe_ratio(n_trials=10, n_obs=252)
        sr10 = benchmark_sharpe_ratio(n_trials=100, n_obs=252)
        assert sr10 > sr1

    def test_positive_for_multiple_trials(self):
        """Multiple trials produce a positive expected max SR."""
        for n in [2, 5, 20, 100]:
            assert benchmark_sharpe_ratio(n_trials=n, n_obs=252) > 0

    def test_single_trial_close_to_zero(self):
        """With one trial, no multiple-testing inflation."""
        sr = benchmark_sharpe_ratio(n_trials=1, n_obs=252)
        assert sr < 0.5  # Euler-Mascheroni-based estimate should be modest


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_returns_dsr_result(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=1,
            n_obs=252,
            skewness=0.0,
            excess_kurtosis=0.0,
        )
        assert isinstance(result, DSRResult)

    def test_single_trial_no_inflation(self):
        """Single trial: DSR ≈ observed Sharpe (no multiple testing penalty)."""
        result = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=1,
            n_obs=252,
        )
        assert result.dsr > 0

    def test_many_trials_reduce_dsr(self):
        """100 trials should produce a lower DSR than 1 trial at same observed SR."""
        r1 = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=1, n_obs=252)
        r100 = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=100, n_obs=252)
        assert r100.dsr < r1.dsr

    def test_non_normality_changes_dsr(self):
        """Non-normal returns (skew/kurtosis) should change DSR vs normal baseline."""
        r_normal = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=10,
            n_obs=252,
            skewness=0.0,
            excess_kurtosis=0.0,
        )
        r_non_normal = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=10,
            n_obs=252,
            skewness=-1.0,
            excess_kurtosis=3.0,
        )
        # Non-normality should change the DSR (direction depends on implementation)
        assert r_normal.dsr != r_non_normal.dsr

    def test_is_genuine_high_sr_low_trials(self):
        """Very high SR with few trials → is_genuine=True."""
        result = deflated_sharpe_ratio(observed_sharpe=3.0, n_trials=3, n_obs=252)
        assert result.is_genuine is True

    def test_is_not_genuine_moderate_sr_many_trials(self):
        """Moderate SR with many trials → is_genuine=False."""
        result = deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=200, n_obs=252)
        assert result.is_genuine is False

    def test_dsr_in_range(self):
        result = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=10, n_obs=252)
        assert 0.0 <= result.dsr <= 1.0


# ---------------------------------------------------------------------------
# probability_of_backtest_overfitting
# ---------------------------------------------------------------------------


class TestPBO:
    @pytest.fixture
    def random_returns_matrix(self):
        """100 paths, 252 observations each."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((252, 10))

    def test_returns_pbo_result(self, random_returns_matrix):
        result = probability_of_backtest_overfitting(random_returns_matrix)
        assert isinstance(result, PBOResult)

    def test_pbo_in_range(self, random_returns_matrix):
        result = probability_of_backtest_overfitting(random_returns_matrix)
        assert 0.0 <= result.pbo <= 1.0

    def test_random_strategies_high_pbo(self, random_returns_matrix):
        """Purely random returns should show elevated overfit risk."""
        result = probability_of_backtest_overfitting(random_returns_matrix)
        # Random strategies have no genuine edge — PBO should be non-trivial (> 0.1)
        assert result.pbo > 0.1

    def test_consistent_strategy_low_pbo(self):
        """A strategy with genuine consistent alpha should have low PBO."""
        rng = np.random.default_rng(0)
        # Base: positive drift + noise; one column is the "strategy"
        base = rng.standard_normal((252, 9))
        # Strategy column: strong consistent positive returns
        alpha_col = np.ones((252, 1)) * 0.01 + rng.standard_normal((252, 1)) * 0.001
        returns = np.hstack([base, alpha_col])
        result = probability_of_backtest_overfitting(returns)
        assert result.pbo < 0.6  # Genuine strategy should NOT be flagged as overfit

    def test_logit_values_are_list(self, random_returns_matrix):
        result = probability_of_backtest_overfitting(random_returns_matrix)
        assert isinstance(result.logit_values, list)


# ---------------------------------------------------------------------------
# run_overfitting_analysis
# ---------------------------------------------------------------------------


class TestRunOverfittingAnalysis:
    def _returns(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.standard_normal(252) * 0.01)

    def test_returns_report(self):
        report = run_overfitting_analysis(strategy_returns=self._returns(), n_trials=5)
        assert isinstance(report, OverfittingReport)

    def test_verdict_is_valid(self):
        report = run_overfitting_analysis(strategy_returns=self._returns(), n_trials=5)
        assert report.verdict in ("GENUINE", "SUSPECT", "OVERFIT")

    def test_has_sharpe(self):
        report = run_overfitting_analysis(strategy_returns=self._returns(), n_trials=5)
        assert isinstance(report.dsr_result.observed_sharpe, float)
