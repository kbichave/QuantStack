# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quantcore.portfolio.optimizer — Sprint 3.

Tests MeanVarianceOptimizer across objectives and constraint configurations.
All tests are pure in-memory, no external I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.portfolio.optimizer import (
    MeanVarianceOptimizer,
    OptimizationObjective,
    OptimizationResult,
    PortfolioConstraints,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def symbols():
    return ["SPY", "QQQ", "GLD", "TLT", "IWM"]


@pytest.fixture
def signals(symbols):
    """Mildly positive expected returns for all symbols."""
    return {s: v for s, v in zip(symbols, [0.08, 0.12, 0.04, 0.03, 0.09])}


@pytest.fixture
def cov_matrix(symbols):
    """Synthetic covariance matrix with reasonable correlations."""
    np.random.seed(42)
    n = len(symbols)
    # Positive-definite covariance: small random off-diagonal
    A = np.random.randn(n, n) * 0.03
    cov = A @ A.T + np.diag([0.04] * n)  # Diagonal dominance
    return pd.DataFrame(cov, index=symbols, columns=symbols)


@pytest.fixture
def optimizer():
    return MeanVarianceOptimizer(risk_free_rate=0.05)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasicOptimization:
    def test_returns_optimization_result(self, optimizer, signals, cov_matrix):
        result = optimizer.optimize(signals, cov_matrix)
        assert isinstance(result, OptimizationResult)

    def test_weights_sum_to_one(self, optimizer, signals, cov_matrix):
        result = optimizer.optimize(signals, cov_matrix)
        total = sum(result.target_weights.values())
        assert abs(total - 1.0) < 1e-4

    def test_all_symbols_in_result(self, optimizer, signals, cov_matrix, symbols):
        result = optimizer.optimize(signals, cov_matrix)
        for s in symbols:
            assert s in result.target_weights

    def test_weights_non_negative_long_only(self, optimizer, signals, cov_matrix):
        result = optimizer.optimize(
            signals, cov_matrix, PortfolioConstraints(min_weight=0.0)
        )
        for w in result.target_weights.values():
            assert w >= -1e-6  # Allow tiny floating point errors

    def test_expected_sharpe_positive(self, optimizer, signals, cov_matrix):
        result = optimizer.optimize(signals, cov_matrix)
        assert result.expected_sharpe > 0

    def test_no_symbol_overlap_raises(self, optimizer):
        with pytest.raises(ValueError, match="No symbols overlap"):
            optimizer.optimize(
                {"ABC": 0.10},
                pd.DataFrame([[0.04]], index=["XYZ"], columns=["XYZ"]),
            )


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_max_weight_respected(self, optimizer, signals, cov_matrix):
        constraints = PortfolioConstraints(max_weight=0.25)
        result = optimizer.optimize(signals, cov_matrix, constraints)
        for w in result.target_weights.values():
            assert w <= 0.25 + 1e-4  # small floating-point tolerance

    def test_min_weight_respected(self, optimizer, signals, cov_matrix):
        constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.40)
        result = optimizer.optimize(signals, cov_matrix, constraints)
        # Positive weights should be >= min (0.0 weights allowed for infeasible)
        for w in result.target_weights.values():
            assert w >= 0.05 - 1e-4 or w < 1e-6


# ---------------------------------------------------------------------------
# Multiple objectives
# ---------------------------------------------------------------------------


class TestObjectives:
    def test_min_variance_converges(self, optimizer, signals, cov_matrix):
        """MIN_VARIANCE optimization should converge successfully."""
        result = optimizer.optimize(
            signals, cov_matrix, objective=OptimizationObjective.MIN_VARIANCE
        )
        assert result.converged
        assert result.expected_volatility > 0

    def test_risk_parity_result(self, optimizer, signals, cov_matrix):
        result = optimizer.optimize(
            signals, cov_matrix, objective=OptimizationObjective.RISK_PARITY
        )
        assert isinstance(result, OptimizationResult)
        assert abs(sum(result.target_weights.values()) - 1.0) < 1e-4

    def test_max_diversification_result(self, optimizer, signals, cov_matrix):
        result = optimizer.optimize(
            signals, cov_matrix, objective=OptimizationObjective.MAX_DIVERSIFICATION
        )
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Required trades (current → target)
# ---------------------------------------------------------------------------


class TestRequiredTrades:
    def test_required_trades_computed_when_current_given(
        self, optimizer, signals, cov_matrix, symbols
    ):
        current = {s: 1.0 / len(symbols) for s in symbols}  # Equal weight
        result = optimizer.optimize(signals, cov_matrix, current_weights=current)
        assert result.required_trades is not None

    def test_required_trades_none_when_current_not_given(
        self, optimizer, signals, cov_matrix
    ):
        result = optimizer.optimize(signals, cov_matrix)
        assert result.required_trades is None

    def test_required_trades_sum_near_zero(self, optimizer, signals, cov_matrix, symbols):
        """Net trades should sum close to zero (portfolio stays fully invested)."""
        current = {s: 1.0 / len(symbols) for s in symbols}
        result = optimizer.optimize(signals, cov_matrix, current_weights=current)
        net = sum(result.required_trades.values())
        assert abs(net) < 0.05  # Small net due to floating point
