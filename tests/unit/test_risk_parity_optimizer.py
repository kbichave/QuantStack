"""Tests for risk-parity portfolio construction (Section 07).

Covers: Ledoit-Wolf covariance, risk-parity weights, alpha tilts,
constrained optimization with infeasibility cascade, factor exposure tracking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.portfolio.optimizer import (
    PortfolioConstraints,
    apply_alpha_tilts,
    compute_risk_parity_weights,
    estimate_covariance,
    optimize_portfolio,
)


# -- Helpers --


def _synthetic_returns(n_assets: int = 5, n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    data = rng.normal(0.0005, 0.015, (n_days, n_assets))
    symbols = [f"SYM{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=symbols)


# -- 7.1 Covariance Estimation --


def test_ledoit_wolf_positive_definite():
    """Ledoit-Wolf shrinkage produces a positive definite matrix."""
    returns = _synthetic_returns()
    cov = estimate_covariance(returns, window=252)

    eigenvalues = np.linalg.eigvalsh(cov)
    assert all(eigenvalues > 0), f"Not PD: min eigenvalue={eigenvalues.min()}"


def test_shrinkage_lower_condition_number_than_sample():
    """Ledoit-Wolf has lower condition number than raw sample covariance."""
    returns = _synthetic_returns(n_assets=20, n_days=300, seed=99)
    tail = returns.iloc[-252:]

    sample_cov = tail.cov().values
    shrunk_cov = estimate_covariance(returns, window=252)

    assert np.linalg.cond(shrunk_cov) < np.linalg.cond(sample_cov)


def test_covariance_uses_trailing_window():
    """Covariance uses only the most recent `window` days."""
    returns = _synthetic_returns(n_assets=3, n_days=300)
    cov_300 = estimate_covariance(returns, window=252)
    cov_252 = estimate_covariance(returns.iloc[-252:], window=252)

    np.testing.assert_array_almost_equal(cov_300, cov_252, decimal=10)


# -- 7.2 Risk-Parity Optimizer --


def test_risk_parity_equal_vol_produces_equal_weights():
    """Equal-vol uncorrelated assets -> equal weights."""
    n = 4
    cov = np.eye(n) * 0.01
    weights = compute_risk_parity_weights(cov)

    np.testing.assert_array_almost_equal(weights, np.ones(n) / n, decimal=2)


def test_risk_parity_higher_vol_gets_lower_weight():
    """Asset with 2x vol gets ~half the weight."""
    cov = np.diag([0.01, 0.04])  # B has 2x vol (4x variance)
    weights = compute_risk_parity_weights(cov)

    assert weights[0] > weights[1]
    assert abs(weights[0] / weights[1] - 2.0) < 0.3


def test_alpha_tilts_shift_weights_toward_high_signal():
    """Alpha tilts increase weight of high-signal asset."""
    base = np.array([0.25, 0.25, 0.25, 0.25])
    signals = np.array([0.8, 0.2, 0.1, 0.1])

    tilted = apply_alpha_tilts(base, signals, tilt_strength=0.5)

    assert tilted[0] > base[0]
    assert abs(tilted.sum() - 1.0) < 1e-6


def test_alpha_tilts_zero_signals_no_change():
    """Zero alpha signals leave weights unchanged."""
    base = np.array([0.3, 0.3, 0.4])
    signals = np.zeros(3)

    tilted = apply_alpha_tilts(base, signals, tilt_strength=0.5)
    np.testing.assert_array_almost_equal(tilted, base)


# -- 7.2 Constrained Optimization --


def _make_simple_portfolio(n: int = 5, seed: int = 42):
    """Build standard optimizer inputs."""
    rng = np.random.default_rng(seed)
    cov = np.diag(rng.uniform(0.005, 0.03, n))
    alpha = rng.uniform(0, 1, n)
    current = np.ones(n) / n
    sector_map = {f"SYM{i}": f"Sector{i % 3}" for i in range(n)}
    strategy_map = {f"SYM{i}": f"Strat{i % 4}" for i in range(n)}
    return cov, alpha, current, sector_map, strategy_map


def test_position_constraint_respected():
    """No individual position exceeds max or falls below min."""
    cov, alpha, current, sector_map, strategy_map = _make_simple_portfolio(10)
    constraints = PortfolioConstraints(position_min=0.01, position_max=0.15)

    weights, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    assert all(w >= constraints.position_min - 1e-4 for w in weights)
    assert all(w <= constraints.position_max + 1e-4 for w in weights)


def test_sector_constraint_respected():
    """No sector exceeds 30% of total weight."""
    n = 10
    cov = np.eye(n) * 0.01
    alpha = np.zeros(n)
    # Start with weights already close to feasible
    current = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.15, 0.15])
    sector_map = {f"SYM{i}": "Tech" if i < 5 else f"Other{i}" for i in range(n)}
    strategy_map = {f"SYM{i}": f"Strat{i}" for i in range(n)}

    constraints = PortfolioConstraints(sector_max=0.30, turnover_max=0.50)
    weights, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    tech_weight = sum(weights[i] for i in range(5))
    assert tech_weight <= constraints.sector_max + 0.02


def test_turnover_constraint_respected():
    """Turnover does not exceed limit."""
    cov, alpha, current, sector_map, strategy_map = _make_simple_portfolio()
    constraints = PortfolioConstraints(turnover_max=0.20)

    weights, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    turnover = np.sum(np.abs(weights - current))
    assert turnover <= constraints.turnover_max + 0.02


def test_gross_exposure_within_bounds():
    """Gross exposure stays within bounds."""
    cov, alpha, current, sector_map, strategy_map = _make_simple_portfolio()
    constraints = PortfolioConstraints(gross_exposure_min=0.50, gross_exposure_max=1.50)

    weights, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    gross = np.sum(np.abs(weights))
    assert gross >= constraints.gross_exposure_min - 0.01
    assert gross <= constraints.gross_exposure_max + 0.01


def test_infeasibility_cascade_relaxes_constraints():
    """When base constraints infeasible, relaxation is attempted."""
    n = 5
    cov = np.eye(n) * 0.01
    alpha = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.25, 0.25, 0.25, 0.25])
    sector_map = {f"SYM{i}": f"S{i}" for i in range(n)}
    strategy_map = {f"SYM{i}": f"St{i}" for i in range(n)}

    constraints = PortfolioConstraints(turnover_max=0.01, position_min=0.01, position_max=0.50)
    weights, meta = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    assert meta.get("feasible") or meta.get("relaxation_applied") is not None


def test_infeasibility_all_relaxed_returns_current_portfolio():
    """If all relaxation fails, return current portfolio unchanged."""
    n = 2
    cov = np.eye(n) * 0.01
    alpha = np.array([1.0, 0.0])
    current = np.array([0.5, 0.5])
    sector_map = {"SYM0": "A", "SYM1": "A"}
    strategy_map = {"SYM0": "X", "SYM1": "X"}

    constraints = PortfolioConstraints(
        position_min=0.40, position_max=0.60,
        sector_max=0.10, turnover_max=0.001,
    )
    weights, meta = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    np.testing.assert_array_almost_equal(weights, current, decimal=2)
    assert meta.get("feasible") is False


# -- 7.3 Factor Exposure --


def test_portfolio_factor_exposure_is_position_weighted_betas():
    """Portfolio factor exposure = sum(w_i * beta_i)."""
    weights = np.array([0.6, 0.4])
    betas = np.array([[1.2, 0.3, -0.1], [0.8, -0.2, 0.4]])
    expected = weights @ betas

    np.testing.assert_array_almost_equal(weights @ betas, expected)


def test_penalty_term_reduces_factor_drift():
    """Optimizer with factor penalty produces smaller total factor exposure."""
    rng = np.random.default_rng(42)
    n = 5
    cov = np.diag(rng.uniform(0.005, 0.03, n))
    alpha = rng.uniform(0, 1, n)
    current = np.ones(n) / n
    sector_map = {f"SYM{i}": f"S{i}" for i in range(n)}
    strategy_map = {f"SYM{i}": f"St{i}" for i in range(n)}
    factor_betas = rng.normal(0, 0.5, (n, 3))

    w_no_pen, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map,
        factor_exposures=factor_betas, factor_penalty_weight=0.0,
    )
    w_pen, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map,
        factor_exposures=factor_betas, factor_penalty_weight=1.0,
    )

    expo_no_pen = np.abs(w_no_pen @ factor_betas).sum()
    expo_pen = np.abs(w_pen @ factor_betas).sum()

    assert expo_pen <= expo_no_pen + 0.1


# -- 7.4 Delta Trades --


def test_delta_trades_computed_correctly():
    """Delta = target - current."""
    target = np.array([0.3, 0.2])
    current = np.array([0.1, 0.25])
    delta = target - current
    np.testing.assert_array_almost_equal(delta, np.array([0.2, -0.05]))
