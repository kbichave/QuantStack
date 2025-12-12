"""
Optimization Utilities for Quantitative Finance.

Portfolio optimization and constrained minimization.
"""

import numpy as np
from typing import Optional
from scipy.optimize import minimize


def portfolio_optimize(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    max_weight: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Mean-variance portfolio optimization.

    Maximizes: E[R] - (risk_aversion/2) * Var[R]

    Args:
        expected_returns: Expected return for each asset
        covariance: Covariance matrix
        risk_aversion: Risk aversion parameter
        max_weight: Maximum weight per asset
        long_only: If True, no short selling

    Returns:
        Optimal weights
    """
    n_assets = len(expected_returns)

    def objective(w):
        port_return = w @ expected_returns
        port_variance = w @ covariance @ w
        return -port_return + 0.5 * risk_aversion * port_variance

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if long_only:
        bounds = [(0, max_weight) for _ in range(n_assets)]
    else:
        bounds = [(-max_weight, max_weight) for _ in range(n_assets)]

    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )

    return result.x


def minimum_variance_portfolio(
    covariance: np.ndarray, long_only: bool = True
) -> np.ndarray:
    """Compute minimum variance portfolio."""
    n = covariance.shape[0]

    if not long_only:
        ones = np.ones(n)
        cov_inv = np.linalg.inv(covariance)
        return cov_inv @ ones / (ones @ cov_inv @ ones)

    def objective(w):
        return w @ covariance @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def maximum_sharpe_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> np.ndarray:
    """Compute maximum Sharpe ratio portfolio."""
    n = len(expected_returns)
    excess_returns = expected_returns - risk_free_rate

    def neg_sharpe(w):
        port_return = w @ excess_returns
        port_vol = np.sqrt(w @ covariance @ w)
        if port_vol < 1e-10:
            return 0
        return -port_return / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) if long_only else (-1, 1) for _ in range(n)]

    best_result = None
    best_sharpe = -np.inf

    for _ in range(5):
        w0 = np.random.dirichlet(np.ones(n))
        result = minimize(
            neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success and -result.fun > best_sharpe:
            best_sharpe = -result.fun
            best_result = result

    return best_result.x if best_result else np.ones(n) / n
