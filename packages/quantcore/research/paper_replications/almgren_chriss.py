"""
Almgren-Chriss Optimal Execution Model.

References:
    Almgren, R., & Chriss, N. (2001).
    "Optimal execution of portfolio transactions."
    Journal of Risk, 3, 5-40.

Key Results:
    - Trade-off between execution risk and market impact
    - Optimal trajectories depend on urgency (risk aversion)
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ExecutionParams:
    """Parameters for optimal execution."""

    total_shares: float
    horizon: int
    volatility: float
    daily_volume: float
    eta: float  # Temporary impact
    gamma: float  # Permanent impact
    lambda_risk: float  # Risk aversion


@dataclass
class ExecutionResult:
    """Result of optimal execution."""

    trajectory: np.ndarray
    trade_schedule: np.ndarray
    expected_cost: float
    variance: float


def optimal_trajectory(params: ExecutionParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal execution trajectory.

    Args:
        params: Execution parameters

    Returns:
        Tuple of (trajectory, trade_schedule)
    """
    T = params.horizon
    X = params.total_shares
    sigma = params.volatility
    eta = params.eta
    lam = params.lambda_risk

    kappa = np.sqrt(lam * sigma**2 / (eta + 1e-10))
    times = np.arange(T + 1)

    if kappa * T < 0.001:
        trajectory = X * (1 - times / T)
    else:
        sinh_kappa_T = np.sinh(kappa * T)
        trajectory = X * np.sinh(kappa * (T - times)) / sinh_kappa_T

    trade_schedule = -np.diff(trajectory)
    return trajectory, trade_schedule


def execution_cost(
    trade_schedule: np.ndarray, params: ExecutionParams
) -> Tuple[float, float]:
    """
    Compute expected cost and variance for a trade schedule.

    Args:
        trade_schedule: Shares to trade each period
        params: Execution parameters

    Returns:
        Tuple of (expected_cost, variance)
    """
    T = len(trade_schedule)
    sigma = params.volatility
    eta = params.eta
    gamma = params.gamma

    cumsum = np.cumsum(trade_schedule)
    permanent_cost = gamma * np.sum(trade_schedule * cumsum)
    temporary_cost = eta * np.sum(trade_schedule**2)
    expected_cost = permanent_cost + temporary_cost

    remaining = np.array([np.sum(trade_schedule[j:]) for j in range(T)])
    variance = sigma**2 * np.sum(remaining**2)

    return expected_cost, variance


class AlmgrenChrissExecutor:
    """
    Almgren-Chriss Optimal Execution.

    Example:
        executor = AlmgrenChrissExecutor(volatility=0.02, daily_volume=1_000_000)
        schedule = executor.optimal_schedule(shares=10000, horizon=20, risk_aversion=1e-6)
    """

    def __init__(
        self,
        volatility: float,
        daily_volume: float,
        eta: float = 2.5e-7,
        gamma: float = 2.5e-7,
    ):
        self.volatility = volatility
        self.daily_volume = daily_volume
        self.eta = eta
        self.gamma = gamma

    def optimal_schedule(
        self,
        shares: float,
        horizon: int,
        risk_aversion: float = 1e-6,
    ) -> ExecutionResult:
        """Compute optimal execution schedule."""
        params = ExecutionParams(
            total_shares=shares,
            horizon=horizon,
            volatility=self.volatility,
            daily_volume=self.daily_volume,
            eta=self.eta,
            gamma=self.gamma,
            lambda_risk=risk_aversion,
        )

        trajectory, schedule = optimal_trajectory(params)
        exp_cost, variance = execution_cost(schedule, params)

        return ExecutionResult(
            trajectory=trajectory,
            trade_schedule=schedule,
            expected_cost=exp_cost,
            variance=variance,
        )

    def twap_schedule(self, shares: float, horizon: int) -> np.ndarray:
        """Generate TWAP schedule."""
        return np.full(horizon, shares / horizon)

    def efficient_frontier(
        self, shares: float, horizon: int, n_points: int = 20
    ) -> dict:
        """Compute efficient frontier of execution strategies."""
        lambdas = np.logspace(-8, -4, n_points)
        costs = []
        variances = []

        for lam in lambdas:
            result = self.optimal_schedule(shares, horizon, lam)
            costs.append(result.expected_cost)
            variances.append(result.variance)

        return {
            "risk_aversion": lambdas,
            "expected_cost": np.array(costs),
            "variance": np.array(variances),
        }
