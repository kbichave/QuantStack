"""
Stochastic Volatility Models.

Implementation of the Heston model and related stochastic volatility processes.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HestonParams:
    """Heston model parameters."""

    S0: float  # Initial stock price
    v0: float  # Initial variance
    kappa: float  # Mean reversion speed
    theta: float  # Long-run variance
    sigma: float  # Vol of vol
    rho: float  # Correlation between price and vol
    r: float  # Risk-free rate


def simulate_heston(
    params: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Heston stochastic volatility model.

    dS = r*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2

    Args:
        params: Heston model parameters
        T: Time horizon
        n_steps: Number of time steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        Tuple of (time_grid, price_paths, variance_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = params.S0
    v[:, 0] = params.v0

    for i in range(n_steps):
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = np.random.normal(0, 1, n_paths)

        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (params.rho * Z1 + np.sqrt(1 - params.rho**2) * Z2)

        v_current = np.maximum(v[:, i], 0)

        v[:, i + 1] = (
            v_current
            + params.kappa * (params.theta - v_current) * dt
            + params.sigma * np.sqrt(v_current) * dW2
        )
        v[:, i + 1] = np.maximum(v[:, i + 1], 0)

        S[:, i + 1] = S[:, i] * np.exp(
            (params.r - 0.5 * v_current) * dt + np.sqrt(v_current) * dW1
        )

    return t, S, v


class HestonModel:
    """
    Heston Stochastic Volatility Model.

    Example:
        model = HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04,
                           sigma=0.3, rho=-0.7, r=0.05)
        t, S, v = model.simulate(T=1.0, n_steps=252, n_paths=10000)
    """

    def __init__(
        self,
        S0: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        r: float = 0.0,
    ):
        self.params = HestonParams(S0, v0, kappa, theta, sigma, rho, r)
        self.feller_satisfied = 2 * kappa * theta >= sigma**2

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate price and variance paths."""
        return simulate_heston(self.params, T, n_steps, n_paths, seed)
