"""
Brownian Motion and Geometric Brownian Motion.

Fundamental stochastic processes for financial modeling.
"""

import numpy as np
from typing import Optional, Tuple


def brownian_motion(
    n_steps: int,
    dt: float = 1.0,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate standard Brownian motion (Wiener process).

    Args:
        n_steps: Number of time steps
        dt: Time step size
        n_paths: Number of independent paths
        seed: Random seed

    Returns:
        Array of shape (n_paths, n_steps+1) with paths
    """
    if seed is not None:
        np.random.seed(seed)

    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    return W


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Geometric Brownian Motion (GBM).

    dS = mu * S * dt + sigma * S * dW

    Args:
        S0: Initial price
        mu: Drift (annualized return)
        sigma: Volatility (annualized)
        T: Time horizon (years)
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        seed: Random seed

    Returns:
        Tuple of (time_grid, paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    W = brownian_motion(n_steps, dt, n_paths)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W

    S = S0 * np.exp(drift + diffusion)

    return t, S


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate GBM paths with variance reduction.

    Args:
        S0: Initial price
        mu: Drift
        sigma: Volatility
        T: Time horizon
        n_steps: Number of steps
        n_paths: Number of paths
        antithetic: Use antithetic variates
        seed: Random seed

    Returns:
        Array of terminal prices
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    Z = np.random.normal(0, 1, size=(n_paths, n_steps))

    if antithetic:
        Z = np.vstack([Z, -Z])

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.log(S0) + np.cumsum(log_returns, axis=1)

    return np.exp(log_S[:, -1])
