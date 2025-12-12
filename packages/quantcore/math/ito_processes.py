"""
Ito Processes and Stochastic Integration.

Numerical methods for solving stochastic differential equations (SDEs).
"""

import numpy as np
from typing import Callable, Optional, Tuple


def euler_maruyama(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama method for SDE simulation.

    Solves: dX = a(t, X) dt + b(t, X) dW

    Args:
        drift: Drift function a(t, x)
        diffusion: Diffusion function b(t, x)
        x0: Initial value
        T: Time horizon
        n_steps: Number of steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        Tuple of (time_grid, paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0

    for i in range(n_steps):
        dW = np.random.normal(0, sqrt_dt, n_paths)
        a = drift(t[i], X[:, i])
        b = diffusion(t[i], X[:, i])
        X[:, i + 1] = X[:, i] + a * dt + b * dW

    return t, X


def milstein(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    diffusion_deriv: Callable[[float, float], float],
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Milstein method for SDE simulation (higher order).

    Args:
        drift: Drift function a(t, x)
        diffusion: Diffusion function b(t, x)
        diffusion_deriv: Derivative db/dx
        x0: Initial value
        T: Time horizon
        n_steps: Number of steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        Tuple of (time_grid, paths)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0

    for i in range(n_steps):
        dW = np.random.normal(0, sqrt_dt, n_paths)
        a = drift(t[i], X[:, i])
        b = diffusion(t[i], X[:, i])
        b_prime = diffusion_deriv(t[i], X[:, i])

        X[:, i + 1] = X[:, i] + a * dt + b * dW + 0.5 * b * b_prime * (dW**2 - dt)

    return t, X


def ornstein_uhlenbeck(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Ornstein-Uhlenbeck process.

    dX = theta * (mu - X) dt + sigma dW

    Args:
        theta: Mean reversion speed
        mu: Long-run mean
        sigma: Volatility
        x0: Initial value
        T: Time horizon
        n_steps: Number of steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        Tuple of (time_grid, paths)
    """

    def drift(t, x):
        return theta * (mu - x)

    def diffusion(t, x):
        return sigma * np.ones_like(x)

    return euler_maruyama(drift, diffusion, x0, T, n_steps, n_paths, seed)
