"""
Block bootstrap Monte Carlo for Sharpe ratio confidence intervals.

Complements the operational Monte Carlo in ``monte_carlo.py`` (which
perturbs timing and slippage for WTI spread strategies) with a
statistical bootstrap that resamples daily returns to estimate the
sampling distribution of the Sharpe ratio.

Block bootstrap preserves serial autocorrelation in returns, giving
more realistic confidence intervals than iid resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class BootstrapResult:
    """Result of bootstrap Sharpe ratio estimation."""

    mean_sharpe: float
    std_sharpe: float
    ci_5: float
    ci_25: float
    ci_75: float
    ci_95: float
    prob_negative: float
    n_simulations: int
    block_size: int
    n_returns: int


def _compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a return array."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _block_resample(
    returns: np.ndarray, n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a single block-bootstrap resample of length *n*."""
    blocks_needed = int(np.ceil(n / block_size))
    max_start = len(returns) - block_size
    if max_start <= 0:
        # Fall back to iid if series is shorter than block
        return rng.choice(returns, size=n, replace=True)
    starts = rng.integers(0, max_start + 1, size=blocks_needed)
    resampled = np.concatenate(
        [returns[s : s + block_size] for s in starts]
    )
    return resampled[:n]


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    n_simulations: int = 1000,
    block_size: int | None = None,
    periods_per_year: int = 252,
    seed: int = 42,
) -> BootstrapResult:
    """Estimate the sampling distribution of the Sharpe ratio via block bootstrap.

    Args:
        returns: Daily return series (simple returns, not log).
        n_simulations: Number of bootstrap replications.
        block_size: Block length for the bootstrap. Defaults to
            ``max(5, len(returns) // 20)`` — roughly the square root of
            sample size, a common heuristic.
        periods_per_year: Annualisation factor (252 for daily).
        seed: Random seed for reproducibility.

    Returns:
        :class:`BootstrapResult` with Sharpe CI and tail probabilities.

    Raises:
        ValueError: If fewer than 30 return observations are provided.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 30:
        raise ValueError(
            f"bootstrap_sharpe_ci requires >= 30 returns, got {len(arr)}"
        )

    if block_size is None:
        block_size = max(5, len(arr) // 20)

    rng = np.random.default_rng(seed)
    sharpe_dist = np.empty(n_simulations)

    for i in range(n_simulations):
        resampled = _block_resample(arr, len(arr), block_size, rng)
        sharpe_dist[i] = _compute_sharpe(resampled, periods_per_year)

    result = BootstrapResult(
        mean_sharpe=float(np.mean(sharpe_dist)),
        std_sharpe=float(np.std(sharpe_dist, ddof=1)),
        ci_5=float(np.percentile(sharpe_dist, 5)),
        ci_25=float(np.percentile(sharpe_dist, 25)),
        ci_75=float(np.percentile(sharpe_dist, 75)),
        ci_95=float(np.percentile(sharpe_dist, 95)),
        prob_negative=float(np.mean(sharpe_dist < 0)),
        n_simulations=n_simulations,
        block_size=block_size,
        n_returns=len(arr),
    )

    logger.debug(
        f"Bootstrap MC: Sharpe mean={result.mean_sharpe:.3f} "
        f"CI=[{result.ci_5:.3f}, {result.ci_95:.3f}] "
        f"P(negative)={result.prob_negative:.2%}"
    )
    return result
