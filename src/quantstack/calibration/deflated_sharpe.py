"""Deflated Sharpe Ratio (DSR) implementation.

Reference: Bailey, D.H. and Lopez de Prado, M. (2014)
'The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting, and the Volatility of Volatility'.
"""

from __future__ import annotations

import math

from scipy import stats


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_strategies_tested: int,
    num_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that observed Sharpe is genuine after adjusting for multiple testing.

    Args:
        observed_sharpe: The observed (sample) Sharpe ratio.
        num_strategies_tested: Total strategies tested (including this one).
        num_returns: Number of return observations.
        skewness: Skewness of the return distribution (0 for normal).
        kurtosis: Kurtosis of the return distribution (3 for normal).

    Returns:
        Probability [0, 1] that the Sharpe ratio is genuine (not due to luck).
        Higher is better.
    """
    if num_strategies_tested < 1:
        return 0.0
    if num_returns < 2:
        return 0.0

    # Expected maximum Sharpe under null (all strategies have zero true Sharpe)
    # E[max(Z_1, ..., Z_N)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    euler_mascheroni = 0.5772156649
    n = max(num_strategies_tested, 1)

    # Approximation of expected max of N standard normals
    if n == 1:
        expected_max_sharpe = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n)
        z2_arg = 1.0 - 1.0 / (n * math.e)
        z2 = stats.norm.ppf(min(z2_arg, 0.9999))
        expected_max_sharpe = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2

    # Variance of the Sharpe ratio estimator (accounting for non-normality)
    # Var[SR] ≈ (1/T) * (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2)
    sr = observed_sharpe
    var_sr = (
        1.0
        + 0.5 * sr**2
        - skewness * sr
        + (kurtosis - 3.0) / 4.0 * sr**2
    ) / num_returns

    if var_sr <= 0:
        return 0.0

    std_sr = math.sqrt(var_sr)

    # PSR (Probabilistic Sharpe Ratio) — probability that true Sharpe > expected max
    if std_sr == 0:
        return 0.0

    z_score = (observed_sharpe - expected_max_sharpe) / std_sr
    return float(stats.norm.cdf(z_score))
