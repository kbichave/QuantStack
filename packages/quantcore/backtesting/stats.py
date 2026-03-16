# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Backtesting statistical rigor — what institutions require before deploying capital.

Functions:
  - sharpe_ratio_with_ci()     Sharpe ratio + 95% confidence interval
  - monte_carlo_permutation()  Null hypothesis significance test
  - walk_forward_summary()     Aggregate walk-forward fold statistics
  - degradation_report()       Out-of-sample performance decay tracker
  - min_sample_size_check()    Is N trades enough to claim edge?

Usage:
    from quantcore.backtesting.stats import (
        sharpe_ratio_with_ci,
        monte_carlo_permutation,
        min_sample_size_check,
    )

    sharpe, (lo, hi) = sharpe_ratio_with_ci(returns, confidence=0.95)
    p_value = monte_carlo_permutation(returns, observed_sharpe=sharpe)
    ok, msg = min_sample_size_check(n_trades=150, min_required=100)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# =============================================================================
# SHARPE RATIO WITH CONFIDENCE INTERVAL
# =============================================================================


def sharpe_ratio_with_ci(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute annualized Sharpe ratio with a confidence interval.

    Uses Lo (2002) asymptotic formula for the Sharpe ratio CI.
    A point estimate without a CI is not useful for institutional review.

    Args:
        returns: List of period returns (e.g. daily returns as fractions)
        risk_free_rate: Annual risk-free rate (e.g. 0.05 for 5%)
        periods_per_year: 252 for daily, 52 for weekly, 12 for monthly
        confidence: Confidence level (0.95 = 95% CI)

    Returns:
        (sharpe, (lower_bound, upper_bound))
    """
    r = np.array(returns, dtype=float)
    n = len(r)

    if n < 10:
        return 0.0, (0.0, 0.0)

    period_rf = risk_free_rate / periods_per_year
    excess = r - period_rf

    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    if std_excess == 0:
        return 0.0, (0.0, 0.0)

    # Annualized Sharpe
    sharpe = (mean_excess / std_excess) * math.sqrt(periods_per_year)

    # Lo (2002): asymptotic variance of the Sharpe ratio estimator
    # Var(SR) ≈ (1/n) * (1 + SR²/2) for IID returns
    # More precisely, accounts for autocorrelation via skewness & kurtosis
    skew = float(stats.skew(excess))
    kurt = float(stats.kurtosis(excess))  # excess kurtosis

    # Variance of the annualized Sharpe ratio
    sr_period = mean_excess / std_excess  # Period Sharpe (not annualized yet)
    var_sr = (1 / n) * (
        1 + (sr_period ** 2 / 2) - skew * sr_period + ((kurt + 2) / 4) * sr_period ** 2
    )
    se_sr_period = math.sqrt(max(0, var_sr))
    se_sr_annualized = se_sr_period * math.sqrt(periods_per_year)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower = sharpe - z * se_sr_annualized
    upper = sharpe + z * se_sr_annualized

    return round(sharpe, 4), (round(lower, 4), round(upper, 4))


# =============================================================================
# MONTE CARLO PERMUTATION TEST
# =============================================================================


def monte_carlo_permutation(
    returns: List[float],
    observed_sharpe: float,
    n_permutations: int = 1000,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    seed: int = 42,
) -> float:
    """
    Null hypothesis significance test for strategy Sharpe ratio.

    Shuffles returns N times and computes Sharpe on each permutation.
    Returns the p-value: fraction of permutations with Sharpe >= observed.

    A low p-value (< 0.05) means the strategy's edge is unlikely to be
    random luck. A high p-value means you cannot reject the null hypothesis
    that the strategy has no edge.

    Args:
        returns: Observed strategy returns
        observed_sharpe: The Sharpe ratio to test
        n_permutations: Number of random shuffles
        risk_free_rate: Annual risk-free rate
        periods_per_year: 252 for daily
        seed: Random seed for reproducibility

    Returns:
        p-value (0.0–1.0). Values < 0.05 suggest real edge.
    """
    rng = np.random.default_rng(seed)
    r = np.array(returns, dtype=float)
    period_rf = risk_free_rate / periods_per_year

    beat_count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(r)
        excess = shuffled - period_rf
        std = np.std(excess, ddof=1)
        if std == 0:
            continue
        perm_sharpe = (np.mean(excess) / std) * math.sqrt(periods_per_year)
        if perm_sharpe >= observed_sharpe:
            beat_count += 1

    return beat_count / n_permutations


# =============================================================================
# WALK-FORWARD SUMMARY
# =============================================================================


@dataclass
class WalkForwardFold:
    """Results of a single walk-forward fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    test_max_drawdown: float
    n_trades: int


@dataclass
class WalkForwardSummary:
    """Aggregate statistics across all walk-forward folds."""

    n_folds: int
    avg_test_sharpe: float
    median_test_sharpe: float
    pct_positive_folds: float       # Fraction of folds with positive test Sharpe
    avg_test_return: float
    avg_max_drawdown: float
    sharpe_degradation: float       # avg_train_sharpe - avg_test_sharpe
    is_statistically_significant: bool
    notes: List[str]


def walk_forward_summary(folds: List[WalkForwardFold]) -> WalkForwardSummary:
    """
    Aggregate walk-forward fold results into a summary.

    The key metric is `sharpe_degradation`: how much worse is the
    test Sharpe vs the training Sharpe. If test_sharpe ≈ train_sharpe,
    the strategy generalizes. If test << train, it's overfit.
    """
    if not folds:
        return WalkForwardSummary(
            n_folds=0,
            avg_test_sharpe=0.0,
            median_test_sharpe=0.0,
            pct_positive_folds=0.0,
            avg_test_return=0.0,
            avg_max_drawdown=0.0,
            sharpe_degradation=0.0,
            is_statistically_significant=False,
            notes=["No folds provided"],
        )

    test_sharpes = [f.test_sharpe for f in folds]
    train_sharpes = [f.train_sharpe for f in folds]
    test_returns = [f.test_return for f in folds]
    drawdowns = [f.test_max_drawdown for f in folds]

    avg_test = float(np.mean(test_sharpes))
    avg_train = float(np.mean(train_sharpes))
    median_test = float(np.median(test_sharpes))
    pct_positive = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
    avg_return = float(np.mean(test_returns))
    avg_dd = float(np.mean(drawdowns))
    degradation = avg_train - avg_test

    notes = []
    if degradation > 0.5:
        notes.append(
            f"High degradation ({degradation:.2f}): "
            "strategy likely overfit to training data"
        )
    if pct_positive < 0.6:
        notes.append(
            f"Only {pct_positive:.0%} of folds positive: strategy is inconsistent"
        )
    if avg_test < 0.5:
        notes.append(f"Avg test Sharpe {avg_test:.2f} < 0.5: weak edge")

    # Statistically significant: >60% positive folds AND test Sharpe > 0.3
    is_sig = pct_positive >= 0.6 and avg_test >= 0.3

    return WalkForwardSummary(
        n_folds=len(folds),
        avg_test_sharpe=round(avg_test, 4),
        median_test_sharpe=round(median_test, 4),
        pct_positive_folds=round(pct_positive, 4),
        avg_test_return=round(avg_return, 4),
        avg_max_drawdown=round(avg_dd, 4),
        sharpe_degradation=round(degradation, 4),
        is_statistically_significant=is_sig,
        notes=notes,
    )


# =============================================================================
# MINIMUM SAMPLE SIZE
# =============================================================================


def min_sample_size_check(
    n_trades: int, min_required: int = 100
) -> Tuple[bool, str]:
    """
    Check if we have enough trades to claim statistical edge.

    With < 100 trades, a 60% win rate could easily be luck (p > 0.05).
    With 100+ trades, a 55% win rate is statistically significant.

    Args:
        n_trades: Number of closed trades
        min_required: Minimum to trust the statistics

    Returns:
        (passes, message)
    """
    if n_trades >= min_required:
        return True, f"Sample size OK: {n_trades} trades >= minimum {min_required}"
    else:
        pct = n_trades / min_required * 100
        return (
            False,
            f"Insufficient sample: {n_trades} trades ({pct:.0f}% of "
            f"required {min_required}). Do not claim edge from this data.",
        )


# =============================================================================
# CALMAR RATIO
# =============================================================================


def calmar_ratio(
    returns: List[float],
    periods_per_year: int = 252,
) -> float:
    """
    Calmar ratio: annualized return / max drawdown.

    More informative than Sharpe for drawdown-sensitive strategies.
    """
    r = np.array(returns, dtype=float)
    if len(r) == 0:
        return 0.0

    annual_return = float(np.mean(r)) * periods_per_year

    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    drawdowns = (cum - peak) / peak
    max_dd = float(np.min(drawdowns))

    if max_dd == 0:
        return 0.0

    return round(annual_return / abs(max_dd), 4)


# =============================================================================
# DEGRADATION REPORT
# =============================================================================


def degradation_report(
    in_sample_metrics: Dict[str, float],
    out_of_sample_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare in-sample vs out-of-sample performance.

    Returns a dict with degradation pct for each metric and an overall verdict.
    """
    from typing import Any  # local import to avoid circular
    report: Dict[str, Any] = {}
    degraded_metrics = []

    for metric in ["sharpe", "win_rate", "profit_factor", "calmar"]:
        is_val = in_sample_metrics.get(metric)
        oos_val = out_of_sample_metrics.get(metric)
        if is_val is None or oos_val is None:
            continue

        if is_val != 0:
            degradation_pct = (is_val - oos_val) / abs(is_val) * 100
        else:
            degradation_pct = 0.0

        report[metric] = {
            "in_sample": round(is_val, 4),
            "out_of_sample": round(oos_val, 4),
            "degradation_pct": round(degradation_pct, 1),
        }

        if degradation_pct > 30:
            degraded_metrics.append(f"{metric} ({degradation_pct:.0f}% worse OOS)")

    if degraded_metrics:
        report["verdict"] = f"CAUTION: significant degradation in {', '.join(degraded_metrics)}"
    else:
        report["verdict"] = "OK: out-of-sample performance tracks in-sample"

    return report
