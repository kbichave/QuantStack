"""
Backtest overfitting detection.

Implements:
- Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014)
  Adjusts for non-normality and multiple-testing inflation.
- Probability of Backtest Overfitting (PBO) — Bailey et al. (2015)
  Derived from Combinatorial Purged Cross-Validation splits.
  Returns a scalar in [0, 1]; values above 0.5 indicate likely overfitting.

References:
  Bailey, D.H., Borwein, J., López de Prado, M., Zhu, Q.J. (2014).
  "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting and Non-Normality." Journal of Portfolio Management.

  Bailey, D.H., and López de Prado, M. (2015).
  "The Sharpe Ratio Efficient Frontier." Journal of Risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


@dataclass
class DSRResult:
    """Result from Deflated Sharpe Ratio calculation."""

    observed_sharpe: float
    benchmark_sharpe: float  # SR* — expected max SR under null
    dsr: float  # Probability that observed SR > SR*
    is_genuine: bool  # True when DSR >= significance_level
    n_trials: int
    skewness: float
    kurtosis: float  # Excess kurtosis


def benchmark_sharpe_ratio(
    n_trials: int,
    n_obs: int,
    sr_std: float = 1.0,
) -> float:
    """
    Compute the expected maximum Sharpe Ratio under the null hypothesis
    that all n_trials strategies are noise (SR* from Equation 10, Bailey 2014).

    The expected maximum order statistic of n_trials independent standard
    normals approximates the benchmark:
        SR* ≈ (1 - γ) × z(1 - 1/n) + γ × z(1 - 1/(n × e))
    where γ is the Euler–Mascheroni constant.

    Args:
        n_trials: Number of strategy variants tried (parameter sets, signals, etc.)
        n_obs: Number of observations in the return series.
        sr_std: Standard deviation of the SR distribution (default 1.0).

    Returns:
        SR* — benchmark Sharpe Ratio to test against.
    """
    if n_trials <= 1:
        return 0.0

    euler_mascheroni = 0.5772156649

    # Expected maximum of n standard normals
    e_max = (1.0 - euler_mascheroni) * norm.ppf(
        1.0 - 1.0 / n_trials
    ) + euler_mascheroni * norm.ppf(1.0 - 1.0 / (n_trials * np.e))

    # Scale by annualised SR standard deviation
    # Var(SR) ≈ 1/T × (1 + 0.5×SR²) for large T — use sr_std directly
    return e_max * sr_std


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    sr_std: float = 1.0,
    significance_level: float = 0.95,
) -> DSRResult:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    The DSR is the probability that the observed Sharpe Ratio is
    greater than the benchmark SR*, after correcting for:
      - Multiple testing across n_trials strategy variants
      - Non-normality of returns (skewness and excess kurtosis)
      - Small-sample bias

    A DSR below significance_level (typically 0.95) indicates the
    strategy is likely a product of overfitting / data mining.

    Args:
        observed_sharpe: Annualised in-sample Sharpe Ratio.
        n_trials: Number of strategy variants tried before selecting this one.
        n_obs: Number of daily return observations.
        skewness: Skewness of the daily return series (0 = normal).
        excess_kurtosis: Excess kurtosis of daily returns (0 = normal).
        sr_std: Standard deviation of Sharpe Ratios across the n_trials.
        significance_level: Threshold for declaring the SR genuine (default 0.95).

    Returns:
        DSRResult with deflated probability and supporting stats.
    """
    if n_obs < 2:
        logger.warning("DSR requires at least 2 observations")
        return DSRResult(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=0.0,
            dsr=0.0,
            is_genuine=False,
            n_trials=n_trials,
            skewness=skewness,
            kurtosis=excess_kurtosis,
        )

    sr_star = benchmark_sharpe_ratio(n_trials, n_obs, sr_std)

    # Non-normality adjustment: variance of the SR estimator
    # V(SR̂) ≈ (1/T) × (1 + 0.5×SR² - γ₃×SR + (γ₄/4)×(SR²-1)²)
    # where γ₃ = skewness, γ₄ = excess kurtosis  (simplified form)
    sr_var = (1.0 / n_obs) * (
        1.0
        + 0.5 * observed_sharpe**2
        - skewness * observed_sharpe
        + (excess_kurtosis / 4.0) * (observed_sharpe**2 - 1.0) ** 2
    )
    sr_var = max(sr_var, 1e-9)  # Guard against numerical zero

    # DSR = Φ((SR_obs - SR*) / √V(SR̂))
    z_score = (observed_sharpe - sr_star) / np.sqrt(sr_var)
    dsr = float(norm.cdf(z_score))

    return DSRResult(
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=sr_star,
        dsr=dsr,
        is_genuine=dsr >= significance_level,
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=excess_kurtosis,
    )


def returns_statistics(returns: pd.Series) -> tuple[float, float, float, float]:
    """
    Compute annualised Sharpe, skewness, excess kurtosis, and SR std
    from a daily return series.

    Args:
        returns: Daily return series (fraction, not percentage).

    Returns:
        (sharpe_annual, skewness, excess_kurtosis, sr_std)
    """
    clean = returns.dropna()
    if len(clean) < 10:
        return 0.0, 0.0, 0.0, 1.0

    mean_r = clean.mean()
    std_r = clean.std(ddof=1)

    if std_r < 1e-10:
        return 0.0, 0.0, 0.0, 1.0

    sharpe = mean_r / std_r * np.sqrt(252)
    skew = float(clean.skew())
    kurt = float(clean.kurtosis())  # pandas returns excess kurtosis

    # Estimated std of SR across trials — use 1.0 as default when unknown
    # Calibrate from SR distribution if multiple strategies are available
    sr_std = 1.0

    return sharpe, skew, kurt, sr_std


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (PBO)
# ---------------------------------------------------------------------------


@dataclass
class PBOResult:
    """Result from Probability of Backtest Overfitting calculation."""

    pbo: float  # Probability of overfitting in [0, 1]
    is_overfit: bool  # True when pbo > 0.5
    n_paths: int  # Number of CPCV paths evaluated
    oos_sharpes: list[float]  # Out-of-sample Sharpe per path
    is_sharpes: list[float]  # In-sample Sharpe per path (for best strategy each path)
    logit_values: list[float]  # Logit(ω) per path (for diagnostic)
    pbo_curve: pd.Series  # CDF of logit values (for plotting)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> PBOResult:
    """
    Compute the Probability of Backtest Overfitting (PBO).

    Uses Combinatorial Purged Cross-Validation to generate multiple
    train/test splits, then computes the fraction of paths where the
    in-sample best strategy ranks below median out-of-sample.

    Args:
        returns_matrix: (T × N) array of daily returns for N strategies/parameters.
                        Each column is a different strategy variant.
        n_splits: Number of CPCV groups (default 6).
        n_test_splits: Number of groups to use as test (default 2 of 6).
        embargo_pct: Embargo fraction between train and test groups.

    Returns:
        PBOResult with PBO scalar and supporting diagnostics.
    """
    T, N = returns_matrix.shape
    if T < 20 or N < 2:
        logger.warning(
            f"PBO requires at least 20 time steps and 2 strategies; got {T}×{N}"
        )
        return PBOResult(
            pbo=0.5,
            is_overfit=False,
            n_paths=0,
            oos_sharpes=[],
            is_sharpes=[],
            logit_values=[],
            pbo_curve=pd.Series(dtype=float),
        )

    group_size = T // n_splits
    embargo_size = max(1, int(T * embargo_pct))

    # Build group index ranges
    groups = []
    for i in range(n_splits):
        start = i * group_size
        end = (i + 1) * group_size if i < n_splits - 1 else T
        groups.append((start, end))

    logit_values: list[float] = []
    oos_sharpes: list[float] = []
    is_sharpes: list[float] = []

    for test_group_ids in combinations(range(n_splits), n_test_splits):
        test_idx = []
        purge_idx: set = set()

        for g in test_group_ids:
            s, e = groups[g]
            test_idx.extend(range(s, e))
            purge_idx.update(range(e, min(e + embargo_size, T)))
            purge_idx.update(range(max(0, s - embargo_size), s))

        all_test = set(test_idx)
        train_idx = sorted(
            i for i in range(T) if i not in all_test and i not in purge_idx
        )
        test_idx_sorted = sorted(test_idx)

        if len(train_idx) < 10 or len(test_idx_sorted) < 5:
            continue

        r_train = returns_matrix[train_idx, :]
        r_test = returns_matrix[test_idx_sorted, :]

        # IS Sharpe for each strategy
        is_sr = _col_sharpes(r_train)
        # OOS Sharpe for each strategy
        oos_sr = _col_sharpes(r_test)

        # Best IS strategy
        best_is_idx = int(np.argmax(is_sr))
        best_oos_sr = oos_sr[best_is_idx]

        # Rank of the IS-best strategy in OOS distribution
        oos_rank = float(np.sum(oos_sr < best_oos_sr)) / max(N - 1, 1)
        # ω = rank of best_oos_sr relative to median of OOS SRs
        float(np.median(oos_sr))

        # Logit(ω) where ω is the relative rank vs median (0 = at median, <0 = below)
        oos_rank - 0.5
        logit = float(np.log(max(1e-9, oos_rank) / max(1e-9, 1.0 - oos_rank)))

        logit_values.append(logit)
        oos_sharpes.append(best_oos_sr)
        is_sharpes.append(float(is_sr[best_is_idx]))

    if not logit_values:
        return PBOResult(
            pbo=0.5,
            is_overfit=False,
            n_paths=0,
            oos_sharpes=[],
            is_sharpes=[],
            logit_values=[],
            pbo_curve=pd.Series(dtype=float),
        )

    # PBO = fraction of paths where IS-best ranks below median OOS
    pbo = float(np.mean([lv < 0 for lv in logit_values]))

    # Build empirical CDF of logit values for plotting
    sorted_logits = sorted(logit_values)
    n = len(sorted_logits)
    pbo_curve = pd.Series(
        [(i + 1) / n for i in range(n)],
        index=sorted_logits,
        name="pbo_cdf",
    )

    return PBOResult(
        pbo=pbo,
        is_overfit=pbo > 0.5,
        n_paths=len(logit_values),
        oos_sharpes=oos_sharpes,
        is_sharpes=is_sharpes,
        logit_values=logit_values,
        pbo_curve=pbo_curve,
    )


def _col_sharpes(returns: np.ndarray, ann_factor: float = 252.0) -> np.ndarray:
    """Annualised Sharpe Ratio for each column in a (T × N) returns array."""
    means = returns.mean(axis=0)
    stds = returns.std(axis=0, ddof=1)
    stds = np.where(stds < 1e-10, 1e-10, stds)
    return means / stds * np.sqrt(ann_factor)


# ---------------------------------------------------------------------------
# Convenience: full overfitting report
# ---------------------------------------------------------------------------


@dataclass
class OverfittingReport:
    """Combined DSR + PBO overfitting analysis."""

    dsr_result: DSRResult
    pbo_result: PBOResult | None
    verdict: str  # "GENUINE" | "SUSPECT" | "OVERFIT"
    summary: str


def run_overfitting_analysis(
    strategy_returns: pd.Series,
    n_trials: int,
    all_strategy_returns: np.ndarray | None = None,
    n_cpcv_splits: int = 6,
    significance_level: float = 0.95,
) -> OverfittingReport:
    """
    Run the full overfitting battery: DSR + optional PBO.

    Args:
        strategy_returns: Daily returns of the selected strategy.
        n_trials: Total number of strategy variants evaluated before selecting
                  this one (parameter sweeps, indicator choices, etc.).
        all_strategy_returns: (T × N) matrix of returns for all N variants.
                               Required for PBO calculation. If None, skips PBO.
        n_cpcv_splits: CPCV groups for PBO.
        significance_level: DSR confidence threshold.

    Returns:
        OverfittingReport with DSR, optional PBO, and a plain-text verdict.
    """
    sharpe, skew, kurt, sr_std = returns_statistics(strategy_returns)
    n_obs = len(strategy_returns.dropna())

    dsr = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=n_trials,
        n_obs=n_obs,
        skewness=skew,
        excess_kurtosis=kurt,
        sr_std=sr_std,
        significance_level=significance_level,
    )

    pbo: PBOResult | None = None
    if all_strategy_returns is not None:
        pbo = probability_of_backtest_overfitting(
            all_strategy_returns,
            n_splits=n_cpcv_splits,
        )

    # Verdict logic
    if not dsr.is_genuine:
        verdict = "OVERFIT"
    elif pbo is not None and pbo.is_overfit:
        verdict = "SUSPECT"
    else:
        verdict = "GENUINE"

    lines = [
        "=== Overfitting Analysis ===",
        f"Observed Sharpe  : {dsr.observed_sharpe:.3f}",
        f"Benchmark SR*    : {dsr.benchmark_sharpe:.3f}  (n_trials={n_trials})",
        f"Deflated SR (DSR): {dsr.dsr:.3f}  {'✓ GENUINE' if dsr.is_genuine else '✗ FAIL'}",
        f"Returns skewness : {dsr.skewness:.3f}",
        f"Excess kurtosis  : {dsr.kurtosis:.3f}",
    ]
    if pbo is not None:
        lines += [
            f"PBO              : {pbo.pbo:.3f}  (n_paths={pbo.n_paths})"
            f"  {'✓ NOT OVERFIT' if not pbo.is_overfit else '✗ OVERFIT'}",
        ]
    lines += [
        "",
        f"VERDICT: {verdict}",
    ]

    return OverfittingReport(
        dsr_result=dsr,
        pbo_result=pbo,
        verdict=verdict,
        summary="\n".join(lines),
    )
