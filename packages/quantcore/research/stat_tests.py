"""
Statistical Tests for Signal Validation.

Provides rigorous hypothesis testing for trading signals:
- ADF stationarity tests
- Lagged cross-correlation analysis
- Regime-switching tests
- Harvey-Liu multiple test correction
- Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from loguru import logger


@dataclass
class TestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    critical_values: Optional[Dict[str, float]] = None
    additional_info: Optional[Dict] = None


def adf_test(
    series: pd.Series,
    max_lags: Optional[int] = None,
    regression: str = "c",
    significance_level: float = 0.05,
) -> TestResult:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test
        max_lags: Maximum number of lags to include
        regression: Type of regression ('c' = constant, 'ct' = constant + trend, 'n' = none)
        significance_level: Significance level for hypothesis test

    Returns:
        TestResult with ADF statistic, p-value, and critical values
    """
    from statsmodels.tsa.stattools import adfuller

    series = series.dropna()

    if len(series) < 20:
        logger.warning("ADF test requires at least 20 observations")
        return TestResult(
            test_name="ADF",
            statistic=np.nan,
            p_value=1.0,
            is_significant=False,
            additional_info={"error": "insufficient_data"},
        )

    result = adfuller(series, maxlag=max_lags, regression=regression, autolag="AIC")

    adf_stat, p_value, used_lag, nobs, critical_values, icbest = result

    return TestResult(
        test_name="ADF",
        statistic=adf_stat,
        p_value=p_value,
        is_significant=p_value < significance_level,
        critical_values=critical_values,
        additional_info={
            "used_lag": used_lag,
            "nobs": nobs,
            "ic_best": icbest,
            "regression": regression,
            "interpretation": (
                "stationary" if p_value < significance_level else "non-stationary"
            ),
        },
    )


def lagged_cross_correlation(
    signal: pd.Series,
    returns: pd.Series,
    max_lag: int = 20,
    min_lag: int = 1,
) -> Dict[int, float]:
    """
    Compute lagged cross-correlation between signal and future returns.

    This is the core IC (Information Coefficient) analysis.

    Args:
        signal: Predictive signal series
        returns: Return series
        max_lag: Maximum forward lag to test
        min_lag: Minimum forward lag (typically 1 to avoid lookahead)

    Returns:
        Dictionary mapping lag -> correlation coefficient
    """
    correlations = {}

    # Align series
    common_idx = signal.index.intersection(returns.index)
    signal = signal.loc[common_idx]
    returns = returns.loc[common_idx]

    for lag in range(min_lag, max_lag + 1):
        # Signal at t predicts returns at t+lag
        lagged_returns = returns.shift(-lag)

        # Remove NaNs
        valid = ~(signal.isna() | lagged_returns.isna())

        if valid.sum() < 30:
            correlations[lag] = np.nan
            continue

        corr, _ = stats.spearmanr(signal[valid], lagged_returns[valid])
        correlations[lag] = corr

    return correlations


def regime_switching_test(
    series: pd.Series,
    n_regimes: int = 2,
    significance_level: float = 0.05,
) -> TestResult:
    """
    Test for regime switching behavior using Markov Switching model.

    Tests whether a series exhibits distinct regimes (e.g., bull/bear markets).

    Args:
        series: Time series to test
        n_regimes: Number of regimes to test for
        significance_level: Significance level

    Returns:
        TestResult with likelihood ratio test statistic
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        logger.warning("statsmodels MarkovRegression not available")
        return TestResult(
            test_name="RegimeSwitching",
            statistic=np.nan,
            p_value=1.0,
            is_significant=False,
            additional_info={"error": "module_not_available"},
        )

    series = series.dropna()

    if len(series) < 100:
        return TestResult(
            test_name="RegimeSwitching",
            statistic=np.nan,
            p_value=1.0,
            is_significant=False,
            additional_info={"error": "insufficient_data"},
        )

    try:
        # Fit single-regime model (null hypothesis)
        model_1 = MarkovRegression(series, k_regimes=1, trend="c")
        result_1 = model_1.fit(disp=False)

        # Fit multi-regime model (alternative)
        model_k = MarkovRegression(series, k_regimes=n_regimes, trend="c")
        result_k = model_k.fit(disp=False)

        # Likelihood ratio test
        lr_stat = 2 * (result_k.llf - result_1.llf)
        df = (n_regimes - 1) * 3  # Parameters difference
        p_value = 1 - stats.chi2.cdf(lr_stat, df)

        return TestResult(
            test_name="RegimeSwitching",
            statistic=lr_stat,
            p_value=p_value,
            is_significant=p_value < significance_level,
            additional_info={
                "n_regimes": n_regimes,
                "llf_1regime": result_1.llf,
                "llf_kregime": result_k.llf,
                "aic_1regime": result_1.aic,
                "aic_kregime": result_k.aic,
                "interpretation": (
                    f"{n_regimes} regimes detected"
                    if p_value < significance_level
                    else "single regime"
                ),
            },
        )
    except Exception as e:
        logger.warning(f"Regime switching test failed: {e}")
        return TestResult(
            test_name="RegimeSwitching",
            statistic=np.nan,
            p_value=1.0,
            is_significant=False,
            additional_info={"error": str(e)},
        )


def harvey_liu_correction(
    p_values: List[float],
    num_tests: int,
    significance_level: float = 0.05,
) -> Dict[str, any]:
    """
    Harvey-Liu-Zhu (2016) multiple testing correction for trading strategies.

    Adjusts for data snooping bias when testing multiple signals/strategies.
    More appropriate than Bonferroni for correlated tests.

    Reference: "... and the Cross-Section of Expected Returns"

    Args:
        p_values: List of p-values from individual tests
        num_tests: Total number of tests conducted (including unreported)
        significance_level: Nominal significance level

    Returns:
        Dictionary with adjusted significance threshold and results
    """
    p_values = np.array(p_values)
    n = len(p_values)

    # Harvey-Liu adjustment factor
    # Accounts for multiple testing with expected correlation
    adjustment_factor = np.sqrt(np.log(num_tests))

    # Adjusted critical t-statistic (approximately)
    # From HLZ paper: t-stat threshold ≈ sqrt(2 * log(M))
    critical_t = np.sqrt(2 * np.log(num_tests))

    # Convert to adjusted p-value threshold
    adjusted_alpha = 2 * (1 - stats.norm.cdf(critical_t))

    # Which tests pass the adjusted threshold
    significant_original = p_values < significance_level
    significant_adjusted = p_values < adjusted_alpha

    # Benjamini-Hochberg for comparison
    sorted_p = np.sort(p_values)
    bh_thresholds = significance_level * np.arange(1, n + 1) / n
    bh_significant = p_values < bh_thresholds[np.searchsorted(sorted_p, p_values)]

    return {
        "original_alpha": significance_level,
        "adjusted_alpha": adjusted_alpha,
        "critical_t_stat": critical_t,
        "num_tests": num_tests,
        "num_significant_original": significant_original.sum(),
        "num_significant_adjusted": significant_adjusted.sum(),
        "num_significant_bh": bh_significant.sum(),
        "significant_mask_original": significant_original,
        "significant_mask_adjusted": significant_adjusted,
        "interpretation": f"Of {n} tests, {significant_adjusted.sum()} survive HLZ correction (vs {significant_original.sum()} nominal)",
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = "percentile",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        data: Input data array
        statistic_func: Function that computes the statistic (e.g., np.mean, sharpe_ratio)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: 'percentile', 'basic', or 'bca' (bias-corrected accelerated)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    data = np.asarray(data)
    n = len(data)

    # Point estimate
    point_estimate = statistic_func(data)

    # Bootstrap resampling
    np.random.seed(42)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)

    alpha = 1 - confidence_level

    if method == "percentile":
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == "basic":
        lower = 2 * point_estimate - np.percentile(
            bootstrap_stats, 100 * (1 - alpha / 2)
        )
        upper = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * alpha / 2)

    elif method == "bca":
        # Bias-corrected and accelerated
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))

        # Acceleration (jackknife estimate)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic_func(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / (den + 1e-10)

        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        p_high = stats.norm.cdf(
            z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high))
        )

        lower = np.percentile(bootstrap_stats, 100 * p_low)
        upper = np.percentile(bootstrap_stats, 100 * p_high)

    else:
        raise ValueError(f"Unknown method: {method}")

    return point_estimate, lower, upper


def information_coefficient_test(
    signal: pd.Series,
    returns: pd.Series,
    lag: int = 1,
    n_bootstrap: int = 5000,
) -> TestResult:
    """
    Test if Information Coefficient (IC) is significantly different from zero.

    IC = Spearman correlation between signal and forward returns.

    Args:
        signal: Predictive signal
        returns: Return series
        lag: Forward lag for returns
        n_bootstrap: Bootstrap samples for CI

    Returns:
        TestResult with IC estimate and confidence interval
    """
    # Align and lag
    common_idx = signal.index.intersection(returns.index)
    signal = signal.loc[common_idx]
    returns = returns.loc[common_idx]

    lagged_returns = returns.shift(-lag)
    valid = ~(signal.isna() | lagged_returns.isna())

    signal_valid = signal[valid].values
    returns_valid = lagged_returns[valid].values

    # Compute IC
    ic, p_value = stats.spearmanr(signal_valid, returns_valid)

    # Bootstrap CI
    def ic_func(paired_data):
        return stats.spearmanr(paired_data[:, 0], paired_data[:, 1])[0]

    paired = np.column_stack([signal_valid, returns_valid])
    _, ic_lower, ic_upper = bootstrap_confidence_interval(
        paired, ic_func, n_bootstrap=n_bootstrap, method="percentile"
    )

    # Significant if CI doesn't include 0
    is_significant = (ic_lower > 0) or (ic_upper < 0)

    return TestResult(
        test_name="InformationCoefficient",
        statistic=ic,
        p_value=p_value,
        is_significant=is_significant,
        additional_info={
            "ci_lower": ic_lower,
            "ci_upper": ic_upper,
            "lag": lag,
            "n_observations": valid.sum(),
            "interpretation": f"IC = {ic:.4f} [{ic_lower:.4f}, {ic_upper:.4f}]",
        },
    )


def run_signal_validation_suite(
    signal: pd.Series,
    returns: pd.Series,
    price: pd.Series,
    num_total_tests: int = 100,
) -> Dict[str, TestResult]:
    """
    Run full suite of statistical tests on a trading signal.

    Args:
        signal: Predictive signal series
        returns: Return series
        price: Price series (for stationarity tests)
        num_total_tests: Total tests for HLZ correction

    Returns:
        Dictionary of test name -> TestResult
    """
    results = {}

    # 1. Signal stationarity
    results["signal_stationarity"] = adf_test(signal)

    # 2. Price stationarity (should be non-stationary)
    results["price_stationarity"] = adf_test(price)

    # 3. Returns stationarity (should be stationary)
    results["returns_stationarity"] = adf_test(returns)

    # 4. Information Coefficient
    results["ic_test"] = information_coefficient_test(signal, returns, lag=1)

    # 5. Lagged correlations
    lag_corrs = lagged_cross_correlation(signal, returns, max_lag=10)
    max_ic_lag = max(
        lag_corrs, key=lambda k: abs(lag_corrs[k]) if not np.isnan(lag_corrs[k]) else 0
    )
    results["lagged_correlations"] = TestResult(
        test_name="LaggedCorrelations",
        statistic=lag_corrs.get(1, np.nan),
        p_value=np.nan,
        is_significant=abs(lag_corrs.get(1, 0)) > 0.02,
        additional_info={
            "all_correlations": lag_corrs,
            "max_ic_lag": max_ic_lag,
            "max_ic": lag_corrs[max_ic_lag],
        },
    )

    # 6. Regime switching
    results["regime_switching"] = regime_switching_test(returns)

    logger.info(f"Signal validation suite complete: {len(results)} tests run")

    return results
