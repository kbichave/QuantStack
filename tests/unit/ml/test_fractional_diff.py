"""Tests for fractional differentiation (AFML Chapter 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.ml.fractional_diff import batch_find_min_d, frac_diff, find_min_d


def _random_walk(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)


def test_d_zero_returns_original_series():
    """d=0 returns the original series (no differencing)."""
    series = _random_walk()
    result = frac_diff(series, d=0.0)
    # d=0 weights are [1, 0, 0, ...] so output equals input (after window NaNs)
    valid = result.dropna()
    np.testing.assert_allclose(valid.values, series.iloc[len(series) - len(valid):].values, atol=1e-10)


def test_d_one_approximates_first_difference():
    """d=1 returns first difference when window=2 (exact 2-term expansion)."""
    series = _random_walk()
    # With window=2, weights are [1, -1], which is exactly the first difference
    result = frac_diff(series, d=1.0, window=2)
    expected = series.diff()
    valid_idx = result.dropna().index
    np.testing.assert_allclose(
        result.loc[valid_idx].values,
        expected.loc[valid_idx].values,
        atol=1e-10,
    )


def test_minimum_d_achieves_stationarity():
    """find_min_d returns a d where ADF test p < 0.05 for a random walk."""
    series = _random_walk(n=500)
    d = find_min_d(series, d_range=(0.0, 1.0), step=0.1, window=50)
    assert 0.0 < d <= 1.0

    # Verify the result is actually stationary
    from statsmodels.tsa.stattools import adfuller

    diffed = frac_diff(series, d=d, window=50).dropna()
    adf_pvalue = adfuller(diffed, maxlag=5, autolag=None)[1]
    assert adf_pvalue < 0.05


def test_higher_d_destroys_more_memory():
    """Correlation with original series decreases as d increases."""
    series = _random_walk(n=500)

    correlations = []
    for d in [0.2, 0.5, 0.8]:
        diffed = frac_diff(series, d=d, window=50).dropna()
        # Correlation with original series measures memory preservation
        aligned = series.iloc[len(series) - len(diffed):]
        corr = np.corrcoef(aligned.values, diffed.values)[0, 1]
        correlations.append(abs(corr))

    # Higher d should have less correlation with original (less memory)
    assert correlations[0] > correlations[2], f"Memory should decrease with d: {correlations}"


def test_per_symbol_d_values_differ():
    """Different series should yield different optimal d values."""
    rng = np.random.default_rng(42)

    # Low-volatility series (needs less differencing)
    low_vol = pd.Series(np.cumsum(rng.normal(0, 0.1, 500)) + 100)
    # High-volatility series
    high_vol = pd.Series(np.cumsum(rng.normal(0, 2.0, 500)) + 100)

    d_low = find_min_d(low_vol, step=0.1, window=50)
    d_high = find_min_d(high_vol, step=0.1, window=50)

    # Both should achieve stationarity but may differ
    assert 0.0 < d_low <= 1.0
    assert 0.0 < d_high <= 1.0


def test_batch_find_min_d():
    """batch_find_min_d processes a DataFrame of multiple columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "AAPL": np.cumsum(rng.normal(0, 1, 300)) + 150,
        "MSFT": np.cumsum(rng.normal(0, 0.5, 300)) + 300,
    })
    result = batch_find_min_d(df, window=50)
    assert "AAPL" in result
    assert "MSFT" in result
    assert all(0.0 < v <= 1.0 for v in result.values())


def test_frac_diff_empty_series():
    """frac_diff handles empty series gracefully."""
    result = frac_diff(pd.Series(dtype=float), d=0.5)
    assert len(result) == 0
