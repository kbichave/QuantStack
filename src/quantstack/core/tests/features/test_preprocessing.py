# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fractional differentiation preprocessing."""

import numpy as np
import pandas as pd

from quantstack.core.features.preprocessing import FractionalDifferentiator


def _random_walk(n: int = 500, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)


def _stationary(n: int = 500, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.random.randn(n), index=dates)


class TestFractionalDifferentiator:
    def test_transform_returns_series(self):
        series = _random_walk()
        result = FractionalDifferentiator().transform(series, d=0.5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)

    def test_d_zero_identity(self):
        """d=0 should approximate the original series."""
        series = _random_walk()
        result = FractionalDifferentiator().transform(series, d=0.0)
        valid = result.dropna()
        # w_0=1, no other weights → should equal original
        pd.testing.assert_series_equal(
            valid, series.loc[valid.index], check_names=False, rtol=1e-10
        )

    def test_d_one_approximates_diff(self):
        """d=1 should approximate first differences."""
        series = _random_walk()
        result = FractionalDifferentiator().transform(series, d=1.0)
        first_diff = series.diff()
        # Compare at valid indices (skip NaN warmup)
        valid_idx = result.dropna().index.intersection(first_diff.dropna().index)
        if len(valid_idx) > 50:
            corr = result.loc[valid_idx].corr(first_diff.loc[valid_idx])
            assert corr > 0.95, f"d=1 should correlate with diff, got {corr}"

    def test_find_min_d_stationary_returns_zero(self):
        """Already-stationary series should return d≈0."""
        series = _stationary()
        d = FractionalDifferentiator().find_min_d(series)
        assert d == 0.0

    def test_find_min_d_random_walk_positive(self):
        """Random walk should need d > 0 for stationarity."""
        series = _random_walk()
        d = FractionalDifferentiator().find_min_d(series)
        assert d > 0.0

    def test_weights_monotonically_decreasing(self):
        fd = FractionalDifferentiator()
        weights = fd._get_weights(0.5, 100)
        abs_weights = np.abs(weights)
        # After first weight, magnitudes should decrease
        for i in range(2, len(abs_weights)):
            assert abs_weights[i] <= abs_weights[i - 1] + 1e-10

    def test_transform_df(self):
        series = _random_walk()
        df = pd.DataFrame({"close": series, "volume": series * 10})
        result = FractionalDifferentiator().transform_df(
            df, ["close"], d_map={"close": 0.5}
        )
        assert "close" in result.columns
        assert "volume" in result.columns
        # Volume should be unchanged
        pd.testing.assert_series_equal(result["volume"], df["volume"])

    def test_short_series_no_crash(self):
        series = pd.Series(
            [100.0, 101.0, 99.0], index=pd.date_range("2020-01-01", periods=3)
        )
        result = FractionalDifferentiator().transform(series, d=0.5)
        assert isinstance(result, pd.Series)

    def test_find_min_d_short_series(self):
        series = pd.Series([100.0] * 5, index=pd.date_range("2020-01-01", periods=5))
        d = FractionalDifferentiator().find_min_d(series)
        assert isinstance(d, float)
