# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SupertrendIndicator, IchimokuCloud, and HullMovingAverage.

Tests verify: expected columns, value-range invariants, direction logic,
no-lookahead property, and edge cases (short series, constant prices).
"""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.trend import (
    SupertrendIndicator,
    IchimokuCloud,
    HullMovingAverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(n: int = 200, seed: int = 7, start: float = 100.0) -> tuple:
    """Return (high, low, close) pd.Series of length n."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = start + np.cumsum(np.random.randn(n) * 0.8)
    high = pd.Series(close + np.abs(np.random.randn(n) * 0.4) + 0.3, index=dates)
    low = pd.Series(close - np.abs(np.random.randn(n) * 0.4) - 0.3, index=dates)
    close = pd.Series(close, index=dates)
    return high, low, close


def _trending_up(n: int = 100) -> tuple:
    """Return strongly upward-trending (high, low, close) series."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100.0 + np.arange(n) * 0.5, index=dates)
    high = close + 0.5
    low = close - 0.5
    return high, low, close


def _trending_down(n: int = 100) -> tuple:
    """Return strongly downward-trending (high, low, close) series."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100.0 - np.arange(n) * 0.5, index=dates)
    high = close + 0.5
    low = close - 0.5
    return high, low, close


# ---------------------------------------------------------------------------
# SupertrendIndicator
# ---------------------------------------------------------------------------


class TestSupertrendIndicator:
    def test_returns_dataframe(self):
        high, low, close = _make_series()
        result = SupertrendIndicator().compute(high, low, close)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        high, low, close = _make_series()
        result = SupertrendIndicator().compute(high, low, close)
        assert "supertrend" in result.columns
        assert "st_direction" in result.columns
        assert "st_uptrend" in result.columns

    def test_same_length(self):
        high, low, close = _make_series(150)
        result = SupertrendIndicator().compute(high, low, close)
        assert len(result) == 150

    def test_direction_values_are_1_or_minus1_or_0(self):
        high, low, close = _make_series()
        result = SupertrendIndicator().compute(high, low, close)
        valid = result["st_direction"].isin([-1, 0, 1])
        assert valid.all()

    def test_direction_is_1_in_strong_uptrend(self):
        high, low, close = _trending_up(100)
        result = SupertrendIndicator(atr_length=5, multiplier=1.0).compute(
            high, low, close
        )
        # After warm-up, should be in an uptrend
        assert result["st_direction"].iloc[-1] == 1

    def test_direction_is_minus1_in_strong_downtrend(self):
        high, low, close = _trending_down(100)
        result = SupertrendIndicator(atr_length=5, multiplier=1.0).compute(
            high, low, close
        )
        assert result["st_direction"].iloc[-1] == -1

    def test_no_lookahead(self):
        """Extending the series by one bar must not change prior values."""
        high, low, close = _make_series(100)
        result_n = SupertrendIndicator().compute(high, low, close)

        high2 = pd.concat(
            [
                high,
                pd.Series(
                    [high.iloc[-1] + 0.5], index=[high.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        low2 = pd.concat(
            [
                low,
                pd.Series(
                    [low.iloc[-1] - 0.5], index=[high.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        close2 = pd.concat(
            [
                close,
                pd.Series(
                    [close.iloc[-1] + 0.2], index=[high.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        result_n1 = SupertrendIndicator().compute(high2, low2, close2)

        np.testing.assert_allclose(
            result_n["supertrend"].values,
            result_n1["supertrend"].iloc[:-1].values,
            equal_nan=True,
        )

    def test_constant_price_no_crash(self):
        n = 50
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        close = pd.Series(100.0, index=dates)
        high = pd.Series(100.5, index=dates)
        low = pd.Series(99.5, index=dates)
        result = SupertrendIndicator().compute(high, low, close)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n

    def test_single_bar_no_crash(self):
        dates = pd.date_range("2022-01-01", periods=1, freq="D")
        close = pd.Series([100.0], index=dates)
        high = pd.Series([101.0], index=dates)
        low = pd.Series([99.0], index=dates)
        result = SupertrendIndicator().compute(high, low, close)
        assert len(result) == 1

    def test_uptrend_indicator_matches_direction(self):
        high, low, close = _make_series()
        result = SupertrendIndicator().compute(high, low, close)
        expected = result["st_direction"] == 1
        pd.testing.assert_series_equal(
            result["st_uptrend"], expected, check_names=False
        )


# ---------------------------------------------------------------------------
# IchimokuCloud
# ---------------------------------------------------------------------------


class TestIchimokuCloud:
    def test_returns_dataframe(self):
        high, low, close = _make_series()
        result = IchimokuCloud().compute(high, low, close)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        high, low, close = _make_series()
        result = IchimokuCloud().compute(high, low, close)
        for col in (
            "tenkan_sen",
            "kijun_sen",
            "senkou_a",
            "senkou_b",
            "chikou_span",
            "price_above_cloud",
            "price_below_cloud",
            "cloud_bullish",
            "tenkan_above_kijun",
        ):
            assert col in result.columns, f"missing: {col}"

    def test_same_length(self):
        high, low, close = _make_series(150)
        result = IchimokuCloud().compute(high, low, close)
        assert len(result) == 150

    def test_tenkan_sen_valid_after_warmup(self):
        """Tenkan-sen should be non-NaN for bars >= tenkan period."""
        high, low, close = _make_series(100)
        result = IchimokuCloud(tenkan=9).compute(high, low, close)
        assert result["tenkan_sen"].iloc[9:].notna().all()

    def test_kijun_sen_valid_after_warmup(self):
        high, low, close = _make_series(100)
        result = IchimokuCloud(kijun=26).compute(high, low, close)
        assert result["kijun_sen"].iloc[26:].notna().all()

    def test_binary_columns_in_0_1(self):
        high, low, close = _make_series()
        result = IchimokuCloud().compute(high, low, close)
        for col in (
            "price_above_cloud",
            "price_below_cloud",
            "cloud_bullish",
            "tenkan_above_kijun",
        ):
            valid = result[col].dropna().isin([0, 1])
            assert valid.all(), f"{col} has values outside {{0, 1}}"

    def test_not_simultaneously_above_and_below_cloud(self):
        high, low, close = _make_series()
        result = IchimokuCloud().compute(high, low, close)
        both = (result["price_above_cloud"] == 1) & (result["price_below_cloud"] == 1)
        assert not both.any()

    def test_tenkan_within_high_low_range(self):
        high, low, close = _make_series(100)
        result = IchimokuCloud(tenkan=9).compute(high, low, close)
        valid = result["tenkan_sen"].iloc[9:]
        assert (valid >= low.rolling(9).min().iloc[9:]).all()
        assert (valid <= high.rolling(9).max().iloc[9:]).all()

    def test_constant_price_no_crash(self):
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        close = pd.Series(100.0, index=dates)
        high = pd.Series(101.0, index=dates)
        low = pd.Series(99.0, index=dates)
        result = IchimokuCloud().compute(high, low, close)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# HullMovingAverage
# ---------------------------------------------------------------------------


class TestHullMovingAverage:
    def test_returns_dataframe(self):
        _, _, close = _make_series()
        result = HullMovingAverage().compute(close)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        _, _, close = _make_series()
        result = HullMovingAverage().compute(close)
        assert "hma" in result.columns
        assert "hma_slope" in result.columns
        assert "hma_uptrend" in result.columns

    def test_same_length(self):
        _, _, close = _make_series(150)
        result = HullMovingAverage(period=20).compute(close)
        assert len(result) == 150

    def test_hma_uptrend_binary(self):
        _, _, close = _make_series()
        result = HullMovingAverage().compute(close)
        valid = result["hma_uptrend"].dropna().isin([0, 1])
        assert valid.all()

    def test_hma_uptrend_true_in_strong_uptrend(self):
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        close = pd.Series(100.0 + np.arange(100) * 1.0, index=dates)
        result = HullMovingAverage(period=10).compute(close)
        assert result["hma_uptrend"].iloc[-1] == 1

    def test_hma_uptrend_false_in_strong_downtrend(self):
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        close = pd.Series(200.0 - np.arange(100) * 1.0, index=dates)
        result = HullMovingAverage(period=10).compute(close)
        assert result["hma_uptrend"].iloc[-1] == 0

    def test_hma_no_lookahead(self):
        """Adding a bar must not change prior HMA values."""
        _, _, close = _make_series(80)
        result_n = HullMovingAverage(period=10).compute(close)

        extra_date = close.index[-1] + pd.Timedelta("1D")
        close_ext = pd.concat(
            [close, pd.Series([close.iloc[-1] + 0.5], index=[extra_date])]
        )
        result_n1 = HullMovingAverage(period=10).compute(close_ext)

        vals_n = result_n["hma"].dropna().values
        vals_n1 = result_n1["hma"].iloc[:-1].dropna().values
        np.testing.assert_allclose(vals_n, vals_n1, equal_nan=True)

    def test_constant_price_no_crash(self):
        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        close = pd.Series(100.0, index=dates)
        result = HullMovingAverage(period=20).compute(close)
        assert isinstance(result, pd.DataFrame)

    def test_slope_positive_in_uptrend(self):
        dates = pd.date_range("2022-01-01", periods=80, freq="D")
        close = pd.Series(100.0 + np.arange(80) * 1.0, index=dates)
        result = HullMovingAverage(period=10).compute(close)
        # Slope should be positive for the last bar
        assert result["hma_slope"].iloc[-1] > 0
