# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for trend indicators: Supertrend, Ichimoku Cloud, HullMA.

These tests verify correctness properties (no lookahead, direction logic,
known-value constraints) following the plan's testing specification.
"""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.trend import (
    HullMovingAverage,
    IchimokuCloud,
    SupertrendIndicator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(n: int = 100, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Synthetic (high, low, close) bars."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.8), index=dates).clip(50)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    return high, low, close


# ---------------------------------------------------------------------------
# SupertrendIndicator
# ---------------------------------------------------------------------------


class TestSupertrend:
    def test_returns_dataframe(self):
        hi, lo, cl = _ohlcv()
        result = SupertrendIndicator().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl = _ohlcv()
        result = SupertrendIndicator().compute(hi, lo, cl)
        for col in ("supertrend", "st_direction", "st_uptrend"):
            assert col in result.columns

    def test_direction_is_plus_minus_one(self):
        hi, lo, cl = _ohlcv()
        result = SupertrendIndicator().compute(hi, lo, cl)
        dirs = result["st_direction"].dropna().unique()
        # 0 is valid during warmup before the first ATR is established
        assert set(dirs).issubset({1, -1, 0})

    def test_no_lookahead(self):
        """Values for bars 0..N-1 must be identical when a new bar N is added."""
        hi, lo, cl = _ohlcv(n=80)
        r1 = SupertrendIndicator().compute(hi, lo, cl)
        # Add one new bar
        extra = pd.concat(
            [
                hi,
                pd.Series(
                    [hi.iloc[-1] * 1.01], index=[hi.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        extra_lo = pd.concat(
            [lo, pd.Series([lo.iloc[-1]], index=[lo.index[-1] + pd.Timedelta("1D")])]
        )
        extra_cl = pd.concat(
            [
                cl,
                pd.Series(
                    [cl.iloc[-1] * 1.01], index=[cl.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        r2 = SupertrendIndicator().compute(extra, extra_lo, extra_cl)
        # All prior values must be unchanged
        pd.testing.assert_series_equal(
            r1["supertrend"].dropna(),
            r2["supertrend"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = SupertrendIndicator().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_constant_price_no_crash(self):
        n = 50
        hi = pd.Series(np.full(n, 100.5))
        lo = pd.Series(np.full(n, 99.5))
        cl = pd.Series(np.full(n, 100.0))
        result = SupertrendIndicator().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_trend_flip_on_crossing(self):
        """Build a series that clearly crosses the band; direction must flip."""
        n = 60
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        # Downtrend for first half, then strong uptrend
        prices = np.concatenate(
            [
                np.linspace(120, 90, 30),
                np.linspace(91, 140, 30),
            ]
        )
        cl = pd.Series(prices, index=dates)
        hi = cl + 1.0
        lo = cl - 1.0
        result = SupertrendIndicator(atr_length=5, multiplier=2.0).compute(hi, lo, cl)
        dirs = result["st_direction"].dropna()
        # Should have both +1 and -1 in the series
        assert 1 in dirs.values and -1 in dirs.values


# ---------------------------------------------------------------------------
# IchimokuCloud
# ---------------------------------------------------------------------------


class TestIchimokuCloud:
    def test_returns_dataframe(self):
        hi, lo, cl = _ohlcv(n=120)
        result = IchimokuCloud().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_five_components_present(self):
        hi, lo, cl = _ohlcv(n=120)
        result = IchimokuCloud().compute(hi, lo, cl)
        for col in ("tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou_span"):
            assert col in result.columns, f"{col} missing"

    def test_cloud_signal_columns(self):
        hi, lo, cl = _ohlcv(n=120)
        result = IchimokuCloud().compute(hi, lo, cl)
        for col in (
            "cloud_bullish",
            "price_above_cloud",
            "price_below_cloud",
            "tenkan_above_kijun",
        ):
            assert col in result.columns

    def test_cloud_bullish_binary(self):
        hi, lo, cl = _ohlcv(n=120)
        result = IchimokuCloud().compute(hi, lo, cl)
        vals = result["cloud_bullish"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_tenkan_kijun_span_correct_period(self):
        """Tenkan should be the 9-period midline; verify against manual calc."""
        n = 30
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        hi = pd.Series(np.arange(n, dtype=float) + 1.5, index=dates)
        lo = pd.Series(np.arange(n, dtype=float) - 0.5, index=dates)
        cl = pd.Series(np.arange(n, dtype=float) + 0.5, index=dates)
        result = IchimokuCloud(tenkan=9).compute(hi, lo, cl)
        # At index 8 (9th bar): tenkan = (max(hi[0:9]) + min(lo[0:9])) / 2
        expected_tenkan = (hi.iloc[:9].max() + lo.iloc[:9].min()) / 2
        assert abs(result["tenkan_sen"].iloc[8] - expected_tenkan) < 1e-6

    def test_no_crash_insufficient_bars(self):
        """Short series (< displacement period) should not raise."""
        hi, lo, cl = _ohlcv(n=10)
        result = IchimokuCloud().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# HullMovingAverage
# ---------------------------------------------------------------------------


class TestHullMovingAverage:
    def test_returns_dataframe(self):
        _, _, cl = _ohlcv()
        result = HullMovingAverage(period=20).compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, _, cl = _ohlcv()
        result = HullMovingAverage(period=20).compute(cl)
        for col in ("hma", "hma_slope", "hma_uptrend"):
            assert col in result.columns

    def test_hma_uptrend_binary(self):
        _, _, cl = _ohlcv()
        result = HullMovingAverage(period=20).compute(cl)
        vals = result["hma_uptrend"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_hma_uptrend_in_uptrend(self):
        """Strong monotone uptrend → hma_uptrend should be 1 after warmup."""
        n = 60
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 160, n), index=dates)
        result = HullMovingAverage(period=10).compute(cl)
        assert result["hma_uptrend"].iloc[20:].mean() == 1.0

    def test_hma_lag_less_than_sma(self):
        """HMA should track price more tightly than SMA of same period."""
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 150, n), index=dates)
        result = HullMovingAverage(period=20).compute(cl)
        sma = cl.rolling(20).mean()
        # At the last bar, HMA should be closer to actual close than SMA
        hma_err = abs(result["hma"].iloc[-1] - cl.iloc[-1])
        sma_err = abs(sma.iloc[-1] - cl.iloc[-1])
        assert hma_err < sma_err

    def test_no_lookahead(self):
        _, _, cl = _ohlcv(n=60)
        r1 = HullMovingAverage(period=10).compute(cl)
        cl2 = pd.concat(
            [
                cl,
                pd.Series(
                    [cl.iloc[-1] * 1.02], index=[cl.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        r2 = HullMovingAverage(period=10).compute(cl2)
        pd.testing.assert_series_equal(
            r1["hma"].dropna(),
            r2["hma"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )
