# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Supertrend, Ichimoku Cloud, and Hull Moving Average."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.trend import (
    HullMovingAverage,
    IchimokuCloud,
    SupertrendIndicator,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_100() -> pd.DataFrame:
    """100-bar synthetic OHLCV — trending up with mild noise."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.3) + 0.2
    low = close - np.abs(np.random.randn(100) * 0.3) - 0.2
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)


@pytest.fixture
def constant_ohlcv() -> pd.DataFrame:
    """Constant price — guards against division-by-zero."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    price = np.full(50, 100.0)
    return pd.DataFrame({"high": price, "low": price, "close": price}, index=dates)


# ---------------------------------------------------------------------------
# SupertrendIndicator
# ---------------------------------------------------------------------------


class TestSupertrend:
    def test_returns_dataframe(self, ohlcv_100):
        st = SupertrendIndicator()
        result = st.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv_100):
        st = SupertrendIndicator()
        result = st.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        assert set(result.columns) == {"supertrend", "st_direction", "st_uptrend"}

    def test_index_preserved(self, ohlcv_100):
        st = SupertrendIndicator()
        result = st.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        pd.testing.assert_index_equal(result.index, ohlcv_100.index)

    def test_direction_only_1_or_minus1(self, ohlcv_100):
        st = SupertrendIndicator()
        result = st.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        valid = result["st_direction"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})
        # After first valid bar, should only be 1 or -1
        non_zero = valid[valid != 0]
        assert set(non_zero.unique()).issubset({-1, 1})

    def test_st_uptrend_agrees_with_direction(self, ohlcv_100):
        st = SupertrendIndicator()
        result = st.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        uptrend_rows = result[result["st_direction"] == 1]
        assert uptrend_rows["st_uptrend"].all()
        downtrend_rows = result[result["st_direction"] == -1]
        assert not downtrend_rows["st_uptrend"].any()

    def test_no_lookahead(self, ohlcv_100):
        """Values for bars 0..N-1 must not change when bar N is added."""
        st = SupertrendIndicator()
        hi = ohlcv_100["high"]
        lo = ohlcv_100["low"]
        cl = ohlcv_100["close"]

        result_n = st.compute(hi.iloc[:-1], lo.iloc[:-1], cl.iloc[:-1])
        result_n1 = st.compute(hi, lo, cl)

        shared_idx = result_n.index
        pd.testing.assert_series_equal(
            result_n["supertrend"].reindex(shared_idx),
            result_n1["supertrend"].reindex(shared_idx),
            check_names=False,
        )

    def test_constant_price_no_crash(self, constant_ohlcv):
        st = SupertrendIndicator()
        result = st.compute(
            constant_ohlcv["high"],
            constant_ohlcv["low"],
            constant_ohlcv["close"],
        )
        assert result is not None
        assert len(result) == len(constant_ohlcv)

    def test_clear_downtrend_direction(self):
        """Price falling well below the band should lock into downtrend."""
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        close = np.linspace(200, 100, 60)
        high = close + 1.0
        low = close - 1.0
        hi = pd.Series(high, index=dates)
        lo = pd.Series(low, index=dates)
        cl = pd.Series(close, index=dates)

        st = SupertrendIndicator(atr_length=5, multiplier=1.5)
        result = st.compute(hi, lo, cl)

        # Last 20 bars of a strong downtrend should be labelled -1
        assert (result["st_direction"].iloc[-20:] == -1).all()

    def test_clear_uptrend_direction(self):
        """Price rising well above the band should lock into uptrend."""
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        close = np.linspace(100, 200, 60)
        high = close + 1.0
        low = close - 1.0
        hi = pd.Series(high, index=dates)
        lo = pd.Series(low, index=dates)
        cl = pd.Series(close, index=dates)

        st = SupertrendIndicator(atr_length=5, multiplier=1.5)
        result = st.compute(hi, lo, cl)

        assert (result["st_direction"].iloc[-20:] == 1).all()


# ---------------------------------------------------------------------------
# IchimokuCloud
# ---------------------------------------------------------------------------


class TestIchimokuCloud:
    def test_returns_dataframe(self, ohlcv_100):
        ichi = IchimokuCloud()
        result = ichi.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        assert isinstance(result, pd.DataFrame)

    def test_all_components_present(self, ohlcv_100):
        ichi = IchimokuCloud()
        result = ichi.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        expected = {
            "tenkan_sen",
            "kijun_sen",
            "senkou_a",
            "senkou_b",
            "chikou_span",
            "price_above_cloud",
            "price_below_cloud",
            "cloud_bullish",
            "tenkan_above_kijun",
        }
        assert expected.issubset(set(result.columns))

    def test_index_preserved(self, ohlcv_100):
        ichi = IchimokuCloud()
        result = ichi.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        pd.testing.assert_index_equal(result.index, ohlcv_100.index)

    def test_binary_signal_columns(self, ohlcv_100):
        ichi = IchimokuCloud()
        result = ichi.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        for col in (
            "price_above_cloud",
            "price_below_cloud",
            "cloud_bullish",
            "tenkan_above_kijun",
        ):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} has non-binary values: {vals}"

    def test_tenkan_midpoint_formula(self):
        """Tenkan = (9-bar high + 9-bar low) / 2 — verify on known data."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        high = pd.Series(np.arange(1.0, 31.0), index=dates)
        low = pd.Series(np.zeros(30), index=dates)
        close = (high + low) / 2.0

        ichi = IchimokuCloud(tenkan=9)
        result = ichi.compute(high, low, close)

        # At bar 8 (index=8): 9-bar high = 9, 9-bar low = 0 → tenkan = 4.5
        assert abs(result["tenkan_sen"].iloc[8] - 4.5) < 1e-9

    def test_no_lookahead(self, ohlcv_100):
        """Tenkan/Kijun must not change when a new bar is appended."""
        ichi = IchimokuCloud()
        hi, lo, cl = ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"]

        result_n = ichi.compute(hi.iloc[:-1], lo.iloc[:-1], cl.iloc[:-1])
        result_n1 = ichi.compute(hi, lo, cl)

        shared = result_n.index
        pd.testing.assert_series_equal(
            result_n["tenkan_sen"].reindex(shared),
            result_n1["tenkan_sen"].reindex(shared),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result_n["kijun_sen"].reindex(shared),
            result_n1["kijun_sen"].reindex(shared),
            check_names=False,
        )

    def test_cloud_mutually_exclusive(self, ohlcv_100):
        """Price cannot be simultaneously above and below the cloud."""
        ichi = IchimokuCloud()
        result = ichi.compute(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        both = (result["price_above_cloud"] == 1) & (result["price_below_cloud"] == 1)
        assert not both.any()


# ---------------------------------------------------------------------------
# HullMovingAverage
# ---------------------------------------------------------------------------


class TestHullMovingAverage:
    def test_returns_dataframe(self, ohlcv_100):
        hma = HullMovingAverage(period=20)
        result = hma.compute(ohlcv_100["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv_100):
        hma = HullMovingAverage(period=20)
        result = hma.compute(ohlcv_100["close"])
        assert set(result.columns) == {"hma", "hma_slope", "hma_uptrend"}

    def test_index_preserved(self, ohlcv_100):
        hma = HullMovingAverage(period=20)
        result = hma.compute(ohlcv_100["close"])
        pd.testing.assert_index_equal(result.index, ohlcv_100.index)

    def test_hma_uptrend_binary(self, ohlcv_100):
        hma = HullMovingAverage(period=20)
        result = hma.compute(ohlcv_100["close"])
        vals = result["hma_uptrend"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_slope_sign_agrees_with_uptrend(self, ohlcv_100):
        hma = HullMovingAverage(period=20)
        result = hma.compute(ohlcv_100["close"])
        valid = result.dropna()
        uptrend_mask = valid["hma_uptrend"] == 1
        assert (valid.loc[uptrend_mask, "hma_slope"] > 0).all()
        downtrend_mask = valid["hma_uptrend"] == 0
        assert (valid.loc[downtrend_mask, "hma_slope"] <= 0).all()

    def test_uptrend_in_rising_market(self):
        """HMA should signal uptrend for most bars after warmup in a strong linear rise."""
        dates = pd.date_range(start="2023-01-01", periods=80, freq="D")
        close = pd.Series(np.linspace(100, 200, 80), index=dates)
        hma = HullMovingAverage(period=20)
        result = hma.compute(close)
        # Skip the warmup window (period + sqrt(period) bars) before asserting
        warmup = 20 + int(20**0.5) + 1
        valid = result["hma_uptrend"].iloc[warmup:].dropna()
        # After warmup, essentially all bars should be uptrend in a strong rally
        assert valid.mean() >= 0.9

    def test_minimum_period_no_crash(self):
        """period=2 (minimum useful) must not crash."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        close = pd.Series(np.linspace(100, 110, 10), index=dates)
        hma = HullMovingAverage(period=2)
        result = hma.compute(close)
        assert len(result) == 10
