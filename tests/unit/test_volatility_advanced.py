# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Williams VIX Fix indicator."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.volatility import WilliamsVIXFix


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_ohlcv():
    """100-bar OHLCV — gentle uptrend with noise."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.3) + 0.2
    low = close - np.abs(np.random.randn(100) * 0.5) - 0.2
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)


@pytest.fixture
def crash_ohlcv():
    """Simulates a sharp price crash — WVF should spike."""
    dates = pd.date_range(start="2023-01-01", periods=80, freq="D")
    # Slow run-up then sudden crash
    close_up = np.linspace(100, 150, 50)
    close_down = np.linspace(150, 80, 30)
    close = np.concatenate([close_up, close_down])
    high = close + 1.0
    # During crash, lows extend further
    low = np.concatenate([close_up - 1.0, close_down - 5.0])
    return pd.DataFrame(
        {
            "high": pd.Series(high, index=dates),
            "low": pd.Series(low, index=dates),
            "close": pd.Series(close, index=dates),
        }
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestWilliamsVIXFixStructure:
    def test_returns_dataframe(self, trending_ohlcv):
        wvf = WilliamsVIXFix()
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, trending_ohlcv):
        wvf = WilliamsVIXFix()
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        assert set(result.columns) == {
            "wvf",
            "wvf_bb_upper",
            "wvf_bb_lower",
            "wvf_extreme",
        }

    def test_index_preserved(self, trending_ohlcv):
        wvf = WilliamsVIXFix()
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        pd.testing.assert_index_equal(result.index, trending_ohlcv.index)

    def test_wvf_non_negative(self, trending_ohlcv):
        """WVF measures gap from recent high to current low — always >= 0."""
        wvf = WilliamsVIXFix()
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        valid = result["wvf"].dropna()
        assert (valid >= 0).all()

    def test_wvf_extreme_binary(self, trending_ohlcv):
        wvf = WilliamsVIXFix()
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        vals = result["wvf_extreme"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_bb_upper_above_lower(self, trending_ohlcv):
        wvf = WilliamsVIXFix()
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        valid = result.dropna()
        assert (valid["wvf_bb_upper"] >= valid["wvf_bb_lower"]).all()


# ---------------------------------------------------------------------------
# Functional / directional tests
# ---------------------------------------------------------------------------


class TestWilliamsVIXFixFunctionality:
    def test_wvf_formula_known_value(self):
        """
        Hand-verify formula on 5-bar constant price then 1 crash bar.
        rolling_max(close, 5) = 100; low[5] = 60
        wvf[5] = (100 - 60) / 100 * 100 = 40.0
        """
        dates = pd.date_range(start="2023-01-01", periods=6, freq="D")
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 90.0], index=dates)
        high = close + 0.5
        low = pd.Series([99.0, 99.0, 99.0, 99.0, 99.0, 60.0], index=dates)

        wvf = WilliamsVIXFix(lookback=5, bb_period=3, bb_dev=2.0)
        result = wvf.compute(high, low, close)

        # At bar index 4 (5th bar, lookback=5): highest_close=100, low=99 → wvf=(100-99)/100*100=1.0
        assert abs(result["wvf"].iloc[4] - 1.0) < 1e-9
        # At bar index 5 (6th bar): highest_close=100, low=60 → wvf=(100-60)/100*100=40.0
        assert abs(result["wvf"].iloc[5] - 40.0) < 1e-9

    def test_wvf_spikes_during_crash(self, crash_ohlcv):
        """WVF should be materially higher during the crash period than the uptrend."""
        wvf = WilliamsVIXFix(lookback=22, bb_period=20, bb_dev=2.0)
        result = wvf.compute(
            crash_ohlcv["high"],
            crash_ohlcv["low"],
            crash_ohlcv["close"],
        )
        uptrend_wvf = result["wvf"].iloc[22:50].mean()  # skip warmup
        crash_wvf = result["wvf"].iloc[60:].mean()
        assert crash_wvf > uptrend_wvf

    def test_wvf_extreme_fires_at_crash_bottom(self, crash_ohlcv):
        """The extreme signal should fire during or near the crash."""
        wvf = WilliamsVIXFix(lookback=22, bb_period=20, bb_dev=2.0)
        result = wvf.compute(
            crash_ohlcv["high"],
            crash_ohlcv["low"],
            crash_ohlcv["close"],
        )
        # At least one extreme signal during the crash window
        assert result["wvf_extreme"].iloc[50:].sum() > 0

    def test_wvf_near_zero_in_stable_uptrend(self):
        """Steady uptrend: current low never far from recent close high → WVF stays small."""
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        close = pd.Series(np.linspace(100, 110, 60), index=dates)
        high = close + 0.1
        low = close - 0.1

        wvf = WilliamsVIXFix(lookback=22)
        result = wvf.compute(high, low, close)
        valid = result["wvf"].dropna()
        # In a smooth linear uptrend, WVF should be very small
        assert valid.mean() < 5.0


# ---------------------------------------------------------------------------
# No-lookahead test
# ---------------------------------------------------------------------------


class TestWilliamsVIXFixNoLookahead:
    def test_no_lookahead(self, trending_ohlcv):
        wvf = WilliamsVIXFix()
        hi = trending_ohlcv["high"]
        lo = trending_ohlcv["low"]
        cl = trending_ohlcv["close"]

        result_n = wvf.compute(hi.iloc[:-1], lo.iloc[:-1], cl.iloc[:-1])
        result_n1 = wvf.compute(hi, lo, cl)

        shared = result_n.index
        pd.testing.assert_series_equal(
            result_n["wvf"].reindex(shared),
            result_n1["wvf"].reindex(shared),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWilliamsVIXFixEdgeCases:
    def test_constant_price_no_crash(self):
        """Constant price: low = close = highest close → wvf = 0 after warmup."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        price = pd.Series(np.full(50, 100.0), index=dates)
        wvf = WilliamsVIXFix(lookback=10)
        result = wvf.compute(price, price, price)
        valid = result["wvf"].dropna()
        assert (valid == 0.0).all()

    def test_low_below_zero_no_crash(self):
        """Negative prices (e.g., futures) should not crash."""
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        close = pd.Series(np.linspace(-50, -10, 60), index=dates)
        high = close + 1.0
        low = close - 1.0
        wvf = WilliamsVIXFix(lookback=10)
        result = wvf.compute(high, low, close)
        assert len(result) == 60

    def test_custom_parameters(self, trending_ohlcv):
        """Verify custom lookback/bb_period/bb_dev don't crash."""
        wvf = WilliamsVIXFix(lookback=10, bb_period=10, bb_dev=1.5)
        result = wvf.compute(
            trending_ohlcv["high"],
            trending_ohlcv["low"],
            trending_ohlcv["close"],
        )
        assert result is not None
        assert len(result) == len(trending_ohlcv)
