# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for volatility indicators: WilliamsVIXFix.
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.volatility import WilliamsVIXFix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(n: int = 100, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.8), index=dates).clip(50)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    return high, low, close


# ---------------------------------------------------------------------------
# WilliamsVIXFix
# ---------------------------------------------------------------------------


class TestWilliamsVIXFix:
    def test_returns_dataframe(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        for col in ("wvf", "wvf_bb_upper", "wvf_bb_lower", "wvf_extreme"):
            assert col in result.columns, f"{col} missing"

    def test_wvf_non_negative(self):
        """WVF = (rolling_max_close - low) / rolling_max_close × 100 — always >= 0."""
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        vals = result["wvf"].dropna()
        assert (vals >= 0).all()

    def test_wvf_extreme_binary(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        vals = result["wvf_extreme"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_bb_upper_above_lower(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        valid = result[["wvf_bb_upper", "wvf_bb_lower"]].dropna()
        assert (valid["wvf_bb_upper"] >= valid["wvf_bb_lower"]).all()

    def test_extreme_fires_on_crash(self):
        """Sharp market crash followed by new lows should fire wvf_extreme."""
        n = 80
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        # Stable prices then crash
        prices = np.concatenate([np.linspace(100, 100, 50), np.linspace(100, 50, 30)])
        cl = pd.Series(prices, index=dates)
        hi = cl + 1.0
        lo = cl - 2.0
        result = WilliamsVIXFix(lookback=22, bb_period=20, bb_dev=2.0).compute(hi, lo, cl)
        # Extreme should fire at least once after the crash begins
        assert result["wvf_extreme"].iloc[55:].sum() > 0

    def test_known_value(self):
        """
        Manual check: for a constant series low=90, close_rolling_max=100
        wvf[i] = (100 - 90) / 100 * 100 = 10.0
        """
        n = 50
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.full(n, 100.0), index=dates)
        lo = pd.Series(np.full(n, 90.0), index=dates)
        hi = pd.Series(np.full(n, 101.0), index=dates)
        result = WilliamsVIXFix(lookback=22).compute(hi, lo, cl)
        # After warmup (bar 22+), wvf should be exactly 10.0
        wvf_stable = result["wvf"].iloc[22:].dropna()
        assert len(wvf_stable) > 0
        assert (wvf_stable - 10.0).abs().max() < 1e-6

    def test_no_crash_short_series(self):
        hi = pd.Series([101.0, 100.5, 99.0])
        lo = pd.Series([99.0, 98.5, 97.0])
        cl = pd.Series([100.0, 99.5, 98.0])
        result = WilliamsVIXFix().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_no_lookahead(self):
        hi, lo, cl = _ohlcv(n=60)
        r1 = WilliamsVIXFix().compute(hi, lo, cl)
        hi2 = pd.concat([hi, pd.Series([hi.iloc[-1]], index=[hi.index[-1] + pd.Timedelta("1D")])])
        lo2 = pd.concat([lo, pd.Series([lo.iloc[-1] * 0.99], index=[lo.index[-1] + pd.Timedelta("1D")])])
        cl2 = pd.concat([cl, pd.Series([cl.iloc[-1] * 0.99], index=[cl.index[-1] + pd.Timedelta("1D")])])
        r2 = WilliamsVIXFix().compute(hi2, lo2, cl2)
        pd.testing.assert_series_equal(
            r1["wvf"].dropna(),
            r2["wvf"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )

    def test_custom_params_accepted(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix(lookback=14, bb_period=10, bb_dev=1.5).compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)
        assert "wvf" in result.columns
