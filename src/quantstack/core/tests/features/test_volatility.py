# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for volatility indicators: WilliamsVIXFix.

Tests verify correctness properties (no lookahead, boundary behavior,
vol-expansion detection) following the plan's testing specification.
"""

import numpy as np
import pandas as pd

from quantstack.core.features.volatility import WilliamsVIXFix


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
        """WVF = (highest_close - low) / highest_close * 100; should be >= 0."""
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        valid = result["wvf"].dropna()
        assert valid.min() >= -0.01

    def test_wvf_extreme_binary(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        vals = result["wvf_extreme"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_extreme_fires_on_selloff(self):
        """Sharp selloff should produce wvf_extreme=1."""
        n = 80
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = np.concatenate(
            [
                np.full(40, 150.0),
                np.linspace(150, 100, 40),
            ]
        )
        cl = pd.Series(prices, index=dates)
        hi = cl + 1.0
        lo = cl - 1.0
        result = WilliamsVIXFix(lookback=22, bb_period=20).compute(hi, lo, cl)
        assert result["wvf_extreme"].sum() > 0

    def test_bb_upper_above_lower(self):
        hi, lo, cl = _ohlcv()
        result = WilliamsVIXFix().compute(hi, lo, cl)
        valid = result.dropna(subset=["wvf_bb_upper", "wvf_bb_lower"])
        if len(valid) > 0:
            assert (valid["wvf_bb_upper"] >= valid["wvf_bb_lower"]).all()

    def test_no_lookahead(self):
        hi, lo, cl = _ohlcv(n=80)
        r1 = WilliamsVIXFix(lookback=10, bb_period=10).compute(hi, lo, cl)
        hi2 = pd.concat(
            [hi, pd.Series([hi.iloc[-1]], index=[hi.index[-1] + pd.Timedelta("1D")])]
        )
        lo2 = pd.concat(
            [
                lo,
                pd.Series(
                    [lo.iloc[-1] * 0.9], index=[lo.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        cl2 = pd.concat(
            [
                cl,
                pd.Series(
                    [cl.iloc[-1] * 0.9], index=[cl.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        r2 = WilliamsVIXFix(lookback=10, bb_period=10).compute(hi2, lo2, cl2)
        pd.testing.assert_series_equal(
            r1["wvf"].dropna(),
            r2["wvf"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = WilliamsVIXFix().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_constant_price(self):
        """Flat price -> wvf should be near 0."""
        n = 60
        hi = pd.Series(np.full(n, 100.5))
        lo = pd.Series(np.full(n, 99.5))
        cl = pd.Series(np.full(n, 100.0))
        result = WilliamsVIXFix().compute(hi, lo, cl)
        valid = result["wvf"].dropna()
        if len(valid) > 0:
            assert valid.mean() < 5.0
