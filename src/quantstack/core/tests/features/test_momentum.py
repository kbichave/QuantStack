# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for momentum indicators: PercentRExhaustion, LaguerreRSI.

Tests verify correctness properties (no lookahead, boundary behavior,
known-value constraints) following the plan's testing specification.
"""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.momentum import LaguerreRSI, PercentRExhaustion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(n: int = 200, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Synthetic (high, low, close) bars."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.8), index=dates).clip(50)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    return high, low, close


# ---------------------------------------------------------------------------
# PercentRExhaustion
# ---------------------------------------------------------------------------


class TestPercentRExhaustion:
    def test_returns_dataframe(self):
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        for col in ("pct_r_short", "pct_r_long", "exhaustion_top", "exhaustion_bottom"):
            assert col in result.columns, f"{col} missing"

    def test_pct_r_range(self):
        """Williams %R is bounded in [-100, 0]."""
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        for col in ("pct_r_short", "pct_r_long"):
            valid = result[col].dropna()
            assert valid.min() >= -100.0, f"{col} below -100"
            assert valid.max() <= 0.0, f"{col} above 0"

    def test_exhaustion_binary(self):
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        for col in ("exhaustion_top", "exhaustion_bottom"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_exhaustion_top_at_highs(self):
        """Strong rally should trigger exhaustion_top at least once."""
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = np.linspace(100, 250, n)
        cl = pd.Series(prices, index=dates)
        hi = cl + 0.5
        lo = cl - 0.5
        result = PercentRExhaustion(short=14, long=112).compute(hi, lo, cl)
        assert result["exhaustion_top"].sum() > 0

    def test_exhaustion_bottom_at_lows(self):
        """Strong selloff should trigger exhaustion_bottom at least once."""
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        prices = np.linspace(250, 80, n)
        cl = pd.Series(prices, index=dates)
        hi = cl + 0.5
        lo = cl - 0.5
        result = PercentRExhaustion(short=14, long=112).compute(hi, lo, cl)
        assert result["exhaustion_bottom"].sum() > 0

    def test_no_lookahead(self):
        hi, lo, cl = _ohlcv(n=150)
        r1 = PercentRExhaustion(short=14, long=50).compute(hi, lo, cl)
        hi2 = pd.concat(
            [hi, pd.Series([hi.iloc[-1]], index=[hi.index[-1] + pd.Timedelta("1D")])]
        )
        lo2 = pd.concat(
            [lo, pd.Series([lo.iloc[-1]], index=[lo.index[-1] + pd.Timedelta("1D")])]
        )
        cl2 = pd.concat(
            [
                cl,
                pd.Series(
                    [cl.iloc[-1] * 0.95], index=[cl.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        r2 = PercentRExhaustion(short=14, long=50).compute(hi2, lo2, cl2)
        pd.testing.assert_series_equal(
            r1["pct_r_short"].dropna(),
            r2["pct_r_short"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = PercentRExhaustion().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_constant_price(self):
        n = 200
        hi = pd.Series(np.full(n, 100.5))
        lo = pd.Series(np.full(n, 99.5))
        cl = pd.Series(np.full(n, 100.0))
        result = PercentRExhaustion().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# LaguerreRSI
# ---------------------------------------------------------------------------


class TestLaguerreRSI:
    def test_returns_dataframe(self):
        _, _, cl = _ohlcv()
        result = LaguerreRSI().compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, _, cl = _ohlcv()
        result = LaguerreRSI().compute(cl)
        for col in ("lrsi", "lma", "lrsi_ob", "lrsi_os"):
            assert col in result.columns, f"{col} missing"

    def test_lrsi_bounded_zero_one(self):
        _, _, cl = _ohlcv()
        result = LaguerreRSI().compute(cl)
        valid = result["lrsi"].dropna()
        assert valid.min() >= -0.01, "lrsi below 0"
        assert valid.max() <= 1.01, "lrsi above 1"

    def test_lrsi_ob_os_binary(self):
        _, _, cl = _ohlcv()
        result = LaguerreRSI().compute(cl)
        for col in ("lrsi_ob", "lrsi_os"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError):
            LaguerreRSI(gamma=0.0)
        with pytest.raises(ValueError):
            LaguerreRSI(gamma=1.0)

    def test_strong_uptrend_overbought(self):
        """Monotone rally -> lrsi_ob should fire."""
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 200, n), index=dates)
        result = LaguerreRSI(gamma=0.5).compute(cl)
        assert result["lrsi_ob"].sum() > 0

    def test_strong_downtrend_oversold(self):
        """Monotone selloff -> lrsi_os should fire."""
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(200, 80, n), index=dates)
        result = LaguerreRSI(gamma=0.5).compute(cl)
        assert result["lrsi_os"].sum() > 0

    def test_no_lookahead(self):
        _, _, cl = _ohlcv(n=80)
        r1 = LaguerreRSI().compute(cl)
        cl2 = pd.concat(
            [
                cl,
                pd.Series(
                    [cl.iloc[-1] * 1.05], index=[cl.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        r2 = LaguerreRSI().compute(cl2)
        pd.testing.assert_series_equal(
            r1["lrsi"].dropna(),
            r2["lrsi"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )

    def test_single_bar_no_crash(self):
        cl = pd.Series([100.0])
        result = LaguerreRSI().compute(cl)
        assert isinstance(result, pd.DataFrame)
