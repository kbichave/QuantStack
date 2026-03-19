# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for momentum indicators: PercentRExhaustion, LaguerreRSI.
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.momentum import LaguerreRSI, PercentRExhaustion


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
        """Williams %R must be in [-100, 0]."""
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        for col in ("pct_r_short", "pct_r_long"):
            vals = result[col].dropna()
            assert (vals >= -100).all() and (vals <= 0).all(), f"{col} out of range"

    def test_exhaustion_top_binary(self):
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        vals = result["exhaustion_top"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_exhaustion_bottom_binary(self):
        hi, lo, cl = _ohlcv()
        result = PercentRExhaustion().compute(hi, lo, cl)
        vals = result["exhaustion_bottom"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_exhaustion_top_fires_in_overbought(self):
        """Monotone rising series should produce exhaustion_top near the end."""
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 250, n), index=dates)
        hi = cl + 0.1
        lo = cl - 0.1
        result = PercentRExhaustion(short=14, long=112, ob_threshold=-20.0).compute(hi, lo, cl)
        assert result["exhaustion_top"].iloc[120:].sum() > 0, "No overbought exhaustion fired"

    def test_exhaustion_bottom_fires_in_oversold(self):
        """Monotone falling series should produce exhaustion_bottom near the end."""
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(200, 50, n), index=dates)
        hi = cl + 0.1
        lo = cl - 0.1
        result = PercentRExhaustion(short=14, long=112, os_threshold=-80.0).compute(hi, lo, cl)
        assert result["exhaustion_bottom"].iloc[120:].sum() > 0, "No oversold exhaustion fired"

    def test_no_crash_short_series(self):
        hi = pd.Series([101.0, 102.0, 100.0])
        lo = pd.Series([99.0, 100.0, 98.0])
        cl = pd.Series([100.0, 101.0, 99.0])
        result = PercentRExhaustion().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_no_lookahead(self):
        hi, lo, cl = _ohlcv(n=80)
        r1 = PercentRExhaustion().compute(hi, lo, cl)
        hi2 = pd.concat([hi, pd.Series([hi.iloc[-1]], index=[hi.index[-1] + pd.Timedelta("1D")])])
        lo2 = pd.concat([lo, pd.Series([lo.iloc[-1]], index=[lo.index[-1] + pd.Timedelta("1D")])])
        cl2 = pd.concat([cl, pd.Series([cl.iloc[-1] * 1.01], index=[cl.index[-1] + pd.Timedelta("1D")])])
        r2 = PercentRExhaustion().compute(hi2, lo2, cl2)
        pd.testing.assert_series_equal(
            r1["pct_r_short"].dropna(),
            r2["pct_r_short"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )


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

    def test_lrsi_range(self):
        """Laguerre RSI must be in [0, 1]."""
        _, _, cl = _ohlcv()
        result = LaguerreRSI().compute(cl)
        vals = result["lrsi"].dropna()
        assert (vals >= 0).all() and (vals <= 1).all()

    def test_lrsi_ob_os_binary(self):
        _, _, cl = _ohlcv()
        result = LaguerreRSI().compute(cl)
        for col in ("lrsi_ob", "lrsi_os"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_lrsi_ob_fires_in_uptrend(self):
        """Sustained uptrend should produce at least one overbought reading."""
        n = 150
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 200, n), index=dates)
        result = LaguerreRSI(gamma=0.5).compute(cl)
        assert result["lrsi_ob"].sum() > 0

    def test_lrsi_os_fires_in_downtrend(self):
        """Sustained downtrend should produce at least one oversold reading."""
        n = 150
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(200, 100, n), index=dates)
        result = LaguerreRSI(gamma=0.5).compute(cl)
        assert result["lrsi_os"].sum() > 0

    def test_gamma_sensitivity(self):
        """Lower gamma = more lag-prone; at steady state lrsi with gamma=0.1 vs 0.9 differ."""
        _, _, cl = _ohlcv()
        r_low = LaguerreRSI(gamma=0.1).compute(cl)
        r_high = LaguerreRSI(gamma=0.9).compute(cl)
        # They should produce different outputs (gamma has effect)
        assert not r_low["lrsi"].equals(r_high["lrsi"])

    def test_no_lookahead(self):
        _, _, cl = _ohlcv(n=60)
        r1 = LaguerreRSI().compute(cl)
        cl2 = pd.concat([cl, pd.Series([cl.iloc[-1] * 1.02],
                                        index=[cl.index[-1] + pd.Timedelta("1D")])])
        r2 = LaguerreRSI().compute(cl2)
        pd.testing.assert_series_equal(
            r1["lrsi"].dropna(),
            r2["lrsi"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )
