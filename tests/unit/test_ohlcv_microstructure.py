# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.microstructure — OHLCV-derived microstructure signals."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.microstructure import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    OvernightGapPersistence,
    RealizedVarianceDecomposition,
    RollImpliedSpread,
    VWAPSessionDeviation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv():
    """100-bar OHLCV with volume."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.3) + 0.2
    low = close - np.abs(np.random.randn(100) * 0.3) - 0.2
    open_ = close + np.random.randn(100) * 0.1
    volume = np.random.randint(500_000, 5_000_000, 100).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# AmihudIlliquidity
# ---------------------------------------------------------------------------


class TestAmihudIlliquidity:
    def test_returns_dataframe(self, ohlcv):
        result = AmihudIlliquidity().compute(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = AmihudIlliquidity().compute(ohlcv["close"], ohlcv["volume"])
        assert {"amihud_raw", "amihud", "amihud_zscore"}.issubset(set(result.columns))

    def test_non_negative(self, ohlcv):
        result = AmihudIlliquidity().compute(ohlcv["close"], ohlcv["volume"])
        assert (result["amihud"].dropna() >= 0).all()
        assert (result["amihud_raw"].dropna() >= 0).all()

    def test_higher_when_volume_low(self):
        """Lower volume → higher illiquidity ratio."""
        np.random.seed(0)
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(50)), index=dates)
        vol_high = pd.Series(np.full(50, 5_000_000.0), index=dates)
        vol_low = pd.Series(np.full(50, 50_000.0), index=dates)

        r_high_vol = AmihudIlliquidity(period=10).compute(close, vol_high)["amihud"].dropna().mean()
        r_low_vol = AmihudIlliquidity(period=10).compute(close, vol_low)["amihud"].dropna().mean()

        assert r_low_vol > r_high_vol

    def test_zero_volume_no_crash(self, ohlcv):
        zero_vol = pd.Series(np.zeros(100), index=ohlcv.index)
        result = AmihudIlliquidity().compute(ohlcv["close"], zero_vol)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# RollImpliedSpread
# ---------------------------------------------------------------------------


class TestRollImpliedSpread:
    def test_returns_dataframe(self, ohlcv):
        result = RollImpliedSpread().compute(ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = RollImpliedSpread().compute(ohlcv["close"])
        assert {"roll_spread", "roll_spread_pct"}.issubset(set(result.columns))

    def test_spread_non_negative_where_defined(self, ohlcv):
        result = RollImpliedSpread().compute(ohlcv["close"])
        valid = result["roll_spread"].dropna()
        assert (valid >= 0).all()

    def test_pct_spread_reasonable_range(self, ohlcv):
        result = RollImpliedSpread().compute(ohlcv["close"])
        valid = result["roll_spread_pct"].dropna()
        assert (valid >= 0).all()
        assert (valid < 50).all()


# ---------------------------------------------------------------------------
# CorwinSchultzSpread
# ---------------------------------------------------------------------------


class TestCorwinSchultzSpread:
    def test_returns_dataframe(self, ohlcv):
        result = CorwinSchultzSpread().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = CorwinSchultzSpread().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert {"cs_spread", "cs_spread_pct", "cs_spread_ma"}.issubset(set(result.columns))

    def test_spread_non_negative_where_defined(self, ohlcv):
        result = CorwinSchultzSpread().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result["cs_spread"].dropna()
        assert (valid >= 0).all()

    def test_larger_hl_range_gives_larger_spread(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(1)
        close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.3), index=dates)

        high_narrow = close + 0.1
        low_narrow = close - 0.1
        high_wide = close + 3.0
        low_wide = close - 3.0

        spread_narrow = CorwinSchultzSpread(period=10).compute(high_narrow, low_narrow, close)["cs_spread"].dropna()
        spread_wide = CorwinSchultzSpread(period=10).compute(high_wide, low_wide, close)["cs_spread"].dropna()

        # Both must have some valid values; wide range should produce larger spread
        assert len(spread_wide) > 0
        if len(spread_narrow) > 0:
            assert spread_wide.mean() > spread_narrow.mean()


# ---------------------------------------------------------------------------
# RealizedVarianceDecomposition
# ---------------------------------------------------------------------------


class TestRealizedVarianceDecomposition:
    def test_returns_dataframe(self, ohlcv):
        result = RealizedVarianceDecomposition().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = RealizedVarianceDecomposition().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        expected = {
            "overnight_var", "intraday_var", "total_var",
            "overnight_var_ratio", "rv_overnight_ma", "rv_intraday_ma",
        }
        assert expected.issubset(set(result.columns))

    def test_components_sum_to_total(self, ohlcv):
        result = RealizedVarianceDecomposition().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        diff = (result["overnight_var"] + result["intraday_var"] - result["total_var"]).dropna().abs()
        assert (diff < 1e-10).all()

    def test_ratio_in_zero_one(self, ohlcv):
        result = RealizedVarianceDecomposition().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        ratio = result["overnight_var_ratio"].dropna()
        assert (ratio >= 0).all()
        assert (ratio <= 1).all()


# ---------------------------------------------------------------------------
# VWAPSessionDeviation
# ---------------------------------------------------------------------------


class TestVWAPSessionDeviation:
    def test_returns_dataframe(self, ohlcv):
        result = VWAPSessionDeviation().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = VWAPSessionDeviation().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert {"vwap_rolling", "vwap_deviation", "vwap_deviation_zscore"}.issubset(set(result.columns))

    def test_zero_deviation_when_close_equals_typical(self):
        """When close = typical price (H+L+C)/3 = (101+99+100)/3=100, deviation ≈ 0."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(np.full(50, 100.0), index=dates)
        high = pd.Series(np.full(50, 101.0), index=dates)
        low = pd.Series(np.full(50, 99.0), index=dates)
        volume = pd.Series(np.full(50, 1_000_000.0), index=dates)

        result = VWAPSessionDeviation(period=10).compute(high, low, close, volume)
        dev = result["vwap_deviation"].dropna()
        assert (dev.abs() < 1e-6).all()


# ---------------------------------------------------------------------------
# OvernightGapPersistence
# ---------------------------------------------------------------------------


class TestOvernightGapPersistence:
    def test_returns_dataframe(self, ohlcv):
        result = OvernightGapPersistence().compute(ohlcv["open"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = OvernightGapPersistence().compute(ohlcv["open"], ohlcv["close"])
        expected = {"gap_pct", "gap_up", "gap_down", "gap_filled", "gap_persisted", "gap_filled_pct"}
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self, ohlcv):
        result = OvernightGapPersistence().compute(ohlcv["open"], ohlcv["close"])
        for col in ("gap_up", "gap_down", "gap_filled", "gap_persisted"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_gap_up_down_mutually_exclusive(self, ohlcv):
        result = OvernightGapPersistence().compute(ohlcv["open"], ohlcv["close"])
        both = (result["gap_up"] == 1) & (result["gap_down"] == 1)
        assert not both.any()

    def test_gap_fill_vs_persist_mutually_exclusive(self, ohlcv):
        result = OvernightGapPersistence().compute(ohlcv["open"], ohlcv["close"])
        both = (result["gap_filled"] == 1) & (result["gap_persisted"] == 1)
        assert not both.any()

    def test_gap_persists_when_open_equals_close(self):
        """Gap opens up, price does NOT retrace (close >= open) → gap persists."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        # yesterday's close rising slowly
        prior_close = pd.Series(np.linspace(100, 115, 30), index=dates)
        # each day: open 2% above prior close, close AT open (no intraday move)
        open_ = prior_close * 1.02
        close = open_.copy()  # no intraday movement

        result = OvernightGapPersistence(min_gap_pct=0.5).compute(open_, close)
        # gap_pct = (open - prior_close) / prior_close; since close=open,
        # intraday_move = 0 → not negative → gap not filled → persisted
        gap_up_bars = result[result["gap_up"] == 1]
        assert len(gap_up_bars) > 0, "expected some gap-up bars"
        assert (gap_up_bars["gap_persisted"] == 1).all()
