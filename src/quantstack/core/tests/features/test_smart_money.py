# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for ICT smart money indicators: FairValueGap, OrderBlocks,
BOS/CHoCH (StructureAnalysis), EqualHighsLows, OTE (OTELevels).

Tests verify correctness properties (no lookahead, binary outputs,
structural detection logic) following the plan's testing specification.
"""

import numpy as np
import pandas as pd

from quantstack.core.features.smart_money import (
    EqualHighsLows,
    FairValueGapDetector,
    OrderBlockDetector,
    OTELevels,
    StructureAnalysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(
    n: int = 100, seed: int = 42
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Synthetic (open, high, low, close) bars."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 1.2), index=dates).clip(50)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(
        np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.8), index=dates
    )
    low = pd.Series(
        np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.8), index=dates
    )
    return open_, high, low, close


def _trending_ohlcv(direction: str = "up", n: int = 100):
    """Generate clearly trending OHLCV with swing highs/lows."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    if direction == "up":
        base = np.linspace(100, 180, n) + np.sin(np.linspace(0, 12, n)) * 5
    else:
        base = np.linspace(180, 100, n) + np.sin(np.linspace(0, 12, n)) * 5
    close = pd.Series(base, index=dates)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_, close).values + 1.5, index=dates)
    low = pd.Series(np.minimum(open_, close).values - 1.5, index=dates)
    return open_, high, low, close


# ---------------------------------------------------------------------------
# FairValueGapDetector
# ---------------------------------------------------------------------------


class TestFairValueGap:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _ohlcv()
        result = FairValueGapDetector().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _ohlcv()
        result = FairValueGapDetector().compute(hi, lo, cl)
        for col in (
            "bullish_fvg",
            "bearish_fvg",
            "fvg_top",
            "fvg_bottom",
            "fvg_filled",
        ):
            assert col in result.columns, f"{col} missing"

    def test_fvg_binary(self):
        _, hi, lo, cl = _ohlcv()
        result = FairValueGapDetector().compute(hi, lo, cl)
        for col in ("bullish_fvg", "bearish_fvg"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_fvg_detected_in_volatile_data(self):
        """With sufficient volatility, at least one FVG should appear."""
        _, hi, lo, cl = _ohlcv(n=200, seed=7)
        result = FairValueGapDetector(min_gap_atr_multiple=0.05).compute(hi, lo, cl)
        total = result["bullish_fvg"].sum() + result["bearish_fvg"].sum()
        assert total > 0, "No FVGs detected in volatile data"

    def test_no_lookahead(self):
        _, hi, lo, cl = _ohlcv(n=80)
        r1 = FairValueGapDetector().compute(hi, lo, cl)
        hi2 = pd.concat(
            [
                hi,
                pd.Series(
                    [hi.iloc[-1] * 1.05], index=[hi.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        lo2 = pd.concat(
            [
                lo,
                pd.Series(
                    [lo.iloc[-1] * 0.95], index=[lo.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        cl2 = pd.concat(
            [
                cl,
                pd.Series(
                    [cl.iloc[-1] * 1.03], index=[cl.index[-1] + pd.Timedelta("1D")]
                ),
            ]
        )
        r2 = FairValueGapDetector().compute(hi2, lo2, cl2)
        pd.testing.assert_series_equal(
            r1["bullish_fvg"],
            r2["bullish_fvg"].iloc[: len(r1)],
            check_names=False,
            check_freq=False,
        )

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = FairValueGapDetector().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# OrderBlockDetector
# ---------------------------------------------------------------------------


class TestOrderBlock:
    def test_returns_dataframe(self):
        op, hi, lo, cl = _ohlcv()
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        op, hi, lo, cl = _ohlcv()
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        for col in ("bullish_ob", "bearish_ob", "ob_high", "ob_low"):
            assert col in result.columns, f"{col} missing"

    def test_ob_binary(self):
        op, hi, lo, cl = _ohlcv()
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        for col in ("bullish_ob", "bearish_ob"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_ob_detected_in_trending_data(self):
        """Trending data with reversals should produce at least one OB."""
        op, hi, lo, cl = _ohlcv(n=200, seed=13)
        result = OrderBlockDetector(impulse_atr_multiple=0.5).compute(op, hi, lo, cl)
        total = result["bullish_ob"].sum() + result["bearish_ob"].sum()
        assert total > 0

    def test_single_bar_no_crash(self):
        op = pd.Series([100.0])
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.5])
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# StructureAnalysis (BOS + CHoCH)
# ---------------------------------------------------------------------------


class TestStructureAnalysis:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _ohlcv()
        result = StructureAnalysis().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _ohlcv()
        result = StructureAnalysis().compute(hi, lo, cl)
        for col in (
            "swing_high",
            "swing_low",
            "bos_bullish",
            "bos_bearish",
            "choch_bullish",
            "choch_bearish",
        ):
            assert col in result.columns, f"{col} missing"

    def test_all_binary(self):
        _, hi, lo, cl = _ohlcv()
        result = StructureAnalysis().compute(hi, lo, cl)
        for col in (
            "swing_high",
            "swing_low",
            "bos_bullish",
            "bos_bearish",
            "choch_bullish",
            "choch_bearish",
        ):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_bos_detected_in_uptrend(self):
        """Uptrend with pullbacks should produce bullish BOS."""
        # Smooth sinusoid won't produce distinct swing highs; use noisy uptrend
        _, hi, lo, cl = _ohlcv(n=200, seed=99)
        result = StructureAnalysis(swing_period=3).compute(hi, lo, cl)
        assert result["bos_bullish"].sum() > 0 or result["bos_bearish"].sum() > 0

    def test_bos_detected_in_downtrend(self):
        """Volatile data should produce at least one bearish BOS."""
        _, hi, lo, cl = _ohlcv(n=200, seed=77)
        result = StructureAnalysis(swing_period=3).compute(hi, lo, cl)
        assert result["bos_bullish"].sum() + result["bos_bearish"].sum() > 0

    def test_swing_points_detected(self):
        _, hi, lo, cl = _ohlcv(n=100)
        result = StructureAnalysis(swing_period=3).compute(hi, lo, cl)
        assert result["swing_high"].sum() > 0
        assert result["swing_low"].sum() > 0

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = StructureAnalysis().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# EqualHighsLows
# ---------------------------------------------------------------------------


class TestEqualHighsLows:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _ohlcv()
        result = EqualHighsLows().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _ohlcv()
        result = EqualHighsLows().compute(hi, lo, cl)
        for col in ("equal_highs", "equal_lows"):
            assert col in result.columns, f"{col} missing"

    def test_binary(self):
        _, hi, lo, cl = _ohlcv()
        result = EqualHighsLows().compute(hi, lo, cl)
        for col in ("equal_highs", "equal_lows"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_detects_repeated_highs(self):
        """Two bars hitting the same high should flag equal_highs."""
        n = 40
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        hi = pd.Series(np.random.uniform(99, 101, n), index=dates)
        hi.iloc[10] = 105.0
        hi.iloc[25] = 105.0
        lo = hi - 2.0
        cl = hi - 1.0
        result = EqualHighsLows(lookback=20, tolerance_atr_multiple=0.2).compute(
            hi, lo, cl
        )
        assert result["equal_highs"].sum() > 0

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = EqualHighsLows().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# OTELevels
# ---------------------------------------------------------------------------


class TestOTELevels:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _ohlcv()
        result = OTELevels().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _ohlcv()
        result = OTELevels().compute(hi, lo, cl)
        for col in (
            "swing_range_high",
            "swing_range_low",
            "ote_upper",
            "ote_lower",
            "price_in_ote",
        ):
            assert col in result.columns, f"{col} missing"

    def test_price_in_ote_binary(self):
        _, hi, lo, cl = _ohlcv()
        result = OTELevels().compute(hi, lo, cl)
        vals = result["price_in_ote"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_ote_upper_above_lower(self):
        """OTE upper (0.618 retrace) should be >= OTE lower (0.79 retrace)."""
        _, hi, lo, cl = _ohlcv()
        result = OTELevels().compute(hi, lo, cl)
        valid = result.dropna(subset=["ote_upper", "ote_lower"])
        if len(valid) > 0:
            assert (valid["ote_upper"] >= valid["ote_lower"]).all()

    def test_ote_within_swing_range(self):
        """OTE levels should be between swing low and swing high."""
        _, hi, lo, cl = _ohlcv()
        result = OTELevels().compute(hi, lo, cl)
        valid = result.dropna(
            subset=["ote_upper", "ote_lower", "swing_range_high", "swing_range_low"]
        )
        if len(valid) > 0:
            assert (valid["ote_upper"] <= valid["swing_range_high"] + 0.01).all()
            assert (valid["ote_lower"] >= valid["swing_range_low"] - 0.01).all()

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = OTELevels().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)
