# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for ICT Smart Money Concepts:
FairValueGapDetector, OrderBlockDetector, BreakerBlockDetector,
StructureAnalysis, EqualHighsLows, OTELevels.
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.smart_money import (
    BreakerBlockDetector,
    EqualHighsLows,
    FairValueGapDetector,
    OTELevels,
    OrderBlockDetector,
    StructureAnalysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bars(n: int = 80, seed: int = 7) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.4), index=dates).clip(50)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close.shift(1).fillna(close)
    return open_, high, low, close


# ---------------------------------------------------------------------------
# FairValueGapDetector
# ---------------------------------------------------------------------------


class TestFairValueGapDetector:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _bars()
        result = FairValueGapDetector().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _bars()
        result = FairValueGapDetector().compute(hi, lo, cl)
        for col in ("bullish_fvg", "bearish_fvg", "fvg_top", "fvg_bottom", "fvg_filled"):
            assert col in result.columns, f"{col} missing"

    def test_fvg_flags_binary(self):
        _, hi, lo, cl = _bars()
        result = FairValueGapDetector().compute(hi, lo, cl)
        for col in ("bullish_fvg", "bearish_fvg", "fvg_filled"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_bullish_fvg_known_pattern(self):
        """
        Bullish FVG: low[i] > high[i-2]. Construct explicit 3-candle gap up.
        Bar 0: H=101, L=99
        Bar 1: H=103, L=101  (middle candle)
        Bar 2: H=106, L=104  -> low[2]=104 > high[0]=101 → bullish FVG at bar 2
        """
        dates = pd.date_range("2022-01-01", periods=5, freq="1h")
        hi = pd.Series([101, 103, 106, 107, 108], index=dates, dtype=float)
        lo = pd.Series([99, 101, 104, 105, 106], index=dates, dtype=float)
        cl = pd.Series([100, 102, 105, 106, 107], index=dates, dtype=float)
        result = FairValueGapDetector().compute(hi, lo, cl)
        assert result["bullish_fvg"].iloc[2] == 1

    def test_bearish_fvg_known_pattern(self):
        """
        Bearish FVG: high[i] < low[i-2]. Construct explicit 3-candle gap down.
        Bar 0: H=101, L=99
        Bar 1: H=99, L=97
        Bar 2: H=96, L=94  -> high[2]=96 < low[0]=99 → bearish FVG at bar 2
        """
        dates = pd.date_range("2022-01-01", periods=5, freq="1h")
        hi = pd.Series([101, 99, 96, 95, 94], index=dates, dtype=float)
        lo = pd.Series([99, 97, 94, 93, 92], index=dates, dtype=float)
        cl = pd.Series([100, 98, 95, 94, 93], index=dates, dtype=float)
        result = FairValueGapDetector().compute(hi, lo, cl)
        assert result["bearish_fvg"].iloc[2] == 1

    def test_no_crash_single_bar(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        result = FairValueGapDetector().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# OrderBlockDetector
# ---------------------------------------------------------------------------


class TestOrderBlockDetector:
    def test_returns_dataframe(self):
        op, hi, lo, cl = _bars()
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        op, hi, lo, cl = _bars()
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        for col in ("bullish_ob", "bearish_ob", "ob_high", "ob_low"):
            assert col in result.columns, f"{col} missing"

    def test_ob_flags_binary(self):
        op, hi, lo, cl = _bars()
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        for col in ("bullish_ob", "bearish_ob"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_no_crash_constant_price(self):
        n = 30
        op = pd.Series(np.full(n, 100.0))
        hi = pd.Series(np.full(n, 100.5))
        lo = pd.Series(np.full(n, 99.5))
        cl = pd.Series(np.full(n, 100.0))
        result = OrderBlockDetector().compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# BreakerBlockDetector
# ---------------------------------------------------------------------------


class TestBreakerBlockDetector:
    def test_returns_dataframe(self):
        op, hi, lo, cl = _bars()
        result = BreakerBlockDetector().compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        op, hi, lo, cl = _bars()
        result = BreakerBlockDetector().compute(op, hi, lo, cl)
        for col in ("bullish_breaker", "bearish_breaker", "breaker_high", "breaker_low"):
            assert col in result.columns, f"{col} missing"

    def test_breaker_flags_binary(self):
        op, hi, lo, cl = _bars()
        result = BreakerBlockDetector().compute(op, hi, lo, cl)
        for col in ("bullish_breaker", "bearish_breaker"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"


# ---------------------------------------------------------------------------
# StructureAnalysis (BOS / CHoCH)
# ---------------------------------------------------------------------------


class TestStructureAnalysis:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _bars()
        result = StructureAnalysis().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _bars()
        result = StructureAnalysis().compute(hi, lo, cl)
        for col in ("swing_high", "swing_low", "bos_bullish", "bos_bearish",
                    "choch_bullish", "choch_bearish"):
            assert col in result.columns, f"{col} missing"

    def test_bos_binary(self):
        _, hi, lo, cl = _bars()
        result = StructureAnalysis().compute(hi, lo, cl)
        for col in ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_bos_bullish_in_uptrend(self):
        """Zigzag uptrend (rising peaks and troughs) should generate bullish BOS breaks."""
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="1h")
        t = np.arange(n)
        # Rising trend with oscillation creates swing highs that get broken
        cl = pd.Series(100 + t * 0.3 + np.sin(t * 0.5) * 3, index=dates)
        hi = cl + 0.5
        lo = cl - 0.5
        result = StructureAnalysis(swing_period=3).compute(hi, lo, cl)
        assert result["bos_bullish"].sum() > 0

    def test_bos_bearish_in_downtrend(self):
        """Zigzag downtrend should generate bearish BOS breaks."""
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="1h")
        t = np.arange(n)
        cl = pd.Series(200 - t * 0.3 + np.sin(t * 0.5) * 3, index=dates)
        hi = cl + 0.5
        lo = cl - 0.5
        result = StructureAnalysis(swing_period=3).compute(hi, lo, cl)
        assert result["bos_bearish"].sum() > 0


# ---------------------------------------------------------------------------
# EqualHighsLows
# ---------------------------------------------------------------------------


class TestEqualHighsLows:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _bars()
        result = EqualHighsLows().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _bars()
        result = EqualHighsLows().compute(hi, lo, cl)
        for col in ("equal_highs", "equal_lows"):
            assert col in result.columns, f"{col} missing"

    def test_flags_binary(self):
        _, hi, lo, cl = _bars()
        result = EqualHighsLows().compute(hi, lo, cl)
        for col in ("equal_highs", "equal_lows"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_detects_equal_highs(self):
        """
        Double-top pattern: two swing highs at approximately the same price
        should flag equal_highs with a loose tolerance.
        """
        n = 40
        dates = pd.date_range("2022-01-01", periods=n, freq="1h")
        # Construct two peaks at ~105 separated by a trough
        prices = ([100, 101, 103, 105, 104, 102, 100, 99, 100, 102,
                   104, 105, 104, 103, 101, 99, 98, 97, 98, 99,
                   100, 101, 103, 105, 104, 102, 100, 99, 100, 102,
                   104, 105, 104, 103, 101, 99, 98, 97, 98, 99])
        cl = pd.Series([float(p) for p in prices], index=dates)
        hi = cl + 0.3
        lo = cl - 0.3
        result = EqualHighsLows(lookback=20, tolerance_atr_multiple=0.5).compute(hi, lo, cl)
        assert result["equal_highs"].sum() > 0


# ---------------------------------------------------------------------------
# OTELevels
# ---------------------------------------------------------------------------


class TestOTELevels:
    def test_returns_dataframe(self):
        _, hi, lo, cl = _bars()
        result = OTELevels().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, hi, lo, cl = _bars()
        result = OTELevels().compute(hi, lo, cl)
        for col in ("swing_range_high", "swing_range_low", "ote_upper", "ote_lower", "price_in_ote"):
            assert col in result.columns, f"{col} missing"

    def test_price_in_ote_binary(self):
        _, hi, lo, cl = _bars()
        result = OTELevels().compute(hi, lo, cl)
        vals = result["price_in_ote"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_ote_bounds_ordering(self):
        """OTE upper should be >= OTE lower where both are defined."""
        _, hi, lo, cl = _bars()
        result = OTELevels().compute(hi, lo, cl)
        valid = result[["ote_upper", "ote_lower"]].dropna()
        assert (valid["ote_upper"] >= valid["ote_lower"]).all()

    def test_no_crash_short_series(self):
        hi = pd.Series([101.0, 102.0, 103.0])
        lo = pd.Series([99.0, 98.0, 97.0])
        cl = pd.Series([100.0, 100.5, 101.0])
        result = OTELevels().compute(hi, lo, cl)
        assert isinstance(result, pd.DataFrame)
