# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.smart_money — ICT concepts."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.smart_money import (
    EqualHighsLows,
    FairValueGapDetector,
    ICTKillZones,
    ICTPowerOfThree,
    OrderBlockDetector,
    OTELevels,
    StructureAnalysis,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv():
    """100-bar daily OHLCV."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.4) + 0.3
    low = close - np.abs(np.random.randn(100) * 0.4) - 0.3
    open_ = close + np.random.randn(100) * 0.1
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=dates,
    )


# ---------------------------------------------------------------------------
# FairValueGapDetector
# ---------------------------------------------------------------------------


class TestFairValueGapDetector:
    def test_returns_dataframe(self, ohlcv):
        result = FairValueGapDetector().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = FairValueGapDetector().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert {
            "bullish_fvg", "bearish_fvg", "fvg_top", "fvg_bottom", "fvg_filled"
        }.issubset(set(result.columns))

    def test_binary_signal_columns(self, ohlcv):
        result = FairValueGapDetector().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for col in ("bullish_fvg", "bearish_fvg", "fvg_filled"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_bullish_fvg_fires_when_gap_exists(self):
        """Construct exact 3-candle bullish FVG: bar[0].high=100, bar[2].low=105."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        high = pd.Series([100.0, 101.0, 106.0, 107.0, 108.0], index=dates)
        low = pd.Series([98.0, 99.0, 105.0, 104.0, 103.0], index=dates)
        close = pd.Series([99.0, 100.0, 105.5, 106.0, 107.0], index=dates)

        # bar[2]: low[2]=105 > high[0]=100 → bullish FVG at bar 2
        result = FairValueGapDetector(min_gap_atr_multiple=0.0).compute(high, low, close)
        assert result["bullish_fvg"].iloc[2] == 1

    def test_bearish_fvg_fires_when_gap_exists(self):
        """Construct exact 3-candle bearish FVG: bar[0].low=100, bar[2].high=95."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        high = pd.Series([102.0, 101.0, 95.0, 94.0, 93.0], index=dates)
        low = pd.Series([100.0, 99.0, 88.0, 87.0, 86.0], index=dates)
        close = pd.Series([101.0, 100.0, 90.0, 89.0, 88.0], index=dates)

        # bar[2]: high[2]=95 < low[0]=100 → bearish FVG at bar 2
        result = FairValueGapDetector(min_gap_atr_multiple=0.0).compute(high, low, close)
        assert result["bearish_fvg"].iloc[2] == 1

    def test_no_fvg_on_overlapping_candles(self, ohlcv):
        """Normal overlapping OHLCV should have few or zero FVGs with tight filter."""
        # With a 5× ATR filter, very few bars should qualify as FVGs in normal data
        result = FairValueGapDetector(min_gap_atr_multiple=5.0).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        total_fvg = result["bullish_fvg"].sum() + result["bearish_fvg"].sum()
        assert total_fvg == 0


# ---------------------------------------------------------------------------
# OrderBlockDetector
# ---------------------------------------------------------------------------


class TestOrderBlockDetector:
    def test_returns_dataframe(self, ohlcv):
        result = OrderBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = OrderBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert {"bullish_ob", "bearish_ob", "ob_high", "ob_low"}.issubset(set(result.columns))

    def test_binary_ob_columns(self, ohlcv):
        result = OrderBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for col in ("bullish_ob", "bearish_ob"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_ob_zone_populated_when_ob_fires(self, ohlcv):
        result = OrderBlockDetector(impulse_atr_multiple=0.5).compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        ob_bars = result[(result["bullish_ob"] == 1) | (result["bearish_ob"] == 1)]
        if len(ob_bars) > 0:
            assert ob_bars["ob_high"].notna().all()
            assert ob_bars["ob_low"].notna().all()


# ---------------------------------------------------------------------------
# StructureAnalysis
# ---------------------------------------------------------------------------


class TestStructureAnalysis:
    def test_returns_dataframe(self, ohlcv):
        result = StructureAnalysis().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = StructureAnalysis().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        expected = {
            "swing_high", "swing_low",
            "bos_bullish", "bos_bearish",
            "choch_bullish", "choch_bearish",
        }
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self, ohlcv):
        result = StructureAnalysis().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for col in [
            "swing_high", "swing_low",
            "bos_bullish", "bos_bearish",
            "choch_bullish", "choch_bearish",
        ]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_bos_in_uptrend(self):
        """Rise to 120, 5-bar pullback to 108, re-rally to 135 → swing high detected and broken."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        # rise → pullback (5 bars below peak) → re-rally past peak
        rise1    = np.linspace(100, 120, 8)   # bars 0-7
        pullback = np.linspace(120, 108, 6)   # bars 8-13: 5 bars clearly below peak
        rise2    = np.linspace(108, 135, 16)  # bars 14-29: crosses above 121 (peak high)
        close_vals = np.concatenate([rise1, pullback, rise2])
        close = pd.Series(close_vals, index=dates)
        high = close + 1.0
        low = close - 1.0
        result = StructureAnalysis(swing_period=3).compute(high, low, close)
        assert result["bos_bullish"].sum() > 0


# ---------------------------------------------------------------------------
# EqualHighsLows
# ---------------------------------------------------------------------------


class TestEqualHighsLows:
    def test_returns_dataframe(self, ohlcv):
        result = EqualHighsLows().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = EqualHighsLows().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert {"equal_highs", "equal_lows"}.issubset(set(result.columns))

    def test_binary_columns(self, ohlcv):
        result = EqualHighsLows().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for col in ("equal_highs", "equal_lows"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_detects_identical_highs(self):
        """Two bars with identical highs should be flagged as equal highs."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        close = pd.Series(np.linspace(100, 110, 30), index=dates)
        high = close + 2.0
        low = close - 1.0
        # Make bar 10 and bar 20 have identical highs
        high.iloc[10] = 110.0
        high.iloc[20] = 110.0

        result = EqualHighsLows(lookback=15, tolerance_atr_multiple=0.5).compute(high, low, close)
        assert result["equal_highs"].iloc[20] == 1


# ---------------------------------------------------------------------------
# OTELevels
# ---------------------------------------------------------------------------


class TestOTELevels:
    def test_returns_dataframe(self, ohlcv):
        result = OTELevels().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = OTELevels().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert {
            "swing_range_high", "swing_range_low",
            "ote_upper", "ote_lower", "price_in_ote",
        }.issubset(set(result.columns))

    def test_ote_upper_above_lower(self, ohlcv):
        result = OTELevels().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid["ote_upper"] >= valid["ote_lower"]).all()

    def test_price_in_ote_binary(self, ohlcv):
        result = OTELevels().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        vals = result["price_in_ote"].unique()
        assert set(vals).issubset({0, 1})


# ---------------------------------------------------------------------------
# ICTKillZones
# ---------------------------------------------------------------------------


class TestICTKillZones:
    def test_returns_dataframe(self):
        idx = pd.date_range(start="2023-01-01 00:00", periods=48, freq="30min")
        result = ICTKillZones().compute(idx)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        idx = pd.date_range(start="2023-01-01 00:00", periods=48, freq="30min")
        result = ICTKillZones().compute(idx)
        expected = {"in_asia_kz", "in_london_kz", "in_ny_am_kz", "in_ny_pm_kz", "in_any_kz"}
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self):
        idx = pd.date_range(start="2023-01-01 00:00", periods=48, freq="30min")
        result = ICTKillZones().compute(idx)
        for col in ("in_asia_kz", "in_london_kz", "in_ny_am_kz", "in_ny_pm_kz", "in_any_kz"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_london_kz_fires_at_correct_hours(self):
        """London KZ is 07:00-10:00 UTC."""
        idx = pd.date_range(start="2023-01-02 07:00", periods=6, freq="60min")
        result = ICTKillZones().compute(idx)
        # 07:00, 08:00, 09:00 are in London KZ; 10:00 is the boundary (exclusive)
        assert result["in_london_kz"].iloc[0] == 1  # 07:00
        assert result["in_london_kz"].iloc[2] == 1  # 09:00
        assert result["in_london_kz"].iloc[3] == 0  # 10:00

    def test_ny_am_fires_at_correct_hours(self):
        """NY AM KZ is 13:30-16:00 UTC."""
        idx = pd.date_range(start="2023-01-02 13:30", periods=4, freq="60min")
        result = ICTKillZones().compute(idx)
        assert result["in_ny_am_kz"].iloc[0] == 1  # 13:30


# ---------------------------------------------------------------------------
# ICTPowerOfThree
# ---------------------------------------------------------------------------


class TestICTPowerOfThree:
    def test_returns_dataframe(self, ohlcv):
        result = ICTPowerOfThree().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = ICTPowerOfThree().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        expected = {
            "session_range_pct", "tight_range",
            "manipulation_up", "manipulation_down",
            "distribution_up", "distribution_down",
        }
        assert expected.issubset(set(result.columns))

    def test_binary_columns(self, ohlcv):
        result = ICTPowerOfThree().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for col in ("tight_range", "manipulation_up", "manipulation_down",
                    "distribution_up", "distribution_down"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_distribution_up_in_strong_trend(self):
        """Close consistently above prior high → distribution_up fires."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        close = pd.Series(np.linspace(100, 200, 30), index=dates)
        high = close + 1.0
        low = close - 1.0
        result = ICTPowerOfThree().compute(close, high, low, close)
        assert result["distribution_up"].sum() > 0
