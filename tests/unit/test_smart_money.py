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


# ---------------------------------------------------------------------------
# BreakerBlockDetector
# ---------------------------------------------------------------------------


from quantcore.features.smart_money import BreakerBlockDetector, SilverBullet, MMXMCycle


class TestBreakerBlockDetector:
    def test_returns_dataframe(self, ohlcv):
        result = BreakerBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = BreakerBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert {"bullish_breaker", "bearish_breaker", "breaker_high", "breaker_low"}.issubset(
            set(result.columns)
        )

    def test_binary_breaker_columns(self, ohlcv):
        result = BreakerBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for col in ("bullish_breaker", "bearish_breaker"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_same_length_as_input(self, ohlcv):
        result = BreakerBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert len(result) == len(ohlcv)

    def test_bearish_breaker_fires_after_bullish_ob_violated(self):
        """Construct explicit OB then price drops through it."""
        # Bar 15 is a bearish candle; bar 16 is a large bullish impulse body
        # → bullish OB registered at bar 16 with zone from bar 15.
        # Then bars 26+ close below ob_low (95.5) → bearish breaker fires.
        n = 40
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        close = np.full(n, 100.0)
        high  = np.full(n, 101.0)
        low   = np.full(n, 99.0)
        open_ = np.full(n, 100.0)

        # Bar 15: bearish candle (OB candidate)
        open_[15] = 101.0; close[15] = 96.0; high[15] = 102.0; low[15] = 95.5
        # Bar 16: large bullish impulse body (close - open_ >> ATR * 0.5)
        open_[16] = 96.5; close[16] = 110.0; high[16] = 111.0; low[16] = 96.0
        # Bars 17-25: hold high level
        for k in range(17, 26):
            open_[k] = 109.0; close[k] = 110.0; high[k] = 111.0; low[k] = 108.5
        # Bars 26-39: crash below OB low (95.5)
        for k in range(26, n):
            open_[k] = 95.0; close[k] = 93.0; high[k] = 95.5; low[k] = 92.5

        result = BreakerBlockDetector(impulse_atr_multiple=0.5).compute(
            pd.Series(open_, index=dates),
            pd.Series(high, index=dates),
            pd.Series(low, index=dates),
            pd.Series(close, index=dates),
        )
        # After the OB is violated, at least one bearish breaker should fire
        assert result["bearish_breaker"].sum() > 0

    def test_breaker_zones_not_nan_when_breaker_fires(self, ohlcv):
        result = BreakerBlockDetector().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        breaker_bars = result[(result["bullish_breaker"] == 1) | (result["bearish_breaker"] == 1)]
        if len(breaker_bars) > 0:
            assert not breaker_bars["breaker_high"].isna().any()
            assert not breaker_bars["breaker_low"].isna().any()

    def test_no_crash_single_bar(self):
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        result = BreakerBlockDetector().compute(
            pd.Series([100.0], index=dates),
            pd.Series([101.0], index=dates),
            pd.Series([99.0], index=dates),
            pd.Series([100.0], index=dates),
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# SilverBullet
# ---------------------------------------------------------------------------


class TestSilverBullet:
    def _make_intraday(self, start="2023-01-02 09:00", periods=120, freq="1min"):
        """1-minute intraday bars covering the 10–11 AM NY window."""
        dates = pd.date_range(start=start, periods=periods, freq=freq)
        np.random.seed(99)
        close = 100 + np.cumsum(np.random.randn(periods) * 0.05)
        high  = close + np.abs(np.random.randn(periods) * 0.05) + 0.1
        low   = close - np.abs(np.random.randn(periods) * 0.05) - 0.1
        return pd.Series(high, index=dates), pd.Series(low, index=dates), pd.Series(close, index=dates)

    def test_returns_dataframe(self):
        high, low, close = self._make_intraday()
        result = SilverBullet().compute(high, low, close)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        high, low, close = self._make_intraday()
        result = SilverBullet().compute(high, low, close)
        assert {"sb_bullish", "sb_bearish", "sb_fvg_top", "sb_fvg_bot"}.issubset(
            set(result.columns)
        )

    def test_binary_signal_columns(self):
        high, low, close = self._make_intraday()
        result = SilverBullet().compute(high, low, close)
        for col in ("sb_bullish", "sb_bearish"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_same_length_as_input(self):
        high, low, close = self._make_intraday()
        result = SilverBullet().compute(high, low, close)
        assert len(result) == len(close)

    def test_no_signal_outside_window(self):
        """Bars from 15:00–16:00 are outside the 10–11 AM window — no Silver Bullet."""
        dates = pd.date_range(start="2023-01-02 15:00", periods=60, freq="1min")
        np.random.seed(7)
        close = pd.Series(100 + np.cumsum(np.random.randn(60) * 0.05), index=dates)
        high  = close + 0.1
        low   = close - 0.1
        result = SilverBullet().compute(high, low, close)
        assert result["sb_bullish"].sum() == 0
        assert result["sb_bearish"].sum() == 0

    def test_no_crash_daily_data(self, ohlcv):
        """Daily data has no intraday timestamps → window check should safely return zeros."""
        result = SilverBullet().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(ohlcv)


# ---------------------------------------------------------------------------
# MMXMCycle
# ---------------------------------------------------------------------------


class TestMMXMCycle:
    def test_returns_dataframe(self, ohlcv):
        result = MMXMCycle().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = MMXMCycle().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert {
            "mmxm_phase", "mmxm_label", "in_consolidation",
            "in_manipulation", "in_expansion", "in_retracement",
        }.issubset(set(result.columns))

    def test_phase_values_in_range(self, ohlcv):
        result = MMXMCycle().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result["mmxm_phase"].isin([0, 1, 2, 3]).all()

    def test_binary_indicator_columns(self, ohlcv):
        result = MMXMCycle().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for col in ("in_consolidation", "in_manipulation", "in_expansion", "in_retracement"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_labels_consistent_with_phase(self, ohlcv):
        result = MMXMCycle().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        label_map = {0: "consolidation", 1: "manipulation", 2: "expansion", 3: "retracement"}
        for phase_val, label_val in label_map.items():
            mask = result["mmxm_phase"] == phase_val
            if mask.any():
                assert (result.loc[mask, "mmxm_label"] == label_val).all()

    def test_expansion_detected_on_large_candle(self):
        """Explicitly construct a large candle that should trigger expansion."""
        n = 50
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        # First 30 bars: moderate noise
        close = np.full(n, 100.0)
        high  = np.full(n, 101.0)
        low   = np.full(n, 99.0)
        # Bar 35: huge expansion candle (range = 20 vs ATR ~2)
        high[35] = 120.0; low[35] = 100.0; close[35] = 119.0
        result = MMXMCycle(expansion_multiple=1.5).compute(
            pd.Series(high, index=dates),
            pd.Series(low, index=dates),
            pd.Series(close, index=dates),
        )
        assert result["in_expansion"].iloc[35] == 1

    def test_consolidation_on_tight_range(self):
        """Constant price produces low ATR → consolidation phase."""
        n = 60
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        close = pd.Series(np.full(n, 100.0), index=dates)
        high  = pd.Series(np.full(n, 100.01), index=dates)
        low   = pd.Series(np.full(n, 99.99), index=dates)
        result = MMXMCycle(atr_contraction_threshold=1.5).compute(high, low, close)
        # Later bars (past warmup) should be consolidation
        assert result["in_consolidation"].iloc[40:].sum() > 0

    def test_same_length_as_input(self, ohlcv):
        result = MMXMCycle().compute(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result) == len(ohlcv)

    def test_no_crash_short_series(self):
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        c = pd.Series([100.0, 101.0, 99.5, 102.0, 98.0], index=dates)
        h = c + 0.5
        lo = c - 0.5
        result = MMXMCycle().compute(h, lo, c)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# SMTDivergence
# ---------------------------------------------------------------------------


from quantcore.features.smart_money import SMTDivergence


class TestSMTDivergence:
    @pytest.fixture
    def dual_ohlcv(self):
        """Two correlated 100-bar instruments slightly offset."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(1)
        base = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_a = pd.Series(base + 1.0, index=dates)
        low_a  = pd.Series(base - 1.0, index=dates)
        high_b = pd.Series(base * 1.005 + 1.0, index=dates)   # slightly different scale
        low_b  = pd.Series(base * 1.005 - 1.0, index=dates)
        return high_a, low_a, high_b, low_b

    def test_returns_dataframe(self, dual_ohlcv):
        ha, la, hb, lb = dual_ohlcv
        result = SMTDivergence().compute(ha, la, hb, lb)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, dual_ohlcv):
        ha, la, hb, lb = dual_ohlcv
        result = SMTDivergence().compute(ha, la, hb, lb)
        assert {"bearish_smt", "bullish_smt", "smt_strength", "divergence_direction"}.issubset(
            set(result.columns)
        )

    def test_binary_signal_columns(self, dual_ohlcv):
        ha, la, hb, lb = dual_ohlcv
        result = SMTDivergence().compute(ha, la, hb, lb)
        for col in ("bearish_smt", "bullish_smt"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_same_length_as_input(self, dual_ohlcv):
        ha, la, hb, lb = dual_ohlcv
        result = SMTDivergence().compute(ha, la, hb, lb)
        assert len(result) == 100

    def test_smt_strength_nonnegative(self, dual_ohlcv):
        ha, la, hb, lb = dual_ohlcv
        result = SMTDivergence().compute(ha, la, hb, lb)
        assert (result["smt_strength"] >= 0).all()

    def test_bearish_smt_detected_on_synthetic_divergence(self):
        """A makes new swing high; B fails to confirm — bearish SMT fires."""
        # Use explicit arrays to avoid linspace endpoint ties that break swing detection.
        # Pattern: B has a clear swing high at bar 10, then declines.
        #          A then makes a new swing high at bar 20; B's high at bar 20 is well below bar 10's high.
        n = 40
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

        # Instrument A: steady rise to swing high at bar 20, then decline
        h_a = [100 + i for i in range(20)] + [119.5] + [118 - i * 0.5 for i in range(19)]
        high_a = pd.Series(h_a, index=dates)
        low_a  = pd.Series([v - 1.0 for v in h_a], index=dates)

        # Instrument B: rises to swing high of 115 at bar 10, then drops to ~108 and stays
        h_b = ([100 + i for i in range(10)] + [114.5] +
               [113 - i for i in range(8)] + [105.5] + [105.0] * 20)
        high_b = pd.Series(h_b, index=dates)
        low_b  = pd.Series([v - 1.0 for v in h_b], index=dates)

        result = SMTDivergence(swing_period=3, atr_tolerance=0.0001).compute(high_a, low_a, high_b, low_b)
        assert result["bearish_smt"].sum() > 0

    def test_direction_column_matches_signals(self, dual_ohlcv):
        ha, la, hb, lb = dual_ohlcv
        result = SMTDivergence().compute(ha, la, hb, lb)
        # Where bearish_smt==1, direction should be 'bearish'
        if result["bearish_smt"].sum() > 0:
            bear_bars = result[result["bearish_smt"] == 1]
            assert (bear_bars["divergence_direction"] == "bearish").all()
        # Where bullish_smt==1, direction should be 'bullish'
        if result["bullish_smt"].sum() > 0:
            bull_bars = result[result["bullish_smt"] == 1]
            assert (bull_bars["divergence_direction"] == "bullish").all()

    def test_no_crash_short_series(self):
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        ha = pd.Series([100.0, 101.0, 102.0, 101.5, 100.5], index=dates)
        la = pd.Series([99.0, 100.0, 101.0, 100.5, 99.5], index=dates)
        hb = pd.Series([200.0, 201.0, 202.0, 201.5, 200.5], index=dates)
        lb = pd.Series([199.0, 200.0, 201.0, 200.5, 199.5], index=dates)
        result = SMTDivergence().compute(ha, la, hb, lb)
        assert len(result) == 5
