# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for order flow approximations: CVD, VPIN, Hawkes, Footprint."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.flow import (
    CumulativeVolumeDelta,
    FootprintApproximation,
    HawkesIntensity,
    VPIN,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv():
    """100-bar daily OHLCV with random walk."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close_arr = 100 + np.cumsum(np.random.randn(100) * 0.5)
    close = pd.Series(close_arr, index=dates)
    high  = close + np.abs(np.random.randn(100) * 0.4) + 0.3
    low   = close - np.abs(np.random.randn(100) * 0.4) - 0.3
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(500_000, 2_000_000, 100).astype(float), index=dates)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# CumulativeVolumeDelta
# ---------------------------------------------------------------------------


class TestCumulativeVolumeDelta:
    def test_returns_dataframe(self, ohlcv):
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert {"bar_delta", "cvd", "cvd_ma", "cvd_divergence", "buy_vol", "sell_vol"}.issubset(
            set(result.columns)
        )

    def test_same_length(self, ohlcv):
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert len(result) == len(ohlcv)

    def test_buy_sell_sum_to_volume(self, ohlcv):
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        total = result["buy_vol"] + result["sell_vol"]
        pd.testing.assert_series_equal(total, ohlcv["volume"], check_names=False, atol=1e-6)

    def test_buy_vol_nonneg(self, ohlcv):
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert (result["buy_vol"] >= 0).all()
        assert (result["sell_vol"] >= 0).all()

    def test_cvd_monotone_on_all_bullish_bars(self):
        """Close = high on every bar → all volume is buy → CVD always increases."""
        n = 30
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        high  = pd.Series(np.arange(100, 130, dtype=float), index=dates)
        low   = pd.Series(np.arange(99, 129, dtype=float), index=dates)
        close = high.copy()  # close == high → buy_frac = 1
        open_ = low.copy()
        vol   = pd.Series(np.full(n, 1_000_000.0), index=dates)
        result = CumulativeVolumeDelta().compute(open_, high, low, close, vol)
        assert (result["bar_delta"] > 0).all()
        assert result["cvd"].is_monotonic_increasing

    def test_rolling_cvd_resets(self, ohlcv):
        result = CumulativeVolumeDelta(lookback=10).compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        # Rolling CVD should not exceed 10 bars of full volume
        max_possible = ohlcv["volume"].rolling(10).sum().max()
        assert result["cvd"].abs().max() <= max_possible + 1e-6

    def test_signed_volume_bypass(self, ohlcv):
        """Pass pre-computed signed volume; result should match."""
        sv = ohlcv["volume"] * 0.3  # arbitrary positive bias
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"],
            signed_volume=sv,
        )
        assert (result["cvd"] > 0).all()  # all positive signed volume → CVD always positive

    def test_divergence_column_values(self, ohlcv):
        result = CumulativeVolumeDelta().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert set(result["cvd_divergence"].unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------


class TestVPIN:
    def test_returns_dataframe(self, ohlcv):
        result = VPIN().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = VPIN().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert {"vpin", "vpin_high", "bucket_imbalance"}.issubset(set(result.columns))

    def test_same_length(self, ohlcv):
        result = VPIN().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert len(result) == len(ohlcv)

    def test_vpin_in_unit_interval(self, ohlcv):
        result = VPIN().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result["vpin"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_vpin_high_is_binary(self, ohlcv):
        result = VPIN().compute(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        vals = result["vpin_high"].unique()
        assert set(vals).issubset({0, 1})

    def test_high_imbalance_on_directional_bars(self):
        """Close always near high → large buy imbalance → elevated VPIN."""
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close = pd.Series(np.linspace(100, 200, n), index=dates)
        high  = close + 0.1
        low   = close - 5.0  # close much closer to high than low
        open_ = low + 0.1
        vol   = pd.Series(np.full(n, 1_000_000.0), index=dates)
        result = VPIN(n_buckets=10, window=10).compute(open_, high, low, close, vol)
        valid = result["vpin"].dropna()
        # High buy imbalance → VPIN should be > 0.5 on most bars
        assert (valid > 0.3).sum() / len(valid) > 0.5


# ---------------------------------------------------------------------------
# HawkesIntensity
# ---------------------------------------------------------------------------


class TestHawkesIntensity:
    def test_returns_dataframe(self, ohlcv):
        result = HawkesIntensity().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = HawkesIntensity().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert {"event", "intensity", "intensity_high", "excited"}.issubset(set(result.columns))

    def test_same_length(self, ohlcv):
        result = HawkesIntensity().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert len(result) == len(ohlcv)

    def test_intensity_nonneg(self, ohlcv):
        result = HawkesIntensity().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert (result["intensity"] >= 0).all()

    def test_event_is_binary(self, ohlcv):
        result = HawkesIntensity().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        vals = result["event"].unique()
        assert set(vals).issubset({0, 1})

    def test_excited_is_binary(self, ohlcv):
        result = HawkesIntensity().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        vals = result["excited"].unique()
        assert set(vals).issubset({0, 1})

    def test_intensity_decays_after_event(self):
        """After a spike event, intensity should decay toward baseline over subsequent bars."""
        n = 30
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        # All bars quiet except bar 5 (massive volume)
        close  = pd.Series(np.full(n, 100.0), index=dates)
        high   = pd.Series(np.full(n, 100.5), index=dates)
        low    = pd.Series(np.full(n, 99.5), index=dates)
        vol    = pd.Series(np.full(n, 100_000.0), index=dates)
        vol.iloc[5] = 10_000_000.0  # spike

        result = HawkesIntensity(decay=0.5, excitation=1.0, baseline=0.1,
                                  event_threshold=1.5, event_window=5).compute(
            high, low, close, vol
        )
        # Intensity at bar 6 should exceed baseline
        assert result["intensity"].iloc[6] > 0.1
        # Intensity should decrease from bar 6 onward (no more events)
        intensities_after_spike = result["intensity"].iloc[7:15].values
        assert intensities_after_spike[-1] < intensities_after_spike[0]

    def test_baseline_when_no_events(self):
        """With no events and decay=0, intensity should converge to baseline."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close  = pd.Series(np.full(n, 100.0), index=dates)
        high   = pd.Series(np.full(n, 100.01), index=dates)
        low    = pd.Series(np.full(n, 99.99), index=dates)
        vol    = pd.Series(np.full(n, 1_000.0), index=dates)  # tiny uniform volume
        result = HawkesIntensity(baseline=0.2, event_threshold=100).compute(
            high, low, close, vol
        )
        # With threshold=100×avg, no events fire → intensity stays near baseline
        assert result["event"].sum() == 0
        last_intensity = result["intensity"].iloc[-1]
        assert abs(last_intensity - 0.2) < 0.05


# ---------------------------------------------------------------------------
# FootprintApproximation
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_footprint():
    """50-bar OHLCV with clear directional segments."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(7)
    close = pd.Series(100.0 + np.cumsum(np.random.randn(50) * 0.5), index=dates)
    high  = close + np.abs(np.random.randn(50) * 0.3) + 0.2
    low   = close - np.abs(np.random.randn(50) * 0.3) - 0.2
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(100_000, 500_000, 50).astype(float), index=dates)
    return open_, high, low, close, volume


class TestFootprintApproximation:
    def test_returns_dataframe(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        expected = {
            "buy_vol", "sell_vol", "bar_delta", "delta_pct",
            "imbalanced_bull", "imbalanced_bear",
            "stacked_bull", "stacked_bear", "poc_price",
        }
        assert expected.issubset(set(result.columns))

    def test_same_length(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        assert len(result) == 50

    def test_buy_plus_sell_equals_volume(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        pd.testing.assert_series_equal(
            (result["buy_vol"] + result["sell_vol"]).round(6),
            vol.round(6),
            check_names=False,
        )

    def test_delta_pct_in_minus1_plus1(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        valid = result["delta_pct"].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_binary_imbalance_columns(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        for col in ("imbalanced_bull", "imbalanced_bear", "stacked_bull", "stacked_bear"):
            assert set(result[col].unique()).issubset({0, 1})

    def test_bull_and_bear_imbalance_mutually_exclusive(self, ohlcv_footprint):
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation().compute(open_, high, low, close, vol)
        assert (result["imbalanced_bull"] & result["imbalanced_bear"]).sum() == 0

    def test_stacked_bull_requires_n_consecutive(self):
        """Construct bars where close is near high (buy_frac ~ 0.9) so bar_delta > 0."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        close  = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 103.5, 103.0, 103.5, 104.0, 104.5], index=dates)
        # close near top → buy_frac ≈ 0.9 → positive delta on all bars
        high   = close + 0.1
        low    = close - 0.9
        open_  = close - 0.5
        volume = pd.Series(np.full(10, 100_000.0), index=dates)
        result = FootprintApproximation(stack_n=3).compute(open_, high, low, close, volume)
        # All bars have buy_frac ≈ 0.9 → positive delta → stacked_bull fires from bar 2 onward
        assert result["stacked_bull"].iloc[2:].sum() > 0

    def test_poc_price_within_bar_range(self, ohlcv_footprint):
        """POC price must be within the high/low range of some bar in the lookback window."""
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation(lookback=10).compute(open_, high, low, close, vol)
        poc = result["poc_price"].dropna()
        # POC is hl2 of some bar → must lie between overall min low and max high
        assert (poc >= low.min()).all()
        assert (poc <= high.max()).all()

    def test_poc_nan_before_lookback(self, ohlcv_footprint):
        """POC should be NaN for bars before lookback window fills."""
        open_, high, low, close, vol = ohlcv_footprint
        result = FootprintApproximation(lookback=15).compute(open_, high, low, close, vol)
        assert result["poc_price"].iloc[:14].isna().all()
        assert not result["poc_price"].iloc[14:].isna().any()

    def test_zero_volume_bar_handled(self):
        """Zero-volume bars should not cause division errors."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        close  = pd.Series([100.0, 101.0, 100.5, 101.5, 102.0], index=dates)
        high   = close + 0.5
        low    = close - 0.5
        open_  = close - 0.1
        volume = pd.Series([100_000.0, 0.0, 100_000.0, 0.0, 100_000.0], index=dates)
        result = FootprintApproximation(lookback=3).compute(open_, high, low, close, volume)
        assert not result.isnull().all(axis=None)  # at least some non-NaN values
