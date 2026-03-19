# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for order flow approximations: CVD, VPIN, Hawkes."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.flow import CumulativeVolumeDelta, HawkesIntensity, VPIN


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
