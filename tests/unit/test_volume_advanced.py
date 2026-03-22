# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for VolumePointOfControl (VPOC/VAH/VAL) and AnchoredVWAP."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.volume import AnchoredVWAP, VolumePointOfControl


@pytest.fixture
def ohlcv():
    """100-bar daily OHLCV with volume."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.4) + 0.3
    low = close - np.abs(np.random.randn(100) * 0.4) - 0.3
    volume = np.random.randint(100_000, 1_000_000, 100).astype(float)
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# VolumePointOfControl
# ---------------------------------------------------------------------------


class TestVolumePointOfControl:
    def test_returns_dataframe(self, ohlcv):
        result = VolumePointOfControl().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = VolumePointOfControl().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert {"vpoc", "vah", "val", "above_vah", "below_val", "in_value"}.issubset(
            set(result.columns)
        )

    def test_val_below_vpoc_below_vah(self, ohlcv):
        result = VolumePointOfControl().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result.dropna()
        assert (valid["val"] <= valid["vpoc"]).all()
        assert (valid["vpoc"] <= valid["vah"]).all()

    def test_binary_flag_columns(self, ohlcv):
        result = VolumePointOfControl().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        for col in ("above_vah", "below_val", "in_value"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_vpoc_within_price_range(self, ohlcv):
        result = VolumePointOfControl().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result.dropna()
        price_lo = ohlcv["low"].expanding().min()
        price_hi = ohlcv["high"].expanding().max()
        assert (valid["vpoc"] >= price_lo.loc[valid.index]).all()
        assert (valid["vpoc"] <= price_hi.loc[valid.index]).all()

    def test_warmup_bars_are_nan(self, ohlcv):
        lookback = 20
        result = VolumePointOfControl(lookback=lookback).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["vpoc"].iloc[: lookback - 1].isna().all()

    def test_vah_wider_value_area(self, ohlcv):
        """Wider value area (90%) → VAH/VAL spread should be >= default (68.2%)."""
        r_default = VolumePointOfControl(value_area_pct=0.682).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        r_wide = VolumePointOfControl(value_area_pct=0.90).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        spread_default = (r_default["vah"] - r_default["val"]).dropna().mean()
        spread_wide = (r_wide["vah"] - r_wide["val"]).dropna().mean()
        assert spread_wide >= spread_default

    def test_in_value_area_flag_correct(self, ohlcv):
        result = VolumePointOfControl().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result.dropna()
        close_valid = ohlcv["close"].loc[valid.index]
        expected_in_value = (
            (close_valid >= valid["val"]) & (close_valid <= valid["vah"])
        ).astype(int)
        pd.testing.assert_series_equal(
            valid["in_value"], expected_in_value, check_names=False
        )


# ---------------------------------------------------------------------------
# AnchoredVWAP
# ---------------------------------------------------------------------------


class TestAnchoredVWAP:
    def test_returns_dataframe(self, ohlcv):
        result = AnchoredVWAP(anchor=0).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = AnchoredVWAP(anchor=0).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert {"avwap", "avwap_deviation", "above_avwap"}.issubset(set(result.columns))

    def test_no_nan_after_anchor(self, ohlcv):
        result = AnchoredVWAP(anchor=10).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["avwap"].iloc[10:].notna().all()

    def test_nan_before_anchor(self, ohlcv):
        result = AnchoredVWAP(anchor=10).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["avwap"].iloc[:10].isna().all()

    def test_anchor_zero_no_nans(self, ohlcv):
        result = AnchoredVWAP(anchor=0).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["avwap"].notna().all()

    def test_avwap_within_high_low(self, ohlcv):
        result = AnchoredVWAP(anchor=0).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        # VWAP should be within the overall price range
        assert (result["avwap"] >= ohlcv["low"].min() * 0.99).all()
        assert (result["avwap"] <= ohlcv["high"].max() * 1.01).all()

    def test_above_avwap_binary(self, ohlcv):
        result = AnchoredVWAP(anchor=0).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        vals = result["above_avwap"].unique()
        assert set(vals).issubset({0, 1})

    def test_deviation_zero_when_close_equals_avwap(self):
        """Single-bar anchor: VWAP = typical price; with same open/close, deviation = 0."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates)
        high = pd.Series([101.0, 101.0, 101.0, 101.0, 101.0], index=dates)
        low = pd.Series([99.0, 99.0, 99.0, 99.0, 99.0], index=dates)
        vol = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 1000.0], index=dates)
        # Typical = (101 + 99 + 100) / 3 = 100; VWAP = 100; deviation = 0
        result = AnchoredVWAP(anchor=0).compute(high, low, close, vol)
        assert (result["avwap_deviation"].abs() < 1e-9).all()

    def test_anchor_by_datetime(self, ohlcv):
        """Anchor by timestamp should resolve correctly."""
        anchor_ts = ohlcv.index[20]
        result = AnchoredVWAP(anchor=anchor_ts).compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["avwap"].iloc[:20].isna().all()
        assert result["avwap"].iloc[20:].notna().all()

    def test_accumulation_monotone_volume_weighting(self):
        """With equal-price bars, VWAP should equal that constant price regardless of volume."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        close = pd.Series(np.full(20, 100.0), index=dates)
        high = pd.Series(np.full(20, 101.0), index=dates)
        low = pd.Series(np.full(20, 99.0), index=dates)
        vol = pd.Series(np.random.randint(100, 10000, 20).astype(float), index=dates)
        result = AnchoredVWAP(anchor=0).compute(high, low, close, vol)
        # Typical price = 100; VWAP should always = 100
        assert (result["avwap"] - 100.0).abs().max() < 1e-9
