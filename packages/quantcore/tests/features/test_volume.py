# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for volume indicators: VolumePointOfControl (VPOC/VAH/VAL),
AnchoredVWAP.

Tests verify correctness properties (no lookahead, value area logic,
VWAP tracking) following the plan's testing specification.
"""

import numpy as np
import pandas as pd

from quantcore.features.volume import AnchoredVWAP, VolumePointOfControl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(
    n: int = 100, seed: int = 42
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Synthetic (high, low, close, volume) bars."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.8), index=dates).clip(50)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    volume = pd.Series(np.random.randint(100_000, 1_000_000, n).astype(float), index=dates)
    return high, low, close, volume


# ---------------------------------------------------------------------------
# VolumePointOfControl
# ---------------------------------------------------------------------------


class TestVolumePointOfControl:
    def test_returns_dataframe(self):
        hi, lo, cl, vol = _ohlcv()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl, vol = _ohlcv()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        for col in ("vpoc", "vah", "val", "above_vah", "below_val", "in_value"):
            assert col in result.columns, f"{col} missing"

    def test_vah_above_val(self):
        """Value Area High must be >= Value Area Low."""
        hi, lo, cl, vol = _ohlcv()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        valid = result.dropna(subset=["vah", "val"])
        if len(valid) > 0:
            assert (valid["vah"] >= valid["val"]).all()

    def test_vpoc_within_range(self):
        """VPOC should be between low and high of the lookback window."""
        hi, lo, cl, vol = _ohlcv()
        result = VolumePointOfControl(lookback=20).compute(hi, lo, cl, vol)
        valid = result.dropna(subset=["vpoc"])
        if len(valid) > 0:
            min_price = lo.min()
            max_price = hi.max()
            assert (valid["vpoc"] >= min_price - 1.0).all()
            assert (valid["vpoc"] <= max_price + 1.0).all()

    def test_zone_columns_binary(self):
        hi, lo, cl, vol = _ohlcv()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        for col in ("above_vah", "below_val", "in_value"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_zone_mutually_exclusive(self):
        """Where vpoc is computed, bar should be in exactly one zone."""
        hi, lo, cl, vol = _ohlcv()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        # Only check bars where the profile is computed (vpoc not NaN)
        valid = result.dropna(subset=["vpoc", "above_vah", "below_val", "in_value"])
        if len(valid) > 0:
            zone_sum = valid["above_vah"] + valid["below_val"] + valid["in_value"]
            assert (zone_sum == 1).all(), "Zone columns are not mutually exclusive"

    def test_no_lookahead(self):
        hi, lo, cl, vol = _ohlcv(n=60)
        r1 = VolumePointOfControl(lookback=10).compute(hi, lo, cl, vol)
        extra_idx = hi.index[-1] + pd.Timedelta("1D")
        hi2 = pd.concat([hi, pd.Series([hi.iloc[-1]], index=[extra_idx])])
        lo2 = pd.concat([lo, pd.Series([lo.iloc[-1]], index=[extra_idx])])
        cl2 = pd.concat([cl, pd.Series([cl.iloc[-1] * 1.02], index=[extra_idx])])
        vol2 = pd.concat([vol, pd.Series([vol.iloc[-1] * 3], index=[extra_idx])])
        r2 = VolumePointOfControl(lookback=10).compute(hi2, lo2, cl2, vol2)
        pd.testing.assert_series_equal(
            r1["vpoc"].dropna(),
            r2["vpoc"].iloc[: len(r1)].dropna(),
            check_names=False, check_freq=False, rtol=1e-5,
        )

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        vol = pd.Series([500_000.0])
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# AnchoredVWAP
# ---------------------------------------------------------------------------


class TestAnchoredVWAP:
    def test_returns_dataframe(self):
        hi, lo, cl, vol = _ohlcv()
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl, vol = _ohlcv()
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        for col in ("avwap", "avwap_deviation", "above_avwap"):
            assert col in result.columns, f"{col} missing"

    def test_above_avwap_binary(self):
        hi, lo, cl, vol = _ohlcv()
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        vals = result["above_avwap"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_avwap_tracks_price(self):
        """AVWAP should be within a reasonable range of the price series."""
        hi, lo, cl, vol = _ohlcv()
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        valid = result["avwap"].dropna()
        if len(valid) > 0:
            assert valid.min() > cl.min() * 0.5
            assert valid.max() < cl.max() * 2.0

    def test_avwap_at_anchor_is_typical_price(self):
        """At the anchor bar, AVWAP should equal the typical price."""
        hi, lo, cl, vol = _ohlcv()
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        typical = (hi.iloc[0] + lo.iloc[0] + cl.iloc[0]) / 3.0
        assert abs(result["avwap"].iloc[0] - typical) < 0.01

    def test_datetime_anchor(self):
        """Anchor by datetime index value."""
        hi, lo, cl, vol = _ohlcv()
        anchor_date = hi.index[10]
        result = AnchoredVWAP(anchor=anchor_date).compute(hi, lo, cl, vol)
        valid = result["avwap"].dropna()
        assert result["avwap"].iloc[:10].isna().all()
        assert len(valid) > 0

    def test_deviation_sign_matches_position(self):
        """When close > avwap, deviation should be positive and above_avwap=1."""
        hi, lo, cl, vol = _ohlcv()
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        valid = result.dropna(subset=["avwap", "avwap_deviation", "above_avwap"])
        if len(valid) > 0:
            above = valid[valid["above_avwap"] == 1]
            if len(above) > 0:
                assert (above["avwap_deviation"] > -0.01).all()

    def test_single_bar_no_crash(self):
        hi = pd.Series([101.0])
        lo = pd.Series([99.0])
        cl = pd.Series([100.0])
        vol = pd.Series([500_000.0])
        result = AnchoredVWAP(anchor=0).compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)
