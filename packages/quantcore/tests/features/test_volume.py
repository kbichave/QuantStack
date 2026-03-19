# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for volume indicators: VolumePointOfControl, AnchoredVWAP.
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.volume import AnchoredVWAP, VolumePointOfControl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bars(n: int = 80, seed: int = 11) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.6), index=dates).clip(50)
    high = close + np.abs(np.random.randn(n) * 0.4)
    low = close - np.abs(np.random.randn(n) * 0.4)
    volume = pd.Series(np.random.randint(100_000, 500_000, n).astype(float), index=dates)
    return high, low, close, volume


# ---------------------------------------------------------------------------
# VolumePointOfControl (VPOC / VAH / VAL)
# ---------------------------------------------------------------------------


class TestVolumePointOfControl:
    def test_returns_dataframe(self):
        hi, lo, cl, vol = _bars()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl, vol = _bars()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        for col in ("vpoc", "vah", "val", "above_vah", "below_val", "in_value"):
            assert col in result.columns, f"{col} missing"

    def test_binary_flags(self):
        hi, lo, cl, vol = _bars()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        for col in ("above_vah", "below_val", "in_value"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} not binary"

    def test_vpoc_within_historical_range(self):
        """
        VPOC is the rolling-window price with highest volume. It must fall within
        the cumulative (running) min of low and max of high seen so far.
        """
        hi, lo, cl, vol = _bars()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        valid_vpoc = result["vpoc"].dropna()
        running_lo = lo.expanding().min()
        running_hi = hi.expanding().max()
        # Align
        common_idx = valid_vpoc.index
        assert (valid_vpoc >= running_lo.loc[common_idx]).all()
        assert (valid_vpoc <= running_hi.loc[common_idx]).all()

    def test_vah_above_val(self):
        """Value Area High must be >= Value Area Low at every valid row."""
        hi, lo, cl, vol = _bars()
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        valid = result[["vah", "val"]].dropna()
        assert (valid["vah"] >= valid["val"]).all()

    def test_vpoc_attracts_price(self):
        """
        Build a synthetic series where the highest volume concentrates at 105.
        VPOC over the full window should be close to 105.
        """
        n = 50
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        # Uniform prices near 105
        cl = pd.Series(np.full(n, 105.0), index=dates)
        hi = cl + 0.5
        lo = cl - 0.5
        # Heavy volume at bars 20-29
        vol_arr = np.ones(n) * 100_000.0
        vol_arr[20:30] = 1_000_000.0
        vol = pd.Series(vol_arr, index=dates)
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        vpoc = result["vpoc"].dropna().iloc[-1]
        assert abs(vpoc - 105.0) < 2.0  # Within 2 dollars of the concentrated price

    def test_no_crash_short_series(self):
        hi = pd.Series([101.0, 102.0, 100.0])
        lo = pd.Series([99.0, 100.0, 98.0])
        cl = pd.Series([100.0, 101.0, 99.0])
        vol = pd.Series([100_000.0, 200_000.0, 150_000.0])
        result = VolumePointOfControl().compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# AnchoredVWAP
# ---------------------------------------------------------------------------


class TestAnchoredVWAP:
    def test_returns_dataframe(self):
        hi, lo, cl, vol = _bars()
        result = AnchoredVWAP().compute(hi, lo, cl, vol)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        hi, lo, cl, vol = _bars()
        result = AnchoredVWAP().compute(hi, lo, cl, vol)
        for col in ("avwap", "avwap_deviation", "above_avwap"):
            assert col in result.columns, f"{col} missing"

    def test_above_avwap_binary(self):
        hi, lo, cl, vol = _bars()
        result = AnchoredVWAP().compute(hi, lo, cl, vol)
        vals = result["above_avwap"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_avwap_at_first_bar(self):
        """At the anchor bar, AVWAP = typical price of that bar."""
        n = 20
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        hi = pd.Series(np.full(n, 102.0), index=dates)
        lo = pd.Series(np.full(n, 98.0), index=dates)
        cl = pd.Series(np.full(n, 100.0), index=dates)
        vol = pd.Series(np.full(n, 100_000.0), index=dates)
        result = AnchoredVWAP().compute(hi, lo, cl, vol)
        # typical price = (102 + 98 + 100) / 3 = 100.0; AVWAP should equal this
        avwap_first = result["avwap"].dropna().iloc[0]
        expected_tp = (102.0 + 98.0 + 100.0) / 3.0
        assert abs(avwap_first - expected_tp) < 1e-4

    def test_avwap_equals_price_for_constant_series(self):
        """For constant-price bars, AVWAP must equal the constant price at every bar."""
        n = 30
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        hi = pd.Series(np.full(n, 101.0), index=dates)
        lo = pd.Series(np.full(n, 99.0), index=dates)
        cl = pd.Series(np.full(n, 100.0), index=dates)
        vol = pd.Series(np.full(n, 200_000.0), index=dates)
        result = AnchoredVWAP().compute(hi, lo, cl, vol)
        expected_tp = (101.0 + 99.0 + 100.0) / 3.0
        avwap_vals = result["avwap"].dropna()
        assert (avwap_vals - expected_tp).abs().max() < 1e-4

    def test_above_avwap_consistent_with_price(self):
        """above_avwap=1 iff close > avwap."""
        hi, lo, cl, vol = _bars()
        result = AnchoredVWAP().compute(hi, lo, cl, vol)
        joined = result[["avwap", "above_avwap"]].copy()
        joined["close"] = cl
        valid = joined.dropna()
        expected = (valid["close"] > valid["avwap"]).astype(int)
        pd.testing.assert_series_equal(
            valid["above_avwap"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_lookahead(self):
        """Adding a future bar must not change historical AVWAP values."""
        hi, lo, cl, vol = _bars(n=60)
        r1 = AnchoredVWAP().compute(hi, lo, cl, vol)
        hi2 = pd.concat([hi, pd.Series([hi.iloc[-1]], index=[hi.index[-1] + pd.Timedelta("1D")])])
        lo2 = pd.concat([lo, pd.Series([lo.iloc[-1]], index=[lo.index[-1] + pd.Timedelta("1D")])])
        cl2 = pd.concat([cl, pd.Series([cl.iloc[-1] * 1.01], index=[cl.index[-1] + pd.Timedelta("1D")])])
        vol2 = pd.concat([vol, pd.Series([vol.iloc[-1]], index=[vol.index[-1] + pd.Timedelta("1D")])])
        r2 = AnchoredVWAP().compute(hi2, lo2, cl2, vol2)
        pd.testing.assert_series_equal(
            r1["avwap"].dropna(),
            r2["avwap"].iloc[: len(r1)].dropna(),
            check_names=False,
            check_freq=False,
            rtol=1e-5,
        )
