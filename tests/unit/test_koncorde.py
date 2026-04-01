# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Koncorde 6-component composite indicator."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.koncorde import Koncorde


@pytest.fixture
def ohlcv():
    """300-bar daily OHLCV (longer for NVI normalisation warmup)."""
    dates = pd.date_range(start="2022-01-01", periods=300, freq="D")
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5), index=dates)
    high = close + np.abs(np.random.randn(300) * 0.4) + 0.3
    low = close - np.abs(np.random.randn(300) * 0.4) - 0.3
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(
        np.random.randint(500_000, 2_000_000, 300).astype(float), index=dates
    )
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestKoncorde:
    def test_returns_dataframe(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        expected = {
            "rsi",
            "mfi",
            "bb_pct_b",
            "stochastic",
            "pvi",
            "nvi",
            "nvi_signal",
            "green_line",
            "blue_line",
            "green_positive",
            "blue_positive",
            "agreement",
            "divergence",
        }
        assert expected.issubset(set(result.columns))

    def test_same_length(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert len(result) == len(ohlcv)

    def test_rsi_in_range(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_mfi_in_range(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result["mfi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_blue_line_in_range(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result["blue_line"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_green_line_in_range_after_warmup(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        valid = result["green_line"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_binary_positive_columns(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        for col in ("green_positive", "blue_positive", "divergence"):
            vals = result[col].unique()
            assert set(vals).issubset({0, 1})

    def test_agreement_in_neg1_0_1(self, ohlcv):
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        vals = result["agreement"].unique()
        assert set(vals).issubset({-1, 0, 1})

    def test_pvi_nvi_start_equal(self, ohlcv):
        """Both PVI and NVI initialise at 1000."""
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["pvi"].iloc[0] == pytest.approx(1000.0)
        assert result["nvi"].iloc[0] == pytest.approx(1000.0)

    def test_pvi_changes_on_high_volume_days(self, ohlcv):
        """PVI changes exactly when volume[i] > volume[i-1]."""
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        pvi = result["pvi"]
        vol = ohlcv["volume"]
        high_vol_mask = vol > vol.shift(1)
        pvi_changed = pvi != pvi.shift(1)
        # Where volume is NOT higher, PVI should NOT change (except bar 0)
        low_vol = ~high_vol_mask
        low_vol.iloc[0] = False  # skip first bar
        assert (pvi_changed & low_vol).sum() == 0

    def test_nvi_changes_on_low_volume_days(self, ohlcv):
        """NVI changes exactly when volume[i] < volume[i-1]."""
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        nvi = result["nvi"]
        vol = ohlcv["volume"]
        low_vol_mask = vol < vol.shift(1)
        nvi_changed = nvi != nvi.shift(1)
        # Where volume is NOT lower, NVI should NOT change
        not_low_vol = ~low_vol_mask
        not_low_vol.iloc[0] = False
        assert (nvi_changed & not_low_vol).sum() == 0

    def test_divergence_mutual_exclusion_with_agreement(self, ohlcv):
        """A bar cannot be both in agreement and divergence simultaneously."""
        result = Koncorde().compute(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        both = (result["agreement"] != 0) & (result["divergence"] == 1)
        assert both.sum() == 0

    def test_no_crash_short_series(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        np.random.seed(0)
        c = pd.Series(100 + np.cumsum(np.random.randn(30) * 0.5), index=dates)
        h = c + 0.5
        lo = c - 0.5
        v = pd.Series(
            np.random.randint(100_000, 500_000, 30).astype(float), index=dates
        )
        result = Koncorde().compute(h, lo, c, v)
        assert len(result) == 30

    def test_trending_market_blue_line_high(self):
        """Steadily rising price + noisy volume → RSI/MFI/Stoch elevated → blue_line > 50."""
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        np.random.seed(5)
        close = pd.Series(
            np.linspace(100, 200, n) + np.random.randn(n) * 0.3, index=dates
        )
        high = close + 1.0
        low = close - 0.5
        # Volume must vary so MFI has both positive and negative flows
        volume = pd.Series(
            1_000_000.0 + np.random.randn(n) * 200_000, index=dates
        ).clip(lower=100_000)
        result = Koncorde().compute(high, low, close, volume)
        valid = result["blue_line"].dropna()
        # In a strong uptrend, majority of bars should show blue > 50
        assert len(valid) > 0
        assert (valid > 50).sum() / len(valid) > 0.5
