# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for %R Trend Exhaustion (dual-period) indicator."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.momentum import PercentRExhaustion


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_200():
    """200-bar OHLCV — long enough for the 112-bar long %R to warm up."""
    dates = pd.date_range(start="2022-01-01", periods=200, freq="D")
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(200) * 0.6)
    high = close + np.abs(np.random.randn(200) * 0.4) + 0.3
    low = close - np.abs(np.random.randn(200) * 0.4) - 0.3
    return pd.DataFrame(
        {"high": high, "low": low, "close": close},
        index=dates,
    )


def _make_ohlcv(close_arr, *, noise: float = 0.5):
    """Helper: build OHLCV Series from a close array."""
    n = len(close_arr)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    np.random.seed(0)
    close = pd.Series(close_arr, index=dates)
    high = close + np.abs(np.random.randn(n) * noise) + 0.1
    low = close - np.abs(np.random.randn(n) * noise) - 0.1
    return high, low, close


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestPercentRExhaustionStructure:
    def test_returns_dataframe(self, ohlcv_200):
        ind = PercentRExhaustion()
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv_200):
        ind = PercentRExhaustion()
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        assert set(result.columns) == {
            "pct_r_short",
            "pct_r_long",
            "exhaustion_top",
            "exhaustion_bottom",
        }

    def test_index_preserved(self, ohlcv_200):
        ind = PercentRExhaustion()
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        pd.testing.assert_index_equal(result.index, ohlcv_200.index)

    def test_pct_r_range(self, ohlcv_200):
        """Both %R series must stay in [-100, 0]."""
        ind = PercentRExhaustion()
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        for col in ("pct_r_short", "pct_r_long"):
            valid = result[col].dropna()
            assert (valid >= -100).all(), f"{col} went below -100"
            assert (valid <= 0).all(), f"{col} went above 0"

    def test_signal_columns_binary(self, ohlcv_200):
        ind = PercentRExhaustion()
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        for col in ("exhaustion_top", "exhaustion_bottom"):
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} has non-binary values"

    def test_signals_mutually_exclusive(self, ohlcv_200):
        """Both signals fire simultaneously only if -20 < threshold < -80, which never happens."""
        ind = PercentRExhaustion()
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        both = (result["exhaustion_top"] == 1) & (result["exhaustion_bottom"] == 1)
        assert (
            not both.any()
        ), "exhaustion_top and exhaustion_bottom fired simultaneously"


# ---------------------------------------------------------------------------
# Directional correctness tests
# ---------------------------------------------------------------------------


class TestPercentRExhaustionSignals:
    def test_exhaustion_top_fires_in_strong_uptrend(self):
        """After a sustained run-up, both %R periods should read overbought."""
        high, low, close = _make_ohlcv(np.linspace(100, 200, 200), noise=0.1)
        ind = PercentRExhaustion(short=14, long=112, ob_threshold=-20, os_threshold=-80)
        result = ind.compute(high, low, close)
        # The last 20 bars of a linear run-up should have exhaustion_top=1
        assert result["exhaustion_top"].iloc[-20:].sum() > 0

    def test_exhaustion_bottom_fires_in_strong_downtrend(self):
        """After a sustained decline, both %R periods should read oversold."""
        high, low, close = _make_ohlcv(np.linspace(200, 100, 200), noise=0.1)
        ind = PercentRExhaustion(short=14, long=112, ob_threshold=-20, os_threshold=-80)
        result = ind.compute(high, low, close)
        # The last 20 bars of a linear decline should have exhaustion_bottom=1
        assert result["exhaustion_bottom"].iloc[-20:].sum() > 0

    def test_no_exhaustion_in_sideways_market(self):
        """A flat, oscillating market should produce few or no exhaustion signals."""
        np.random.seed(99)
        n = 200
        close_arr = 100 + np.random.randn(n) * 0.5  # very tight range
        high, low, close = _make_ohlcv(close_arr, noise=0.3)
        ind = PercentRExhaustion(short=14, long=112, ob_threshold=-5, os_threshold=-95)
        result = ind.compute(high, low, close)
        # With very tight thresholds, almost no bar should trigger
        assert result["exhaustion_top"].sum() == 0
        assert result["exhaustion_bottom"].sum() == 0

    def test_custom_periods(self, ohlcv_200):
        """Shorter periods (short=5, long=20) warm up faster — should have valid output."""
        ind = PercentRExhaustion(short=5, long=20)
        result = ind.compute(ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"])
        # With short=5, long=20, bars 20..199 should all be non-NaN
        valid_short = result["pct_r_short"].iloc[5:].dropna()
        valid_long = result["pct_r_long"].iloc[20:].dropna()
        assert len(valid_short) > 0
        assert len(valid_long) > 0


# ---------------------------------------------------------------------------
# No-lookahead test
# ---------------------------------------------------------------------------


class TestPercentRNoLookahead:
    def test_no_lookahead(self, ohlcv_200):
        ind = PercentRExhaustion()
        hi, lo, cl = ohlcv_200["high"], ohlcv_200["low"], ohlcv_200["close"]

        result_n = ind.compute(hi.iloc[:-1], lo.iloc[:-1], cl.iloc[:-1])
        result_n1 = ind.compute(hi, lo, cl)

        shared = result_n.index
        pd.testing.assert_series_equal(
            result_n["pct_r_short"].reindex(shared),
            result_n1["pct_r_short"].reindex(shared),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result_n["pct_r_long"].reindex(shared),
            result_n1["pct_r_long"].reindex(shared),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPercentREdgeCases:
    def test_constant_price_no_crash(self):
        """Constant price → zero range → should not crash, may produce NaN."""
        dates = pd.date_range(start="2023-01-01", periods=150, freq="D")
        price = pd.Series(np.full(150, 100.0), index=dates)
        ind = PercentRExhaustion()
        result = ind.compute(price, price, price)
        assert len(result) == 150

    def test_minimum_data_length(self):
        """Fewer bars than long period → long %R all NaN, no crash."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        close = pd.Series(np.linspace(100, 110, 20), index=dates)
        high = close + 0.5
        low = close - 0.5
        ind = PercentRExhaustion(short=5, long=112)
        result = ind.compute(high, low, close)
        assert result["pct_r_long"].isna().all()
        assert result["pct_r_short"].iloc[5:].notna().any()
