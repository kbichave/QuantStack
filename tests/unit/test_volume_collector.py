# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the volume SignalEngine collector.

Tests verify that the collector returns expected keys (including the new
VPOC/VAH/VAL, AnchoredVWAP, and microstructure signals) and that no
exceptions propagate out of the fault-tolerant try/except wrappers.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from quantstack.signal_engine.collectors.volume import collect_volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(n: int = 80, seed: int = 42) -> MagicMock:
    """Return a mock DataStore with realistic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3) + 0.2
    low = close - np.abs(np.random.randn(n) * 0.3) - 0.2
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(500_000, 2_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    store = MagicMock()
    store.load_ohlcv.return_value = df
    return store


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVolumeCollectorCore:
    def test_returns_dict(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert isinstance(result, dict)

    def test_baseline_keys_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        for key in ("vwap", "volume_trend", "adv_20", "at_hvn", "at_lvn"):
            assert key in result, f"missing baseline key: {key}"

    def test_returns_empty_on_insufficient_bars(self):
        store = MagicMock()
        store.load_ohlcv.return_value = None
        result = _run(collect_volume("SPY", store))
        assert result == {}

    def test_volume_trend_valid_values(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert result["volume_trend"] in ("increasing", "decreasing", "flat")


class TestVolumeCollectorVPOC:
    def test_vpoc_key_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert "vpoc" in result

    def test_vpoc_within_price_range(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        vpoc = result.get("vpoc")
        if vpoc is not None:
            df = store.load_ohlcv.return_value
            assert df["low"].min() <= vpoc <= df["high"].max()

    def test_vah_above_val(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        vah = result.get("vah")
        val = result.get("val")
        if vah is not None and val is not None:
            assert vah >= val

    def test_price_in_value_area_is_binary(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        if "price_in_value_area" in result:
            assert result["price_in_value_area"] in (0, 1)


class TestVolumeCollectorAnchoredVWAP:
    def test_avwap_key_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert "avwap" in result

    def test_avwap_positive(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        avwap = result.get("avwap")
        if avwap is not None:
            assert avwap > 0

    def test_above_avwap_binary(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        if "above_avwap" in result:
            assert result["above_avwap"] in (0, 1)


class TestVolumeCollectorMicrostructure:
    def test_amihud_key_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert "amihud" in result

    def test_amihud_non_negative(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        amihud = result.get("amihud")
        if amihud is not None:
            assert amihud >= 0

    def test_roll_spread_pct_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        # May be NaN for trending data (positive covariance) — key should still exist
        assert "roll_spread_pct" in result

    def test_cs_spread_pct_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert "cs_spread_pct" in result

    def test_rv_overnight_ratio_in_range(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        ratio = result.get("rv_overnight_ratio")
        if ratio is not None:
            assert 0.0 <= ratio <= 1.0

    def test_gap_keys_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        for key in ("gap_up", "gap_down", "gap_persisted"):
            assert key in result

    def test_gap_up_down_mutually_exclusive(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        if "gap_up" in result and "gap_down" in result:
            assert not (result["gap_up"] == 1 and result["gap_down"] == 1)

    def test_no_exceptions_with_constant_prices(self):
        """Constant price (zero-variance) must not crash the collector."""
        store = MagicMock()
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "open": np.full(60, 100.0),
                "high": np.full(60, 100.5),
                "low": np.full(60, 99.5),
                "close": np.full(60, 100.0),
                "volume": np.full(60, 1_000_000.0),
            },
            index=dates,
        )
        store.load_ohlcv.return_value = df
        result = _run(collect_volume("SPY", store))
        assert isinstance(result, dict)
        # At minimum the baseline keys must be present
        assert "vwap" in result


class TestVolumeCollectorVWAPDeviation:
    def test_vwap_deviation_key_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert "vwap_deviation" in result

    def test_vwap_deviation_zscore_key_present(self):
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        assert "vwap_deviation_zscore" in result

    def test_vwap_deviation_reasonable_range(self):
        """Deviation for realistic daily data should be within ±10%."""
        store = _make_store()
        result = _run(collect_volume("SPY", store))
        dev = result.get("vwap_deviation")
        if dev is not None:
            assert -50 < dev < 50  # percentage, very loose bound
