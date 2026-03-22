# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the technical SignalEngine collector.

Tests verify key output contracts — presence of expected signal keys
and value-range invariants — using a mock DataStore.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from quantstack.signal_engine.collectors.technical import collect_technical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(n: int = 300, seed: int = 7) -> MagicMock:
    """Return a mock DataStore with realistic OHLCV data (daily + weekly)."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.8)
    high = close + np.abs(np.random.randn(n) * 0.4) + 0.3
    low = close - np.abs(np.random.randn(n) * 0.4) - 0.3
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)

    daily_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    # Weekly: ~52 bars
    w_dates = pd.date_range("2022-01-03", periods=60, freq="W")
    wc = 100.0 + np.cumsum(np.random.randn(60) * 1.5)
    weekly_df = pd.DataFrame(
        {
            "open": np.roll(wc, 1),
            "high": wc + 1.0,
            "low": wc - 1.0,
            "close": wc,
            "volume": np.full(60, 5_000_000.0),
        },
        index=w_dates,
    )

    store = MagicMock()
    store.load_ohlcv.side_effect = lambda sym, tf: (
        weekly_df if str(tf) == "1W" else daily_df
    )
    return store


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------


class TestTechnicalCollectorCore:
    def test_returns_dict(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert isinstance(result, dict)

    def test_empty_on_insufficient_bars(self):
        store = MagicMock()
        store.load_ohlcv.return_value = None
        result = _run(collect_technical("SPY", store))
        assert result == {}

    def test_baseline_momentum_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        for key in ("rsi_14", "macd_hist", "adx_14"):
            assert key in result, f"missing: {key}"

    def test_weekly_trend_valid_value(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert result.get("weekly_trend") in (
            "bullish",
            "bearish",
            "neutral",
            "unknown",
        )


# ---------------------------------------------------------------------------
# Phase 1 — trend indicators
# ---------------------------------------------------------------------------


class TestTechnicalCollectorTrend:
    def test_supertrend_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "supertrend" in result
        assert "st_direction" in result
        assert result["st_direction"] in (-1, 1)

    def test_ichimoku_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        for key in ("tenkan_sen", "kijun_sen", "cloud_bullish", "price_above_cloud"):
            assert key in result, f"missing ichimoku key: {key}"

    def test_hma_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "hma" in result
        assert "hma_uptrend" in result
        assert result["hma_uptrend"] in (0, 1)

    def test_pct_r_exhaustion_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "exhaustion_top" in result
        assert "exhaustion_bottom" in result

    def test_williams_vix_fix_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "wvf" in result
        assert "wvf_extreme" in result
        assert result["wvf_extreme"] in (0, 1)


# ---------------------------------------------------------------------------
# Phase 3 — ICT Smart Money
# ---------------------------------------------------------------------------


class TestTechnicalCollectorICT:
    def test_fvg_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "bullish_fvg" in result
        assert "bearish_fvg" in result

    def test_order_block_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "bullish_ob" in result
        assert "bearish_ob" in result

    def test_structure_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        for key in ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish"):
            assert key in result

    def test_breaker_block_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "bullish_breaker" in result
        assert "bearish_breaker" in result

    def test_mmxm_phase_valid(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "mmxm_phase" in result
        assert result.get("mmxm_phase") in (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Order flow + momentum
# ---------------------------------------------------------------------------


class TestTechnicalCollectorOrderFlow:
    def test_laguerre_rsi_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "lrsi" in result
        if result["lrsi"] is not None:
            assert 0.0 <= result["lrsi"] <= 1.0

    def test_dual_momentum_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "momentum_12m1m" in result
        assert "abs_momentum_signal" in result
        if result["abs_momentum_signal"] is not None:
            assert result["abs_momentum_signal"] in (0, 1)

    def test_cvd_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "cvd" in result

    def test_hawkes_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "hawkes_intensity" in result
        assert "hawkes_excited" in result

    def test_koncorde_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "koncorde_green" in result
        assert "koncorde_agreement" in result

    def test_vpin_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "vpin" in result

    def test_footprint_keys(self):
        result = _run(collect_technical("SPY", _make_store()))
        assert "fp_bar_delta" in result
        assert "fp_imbalanced_bull" in result
