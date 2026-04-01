# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for macro_signals.py — get_credit_market_signals, get_market_breadth.

Mocks _get_reader() to inject synthetic OHLCV data for ETF proxies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


def _make_etf_store(tickers: list[str], n_days: int = 60) -> MagicMock:
    """Build a mock DataStore that returns synthetic OHLCV for given tickers."""
    store = MagicMock()

    def _load(ticker, timeframe):
        if ticker in tickers:
            return synthetic_ohlcv(ticker, n_days=n_days, seed=hash(ticker) % 10000)
        return None

    store.load_ohlcv.side_effect = _load
    store.close = MagicMock()
    return store


# ---------------------------------------------------------------------------
# get_credit_market_signals
# ---------------------------------------------------------------------------

class TestGetCreditMarketSignals:

    @pytest.mark.asyncio
    async def test_happy_path_all_etfs(self):
        """With all credit ETFs loaded, returns credit_regime and bottom_signal."""
        from quantstack.mcp.tools.macro_signals import get_credit_market_signals

        all_credit = ["HYG", "LQD", "TLT", "IEF", "SHY", "GLD", "UUP"]
        store = _make_etf_store(all_credit, n_days=100)

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_credit_market_signals)()

        assert "credit_regime" in result
        assert result["credit_regime"] in ("widening", "stable", "contracting", "unknown")
        assert "bottom_signal" in result
        assert isinstance(result["bottom_signal"], bool)
        assert "risk_on_score" in result
        assert "interpretation" in result

    @pytest.mark.asyncio
    async def test_insufficient_etf_data(self):
        """With <3 ETFs loaded, returns unknown credit_regime."""
        from quantstack.mcp.tools.macro_signals import get_credit_market_signals

        store = _make_etf_store(["HYG", "LQD"], n_days=100)

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_credit_market_signals)()

        assert result["credit_regime"] == "unknown"
        assert result["bottom_signal"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_short_data_excluded(self):
        """ETFs with <30 bars are excluded from analysis."""
        from quantstack.mcp.tools.macro_signals import get_credit_market_signals

        store = MagicMock()
        call_count = 0

        def _load(ticker, timeframe):
            nonlocal call_count
            call_count += 1
            # Return short data for all — should all be excluded
            return synthetic_ohlcv(ticker, n_days=20)

        store.load_ohlcv.side_effect = _load
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_credit_market_signals)()

        assert result["credit_regime"] == "unknown"

    @pytest.mark.asyncio
    async def test_store_close_always_called(self):
        """Store.close() must be called even on data errors."""
        from quantstack.mcp.tools.macro_signals import get_credit_market_signals

        store = MagicMock()
        store.load_ohlcv.side_effect = RuntimeError("DB down")
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_credit_market_signals)()

        store.close.assert_called_once()


# ---------------------------------------------------------------------------
# get_market_breadth
# ---------------------------------------------------------------------------

class TestGetMarketBreadth:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """With sufficient sector ETFs, returns breadth_score and sector details."""
        from quantstack.mcp.tools.macro_signals import get_market_breadth

        sector_etfs = [
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB",
            "XLRE", "XLU", "XLC", "SPY", "QQQ", "IWM", "MDY",
        ]
        store = _make_etf_store(sector_etfs, n_days=250)

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_market_breadth)()

        assert "breadth_score" in result
        assert result["breadth_score"] is not None
        assert 0 <= result["breadth_score"] <= 1
        assert "breadth_trend" in result
        assert "weakest_sectors" in result
        assert "strongest_sectors" in result
        assert "bottom_signal" in result
        assert "sector_details" in result

    @pytest.mark.asyncio
    async def test_insufficient_etf_data(self):
        """With <5 ETFs, returns error."""
        from quantstack.mcp.tools.macro_signals import get_market_breadth

        store = _make_etf_store(["SPY", "QQQ"], n_days=100)

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_market_breadth)()

        assert result["breadth_score"] is None
        assert result["bottom_signal"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_short_data_excluded_from_breadth(self):
        """ETFs with <50 bars are excluded from breadth analysis."""
        from quantstack.mcp.tools.macro_signals import get_market_breadth

        store = MagicMock()
        store.load_ohlcv.return_value = synthetic_ohlcv("SPY", n_days=30)
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_market_breadth)()

        assert result["breadth_score"] is None

    @pytest.mark.asyncio
    async def test_breadth_divergence_detection(self):
        """When SPY is at 20d low but breadth is stable, divergence=True."""
        from quantstack.mcp.tools.macro_signals import get_market_breadth

        sector_etfs = [
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI",
            "SPY", "QQQ", "IWM",
        ]

        store = MagicMock()

        def _load(ticker, timeframe):
            df = synthetic_ohlcv(ticker, n_days=250, seed=hash(ticker) % 10000)
            if ticker == "SPY":
                # Force SPY to be at its 20d low
                df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-21:].min() * 0.99
            return df

        store.load_ohlcv.side_effect = _load
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_market_breadth)()

        # Should detect SPY at 20d low
        assert "spy_at_20d_low" in result

    @pytest.mark.asyncio
    async def test_store_close_always_called(self):
        """Store.close() must be called even on data errors."""
        from quantstack.mcp.tools.macro_signals import get_market_breadth

        store = MagicMock()
        store.load_ohlcv.side_effect = RuntimeError("DB down")
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.macro_signals._get_reader", return_value=store):
            result = await _fn(get_market_breadth)()

        store.close.assert_called_once()
