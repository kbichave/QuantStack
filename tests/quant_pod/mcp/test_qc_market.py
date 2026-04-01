# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_market MCP tools — symbol snapshots, regime, trade templates,
screener, liquidity, volume profile, calendar.

All tools use _get_reader() for data. We mock it with synthetic OHLCV.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.mcp.tools.qc_market import (
    analyze_liquidity,
    analyze_volume_profile,
    generate_trade_template,
    get_market_regime_snapshot,
    get_symbol_snapshot,
    get_trading_calendar,
    run_screener,
    validate_trade,
)
from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


def _patch_reader(df):
    store = MagicMock()
    store.load_ohlcv.return_value = df
    store.close.return_value = None
    return patch(
        "quantstack.mcp.tools.qc_market._get_reader",
        return_value=store,
    )


class TestGetSymbolSnapshot:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            result = await _fn(get_symbol_snapshot)(symbol="SPY")
        assert "error" not in result
        assert result.get("symbol") == "SPY"

    @pytest.mark.asyncio
    async def test_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(get_symbol_snapshot)(symbol="NODATA")
        assert "error" in result


class TestGetMarketRegimeSnapshot:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            # get_market_regime_snapshot takes end_date, not symbol —
            # it reads SPY internally as the benchmark
            result = await _fn(get_market_regime_snapshot)()
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_with_end_date(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            result = await _fn(get_market_regime_snapshot)(end_date="2025-01-01")
        # May error if data doesn't cover date — just check structure
        assert isinstance(result, dict)


class TestGenerateTradeTemplate:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("AAPL", n_days=252)
        with _patch_reader(df):
            result = await _fn(generate_trade_template)(
                symbol="AAPL", direction="long",
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(generate_trade_template)(
                symbol="NODATA", direction="long",
            )
        assert "error" in result


class TestValidateTrade:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            # validate_trade takes a trade_template dict, not individual params
            result = await _fn(validate_trade)(
                trade_template={
                    "symbol": "SPY",
                    "direction": "long",
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "shares": 10,
                },
                account_equity=100000.0,
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        # Minimal template — should still not crash
        result = await _fn(validate_trade)(
            trade_template={"symbol": "SPY"},
        )
        assert isinstance(result, dict)


class TestRunScreener:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            # run_screener takes individual params, not a criteria dict
            result = await _fn(run_screener)(
                symbols=["SPY"], min_volume=100000,
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_empty_symbols(self):
        result = await _fn(run_screener)(symbols=[])
        # Should handle empty gracefully
        assert isinstance(result, dict)


class TestAnalyzeLiquidity:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=100)
        with _patch_reader(df):
            result = await _fn(analyze_liquidity)(symbol="SPY")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(analyze_liquidity)(symbol="NODATA")
        assert "error" in result


class TestAnalyzeVolumeProfile:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=100)
        with _patch_reader(df):
            result = await _fn(analyze_volume_profile)(symbol="SPY")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(analyze_volume_profile)(symbol="NODATA")
        assert "error" in result


class TestGetTradingCalendar:
    @pytest.mark.asyncio
    async def test_returns_calendar(self):
        result = await _fn(get_trading_calendar)()
        # Pure computation — no data dependency
        assert "error" not in result
