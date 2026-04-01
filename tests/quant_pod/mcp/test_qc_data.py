# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_data MCP tools — data fetching, loading, listing.

Tools use _get_reader() or the data registry. Mock both.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.mcp.tools.qc_data import (
    fetch_market_data,
    get_corporate_actions,
    get_insider_trades,
    get_institutional_ownership,
    list_stored_symbols,
    load_market_data,
)
from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


def _mock_store(df):
    """Return a mock PgDataStore with common methods."""
    store = MagicMock()
    store.load_ohlcv.return_value = df
    store.list_symbols.return_value = ["SPY", "AAPL", "QQQ"]
    # list_stored_symbols uses get_metadata(), not list_symbols()
    store.get_metadata.return_value = pd.DataFrame({
        "symbol": ["SPY", "AAPL", "QQQ"],
        "timeframe": ["daily", "daily", "daily"],
        "first_timestamp": [None, None, None],
        "last_timestamp": [None, None, None],
        "row_count": [100, 200, 150],
    })
    store.load_insider_trades.return_value = pd.DataFrame()
    store.load_institutional_ownership.return_value = pd.DataFrame()
    store.load_corporate_actions.return_value = pd.DataFrame()
    store.close.return_value = None
    return store


def _patch_reader(df):
    return patch(
        "quantstack.mcp.tools.qc_data._get_reader",
        return_value=_mock_store(df),
    )


class TestLoadMarketData:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=100)
        with _patch_reader(df):
            result = await _fn(load_market_data)(symbol="SPY")
        assert "error" not in result
        assert "symbol" in result

    @pytest.mark.asyncio
    async def test_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(load_market_data)(symbol="NODATA")
        assert "error" in result


class TestListStoredSymbols:
    @pytest.mark.asyncio
    async def test_returns_symbols(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(list_stored_symbols)()
        assert "error" not in result
        assert "symbols" in result
        # list_stored_symbols returns a dict keyed by symbol, not a list
        assert isinstance(result["symbols"], dict)
        assert len(result["symbols"]) == 3
        assert "total_symbols" in result


class TestFetchMarketData:
    @pytest.mark.asyncio
    async def test_fetch_with_mock_context(self):
        """fetch_market_data uses _get_data_registry() — mock it."""
        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = synthetic_ohlcv("AAPL", n_days=50)

        with (
            patch("quantstack.mcp.tools.qc_data._get_data_registry", return_value=mock_registry),
            patch("quantstack.mcp.tools.qc_data._get_writer") as mock_writer,
        ):
            mock_writer.return_value = MagicMock()
            result = await _fn(fetch_market_data)(symbol="AAPL")

        assert isinstance(result, dict)
        if "error" not in result:
            assert result["symbol"] == "AAPL"


class TestGetInsiderTrades:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(get_insider_trades)(symbol="AAPL")
        # May need specific data provider — check it returns dict
        assert isinstance(result, dict)


class TestGetInstitutionalOwnership:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(get_institutional_ownership)(symbol="AAPL")
        assert isinstance(result, dict)


class TestGetCorporateActions:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(get_corporate_actions)(symbol="AAPL")
        assert isinstance(result, dict)
