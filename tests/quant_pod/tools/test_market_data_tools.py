# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for market data MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quantstack.tools.mcp_bridge import (
    MCPBridge,
    fetch_market_data_tool,
    get_symbol_snapshot_tool,
    list_stored_symbols_tool,
)


class TestMarketDataTools:
    """Test market data tool wrappers."""

    def test_fetch_market_data_tool(self):
        """Test fetch_market_data_tool returns OHLCV data."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "data": [
                    {
                        "date": "2024-01-01",
                        "open": 470,
                        "high": 472,
                        "low": 469,
                        "close": 471,
                        "volume": 1000000,
                    }
                ],
            }

            tool = fetch_market_data_tool()
            result = tool._run(symbol="SPY", timeframe="daily", outputsize="compact")

            data = json.loads(result)
            assert data["symbol"] == "SPY"
            assert "data" in data

    def test_get_symbol_snapshot_tool(self):
        """Test get_symbol_snapshot_tool returns comprehensive snapshot."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "AAPL",
                "price": 175.50,
                "indicators": {"rsi": 55, "macd": 0.5},
                "regime": "trending_up",
            }

            tool = get_symbol_snapshot_tool()
            result = tool._run(symbol="AAPL")

            data = json.loads(result)
            assert data["symbol"] == "AAPL"
            assert "price" in data
            assert "indicators" in data

    def test_list_stored_symbols_tool(self):
        """Test list_stored_symbols_tool returns symbol list."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbols": ["SPY", "QQQ", "AAPL", "MSFT"],
                "count": 4,
            }

            tool = list_stored_symbols_tool()
            result = tool._run()

            data = json.loads(result)
            assert "symbols" in data
            assert len(data["symbols"]) == 4
