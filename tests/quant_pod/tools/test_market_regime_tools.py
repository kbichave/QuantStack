# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for market regime MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quantstack.tools.mcp_bridge import (
    MCPBridge,
    analyze_volume_profile_tool,
    get_event_calendar_tool,
    get_market_regime_snapshot_tool,
)


class TestMarketRegimeTools:
    """Test market regime tool wrappers."""

    def test_get_market_regime_snapshot_tool(self):
        """Test get_market_regime_snapshot_tool returns regime classification."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "regime": "trending_up",
                "confidence": 0.85,
                "volatility_regime": "normal",
                "signals": {"adx": 32, "trend_strength": 0.7},
            }

            tool = get_market_regime_snapshot_tool()
            result = tool._run()

            data = json.loads(result)
            assert "regime" in data
            assert "confidence" in data

    def test_analyze_volume_profile_tool(self):
        """Test analyze_volume_profile_tool returns volume analysis."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "poc": 470.50,  # Point of Control
                "value_area_high": 472.00,
                "value_area_low": 468.50,
                "support_levels": [468, 465],
                "resistance_levels": [473, 475],
            }

            tool = analyze_volume_profile_tool()
            result = tool._run(symbol="SPY", timeframe="daily", num_bins=20)

            data = json.loads(result)
            assert "poc" in data
            assert "value_area_high" in data

    def test_get_event_calendar_tool(self):
        """Test get_event_calendar_tool returns upcoming events."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "events": [
                    {"date": "2024-01-15", "type": "earnings", "symbol": "AAPL"},
                    {
                        "date": "2024-01-17",
                        "type": "fomc",
                        "description": "Fed Meeting",
                    },
                ],
                "count": 2,
            }

            tool = get_event_calendar_tool()
            result = tool._run(days_ahead=7)

            data = json.loads(result)
            assert "events" in data
            assert len(data["events"]) > 0
