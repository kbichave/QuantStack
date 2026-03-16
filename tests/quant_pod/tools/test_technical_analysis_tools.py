# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for technical analysis MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quant_pod.tools.mcp_bridge import MCPBridge, compute_all_features_tool, compute_indicators_tool, list_available_indicators_tool


class TestTechnicalAnalysisTools:
    """Test technical analysis tool wrappers."""

    def test_compute_indicators_tool(self):
        """Test compute_indicators_tool returns indicator values."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "indicators": {"rsi_14": 55.5, "macd": 1.2, "atr_14": 5.5},
            }

            tool = compute_indicators_tool()
            result = tool._run(symbol="SPY", timeframe="daily", indicators=["rsi_14", "macd"])

            data = json.loads(result)
            assert data["symbol"] == "SPY"
            assert "indicators" in data
            assert "rsi_14" in data["indicators"]

    def test_compute_all_features_tool(self):
        """Test compute_all_features_tool returns 200+ features."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "feature_count": 215,
                "features": {"trend_sma_20": 465.5, "momentum_rsi": 55},
            }

            tool = compute_all_features_tool()
            result = tool._run(symbol="SPY", timeframe="daily")

            data = json.loads(result)
            assert data["symbol"] == "SPY"
            assert data["feature_count"] > 0

    def test_list_available_indicators_tool(self):
        """Test list_available_indicators_tool returns indicator catalog."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "indicators": [
                    {"name": "RSI", "description": "Relative Strength Index"},
                    {
                        "name": "MACD",
                        "description": "Moving Average Convergence Divergence",
                    },
                ],
                "count": 200,
            }

            tool = list_available_indicators_tool()
            result = tool._run()

            data = json.loads(result)
            assert "indicators" in data
            assert data["count"] > 0
