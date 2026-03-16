# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for error handling in MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quant_pod.tools.mcp_bridge import (
    MCPBridge,
    compute_indicators_tool,
    get_symbol_snapshot_tool,
    run_backtest_tool,
)


class TestErrorHandling:
    """Test error handling in tool wrappers."""

    def test_quantcore_unavailable(self):
        """Test error when QuantCore MCP is unavailable."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"error": "QuantCore MCP not available"}

            tool = compute_indicators_tool()
            result = tool._run(symbol="SPY")

            data = json.loads(result)
            assert "error" in data
            assert "not available" in data["error"]

    def test_invalid_tool_response(self):
        """Test handling of tool not found error."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"error": "Tool nonexistent_tool not found in QuantCore MCP"}

            tool = run_backtest_tool()
            result = tool._run(symbol="SPY")

            data = json.loads(result)
            assert "error" in data

    def test_exception_handling(self):
        """Test exception from MCP call is returned as error dict."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            # Simulate the bridge returning an error dict (as it does when catching exceptions)
            mock_call.return_value = {"error": "Connection timeout"}

            tool = get_symbol_snapshot_tool()
            result = tool._run(symbol="SPY")

            # Error should be in the JSON response
            data = json.loads(result)
            assert "error" in data
            assert "timeout" in data["error"]
