# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for statistical analysis MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quantstack.tools.mcp_bridge import (
    MCPBridge,
    compute_alpha_decay_tool,
    compute_information_coefficient_tool,
    run_adf_test_tool,
    validate_signal_tool,
)


class TestStatisticalTools:
    """Test statistical analysis tool wrappers."""

    def test_run_adf_test_tool(self):
        """Test run_adf_test_tool returns stationarity result."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "adf_statistic": -3.5,
                "p_value": 0.008,
                "is_stationary": True,
                "interpretation": "Series is stationary at 99% confidence",
            }

            tool = run_adf_test_tool()
            result = tool._run(symbol="SPY", timeframe="daily", column="close")

            data = json.loads(result)
            assert "p_value" in data
            assert "is_stationary" in data

    def test_compute_alpha_decay_tool(self):
        """Test compute_alpha_decay_tool returns decay analysis."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "signal_column": "rsi_14",
                "decay_profile": [0.05, 0.04, 0.03, 0.02, 0.01],
                "optimal_horizon": 3,
                "half_life": 2.5,
            }

            tool = compute_alpha_decay_tool()
            result = tool._run(symbol="SPY", timeframe="daily", signal_column="rsi_14")

            data = json.loads(result)
            assert "optimal_horizon" in data
            assert "half_life" in data

    def test_compute_information_coefficient_tool(self):
        """Test compute_information_coefficient_tool returns IC value."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "signal_column": "rsi_14",
                "ic": 0.08,
                "ic_ir": 0.45,
                "is_significant": True,
            }

            tool = compute_information_coefficient_tool()
            result = tool._run(symbol="SPY", timeframe="daily", signal_column="rsi_14")

            data = json.loads(result)
            assert "ic" in data
            assert data["ic"] > 0

    def test_validate_signal_tool(self):
        """Test validate_signal_tool returns validation result."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "signal_column": "momentum_rsi",
                "is_valid": True,
                "checks": {
                    "has_predictive_power": True,
                    "no_lookahead_bias": True,
                    "sufficient_data": True,
                },
            }

            tool = validate_signal_tool()
            result = tool._run(symbol="SPY", signal_column="momentum_rsi")

            data = json.loads(result)
            assert "is_valid" in data
            assert "checks" in data
