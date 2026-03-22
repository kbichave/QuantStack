# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for risk management MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quantstack.tools.mcp_bridge import (
    MCPBridge,
    check_risk_limits_tool,
    compute_position_size_tool,
    compute_var_tool,
    stress_test_portfolio_tool,
)


class TestRiskManagementTools:
    """Test risk management tool wrappers."""

    def test_compute_position_size_tool(self):
        """Test compute_position_size_tool returns position size."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "shares": 50,
                "dollar_amount": 5000,
                "risk_amount": 250,
                "position_pct": 0.05,
            }

            tool = compute_position_size_tool()
            result = tool._run(equity=100000, entry_price=100, stop_loss_price=95)

            data = json.loads(result)
            assert "shares" in data
            assert data["shares"] > 0

    def test_compute_var_tool(self):
        """Test compute_var_tool returns VaR metrics."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "var_95": 2500,
                "var_99": 3500,
                "cvar_95": 3000,
                "confidence_level": 0.95,
            }

            tool = compute_var_tool()
            result = tool._run(
                returns=[0.01, -0.02, 0.015, -0.01], portfolio_value=100000
            )

            data = json.loads(result)
            assert "var_95" in data
            assert data["var_95"] > 0

    def test_stress_test_portfolio_tool(self):
        """Test stress_test_portfolio_tool returns scenario results."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "scenarios": [
                    {"name": "Market Crash", "pnl": -15000},
                    {"name": "Vol Spike", "pnl": -5000},
                ],
                "worst_case": -15000,
            }

            tool = stress_test_portfolio_tool()
            positions = json.dumps([{"symbol": "SPY", "quantity": 100, "delta": 1.0}])
            scenarios = json.dumps([{"name": "Market Crash", "price_change": -0.10}])
            result = tool._run(positions=positions, scenarios=scenarios)

            data = json.loads(result)
            assert "scenarios" in data
            assert "worst_case" in data

    def test_check_risk_limits_tool(self):
        """Test check_risk_limits_tool returns pass/fail status."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "within_limits": True,
                "checks": {"delta_ok": True, "gamma_ok": True, "vega_ok": True},
                "utilization": {"delta": 0.5, "gamma": 0.3, "vega": 0.4},
            }

            tool = check_risk_limits_tool()
            result = tool._run(
                portfolio_delta=50, portfolio_gamma=25, portfolio_vega=2500
            )

            data = json.loads(result)
            assert "within_limits" in data
            assert "checks" in data
