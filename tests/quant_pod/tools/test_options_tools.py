# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for options pricing MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quant_pod.tools.mcp_bridge import (
    MCPBridge,
    analyze_option_structure_tool,
    compute_greeks_tool,
    compute_implied_vol_tool,
    compute_option_chain_tool,
    price_option_tool,
)


class TestOptionsTools:
    """Test options pricing tool wrappers."""

    def test_price_option_tool(self):
        """Test price_option_tool returns Black-Scholes price."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "option_type": "call",
                "price": 5.25,
                "spot": 100,
                "strike": 100,
                "greeks": {"delta": 0.52, "gamma": 0.04},
            }

            tool = price_option_tool()
            result = tool._run(spot=100, strike=100, time_to_expiry=0.25, volatility=0.20)

            data = json.loads(result)
            assert "price" in data
            assert data["price"] > 0

    def test_compute_greeks_tool(self):
        """Test compute_greeks_tool returns all Greeks."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "option_type": "call",
                "greeks": {
                    "delta": 0.52,
                    "gamma": 0.04,
                    "theta": -0.05,
                    "vega": 0.25,
                    "rho": 0.12,
                },
            }

            tool = compute_greeks_tool()
            result = tool._run(spot=100, strike=100, time_to_expiry=0.25, volatility=0.20)

            data = json.loads(result)
            assert "greeks" in data
            assert "delta" in data["greeks"]
            assert "gamma" in data["greeks"]

    def test_compute_implied_vol_tool(self):
        """Test compute_implied_vol_tool returns IV."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "implied_volatility": 0.22,
                "option_price": 5.50,
                "convergence": True,
            }

            tool = compute_implied_vol_tool()
            result = tool._run(option_price=5.50, spot=100, strike=100, time_to_expiry=0.25)

            data = json.loads(result)
            assert "implied_volatility" in data
            assert data["implied_volatility"] > 0

    def test_analyze_option_structure_tool(self):
        """Test analyze_option_structure_tool returns P&L profile."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "structure_type": "VERTICAL_SPREAD",
                "max_profit": 200,
                "max_loss": 300,
                "breakeven": [102.5],
                "net_greeks": {"delta": 0.25, "theta": -0.02},
            }

            tool = analyze_option_structure_tool()
            legs = json.dumps(
                [
                    {"strike": 100, "option_type": "call", "quantity": 1},
                    {"strike": 105, "option_type": "call", "quantity": -1},
                ]
            )
            result = tool._run(structure_type="VERTICAL_SPREAD", legs=legs, spot=100)

            data = json.loads(result)
            assert "max_profit" in data
            assert "max_loss" in data

    def test_compute_option_chain_tool(self):
        """Test compute_option_chain_tool returns theoretical chain."""
        with patch.object(MCPBridge, "call_quantcore", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "expiry_date": "2024-02-16",
                "calls": [{"strike": 470, "price": 5.50, "delta": 0.45}],
                "puts": [{"strike": 470, "price": 4.80, "delta": -0.55}],
            }

            tool = compute_option_chain_tool()
            result = tool._run(symbol="SPY", spot_price=470, volatility=0.18, days_to_expiry=30)

            data = json.loads(result)
            assert "calls" in data
            assert "puts" in data
