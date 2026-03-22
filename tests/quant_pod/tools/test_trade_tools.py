# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for trade MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quantstack.tools.mcp_bridge import (
    MCPBridge,
    generate_trade_template_tool,
    score_trade_structure_tool,
    validate_trade_tool,
)


class TestTradeTools:
    """Test trade tool wrappers."""

    def test_validate_trade_tool(self):
        """Test validate_trade_tool returns validation result."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "is_valid": True,
                "checks": {
                    "risk_reward_ok": True,
                    "position_size_ok": True,
                    "stop_loss_ok": True,
                },
                "warnings": [],
            }

            tool = validate_trade_tool()
            result = tool._run(
                symbol="SPY",
                direction="LONG",
                entry_price=470,
                stop_loss=465,
                position_size=5000,
            )

            data = json.loads(result)
            assert "is_valid" in data
            assert "checks" in data

    def test_score_trade_structure_tool(self):
        """Test score_trade_structure_tool returns score."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "total_score": 75,
                "grade": "B",
                "components": {
                    "risk_reward_score": 80,
                    "probability_score": 70,
                    "time_decay_score": 75,
                },
            }

            tool = score_trade_structure_tool()
            result = tool._run(
                structure_type="VERTICAL_SPREAD",
                max_profit=200,
                max_loss=100,
                probability_of_profit=0.65,
                days_to_expiry=30,
            )

            data = json.loads(result)
            assert "total_score" in data
            assert "grade" in data

    def test_generate_trade_template_tool(self):
        """Test generate_trade_template_tool returns template."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "direction": "LONG",
                "structure": "VERTICAL_SPREAD",
                "legs": [
                    {"strike": 470, "option_type": "call", "action": "BUY"},
                    {"strike": 475, "option_type": "call", "action": "SELL"},
                ],
                "estimated_cost": 250,
                "max_profit": 250,
                "max_loss": 250,
            }

            tool = generate_trade_template_tool()
            result = tool._run(
                symbol="SPY", direction="LONG", structure_type="VERTICAL_SPREAD"
            )

            data = json.loads(result)
            assert "legs" in data
            assert len(data["legs"]) > 0
