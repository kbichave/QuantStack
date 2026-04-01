# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for backtesting MCP Bridge tool wrappers."""

import json
from unittest.mock import AsyncMock, patch

from quantstack.tools.mcp_bridge import (
    MCPBridge,
    run_backtest_tool,
    run_monte_carlo_tool,
    run_walkforward_tool,
)


class TestBacktestingTools:
    """Test backtesting tool wrappers."""

    def test_run_backtest_tool(self):
        """Test run_backtest_tool returns backtest metrics."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "strategy": "mean_reversion",
                "metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.08,
                    "win_rate": 0.55,
                    "total_trades": 42,
                },
            }

            tool = run_backtest_tool()
            result = tool._run(symbol="SPY", strategy_type="mean_reversion")

            data = json.loads(result)
            assert data["symbol"] == "SPY"
            assert "metrics" in data
            assert data["metrics"]["sharpe_ratio"] > 0

    def test_run_monte_carlo_tool(self):
        """Test run_monte_carlo_tool returns simulation results."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "n_simulations": 1000,
                "results": {
                    "mean_return": 0.12,
                    "std_return": 0.08,
                    "percentile_5": 0.02,
                    "percentile_95": 0.25,
                },
            }

            tool = run_monte_carlo_tool()
            result = tool._run(symbol="SPY", timeframe="daily", n_simulations=1000)

            data = json.loads(result)
            assert data["n_simulations"] == 1000
            assert "results" in data

    def test_run_walkforward_tool(self):
        """Test run_walkforward_tool returns walk-forward splits."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "n_splits": 5,
                "splits": [
                    {
                        "train_start": "2023-01-01",
                        "train_end": "2023-06-30",
                        "test_sharpe": 1.1,
                    }
                ],
                "average_oos_sharpe": 0.95,
            }

            tool = run_walkforward_tool()
            result = tool._run(symbol="SPY", timeframe="daily", n_splits=5)

            data = json.loads(result)
            assert data["n_splits"] == 5
            assert "splits" in data
