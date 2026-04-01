# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool factory functions in MCP Bridge."""

from quantstack.tools.mcp_bridge import (
    ComputeAllFeaturesTool,
    ComputeGreeksTool,
    ComputeImpliedVolTool,
    ComputeIndicatorsTool,
    ComputePositionSizeTool,
    ComputeVaRTool,
    FetchMarketDataTool,
    GetSymbolSnapshotTool,
    ListAvailableIndicatorsTool,
    ListStoredSymbolsTool,
    LoadMarketDataTool,
    PriceOptionTool,
    RunBacktestTool,
    RunMonteCarloTool,
    RunWalkForwardTool,
    StressTestPortfolioTool,
    compute_all_features_tool,
    compute_greeks_tool,
    compute_implied_vol_tool,
    compute_indicators_tool,
    compute_position_size_tool,
    compute_var_tool,
    fetch_market_data_tool,
    get_symbol_snapshot_tool,
    list_available_indicators_tool,
    list_stored_symbols_tool,
    load_market_data_tool,
    price_option_tool,
    run_backtest_tool,
    run_monte_carlo_tool,
    run_walkforward_tool,
    stress_test_portfolio_tool,
)


class TestFactoryFunctions:
    """Test tool factory functions return correct instances."""

    def test_market_data_factories(self):
        """Test market data tool factories."""
        assert isinstance(fetch_market_data_tool(), FetchMarketDataTool)
        assert isinstance(load_market_data_tool(), LoadMarketDataTool)
        assert isinstance(list_stored_symbols_tool(), ListStoredSymbolsTool)
        assert isinstance(get_symbol_snapshot_tool(), GetSymbolSnapshotTool)

    def test_technical_analysis_factories(self):
        """Test technical analysis tool factories."""
        assert isinstance(compute_indicators_tool(), ComputeIndicatorsTool)
        assert isinstance(compute_all_features_tool(), ComputeAllFeaturesTool)
        assert isinstance(list_available_indicators_tool(), ListAvailableIndicatorsTool)

    def test_backtesting_factories(self):
        """Test backtesting tool factories."""
        assert isinstance(run_backtest_tool(), RunBacktestTool)
        assert isinstance(run_monte_carlo_tool(), RunMonteCarloTool)
        assert isinstance(run_walkforward_tool(), RunWalkForwardTool)

    def test_options_factories(self):
        """Test options tool factories."""
        assert isinstance(price_option_tool(), PriceOptionTool)
        assert isinstance(compute_greeks_tool(), ComputeGreeksTool)
        assert isinstance(compute_implied_vol_tool(), ComputeImpliedVolTool)

    def test_risk_factories(self):
        """Test risk management tool factories."""
        assert isinstance(compute_position_size_tool(), ComputePositionSizeTool)
        assert isinstance(compute_var_tool(), ComputeVaRTool)
        assert isinstance(stress_test_portfolio_tool(), StressTestPortfolioTool)
