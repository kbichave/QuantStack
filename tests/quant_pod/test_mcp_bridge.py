# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MCP Bridge tool wrappers."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from quant_pod.tools.mcp_bridge import (
    # Bridge
    MCPBridge,
    get_bridge,
    # Market Data Tools
    FetchMarketDataTool,
    LoadMarketDataTool,
    ListStoredSymbolsTool,
    GetSymbolSnapshotTool,
    fetch_market_data_tool,
    load_market_data_tool,
    list_stored_symbols_tool,
    get_symbol_snapshot_tool,
    # Technical Analysis Tools
    ComputeIndicatorsTool,
    ComputeAllFeaturesTool,
    ListAvailableIndicatorsTool,
    compute_indicators_tool,
    compute_all_features_tool,
    list_available_indicators_tool,
    # Backtesting Tools
    RunBacktestTool,
    RunMonteCarloTool,
    RunWalkForwardTool,
    run_backtest_tool,
    run_monte_carlo_tool,
    run_walkforward_tool,
    # Statistical Tools
    RunADFTestTool,
    ComputeAlphaDecayTool,
    ComputeInformationCoefficientTool,
    ValidateSignalTool,
    run_adf_test_tool,
    compute_alpha_decay_tool,
    compute_information_coefficient_tool,
    validate_signal_tool,
    # Options Tools
    PriceOptionTool,
    ComputeGreeksTool,
    ComputeImpliedVolTool,
    AnalyzeOptionStructureTool,
    ComputeOptionChainTool,
    price_option_tool,
    compute_greeks_tool,
    compute_implied_vol_tool,
    analyze_option_structure_tool,
    compute_option_chain_tool,
    # Risk Management Tools
    ComputePositionSizeTool,
    ComputeVaRTool,
    StressTestPortfolioTool,
    CheckRiskLimitsTool,
    compute_position_size_tool,
    compute_var_tool,
    stress_test_portfolio_tool,
    check_risk_limits_tool,
    # Market/Regime Tools
    GetMarketRegimeSnapshotTool,
    AnalyzeVolumeProfileTool,
    GetEventCalendarTool,
    get_market_regime_snapshot_tool,
    analyze_volume_profile_tool,
    get_event_calendar_tool,
    # Trade Tools
    ValidateTradeTool,
    ScoreTradeStructureTool,
    GenerateTradeTemplateTool,
    validate_trade_tool,
    score_trade_structure_tool,
    generate_trade_template_tool,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_bridge():
    """Create a mock MCP bridge."""
    bridge = MagicMock(spec=MCPBridge)
    bridge._quantcore_available = True
    bridge._etrade_available = True
    return bridge


@pytest.fixture
def mock_quantcore_response():
    """Mock successful QuantCore response."""

    async def _mock_call(tool_name, **kwargs):
        return {"status": "success", "tool": tool_name, "params": kwargs}

    return _mock_call


@pytest.fixture
def mock_quantcore_error():
    """Mock QuantCore unavailable response."""

    async def _mock_call(tool_name, **kwargs):
        return {"error": "QuantCore MCP not available"}

    return _mock_call


# =============================================================================
# TEST MARKET DATA TOOLS
# =============================================================================


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


# =============================================================================
# TEST TECHNICAL ANALYSIS TOOLS
# =============================================================================


class TestTechnicalAnalysisTools:
    """Test technical analysis tool wrappers."""

    def test_compute_indicators_tool(self):
        """Test compute_indicators_tool returns indicator values."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "indicators": {"rsi_14": 55.5, "macd": 1.2, "atr_14": 5.5},
            }

            tool = compute_indicators_tool()
            result = tool._run(
                symbol="SPY", timeframe="daily", indicators=["rsi_14", "macd"]
            )

            data = json.loads(result)
            assert data["symbol"] == "SPY"
            assert "indicators" in data
            assert "rsi_14" in data["indicators"]

    def test_compute_all_features_tool(self):
        """Test compute_all_features_tool returns 200+ features."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
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
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
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


# =============================================================================
# TEST BACKTESTING TOOLS
# =============================================================================


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


# =============================================================================
# TEST STATISTICAL TOOLS
# =============================================================================


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


# =============================================================================
# TEST OPTIONS TOOLS
# =============================================================================


class TestOptionsTools:
    """Test options pricing tool wrappers."""

    def test_price_option_tool(self):
        """Test price_option_tool returns Black-Scholes price."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "option_type": "call",
                "price": 5.25,
                "spot": 100,
                "strike": 100,
                "greeks": {"delta": 0.52, "gamma": 0.04},
            }

            tool = price_option_tool()
            result = tool._run(
                spot=100, strike=100, time_to_expiry=0.25, volatility=0.20
            )

            data = json.loads(result)
            assert "price" in data
            assert data["price"] > 0

    def test_compute_greeks_tool(self):
        """Test compute_greeks_tool returns all Greeks."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
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
            result = tool._run(
                spot=100, strike=100, time_to_expiry=0.25, volatility=0.20
            )

            data = json.loads(result)
            assert "greeks" in data
            assert "delta" in data["greeks"]
            assert "gamma" in data["greeks"]

    def test_compute_implied_vol_tool(self):
        """Test compute_implied_vol_tool returns IV."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "implied_volatility": 0.22,
                "option_price": 5.50,
                "convergence": True,
            }

            tool = compute_implied_vol_tool()
            result = tool._run(
                option_price=5.50, spot=100, strike=100, time_to_expiry=0.25
            )

            data = json.loads(result)
            assert "implied_volatility" in data
            assert data["implied_volatility"] > 0

    def test_analyze_option_structure_tool(self):
        """Test analyze_option_structure_tool returns P&L profile."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
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
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "symbol": "SPY",
                "expiry_date": "2024-02-16",
                "calls": [{"strike": 470, "price": 5.50, "delta": 0.45}],
                "puts": [{"strike": 470, "price": 4.80, "delta": -0.55}],
            }

            tool = compute_option_chain_tool()
            result = tool._run(
                symbol="SPY", spot_price=470, volatility=0.18, days_to_expiry=30
            )

            data = json.loads(result)
            assert "calls" in data
            assert "puts" in data


# =============================================================================
# TEST RISK MANAGEMENT TOOLS
# =============================================================================


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


# =============================================================================
# TEST MARKET REGIME TOOLS
# =============================================================================


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


# =============================================================================
# TEST TRADE TOOLS
# =============================================================================


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


# =============================================================================
# TEST ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Test error handling in tool wrappers."""

    def test_quantcore_unavailable(self):
        """Test error when QuantCore MCP is unavailable."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"error": "QuantCore MCP not available"}

            tool = compute_indicators_tool()
            result = tool._run(symbol="SPY")

            data = json.loads(result)
            assert "error" in data
            assert "not available" in data["error"]

    def test_invalid_tool_response(self):
        """Test handling of tool not found error."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {
                "error": "Tool nonexistent_tool not found in QuantCore MCP"
            }

            tool = run_backtest_tool()
            result = tool._run(symbol="SPY")

            data = json.loads(result)
            assert "error" in data

    def test_exception_handling(self):
        """Test exception from MCP call is returned as error dict."""
        with patch.object(
            MCPBridge, "call_quantcore", new_callable=AsyncMock
        ) as mock_call:
            # Simulate the bridge returning an error dict (as it does when catching exceptions)
            mock_call.return_value = {"error": "Connection timeout"}

            tool = get_symbol_snapshot_tool()
            result = tool._run(symbol="SPY")

            # Error should be in the JSON response
            data = json.loads(result)
            assert "error" in data
            assert "timeout" in data["error"]


# =============================================================================
# TEST FACTORY FUNCTIONS
# =============================================================================


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
