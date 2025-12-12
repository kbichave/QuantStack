# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MCP Bridge - CrewAI tools that wrap MCP server calls.

Provides CrewAI-compatible tools that call the QuantCore and eTrade MCP servers.

QuantCore Tools:
- Market Data: fetch_market_data, load_market_data, list_stored_symbols, get_symbol_snapshot
- Technical Analysis: compute_indicators, compute_all_features, list_available_indicators
- Backtesting: run_backtest, get_backtest_metrics, run_monte_carlo, run_walkforward
- Statistical: run_adf_test, compute_alpha_decay, compute_information_coefficient
- Options: price_option, compute_greeks, compute_implied_vol, analyze_option_structure
- Risk: compute_position_size, compute_var, stress_test_portfolio, check_risk_limits
- Market: get_market_regime_snapshot, analyze_volume_profile, get_trading_calendar

eTrade Tools:
- Quotes: get_quote, get_option_chains
- Orders: preview_order, place_order
- Account: get_positions, get_account_balance
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type

from quant_pod.crewai_compat import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


# =============================================================================
# MCP BRIDGE CLASS
# =============================================================================


class MCPBridge:
    """
    Bridge between CrewAI agents and MCP servers.

    Handles communication with:
    - QuantCore MCP (technical analysis, backtesting, options, risk)
    - eTrade MCP (trading, account management)
    """

    def __init__(self):
        """Initialize MCP bridge."""
        self._quantcore_available = False
        self._etrade_available = False
        self._check_servers()

    def _check_servers(self) -> None:
        """Check which MCP servers are available."""
        try:
            from quantcore.mcp.server import mcp as quantcore_mcp

            self._quantcore_available = True
            logger.info("QuantCore MCP server available")
        except ImportError:
            logger.warning("QuantCore MCP server not available")

        try:
            from etrade_mcp.server import mcp as etrade_mcp

            self._etrade_available = True
            logger.info("eTrade MCP server available")
        except ImportError:
            logger.warning("eTrade MCP server not available")

    async def call_quantcore(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a QuantCore MCP tool."""
        if not self._quantcore_available:
            return {"error": "QuantCore MCP not available"}

        try:
            from quantcore.mcp.server import mcp as quantcore_mcp

            tool_func = getattr(quantcore_mcp, tool_name, None)
            if tool_func is None:
                return {"error": f"Tool {tool_name} not found in QuantCore MCP"}
            result = await tool_func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"QuantCore MCP call failed: {e}")
            return {"error": str(e)}

    async def call_etrade(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call an eTrade MCP tool."""
        if not self._etrade_available:
            return {"error": "eTrade MCP not available"}

        try:
            from etrade_mcp.server import mcp as etrade_mcp

            tool_func = getattr(etrade_mcp, tool_name, None)
            if tool_func is None:
                return {"error": f"Tool {tool_name} not found in eTrade MCP"}
            result = await tool_func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"eTrade MCP call failed: {e}")
            return {"error": str(e)}


# Global bridge instance
_bridge: Optional[MCPBridge] = None


def get_bridge() -> MCPBridge:
    """Get or create the global MCP bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = MCPBridge()
    return _bridge


def _run_async(coro):
    """Helper to run async coroutine synchronously."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# =============================================================================
# INPUT SCHEMAS - eTrade Tools
# =============================================================================


class QuoteInput(BaseModel):
    """Input for get_quote tool."""

    symbols: str = Field(
        ..., description="Comma-separated list of symbols (e.g., 'SPY,AAPL,MSFT')"
    )


class OptionChainInput(BaseModel):
    """Input for get_option_chains tool."""

    symbol: str = Field(..., description="Underlying symbol")
    expiration_date: Optional[str] = Field(
        None, description="Expiration date (YYYY-MM-DD)"
    )
    strike_price_near: Optional[float] = Field(
        None, description="Center strikes around this price"
    )
    no_of_strikes: int = Field(10, description="Number of strikes to return")


class PreviewOrderInput(BaseModel):
    """Input for preview_order tool."""

    account_id_key: str = Field(..., description="Account ID key")
    symbol: str = Field(..., description="Symbol to trade")
    action: str = Field(..., description="BUY, SELL, BUY_TO_OPEN, etc.")
    quantity: int = Field(..., description="Number of shares/contracts")
    order_type: str = Field("LIMIT", description="MARKET, LIMIT, STOP")
    limit_price: Optional[float] = Field(None, description="Limit price")


class PlaceOrderInput(BaseModel):
    """Input for place_order tool."""

    account_id_key: str = Field(..., description="Account ID key")
    symbol: str = Field(..., description="Symbol to trade")
    action: str = Field(..., description="BUY, SELL, BUY_TO_OPEN, etc.")
    quantity: int = Field(..., description="Number of shares/contracts")
    order_type: str = Field("LIMIT", description="MARKET, LIMIT, STOP")
    limit_price: Optional[float] = Field(None, description="Limit price")
    preview_id: Optional[str] = Field(None, description="Preview ID from preview_order")


class PositionsInput(BaseModel):
    """Input for get_positions tool."""

    account_id_key: str = Field(..., description="Account ID key")
    symbol: Optional[str] = Field(None, description="Optional symbol filter")


class BalanceInput(BaseModel):
    """Input for get_account_balance tool."""

    account_id_key: str = Field(..., description="Account ID key")


# =============================================================================
# INPUT SCHEMAS - QuantCore Market Data Tools
# =============================================================================


class FetchMarketDataInput(BaseModel):
    """Input for fetch_market_data tool."""

    symbol: str = Field(..., description="Stock/ETF symbol (e.g., 'SPY', 'AAPL')")
    timeframe: str = Field("daily", description="Data frequency: daily, 1h, 4h, weekly")
    outputsize: str = Field(
        "compact", description="'compact' (100 bars) or 'full' (20+ years)"
    )


class LoadMarketDataInput(BaseModel):
    """Input for load_market_data tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe: 1h, 4h, daily, weekly")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class SymbolInput(BaseModel):
    """Input for symbol-based tools."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe: 1h, 4h, daily, weekly")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class IndicatorsInput(BaseModel):
    """Input for compute_indicators tool."""

    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field("daily", description="Timeframe: 1h, 4h, daily, weekly")
    indicators: Optional[List[str]] = Field(
        None, description="List of indicators (RSI, MACD, ATR, etc.)"
    )
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


# =============================================================================
# INPUT SCHEMAS - QuantCore Backtesting Tools
# =============================================================================


class BacktestInput(BaseModel):
    """Input for run_backtest tool."""

    symbol: str = Field(..., description="Symbol to backtest")
    strategy_type: str = Field(
        "mean_reversion", description="mean_reversion, trend_following, momentum"
    )
    timeframe: str = Field("daily", description="Timeframe")
    initial_capital: float = Field(100000, description="Starting capital")
    position_size_pct: float = Field(10, description="Position size as % of equity")
    stop_loss_atr: float = Field(2, description="Stop loss in ATR multiples")
    take_profit_atr: float = Field(3, description="Take profit in ATR multiples")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class BacktestMetricsInput(BaseModel):
    """Input for get_backtest_metrics tool."""

    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Risk-adjusted return metric")
    max_drawdown: float = Field(..., description="Maximum peak-to-trough decline")
    win_rate: float = Field(..., description="Percentage of winning trades")
    total_trades: int = Field(..., description="Total number of trades")


class MonteCarloInput(BaseModel):
    """Input for run_monte_carlo tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    n_simulations: int = Field(1000, description="Number of simulations to run")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class WalkForwardInput(BaseModel):
    """Input for run_walkforward tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe")
    n_splits: int = Field(5, description="Number of train/test splits")
    train_pct: float = Field(0.7, description="Percentage for training")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


# =============================================================================
# INPUT SCHEMAS - QuantCore Statistical Tools
# =============================================================================


class ADFTestInput(BaseModel):
    """Input for run_adf_test tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    column: str = Field("close", description="Column to test: close, returns, spread")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class AlphaDecayInput(BaseModel):
    """Input for compute_alpha_decay tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    signal_column: str = Field("rsi_14", description="Feature to analyze as signal")
    max_lag: int = Field(20, description="Maximum forward lag to analyze")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class ICInput(BaseModel):
    """Input for compute_information_coefficient tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Data timeframe")
    signal_column: str = Field("rsi_14", description="Feature to analyze")
    forward_return_periods: int = Field(5, description="Forward return horizon in bars")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class ValidateSignalInput(BaseModel):
    """Input for validate_signal tool."""

    symbol: str = Field(..., description="Stock symbol")
    signal_column: str = Field(..., description="Signal column to validate")
    timeframe: str = Field("daily", description="Timeframe")


class DiagnoseSignalInput(BaseModel):
    """Input for diagnose_signal tool."""

    symbol: str = Field(..., description="Stock symbol")
    signal_column: str = Field(..., description="Signal column to diagnose")
    timeframe: str = Field("daily", description="Timeframe")


# =============================================================================
# INPUT SCHEMAS - QuantCore Options Tools
# =============================================================================


class PriceOptionInput(BaseModel):
    """Input for price_option tool."""

    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Option strike price")
    time_to_expiry: float = Field(
        ..., description="Time to expiration in years (e.g., 0.25 for 3 months)"
    )
    volatility: float = Field(
        ..., description="Annualized volatility (e.g., 0.20 for 20%)"
    )
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    option_type: str = Field("call", description="'call' or 'put'")
    dividend_yield: float = Field(0, description="Continuous dividend yield")


class ComputeGreeksInput(BaseModel):
    """Input for compute_greeks tool."""

    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Option strike price")
    time_to_expiry: float = Field(..., description="Time to expiration in years")
    volatility: float = Field(..., description="Annualized volatility")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    option_type: str = Field("call", description="'call' or 'put'")
    dividend_yield: float = Field(0, description="Continuous dividend yield")


class ImpliedVolInput(BaseModel):
    """Input for compute_implied_vol tool."""

    option_price: float = Field(..., description="Market option price")
    spot: float = Field(..., description="Current stock price")
    strike: float = Field(..., description="Strike price")
    time_to_expiry: float = Field(..., description="Time to expiry in years")
    risk_free_rate: float = Field(0.05, description="Risk-free rate")
    option_type: str = Field("call", description="'call' or 'put'")


class AnalyzeOptionStructureInput(BaseModel):
    """Input for analyze_option_structure tool."""

    structure_type: str = Field(
        ..., description="VERTICAL_SPREAD, IRON_CONDOR, STRADDLE, etc."
    )
    legs: str = Field(
        ..., description="JSON array of legs with strike, option_type, quantity"
    )
    spot: float = Field(..., description="Current spot price")
    volatility: float = Field(0.25, description="Implied volatility")
    time_to_expiry: float = Field(0.083, description="Time to expiry in years")


class ComputeOptionChainInput(BaseModel):
    """Input for compute_option_chain tool."""

    symbol: str = Field(..., description="Underlying symbol")
    spot_price: float = Field(..., description="Current spot price")
    volatility: float = Field(0.25, description="Implied volatility")
    days_to_expiry: int = Field(30, description="Days until expiration")
    num_strikes: int = Field(10, description="Number of strikes each side of ATM")


class MultiLegPriceInput(BaseModel):
    """Input for compute_multi_leg_price tool."""

    legs: str = Field(
        ..., description="JSON array of legs with strike, option_type, quantity, price"
    )
    spot: float = Field(..., description="Current spot price")


# =============================================================================
# INPUT SCHEMAS - QuantCore Risk Management Tools
# =============================================================================


class PositionSizeInput(BaseModel):
    """Input for compute_position_size tool."""

    equity: float = Field(..., description="Total account equity")
    entry_price: float = Field(..., description="Planned entry price")
    stop_loss_price: float = Field(..., description="Stop loss price level")
    risk_per_trade_pct: float = Field(
        1, description="Percentage of equity to risk per trade"
    )
    max_position_pct: float = Field(20, description="Maximum position as % of equity")
    alignment_score: float = Field(
        1, description="Cross-timeframe alignment score (0-1)"
    )


class MaxDrawdownInput(BaseModel):
    """Input for compute_max_drawdown tool."""

    equity_curve: List[float] = Field(
        ..., description="List of equity values over time"
    )


class PortfolioStatsInput(BaseModel):
    """Input for compute_portfolio_stats tool."""

    returns: List[float] = Field(..., description="List of period returns")
    risk_free_rate: float = Field(0.02, description="Annual risk-free rate")


class VaRInput(BaseModel):
    """Input for compute_var tool."""

    returns: List[float] = Field(..., description="Historical returns")
    confidence_level: float = Field(0.95, description="VaR confidence level")
    portfolio_value: float = Field(100000, description="Portfolio value")


class StressTestInput(BaseModel):
    """Input for stress_test_portfolio tool."""

    positions: str = Field(
        ..., description="JSON array of positions with symbol, quantity, delta, gamma"
    )
    scenarios: str = Field(
        ..., description="JSON array of scenarios with name, price_change, vol_change"
    )


class RiskLimitsInput(BaseModel):
    """Input for check_risk_limits tool."""

    portfolio_delta: float = Field(..., description="Net portfolio delta")
    portfolio_gamma: float = Field(..., description="Net portfolio gamma")
    portfolio_vega: float = Field(..., description="Net portfolio vega")
    max_delta: float = Field(100, description="Maximum allowed delta")
    max_gamma: float = Field(50, description="Maximum allowed gamma")
    max_vega: float = Field(5000, description="Maximum allowed vega")


class LiquidityInput(BaseModel):
    """Input for analyze_liquidity tool."""

    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field("daily", description="Timeframe")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


# =============================================================================
# INPUT SCHEMAS - QuantCore Market/Regime Tools
# =============================================================================


class SymbolSnapshotInput(BaseModel):
    """Input for get_symbol_snapshot tool."""

    symbol: str = Field(..., description="Stock symbol")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class VolumeProfileInput(BaseModel):
    """Input for analyze_volume_profile tool."""

    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field("daily", description="Timeframe")
    num_bins: int = Field(20, description="Number of price bins")
    end_date: Optional[str] = Field(
        None, description="End date filter (YYYY-MM-DD) for historical simulation"
    )


class TradingCalendarInput(BaseModel):
    """Input for get_trading_calendar tool."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class EventCalendarInput(BaseModel):
    """Input for get_event_calendar tool."""

    symbol: Optional[str] = Field(None, description="Optional symbol filter")
    days_ahead: int = Field(7, description="Days to look ahead")


# =============================================================================
# INPUT SCHEMAS - QuantCore Trade Tools
# =============================================================================


class TradeTemplateInput(BaseModel):
    """Input for generate_trade_template tool."""

    symbol: str = Field(..., description="Stock symbol")
    direction: str = Field(..., description="LONG or SHORT")
    structure_type: str = Field("VERTICAL_SPREAD", description="Option structure type")
    max_risk: float = Field(500, description="Maximum dollar risk")


class ValidateTradeInput(BaseModel):
    """Input for validate_trade tool."""

    symbol: str = Field(..., description="Stock symbol")
    direction: str = Field(..., description="LONG or SHORT")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    position_size: float = Field(..., description="Position size in dollars")


class ScoreTradeInput(BaseModel):
    """Input for score_trade_structure tool."""

    structure_type: str = Field(..., description="Option structure type")
    max_profit: float = Field(..., description="Maximum profit potential")
    max_loss: float = Field(..., description="Maximum loss potential")
    probability_of_profit: float = Field(..., description="POP estimate (0-1)")
    days_to_expiry: int = Field(..., description="DTE")


class SimulateTradeInput(BaseModel):
    """Input for simulate_trade_outcome tool."""

    symbol: str = Field(..., description="Stock symbol")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss")
    take_profit: float = Field(..., description="Take profit")
    position_size: int = Field(..., description="Number of shares")
    days_to_hold: int = Field(20, description="Holding period")


class ScreenerInput(BaseModel):
    """Input for run_screener tool."""

    min_price: float = Field(10, description="Minimum stock price")
    max_price: float = Field(500, description="Maximum stock price")
    min_volume: int = Field(1000000, description="Minimum daily volume")
    rsi_oversold: float = Field(30, description="RSI oversold threshold")
    rsi_overbought: float = Field(70, description="RSI overbought threshold")


# =============================================================================
# ETRADE TOOL CLASSES
# =============================================================================


class GetQuoteTool(BaseTool):
    """Tool to get real-time quotes."""

    name: str = "get_quote"
    description: str = (
        "Get real-time quotes for one or more symbols. Returns bid/ask, last price, volume."
    )
    args_schema: Type[BaseModel] = QuoteInput

    def _run(self, symbols: str) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade("get_quote", symbols=symbols)

        return json.dumps(_run_async(_exec()), indent=2)


class GetOptionChainsTool(BaseTool):
    """Tool to get option chains from eTrade."""

    name: str = "get_option_chains"
    description: str = (
        "Get option chain for a symbol with calls and puts, Greeks, and open interest."
    )
    args_schema: Type[BaseModel] = OptionChainInput

    def _run(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        strike_price_near: Optional[float] = None,
        no_of_strikes: int = 10,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "get_option_chains",
                symbol=symbol,
                expiration_date=expiration_date,
                strike_price_near=strike_price_near,
                no_of_strikes=no_of_strikes,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class PreviewOrderTool(BaseTool):
    """Tool to preview an order before placement."""

    name: str = "preview_order"
    description: str = (
        "Preview an order to see estimated costs. ALWAYS use before placing orders."
    )
    args_schema: Type[BaseModel] = PreviewOrderInput

    def _run(
        self,
        account_id_key: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "LIMIT",
        limit_price: Optional[float] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "preview_order",
                account_id_key=account_id_key,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class PlaceOrderTool(BaseTool):
    """Tool to place an order."""

    name: str = "place_order"
    description: str = (
        "Place an order. ALWAYS preview first. This commits real money in production."
    )
    args_schema: Type[BaseModel] = PlaceOrderInput

    def _run(
        self,
        account_id_key: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "LIMIT",
        limit_price: Optional[float] = None,
        preview_id: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "place_order",
                account_id_key=account_id_key,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                preview_id=preview_id,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetPositionsTool(BaseTool):
    """Tool to get account positions."""

    name: str = "get_positions"
    description: str = "Get current positions for an account with P&L information."
    args_schema: Type[BaseModel] = PositionsInput

    def _run(self, account_id_key: str, symbol: Optional[str] = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "get_positions", account_id_key=account_id_key, symbol=symbol
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetAccountBalanceTool(BaseTool):
    """Tool to get account balance."""

    name: str = "get_account_balance"
    description: str = (
        "Get account balance including cash, buying power, and margin info."
    )
    args_schema: Type[BaseModel] = BalanceInput

    def _run(self, account_id_key: str) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_etrade(
                "get_account_balance", account_id_key=account_id_key
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE MARKET DATA TOOL CLASSES
# =============================================================================


class FetchMarketDataTool(BaseTool):
    """Tool to fetch OHLCV market data from Alpha Vantage."""

    name: str = "fetch_market_data"
    description: str = (
        "Fetch OHLCV market data for a symbol. Use for getting fresh data from Alpha Vantage API."
    )
    args_schema: Type[BaseModel] = FetchMarketDataInput

    def _run(
        self, symbol: str, timeframe: str = "daily", outputsize: str = "compact"
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "fetch_market_data",
                symbol=symbol,
                timeframe=timeframe,
                outputsize=outputsize,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class LoadMarketDataTool(BaseTool):
    """Tool to load OHLCV data from local storage."""

    name: str = "load_market_data"
    description: str = (
        "Load OHLCV data from local DuckDB storage. Faster than fetching from API."
    )
    args_schema: Type[BaseModel] = LoadMarketDataInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "load_market_data",
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ListStoredSymbolsTool(BaseTool):
    """Tool to list all symbols in local database."""

    name: str = "list_stored_symbols"
    description: str = (
        "List all symbols stored in the local database with their available timeframes."
    )

    def _run(self) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore("list_stored_symbols")

        return json.dumps(_run_async(_exec()), indent=2)


class GetSymbolSnapshotTool(BaseTool):
    """Tool to get comprehensive symbol snapshot."""

    name: str = "get_symbol_snapshot"
    description: str = (
        "Get a comprehensive snapshot of a symbol including price, indicators, and regime."
    )
    args_schema: Type[BaseModel] = SymbolSnapshotInput

    def _run(self, symbol: str, end_date: Optional[str] = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_symbol_snapshot", symbol=symbol, end_date=end_date
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE TECHNICAL ANALYSIS TOOL CLASSES
# =============================================================================


class ComputeIndicatorsTool(BaseTool):
    """Tool to compute technical indicators."""

    name: str = "compute_indicators"
    description: str = (
        "Compute technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.) for a symbol."
    )
    args_schema: Type[BaseModel] = IndicatorsInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        indicators: Optional[List[str]] = None,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_technical_indicators",
                symbol=symbol,
                timeframe=timeframe,
                indicators=indicators,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeAllFeaturesTool(BaseTool):
    """Tool to compute all 200+ features."""

    name: str = "compute_all_features"
    description: str = (
        "Compute all available features (200+) including trend, momentum, volatility, volume, and patterns."
    )
    args_schema: Type[BaseModel] = SymbolInput

    def _run(
        self, symbol: str, timeframe: str = "daily", end_date: Optional[str] = None
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_all_features",
                symbol=symbol,
                timeframe=timeframe,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ListAvailableIndicatorsTool(BaseTool):
    """Tool to list all available indicators."""

    name: str = "list_available_indicators"
    description: str = (
        "List all available technical indicators with their descriptions."
    )

    def _run(self) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore("list_available_indicators")

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE BACKTESTING TOOL CLASSES
# =============================================================================


class RunBacktestTool(BaseTool):
    """Tool to run backtests."""

    name: str = "run_backtest"
    description: str = (
        "Run a backtest on historical data to validate a trading strategy."
    )
    args_schema: Type[BaseModel] = BacktestInput

    def _run(
        self,
        symbol: str,
        strategy_type: str = "mean_reversion",
        timeframe: str = "daily",
        initial_capital: float = 100000,
        position_size_pct: float = 10,
        stop_loss_atr: float = 2,
        take_profit_atr: float = 3,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_backtest",
                symbol=symbol,
                strategy_type=strategy_type,
                timeframe=timeframe,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                stop_loss_atr=stop_loss_atr,
                take_profit_atr=take_profit_atr,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetBacktestMetricsTool(BaseTool):
    """Tool to analyze backtest metrics."""

    name: str = "get_backtest_metrics"
    description: str = (
        "Analyze and interpret backtest metrics to assess strategy quality."
    )
    args_schema: Type[BaseModel] = BacktestMetricsInput

    def _run(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_backtest_metrics",
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class RunMonteCarloTool(BaseTool):
    """Tool to run Monte Carlo simulations."""

    name: str = "run_monte_carlo"
    description: str = (
        "Run Monte Carlo simulation to test strategy robustness under varying conditions."
    )
    args_schema: Type[BaseModel] = MonteCarloInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        n_simulations: int = 1000,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_monte_carlo",
                symbol=symbol,
                timeframe=timeframe,
                n_simulations=n_simulations,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class RunWalkForwardTool(BaseTool):
    """Tool to run walk-forward optimization."""

    name: str = "run_walkforward"
    description: str = (
        "Run walk-forward optimization to validate strategy out-of-sample."
    )
    args_schema: Type[BaseModel] = WalkForwardInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        n_splits: int = 5,
        train_pct: float = 0.7,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_walkforward",
                symbol=symbol,
                timeframe=timeframe,
                n_splits=n_splits,
                train_pct=train_pct,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE STATISTICAL ANALYSIS TOOL CLASSES
# =============================================================================


class RunADFTestTool(BaseTool):
    """Tool to run Augmented Dickey-Fuller stationarity test."""

    name: str = "run_adf_test"
    description: str = (
        "Run ADF test to check if a time series is stationary (mean-reverting). p-value < 0.05 indicates stationarity."
    )
    args_schema: Type[BaseModel] = ADFTestInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        column: str = "close",
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_adf_test",
                symbol=symbol,
                timeframe=timeframe,
                column=column,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeAlphaDecayTool(BaseTool):
    """Tool to analyze signal alpha decay."""

    name: str = "compute_alpha_decay"
    description: str = (
        "Analyze how a trading signal's predictive power decays over time. Returns optimal holding period."
    )
    args_schema: Type[BaseModel] = AlphaDecayInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        signal_column: str = "rsi_14",
        max_lag: int = 20,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_alpha_decay",
                symbol=symbol,
                timeframe=timeframe,
                signal_column=signal_column,
                max_lag=max_lag,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeInformationCoefficientTool(BaseTool):
    """Tool to compute Information Coefficient."""

    name: str = "compute_information_coefficient"
    description: str = (
        "Compute IC between a signal and forward returns. IC > 0.05 is generally meaningful."
    )
    args_schema: Type[BaseModel] = ICInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        signal_column: str = "rsi_14",
        forward_return_periods: int = 5,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_information_coefficient",
                symbol=symbol,
                timeframe=timeframe,
                signal_column=signal_column,
                forward_return_periods=forward_return_periods,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ValidateSignalTool(BaseTool):
    """Tool to validate a trading signal."""

    name: str = "validate_signal"
    description: str = (
        "Validate a trading signal for statistical significance and predictive power."
    )
    args_schema: Type[BaseModel] = ValidateSignalInput

    def _run(self, symbol: str, signal_column: str, timeframe: str = "daily") -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "validate_signal",
                symbol=symbol,
                signal_column=signal_column,
                timeframe=timeframe,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class DiagnoseSignalTool(BaseTool):
    """Tool to diagnose signal issues."""

    name: str = "diagnose_signal"
    description: str = (
        "Diagnose potential issues with a trading signal (noise, lag, correlation)."
    )
    args_schema: Type[BaseModel] = DiagnoseSignalInput

    def _run(self, symbol: str, signal_column: str, timeframe: str = "daily") -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "diagnose_signal",
                symbol=symbol,
                signal_column=signal_column,
                timeframe=timeframe,
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE OPTIONS TOOL CLASSES
# =============================================================================


class PriceOptionTool(BaseTool):
    """Tool to price options using Black-Scholes."""

    name: str = "price_option"
    description: str = (
        "Calculate option price using Black-Scholes-Merton model. Returns price and Greeks."
    )
    args_schema: Type[BaseModel] = PriceOptionInput

    def _run(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
        dividend_yield: float = 0,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "price_option",
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeGreeksTool(BaseTool):
    """Tool to compute option Greeks."""

    name: str = "compute_greeks"
    description: str = (
        "Compute option Greeks (Delta, Gamma, Theta, Vega, Rho) with interpretations."
    )
    args_schema: Type[BaseModel] = ComputeGreeksInput

    def _run(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
        dividend_yield: float = 0,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_greeks",
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeImpliedVolTool(BaseTool):
    """Tool to compute implied volatility."""

    name: str = "compute_implied_vol"
    description: str = (
        "Calculate implied volatility from market option price using Newton-Raphson method."
    )
    args_schema: Type[BaseModel] = ImpliedVolInput

    def _run(
        self,
        option_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_implied_vol",
                option_price=option_price,
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class AnalyzeOptionStructureTool(BaseTool):
    """Tool to analyze option structures."""

    name: str = "analyze_option_structure"
    description: str = (
        "Analyze an options structure (spread, condor, etc.) for P&L, breakevens, and Greeks."
    )
    args_schema: Type[BaseModel] = AnalyzeOptionStructureInput

    def _run(
        self,
        structure_type: str,
        legs: str,
        spot: float,
        volatility: float = 0.25,
        time_to_expiry: float = 0.083,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            legs_list = json.loads(legs)
            return await bridge.call_quantcore(
                "analyze_option_structure",
                structure_type=structure_type,
                legs=legs_list,
                spot=spot,
                volatility=volatility,
                time_to_expiry=time_to_expiry,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeOptionChainTool(BaseTool):
    """Tool to compute theoretical option chain."""

    name: str = "compute_option_chain"
    description: str = (
        "Compute a theoretical option chain with prices and Greeks for multiple strikes."
    )
    args_schema: Type[BaseModel] = ComputeOptionChainInput

    def _run(
        self,
        symbol: str,
        spot_price: float,
        volatility: float = 0.25,
        days_to_expiry: int = 30,
        num_strikes: int = 10,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_option_chain",
                symbol=symbol,
                spot_price=spot_price,
                volatility=volatility,
                days_to_expiry=days_to_expiry,
                num_strikes=num_strikes,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeMultiLegPriceTool(BaseTool):
    """Tool to compute multi-leg option structure price."""

    name: str = "compute_multi_leg_price"
    description: str = (
        "Calculate net debit/credit and combined Greeks for a multi-leg options trade."
    )
    args_schema: Type[BaseModel] = MultiLegPriceInput

    def _run(self, legs: str, spot: float) -> str:
        async def _exec():
            bridge = get_bridge()
            legs_list = json.loads(legs)
            return await bridge.call_quantcore(
                "compute_multi_leg_price", legs=legs_list, spot=spot
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE RISK MANAGEMENT TOOL CLASSES
# =============================================================================


class ComputePositionSizeTool(BaseTool):
    """Tool to calculate position size."""

    name: str = "compute_position_size"
    description: str = (
        "Calculate optimal position size using ATR-based risk management and Kelly criterion."
    )
    args_schema: Type[BaseModel] = PositionSizeInput

    def _run(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: float = 1,
        max_position_pct: float = 20,
        alignment_score: float = 1,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_position_size",
                equity=equity,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
                alignment_score=alignment_score,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeMaxDrawdownTool(BaseTool):
    """Tool to compute maximum drawdown."""

    name: str = "compute_max_drawdown"
    description: str = (
        "Compute maximum drawdown and drawdown statistics from equity curve."
    )
    args_schema: Type[BaseModel] = MaxDrawdownInput

    def _run(self, equity_curve: List[float]) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_max_drawdown", equity_curve=equity_curve
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputePortfolioStatsTool(BaseTool):
    """Tool to compute portfolio statistics."""

    name: str = "compute_portfolio_stats"
    description: str = (
        "Compute portfolio statistics including Sharpe ratio, volatility, and risk metrics."
    )
    args_schema: Type[BaseModel] = PortfolioStatsInput

    def _run(self, returns: List[float], risk_free_rate: float = 0.02) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_portfolio_stats",
                returns=returns,
                risk_free_rate=risk_free_rate,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ComputeVaRTool(BaseTool):
    """Tool to compute Value at Risk."""

    name: str = "compute_var"
    description: str = "Compute Value at Risk (VaR) using historical simulation method."
    args_schema: Type[BaseModel] = VaRInput

    def _run(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
        portfolio_value: float = 100000,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "compute_var",
                returns=returns,
                confidence_level=confidence_level,
                portfolio_value=portfolio_value,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class StressTestPortfolioTool(BaseTool):
    """Tool to stress test portfolio."""

    name: str = "stress_test_portfolio"
    description: str = "Run stress tests on portfolio with various market scenarios."
    args_schema: Type[BaseModel] = StressTestInput

    def _run(self, positions: str, scenarios: str) -> str:
        async def _exec():
            bridge = get_bridge()
            pos_list = json.loads(positions)
            scen_list = json.loads(scenarios)
            return await bridge.call_quantcore(
                "stress_test_portfolio", positions=pos_list, scenarios=scen_list
            )

        return json.dumps(_run_async(_exec()), indent=2)


class CheckRiskLimitsTool(BaseTool):
    """Tool to check risk limits."""

    name: str = "check_risk_limits"
    description: str = "Check if portfolio Greeks are within risk limits."
    args_schema: Type[BaseModel] = RiskLimitsInput

    def _run(
        self,
        portfolio_delta: float,
        portfolio_gamma: float,
        portfolio_vega: float,
        max_delta: float = 100,
        max_gamma: float = 50,
        max_vega: float = 5000,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "check_risk_limits",
                portfolio_delta=portfolio_delta,
                portfolio_gamma=portfolio_gamma,
                portfolio_vega=portfolio_vega,
                max_delta=max_delta,
                max_gamma=max_gamma,
                max_vega=max_vega,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class AnalyzeLiquidityTool(BaseTool):
    """Tool to analyze market liquidity."""

    name: str = "analyze_liquidity"
    description: str = (
        "Analyze market liquidity including bid-ask spread, volume, and market impact."
    )
    args_schema: Type[BaseModel] = LiquidityInput

    def _run(
        self, symbol: str, timeframe: str = "daily", end_date: Optional[str] = None
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "analyze_liquidity",
                symbol=symbol,
                timeframe=timeframe,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE MARKET/REGIME TOOL CLASSES
# =============================================================================


class GetMarketRegimeSnapshotTool(BaseTool):
    """Tool to get market regime snapshot."""

    name: str = "get_market_regime_snapshot"
    description: str = (
        "Get current market regime classification (trending, ranging, volatile) with confidence."
    )

    def _run(self, end_date: Optional[str] = None) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_market_regime_snapshot", end_date=end_date
            )

        return json.dumps(_run_async(_exec()), indent=2)


class AnalyzeVolumeProfileTool(BaseTool):
    """Tool to analyze volume profile."""

    name: str = "analyze_volume_profile"
    description: str = (
        "Analyze volume profile to identify support/resistance levels and value areas."
    )
    args_schema: Type[BaseModel] = VolumeProfileInput

    def _run(
        self,
        symbol: str,
        timeframe: str = "daily",
        num_bins: int = 20,
        end_date: Optional[str] = None,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "analyze_volume_profile",
                symbol=symbol,
                timeframe=timeframe,
                num_bins=num_bins,
                end_date=end_date,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetTradingCalendarTool(BaseTool):
    """Tool to get trading calendar."""

    name: str = "get_trading_calendar"
    description: str = "Get trading calendar with market holidays and trading days."
    args_schema: Type[BaseModel] = TradingCalendarInput

    def _run(self, start_date: str, end_date: str) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_trading_calendar", start_date=start_date, end_date=end_date
            )

        return json.dumps(_run_async(_exec()), indent=2)


class GetEventCalendarTool(BaseTool):
    """Tool to get event calendar."""

    name: str = "get_event_calendar"
    description: str = (
        "Get upcoming market events (earnings, Fed meetings, economic releases)."
    )
    args_schema: Type[BaseModel] = EventCalendarInput

    def _run(self, symbol: Optional[str] = None, days_ahead: int = 7) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "get_event_calendar", symbol=symbol, days_ahead=days_ahead
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# QUANTCORE TRADE TOOL CLASSES
# =============================================================================


class GenerateTradeTemplateTool(BaseTool):
    """Tool to generate trade template."""

    name: str = "generate_trade_template"
    description: str = (
        "Generate a trade template with entry, stop, target based on symbol analysis."
    )
    args_schema: Type[BaseModel] = TradeTemplateInput

    def _run(
        self,
        symbol: str,
        direction: str,
        structure_type: str = "VERTICAL_SPREAD",
        max_risk: float = 500,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "generate_trade_template",
                symbol=symbol,
                direction=direction,
                structure_type=structure_type,
                max_risk=max_risk,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ValidateTradeTool(BaseTool):
    """Tool to validate a trade setup."""

    name: str = "validate_trade"
    description: str = (
        "Validate a trade setup against risk rules and market conditions."
    )
    args_schema: Type[BaseModel] = ValidateTradeInput

    def _run(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        position_size: float,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "validate_trade",
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                position_size=position_size,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class ScoreTradeStructureTool(BaseTool):
    """Tool to score a trade structure."""

    name: str = "score_trade_structure"
    description: str = (
        "Score an options trade structure based on risk/reward, probability, and Greeks."
    )
    args_schema: Type[BaseModel] = ScoreTradeInput

    def _run(
        self,
        structure_type: str,
        max_profit: float,
        max_loss: float,
        probability_of_profit: float,
        days_to_expiry: int,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "score_trade_structure",
                structure_type=structure_type,
                max_profit=max_profit,
                max_loss=max_loss,
                probability_of_profit=probability_of_profit,
                days_to_expiry=days_to_expiry,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class SimulateTradeOutcomeTool(BaseTool):
    """Tool to simulate trade outcomes."""

    name: str = "simulate_trade_outcome"
    description: str = (
        "Simulate potential trade outcomes using Monte Carlo based on historical volatility."
    )
    args_schema: Type[BaseModel] = SimulateTradeInput

    def _run(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: int,
        days_to_hold: int = 20,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "simulate_trade_outcome",
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                days_to_hold=days_to_hold,
            )

        return json.dumps(_run_async(_exec()), indent=2)


class RunScreenerTool(BaseTool):
    """Tool to run market screener."""

    name: str = "run_screener"
    description: str = "Screen stocks based on technical and price criteria."
    args_schema: Type[BaseModel] = ScreenerInput

    def _run(
        self,
        min_price: float = 10,
        max_price: float = 500,
        min_volume: int = 1000000,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ) -> str:
        async def _exec():
            bridge = get_bridge()
            return await bridge.call_quantcore(
                "run_screener",
                min_price=min_price,
                max_price=max_price,
                min_volume=min_volume,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
            )

        return json.dumps(_run_async(_exec()), indent=2)


# =============================================================================
# TOOL FACTORY FUNCTIONS - eTrade
# =============================================================================


def get_quote_tool() -> GetQuoteTool:
    """Get the quote tool instance."""
    return GetQuoteTool()


def get_option_chains_tool() -> GetOptionChainsTool:
    """Get the option chains tool instance."""
    return GetOptionChainsTool()


def preview_order_tool() -> PreviewOrderTool:
    """Get the preview order tool instance."""
    return PreviewOrderTool()


def place_order_tool() -> PlaceOrderTool:
    """Get the place order tool instance."""
    return PlaceOrderTool()


def get_positions_tool() -> GetPositionsTool:
    """Get the positions tool instance."""
    return GetPositionsTool()


def get_account_balance_tool() -> GetAccountBalanceTool:
    """Get the account balance tool instance."""
    return GetAccountBalanceTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Market Data
# =============================================================================


def fetch_market_data_tool() -> FetchMarketDataTool:
    """Get the fetch market data tool instance."""
    return FetchMarketDataTool()


def load_market_data_tool() -> LoadMarketDataTool:
    """Get the load market data tool instance."""
    return LoadMarketDataTool()


def list_stored_symbols_tool() -> ListStoredSymbolsTool:
    """Get the list stored symbols tool instance."""
    return ListStoredSymbolsTool()


def get_symbol_snapshot_tool() -> GetSymbolSnapshotTool:
    """Get the symbol snapshot tool instance."""
    return GetSymbolSnapshotTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Technical Analysis
# =============================================================================


def compute_indicators_tool() -> ComputeIndicatorsTool:
    """Get the indicators tool instance."""
    return ComputeIndicatorsTool()


def compute_all_features_tool() -> ComputeAllFeaturesTool:
    """Get the all features tool instance."""
    return ComputeAllFeaturesTool()


def list_available_indicators_tool() -> ListAvailableIndicatorsTool:
    """Get the list indicators tool instance."""
    return ListAvailableIndicatorsTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Backtesting
# =============================================================================


def run_backtest_tool() -> RunBacktestTool:
    """Get the backtest tool instance."""
    return RunBacktestTool()


def get_backtest_metrics_tool() -> GetBacktestMetricsTool:
    """Get the backtest metrics tool instance."""
    return GetBacktestMetricsTool()


def run_monte_carlo_tool() -> RunMonteCarloTool:
    """Get the Monte Carlo tool instance."""
    return RunMonteCarloTool()


def run_walkforward_tool() -> RunWalkForwardTool:
    """Get the walk-forward tool instance."""
    return RunWalkForwardTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Statistical
# =============================================================================


def run_adf_test_tool() -> RunADFTestTool:
    """Get the ADF test tool instance."""
    return RunADFTestTool()


def compute_alpha_decay_tool() -> ComputeAlphaDecayTool:
    """Get the alpha decay tool instance."""
    return ComputeAlphaDecayTool()


def compute_information_coefficient_tool() -> ComputeInformationCoefficientTool:
    """Get the IC tool instance."""
    return ComputeInformationCoefficientTool()


def validate_signal_tool() -> ValidateSignalTool:
    """Get the validate signal tool instance."""
    return ValidateSignalTool()


def diagnose_signal_tool() -> DiagnoseSignalTool:
    """Get the diagnose signal tool instance."""
    return DiagnoseSignalTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Options
# =============================================================================


def price_option_tool() -> PriceOptionTool:
    """Get the price option tool instance."""
    return PriceOptionTool()


def compute_greeks_tool() -> ComputeGreeksTool:
    """Get the Greeks tool instance."""
    return ComputeGreeksTool()


def compute_implied_vol_tool() -> ComputeImpliedVolTool:
    """Get the implied vol tool instance."""
    return ComputeImpliedVolTool()


def analyze_option_structure_tool() -> AnalyzeOptionStructureTool:
    """Get the option structure analysis tool instance."""
    return AnalyzeOptionStructureTool()


def compute_option_chain_tool() -> ComputeOptionChainTool:
    """Get the option chain tool instance."""
    return ComputeOptionChainTool()


def compute_multi_leg_price_tool() -> ComputeMultiLegPriceTool:
    """Get the multi-leg price tool instance."""
    return ComputeMultiLegPriceTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Risk Management
# =============================================================================


def compute_position_size_tool() -> ComputePositionSizeTool:
    """Get the position size tool instance."""
    return ComputePositionSizeTool()


def compute_max_drawdown_tool() -> ComputeMaxDrawdownTool:
    """Get the max drawdown tool instance."""
    return ComputeMaxDrawdownTool()


def compute_portfolio_stats_tool() -> ComputePortfolioStatsTool:
    """Get the portfolio stats tool instance."""
    return ComputePortfolioStatsTool()


def compute_var_tool() -> ComputeVaRTool:
    """Get the VaR tool instance."""
    return ComputeVaRTool()


def stress_test_portfolio_tool() -> StressTestPortfolioTool:
    """Get the stress test tool instance."""
    return StressTestPortfolioTool()


def check_risk_limits_tool() -> CheckRiskLimitsTool:
    """Get the risk limits tool instance."""
    return CheckRiskLimitsTool()


def analyze_liquidity_tool() -> AnalyzeLiquidityTool:
    """Get the liquidity analysis tool instance."""
    return AnalyzeLiquidityTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Market/Regime
# =============================================================================


def get_market_regime_snapshot_tool() -> GetMarketRegimeSnapshotTool:
    """Get the market regime snapshot tool instance."""
    return GetMarketRegimeSnapshotTool()


def analyze_volume_profile_tool() -> AnalyzeVolumeProfileTool:
    """Get the volume profile tool instance."""
    return AnalyzeVolumeProfileTool()


def get_trading_calendar_tool() -> GetTradingCalendarTool:
    """Get the trading calendar tool instance."""
    return GetTradingCalendarTool()


def get_event_calendar_tool() -> GetEventCalendarTool:
    """Get the event calendar tool instance."""
    return GetEventCalendarTool()


# =============================================================================
# TOOL FACTORY FUNCTIONS - QuantCore Trade
# =============================================================================


def generate_trade_template_tool() -> GenerateTradeTemplateTool:
    """Get the trade template tool instance."""
    return GenerateTradeTemplateTool()


def validate_trade_tool() -> ValidateTradeTool:
    """Get the validate trade tool instance."""
    return ValidateTradeTool()


def score_trade_structure_tool() -> ScoreTradeStructureTool:
    """Get the score trade tool instance."""
    return ScoreTradeStructureTool()


def simulate_trade_outcome_tool() -> SimulateTradeOutcomeTool:
    """Get the simulate trade tool instance."""
    return SimulateTradeOutcomeTool()


def run_screener_tool() -> RunScreenerTool:
    """Get the screener tool instance."""
    return RunScreenerTool()
