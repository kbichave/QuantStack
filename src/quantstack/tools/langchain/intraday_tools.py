"""Intraday execution tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def get_intraday_status() -> str:
    """Retrieves the current intraday trading loop status including open positions, realized P&L, and execution metrics. Use when monitoring daytrading activity, checking whether the intraday scanner is running, or reviewing session performance. Returns JSON with running state, positions_held count, realized_pnl dollar amount, trades_today count, bars_processed, flattened flag, and active symbols list."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_tca_report(
    lookback_days: Annotated[int, Field(description="Number of historical days to include in the TCA analysis window, e.g. 7, 30, 90")] = 30,
    symbol: Annotated[str | None, Field(description="Optional ticker symbol filter to restrict TCA report to a single stock, e.g. 'AAPL'. Use None for all symbols")] = None,
) -> str:
    """Retrieves aggregate Transaction Cost Analysis (TCA) statistics measuring execution quality, slippage, and fill performance. Use when reviewing trade execution efficiency, tracking slippage trends over time, identifying worst fills, or evaluating algorithm recommendation accuracy. Computes average slippage in basis points, algo-level breakdowns, and an overall execution_quality verdict. Returns JSON with avg_slippage_bps, worst_fills list, algo_breakdown, execution_quality score, and total trade count."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_algo_recommendation(
    symbol: Annotated[str, Field(description="Ticker symbol to get execution algorithm recommendation for, e.g. 'AAPL', 'SPY'")],
    side: Annotated[str, Field(description="Trade direction: 'buy' for purchase or 'sell' for liquidation")],
    shares: Annotated[float, Field(description="Number of shares to trade in this order")],
    current_price: Annotated[float, Field(description="Current last-trade price of the stock in dollars")],
    adv: Annotated[float, Field(description="Average daily volume in shares, used to gauge liquidity and market impact")],
    daily_vol_pct: Annotated[float, Field(description="Daily return volatility as a percentage, e.g. 1.5 for 1.5% daily vol")],
    spread_bps: Annotated[float, Field(description="Current bid-ask spread in basis points, e.g. 5.0 for typical liquid stocks")] = 5.0,
    urgency: Annotated[str, Field(description="Execution urgency level: 'stop_loss', 'high', 'normal', or 'low'")] = "normal",
    vix: Annotated[float, Field(description="Current VIX volatility index level; use 0 if unknown")] = 0.0,
    earnings_within_24h: Annotated[bool, Field(description="Whether an earnings report is expected within 24 hours, triggering special handling")] = False,
    bid: Annotated[float | None, Field(description="Current best bid price in dollars; improves limit price calculation when provided")] = None,
    ask: Annotated[float | None, Field(description="Current best ask price in dollars; improves limit price calculation when provided")] = None,
) -> str:
    """Provides an urgency-aware execution algorithm recommendation with cost estimates for optimal order routing. Use when deciding between TWAP, VWAP, limit, or market orders based on liquidity, volatility, and special situations. Computes pre-trade cost forecasts and applies override rules for stop-loss exits, high VIX regimes, earnings proximity, and low-liquidity tickers. Returns JSON with recommended_algo name, limit_price, urgency classification, expected_cost_bps, override_reason, execution_window, and detailed TCA forecast breakdown."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
