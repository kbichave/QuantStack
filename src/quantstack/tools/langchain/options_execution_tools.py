"""Options execution tools for LangGraph agents."""

import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field


@tool
async def execute_options_trade(
    symbol: Annotated[str, Field(description="Underlying stock ticker symbol for the options contract, e.g. 'SPY', 'AAPL', 'QQQ'")],
    option_type: Annotated[str, Field(description="Contract type: 'call' for call option or 'put' for put option")],
    strike: Annotated[float, Field(description="Strike price (exercise price) of the options contract in dollars")],
    expiry_date: Annotated[str, Field(description="Options expiration date in YYYY-MM-DD format, e.g. '2026-04-18'")],
    action: Annotated[str, Field(description="Trade action: 'buy' to go long (purchase) or 'sell' to go short (write) the contract")],
    contracts: Annotated[int, Field(description="Number of options contracts to trade; each contract represents 100 shares of underlying")],
    reasoning: Annotated[str, Field(description="Required trade justification explaining the thesis, catalyst, and expected outcome")],
    confidence: Annotated[float, Field(description="Required confidence score from 0.0 (no confidence) to 1.0 (maximum conviction)")],
    strategy_id: Annotated[Optional[str], Field(description="Optional strategy identifier to link this trade to a registered strategy for tracking")] = None,
    order_type: Annotated[str, Field(description="Order type: 'market' for immediate fill or 'limit' for price-constrained execution")] = "market",
    limit_price: Annotated[Optional[float], Field(description="Per-contract limit price in dollars; required when order_type is 'limit'")] = None,
    paper_mode: Annotated[bool, Field(description="Paper trading mode flag; must be explicitly set to False for live order execution")] = True,
) -> str:
    """Executes an options trade (call or put) through the mandatory risk gate and broker interface. Use when placing options orders for directional bets, hedging, or volatility strategies. Validates premium-at-risk limits, days-to-expiration (DTE) bounds, and daily loss caps before routing to the broker. Computes Black-Scholes theoretical pricing with 20-day realized volatility in paper mode. Returns JSON with fill details including execution price, total premium paid or received, option Greeks (delta, gamma, theta, vega), or a rejection reason if the risk gate blocks the trade."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
