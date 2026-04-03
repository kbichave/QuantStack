"""Options execution tools for LangGraph agents."""

import json
from typing import Optional

from langchain_core.tools import tool


@tool
async def execute_options_trade(
    symbol: str,
    option_type: str,
    strike: float,
    expiry_date: str,
    action: str,
    contracts: int,
    reasoning: str,
    confidence: float,
    strategy_id: Optional[str] = None,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    paper_mode: bool = True,
) -> str:
    """Execute an options trade through the risk gate and broker.

    The risk gate checks premium-at-risk, DTE bounds, and daily loss limits.
    Paper mode uses Black-Scholes pricing with 20-day realized vol.

    Args:
        symbol: Underlying ticker (e.g., "SPY").
        option_type: "call" or "put".
        strike: Strike price.
        expiry_date: Expiration date (YYYY-MM-DD).
        action: "buy" or "sell" (buy = long, sell = short/write).
        contracts: Number of contracts (each = 100 shares).
        reasoning: REQUIRED. Why you are making this trade.
        confidence: REQUIRED. 0-1 confidence score.
        strategy_id: Links trade to a registered strategy.
        order_type: "market" or "limit".
        limit_price: Per-contract limit price (required for limit orders).
        paper_mode: Must be explicitly False for live trading.

    Returns JSON with fill details, premium paid/received, Greeks, or rejection reason.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
