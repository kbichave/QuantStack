"""Trade execution tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def execute_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
) -> str:
    """Execute a trade order through the broker.

    Only call this after the fund manager has approved the trade.
    The risk gate and kill switch are checked automatically.

    Args:
        symbol: Ticker symbol.
        side: "buy" or "sell".
        quantity: Number of shares/contracts.
        order_type: "market" or "limit".
        limit_price: Required for limit orders.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "execute_trade",
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
    )
    return json.dumps(result, default=str)
