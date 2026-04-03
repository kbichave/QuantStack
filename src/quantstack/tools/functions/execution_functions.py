"""Execution functions called directly by graph nodes."""

from typing import Any

from quantstack.tools.mcp_bridge._bridge import get_bridge


async def submit_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
) -> dict[str, Any]:
    """Submit an order through the broker.

    Called by the execute_entries node after all gates are passed.
    Raises on critical failure (broker unreachable).
    """
    bridge = get_bridge()
    return await bridge.call_quantcore(
        "execute_trade",
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
    )


async def close_position(symbol: str, reason: str) -> dict[str, Any]:
    """Close an open position.

    Called by the execute_exits node.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore(
        "close_position", symbol=symbol, reason=reason
    )
