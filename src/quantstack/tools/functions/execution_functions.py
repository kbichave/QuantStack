"""Execution functions called directly by graph nodes."""

from typing import Any

from loguru import logger

from quantstack.tools._state import live_db_or_error, _serialize


async def submit_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
) -> dict[str, Any]:
    """Submit an order through the broker.

    Called by the execute_entries node after all gates are passed.
    Uses the same risk gate + broker execution path as the full execute_trade,
    with sensible defaults for audit parameters.
    """
    from quantstack.tools.langchain.execution_tools import (
        execute_order,
    )

    # execute_order is a @tool that returns JSON; we need a dict
    import json

    result_str = await execute_order.ainvoke({
        "symbol": symbol,
        "action": side,
        "reasoning": "graph-node-auto-order",
        "confidence": 1.0,
        "quantity": int(quantity),
        "order_type": order_type,
        "limit_price": limit_price,
        "paper_mode": True,
    })
    try:
        return json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return {"success": False, "error": f"Unexpected result: {result_str}"}


async def close_position(symbol: str, reason: str) -> dict[str, Any]:
    """Close an open position.

    Called by the execute_exits node.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        pos = ctx.portfolio.get_position(symbol)
        if pos is None:
            return {"success": False, "error": f"No open position for {symbol}"}

        close_qty = abs(pos.quantity)
        close_action = "sell" if pos.side == "long" else "buy"

        return await submit_order(
            symbol=symbol,
            side=close_action,
            quantity=close_qty,
            order_type="market",
        )
    except Exception as e:
        logger.error(f"close_position({symbol}) failed: {e}")
        return {"success": False, "error": str(e)}
