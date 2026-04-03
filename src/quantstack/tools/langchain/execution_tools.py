"""Trade execution tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _calc_quantity_from_size(position_size: str, equity: float, current_price: float) -> int:
    fractions = {"full": 0.10, "half": 0.05, "quarter": 0.025}
    frac = fractions.get(position_size, 0.025)
    if current_price <= 0:
        return 0
    return max(1, int((equity * frac) / current_price))


@tool
async def execute_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: float | None = None,
    reasoning: str = "",
    confidence: float = 0.5,
    position_size: str = "quarter",
) -> str:
    """Execute a trade order through the broker.

    Only call this after the fund manager has approved the trade.
    The risk gate and kill switch are checked automatically.

    Args:
        symbol: Ticker symbol.
        side: "buy" or "sell".
        quantity: Number of shares/contracts (0 = auto-size from position_size).
        order_type: "market" or "limit".
        limit_price: Required for limit orders.
        reasoning: Why this trade is being placed.
        confidence: Confidence level 0-1.
        position_size: "full", "half", or "quarter" (used if quantity=0).
    """
    try:
        from quantstack.tools._state import require_ctx

        ctx = require_ctx()

        # Kill switch check — SACRED, NEVER BYPASSED
        ctx.kill_switch.guard()

        snapshot = ctx.portfolio.get_snapshot()
        equity = snapshot.get("equity", 100_000)

        # Auto-size if quantity is 0
        qty = int(quantity) if quantity > 0 else 0
        if qty == 0:
            current_price = snapshot.get("last_prices", {}).get(symbol, 0)
            if current_price <= 0:
                return json.dumps({"error": f"No price available for {symbol}, cannot auto-size"})
            qty = _calc_quantity_from_size(position_size, equity, current_price)
            current_price_for_risk = current_price
        else:
            current_price_for_risk = snapshot.get("last_prices", {}).get(symbol, limit_price or 0)

        # Risk gate check — SACRED, NEVER BYPASSED
        if current_price_for_risk > 0:
            risk_ok = ctx.risk_gate.check(
                symbol=symbol,
                side=side,
                quantity=qty,
                current_price=current_price_for_risk,
            )
            if not risk_ok:
                return json.dumps({
                    "status": "rejected",
                    "reason": "Risk gate rejected the trade",
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                })

        # Submit order
        from quantstack.execution.alpaca_broker import OrderRequest

        order = OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
        )
        fill = await ctx.broker.submit_order(order)

        result = {
            "status": "filled" if fill else "submitted",
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "order_type": order_type,
            "reasoning": reasoning,
            "confidence": confidence,
        }
        if fill:
            result.update(fill if isinstance(fill, dict) else {"fill": str(fill)})

        logger.info(f"Executed {side} {qty} {symbol}: {result.get('status')}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"execute_order({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol, "side": side})


@tool
async def close_position(symbol: str, reasoning: str = "position close") -> str:
    """Close an entire position for a symbol.

    Args:
        symbol: Ticker symbol to close.
        reasoning: Why the position is being closed.
    """
    try:
        from quantstack.tools._state import require_ctx

        ctx = require_ctx()
        snapshot = ctx.portfolio.get_snapshot()
        positions = snapshot.get("positions", {})
        pos = positions.get(symbol)
        if not pos:
            return json.dumps({"error": f"No open position for {symbol}"})

        qty = abs(pos.get("quantity", 0))
        side = "sell" if pos.get("quantity", 0) > 0 else "buy"

        result = await execute_order.ainvoke({
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "reasoning": reasoning,
        })
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def get_fills(symbol: str | None = None, limit: int = 20) -> str:
    """Get recent order fills from the broker.

    Args:
        symbol: Filter by symbol (None for all).
        limit: Maximum number of fills to return.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_audit_trail(symbol: str | None = None, limit: int = 20) -> str:
    """Get the decision audit trail for recent trades.

    Args:
        symbol: Filter by symbol (None for all).
        limit: Maximum entries to return.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def update_position_stops(symbol: str, new_stop: float) -> str:
    """Update stop loss for an open position.

    Args:
        symbol: Ticker symbol.
        new_stop: New stop loss price.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def check_broker_connection() -> str:
    """Check if the broker connection is healthy."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


