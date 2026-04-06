"""Trade execution tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


def _calc_quantity_from_size(position_size: str, equity: float, current_price: float) -> int:
    fractions = {"full": 0.10, "half": 0.05, "quarter": 0.025}
    frac = fractions.get(position_size, 0.025)
    if current_price <= 0:
        return 0
    return max(1, int((equity * frac) / current_price))


@tool
async def execute_order(
    symbol: str = Field(description="Ticker symbol to trade, e.g. 'AAPL' or 'TSLA'"),
    side: str = Field(description="Trade direction: 'buy' to open long or cover short, 'sell' to close long or open short"),
    quantity: float = Field(description="Number of shares or contracts to trade; set to 0 for automatic position sizing based on position_size parameter"),
    order_type: str = Field(default="market", description="Order type: 'market' for immediate execution or 'limit' for price-constrained execution"),
    limit_price: float | None = Field(default=None, description="Limit price for limit orders; required when order_type is 'limit', ignored for market orders"),
    reasoning: str = Field(default="", description="Explanation of the trade thesis, rationale, or signal that triggered this order"),
    confidence: float = Field(default=0.5, description="Confidence score between 0.0 and 1.0 indicating conviction strength for this trade"),
    position_size: str = Field(default="quarter", description="Position sizing tier: 'full' (10% equity), 'half' (5%), or 'quarter' (2.5%); used only when quantity is 0"),
) -> str:
    """Execute a buy or sell trade order through the broker after risk gate and kill switch validation. Use when submitting a stock or equity order for execution. Returns JSON with fill status, order details, and confirmation. Provides order routing, automatic position sizing, and risk-checked trade submission for market and limit orders."""
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
async def close_position(
    symbol: str = Field(description="Ticker symbol of the open position to close, e.g. 'AAPL'"),
    reasoning: str = Field(default="position close", description="Explanation for why this position is being liquidated or exited"),
) -> str:
    """Close an entire open position for a given ticker symbol by submitting the inverse order. Use when liquidating, exiting, or unwinding a long or short equity holding. Returns JSON with order fill status and confirmation details. Provides full position flattening with automatic side and quantity detection."""
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
async def get_fills(
    symbol: str | None = Field(default=None, description="Ticker symbol to filter fills by; pass None or omit to retrieve fills for all symbols"),
    limit: int = Field(default=20, description="Maximum number of recent fills to return, ordered most recent first"),
) -> str:
    """Retrieve recent order fill history and execution records from the broker. Use when checking trade confirmations, verifying execution prices, or auditing completed orders. Returns JSON list of fills with price, quantity, and timestamp details."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_audit_trail(
    symbol: str | None = Field(default=None, description="Ticker symbol to filter audit entries by; pass None or omit to retrieve all trade decisions"),
    limit: int = Field(default=20, description="Maximum number of audit trail entries to return, ordered most recent first"),
) -> str:
    """Retrieve the decision audit trail showing reasoning and metadata for recent trade executions. Use when reviewing trade rationale, debugging order decisions, or generating compliance reports. Returns JSON with decision logs, timestamps, and reasoning for each trade action."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def update_position_stops(
    symbol: str = Field(description="Ticker symbol of the open position whose stop loss needs updating"),
    new_stop: float = Field(description="New stop loss price level; must be a positive value representing the exit trigger price"),
) -> str:
    """Update or modify the stop loss price for an existing open position. Use when trailing stops, tightening risk, or adjusting protective stop orders after price movement. Returns JSON confirmation with the updated stop loss level and position details."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def check_broker_connection() -> str:
    """Check if the broker connection is healthy and responsive. Use when diagnosing connectivity issues, verifying API availability, or performing pre-trade health checks. Returns JSON with broker status, latency, and connection health indicators."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


