"""Portfolio management tools for LangGraph agents."""

import json

from langchain_core.tools import tool
from loguru import logger

from quantstack.tools._state import live_db_or_error, _serialize


@tool
async def fetch_portfolio() -> str:
    """Get current portfolio state including positions, P&L, and exposure.

    Returns JSON with holdings, unrealized P&L, cash balance,
    and gross/net exposure.
    """
    ctx, err = live_db_or_error()
    if err:
        return json.dumps(err, default=str)
    try:
        snapshot = _serialize(ctx.portfolio.get_snapshot())
        positions = [_serialize(p) for p in ctx.portfolio.get_positions()]
        context_string = ctx.portfolio.as_context_string()
        result = {
            "success": True,
            "snapshot": snapshot,
            "positions": positions,
            "context_string": context_string,
        }
    except Exception as e:
        logger.error(f"fetch_portfolio failed: {e}")
        result = {"success": False, "error": str(e)}
    return json.dumps(result, default=str)
