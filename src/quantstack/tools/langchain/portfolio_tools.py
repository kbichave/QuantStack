"""Portfolio management tools for LangGraph agents."""

import json

from langchain_core.tools import tool
from loguru import logger

from quantstack.tools._state import live_db_or_error, _serialize


@tool
async def fetch_portfolio() -> str:
    """Retrieve the current portfolio snapshot including all open positions, P&L, cash balance, and exposure metrics. Use when you need to assess holdings, unrealized profit and loss, buying power, gross and net exposure, or sector allocation before making trade decisions. Returns JSON with position details, per-position unrealized P&L, total cash balance, and a human-readable context string. Provides the foundation for risk assessment, rebalancing, and position sizing. Synonyms: holdings, account summary, positions, balance, equity, allocation, portfolio state."""
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
