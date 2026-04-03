"""Portfolio management tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def fetch_portfolio() -> str:
    """Get current portfolio state including positions, P&L, and exposure.

    Returns JSON with holdings, unrealized P&L, cash balance,
    and gross/net exposure.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore("get_portfolio_state")
    return json.dumps(result, default=str)
