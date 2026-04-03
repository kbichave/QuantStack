"""Strategy lifecycle tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def fetch_strategy_registry(status: str | None = None) -> str:
    """Fetch strategies from the registry, optionally filtered by status.

    Args:
        status: Filter by status ("active", "forward_testing", "retired", or None for all).

    Returns JSON with strategy details: ID, type, symbol, performance, status.
    """
    bridge = get_bridge()
    kwargs = {}
    if status:
        kwargs["status"] = status
    result = await bridge.call_quantcore("list_strategies", **kwargs)
    return json.dumps(result, default=str)
