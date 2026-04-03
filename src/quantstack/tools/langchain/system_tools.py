"""System health tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def check_system_status() -> str:
    """Check overall system health including kill switch, risk state, and broker mode.

    Called by supervisor graph's safety_check node.
    Returns JSON with service health, kill switch state, and data freshness.
    """
    try:
        from quantstack.tools.functions.system_functions import check_system_status as _fn

        result = await _fn()
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, default=str)


@tool
async def check_heartbeat(service: str) -> str:
    """Check heartbeat freshness for a service (trading-graph, research-graph).

    Args:
        service: Service name ("trading-graph", "research-graph").

    Returns JSON with last_heartbeat timestamp and staleness.
    """
    try:
        from quantstack.tools.functions.system_functions import check_heartbeat as _fn

        result = await _fn(service=service)
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, default=str)


@tool
async def get_recent_decisions(symbol: str | None = None, limit: int = 20) -> str:
    """Query recent audit trail entries for decision review.

    Use during after-market review to inspect today's decisions, when
    debugging why a trade was taken or rejected, or to check for duplicate
    actions before entering a new trade.

    Args:
        symbol: Filter by ticker symbol. None returns all symbols.
        limit: Maximum number of entries to return.

    Returns JSON with decisions list and total count.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
