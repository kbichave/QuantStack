"""System health tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def check_system_status() -> str:
    """Checks overall system health including kill switch state, risk gate status, broker connection mode, and data freshness. Use when verifying the platform is operational before trading, diagnosing outages, or running periodic supervisor health checks. Returns JSON with service health booleans, kill switch active/inactive, paper vs live mode, and last-updated timestamps."""
    try:
        from quantstack.tools.functions.system_functions import check_system_status as _fn

        result = await _fn()
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, default=str)


@tool
async def check_heartbeat(
    service: Annotated[str, Field(description="Service name to check heartbeat for: 'trading-graph', 'research-graph', or 'supervisor-graph'")],
) -> str:
    """Checks heartbeat freshness and liveness for a specific graph service. Use when monitoring whether a service is responsive, detecting stale or crashed graph loops, or triaging alerts about service downtime. Returns JSON with last_heartbeat timestamp, staleness duration in seconds, and alive/dead status."""
    try:
        from quantstack.tools.functions.system_functions import check_heartbeat as _fn

        result = await _fn(service=service)
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, default=str)


@tool
async def get_recent_decisions(
    symbol: Annotated[str | None, Field(description="Ticker symbol to filter decisions by, e.g. 'AAPL', 'SPY'. Pass None to retrieve decisions across all symbols")] = None,
    limit: Annotated[int, Field(description="Maximum number of audit trail entries to return, ordered by most recent first")] = 20,
) -> str:
    """Retrieves recent entries from the decision audit trail for trade review and compliance inspection. Use during after-market review to inspect today's trade decisions, when debugging why a specific entry or exit was taken or rejected, or to check for duplicate actions before placing a new order. Returns JSON with a list of decision records including timestamps, reasoning, action taken, and total count."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
