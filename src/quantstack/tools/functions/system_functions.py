"""System health and coordination functions for graph nodes."""

from typing import Any

from quantstack.tools.mcp_bridge._bridge import get_bridge


async def check_system_status() -> dict[str, Any]:
    """Check overall system health status.

    Called by supervisor graph's safety_check node.
    Returns dict with service health, kill switch state, and data freshness.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore("get_system_status")


async def check_heartbeat(service: str) -> dict[str, Any]:
    """Check heartbeat freshness for a specific service.

    Args:
        service: Service name ("trading-graph", "research-graph").

    Returns dict with last_heartbeat timestamp and staleness.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore("check_heartbeat", service=service)


async def record_heartbeat(service: str) -> dict[str, Any]:
    """Record a heartbeat for the current service.

    Called at the start of each graph cycle.
    """
    bridge = get_bridge()
    return await bridge.call_quantcore("record_heartbeat", service=service)
