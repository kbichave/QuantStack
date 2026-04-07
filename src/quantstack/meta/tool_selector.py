"""Quarterly tool-binding optimizer.

Analyzes tool usage telemetry to recommend adding high-demand deferred tools
and removing tools that are never invoked.  This keeps agent tool sets lean
(reducing prompt token cost) while surfacing useful capabilities that agents
are searching for but don't have bound.
"""

from __future__ import annotations


def find_unused_tools(
    agent_id: str,
    tool_usage: dict[str, int],
    bound_tools: list[str],
) -> list[str]:
    """Return tools bound to *agent_id* that have zero invocations."""
    return [t for t in bound_tools if tool_usage.get(t, 0) == 0]


def find_high_demand_deferred(
    agent_id: str,
    search_log: dict[str, int],
    bound_tools: list[str],
) -> list[str]:
    """Return tools that were searched for but are not currently bound."""
    bound_set = set(bound_tools)
    return [
        tool
        for tool, count in sorted(search_log.items(), key=lambda x: -x[1])
        if tool not in bound_set and count > 0
    ]


def recommend_tool_changes(
    agent_id: str,
    tool_usage: dict[str, int],
    search_log: dict[str, int],
    bound_tools: list[str],
) -> dict[str, list[str]]:
    """Return ``{"add": [...], "remove": [...]}`` recommendations for *agent_id*."""
    return {
        "add": find_high_demand_deferred(agent_id, search_log, bound_tools),
        "remove": find_unused_tools(agent_id, tool_usage, bound_tools),
    }
