"""Market intelligence tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def web_search(query: str) -> str:
    """Search the web for market intelligence, news, and analyst actions.

    Use for real-time information that data APIs don't cover:
    breaking news, analyst upgrades/downgrades, macro events.

    Args:
        query: Search query (e.g., "AAPL earnings guidance 2026").

    Returns JSON with search results including title, snippet, and URL.
    """
    try:
        from quantstack.mcp.tools.web import web_search as mcp_web_search
        result = await mcp_web_search(query=query)
        return json.dumps(result, default=str)
    except ImportError:
        return json.dumps({"error": "Web search not available"})
