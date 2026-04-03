"""Market intelligence tools for LangGraph agents."""

import json

import httpx
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
    # TODO: Wire to a proper search API (SerpAPI, Brave Search, Tavily).
    # For now, return a placeholder indicating the tool exists but needs config.
    return json.dumps({
        "error": "Web search not configured — set SEARCH_API_KEY in .env",
        "query": query,
        "results": [],
    })
