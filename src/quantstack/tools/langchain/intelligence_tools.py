"""Market intelligence tools for LangGraph agents."""

import json
from typing import Annotated

import httpx
from langchain_core.tools import tool
from pydantic import Field


@tool
async def web_search(
    query: Annotated[str, Field(description="Search query string for market intelligence, e.g. 'AAPL earnings guidance 2026', 'Fed rate decision', 'NVDA analyst upgrade'")],
) -> str:
    """Search the web for real-time market intelligence, financial news, and analyst actions. Use when you need breaking news, analyst upgrades or downgrades, macro events, earnings surprises, SEC filings, or sector sentiment that structured data APIs do not cover. Returns JSON with search results including title, snippet, source URL, and publication date. Provides context for fundamental analysis and event-driven trading decisions. Synonyms: news search, headline scan, analyst rating, market sentiment, current events, press release, macro outlook."""
    # TODO: Wire to a proper search API (SerpAPI, Brave Search, Tavily).
    # For now, return a placeholder indicating the tool exists but needs config.
    return json.dumps({
        "error": "Web search not configured — set SEARCH_API_KEY in .env",
        "query": query,
        "results": [],
    })
