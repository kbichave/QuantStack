"""Knowledge base and learning tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for past lessons, strategies, and trade outcomes.

    Use before making trading decisions to learn from past experience.
    Returns JSON with relevant knowledge entries ranked by relevance.

    Args:
        query: Natural language query (e.g., "AAPL earnings trade lessons").
        top_k: Number of results to return.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "search_knowledge_base", query=query, top_k=top_k
    )
    return json.dumps(result, default=str)
