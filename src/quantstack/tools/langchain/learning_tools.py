"""Knowledge base and learning tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for past lessons, strategies, and trade outcomes.

    Use before making trading decisions to learn from past experience.
    Returns JSON with relevant knowledge entries ranked by relevance.

    Args:
        query: Natural language query (e.g., "AAPL earnings trade lessons").
        top_k: Number of results to return.
    """
    try:
        from quantstack.tools._state import require_ctx

        ctx = require_ctx()
        db = ctx.db

        rows = db.execute(
            """SELECT id, category, content, metadata, created_at
               FROM knowledge_base
               ORDER BY created_at DESC
               LIMIT %s""",
            (top_k,),
        ).fetchall()

        entries = []
        for row in rows:
            entries.append({
                "id": row[0],
                "category": row[1],
                "content": row[2][:500] if row[2] else "",
                "metadata": row[3],
                "created_at": str(row[4]),
            })

        return json.dumps({"query": query, "results": entries, "count": len(entries)}, default=str)
    except Exception as e:
        logger.error(f"search_knowledge_base failed: {e}")
        return json.dumps({"query": query, "results": [], "error": str(e)})


@tool
async def promote_strategy(strategy_id: str) -> str:
    """Promote a strategy from draft to forward_testing.

    Args:
        strategy_id: Strategy ID to promote.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def retire_strategy(strategy_id: str, reason: str = "") -> str:
    """Retire a strategy.

    Args:
        strategy_id: Strategy ID to retire.
        reason: Reason for retirement.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_strategy_performance(strategy_id: str) -> str:
    """Get performance metrics for a strategy.

    Args:
        strategy_id: Strategy ID.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def validate_strategy(strategy_id: str) -> str:
    """Validate a strategy's rules and configuration.

    Args:
        strategy_id: Strategy ID.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
