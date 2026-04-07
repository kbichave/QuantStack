"""Knowledge base and learning tools for LangGraph agents."""

import hashlib
import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from quantstack.rag.query import search_knowledge_base as rag_search

logger = logging.getLogger(__name__)


@tool
async def search_knowledge_base(
    query: Annotated[str, Field(description="Natural language search query for the knowledge base, e.g. 'AAPL earnings trade lessons', 'momentum strategy drawdown'")],
    top_k: Annotated[int, Field(description="Maximum number of knowledge entries to return, ranked by semantic relevance")] = 5,
) -> str:
    """Retrieves past lessons, trade outcomes, and strategy notes from the knowledge base using semantic search. Use when preparing for a new trade to recall historical mistakes, reviewing what worked or failed on a specific ticker, or gathering institutional memory before strategy design. Returns JSON with matching knowledge entries including category, content snippet, metadata, and relevance distance."""
    try:
        results = rag_search(query=query, n_results=top_k)
    except (ConnectionError, OSError) as e:
        logger.error("Embedding service unavailable for search_knowledge_base: %s", e)
        return json.dumps({
            "query": query,
            "results": [],
            "error": "Embedding service unavailable — knowledge base search requires Ollama to be running",
        })
    except Exception as e:
        logger.error("search_knowledge_base failed: %s", e)
        return json.dumps({"query": query, "results": [], "error": str(e)})

    entries = []
    for r in results:
        meta = r.get("metadata") or {}
        text = r.get("text", "")
        entry_id = meta.get("id") or hashlib.sha256(text.encode()).hexdigest()[:16]
        entries.append({
            "id": entry_id,
            "category": r.get("collection", "unknown"),
            "content": text[:500],
            "metadata": meta,
            "created_at": meta.get("created_at"),
            "distance": r.get("distance"),
        })

    return json.dumps({"query": query, "results": entries, "count": len(entries)}, default=str)


@tool
async def promote_strategy(
    strategy_id: Annotated[str, Field(description="Unique strategy identifier (UUID) to promote from draft to forward_testing")],
) -> str:
    """Promotes a strategy from draft status to forward_testing, enabling paper-trade validation. Use when a strategy has passed backtesting criteria and is ready for live paper evaluation. Returns JSON confirmation with updated status or error details."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def retire_strategy(
    strategy_id: Annotated[str, Field(description="Unique strategy identifier (UUID) to retire and deactivate")],
    reason: Annotated[str, Field(description="Human-readable explanation for retirement, e.g. 'regime shift invalidated edge', 'max drawdown exceeded threshold'")] = "",
) -> str:
    """Retires and deactivates a strategy, removing it from the active trading roster. Use when a strategy has underperformed, its market regime edge has disappeared, or risk limits have been breached. Returns JSON confirmation with retirement status and timestamp."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_strategy_performance(
    strategy_id: Annotated[str, Field(description="Unique strategy identifier (UUID) to retrieve performance metrics for")],
) -> str:
    """Retrieves performance metrics and statistics for a specific strategy including PnL, win rate, Sharpe ratio, and drawdown. Use when evaluating whether to promote, retire, or adjust a strategy based on its track record. Returns JSON with cumulative and per-trade performance data."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def validate_strategy(
    strategy_id: Annotated[str, Field(description="Unique strategy identifier (UUID) to validate rules and configuration for")],
) -> str:
    """Validates a strategy's entry rules, exit rules, parameters, and configuration for correctness and completeness. Use before promoting a strategy to forward_testing to catch missing fields, invalid thresholds, or incompatible rule combinations. Returns JSON with validation pass/fail status and any identified issues."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
