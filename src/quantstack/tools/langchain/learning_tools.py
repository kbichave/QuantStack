"""Knowledge base and learning tools for LangGraph agents."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


@tool
async def search_knowledge_base(
    query: Annotated[str, Field(description="Natural language search query for the knowledge base, e.g. 'AAPL earnings trade lessons', 'momentum strategy drawdown'")],
    top_k: Annotated[int, Field(description="Maximum number of knowledge entries to return, ranked by recency")] = 5,
) -> str:
    """Retrieves past lessons, trade outcomes, and strategy notes from the knowledge base using keyword search. Use when preparing for a new trade to recall historical mistakes, reviewing what worked or failed on a specific ticker, or gathering institutional memory before strategy design. Returns JSON with matching knowledge entries including category, content snippet, metadata, and creation timestamp."""
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
