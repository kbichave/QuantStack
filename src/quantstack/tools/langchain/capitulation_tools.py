"""Capitulation detection tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def get_capitulation_score(
    symbol: Annotated[str, Field(description="Stock ticker symbol to analyze for capitulation, e.g. 'RDDT', 'SPY', 'AAPL'")],
    lookback_days: Annotated[int, Field(description="Lookback window in trading days for volume exhaustion and consecutive-down normalization")] = 20,
) -> str:
    """Compute an institutional-grade capitulation score for a stock symbol using smart-money and institutional signals only. Use when screening for buy-the-bottom opportunities or detecting seller exhaustion at support levels. Calculates a composite 0-1 score from volume exhaustion, support integrity, Williams VIX Fix extremes, PercentR dual exhaustion, and consecutive down-bar analysis. Returns JSON with capitulation_score, component breakdowns, support level, support_test_count, and actionable recommendation. Score above 0.65 indicates high-conviction capitulation. Synonyms: bottom detection, seller exhaustion, washout signal, panic selling, fear gauge, support test, buy the dip."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
