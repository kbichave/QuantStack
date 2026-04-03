"""Capitulation detection tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def get_capitulation_score(
    symbol: str,
    lookback_days: int = 20,
) -> str:
    """Compute institutional-grade capitulation score for a symbol.

    Uses ONLY tier_2_smart_money and tier_3_institutional signals -- no retail
    indicators (RSI/MACD/BB). Designed for buy-the-bottom strategies.

    Components:
    - Volume exhaustion (0.25): down-day volume declining = sellers running out
    - Support integrity (0.25): 52-week low zone tested 3+ times without breaking
    - Williams VIX Fix extreme (0.20): synthetic fear gauge at Bollinger extreme
    - PercentR dual exhaustion (0.20): both short+long lookback simultaneously at bottom
    - Consecutive down bars (0.10): normalized run-length vs historical distribution

    Score > 0.65: high-conviction capitulation
    Score 0.40-0.65: partial washout -- watch, not yet actionable
    Score < 0.40: no capitulation signal

    Args:
        symbol: Stock ticker (e.g., "RDDT", "SPY").
        lookback_days: Window for volume exhaustion and consecutive-down normalization.

    Returns JSON with capitulation_score (0-1), component scores, support level,
    support_test_count, and recommendation.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
