"""Strategy decoder tools for LangGraph agents."""

import json
from typing import Any

from langchain_core.tools import tool


@tool
async def decode_strategy(
    signals: list[dict[str, Any]],
    source_name: str = "unknown",
    strategy_name: str | None = None,
) -> str:
    """Reverse-engineer a trading strategy from historical trade signals.

    Analyzes entry patterns (timing, direction bias), exit patterns (holding
    period, target vs time-based), sizing patterns, and regime affinity.

    Args:
        signals: List of trade signal dicts. Each must have:
            symbol, direction, entry_time, entry_price, exit_time, exit_price.
            Optional: size, notes.
        source_name: Name of the signal source (e.g., "discord_trader_x").
        strategy_name: If provided, auto-registers the decoded strategy.

    Returns JSON with DecodedStrategy including entry_trigger, exit_trigger,
    timing_pattern, win_rate, regime_affinity, and edge_hypothesis.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def decode_from_trades(
    source: str = "closed_trades",
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source_name: str = "self",
) -> str:
    """Decode strategy patterns from the system's own trade history.

    Pulls trades from the closed_trades or fills table and feeds them
    to the decoder.

    Args:
        source: "closed_trades" or "fills".
        symbol: Filter by symbol. None = all.
        start_date: Start date filter (YYYY-MM-DD).
        end_date: End date filter (YYYY-MM-DD).
        source_name: Label for the decoded source.

    Returns JSON with DecodedStrategy from historical trades.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
