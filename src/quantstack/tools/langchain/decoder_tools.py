"""Strategy decoder tools for LangGraph agents."""

import json
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import Field


@tool
async def decode_strategy(
    signals: Annotated[list[dict[str, Any]], Field(description="List of trade signal dicts, each requiring: symbol, direction, entry_time, entry_price, exit_time, exit_price. Optional: size, notes")],
    source_name: Annotated[str, Field(description="Name of the signal source being decoded, e.g. 'discord_trader_x', 'newsletter_alpha'")] = "unknown",
    strategy_name: Annotated[str | None, Field(description="If provided, auto-registers the decoded strategy under this name")] = None,
) -> str:
    """Reverse-engineer a trading strategy from a list of historical trade signals by analyzing entry patterns, exit patterns, sizing behavior, and regime affinity. Use when you have raw trade data from an external source and want to extract the underlying strategy logic. Returns JSON with DecodedStrategy including entry_trigger, exit_trigger, timing_pattern, win_rate, regime_affinity, and edge_hypothesis. Synonyms: strategy extraction, trade pattern analysis, signal decoding, reverse engineer trades, pattern recognition, strategy mining."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def decode_from_trades(
    source: Annotated[str, Field(description="Trade data source table: 'closed_trades' or 'fills'")] = "closed_trades",
    symbol: Annotated[str | None, Field(description="Filter to a specific ticker symbol. None returns all symbols")] = None,
    start_date: Annotated[str | None, Field(description="Start date filter in ISO format (YYYY-MM-DD). None includes all history")] = None,
    end_date: Annotated[str | None, Field(description="End date filter in ISO format (YYYY-MM-DD). None includes up to today")] = None,
    source_name: Annotated[str, Field(description="Label for the decoded source in the output, e.g. 'self', 'paper_account'")] = "self",
) -> str:
    """Decode strategy patterns from the system's own closed trade history or fill records by pulling data from the database and running pattern analysis. Use when performing self-reflection on past trades to discover implicit strategies, recurring entry/exit patterns, or regime-dependent behaviors. Returns JSON with DecodedStrategy extracted from historical trades. Synonyms: trade history analysis, self-decode, pattern extraction, trade review, backtest decode, own trades, fill analysis."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
