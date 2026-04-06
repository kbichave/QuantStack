"""Execution feedback loop tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def get_fill_quality(
    order_id: Annotated[str, Field(description="Order ID from get_fills output to assess execution quality for")],
) -> str:
    """Assess execution quality for a completed order fill by comparing the fill price against VWAP at fill time. Use when reviewing trade execution during reflect sessions or auditing slippage across orders. Returns JSON with fill_price, slippage_bps, vwap, fill_vs_vwap_bps, and a quality_note rating. Provides basis for tracking execution cost and broker performance over time. Synonyms: slippage analysis, fill review, execution cost, VWAP comparison, order quality, trade execution audit."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_position_monitor(
    symbol: Annotated[str, Field(description="Ticker symbol of the open position to monitor, e.g. 'AAPL', 'SPY'")],
) -> str:
    """Retrieve comprehensive position status for an open holding including price, unrealized P&L, ATR-based stop distance, days held, and regime comparison. Use when reviewing open positions, checking stop distances, or evaluating whether to hold, trim, or close a position. Returns JSON with price, pnl, days_held, current_regime, flags, and recommended_action. Returns has_position=False if no open position exists for the symbol. Synonyms: position check, holding status, open position review, portfolio monitor, position health, unrealized PnL."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
