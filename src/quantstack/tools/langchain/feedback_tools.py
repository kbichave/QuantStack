"""Execution feedback loop tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def get_fill_quality(order_id: str) -> str:
    """Assess execution quality for a completed fill.

    Compares the fill price to VWAP at fill time and returns slippage analysis.
    Use during /reflect sessions to track execution quality over time.

    Args:
        order_id: Order ID from get_fills output.

    Returns JSON with fill_price, slippage_bps, vwap, fill_vs_vwap_bps, quality_note.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_position_monitor(symbol: str) -> str:
    """Comprehensive position status for an open position.

    Returns price, unrealized P&L, ATR-based stop distance, days held,
    and current vs entry regime. Designed for /review position checks.

    Args:
        symbol: Ticker symbol of the open position.

    Returns JSON with price, pnl, days_held, current_regime, flags, recommended_action.
    Returns has_position=False if no open position exists.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
