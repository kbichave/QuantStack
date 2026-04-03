"""P&L attribution tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def get_daily_equity(
    start_date: str = "",
    end_date: str = "",
    include_summary: bool = True,
) -> str:
    """Query the daily equity curve and headline performance stats.

    Returns daily NAV, return, drawdown from the daily_equity table.
    Optionally includes summary stats (total return, Sharpe, Sortino,
    max drawdown, benchmark comparison).

    Args:
        start_date: ISO date string (YYYY-MM-DD). Default: last 30 days.
        end_date: ISO date string (YYYY-MM-DD). Default: today.
        include_summary: Include headline stats (Sharpe, Sortino, etc).

    Returns JSON with equity_curve list and optional summary dict.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_strategy_pnl(
    strategy_id: str = "",
    start_date: str = "",
    end_date: str = "",
    include_credits: bool = False,
) -> str:
    """Query per-strategy P&L attribution with optional step credit breakdown.

    Shows realized/unrealized P&L, trade counts, and win/loss per strategy.
    When include_credits=True, also aggregates step_credits to show which
    decision steps (signal, regime, strategy_selection, sizing, debate)
    contribute most to losses for the given strategy.

    Args:
        strategy_id: Filter to a single strategy. Empty = all strategies.
        start_date: ISO date string. Default: last 30 days.
        end_date: ISO date string. Default: today.
        include_credits: Include step-level credit breakdown.

    Returns JSON with strategy_pnl list and optional credit_breakdown.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
