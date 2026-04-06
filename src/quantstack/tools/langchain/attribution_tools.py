"""P&L attribution tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def get_daily_equity(
    start_date: Annotated[str, Field(description="Start date in ISO format (YYYY-MM-DD). Defaults to 30 days ago if empty")] = "",
    end_date: Annotated[str, Field(description="End date in ISO format (YYYY-MM-DD). Defaults to today if empty")] = "",
    include_summary: Annotated[bool, Field(description="Whether to include headline stats like Sharpe ratio, Sortino ratio, and max drawdown")] = True,
) -> str:
    """Retrieve the daily equity curve, NAV, returns, and drawdown from the portfolio performance table. Use when analyzing portfolio performance, generating equity charts, or computing risk-adjusted return metrics over a date range. Returns JSON with equity_curve list containing daily NAV and return values, plus optional summary dict with total return, Sharpe ratio, Sortino ratio, max drawdown, and benchmark comparison. Synonyms: portfolio performance, NAV history, equity curve, return series, drawdown analysis, performance attribution."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_strategy_pnl(
    strategy_id: Annotated[str, Field(description="Filter to a single strategy by its ID. Empty string returns all strategies")] = "",
    start_date: Annotated[str, Field(description="Start date in ISO format (YYYY-MM-DD). Defaults to 30 days ago if empty")] = "",
    end_date: Annotated[str, Field(description="End date in ISO format (YYYY-MM-DD). Defaults to today if empty")] = "",
    include_credits: Annotated[bool, Field(description="Whether to include step-level credit breakdown showing which decision steps contribute to P&L")] = False,
) -> str:
    """Retrieve per-strategy profit-and-loss attribution with realized/unrealized P&L, trade counts, and win/loss ratios. Use when evaluating strategy effectiveness, identifying losing strategies, or performing post-trade analysis and blame attribution. Returns JSON with strategy_pnl list and optional credit_breakdown showing contribution from signal, regime, sizing, and debate decision steps. Synonyms: strategy performance, PnL breakdown, trade attribution, win rate, strategy report, loss analysis, decision credit."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
