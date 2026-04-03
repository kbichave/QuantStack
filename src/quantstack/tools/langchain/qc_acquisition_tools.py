"""Data acquisition tools for LangGraph agents."""

import json
from typing import Optional

from langchain_core.tools import tool


@tool
async def acquire_historical_data(
    phases: Optional[list[str]] = None,
    symbols: Optional[list[str]] = None,
    m5_lookback_months: int = 24,
    dry_run: bool = False,
) -> str:
    """Run the full-stack data acquisition pipeline via Alpha Vantage.

    Phases (all idempotent, safe to re-run):
      ohlcv_5min, ohlcv_daily, financials, earnings_history, macro,
      insider, institutional, corporate_actions, options, news, fundamentals.

    Each phase checks the DB before calling the API -- only fetches missing data.

    Args:
        phases: Which phases to run. Default: all 11 phases.
        symbols: Which symbols to acquire. Default: full liquid universe (~50).
        m5_lookback_months: Months of 5-min history on cold start (default 24).
        dry_run: If True, return estimated API call counts without making any calls.

    Returns JSON with phase reports, success counts, and failure counts.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def register_ticker(
    symbol: str,
    description: str = "",
    group: str = "general",
    acquire_data: bool = True,
    dry_run: bool = False,
) -> str:
    """Register a ticker: fetch metadata from Alpha Vantage, store in DB,
    and optionally acquire all 12 phases of historical data.

    Use this tool when an agent needs to register a new ticker at runtime
    without a code change to universe.py.

    Args:
        symbol: Ticker symbol (case-insensitive, e.g. "HIMS").
        description: Optional one-line override for what the company does.
        group: Logical category -- "speculative" | "macro_etf" | "large_cap" | "general".
        acquire_data: If True, run all 12 acquisition phases after registering metadata.
        dry_run: If True, fetch and return metadata only -- no DB writes, no acquisition.

    Returns JSON with ticker metadata, registration status, and acquisition report.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
