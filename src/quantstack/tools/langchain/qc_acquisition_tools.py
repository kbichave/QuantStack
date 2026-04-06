"""Data acquisition tools for LangGraph agents."""

import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import Field


@tool
async def acquire_historical_data(
    phases: Optional[list[str]] = Field(
        default=None,
        description="Which acquisition phases to run (ohlcv_5min, ohlcv_daily, financials, earnings_history, macro, insider, institutional, corporate_actions, options, news, fundamentals). Default: all 11 phases.",
    ),
    symbols: Optional[list[str]] = Field(
        default=None,
        description="Ticker symbols to fetch data for (e.g. ['AAPL', 'MSFT']). Default: full liquid universe (~50 tickers).",
    ),
    m5_lookback_months: int = Field(
        default=24,
        description="Months of 5-minute intraday OHLCV history to retrieve on cold start. Default 24 months.",
    ),
    dry_run: bool = Field(
        default=False,
        description="If True, returns estimated API call counts without making any network requests or DB writes.",
    ),
) -> str:
    """Retrieves and ingests multi-phase historical market data from Alpha Vantage into the local database. Use when you need to populate or refresh OHLCV price bars, financials, earnings history, macro indicators, insider transactions, institutional holdings, corporate actions, options chains, news sentiment, or fundamentals for one or more tickers. Each phase is idempotent and checks existing DB state before calling the API, so re-runs only fetch missing data. Returns JSON with per-phase reports including success counts, failure counts, and skipped symbols. Provides a dry-run mode to estimate API call budget before committing."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def register_ticker(
    symbol: str = Field(
        description="Ticker symbol to register, case-insensitive (e.g. 'HIMS', 'AAPL', 'QQQ').",
    ),
    description: str = Field(
        default="",
        description="Optional one-line description override for the company or ETF.",
    ),
    group: str = Field(
        default="general",
        description="Logical category for the ticker: 'speculative', 'macro_etf', 'large_cap', or 'general'.",
    ),
    acquire_data: bool = Field(
        default=True,
        description="If True, runs all 12 acquisition phases (OHLCV, financials, earnings, etc.) after registering metadata.",
    ),
    dry_run: bool = Field(
        default=False,
        description="If True, fetches and returns metadata only with no DB writes and no data acquisition.",
    ),
) -> str:
    """Registers a new ticker symbol into the trading universe at runtime, fetching metadata from Alpha Vantage and storing it in the database. Use when an agent discovers a new stock, ETF, or asset to track and needs to onboard it without modifying universe.py. Optionally triggers all 12 data acquisition phases (OHLCV daily, OHLCV intraday, financials, earnings history, macro, insider, institutional, corporate actions, options, news, fundamentals) after registration. Returns JSON with ticker metadata, registration status, and acquisition report. Provides a dry-run mode for preview without side effects."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
