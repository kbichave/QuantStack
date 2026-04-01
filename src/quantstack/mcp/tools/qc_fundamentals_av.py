# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Alpha Vantage-sourced fundamental tools — earnings transcripts, ETF profiles,
market movers, market status, insider transactions, institutional holdings.

Only loaded when ALPHA_VANTAGE_API_KEY is configured.
"""

from typing import Any

from quantstack.data.fetcher import AlphaVantageClient
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



# =============================================================================
# EARNINGS CALL TRANSCRIPTS
# =============================================================================


@domain(Domain.DATA)
@tool_def()
async def get_earnings_call_transcript(
    ticker: str,
    year: int,
    quarter: int,
) -> dict[str, Any]:
    """
    Fetch earnings call transcript with LLM sentiment signals.
    Useful for earnings-driven strategy design.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")
        year: Calendar year of the earnings call (e.g., 2024)
        quarter: Fiscal quarter (1, 2, 3, or 4)

    Returns:
        Dictionary with transcript text and metadata
    """
    try:
        client = AlphaVantageClient()
        result = client.fetch_earnings_call_transcript(ticker, year, quarter)
        return {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "data": result,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# =============================================================================
# ETF PROFILE
# =============================================================================


@domain(Domain.DATA)
@tool_def()
async def get_etf_profile(
    ticker: str,
) -> dict[str, Any]:
    """
    Fetch ETF profile with holdings and sector weights.
    Essential for SPY/QQQ/IWM analysis.

    Args:
        ticker: ETF symbol (e.g., "SPY", "QQQ", "IWM")

    Returns:
        Dictionary with ETF holdings, sector weights, and top holdings
    """
    try:
        client = AlphaVantageClient()
        result = client.fetch_etf_profile(ticker)
        return {
            "ticker": ticker,
            "data": result,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# =============================================================================
# MARKET MOVERS
# =============================================================================


@domain(Domain.DATA)
@tool_def()
async def get_top_movers() -> dict[str, Any]:
    """
    Fetch top gainers, losers, and most actively traded US tickers.
    Useful for universe screening and momentum signals.

    Returns:
        Dictionary with top gainers, losers, and most active tickers
    """
    try:
        client = AlphaVantageClient()
        result = client.fetch_top_gainers_losers()
        return {"data": result}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# MARKET STATUS
# =============================================================================


@domain(Domain.DATA)
@tool_def()
async def get_market_status() -> dict[str, Any]:
    """
    Check current open/closed status of global trading venues.
    Use for pre-flight checks before trading sessions.

    Returns:
        Dictionary with market open/close status for global venues
    """
    try:
        client = AlphaVantageClient()
        result = client.fetch_market_status()
        return {"data": result}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# INSIDER TRANSACTIONS (Alpha Vantage source)
# =============================================================================


@domain(Domain.DATA)
@tool_def()
async def get_av_insider_transactions(
    ticker: str,
) -> dict[str, Any]:
    """
    Fetch insider trading activity from Alpha Vantage.
    Complements FinancialDatasets insider data with additional coverage.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")

    Returns:
        Dictionary with insider transaction records
    """
    try:
        client = AlphaVantageClient()
        df = client.fetch_insider_transactions(ticker)
        records = df.to_dict(orient="records") if hasattr(df, "to_dict") else df
        return {
            "ticker": ticker,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# =============================================================================
# INSTITUTIONAL HOLDINGS (Alpha Vantage source)
# =============================================================================


@domain(Domain.DATA)
@tool_def()
async def get_av_institutional_holdings(
    ticker: str,
) -> dict[str, Any]:
    """
    Fetch institutional investor holdings from Alpha Vantage.
    Complements FinancialDatasets institutional data.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")

    Returns:
        Dictionary with institutional holder records
    """
    try:
        client = AlphaVantageClient()
        df = client.fetch_institutional_holdings(ticker)
        records = df.to_dict(orient="records") if hasattr(df, "to_dict") else df
        return {
            "ticker": ticker,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
