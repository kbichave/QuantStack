"""Data fetch tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def fetch_market_data(
    symbol: str,
    timeframe: str = "daily",
    outputsize: str = "compact",
) -> str:
    """Fetch OHLCV market data for a symbol from Alpha Vantage API.

    Args:
        symbol: Ticker symbol (e.g., "SPY").
        timeframe: "daily", "weekly", or "monthly".
        outputsize: "compact" (100 bars) or "full" (all history).
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "fetch_market_data",
        symbol=symbol,
        timeframe=timeframe,
        outputsize=outputsize,
    )
    return json.dumps(result, default=str)


@tool
async def fetch_fundamentals(symbol: str) -> str:
    """Fetch fundamental data (financial statements, company facts) for a symbol.

    Returns JSON with earnings, revenue, margins, and valuation metrics.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "get_financial_statements", symbol=symbol
    )
    return json.dumps(result, default=str)


@tool
async def fetch_earnings_data(symbol: str) -> str:
    """Fetch earnings data including estimates, historical moves, and IV premium.

    Use for earnings event analysis. Returns JSON with expected move,
    beat rate, and analyst estimates.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "get_earnings_data", symbol=symbol
    )
    return json.dumps(result, default=str)
