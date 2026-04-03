"""Data fetch tools for LangGraph agents."""

import json
from datetime import datetime, timedelta

from langchain_core.tools import tool
from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.data.base import AssetClass
from quantstack.data.registry import DataProviderRegistry
from quantstack.tools._helpers import _dataframe_to_dict, _get_reader, _get_writer, _parse_timeframe


def _get_data_registry() -> DataProviderRegistry:
    """Get a DataProviderRegistry from current settings."""
    return DataProviderRegistry.from_settings(get_settings())


@tool
async def fetch_market_data(
    symbol: str,
    timeframe: str = "daily",
    outputsize: str = "compact",
) -> str:
    """Fetch OHLCV market data for a symbol using the configured provider chain.

    Args:
        symbol: Ticker symbol (e.g., "SPY").
        timeframe: "daily", "weekly", "1h", "4h".
        outputsize: "compact" (~6 months) or "full" (5+ years).
    """
    try:
        registry = _get_data_registry()
        tf = _parse_timeframe(timeframe)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 6 if outputsize == "full" else 180)

        df = registry.fetch_ohlcv(symbol, AssetClass.EQUITY, tf, start_date, end_date)

        if df.empty:
            return json.dumps({"error": f"No data returned for {symbol}", "symbol": symbol})

        try:
            writer = _get_writer()
            writer.save_ohlcv(df, symbol, tf)
            writer.close()
        except Exception as store_exc:
            logger.warning(f"Data fetched but failed to persist locally: {store_exc}")

        result = {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def fetch_fundamentals(symbol: str) -> str:
    """Fetch fundamental data (financial statements, company facts) for a symbol.

    Returns JSON with earnings, revenue, margins, and valuation metrics.
    """
    try:
        registry = _get_data_registry()
        data = registry.fetch_fundamentals(symbol)
        result = {"symbol": symbol, **data} if isinstance(data, dict) else {"symbol": symbol, "data": data}
    except Exception as e:
        logger.error(f"fetch_fundamentals({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def fetch_earnings_data(symbol: str) -> str:
    """Fetch earnings data including estimates, historical moves, and IV premium.

    Use for earnings event analysis. Returns JSON with expected move,
    beat rate, and analyst estimates.
    """
    try:
        registry = _get_data_registry()
        data = registry.fetch_earnings(symbol)
        result = {"symbol": symbol, **data} if isinstance(data, dict) else {"symbol": symbol, "data": data}
    except Exception as e:
        logger.error(f"fetch_earnings_data({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def load_market_data(symbol: str, timeframe: str = "daily") -> str:
    """Load previously stored market data from the local database.

    Args:
        symbol: Ticker symbol.
        timeframe: "daily", "weekly", "1h", "4h".
    """
    try:
        reader = _get_reader()
        tf = _parse_timeframe(timeframe)
        df = reader.load_ohlcv(symbol, tf)
        reader.close()

        if df.empty:
            return json.dumps({"error": f"No stored data for {symbol}", "symbol": symbol})

        result = {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def list_stored_symbols() -> str:
    """List all symbols with stored OHLCV data in the local database."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_company_facts(symbol: str) -> str:
    """Get SEC company facts (financial statement line items) for a symbol.

    Args:
        symbol: Ticker symbol.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_analyst_estimates(symbol: str) -> str:
    """Get consensus analyst estimates for a symbol.

    Args:
        symbol: Ticker symbol.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def screen_stocks(
    min_market_cap: float | None = None,
    max_pe: float | None = None,
    min_revenue_growth: float | None = None,
    sector: str | None = None,
) -> str:
    """Screen stocks based on fundamental criteria.

    Args:
        min_market_cap: Minimum market cap.
        max_pe: Maximum P/E ratio.
        min_revenue_growth: Minimum revenue growth rate.
        sector: Filter by sector.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
