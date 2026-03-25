# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP data tools — fetch, load, and list market data.

Extracted from ``quantcore.mcp.server`` to keep tool modules focused.
All helpers come from ``quantcore.mcp._helpers``; the ``mcp`` singleton
is imported from ``quantcore.mcp.server``.
"""

from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from quantstack.data.base import AssetClass
from quantstack.mcp._helpers import (
    ServerContext,
    _dataframe_to_dict,
    _get_reader,
    _get_writer,
    _parse_timeframe,
)
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



# =============================================================================
# DATA TOOLS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def fetch_market_data(
    symbol: str,
    timeframe: str = "daily",
    outputsize: str = "compact",
) -> dict[str, Any]:
    """
    Fetch OHLCV market data using the configured provider chain.

    Uses DATA_PROVIDER_PRIORITY (default: alpaca,polygon,alpha_vantage) with
    automatic fallback. Stores fetched data in DuckDB for future load_market_data calls.

    Args:
        symbol: Stock/ETF symbol (e.g., "SPY", "AAPL", "QQQ")
        timeframe: Data frequency - "daily", "1h", "4h", "weekly"
        outputsize: "compact" (~6 months) or "full" (5+ years)

    Returns:
        Dictionary with OHLCV data and metadata
    """
    ctx: ServerContext = mcp.context
    registry = ctx.data_registry
    tf = _parse_timeframe(timeframe)

    # Convert outputsize to date range — providers like Alpaca require
    # explicit start/end rather than Alpha Vantage's compact/full enum
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 6 if outputsize == "full" else 180)

    try:
        df = registry.fetch_ohlcv(symbol, AssetClass.EQUITY, tf, start_date, end_date)

        if df.empty:
            return {"error": f"No data returned for {symbol}", "symbol": symbol}

        # Persist to local DuckDB so load_market_data works without re-fetching.
        # Uses a short-lived write connection — the only writer in this server.
        try:
            writer = _get_writer()
            writer.save_ohlcv(df, symbol, tf)
            writer.close()
        except Exception as store_exc:
            logger.warning(f"Data fetched but failed to persist locally: {store_exc}")

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@domain(Domain.DATA)
@mcp.tool()
async def load_market_data(
    symbol: str,
    timeframe: str = "daily",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Load OHLCV data from local DuckDB storage.

    Args:
        symbol: Stock symbol
        timeframe: "1h", "4h", "daily", "weekly"
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        Dictionary with OHLCV data
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        df = store.load_ohlcv(symbol, tf, start_dt, end_dt)

        if df.empty:
            return {
                "error": f"No data found for {symbol} at {tf.value}",
                "symbol": symbol,
                "hint": "Try fetch_market_data first to download data",
            }

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.DATA)
@mcp.tool()
async def list_stored_symbols() -> dict[str, Any]:
    """
    List all symbols stored in the local database with their metadata.

    Returns:
        Dictionary with symbols and their available timeframes
    """
    store = _get_reader()

    try:
        meta_df = store.get_metadata()

        symbols: dict[str, Any] = {}
        for _, row in meta_df.iterrows():
            sym = row.get("symbol")
            tf = row.get("timeframe")
            if sym not in symbols:
                symbols[sym] = {"timeframes": {}}
            symbols[sym]["timeframes"][tf] = {
                "first_date": str(row.get("first_timestamp")) if row.get("first_timestamp") is not None else None,
                "last_date": str(row.get("last_timestamp")) if row.get("last_timestamp") is not None else None,
                "row_count": row.get("row_count"),
            }

        return {
            "symbols": symbols,
            "total_symbols": len(symbols),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# FUNDAMENTALS & ENRICHMENT TOOLS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_financial_statements(
    symbol: str,
    statement_type: str = "income_statement",
    period_type: str = "quarterly",
    limit: int = 8,
) -> dict[str, Any]:
    """
    Return financial statements (income, balance sheet, or cash flow).

    Args:
        symbol:         Ticker (e.g. "AAPL").
        statement_type: "income_statement" | "balance_sheet" | "cash_flow"
        period_type:    "quarterly" | "annual"
        limit:          Number of periods to return (default 8 = 2 years quarterly).

    Returns:
        List of period rows with key metrics and full JSON data blob.
    """
    store = _get_reader()
    try:
        df = store.load_financial_statements(
            symbol, statement_type=statement_type, period_type=period_type, limit=limit
        )
        if df.empty:
            return {
                "symbol": symbol,
                "statement_type": statement_type,
                "records": [],
                "note": "No data cached — run acquisition pipeline first.",
            }
        return {
            "symbol": symbol,
            "statement_type": statement_type,
            "period_type": period_type,
            "records": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e)}


@domain(Domain.DATA)
@mcp.tool()
async def get_macro_indicator(
    indicator: str,
    start_date: str | None = None,
) -> dict[str, Any]:
    """
    Return a macro economic time series.

    Available indicators (populated by acquisition pipeline):
      REAL_GDP, FEDERAL_FUNDS_RATE, TREASURY_YIELD, CPI, INFLATION,
      RETAIL_SALES, UNEMPLOYMENT, NONFARM_PAYROLL, DURABLES

    Args:
        indicator:  One of the series names above (uppercase).
        start_date: Optional ISO date string to filter from (e.g. "2020-01-01").

    Returns:
        Series of date/value pairs.
    """
    store = _get_reader()
    try:
        df = store.load_macro_indicator(indicator, start_date=start_date)
        if df.empty:
            available = store.list_macro_indicators()
            return {
                "indicator": indicator,
                "records": [],
                "available_indicators": available,
                "note": "No data — run 'macro' phase in acquisition pipeline.",
            }
        return {
            "indicator": indicator,
            "records": _dataframe_to_dict(df),
            "count": len(df),
        }
    except Exception as e:
        return {"error": str(e)}


@domain(Domain.DATA)
@mcp.tool()
async def get_insider_trades(
    symbol: str,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Return recent insider buy/sell transactions for a symbol.

    Args:
        symbol: Ticker (e.g. "NVDA").
        limit:  Max records to return (default 50).

    Returns:
        List of transactions with date, owner, type, shares, price.
    """
    store = _get_reader()
    try:
        df = store.load_insider_trades(symbol, limit=limit)
        if df.empty:
            return {
                "symbol": symbol,
                "records": [],
                "note": "No data — run 'insider' phase in acquisition pipeline.",
            }
        return {"symbol": symbol, "records": _dataframe_to_dict(df), "count": len(df)}
    except Exception as e:
        return {"error": str(e)}


@domain(Domain.DATA)
@mcp.tool()
async def get_institutional_ownership(
    symbol: str,
) -> dict[str, Any]:
    """
    Return top institutional (13F) holders for a symbol.

    Args:
        symbol: Ticker (e.g. "SPY").

    Returns:
        List of institutions with shares held, market value, portfolio weight.
    """
    store = _get_reader()
    try:
        df = store.load_institutional_ownership(symbol)
        if df.empty:
            return {
                "symbol": symbol,
                "records": [],
                "note": "No data — run 'institutional' phase in acquisition pipeline.",
            }
        return {"symbol": symbol, "records": _dataframe_to_dict(df), "count": len(df)}
    except Exception as e:
        return {"error": str(e)}


@domain(Domain.DATA)
@mcp.tool()
async def get_corporate_actions(
    symbol: str,
    action_type: str | None = None,
) -> dict[str, Any]:
    """
    Return dividend history and/or stock split history for a symbol.

    Args:
        symbol:      Ticker (e.g. "AAPL").
        action_type: "dividend" | "split" | None (returns both).

    Returns:
        List of corporate action records with effective date and amount.
    """
    store = _get_reader()
    try:
        df = store.load_corporate_actions(symbol, action_type=action_type)
        if df.empty:
            return {
                "symbol": symbol,
                "action_type": action_type,
                "records": [],
                "note": "No data — run 'corporate_actions' phase in acquisition pipeline.",
            }
        return {
            "symbol": symbol,
            "action_type": action_type or "all",
            "records": _dataframe_to_dict(df),
            "count": len(df),
        }
    except Exception as e:
        return {"error": str(e)}
