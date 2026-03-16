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

from quantcore.mcp._helpers import (
    ServerContext,
    _dataframe_to_dict,
    _get_reader,
    _get_writer,
    _parse_timeframe,
)
from quantcore.mcp.server import mcp


# =============================================================================
# DATA TOOLS
# =============================================================================


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
    from quantcore.data.base import AssetClass

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


@mcp.tool()
async def list_stored_symbols() -> dict[str, Any]:
    """
    List all symbols stored in the local database with their metadata.

    Returns:
        Dictionary with symbols and their available timeframes
    """
    store = _get_reader()

    try:
        # Query metadata table
        result = store.conn.execute(
            """
            SELECT symbol, timeframe,
                   first_timestamp, last_timestamp, row_count
            FROM data_metadata
            ORDER BY symbol, timeframe
        """
        ).fetchall()

        symbols = {}
        for row in result:
            sym, tf, first_ts, last_ts, count = row
            if sym not in symbols:
                symbols[sym] = {"timeframes": {}}
            symbols[sym]["timeframes"][tf] = {
                "first_date": str(first_ts) if first_ts else None,
                "last_date": str(last_ts) if last_ts else None,
                "row_count": count,
            }

        return {
            "symbols": symbols,
            "total_symbols": len(symbols),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()
