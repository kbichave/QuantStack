# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Data provider factory — resolves the active DataProvider at runtime.

Separated from provider.py to break the factory-subclass circular dependency
(provider.py defines the ABC, subclasses import it, factory imports subclasses).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.data.polygon import PolygonProvider
from quantstack.data.provider import DataProvider

if TYPE_CHECKING:
    from quantstack.config.timeframes import Timeframe

_active_provider: DataProvider | None = None


def get_provider() -> DataProvider:
    """
    Return the active data provider.

    Priority:
      1. DATA_PROVIDER env var ("polygon" or "alphavantage")
      2. Auto-detect based on available API keys
         - POLYGON_API_KEY set → PolygonProvider
         - ALPHA_VANTAGE_API_KEY set → AlphaVantageProvider
      3. Raise if neither key is set
    """
    global _active_provider
    if _active_provider is not None:
        return _active_provider

    pref = os.getenv("DATA_PROVIDER", "").lower()

    if pref == "polygon" or (not pref and os.getenv("POLYGON_API_KEY")):
        _active_provider = PolygonProvider()
    elif pref == "alphavantage" or (not pref and os.getenv("ALPHA_VANTAGE_API_KEY")):
        _active_provider = AlphaVantageAdapter()
    else:
        raise RuntimeError(
            "No data provider configured. Set POLYGON_API_KEY (recommended) "
            "or ALPHA_VANTAGE_API_KEY, or DATA_PROVIDER env var."
        )

    return _active_provider


def set_provider(provider: DataProvider) -> None:
    """Override the active provider (useful for testing)."""
    global _active_provider
    _active_provider = provider


def fetch_ohlcv_with_fallback(
    symbol: str,
    timeframe: "Timeframe",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV with Alpaca as fallback when the primary provider returns no data.

    The AV quota guard in fetcher.py returns an empty dict (→ empty DataFrame)
    when the daily cap is reached.  This function catches that and retries via
    the Alpaca adapter, which has no daily cap for free historical bars.

    Call this instead of ``get_provider().fetch_ohlcv()`` anywhere that should
    degrade gracefully under AV quota pressure (signal briefs, ML feature refresh,
    backtests).  Do NOT use for intraday or options data — Alpaca sub-minute and
    options history are separate products.
    """
    primary = get_provider()
    df = primary.fetch_ohlcv(symbol, timeframe, start_date, end_date)

    if not df.empty:
        return df

    # Primary returned nothing — try Alpaca before giving up.
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        return df  # no Alpaca credentials; return empty

    try:
        from quantstack.data.adapters.alpaca import AlpacaAdapter

        fallback = AlpacaAdapter()
        logger.info(
            f"[factory] Primary provider returned empty for {symbol} {timeframe} — "
            "falling back to Alpaca OHLCV"
        )
        return fallback.fetch_ohlcv(symbol, timeframe, start_date, end_date)
    except Exception as exc:
        logger.warning(f"[factory] Alpaca fallback failed for {symbol}: {exc}")
        return df
