# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Abstract DataProvider interface + registry.

All data sources implement DataProvider. Consumers call get_provider()
to get the active implementation without knowing which one it is.

Usage:
    provider = get_provider()

    bars = provider.get_bars("SPY", interval="1d", limit=252)
    quote = provider.get_quote("SPY")
    meta = provider.get_symbol_info("SPY")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel

# =============================================================================
# DATA MODELS
# =============================================================================


class Bar(BaseModel):
    """A single OHLCV bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    interval: str = "1d"

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def body_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return abs(self.close - self.open) / self.open


class Quote(BaseModel):
    """Real-time or delayed quote."""

    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    timestamp: datetime
    delayed: bool = True  # False = real-time


class SymbolInfo(BaseModel):
    """Static metadata about a symbol."""

    symbol: str
    name: str
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    avg_daily_volume: int | None = None
    currency: str = "USD"


# =============================================================================
# ABSTRACT PROVIDER
# =============================================================================


class DataProvider(ABC):
    """
    Abstract base class for all market data providers.

    Implementations:
      - PolygonProvider   (production — unlimited calls, $29/month)
      - AlphaVantageProvider (legacy / backtest — 5 calls/min free tier)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 252,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Bar]:
        """
        Fetch OHLCV bars.

        Args:
            symbol: Ticker symbol
            interval: "1m", "5m", "15m", "1h", "1d", "1w"
            limit: Number of bars (ignored if start/end provided)
            start: Start date (inclusive)
            end: End date (inclusive, defaults to today)

        Returns:
            List of Bar objects, oldest first.
        """
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Fetch latest quote for a symbol."""
        ...

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetch static metadata for a symbol."""
        ...

    def get_bars_df(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 252,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: returns bars as a DataFrame.

        DataFrame has columns: open, high, low, close, volume, vwap
        Index is timestamp (DatetimeIndex).
        """
        bars = self.get_bars(symbol, interval, limit, start, end)
        if not bars:
            return pd.DataFrame()
        records = [b.model_dump() for b in bars]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    def get_multi_bars_df(
        self,
        symbols: list[str],
        interval: str = "1d",
        limit: int = 252,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols. Returns {symbol: DataFrame}."""
        return {sym: self.get_bars_df(sym, interval, limit) for sym in symbols}


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

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
        from quantcore.data.polygon import PolygonProvider

        _active_provider = PolygonProvider()
    elif pref == "alphavantage" or (not pref and os.getenv("ALPHA_VANTAGE_API_KEY")):
        from quantcore.data.alphavantage import AlphaVantageProvider

        _active_provider = AlphaVantageProvider()
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
