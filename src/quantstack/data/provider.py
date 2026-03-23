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

# Factory functions moved to quantstack.data.factory to break circular import.
# Lazy re-export via __getattr__ to avoid loading factory.py at import time
# (which would trigger polygon.py → provider.py → factory.py circular chain).
def __getattr__(name: str):
    if name in ("get_provider", "set_provider"):
        from quantstack.data.factory import get_provider, set_provider
        _exports = {"get_provider": get_provider, "set_provider": set_provider}
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
