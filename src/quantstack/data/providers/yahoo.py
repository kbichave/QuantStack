# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Yahoo Finance adapter — last-resort OHLCV fallback via yfinance.

No auth required. Self-imposed 1 req/sec rate limit.
Unreliable — can break without notice. Cache aggressively.

Data limitations:
  - Daily OHLCV: full history available
  - 1-min intraday: only 7 days of history
  - 1-hour intraday: only 60 days of history
"""

from __future__ import annotations

import threading
import time

import pandas as pd
from loguru import logger

from quantstack.data.providers.base import ConfigurationError, DataProvider

# Cache TTLs in seconds
_DAILY_TTL = 86_400  # 24 hours
_INTRADAY_TTL = 3_600  # 1 hour

# yfinance interval string -> max history period
_INTRADAY_LIMITS = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",
}


class YahooProvider(DataProvider):
    """Yahoo Finance adapter via yfinance — last-resort fallback only."""

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ConfigurationError(
                "yfinance not installed — run: pip install yfinance"
            )
        self._cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._lock = threading.Lock()
        self._last_request_time = 0.0

    def name(self) -> str:
        return "yahoo"

    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        cache_key = f"{symbol}:1d"
        cached = self._get_cached(cache_key, _DAILY_TTL)
        if cached is not None:
            return cached

        df = self._download(symbol, period="2y", interval="1d")
        if df is not None:
            self._set_cached(cache_key, df)
        return df

    def fetch_ohlcv_intraday(
        self, symbol: str, interval: str = "5min"
    ) -> pd.DataFrame | None:
        # Normalize interval string: "5min" -> "5m"
        yf_interval = interval.replace("min", "m")
        if yf_interval not in _INTRADAY_LIMITS:
            logger.warning("[Yahoo] Unsupported interval: %s", interval)
            return None

        cache_key = f"{symbol}:{yf_interval}"
        cached = self._get_cached(cache_key, _INTRADAY_TTL)
        if cached is not None:
            return cached

        period = _INTRADAY_LIMITS[yf_interval]
        df = self._download(symbol, period=period, interval=yf_interval)
        if df is not None:
            self._set_cached(cache_key, df)
        return df

    def _download(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame | None:
        import yfinance as yf

        self._rate_limit()
        try:
            df = yf.download(
                symbol, period=period, interval=interval, progress=False
            )
            if df is None or df.empty:
                return None
            return self._normalize(df, symbol)
        except Exception as exc:
            logger.warning("[Yahoo] Download failed %s: %s", symbol, exc)
            return None

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize yfinance output to standard schema."""
        df = df.copy()
        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # Drop adj_close — QuantStack handles adjustments separately
        df = df.drop(columns=["adj_close"], errors="ignore")
        # Rename index
        df.index.name = "timestamp"
        df = df.reset_index()
        df["symbol"] = symbol
        return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    def _rate_limit(self) -> None:
        """Self-imposed 1 request/second to avoid being blocked."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.monotonic()

    def _get_cached(self, key: str, ttl_seconds: int) -> pd.DataFrame | None:
        with self._lock:
            if key in self._cache:
                df, cached_at = self._cache[key]
                if time.monotonic() - cached_at < ttl_seconds:
                    return df
                del self._cache[key]
        return None

    def _set_cached(self, key: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._cache[key] = (df, time.monotonic())
