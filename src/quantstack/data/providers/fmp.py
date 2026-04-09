# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Financial Modeling Prep adapter for fundamentals fallback.

Provides income statement, balance sheet, cash flow, and key ratios
as a secondary source behind Alpha Vantage. Requires FMP_API_KEY env var.

Free tier: 250 req/day. Paid ($14/mo): 300 req/day.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
from loguru import logger

from quantstack.data.providers.base import ConfigurationError, DataProvider

_BASE_URL = "https://financialmodelingprep.com/api/v3"
_TIMEOUT = 15


class FMPProvider(DataProvider):
    """Financial Modeling Prep data provider — fundamentals fallback."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("FMP_API_KEY")
        if not self._api_key:
            raise ConfigurationError("FMP_API_KEY not set")
        self._session = requests.Session()
        self._session.params = {"apikey": self._api_key}  # type: ignore[assignment]

    def name(self) -> str:
        return "fmp"

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Fetch company profile + key ratios."""
        profile = self._get(f"/profile/{symbol}")
        if not profile:
            return None
        ratios = self._get(f"/ratios/{symbol}", params={"limit": 4})
        result = profile[0] if isinstance(profile, list) else profile
        if ratios:
            result["ratios"] = ratios
        return result

    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None:
        data = self._get(f"/earning_calendar", params={"symbol": symbol})
        if not data:
            return None
        df = pd.DataFrame(data)
        if df.empty:
            return None
        return df

    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        """FMP has historical daily data — useful as a secondary OHLCV source."""
        data = self._get(
            f"/historical-price-full/{symbol}",
            params={"timeseries": 504},  # ~2 years
        )
        if not data or "historical" not in data:
            return None
        df = pd.DataFrame(data["historical"])
        if df.empty:
            return None
        df = df.rename(columns={"date": "timestamp"})
        df["symbol"] = symbol
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        return df[[c for c in cols if c in df.columns]].sort_values("timestamp")

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        url = f"{_BASE_URL}{endpoint}"
        try:
            resp = self._session.get(url, params=params or {}, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("[FMP] Request failed %s: %s", endpoint, exc)
            return None
