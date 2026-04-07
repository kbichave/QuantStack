"""DataProvider ABC and configuration exceptions.

All data providers (Alpha Vantage, FRED, EDGAR, etc.) implement this
interface. The ProviderRegistry routes fetch requests to the appropriate
provider based on data type and availability.

Return value semantics for fetch methods:
  - NotImplementedError raised: provider doesn't support this data type.
  - None returned: provider supports this type but found no data.
  - Empty DataFrame returned: provider returned successfully, no rows.
  - Populated DataFrame returned: success.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ConfigurationError(Exception):
    """Raised when a provider cannot initialize due to missing configuration.

    The registry catches this during provider registration and excludes the
    provider with a warning log, rather than crashing the application.
    """


class DataProvider(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider (e.g., 'alpha_vantage', 'fred')."""

    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        """Fetch daily OHLCV bars for symbol."""
        raise NotImplementedError

    def fetch_ohlcv_intraday(
        self, symbol: str, interval: str = "5min"
    ) -> pd.DataFrame | None:
        """Fetch intraday OHLCV bars for symbol at given interval."""
        raise NotImplementedError

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        """Fetch macro indicator time series. Returns (date, value) DataFrame."""
        raise NotImplementedError

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Fetch company fundamentals (overview, financial ratios)."""
        raise NotImplementedError

    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None:
        """Fetch insider transactions."""
        raise NotImplementedError

    def fetch_institutional_holdings(self, symbol: str) -> pd.DataFrame | None:
        """Fetch institutional holdings."""
        raise NotImplementedError

    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None:
        """Fetch earnings history (reported EPS, estimates, surprises)."""
        raise NotImplementedError

    def fetch_options_chain(
        self, symbol: str, date: str
    ) -> pd.DataFrame | None:
        """Fetch options chain for symbol on given date."""
        raise NotImplementedError

    def fetch_sec_filings(
        self, symbol: str, form_types: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Fetch SEC filing metadata. Optionally filter by form types."""
        raise NotImplementedError

    def fetch_news_sentiment(self, symbol: str) -> pd.DataFrame | None:
        """Fetch news sentiment data for symbol."""
        raise NotImplementedError
