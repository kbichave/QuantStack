"""Alpha Vantage adapter for the DataProvider interface.

Thin wrapper around AlphaVantageClient. All rate limiting and retry
logic remains in the underlying client.
"""

from __future__ import annotations

import pandas as pd

from quantstack.data.fetcher import AlphaVantageClient
from quantstack.data.providers.base import DataProvider


class AVProvider(DataProvider):
    """Alpha Vantage data provider.

    Wraps the existing AlphaVantageClient behind the DataProvider ABC.
    Accepts an existing client instance (dependency injection) to preserve
    shared rate limit state.
    """

    # QuantStack indicator name -> AV fetch_economic_indicator parameters
    MACRO_INDICATOR_MAP: dict[str, dict] = {
        "REAL_GDP": {"function": "REAL_GDP"},
        "REAL_GDP_PER_CAPITA": {"function": "REAL_GDP_PER_CAPITA"},
        "TREASURY_YIELD_10Y": {"function": "TREASURY_YIELD", "maturity": "10year"},
        "TREASURY_YIELD_2Y": {"function": "TREASURY_YIELD", "maturity": "2year"},
        "FEDERAL_FUNDS_RATE": {"function": "FEDERAL_FUNDS_RATE"},
        "FED_FUNDS_RATE": {"function": "FEDERAL_FUNDS_RATE"},
        "CPI": {"function": "CPI"},
        "INFLATION": {"function": "INFLATION"},
        "RETAIL_SALES": {"function": "RETAIL_SALES"},
        "UNEMPLOYMENT": {"function": "UNEMPLOYMENT"},
        "NONFARM_PAYROLL": {"function": "NONFARM_PAYROLL"},
        "DURABLES": {"function": "DURABLES"},
    }

    def __init__(self, client: AlphaVantageClient | None = None) -> None:
        self._client = client or AlphaVantageClient()

    def name(self) -> str:
        return "alpha_vantage"

    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None:
        df = self._client.fetch_daily(symbol)
        return df if df is not None and not df.empty else None

    def fetch_ohlcv_intraday(
        self, symbol: str, interval: str = "5min"
    ) -> pd.DataFrame | None:
        df = self._client.fetch_intraday(symbol, interval)
        return df if df is not None and not df.empty else None

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        params = self.MACRO_INDICATOR_MAP.get(indicator)
        if params is None:
            return None
        function = params["function"]
        kwargs = {k: v for k, v in params.items() if k != "function"}
        df = self._client.fetch_economic_indicator(function, **kwargs)
        return df if df is not None and not df.empty else None

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        overview = self._client.fetch_company_overview(symbol)
        if not overview or "Symbol" not in overview:
            return None
        return overview

    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None:
        df = self._client.fetch_insider_transactions(symbol)
        return df if df is not None and not df.empty else None

    def fetch_institutional_holdings(self, symbol: str) -> pd.DataFrame | None:
        df = self._client.fetch_institutional_holdings(symbol)
        return df if df is not None and not df.empty else None

    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None:
        data = self._client.fetch_earnings_history(symbol)
        if not data:
            return None
        # AV returns a dict with annualEarnings/quarterlyEarnings
        # Return the raw dict wrapped — callers handle parsing
        return data

    def fetch_options_chain(
        self, symbol: str, date: str
    ) -> pd.DataFrame | None:
        df = self._client.fetch_historical_options(symbol, date=date)
        return df if df is not None and not df.empty else None

    def fetch_news_sentiment(self, symbol: str) -> pd.DataFrame | None:
        df = self._client.fetch_news_sentiment(tickers=symbol, limit=50)
        return df if df is not None and not df.empty else None
