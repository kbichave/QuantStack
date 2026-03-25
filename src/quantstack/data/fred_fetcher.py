"""
FRED (Federal Reserve Economic Data) fetcher.

Separate from EconomicFetcher (Alpha Vantage) — different API, auth, rate limits.
Stores data into EconomicStorage for unified downstream consumption.

Requires: FRED_API_KEY environment variable and fredapi package.
Free tier: 120 requests/minute.
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from quantstack.data.economic_storage import EconomicStorage

from fredapi import Fred

# FRED series ID mapping
FRED_SERIES: dict[str, str] = {
    # Tier 2 macro signals
    "tips_5y": "DFII5",
    "tips_10y": "DFII10",
    "breakeven_5y": "T5YIE",
    "breakeven_10y": "T10YIE",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A4CBBB",
    "dxy": "DTWEXBGS",
    "gold": "GOLDAMGBD228NLBM",
    "vix": "VIXCLS",
    # Yield curve (backup for FD.ai)
    "fed_funds_eff": "DFF",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "treasury_3m": "DTB3",
}


class FREDFetcher:
    """
    Fetches economic time series from FRED.

    Uses fredapi library, caches results in EconomicStorage.
    Gracefully returns empty DataFrame if FRED_API_KEY is not set.

    Parameters
    ----------
    storage : EconomicStorage, optional
        Storage backend. Creates default if None.
    """

    def __init__(self, storage: EconomicStorage | None = None) -> None:
        self.api_key = os.environ.get("FRED_API_KEY")
        self.storage = storage or EconomicStorage()
        self._fred = None
        self._last_request_time = 0.0
        self._min_interval = 0.5  # 120 req/min = 0.5s between requests

    @property
    def fred(self):
        """Lazy-initialize Fred client."""
        if self._fred is None:
            if not self.api_key:
                return None
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def _rate_limit(self) -> None:
        """Simple rate limiter for FRED API."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def fetch(
        self,
        name: str,
        days: int = 365,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series by friendly name.

        Checks local cache first. Falls back to FRED API if stale or missing.

        Parameters
        ----------
        name : str
            Friendly name (key in FRED_SERIES, e.g. "hy_oas").
        days : int
            Lookback in calendar days. Default 365.
        force_refresh : bool
            If True, bypass cache. Default False.

        Returns
        -------
        pd.DataFrame with columns [date, value], sorted by date.
        Returns empty DataFrame if FRED_API_KEY is not set or series unavailable.
        """
        if name not in FRED_SERIES:
            logger.warning(f"[FRED] Unknown series name: {name}")
            return pd.DataFrame(columns=["date", "value"])

        series_id = FRED_SERIES[name]
        start_date = date.today() - timedelta(days=days)

        # Try cache first
        if not force_refresh:
            cached = self._read_cache(name, start_date)
            if not cached.empty:
                return cached

        # Fetch from FRED
        if self.fred is None:
            logger.debug("[FRED] No API key — returning empty")
            return pd.DataFrame(columns=["date", "value"])

        try:
            self._rate_limit()
            raw = self.fred.get_series(series_id, observation_start=start_date)
            if raw is None or raw.empty:
                return pd.DataFrame(columns=["date", "value"])

            df = pd.DataFrame({"date": raw.index.date, "value": raw.values})
            df = df.dropna(subset=["value"])

            # Cache to storage
            self._write_cache(name, df)
            logger.debug(f"[FRED] Fetched {name} ({series_id}): {len(df)} rows")
            return df

        except Exception as exc:
            logger.warning(f"[FRED] Failed to fetch {name}: {exc}")
            return self._read_cache(name, start_date)

    def _read_cache(self, name: str, start_date: date) -> pd.DataFrame:
        """Read from EconomicStorage."""
        try:
            df = self.storage.get_indicator(name, start_date=start_date)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return pd.DataFrame(columns=["date", "value"])

    def _write_cache(self, name: str, df: pd.DataFrame) -> None:
        """Write to EconomicStorage."""
        try:
            self.storage.store_indicator(name, df, frequency="daily")
        except Exception as exc:
            logger.debug(f"[FRED] Cache write failed for {name}: {exc}")

    def fetch_many(self, names: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
        """Fetch multiple series. Returns dict of name → DataFrame."""
        return {name: self.fetch(name, days=days) for name in names}
