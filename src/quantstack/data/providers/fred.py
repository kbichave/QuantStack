"""FRED (Federal Reserve Economic Data) provider.

Fetches macroeconomic indicators via the fredapi library. Only implements
fetch_macro_indicator; all other DataProvider methods inherit the default
NotImplementedError from the ABC.

Rate limit: 120 req/min. Enforced via minimum 0.5s gap between requests.
"""

from __future__ import annotations

import os
import time

import pandas as pd
from fredapi import Fred
from loguru import logger

from quantstack.data.providers.base import ConfigurationError, DataProvider

# QuantStack indicator name -> FRED series ID
INDICATOR_TO_FRED: dict[str, str] = {
    "TREASURY_YIELD_10Y": "DGS10",
    "TREASURY_YIELD_2Y": "DGS2",
    "YIELD_CURVE_SPREAD": "T10Y2Y",
    "FED_FUNDS_RATE": "FEDFUNDS",
    "CPI": "CPIAUCSL",
    "UNEMPLOYMENT": "UNRATE",
    "REAL_GDP": "GDP",
    "HIGH_YIELD_OAS": "BAMLH0A0HYM2",
    "INITIAL_CLAIMS": "ICSA",
}

# Reverse lookup: FRED series ID -> QuantStack name
FRED_TO_INDICATOR: dict[str, str] = {v: k for k, v in INDICATOR_TO_FRED.items()}


class FREDProvider(DataProvider):
    """FRED data provider for macroeconomic indicators.

    Capabilities:
    - fetch_macro_indicator: FRED series -> (date, value) DataFrame

    All other DataProvider methods raise NotImplementedError.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("FRED_API_KEY environment variable is required")
        self._client = Fred(api_key=api_key)
        self._last_request_at: float = 0.0

    def name(self) -> str:
        return "fred"

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        """Fetch a macro indicator from FRED.

        Args:
            indicator: QuantStack indicator name (e.g., "TREASURY_YIELD_10Y")
                       or raw FRED series ID (e.g., "DGS10").

        Returns:
            DataFrame with columns (date, value), or None if no data.

        Raises:
            NotImplementedError: if indicator is not in the series mapping.
        """
        # Resolve to FRED series ID
        if indicator in INDICATOR_TO_FRED:
            series_id = INDICATOR_TO_FRED[indicator]
        elif indicator in FRED_TO_INDICATOR:
            series_id = indicator
        else:
            raise NotImplementedError(
                f"Indicator '{indicator}' is not mapped to a FRED series"
            )

        # Throttle: minimum 0.5s between requests
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)

        self._last_request_at = time.monotonic()
        series = self._client.get_series(series_id)

        if series is None or series.empty:
            return None

        # Normalize to (date, value) DataFrame
        df = pd.DataFrame({"date": series.index, "value": series.values})
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            return None

        logger.debug("[FRED] Fetched %s (%s): %d observations", indicator, series_id, len(df))
        return df
