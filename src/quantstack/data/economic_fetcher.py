"""
Fetcher for economic indicators from Alpha Vantage API.

Indicators include:
- REAL_GDP, REAL_GDP_PER_CAPITA
- TREASURY_YIELD (various maturities)
- FEDERAL_FUNDS_RATE
- CPI, INFLATION
- RETAIL_SALES, DURABLES
- UNEMPLOYMENT, NONFARM_PAYROLL
"""

import time

import pandas as pd
import requests
from loguru import logger
from pydantic import BaseModel

from quantstack.config.settings import get_settings


class EconomicIndicator(BaseModel):
    """Configuration for an economic indicator."""

    function: str
    interval: str | None = None  # For indicators with interval param
    maturity: str | None = None  # For TREASURY_YIELD
    name: str  # Friendly name for database
    frequency: str  # daily, weekly, monthly, quarterly, annual


class EconomicFetcher:
    """Fetches economic indicators from Alpha Vantage."""

    INDICATORS = [
        EconomicIndicator(
            function="REAL_GDP",
            interval="quarterly",
            name="real_gdp_quarterly",
            frequency="quarterly",
        ),
        EconomicIndicator(
            function="REAL_GDP_PER_CAPITA",
            name="real_gdp_per_capita",
            frequency="quarterly",
        ),
        EconomicIndicator(
            function="TREASURY_YIELD",
            interval="daily",
            maturity="3month",
            name="treasury_3m",
            frequency="daily",
        ),
        EconomicIndicator(
            function="TREASURY_YIELD",
            interval="daily",
            maturity="2year",
            name="treasury_2y",
            frequency="daily",
        ),
        EconomicIndicator(
            function="TREASURY_YIELD",
            interval="daily",
            maturity="10year",
            name="treasury_10y",
            frequency="daily",
        ),
        EconomicIndicator(
            function="TREASURY_YIELD",
            interval="daily",
            maturity="30year",
            name="treasury_30y",
            frequency="daily",
        ),
        EconomicIndicator(
            function="FEDERAL_FUNDS_RATE",
            interval="monthly",
            name="fed_funds_rate",
            frequency="monthly",
        ),
        EconomicIndicator(
            function="CPI",
            interval="monthly",
            name="cpi",
            frequency="monthly",
        ),
        EconomicIndicator(
            function="INFLATION",
            name="inflation",
            frequency="annual",
        ),
        EconomicIndicator(
            function="RETAIL_SALES",
            name="retail_sales",
            frequency="monthly",
        ),
        EconomicIndicator(
            function="DURABLES",
            name="durables",
            frequency="monthly",
        ),
        EconomicIndicator(
            function="UNEMPLOYMENT",
            name="unemployment",
            frequency="monthly",
        ),
        EconomicIndicator(
            function="NONFARM_PAYROLL",
            name="nonfarm_payroll",
            frequency="monthly",
        ),
    ]

    def __init__(self, api_key: str | None = None):
        """Initialize economic fetcher.

        Args:
            api_key: Alpha Vantage API key. If None, uses settings.
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.alpha_vantage_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12.0  # Alpha Vantage free tier: 5 calls/min

    def fetch_indicator(self, indicator: EconomicIndicator) -> pd.DataFrame | None:
        """Fetch a single economic indicator.

        Args:
            indicator: Indicator configuration

        Returns:
            DataFrame with date and value columns, or None if fetch fails
        """
        params = {
            "function": indicator.function,
            "apikey": self.api_key,
            "datatype": "json",
        }

        if indicator.interval:
            params["interval"] = indicator.interval
        if indicator.maturity:
            params["maturity"] = indicator.maturity

        try:
            logger.info(
                "Fetching economic indicator: {} ({})",
                indicator.name,
                indicator.function,
            )
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                logger.error(
                    "API error for {}: {}", indicator.name, data["Error Message"]
                )
                return None

            if "Note" in data:
                logger.warning(
                    "API rate limit warning for {}: {}", indicator.name, data["Note"]
                )
                return None

            # Parse response based on structure
            df = self._parse_response(data, indicator)

            if df is not None and not df.empty:
                logger.info(
                    "Fetched {} records for {} ({}–{})",
                    len(df),
                    indicator.name,
                    df["date"].min(),
                    df["date"].max(),
                )
                return df
            else:
                logger.warning("No data returned for {}", indicator.name)
                return None

        except requests.RequestException as e:
            logger.error("Request failed for {}: {}", indicator.name, e)
            return None
        except Exception as e:
            logger.error("Unexpected error for {}: {}", indicator.name, e)
            return None

    def _parse_response(
        self, data: dict, indicator: EconomicIndicator
    ) -> pd.DataFrame | None:
        """Parse API response into DataFrame.

        Args:
            data: JSON response from API
            indicator: Indicator configuration

        Returns:
            DataFrame with date and value columns
        """
        # Most economic indicators have "data" key with list of records
        if "data" in data:
            records = data["data"]
            df = pd.DataFrame(records)

            if df.empty:
                return None

            # Standardize column names
            df = df.rename(columns={"date": "date", "value": "value"})

            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])

            # Convert value to float
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Drop rows with missing values
            df = df.dropna()

            # Add metadata
            df["indicator"] = indicator.name
            df["frequency"] = indicator.frequency

            return df[["date", "value", "indicator", "frequency"]].sort_values("date")

        # Some indicators might have different structure
        # Try to handle time series format
        elif any(k.startswith("Time Series") for k in data.keys()):
            # Find the time series key
            ts_key = [k for k in data.keys() if k.startswith("Time Series")][0]
            ts_data = data[ts_key]

            records = []
            for date_str, values in ts_data.items():
                # Get the value (could be different keys)
                value = None
                for val_key in ["value", "close", "price"]:
                    if val_key in values:
                        value = values[val_key]
                        break

                if value is not None:
                    records.append({"date": date_str, "value": value})

            if not records:
                return None

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            df["indicator"] = indicator.name
            df["frequency"] = indicator.frequency

            return df[["date", "value", "indicator", "frequency"]].sort_values("date")

        logger.warning("Unknown response format for {}", indicator.name)
        return None

    def fetch_all_indicators(
        self, indicators: list[EconomicIndicator] | None = None
    ) -> dict[str, pd.DataFrame]:
        """Fetch all economic indicators.

        Args:
            indicators: List of indicators to fetch. If None, fetches all.

        Returns:
            Dict mapping indicator name to DataFrame
        """
        indicators = indicators or self.INDICATORS
        results = {}

        for i, indicator in enumerate(indicators):
            df = self.fetch_indicator(indicator)

            if df is not None:
                results[indicator.name] = df

            # Rate limiting
            if i < len(indicators) - 1:
                logger.debug("Waiting {:.1f}s for rate limit", self.rate_limit_delay)
                time.sleep(self.rate_limit_delay)

        logger.info("Fetched {} of {} indicators", len(results), len(indicators))
        return results

    def update_indicators(
        self,
        existing_data: dict[str, pd.DataFrame],
        indicators: list[EconomicIndicator] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Update existing indicator data with latest values.

        Args:
            existing_data: Dict of existing indicator DataFrames
            indicators: List of indicators to update. If None, updates all.

        Returns:
            Dict mapping indicator name to updated DataFrame
        """
        indicators = indicators or self.INDICATORS
        results = {}

        for i, indicator in enumerate(indicators):
            logger.info("Updating indicator: {}", indicator.name)

            # Fetch latest data
            new_df = self.fetch_indicator(indicator)

            if new_df is None:
                # Keep existing data if fetch fails
                if indicator.name in existing_data:
                    results[indicator.name] = existing_data[indicator.name]
                continue

            # Merge with existing data
            if indicator.name in existing_data:
                old_df = existing_data[indicator.name]
                # Combine and deduplicate
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                results[indicator.name] = combined.sort_values("date")
            else:
                results[indicator.name] = new_df

            # Rate limiting
            if i < len(indicators) - 1:
                logger.debug("Waiting {:.1f}s for rate limit", self.rate_limit_delay)
                time.sleep(self.rate_limit_delay)

        return results
