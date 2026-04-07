"""
Alpha Vantage API client for fetching market data.

Handles rate limiting, retries, and data validation.
"""

import json
import os
import time
from datetime import datetime, timezone
from io import StringIO
from typing import Any

import pandas as pd
import requests
from loguru import logger

from quantstack.config.settings import get_settings

import certifi


# Use system CA bundle if available, otherwise try certifi
def get_ca_bundle():
    """Get CA bundle path for SSL verification."""
    # Check for environment variable first (e.g., Zscaler setup)
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE", "")
    # Validate it's a real path (not a literal ${VAR} string)
    if ca_bundle and not ca_bundle.startswith("${") and os.path.exists(ca_bundle):
        return ca_bundle

    # Check common Zscaler bundle locations
    zscaler_paths = [
        os.path.expanduser("~/.zscaler_certifi_bundle.pem"),
        "/etc/ssl/certs/zscaler.pem",
    ]
    for zscaler_path in zscaler_paths:
        if os.path.exists(zscaler_path):
            return zscaler_path

    return certifi.where()


CA_BUNDLE = get_ca_bundle()


class AlphaVantageClient:
    """
    Client for fetching market data from Alpha Vantage API.

    Features:
    - Rate limiting (respects API limits)
    - Automatic retries with exponential backoff
    - Data validation and cleaning
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Alpha Vantage client.

        Args:
            api_key: API key (uses settings if not provided)
        """
        settings = get_settings()
        self.api_key = api_key or settings.alpha_vantage_api_key
        self.base_url = settings.alpha_vantage_base_url
        self.rate_limit = settings.alpha_vantage_rate_limit
        self._last_call_time: float = 0
        # Fallback in-memory limiter (used when DB is unreachable)
        self._fallback_call_count: int = 0
        self._fallback_minute_start: float = time.time()
        self._using_fallback: bool = False
        # Daily quota guard — prevents runaway loops from exhausting AV credits.
        # Premium plan ($49.99): 75 req/min, no hard daily cap. Default 25,000
        # is ~5.5h of continuous 75/min usage — effectively unlimited for normal ops.
        self._daily_limit: int = int(os.getenv("AV_DAILY_CALL_LIMIT", "25000"))

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits.

        Primary path: PostgreSQL-backed token bucket (shared across containers).
        Fallback path: per-process in-memory limiter (if DB is unreachable).
        """
        self._using_fallback = False
        for attempt in range(60):
            try:
                from quantstack.db import pg_conn

                with pg_conn() as conn:
                    row = conn.execute(
                        "SELECT consume_token('alpha_vantage')"
                    ).fetchone()
                    got_token = row[0] if row else False
                if got_token:
                    return
                logger.debug("Rate limit: waiting for token (bucket: alpha_vantage)")
                time.sleep(1)
            except Exception:
                logger.warning("Rate limiter DB error — falling back to per-process limiter")
                self._using_fallback = True
                self._wait_for_rate_limit_fallback()
                return
        logger.warning("Rate limiter: exhausted 60 retries, skipping call")

    def _wait_for_rate_limit_fallback(self) -> None:
        """Fallback per-process in-memory rate limiter (no DB coordination)."""
        current_time = time.time()
        if current_time - self._fallback_minute_start >= 60:
            self._fallback_call_count = 0
            self._fallback_minute_start = current_time
        if self._fallback_call_count >= self.rate_limit:
            wait_time = 60 - (current_time - self._fallback_minute_start) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached (fallback), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._fallback_call_count = 0
                self._fallback_minute_start = time.time()

    def get_calls_this_minute(self) -> int:
        """Return the fallback call count for the current minute window.

        Exposed for fan-out throttle logic to check quota pressure.
        """
        current_time = time.time()
        if current_time - self._fallback_minute_start >= 60:
            return 0
        return self._fallback_call_count

    def _get_daily_count(self) -> int:
        """Read today's AV call count from system_state. Returns 0 if not set."""
        try:
            from quantstack.db import pg_conn

            today_key = f"av_daily_calls_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            with pg_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = %s", [today_key]
                ).fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            logger.debug(f"[AV quota] Could not read daily count: {exc}")
            return 0

    def _increment_daily_count(self) -> None:
        """Atomically increment today's AV call counter in system_state."""
        try:
            from quantstack.db import pg_conn

            today_key = f"av_daily_calls_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            with pg_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES (%s, '1', NOW())
                    ON CONFLICT (key) DO UPDATE
                      SET value = (CAST(system_state.value AS INTEGER) + 1)::TEXT,
                          updated_at = NOW()
                    """,
                    [today_key],
                )
        except Exception as exc:
            logger.debug(f"[AV quota] Could not increment daily count: {exc}")

    def _make_request(self, function: str, symbol: str, priority: str = "normal", **kwargs) -> dict:
        """
        Make an API request with retry logic.

        Args:
            function: API function name
            symbol: Stock symbol
            **kwargs: Additional parameters

        Returns:
            API response as dictionary
        """
        self._wait_for_rate_limit()

        # Daily quota gate.  Critical requests always proceed; others are shed when quota is near.
        daily_used = self._get_daily_count()
        if priority == "low" and daily_used >= self._daily_limit // 2:
            logger.info(
                f"[AV quota] Skipping low-priority {function}/{symbol} "
                f"(daily used {daily_used}/{self._daily_limit})"
            )
            return {}
        if priority == "normal" and daily_used >= int(self._daily_limit * 0.8):
            logger.warning(
                f"[AV quota] Skipping normal-priority {function}/{symbol} "
                f"(daily used {daily_used}/{self._daily_limit}, >80% budget consumed)"
            )
            return {}
        if priority != "critical" and daily_used >= self._daily_limit:
            logger.error(
                f"[AV quota] Daily quota reached ({daily_used}/{self._daily_limit}). "
                f"Skipping {function}/{symbol}. Only critical requests continue."
            )
            return {}

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            **kwargs,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.base_url, params=params, timeout=30, verify=CA_BUNDLE
                )
                response.raise_for_status()
                self._fallback_call_count += 1
                self._increment_daily_count()

                data = response.json()

                # Check for API errors
                if "Error Message" in data:
                    raise ValueError(f"API Error: {data['Error Message']}")
                if "Note" in data:
                    # Rate limit message from API
                    logger.warning(f"API Note: {data['Note']}")
                    time.sleep(60)
                    continue

                return data

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def fetch_intraday_by_month(
        self,
        symbol: str,
        interval: str = "60min",
        month: str = None,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch intraday data for a specific month or recent data.

        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            month: Month in YYYY-MM format (e.g., "2023-01"). If None, fetches recent data.
            outputsize: 'compact' (100 points) or 'full' (30 days or full month if month specified)

        Returns:
            DataFrame with OHLCV data
        """
        if month:
            logger.info(f"Fetching intraday data for {symbol}, month={month}")
        else:
            logger.info(f"Fetching recent intraday data for {symbol}")

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "adjusted": "true",
            "extended_hours": "true",
            "outputsize": outputsize,
        }

        if month:
            params["month"] = month

        data = self._make_request(
            "TIME_SERIES_INTRADAY",
            symbol,
            interval=interval,
            adjusted="true",
            extended_hours="true",
            outputsize=outputsize,
            **{"month": month} if month else {},
        )

        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.warning(
                f"No time series data found for {symbol} {month or 'recent'}"
            )
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )

        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_intraday(
        self,
        symbol: str,
        interval: str = "60min",
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch intraday data (recent data).

        Args:
            symbol: Stock symbol
            interval: Time interval
            outputsize: 'compact' (100 points) or 'full' (all available)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching intraday data for {symbol}")

        data = self._make_request(
            "TIME_SERIES_INTRADAY",
            symbol,
            interval=interval,
            outputsize=outputsize,
            adjusted="true",
        )

        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.warning(f"No time series data found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )

        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_daily(
        self,
        symbol: str,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data.

        Args:
            symbol: Stock symbol
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching daily data for {symbol}")

        data = self._make_request(
            "TIME_SERIES_DAILY_ADJUSTED",
            symbol,
            outputsize=outputsize,
        )

        if "Time Series (Daily)" not in data:
            logger.warning(f"No daily data found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns (adjusted close)
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. adjusted close": "adj_close",
                "6. volume": "volume",
                "7. dividend amount": "dividend",
                "8. split coefficient": "split_coef",
            }
        )

        # Use adjusted close as close
        if "adj_close" in df.columns:
            # Adjust OHLC for splits/dividends
            adjustment_factor = df["adj_close"].astype(float) / df["close"].astype(
                float
            )
            for col in ["open", "high", "low"]:
                df[col] = df[col].astype(float) * adjustment_factor
            df["close"] = df["adj_close"].astype(float)

        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_weekly(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Fetch weekly OHLCV data.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching weekly data for {symbol}")

        data = self._make_request(
            "TIME_SERIES_WEEKLY_ADJUSTED",
            symbol,
        )

        if "Weekly Adjusted Time Series" not in data:
            logger.warning(f"No weekly data found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data["Weekly Adjusted Time Series"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. adjusted close": "adj_close",
                "6. volume": "volume",
                "7. dividend amount": "dividend",
            }
        )

        # Use adjusted close
        if "adj_close" in df.columns:
            adjustment_factor = df["adj_close"].astype(float) / df["close"].astype(
                float
            )
            for col in ["open", "high", "low"]:
                df[col] = df[col].astype(float) * adjustment_factor
            df["close"] = df["adj_close"].astype(float)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_all_intraday_history(
        self,
        symbol: str,
        interval: str = "60min",
        start_year: int = 2022,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """
        Fetch all available intraday history by month.

        Uses TIME_SERIES_INTRADAY API with month parameter.
        Requires premium API key for outputsize=full with month parameter.

        Args:
            symbol: Stock symbol
            interval: Time interval
            start_year: Starting year
            end_year: Ending year

        Returns:
            DataFrame with complete OHLCV history
        """
        logger.info(
            f"Fetching complete intraday history for {symbol} ({start_year}-{end_year})"
        )
        all_data = []

        # Generate list of months to fetch
        current_date = datetime.now()

        for year in range(start_year, end_year + 1):
            start_month = 1
            end_month = 12

            # Don't fetch future months
            if year == current_date.year:
                end_month = current_date.month

            for month in range(start_month, end_month + 1):
                month_str = f"{year}-{month:02d}"
                try:
                    df = self.fetch_intraday_by_month(
                        symbol, interval, month_str, outputsize="full"
                    )
                    if not df.empty:
                        all_data.append(df)
                        logger.debug(f"Fetched {len(df)} rows from {month_str}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {month_str}: {e}")
                    continue

        if not all_data:
            logger.warning(f"No intraday history found for {symbol}")
            return pd.DataFrame()

        # Combine and deduplicate
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        logger.info(f"Fetched {len(combined)} total rows for {symbol}")
        return combined

    # ========================================
    # Commodity Endpoints
    # ========================================

    def fetch_commodity(
        self,
        commodity: str,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch commodity price data (WTI, BRENT, NATURAL_GAS).

        Args:
            commodity: Commodity code (WTI, BRENT, NATURAL_GAS)
            interval: daily, weekly, or monthly

        Returns:
            DataFrame with date index and 'close' column
        """
        logger.info(f"Fetching {commodity} commodity data ({interval})")

        self._wait_for_rate_limit()

        params = {
            "function": commodity,
            "interval": interval,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")

            # Parse data key
            data_key = "data"
            if data_key not in data:
                logger.warning(f"No data found for {commodity}")
                return pd.DataFrame()

            df = pd.DataFrame(data[data_key])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()

            # Rename value column to close
            df = df.rename(columns={"value": "close"})
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

            # Remove null values
            df = df.dropna()

            # Add OHLV columns (commodity data only has close)
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 0

            df.index.name = "timestamp"
            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Failed to fetch {commodity}: {e}")
            return pd.DataFrame()

    def fetch_economic_indicator(
        self,
        function: str,
        interval: str = "monthly",
        maturity: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch economic indicator data.

        Args:
            function: Indicator function (CPI, FEDERAL_FUNDS_RATE, TREASURY_YIELD, etc.)
            interval: daily, weekly, monthly, etc.
            maturity: For TREASURY_YIELD: 3month, 2year, 5year, 10year, 30year

        Returns:
            DataFrame with date index and 'value' column
        """
        logger.info(f"Fetching {function} economic indicator ({interval})")

        self._wait_for_rate_limit()

        params = {
            "function": function,
            "interval": interval,
            "apikey": self.api_key,
        }

        if maturity:
            params["maturity"] = maturity

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            data_key = "data"
            if data_key not in data:
                logger.warning(f"No data found for {function}")
                return pd.DataFrame()

            df = pd.DataFrame(data[data_key])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()

            df.index.name = "timestamp"
            return df[["value"]]

        except Exception as e:
            logger.error(f"Failed to fetch {function}: {e}")
            return pd.DataFrame()

    def fetch_technical_indicator(
        self,
        function: str,
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ) -> pd.DataFrame:
        """
        Fetch technical indicator from AlphaVantage.

        Args:
            function: Indicator (RSI, SMA, EMA, MACD, BBANDS, ATR, etc.)
            symbol: Stock/ETF symbol
            interval: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
            time_period: Number of data points
            series_type: close, open, high, low

        Returns:
            DataFrame with indicator values
        """
        logger.info(f"Fetching {function} for {symbol} ({interval})")

        self._wait_for_rate_limit()

        params = {
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            # Find the technical analysis key
            ta_key = None
            for key in data.keys():
                if "Technical Analysis" in key:
                    ta_key = key
                    break

            if not ta_key:
                logger.warning(f"No data found for {function}")
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(data[ta_key], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.index.name = "timestamp"
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {function}: {e}")
            return pd.DataFrame()

    def fetch_news_sentiment(
        self,
        tickers: str = "CRUDE",
        topics: str = "energy_transportation",
        time_from: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch news and sentiment data from AlphaVantage.

        Args:
            tickers: Comma-separated tickers (e.g., "CRUDE,XLE,USO")
            topics: Topics to filter (e.g., "energy_transportation")
            time_from: Start time in YYYYMMDDTHHMM format
            limit: Max articles to return (up to 1000)

        Returns:
            DataFrame with news articles and sentiment scores
        """
        logger.info(f"Fetching news sentiment for {tickers}")

        self._wait_for_rate_limit()

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "topics": topics,
            "limit": limit,
            "apikey": self.api_key,
        }

        if time_from:
            params["time_from"] = time_from

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            if "feed" not in data:
                logger.warning("No news feed found")
                return pd.DataFrame()

            articles = []
            for item in data["feed"]:
                article = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "time_published": item.get("time_published", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "overall_sentiment_score": item.get("overall_sentiment_score", 0),
                    "overall_sentiment_label": item.get("overall_sentiment_label", ""),
                }

                # Extract ticker-specific sentiment
                for ticker_data in item.get("ticker_sentiment", []):
                    if ticker_data.get("ticker") in tickers:
                        article["ticker"] = ticker_data.get("ticker")
                        article["ticker_sentiment_score"] = float(
                            ticker_data.get("ticker_sentiment_score", 0)
                        )
                        article["ticker_sentiment_label"] = ticker_data.get(
                            "ticker_sentiment_label", ""
                        )
                        article["relevance_score"] = float(
                            ticker_data.get("relevance_score", 0)
                        )
                        break

                articles.append(article)

            df = pd.DataFrame(articles)

            if not df.empty and "time_published" in df.columns:
                df["time_published"] = pd.to_datetime(
                    df["time_published"], format="%Y%m%dT%H%M%S"
                )
                df = df.set_index("time_published")
                df = df.sort_index()

            logger.info(f"Fetched {len(df)} news articles")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch news sentiment: {e}")
            return pd.DataFrame()

    def fetch_historical_news_sentiment(
        self,
        tickers: str = "CRUDE,XLE,USO",
        topics: str = "energy_transportation",
        start_date: str = "2020-01-01",
        end_date: str | None = None,
        batch_months: int = 3,
        limit_per_batch: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical news sentiment data in batches.

        Batches requests by time windows to get comprehensive historical data
        while respecting API rate limits.

        Args:
            tickers: Comma-separated tickers (e.g., "CRUDE,XLE,USO")
            topics: Topics to filter (e.g., "energy_transportation")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: current date)
            batch_months: Number of months per batch request
            limit_per_batch: Max articles per batch (up to 1000)

        Returns:
            DataFrame with all historical news articles, deduplicated
        """
        logger.info(
            f"Fetching historical news sentiment from {start_date} to {end_date or 'now'}"
        )

        # Parse dates
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

        all_articles = []
        batch_count = 0

        while current < end:
            # Calculate batch end (N months later or end date)
            batch_end = min(current + pd.DateOffset(months=batch_months), end)

            # Format for API (YYYYMMDDTHHMM)
            time_from = current.strftime("%Y%m%dT0000")
            time_to = batch_end.strftime("%Y%m%dT2359")

            logger.info(
                f"Fetching batch {batch_count + 1}: {current.date()} to {batch_end.date()}"
            )

            try:
                self._wait_for_rate_limit()

                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": tickers,
                    "topics": topics,
                    "time_from": time_from,
                    "time_to": time_to,
                    "limit": limit_per_batch,
                    "sort": "EARLIEST",
                    "apikey": self.api_key,
                }

                response = requests.get(
                    self.base_url, params=params, timeout=60, verify=CA_BUNDLE
                )
                response.raise_for_status()
                self._fallback_call_count += 1
                data = response.json()

                if "Error Message" in data:
                    logger.warning(f"API Error for batch: {data['Error Message']}")
                elif "Note" in data:
                    logger.warning(f"API rate limit hit: {data['Note']}")
                    # Wait longer and retry
                    time.sleep(60)
                    continue
                elif "feed" in data:
                    batch_articles = self._parse_news_feed(data["feed"], tickers)
                    all_articles.extend(batch_articles)
                    logger.info(f"  Got {len(batch_articles)} articles")
                else:
                    logger.debug("  No articles in this batch")

                batch_count += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for batch {current.date()}: {e}")
            except Exception as e:
                logger.error(f"Error processing batch {current.date()}: {e}")

            # Move to next batch
            current = batch_end

            # Small delay between batches
            time.sleep(0.5)

        if not all_articles:
            logger.warning("No historical news articles retrieved")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(all_articles)

        # Parse timestamps
        if "time_published" in df.columns:
            df["time_published"] = pd.to_datetime(
                df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce"
            )

            # Drop rows with invalid timestamps
            df = df.dropna(subset=["time_published"])

            # Set index and sort
            df = df.set_index("time_published")
            df = df.sort_index()

        # Remove duplicates (same title + time)
        if "title" in df.columns:
            df = df.reset_index()
            df = df.drop_duplicates(subset=["time_published", "title"], keep="first")
            df = df.set_index("time_published")

        logger.info(
            f"Fetched total {len(df)} unique news articles from {batch_count} batches"
        )
        return df

    def _parse_news_feed(self, feed: list, tickers: str) -> list:
        """
        Parse news feed items into article dictionaries.

        Args:
            feed: List of feed items from API
            tickers: Comma-separated tickers to match

        Returns:
            List of article dictionaries
        """
        articles = []
        ticker_list = tickers.split(",")

        for item in feed:
            article = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "time_published": item.get("time_published", ""),
                "summary": (
                    item.get("summary", "")[:500] if item.get("summary") else ""
                ),  # Truncate
                "source": item.get("source", ""),
                "overall_sentiment_score": float(
                    item.get("overall_sentiment_score", 0) or 0
                ),
                "overall_sentiment_label": item.get("overall_sentiment_label", ""),
            }

            # Extract ticker-specific sentiment (find first matching ticker)
            for ticker_data in item.get("ticker_sentiment", []):
                ticker = ticker_data.get("ticker", "")
                if ticker in ticker_list:
                    article["ticker"] = ticker
                    article["ticker_sentiment_score"] = float(
                        ticker_data.get("ticker_sentiment_score", 0) or 0
                    )
                    article["ticker_sentiment_label"] = ticker_data.get(
                        "ticker_sentiment_label", ""
                    )
                    article["relevance_score"] = float(
                        ticker_data.get("relevance_score", 0) or 0
                    )
                    break

            # Only add if we have a timestamp
            if article["time_published"]:
                articles.append(article)

        return articles

    # ========================================
    # Options Data Endpoints
    # ========================================

    def fetch_realtime_options(
        self,
        symbol: str,
        require_greeks: bool = True,
        contract: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch realtime options chain data from AlphaVantage.

        Args:
            symbol: Underlying stock symbol (e.g., "AAPL")
            require_greeks: Include Greeks and IV in response
            contract: Optional specific contract ID to fetch

        Returns:
            DataFrame with options chain data including:
            - contractID, symbol, expiration, strike, type
            - bid, ask, last, volume, open_interest
            - implied_volatility, delta, gamma, theta, vega, rho (if require_greeks=True)
        """
        logger.info(f"Fetching realtime options for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "REALTIME_OPTIONS",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        if require_greeks:
            params["require_greeks"] = "true"

        if contract:
            params["contract"] = contract

        try:
            response = requests.get(
                self.base_url, params=params, timeout=60, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()

            if "data" not in data:
                logger.warning(f"No options data found for {symbol}")
                return pd.DataFrame()

            options_list = data["data"]
            if not options_list:
                return pd.DataFrame()

            df = pd.DataFrame(options_list)

            # Standardize column names
            column_mapping = {
                "contractID": "contract_id",
                "symbol": "underlying",
                "expiration": "expiry",
                "strike": "strike",
                "type": "option_type",
                "bid": "bid",
                "ask": "ask",
                "last": "last",
                "volume": "volume",
                "open_interest": "open_interest",
                "implied_volatility": "iv",
                "delta": "delta",
                "gamma": "gamma",
                "theta": "theta",
                "vega": "vega",
                "rho": "rho",
            }

            df = df.rename(
                columns={k: v for k, v in column_mapping.items() if k in df.columns}
            )

            # Convert numeric columns
            numeric_cols = [
                "strike",
                "bid",
                "ask",
                "last",
                "volume",
                "open_interest",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert expiry to datetime
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"])

            # Add mid price
            if "bid" in df.columns and "ask" in df.columns:
                df["mid"] = (df["bid"] + df["ask"]) / 2

            logger.info(f"Fetched {len(df)} options contracts for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch realtime options for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_options(
        self,
        symbol: str,
        date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical options chain for a specific date.

        Covers 15+ years of history (since 2008-01-01).
        Includes IV and Greeks.

        Args:
            symbol: Underlying stock symbol
            date: Date in YYYY-MM-DD format. If None, returns previous trading day.

        Returns:
            DataFrame with historical options data
        """
        logger.info(
            f"Fetching historical options for {symbol}"
            + (f" on {date}" if date else "")
        )

        self._wait_for_rate_limit()

        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        if date:
            params["date"] = date

        try:
            response = requests.get(
                self.base_url, params=params, timeout=60, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()

            if "data" not in data:
                logger.warning(f"No historical options data found for {symbol}")
                return pd.DataFrame()

            options_list = data["data"]
            if not options_list:
                return pd.DataFrame()

            df = pd.DataFrame(options_list)

            # Standardize column names (same as realtime)
            column_mapping = {
                "contractID": "contract_id",
                "symbol": "underlying",
                "expiration": "expiry",
                "strike": "strike",
                "type": "option_type",
                "bid": "bid",
                "ask": "ask",
                "last": "last",
                "volume": "volume",
                "open_interest": "open_interest",
                "implied_volatility": "iv",
                "delta": "delta",
                "gamma": "gamma",
                "theta": "theta",
                "vega": "vega",
                "rho": "rho",
            }

            df = df.rename(
                columns={k: v for k, v in column_mapping.items() if k in df.columns}
            )

            # Convert numeric columns
            numeric_cols = [
                "strike",
                "bid",
                "ask",
                "last",
                "volume",
                "open_interest",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert expiry to datetime
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"])

            # Add mid price
            if "bid" in df.columns and "ask" in df.columns:
                df["mid"] = (df["bid"] + df["ask"]) / 2

            # Add the query date as a column
            df["data_date"] = (
                pd.to_datetime(date) if date else pd.Timestamp.now().normalize()
            )

            logger.info(f"Fetched {len(df)} historical options for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical options for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_earnings_calendar(
        self,
        symbol: str | None = None,
        horizon: str = "3month",
    ) -> pd.DataFrame:
        """
        Fetch earnings calendar for upcoming earnings announcements.

        Args:
            symbol: Optional symbol to filter. If None, returns all upcoming earnings.
            horizon: Time horizon - "3month", "6month", or "12month"

        Returns:
            DataFrame with columns:
            - symbol, name, reportDate, fiscalDateEnding
            - estimate, currency
        """
        logger.info("Fetching earnings calendar" + (f" for {symbol}" if symbol else ""))

        self._wait_for_rate_limit()

        params = {
            "function": "EARNINGS_CALENDAR",
            "horizon": horizon,
            "apikey": self.api_key,
        }

        if symbol:
            params["symbol"] = symbol

        try:
            # Earnings calendar returns CSV by default
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1

            # Check if it's an error response (JSON)
            try:
                data = response.json()
                if "Error Message" in data:
                    raise ValueError(f"API Error: {data['Error Message']}")
                if "Note" in data:
                    logger.warning(f"API Note: {data['Note']}")
                    return pd.DataFrame()
            except (ValueError, KeyError):
                pass  # Not JSON, continue with CSV parsing

            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                logger.warning("No earnings data found")
                return pd.DataFrame()

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Convert report date
            if "reportdate" in df.columns:
                df["report_date"] = pd.to_datetime(df["reportdate"], errors="coerce")
                df = df.drop(columns=["reportdate"], errors="ignore")

            if "fiscaldateending" in df.columns:
                df["fiscal_date_ending"] = pd.to_datetime(
                    df["fiscaldateending"], errors="coerce"
                )
                df = df.drop(columns=["fiscaldateending"], errors="ignore")

            # Convert estimate to numeric
            if "estimate" in df.columns:
                df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")

            logger.info(f"Fetched {len(df)} earnings announcements")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch earnings calendar: {e}")
            return pd.DataFrame()

    def fetch_company_overview(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """
        Fetch company fundamentals and overview data.

        Useful for:
        - Dividend dates (ExDividendDate, DividendDate)
        - Sector classification
        - Market cap and fundamental data

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company data including:
            - Symbol, Name, Sector, Industry
            - MarketCapitalization
            - DividendPerShare, DividendYield
            - ExDividendDate, DividendDate
            - 52WeekHigh, 52WeekLow
            - etc.
        """
        logger.info(f"Fetching company overview for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return {}

            if not data or "Symbol" not in data:
                logger.warning(f"No overview data found for {symbol}")
                return {}

            # Convert numeric fields
            numeric_fields = [
                "MarketCapitalization",
                "EBITDA",
                "PERatio",
                "PEGRatio",
                "BookValue",
                "DividendPerShare",
                "DividendYield",
                "EPS",
                "RevenuePerShareTTM",
                "ProfitMargin",
                "OperatingMarginTTM",
                "ReturnOnAssetsTTM",
                "ReturnOnEquityTTM",
                "RevenueTTM",
                "GrossProfitTTM",
                "DilutedEPSTTM",
                "QuarterlyEarningsGrowthYOY",
                "QuarterlyRevenueGrowthYOY",
                "AnalystTargetPrice",
                "TrailingPE",
                "ForwardPE",
                "PriceToSalesRatioTTM",
                "PriceToBookRatio",
                "EVToRevenue",
                "EVToEBITDA",
                "Beta",
                "52WeekHigh",
                "52WeekLow",
                "50DayMovingAverage",
                "200DayMovingAverage",
                "SharesOutstanding",
                "SharesFloat",
                "SharesShort",
                "SharesShortPriorMonth",
                "ShortRatio",
                "ShortPercentOutstanding",
                "ShortPercentFloat",
                "PercentInsiders",
                "PercentInstitutions",
            ]

            for field in numeric_fields:
                if field in data and data[field] not in [None, "None", "-", ""]:
                    try:
                        data[field] = float(data[field])
                    except (ValueError, TypeError):
                        pass

            logger.info(f"Fetched overview for {symbol}: {data.get('Name', 'Unknown')}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch company overview for {symbol}: {e}")
            return {}

    def fetch_bulk_quotes(
        self,
        symbols: list[str],
    ) -> pd.DataFrame:
        """
        Fetch realtime quotes for multiple symbols in bulk.

        Accepts up to 100 symbols per request.

        Args:
            symbols: List of stock symbols (max 100)

        Returns:
            DataFrame with columns:
            - symbol, open, high, low, price, volume
            - latestDay, previousClose, change, changePercent
        """
        if not symbols:
            return pd.DataFrame()

        # Limit to 100 symbols
        if len(symbols) > 100:
            logger.warning(
                f"Bulk quotes limited to 100 symbols, truncating from {len(symbols)}"
            )
            symbols = symbols[:100]

        logger.info(f"Fetching bulk quotes for {len(symbols)} symbols")

        self._wait_for_rate_limit()

        params = {
            "function": "REALTIME_BULK_QUOTES",
            "symbol": ",".join(symbols),
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()

            if "data" not in data:
                logger.warning("No bulk quote data found")
                return pd.DataFrame()

            quotes = data["data"]
            if not quotes:
                return pd.DataFrame()

            df = pd.DataFrame(quotes)

            # Standardize column names
            column_mapping = {
                "01. symbol": "symbol",
                "02. open": "open",
                "03. high": "high",
                "04. low": "low",
                "05. price": "price",
                "06. volume": "volume",
                "07. latest trading day": "latest_day",
                "08. previous close": "previous_close",
                "09. change": "change",
                "10. change percent": "change_percent",
            }

            df = df.rename(
                columns={k: v for k, v in column_mapping.items() if k in df.columns}
            )

            # Convert numeric columns
            numeric_cols = [
                "open",
                "high",
                "low",
                "price",
                "volume",
                "previous_close",
                "change",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parse change percent
            if "change_percent" in df.columns:
                df["change_percent"] = (
                    df["change_percent"].str.rstrip("%").astype(float) / 100
                )

            logger.info(f"Fetched quotes for {len(df)} symbols")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch bulk quotes: {e}")
            return pd.DataFrame()

    def fetch_earnings_call_transcript(
        self,
        symbol: str,
        year: int,
        quarter: int,
    ) -> dict[str, Any]:
        """
        Fetch earnings call transcript for a specific quarter.

        Args:
            symbol: Stock symbol
            year: Fiscal year (e.g. 2024)
            quarter: Fiscal quarter (1-4)

        Returns:
            Dictionary with transcript text and metadata including:
            - symbol, quarter, year
            - transcript content
        """
        logger.info(f"Fetching earnings call transcript for {symbol} Q{quarter} {year}")

        self._wait_for_rate_limit()

        params = {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return {}

            if not data:
                logger.warning(
                    f"No transcript data found for {symbol} Q{quarter} {year}"
                )
                return {}

            logger.info(f"Fetched transcript for {symbol} Q{quarter} {year}")
            return data

        except Exception as e:
            logger.error(
                f"Failed to fetch earnings call transcript for {symbol} Q{quarter} {year}: {e}"
            )
            return {}

    def fetch_etf_profile(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """
        Fetch ETF profile including holdings, sector weights, and metadata.

        Useful for ETFs like SPY, QQQ, IWM to understand composition.

        Args:
            symbol: ETF symbol

        Returns:
            Dictionary with ETF data including:
            - holdings, sector weights, top holdings
            - net assets, expense ratio, turnover
        """
        logger.info(f"Fetching ETF profile for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "ETF_PROFILE",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return {}

            if not data:
                logger.warning(f"No ETF profile data found for {symbol}")
                return {}

            # Convert numeric fields where present
            numeric_fields = [
                "net_assets",
                "net_expense_ratio",
                "portfolio_turnover",
                "dividend_yield",
                "inception_date",
            ]
            for field in numeric_fields:
                if field in data and data[field] not in [None, "None", "-", ""]:
                    try:
                        data[field] = float(data[field])
                    except (ValueError, TypeError):
                        pass

            logger.info(f"Fetched ETF profile for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch ETF profile for {symbol}: {e}")
            return {}

    def fetch_top_gainers_losers(self) -> dict[str, Any]:
        """
        Fetch top gainers, losers, and most actively traded tickers.

        No symbol parameter needed — returns market-wide data.

        Returns:
            Dictionary with keys:
            - top_gainers: list of top gaining tickers
            - top_losers: list of top losing tickers
            - most_actively_traded: list of most active tickers
            Each entry includes ticker, price, change_amount, change_percentage, volume.
        """
        logger.info("Fetching top gainers/losers")

        self._wait_for_rate_limit()

        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return {}

            if not data:
                logger.warning("No top gainers/losers data found")
                return {}

            # Convert numeric fields in each list
            for list_key in ["top_gainers", "top_losers", "most_actively_traded"]:
                if list_key in data and isinstance(data[list_key], list):
                    for entry in data[list_key]:
                        for field in ["price", "change_amount", "volume"]:
                            if field in entry and entry[field] not in [
                                None,
                                "None",
                                "-",
                                "",
                            ]:
                                try:
                                    entry[field] = float(entry[field])
                                except (ValueError, TypeError):
                                    pass
                        if "change_percentage" in entry:
                            raw = entry["change_percentage"]
                            if isinstance(raw, str) and raw.endswith("%"):
                                try:
                                    entry["change_percentage"] = (
                                        float(raw.rstrip("%")) / 100
                                    )
                                except (ValueError, TypeError):
                                    pass

            logger.info(
                f"Fetched {len(data.get('top_gainers', []))} gainers, "
                f"{len(data.get('top_losers', []))} losers, "
                f"{len(data.get('most_actively_traded', []))} active"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to fetch top gainers/losers: {e}")
            return {}

    def fetch_market_status(self) -> dict[str, Any]:
        """
        Fetch current market status for global exchanges.

        No symbol parameter needed — returns status for all tracked markets.

        Returns:
            Dictionary with market status data including:
            - markets: list of exchanges with open/close status
            - Each entry includes market_type, region, primary_exchanges, status
        """
        logger.info("Fetching market status")

        self._wait_for_rate_limit()

        params = {
            "function": "MARKET_STATUS",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return {}

            if not data:
                logger.warning("No market status data found")
                return {}

            logger.info(
                f"Fetched market status for {len(data.get('markets', []))} exchanges"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to fetch market status: {e}")
            return {}

    def fetch_insider_transactions(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Fetch insider transactions (buys/sells by officers, directors, 10% owners).

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with insider transaction data including:
            - transaction_date, shares, acquisition_or_disposition
            - transaction_type, owner_name, owner_title
        """
        logger.info(f"Fetching insider transactions for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "INSIDER_TRANSACTIONS",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()

            # Extract the transaction list from the response
            transactions = data.get("data", data.get("transactions", []))
            if not transactions:
                logger.warning(f"No insider transaction data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(transactions)

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Convert numeric columns
            numeric_cols = ["shares", "share_price"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert date columns
            for col in ["transaction_date", "filing_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            logger.info(f"Fetched {len(df)} insider transactions for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch insider transactions for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_institutional_holdings(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Fetch institutional ownership data (13F filings).

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with institutional holding data including:
            - investor, shares, value, weight
            - date_reported, change_in_shares
        """
        logger.info(f"Fetching institutional holdings for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "INSTITUTIONAL_HOLDINGS",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()

            # Extract the holdings list from the response
            holdings = data.get("data", data.get("holdings", []))
            if not holdings:
                logger.warning(f"No institutional holdings data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(holdings)

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Convert numeric columns
            numeric_cols = ["shares", "value", "weight", "change_in_shares"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert date columns
            for col in ["date_reported"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            logger.info(f"Fetched {len(df)} institutional holders for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch institutional holdings for {symbol}: {e}")
            return pd.DataFrame()

    # ========================================
    # Financial Statements (Free tier)
    # ========================================

    def _fetch_statement(self, function: str, symbol: str) -> dict:
        """Shared helper for INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW."""
        logger.info(f"Fetching {function} for {symbol}")
        self._wait_for_rate_limit()
        params = {"function": function, "symbol": symbol, "apikey": self.api_key}
        response = requests.get(
            self.base_url, params=params, timeout=30, verify=CA_BUNDLE
        )
        response.raise_for_status()
        self._fallback_call_count += 1
        data = response.json()
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Note" in data:
            logger.warning(f"API Note: {data['Note']}")
            return {}
        return data

    def fetch_income_statement(self, symbol: str) -> dict:
        """
        Fetch annual and quarterly income statements.

        Returns dict with keys ``annualReports`` and ``quarterlyReports``,
        each a list of period dicts (totalRevenue, netIncome, grossProfit, etc.).
        Free tier, no rate-limit beyond the standard 75/min.
        """
        return self._fetch_statement("INCOME_STATEMENT", symbol)

    def fetch_balance_sheet(self, symbol: str) -> dict:
        """
        Fetch annual and quarterly balance sheets.

        Returns dict with keys ``annualReports`` and ``quarterlyReports``,
        each a list of period dicts (totalAssets, totalLiabilities, etc.).
        Free tier.
        """
        return self._fetch_statement("BALANCE_SHEET", symbol)

    def fetch_cash_flow(self, symbol: str) -> dict:
        """
        Fetch annual and quarterly cash flow statements.

        Returns dict with keys ``annualReports`` and ``quarterlyReports``,
        each a list of period dicts (operatingCashflow, capitalExpenditures, etc.).
        Free tier.
        """
        return self._fetch_statement("CASH_FLOW", symbol)

    def fetch_earnings_history(self, symbol: str) -> dict:
        """
        Fetch historical EPS — actual vs estimated, annual and quarterly.

        Returns dict with keys ``annualEarnings`` and ``quarterlyEarnings``,
        each a list of period dicts (fiscalDateEnding, reportedEPS, estimatedEPS,
        surprise, surprisePercentage).
        Free tier; more granular than EARNINGS_CALENDAR.
        """
        logger.info(f"Fetching earnings history for {symbol}")
        self._wait_for_rate_limit()
        params = {"function": "EARNINGS", "symbol": symbol, "apikey": self.api_key}
        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return {}
            return data
        except Exception as e:
            logger.error(f"Failed to fetch earnings history for {symbol}: {e}")
            return {}

    # ========================================
    # Corporate Actions (Free tier)
    # ========================================

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        """
        Fetch full dividend history (ex-dividend dates, amounts, payment dates).

        AV function: DIVIDENDS.  Returns a DataFrame with columns:
        ex_dividend_date, declaration_date, record_date, payment_date, amount.
        Free tier.
        """
        logger.info(f"Fetching dividends for {symbol}")
        self._wait_for_rate_limit()
        params = {"function": "DIVIDENDS", "symbol": symbol, "apikey": self.api_key}
        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()
            records = data.get("data", [])
            if not records:
                logger.debug(f"No dividend data for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame(records)
            df.columns = df.columns.str.lower().str.replace(" ", "_")
            for col in [
                "ex_dividend_date",
                "declaration_date",
                "record_date",
                "payment_date",
            ]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            logger.info(f"Fetched {len(df)} dividend records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch dividends for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_stock_splits(self, symbol: str) -> pd.DataFrame:
        """
        Fetch full stock split history.

        AV function: SPLITS.  Returns a DataFrame with columns:
        effective_date, split_factor.
        Free tier.
        """
        logger.info(f"Fetching stock splits for {symbol}")
        self._wait_for_rate_limit()
        params = {"function": "SPLITS", "symbol": symbol, "apikey": self.api_key}
        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            data = response.json()
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()
            records = data.get("data", [])
            if not records:
                logger.debug(f"No split data for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame(records)
            df.columns = df.columns.str.lower().str.replace(" ", "_")
            if "effective_date" in df.columns:
                df["effective_date"] = pd.to_datetime(
                    df["effective_date"], errors="coerce"
                )
            if "split_factor" in df.columns:
                df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")
            logger.info(f"Fetched {len(df)} split records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch stock splits for {symbol}: {e}")
            return pd.DataFrame()

    # ========================================
    # Commodities, Forex, and Listing Status (AV Data Expansion)
    # ========================================

    def fetch_precious_metals_history(
        self,
        interval: str = "monthly",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch gold and silver price history.

        Makes separate GOLD and SILVER API calls.  Each returns a time series
        with date index and 'value' column (USD per troy ounce).

        Args:
            interval: daily, weekly, or monthly

        Returns:
            Tuple of (gold_df, silver_df).  Each is a DataFrame with datetime
            index (name='timestamp') and a float 'value' column.
            Returns (empty, empty) on total failure.
        """
        logger.info(f"Fetching precious metals history ({interval})")

        gold_df = self._fetch_metal("GOLD", interval)
        silver_df = self._fetch_metal("SILVER", interval)

        return gold_df, silver_df

    def _fetch_metal(self, metal: str, interval: str) -> pd.DataFrame:
        """Fetch a single precious metal time series (GOLD or SILVER)."""
        self._wait_for_rate_limit()
        params = {
            "function": metal,
            "interval": interval,
            "apikey": self.api_key,
        }
        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            self._increment_daily_count()
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")

            if "data" not in data:
                logger.warning(f"No data found for {metal}")
                return pd.DataFrame()

            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            df.index.name = "timestamp"
            return df[["value"]]

        except Exception as e:
            logger.error(f"Failed to fetch {metal}: {e}")
            return pd.DataFrame()

    def fetch_commodity_history(
        self,
        commodity: str,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch commodity price data for COPPER, ALL_COMMODITIES, etc.

        Same pattern as ``fetch_economic_indicator()`` — global endpoint,
        parse the 'data' key.  For WTI/BRENT/NATURAL_GAS use the existing
        ``fetch_commodity()`` method instead.

        Args:
            commodity: AV function name (COPPER, ALL_COMMODITIES, ALUMINUM,
                       WHEAT, CORN, COTTON, SUGAR, COFFEE)
            interval: daily, weekly, or monthly

        Returns:
            DataFrame with datetime index (name='timestamp') and float 'value'
            column.  Returns empty DataFrame on failure.
        """
        logger.info(f"Fetching {commodity} commodity history ({interval})")

        self._wait_for_rate_limit()

        params = {
            "function": commodity,
            "interval": interval,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            self._increment_daily_count()
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")

            if "data" not in data:
                logger.warning(f"No data found for {commodity}")
                return pd.DataFrame()

            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            df.index.name = "timestamp"
            return df[["value"]]

        except Exception as e:
            logger.error(f"Failed to fetch {commodity}: {e}")
            return pd.DataFrame()

    def fetch_forex_daily(
        self,
        from_symbol: str,
        to_symbol: str,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch daily foreign-exchange rates via FX_DAILY.

        Args:
            from_symbol: Source currency (e.g. 'EUR')
            to_symbol: Destination currency (e.g. 'USD')
            outputsize: 'compact' (100 points) or 'full' (20+ years)

        Returns:
            DataFrame with datetime index (name='timestamp') and OHLC columns
            (open, high, low, close) as floats.  Returns empty DataFrame on
            failure.
        """
        logger.info(f"Fetching FX daily {from_symbol}/{to_symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            self._increment_daily_count()
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")

            ts_key = "Time Series FX (Daily)"
            if ts_key not in data:
                logger.warning(f"No FX data found for {from_symbol}/{to_symbol}")
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(data[ts_key], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            df = df.rename(
                columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                }
            )

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df.index.name = "timestamp"
            return df[["open", "high", "low", "close"]]

        except Exception as e:
            logger.error(
                f"Failed to fetch FX daily {from_symbol}/{to_symbol}: {e}"
            )
            return pd.DataFrame()

    def fetch_listing_status(
        self,
        state: str = "active",
    ) -> pd.DataFrame:
        """
        Fetch listing status for US-listed equities (CSV endpoint).

        Args:
            state: 'active' or 'delisted'

        Returns:
            DataFrame with columns including symbol, name, exchange,
            assetType, ipoDate, status.  Returns empty DataFrame on failure.
        """
        logger.info(f"Fetching listing status (state={state})")

        self._wait_for_rate_limit()

        params = {
            "function": "LISTING_STATUS",
            "state": state,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            self._increment_daily_count()

            # Check if AV returned a JSON error instead of CSV
            try:
                data = response.json()
                if "Error Message" in data:
                    raise ValueError(f"API Error: {data['Error Message']}")
                if "Note" in data:
                    logger.warning(f"API Note: {data['Note']}")
                    return pd.DataFrame()
            except (ValueError, KeyError):
                pass  # Not JSON — continue with CSV parsing

            if not response.text or not response.text.strip():
                logger.warning("Empty listing status response")
                return pd.DataFrame()

            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                logger.warning("No listing data found")
                return pd.DataFrame()

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Convert date columns
            for col in ["ipodate", "delistingdate"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            logger.info(f"Fetched {len(df)} listings (state={state})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch listing status: {e}")
            return pd.DataFrame()

    def fetch_realtime_pcr(
        self,
        symbol: str,
    ) -> dict | None:
        """
        Fetch realtime put/call ratio for a symbol.

        This endpoint may be blocked or return demo data on some plans.
        Returns None if the data is unavailable or blocked.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with PCR data, or None if blocked/unavailable.
        """
        logger.info(f"Fetching realtime PCR for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "REALTIME_PUT_CALL_RATIO",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            self._increment_daily_count()
            data = response.json()

            if "Error Message" in data:
                logger.warning(f"PCR endpoint error for {symbol}: {data['Error Message']}")
                return None
            if "Information" in data:
                logger.info(f"PCR endpoint blocked/demo for {symbol}")
                return None
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return None

            if not data:
                return None

            logger.info(f"Fetched realtime PCR for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch realtime PCR for {symbol}: {e}")
            return None

    def fetch_historical_pcr(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Fetch historical put/call ratio time series.

        This endpoint may be blocked or return demo data on some plans.
        Returns empty DataFrame if unavailable.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with datetime index and 'put_call_ratio' column.
            Empty DataFrame if blocked or on failure.
        """
        logger.info(f"Fetching historical PCR for {symbol}")

        self._wait_for_rate_limit()

        params = {
            "function": "HISTORICAL_PUT_CALL_RATIO",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                self.base_url, params=params, timeout=30, verify=CA_BUNDLE
            )
            response.raise_for_status()
            self._fallback_call_count += 1
            self._increment_daily_count()
            data = response.json()

            if "Error Message" in data:
                logger.warning(
                    f"Historical PCR error for {symbol}: {data['Error Message']}"
                )
                return pd.DataFrame()
            if "Information" in data:
                logger.info(f"Historical PCR blocked/demo for {symbol}")
                return pd.DataFrame()
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return pd.DataFrame()

            records = data.get("data", [])
            if not records:
                logger.warning(f"No historical PCR data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df = df.sort_index()
            if "put_call_ratio" in df.columns:
                df["put_call_ratio"] = pd.to_numeric(
                    df["put_call_ratio"], errors="coerce"
                )
            df = df.dropna()
            df.index.name = "timestamp"

            logger.info(f"Fetched {len(df)} historical PCR records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical PCR for {symbol}: {e}")
            return pd.DataFrame()
