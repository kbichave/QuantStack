"""
Alpha Vantage API client for fetching market data.

Handles rate limiting, retries, and data validation.
"""

import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from loguru import logger

from quantcore.config.settings import get_settings


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

    # Try certifi as fallback
    try:
        import certifi
        return certifi.where()
    except ImportError:
        # Returning True tells requests to use the system's default CA bundle
        # (i.e. standard SSL verification — NOT disabled).
        return True

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
        self._call_count: int = 0
        self._minute_start: float = time.time()

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self._minute_start >= 60:
            self._call_count = 0
            self._minute_start = current_time

        # Wait if we've hit the rate limit
        if self._call_count >= self.rate_limit:
            wait_time = 60 - (current_time - self._minute_start) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._call_count = 0
                self._minute_start = time.time()

    def _make_request(
        self,
        function: str,
        symbol: str,
        **kwargs
    ) -> dict:
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

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            **kwargs
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
                response.raise_for_status()
                self._call_count += 1

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
                    wait_time = 2 ** attempt
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
            **{"month": month} if month else {}
        )

        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.warning(f"No time series data found for {symbol} {month or 'recent'}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        })

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
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        })

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
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
            "8. split coefficient": "split_coef",
        })

        # Use adjusted close as close
        if "adj_close" in df.columns:
            # Adjust OHLC for splits/dividends
            adjustment_factor = df["adj_close"].astype(float) / df["close"].astype(float)
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
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
        })

        # Use adjusted close
        if "adj_close" in df.columns:
            adjustment_factor = df["adj_close"].astype(float) / df["close"].astype(float)
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
        logger.info(f"Fetching complete intraday history for {symbol} ({start_year}-{end_year})")
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
                    df = self.fetch_intraday_by_month(symbol, interval, month_str, outputsize="full")
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
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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
                        article["ticker_sentiment_score"] = float(ticker_data.get("ticker_sentiment_score", 0))
                        article["ticker_sentiment_label"] = ticker_data.get("ticker_sentiment_label", "")
                        article["relevance_score"] = float(ticker_data.get("relevance_score", 0))
                        break

                articles.append(article)

            df = pd.DataFrame(articles)

            if not df.empty and "time_published" in df.columns:
                df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
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
        logger.info(f"Fetching historical news sentiment from {start_date} to {end_date or 'now'}")

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

            logger.info(f"Fetching batch {batch_count + 1}: {current.date()} to {batch_end.date()}")

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

                response = requests.get(self.base_url, params=params, timeout=60, verify=CA_BUNDLE)
                response.raise_for_status()
                self._call_count += 1
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
            df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")

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

        logger.info(f"Fetched total {len(df)} unique news articles from {batch_count} batches")
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
                "summary": item.get("summary", "")[:500] if item.get("summary") else "",  # Truncate
                "source": item.get("source", ""),
                "overall_sentiment_score": float(item.get("overall_sentiment_score", 0) or 0),
                "overall_sentiment_label": item.get("overall_sentiment_label", ""),
            }

            # Extract ticker-specific sentiment (find first matching ticker)
            for ticker_data in item.get("ticker_sentiment", []):
                ticker = ticker_data.get("ticker", "")
                if ticker in ticker_list:
                    article["ticker"] = ticker
                    article["ticker_sentiment_score"] = float(ticker_data.get("ticker_sentiment_score", 0) or 0)
                    article["ticker_sentiment_label"] = ticker_data.get("ticker_sentiment_label", "")
                    article["relevance_score"] = float(ticker_data.get("relevance_score", 0) or 0)
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
            response = requests.get(self.base_url, params=params, timeout=60, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Convert numeric columns
            numeric_cols = ["strike", "bid", "ask", "last", "volume", "open_interest",
                          "iv", "delta", "gamma", "theta", "vega", "rho"]
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
        logger.info(f"Fetching historical options for {symbol}" + (f" on {date}" if date else ""))

        self._wait_for_rate_limit()

        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        if date:
            params["date"] = date

        try:
            response = requests.get(self.base_url, params=params, timeout=60, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Convert numeric columns
            numeric_cols = ["strike", "bid", "ask", "last", "volume", "open_interest",
                          "iv", "delta", "gamma", "theta", "vega", "rho"]
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
            df["data_date"] = pd.to_datetime(date) if date else pd.Timestamp.now().normalize()

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
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1

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
            from io import StringIO
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
                df["fiscal_date_ending"] = pd.to_datetime(df["fiscaldateending"], errors="coerce")
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
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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
                "MarketCapitalization", "EBITDA", "PERatio", "PEGRatio",
                "BookValue", "DividendPerShare", "DividendYield", "EPS",
                "RevenuePerShareTTM", "ProfitMargin", "OperatingMarginTTM",
                "ReturnOnAssetsTTM", "ReturnOnEquityTTM", "RevenueTTM",
                "GrossProfitTTM", "DilutedEPSTTM", "QuarterlyEarningsGrowthYOY",
                "QuarterlyRevenueGrowthYOY", "AnalystTargetPrice", "TrailingPE",
                "ForwardPE", "PriceToSalesRatioTTM", "PriceToBookRatio",
                "EVToRevenue", "EVToEBITDA", "Beta", "52WeekHigh", "52WeekLow",
                "50DayMovingAverage", "200DayMovingAverage", "SharesOutstanding",
                "SharesFloat", "SharesShort", "SharesShortPriorMonth",
                "ShortRatio", "ShortPercentOutstanding", "ShortPercentFloat",
                "PercentInsiders", "PercentInstitutions",
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
            logger.warning(f"Bulk quotes limited to 100 symbols, truncating from {len(symbols)}")
            symbols = symbols[:100]

        logger.info(f"Fetching bulk quotes for {len(symbols)} symbols")

        self._wait_for_rate_limit()

        params = {
            "function": "REALTIME_BULK_QUOTES",
            "symbol": ",".join(symbols),
            "apikey": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30, verify=CA_BUNDLE)
            response.raise_for_status()
            self._call_count += 1
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

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Convert numeric columns
            numeric_cols = ["open", "high", "low", "price", "volume", "previous_close", "change"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parse change percent
            if "change_percent" in df.columns:
                df["change_percent"] = df["change_percent"].str.rstrip("%").astype(float) / 100

            logger.info(f"Fetched quotes for {len(df)} symbols")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch bulk quotes: {e}")
            return pd.DataFrame()

