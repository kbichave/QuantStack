"""Options chain and news/earnings operations mixin for DataStore.

Covers save/load for options chains, earnings calendar, company overview,
and news sentiment.
"""

from datetime import datetime

import pandas as pd
from loguru import logger


class OptionsNewsMixin:
    """Mixin for options, earnings, company overview, and news sentiment operations."""

    # ========================================
    # Options Data Methods
    # ========================================

    def save_options_chain(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_date: datetime,
        replace: bool = True,
    ) -> int:
        """
        Save options chain data to the database.

        Args:
            df: DataFrame with options data (from fetch_realtime_options or fetch_historical_options)
            symbol: Underlying symbol
            data_date: Date of the data
            replace: If True, replace existing data for this symbol/date

        Returns:
            Number of rows saved
        """
        if df.empty:
            logger.warning(f"Empty options DataFrame for {symbol}")
            return 0

        data = df.copy()

        # Ensure required columns
        if "underlying" not in data.columns:
            data["underlying"] = symbol

        if "data_date" not in data.columns:
            data["data_date"] = data_date

        # Ensure data_date is date type
        if isinstance(data["data_date"].iloc[0], pd.Timestamp):
            data["data_date"] = data["data_date"].dt.date

        if replace:
            # Delete existing data for this symbol/date
            self.conn.execute(
                """
                DELETE FROM options_chains
                WHERE underlying = ? AND data_date = ?
            """,
                [symbol, data_date.date() if hasattr(data_date, "date") else data_date],
            )

        # Select only columns that exist in the table
        valid_columns = [
            "contract_id",
            "underlying",
            "data_date",
            "expiry",
            "strike",
            "option_type",
            "bid",
            "ask",
            "mid",
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

        available_columns = [c for c in valid_columns if c in data.columns]
        data[available_columns]

        # Insert data
        self.conn.execute(f"""
            INSERT OR REPLACE INTO options_chains
            ({", ".join(available_columns)})
            SELECT {", ".join(available_columns)}
            FROM insert_data
        """)

        logger.info(f"Saved {len(data)} options contracts for {symbol} on {data_date}")
        return len(data)

    def load_options_chain(
        self,
        symbol: str,
        data_date: datetime | None = None,
        expiry_min: datetime | None = None,
        expiry_max: datetime | None = None,
        option_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Load options chain data from the database.

        Args:
            symbol: Underlying symbol
            data_date: Specific date to load (if None, loads most recent)
            expiry_min: Minimum expiry date filter
            expiry_max: Maximum expiry date filter
            option_type: Filter by "call" or "put"

        Returns:
            DataFrame with options data
        """
        query = """
            SELECT *
            FROM options_chains
            WHERE underlying = ?
        """
        params = [symbol]

        if data_date:
            query += " AND data_date = ?"
            params.append(data_date.date() if hasattr(data_date, "date") else data_date)
        else:
            # Get most recent date
            query += (
                " AND data_date = (SELECT MAX(data_date) FROM options_chains WHERE underlying = ?)"
            )
            params.append(symbol)

        if expiry_min:
            query += " AND expiry >= ?"
            params.append(expiry_min.date() if hasattr(expiry_min, "date") else expiry_min)

        if expiry_max:
            query += " AND expiry <= ?"
            params.append(expiry_max.date() if hasattr(expiry_max, "date") else expiry_max)

        if option_type:
            query += " AND LOWER(option_type) = ?"
            params.append(option_type.lower())

        query += " ORDER BY expiry, strike"

        df = self.conn.execute(query, params).fetchdf()

        if df.empty:
            logger.debug(f"No options data found for {symbol}")
            return pd.DataFrame()

        # Convert date columns
        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"])
        if "data_date" in df.columns:
            df["data_date"] = pd.to_datetime(df["data_date"])

        logger.debug(f"Loaded {len(df)} options contracts for {symbol}")
        return df

    def save_earnings_calendar(
        self,
        df: pd.DataFrame,
        replace: bool = False,
    ) -> int:
        """
        Save earnings calendar data.

        Args:
            df: DataFrame with earnings data
            replace: If True, replace all existing data

        Returns:
            Number of rows saved
        """
        if df.empty:
            return 0

        data = df.copy()

        # Standardize column names
        if "report_date" not in data.columns and "reportdate" in data.columns:
            data = data.rename(columns={"reportdate": "report_date"})

        if replace:
            self.conn.execute("DELETE FROM earnings_calendar")

        # Select valid columns
        valid_columns = [
            "symbol",
            "report_date",
            "fiscal_date_ending",
            "estimate",
            "reported_eps",
            "surprise",
            "surprise_pct",
        ]

        available_columns = [c for c in valid_columns if c in data.columns]
        data[available_columns]

        self.conn.execute(f"""
            INSERT OR REPLACE INTO earnings_calendar
            ({", ".join(available_columns)})
            SELECT {", ".join(available_columns)}
            FROM insert_data
        """)

        logger.info(f"Saved {len(data)} earnings records")
        return len(data)

    def load_earnings_calendar(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load earnings calendar data.

        Args:
            symbol: Filter by symbol
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with earnings data
        """
        query = "SELECT * FROM earnings_calendar WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND report_date >= ?"
            params.append(start_date.date() if hasattr(start_date, "date") else start_date)

        if end_date:
            query += " AND report_date <= ?"
            params.append(end_date.date() if hasattr(end_date, "date") else end_date)

        query += " ORDER BY report_date"

        df = self.conn.execute(query, params).fetchdf()

        if not df.empty and "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])

        return df

    def get_days_to_earnings(
        self,
        symbol: str,
        as_of_date: datetime | None = None,
    ) -> int | None:
        """
        Get days until next earnings announcement for a symbol.

        Args:
            symbol: Stock symbol
            as_of_date: Reference date (default: today)

        Returns:
            Number of days to next earnings, or None if not found
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        result = self.conn.execute(
            """
            SELECT MIN(report_date) as next_earnings
            FROM earnings_calendar
            WHERE symbol = ? AND report_date >= ?
        """,
            [symbol, as_of_date.date()],
        ).fetchone()

        if result and result[0]:
            next_date = pd.to_datetime(result[0])
            return (next_date - pd.Timestamp(as_of_date)).days

        return None

    def save_company_overview(
        self,
        data: dict,
    ) -> None:
        """
        Save company overview data.

        Args:
            data: Dictionary from fetch_company_overview
        """
        if not data or "Symbol" not in data:
            return

        self.conn.execute(
            """
            INSERT OR REPLACE INTO company_overview
            (symbol, name, sector, industry, market_cap, dividend_yield,
             ex_dividend_date, fifty_two_week_high, fifty_two_week_low, beta, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            [
                data.get("Symbol"),
                data.get("Name"),
                data.get("Sector"),
                data.get("Industry"),
                data.get("MarketCapitalization"),
                data.get("DividendYield"),
                data.get("ExDividendDate"),
                data.get("52WeekHigh"),
                data.get("52WeekLow"),
                data.get("Beta"),
            ],
        )

        logger.debug(f"Saved company overview for {data.get('Symbol')}")

    def load_company_overview(
        self,
        symbol: str,
    ) -> dict | None:
        """
        Load company overview data.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company data or None
        """
        result = self.conn.execute(
            """
            SELECT * FROM company_overview WHERE symbol = ?
        """,
            [symbol],
        ).fetchdf()

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    # ========================================
    # News Sentiment Methods
    # ========================================

    def save_news_sentiment(
        self,
        df: pd.DataFrame,
        replace: bool = False,
    ) -> int:
        """
        Save news sentiment data to the database.

        Args:
            df: DataFrame with news sentiment data
                Expected columns: time_published (index or column),
                title, summary, source, url, ticker,
                overall_sentiment_score, overall_sentiment_label,
                ticker_sentiment_score, ticker_sentiment_label, relevance_score
            replace: If True, delete existing data first

        Returns:
            Number of rows saved
        """
        if df.empty:
            logger.warning("Empty news sentiment DataFrame provided")
            return 0

        data = df.copy()

        # Handle datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            if "index" in data.columns:
                data = data.rename(columns={"index": "time_published"})

        # Ensure time_published exists
        if "time_published" not in data.columns:
            logger.error("No time_published column in news data")
            return 0

        # Convert time_published to timestamp
        if not pd.api.types.is_datetime64_any_dtype(data["time_published"]):
            data["time_published"] = pd.to_datetime(data["time_published"])

        if replace:
            self.conn.execute("DELETE FROM news_sentiment")

        # Valid columns for the table
        valid_columns = [
            "time_published",
            "title",
            "summary",
            "source",
            "url",
            "ticker",
            "overall_sentiment_score",
            "overall_sentiment_label",
            "ticker_sentiment_score",
            "ticker_sentiment_label",
            "relevance_score",
        ]

        available_columns = [c for c in valid_columns if c in data.columns]
        data[available_columns].copy()

        # Handle duplicates by using INSERT OR IGNORE
        self.conn.execute(f"""
            INSERT OR IGNORE INTO news_sentiment
            ({", ".join(available_columns)})
            SELECT {", ".join(available_columns)}
            FROM insert_data
        """)

        logger.info(f"Saved {len(data)} news sentiment records")
        return len(data)

    def load_news_sentiment(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load news sentiment data from the database.

        Args:
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)
            tickers: List of tickers to filter (if None, load all)

        Returns:
            DataFrame with news sentiment data, indexed by time_published
        """
        query = """
            SELECT time_published, title, summary, source, url, ticker,
                   overall_sentiment_score, overall_sentiment_label,
                   ticker_sentiment_score, ticker_sentiment_label, relevance_score
            FROM news_sentiment
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND time_published >= ?"
            params.append(start_date)

        if end_date:
            query += " AND time_published <= ?"
            params.append(end_date)

        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        query += " ORDER BY time_published"

        df = self.conn.execute(query, params).fetchdf()

        if df.empty:
            logger.debug("No news sentiment data found")
            return pd.DataFrame()

        # Set time_published as index
        df["time_published"] = pd.to_datetime(df["time_published"])
        df = df.set_index("time_published")

        logger.debug(f"Loaded {len(df)} news sentiment records")
        return df

    def get_news_date_range(
        self,
        tickers: list[str] | None = None,
    ) -> tuple[datetime | None, datetime | None]:
        """
        Get the date range of available news data.

        Args:
            tickers: Optional list of tickers to filter

        Returns:
            Tuple of (first_date, last_date) or (None, None) if no data
        """
        query = "SELECT MIN(time_published), MAX(time_published) FROM news_sentiment"
        params = []

        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            query += f" WHERE ticker IN ({placeholders})"
            params.extend(tickers)

        result = self.conn.execute(query, params).fetchone()

        if result and result[0]:
            return (pd.to_datetime(result[0]), pd.to_datetime(result[1]))
        return (None, None)

    def get_news_count(
        self,
        tickers: list[str] | None = None,
    ) -> int:
        """
        Get count of news articles in database.

        Args:
            tickers: Optional list of tickers to filter

        Returns:
            Number of news articles
        """
        query = "SELECT COUNT(*) FROM news_sentiment"
        params = []

        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            query += f" WHERE ticker IN ({placeholders})"
            params.extend(tickers)

        result = self.conn.execute(query, params).fetchone()
        return result[0] if result else 0

    def delete_news_sentiment(
        self,
        tickers: list[str] | None = None,
        before_date: datetime | None = None,
    ) -> int:
        """
        Delete news sentiment data.

        Args:
            tickers: Delete only for specific tickers (if None, affects all)
            before_date: Delete only data before this date (if None, deletes all matching)

        Returns:
            Number of rows deleted
        """
        query = "DELETE FROM news_sentiment WHERE 1=1"
        params = []

        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        if before_date:
            query += " AND time_published < ?"
            params.append(before_date)

        result = self.conn.execute(query, params)
        deleted = result.rowcount
        logger.info(f"Deleted {deleted} news sentiment records")
        return deleted
