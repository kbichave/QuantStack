"""OHLCV and intraday data operations mixin for DataStore.

Covers save/load for daily OHLCV, 1-minute bars, tick data, and metadata queries.
"""

from datetime import datetime

import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe


class OHLCVMixin:
    """Mixin for OHLCV, intraday bar, and tick data operations."""

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        replace: bool = False,
    ) -> int:
        """
        Save OHLCV data to the database.

        Args:
            df: DataFrame with OHLCV data and DatetimeIndex
            symbol: Stock symbol
            timeframe: Data timeframe
            replace: If True, replace existing data; if False, upsert

        Returns:
            Number of rows saved
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {symbol} {timeframe.value}")
            return 0

        # Prepare data — keep only OHLCV columns (providers may add vwap, trade_count, etc.)
        data = df.copy()
        core_cols = ["open", "high", "low", "close", "volume"]
        extra = [c for c in data.columns if c.lower() not in core_cols]
        if extra:
            data = data.drop(columns=extra, errors="ignore")
        data = data.reset_index()
        data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        data["symbol"] = symbol
        data["timeframe"] = timeframe.value

        with self._use_conn() as conn:
            if replace:
                conn.execute(
                    "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                    [symbol, timeframe.value],
                )

            conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                FROM data
            """
            )

            # Update metadata inline (avoid nested _use_conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO data_metadata
                (symbol, timeframe, first_timestamp, last_timestamp, row_count, updated_at)
                SELECT symbol, timeframe,
                    MIN(timestamp), MAX(timestamp), COUNT(*), CURRENT_TIMESTAMP
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                GROUP BY symbol, timeframe
            """,
                [symbol, timeframe.value],
            )

        logger.info(f"Saved {len(data)} rows for {symbol} {timeframe.value}")
        return len(data)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from the database.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)

        Returns:
            DataFrame with OHLCV data and DatetimeIndex
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe.value]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self._use_conn() as conn:
            df = conn.execute(query, params).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        logger.debug(f"Loaded {len(df)} rows for {symbol} {timeframe.value}")
        return df

    def load_multi_timeframe(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[Timeframe, pd.DataFrame]:
        """
        Load data for all timeframes for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}
        for tf in Timeframe:
            result[tf] = self.load_ohlcv(symbol, tf, start_date, end_date)
        return result

    def _update_metadata(self, symbol: str, timeframe: Timeframe) -> None:
        """Update metadata for a symbol/timeframe combination."""
        with self._use_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_metadata
                (symbol, timeframe, first_timestamp, last_timestamp, row_count, updated_at)
                SELECT
                    symbol,
                    timeframe,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(*) as row_count,
                    CURRENT_TIMESTAMP as updated_at
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                GROUP BY symbol, timeframe
            """,
                [symbol, timeframe.value],
            )

    def get_metadata(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
    ) -> pd.DataFrame:
        """
        Get data metadata.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe

        Returns:
            DataFrame with metadata
        """
        query = "SELECT * FROM data_metadata WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe.value)

        with self._use_conn() as conn:
            return conn.execute(query, params).fetchdf()

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols in the database."""
        with self._use_conn() as conn:
            result = conn.execute(
                """
                SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol
            """
            ).fetchall()
        return [row[0] for row in result]

    def get_date_range(
        self,
        symbol: str,
        timeframe: Timeframe,
    ) -> tuple[datetime | None, datetime | None]:
        """
        Get the date range for a symbol/timeframe.

        Returns:
            Tuple of (first_date, last_date) or (None, None) if no data
        """
        with self._use_conn() as conn:
            result = conn.execute(
                """
                SELECT MIN(timestamp), MAX(timestamp)
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """,
                [symbol, timeframe.value],
            ).fetchone()

        if result and result[0]:
            return (result[0], result[1])
        return (None, None)

    def has_required_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_bars: int = 100,
        completeness_threshold: float = 0.9,
    ) -> tuple[bool, str]:
        """
        Check if we have sufficient data for a symbol/timeframe/date range.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            start_date: Required start date (inclusive)
            end_date: Required end date (inclusive)
            min_bars: Minimum number of bars required
            completeness_threshold: Minimum fraction of expected bars

        Returns:
            Tuple of (has_data, reason)
        """
        # Get metadata for this symbol/timeframe
        metadata = self.get_metadata(symbol=symbol, timeframe=timeframe)

        if metadata.empty:
            return (False, f"No data for {symbol} {timeframe.value}")

        meta_row = metadata.iloc[0]
        row_count = meta_row.get("row_count", 0)
        first_ts = meta_row.get("first_timestamp")
        last_ts = meta_row.get("last_timestamp")

        # Check minimum bars
        if row_count < min_bars:
            return (False, f"Too few bars: {row_count} < {min_bars}")

        # Check date range coverage
        if start_date and first_ts:
            first_ts = pd.Timestamp(first_ts)
            start_ts = pd.Timestamp(start_date)

            # Allow up to 7 days slack for weekends/holidays
            if first_ts > start_ts + pd.Timedelta(days=7):
                return (False, f"Data starts too late: {first_ts} > {start_date}")

        if end_date and last_ts:
            last_ts = pd.Timestamp(last_ts)
            end_ts = pd.Timestamp(end_date)

            # Allow up to 7 days slack
            if last_ts < end_ts - pd.Timedelta(days=7):
                return (False, f"Data ends too early: {last_ts} < {end_date}")

        # Check completeness if date range provided
        if start_date and end_date:
            days = (end_date - start_date).days

            # Bars per trading day by timeframe (US equity session = 390 min / 6.5 h)
            _bars_per_day = {
                Timeframe.W1: 5 / 7,  # 1 bar per 5 trading days
                Timeframe.D1: 5 / 7,  # 1 bar per trading day
                Timeframe.H4: 5 / 7 * 2,  # ~2 bars per trading day
                Timeframe.H1: 5 / 7 * 6.5,
                Timeframe.M30: 5 / 7 * 13,
                Timeframe.M15: 5 / 7 * 26,
                Timeframe.M5: 5 / 7 * 78,
                Timeframe.M1: 5 / 7 * 390,
                Timeframe.S5: 5 / 7 * 390 * 12,  # 12 × 5s bars per minute
            }
            expected_bars = max(1, int(days * _bars_per_day.get(timeframe, 5 / 7)))

            if expected_bars > 0:
                completeness = row_count / expected_bars
                if completeness < completeness_threshold:
                    return (
                        False,
                        f"Incomplete: {completeness:.1%} < {completeness_threshold:.1%}",
                    )

        return (True, f"Valid: {row_count} bars")

    def delete_symbol(self, symbol: str) -> int:
        """
        Delete all data for a symbol.

        Args:
            symbol: Symbol to delete

        Returns:
            Number of rows deleted
        """
        result = self.conn.execute(
            """
            DELETE FROM ohlcv WHERE symbol = ?
        """,
            [symbol],
        )

        self.conn.execute(
            """
            DELETE FROM data_metadata WHERE symbol = ?
        """,
            [symbol],
        )

        deleted = result.rowcount
        logger.info(f"Deleted {deleted} rows for {symbol}")
        return deleted

    def vacuum(self) -> None:
        """Optimize database storage."""
        self.conn.execute("VACUUM")
        logger.info("Database vacuumed")

    # ── 1-Minute Bar Methods ───────────────────────────────────────────────────

    def save_ohlcv_1m(
        self,
        df: pd.DataFrame,
        symbol: str,
        replace: bool = False,
    ) -> int:
        """Save 1-minute OHLCV bars.

        Args:
            df: DatetimeIndex DataFrame with [open, high, low, close, volume] columns.
                Optional columns: vwap, trade_count.
            symbol: Ticker symbol.
            replace: If True, delete existing rows for this symbol first.

        Returns:
            Number of rows written.
        """
        if df.empty:
            logger.warning(f"Empty 1m DataFrame for {symbol}")
            return 0

        data = df.copy().reset_index()
        data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)
        data["symbol"] = symbol

        # Pad optional columns so the INSERT always has the same shape
        if "vwap" not in data.columns:
            data["vwap"] = None
        if "trade_count" not in data.columns:
            data["trade_count"] = None

        with self._use_conn() as conn:
            if replace:
                conn.execute("DELETE FROM ohlcv_1m WHERE symbol = ?", [symbol])

            conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv_1m
                    (symbol, timestamp, open, high, low, close, volume, vwap, trade_count)
                SELECT symbol, timestamp, open, high, low, close, volume, vwap, trade_count
                FROM data
            """
            )

        logger.info(f"Saved {len(data)} 1m bars for {symbol}")
        return len(data)

    def load_ohlcv_1m(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load 1-minute OHLCV bars.

        Returns:
            DataFrame with DatetimeIndex named 'timestamp' and columns
            [open, high, low, close, volume]; vwap and trade_count included
            when present.
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, vwap, trade_count
            FROM ohlcv_1m
            WHERE symbol = ?
        """
        params: list = [symbol]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self._use_conn() as conn:
            df = conn.execute(query, params).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

        # Drop optional columns when entirely null (providers that don't supply them)
        for col in ("vwap", "trade_count"):
            if col in df.columns and df[col].isna().all():
                df.drop(columns=[col], inplace=True)

        logger.debug(f"Loaded {len(df)} 1m bars for {symbol}")
        return df

    # ── Tick Data Methods ──────────────────────────────────────────────────────

    def save_ticks(
        self,
        ticks: "list",
        symbol: str,
    ) -> int:
        """Persist a list of ``TradeTick`` objects to the ``tick_data`` table.

        Args:
            ticks:  List of ``TradeTick`` instances.
            symbol: Ticker symbol (must match ``tick.symbol`` for all entries).

        Returns:
            Number of rows written.
        """
        if not ticks:
            return 0

        records = [
            {
                "symbol": t.symbol,
                "timestamp_ns": t.timestamp_ns,
                "price": t.price,
                "size": t.size,
                "bid": None,
                "ask": None,
                "bid_size": None,
                "ask_size": None,
                "side": t.side,
                "exchange": t.exchange,
            }
            for t in ticks
        ]
        pd.DataFrame(records)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO tick_data
                (symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange)
            SELECT symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange
            FROM data
        """
        )
        logger.debug(f"Saved {len(records)} ticks for {symbol}")
        return len(records)

    def save_quotes(
        self,
        quotes: "list",
        symbol: str,
    ) -> int:
        """Persist a list of ``QuoteTick`` objects to the ``tick_data`` table.

        Quote ticks are stored with NULL price/size and bid/ask/bid_size/ask_size
        populated.  Side is stored as "quote" to distinguish from trade rows.

        Args:
            quotes: List of ``QuoteTick`` instances.
            symbol: Ticker symbol.

        Returns:
            Number of rows written.
        """
        if not quotes:
            return 0

        records = [
            {
                "symbol": q.symbol,
                "timestamp_ns": q.timestamp_ns,
                "price": q.mid,
                "size": q.bid_size + q.ask_size,
                "bid": q.bid,
                "ask": q.ask,
                "bid_size": q.bid_size,
                "ask_size": q.ask_size,
                "side": "quote",
                "exchange": None,
            }
            for q in quotes
        ]
        pd.DataFrame(records)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO tick_data
                (symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange)
            SELECT symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange
            FROM data
        """
        )
        logger.debug(f"Saved {len(records)} quotes for {symbol}")
        return len(records)

    def load_ticks(
        self,
        symbol: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
        tick_type: str = "trade",
    ) -> pd.DataFrame:
        """Load tick data from the ``tick_data`` table.

        Args:
            symbol:     Ticker symbol.
            start_ns:   Start timestamp in nanoseconds (inclusive).
            end_ns:     End timestamp in nanoseconds (inclusive).
            tick_type:  "trade" for trade ticks, "quote" for NBBO quotes, "all".

        Returns:
            DataFrame with columns matching the ``tick_data`` schema.
            ``timestamp_ns`` is preserved as int64 (no datetime conversion).
        """
        query = "SELECT * FROM tick_data WHERE symbol = ?"
        params: list = [symbol]

        if tick_type == "trade":
            query += " AND side != 'quote'"
        elif tick_type == "quote":
            query += " AND side = 'quote'"

        if start_ns is not None:
            query += " AND timestamp_ns >= ?"
            params.append(start_ns)
        if end_ns is not None:
            query += " AND timestamp_ns <= ?"
            params.append(end_ns)

        query += " ORDER BY timestamp_ns"

        df = self.conn.execute(query, params).fetchdf()
        logger.debug(f"Loaded {len(df)} {tick_type} rows for {symbol}")
        return df
