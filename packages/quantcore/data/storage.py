"""
DuckDB-based data storage for multi-timeframe market data.

Provides efficient storage and retrieval with partitioning by symbol and timeframe.
"""

import os
import threading
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import duckdb
import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.config.settings import get_settings


class DataStore:
    """
    DuckDB-based storage for OHLCV market data.
    
    Features:
    - Efficient columnar storage
    - Partitioning by symbol and timeframe
    - Fast analytical queries
    - Support for multi-timeframe data
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the data store.

        Args:
            db_path: Path to database file (uses settings if not provided)
        """
        settings = get_settings()
        self.db_path = db_path or settings.database_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._conn_lock = threading.Lock()
        self._init_schema()
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection (thread-safe double-checked locking)."""
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:
                    self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Main OHLCV table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        
        # Metadata table for tracking data freshness
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                first_timestamp TIMESTAMP,
                last_timestamp TIMESTAMP,
                row_count INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe)
            )
        """)
        
        # Index for the most common access pattern: symbol + timeframe range scans
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
            ON ohlcv (symbol, timeframe, timestamp)
        """)

        # Initialize options schema
        self._init_options_schema()

        # Initialize news sentiment schema
        self._init_news_sentiment_schema()

        # Initialize intraday / tick schema
        self._init_intraday_schema()

        logger.debug("Database schema initialized")
    
    def _init_intraday_schema(self) -> None:
        """Initialize high-frequency tables for intraday and tick data.

        Kept separate from `ohlcv` to avoid index bloat: 1-minute bars produce
        ~390 rows/day/symbol; at 100 symbols × 5 years that is ~50M rows.
        Keeping them in a dedicated table lets DuckDB scan them with predicate
        pushdown on (symbol, timestamp) without touching the swing/daily index.
        """
        # 1-minute bars — TIMESTAMPTZ for unambiguous intraday storage.
        # vwap and trade_count are optional (Alpaca/Polygon provide them; AV does not).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_1m (
                symbol      VARCHAR   NOT NULL,
                timestamp   TIMESTAMPTZ NOT NULL,
                open        DOUBLE    NOT NULL,
                high        DOUBLE    NOT NULL,
                low         DOUBLE    NOT NULL,
                close       DOUBLE    NOT NULL,
                volume      DOUBLE    NOT NULL,
                vwap        DOUBLE,
                trade_count INTEGER,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_ts
            ON ohlcv_1m (symbol, timestamp)
        """)

        # Tick data — nanosecond int64 timestamps avoid float precision loss.
        # bid/ask/bid_size/ask_size are NULL for trade-only ticks.
        # side: 'buy' | 'sell' | 'unknown' (derived from aggressor flag or rule).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tick_data (
                symbol       VARCHAR NOT NULL,
                timestamp_ns BIGINT  NOT NULL,
                price        DOUBLE  NOT NULL,
                size         DOUBLE  NOT NULL,
                bid          DOUBLE,
                ask          DOUBLE,
                bid_size     DOUBLE,
                ask_size     DOUBLE,
                side         VARCHAR,
                exchange     VARCHAR,
                PRIMARY KEY (symbol, timestamp_ns)
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_ts
            ON tick_data (symbol, timestamp_ns)
        """)

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

        if replace:
            self.conn.execute("DELETE FROM ohlcv_1m WHERE symbol = ?", [symbol])

        self.conn.execute("""
            INSERT OR REPLACE INTO ohlcv_1m
                (symbol, timestamp, open, high, low, close, volume, vwap, trade_count)
            SELECT symbol, timestamp, open, high, low, close, volume, vwap, trade_count
            FROM data
        """)

        logger.info(f"Saved {len(data)} 1m bars for {symbol}")
        return len(data)

    def load_ohlcv_1m(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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

        df = self.conn.execute(query, params).fetchdf()

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

        from quantcore.data.streaming.tick_models import TradeTick
        records = [
            {
                "symbol":       t.symbol,
                "timestamp_ns": t.timestamp_ns,
                "price":        t.price,
                "size":         t.size,
                "bid":          None,
                "ask":          None,
                "bid_size":     None,
                "ask_size":     None,
                "side":         t.side,
                "exchange":     t.exchange,
            }
            for t in ticks
        ]
        data = pd.DataFrame(records)
        self.conn.execute("""
            INSERT OR IGNORE INTO tick_data
                (symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange)
            SELECT symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange
            FROM data
        """)
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

        from quantcore.data.streaming.tick_models import QuoteTick
        records = [
            {
                "symbol":       q.symbol,
                "timestamp_ns": q.timestamp_ns,
                "price":        q.mid,
                "size":         q.bid_size + q.ask_size,
                "bid":          q.bid,
                "ask":          q.ask,
                "bid_size":     q.bid_size,
                "ask_size":     q.ask_size,
                "side":         "quote",
                "exchange":     None,
            }
            for q in quotes
        ]
        data = pd.DataFrame(records)
        self.conn.execute("""
            INSERT OR IGNORE INTO tick_data
                (symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange)
            SELECT symbol, timestamp_ns, price, size, bid, ask, bid_size, ask_size, side, exchange
            FROM data
        """)
        logger.debug(f"Saved {len(records)} quotes for {symbol}")
        return len(records)

    def load_ticks(
        self,
        symbol: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
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

    def _init_options_schema(self) -> None:
        """Initialize options-specific tables."""
        # Options chains table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chains (
                contract_id VARCHAR NOT NULL,
                underlying VARCHAR NOT NULL,
                data_date DATE NOT NULL,
                expiry DATE NOT NULL,
                strike DOUBLE NOT NULL,
                option_type VARCHAR NOT NULL,
                bid DOUBLE,
                ask DOUBLE,
                mid DOUBLE,
                last DOUBLE,
                volume INTEGER,
                open_interest INTEGER,
                iv DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                rho DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (contract_id, data_date)
            )
        """)
        
        # Earnings calendar table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                symbol VARCHAR NOT NULL,
                report_date DATE NOT NULL,
                fiscal_date_ending DATE,
                estimate DOUBLE,
                reported_eps DOUBLE,
                surprise DOUBLE,
                surprise_pct DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, report_date)
            )
        """)
        
        # Company overview cache table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS company_overview (
                symbol VARCHAR NOT NULL PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap DOUBLE,
                dividend_yield DOUBLE,
                ex_dividend_date DATE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low DOUBLE,
                beta DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for options queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_options_underlying_date 
            ON options_chains (underlying, data_date)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_options_expiry 
            ON options_chains (expiry)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_earnings_date 
            ON earnings_calendar (report_date)
        """)
    
    def _init_news_sentiment_schema(self) -> None:
        """Initialize news sentiment table."""
        # News sentiment table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY,
                time_published TIMESTAMP NOT NULL,
                title VARCHAR,
                summary VARCHAR,
                source VARCHAR,
                url VARCHAR,
                ticker VARCHAR,
                overall_sentiment_score DOUBLE,
                overall_sentiment_label VARCHAR,
                ticker_sentiment_score DOUBLE,
                ticker_sentiment_label VARCHAR,
                relevance_score DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (time_published, title, ticker)
            )
        """)
        
        # Create indexes for news queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_time 
            ON news_sentiment (time_published)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker 
            ON news_sentiment (ticker)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker_time 
            ON news_sentiment (ticker, time_published)
        """)
    
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
        
        # Prepare data
        data = df.copy()
        data = data.reset_index()
        data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        data["symbol"] = symbol
        data["timeframe"] = timeframe.value
        
        if replace:
            # Delete existing data for this symbol/timeframe
            self.conn.execute("""
                DELETE FROM ohlcv 
                WHERE symbol = ? AND timeframe = ?
            """, [symbol, timeframe.value])
        
        # Insert data (with conflict resolution)
        self.conn.execute("""
            INSERT OR REPLACE INTO ohlcv 
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume
            FROM data
        """)
        
        # Update metadata
        self._update_metadata(symbol, timeframe)
        
        logger.info(f"Saved {len(data)} rows for {symbol} {timeframe.value}")
        return len(data)
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
        
        df = self.conn.execute(query, params).fetchdf()
        
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        logger.debug(f"Loaded {len(df)} rows for {symbol} {timeframe.value}")
        return df
    
    def load_multi_timeframe(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[Timeframe, pd.DataFrame]:
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
        self.conn.execute("""
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
        """, [symbol, timeframe.value])
    
    def get_metadata(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[Timeframe] = None,
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
        
        return self.conn.execute(query, params).fetchdf()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols in the database."""
        result = self.conn.execute("""
            SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol
        """).fetchall()
        return [row[0] for row in result]
    
    def get_date_range(
        self,
        symbol: str,
        timeframe: Timeframe,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the date range for a symbol/timeframe.
        
        Returns:
            Tuple of (first_date, last_date) or (None, None) if no data
        """
        result = self.conn.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, [symbol, timeframe.value]).fetchone()
        
        if result and result[0]:
            return (result[0], result[1])
        return (None, None)
    
    def has_required_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
                Timeframe.W1:  5 / 7,      # 1 bar per 5 trading days
                Timeframe.D1:  5 / 7,      # 1 bar per trading day
                Timeframe.H4:  5 / 7 * 2,  # ~2 bars per trading day
                Timeframe.H1:  5 / 7 * 6.5,
                Timeframe.M30: 5 / 7 * 13,
                Timeframe.M15: 5 / 7 * 26,
                Timeframe.M5:  5 / 7 * 78,
                Timeframe.M1:  5 / 7 * 390,
                Timeframe.S5:  5 / 7 * 390 * 12,  # 12 × 5s bars per minute
            }
            expected_bars = max(1, int(days * _bars_per_day.get(timeframe, 5 / 7)))
            
            if expected_bars > 0:
                completeness = row_count / expected_bars
                if completeness < completeness_threshold:
                    return (False, f"Incomplete: {completeness:.1%} < {completeness_threshold:.1%}")
        
        return (True, f"Valid: {row_count} bars")
    
    def delete_symbol(self, symbol: str) -> int:
        """
        Delete all data for a symbol.
        
        Args:
            symbol: Symbol to delete
            
        Returns:
            Number of rows deleted
        """
        result = self.conn.execute("""
            DELETE FROM ohlcv WHERE symbol = ?
        """, [symbol])
        
        self.conn.execute("""
            DELETE FROM data_metadata WHERE symbol = ?
        """, [symbol])
        
        deleted = result.rowcount
        logger.info(f"Deleted {deleted} rows for {symbol}")
        return deleted
    
    def vacuum(self) -> None:
        """Optimize database storage."""
        self.conn.execute("VACUUM")
        logger.info("Database vacuumed")
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
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
            self.conn.execute("""
                DELETE FROM options_chains 
                WHERE underlying = ? AND data_date = ?
            """, [symbol, data_date.date() if hasattr(data_date, 'date') else data_date])
        
        # Select only columns that exist in the table
        valid_columns = [
            "contract_id", "underlying", "data_date", "expiry", "strike",
            "option_type", "bid", "ask", "mid", "last", "volume",
            "open_interest", "iv", "delta", "gamma", "theta", "vega", "rho"
        ]
        
        available_columns = [c for c in valid_columns if c in data.columns]
        insert_data = data[available_columns]
        
        # Insert data
        self.conn.execute(f"""
            INSERT OR REPLACE INTO options_chains 
            ({', '.join(available_columns)})
            SELECT {', '.join(available_columns)}
            FROM insert_data
        """)
        
        logger.info(f"Saved {len(data)} options contracts for {symbol} on {data_date}")
        return len(data)
    
    def load_options_chain(
        self,
        symbol: str,
        data_date: Optional[datetime] = None,
        expiry_min: Optional[datetime] = None,
        expiry_max: Optional[datetime] = None,
        option_type: Optional[str] = None,
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
            params.append(data_date.date() if hasattr(data_date, 'date') else data_date)
        else:
            # Get most recent date
            query += " AND data_date = (SELECT MAX(data_date) FROM options_chains WHERE underlying = ?)"
            params.append(symbol)
        
        if expiry_min:
            query += " AND expiry >= ?"
            params.append(expiry_min.date() if hasattr(expiry_min, 'date') else expiry_min)
        
        if expiry_max:
            query += " AND expiry <= ?"
            params.append(expiry_max.date() if hasattr(expiry_max, 'date') else expiry_max)
        
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
            "symbol", "report_date", "fiscal_date_ending", "estimate",
            "reported_eps", "surprise", "surprise_pct"
        ]
        
        available_columns = [c for c in valid_columns if c in data.columns]
        insert_data = data[available_columns]
        
        self.conn.execute(f"""
            INSERT OR REPLACE INTO earnings_calendar 
            ({', '.join(available_columns)})
            SELECT {', '.join(available_columns)}
            FROM insert_data
        """)
        
        logger.info(f"Saved {len(data)} earnings records")
        return len(data)
    
    def load_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
            params.append(start_date.date() if hasattr(start_date, 'date') else start_date)
        
        if end_date:
            query += " AND report_date <= ?"
            params.append(end_date.date() if hasattr(end_date, 'date') else end_date)
        
        query += " ORDER BY report_date"
        
        df = self.conn.execute(query, params).fetchdf()
        
        if not df.empty and "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
        
        return df
    
    def get_days_to_earnings(
        self,
        symbol: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[int]:
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
        
        result = self.conn.execute("""
            SELECT MIN(report_date) as next_earnings
            FROM earnings_calendar
            WHERE symbol = ? AND report_date >= ?
        """, [symbol, as_of_date.date()]).fetchone()
        
        if result and result[0]:
            next_date = pd.to_datetime(result[0])
            return (next_date - pd.Timestamp(as_of_date)).days
        
        return None
    
    def save_company_overview(
        self,
        data: Dict,
    ) -> None:
        """
        Save company overview data.
        
        Args:
            data: Dictionary from fetch_company_overview
        """
        if not data or "Symbol" not in data:
            return
        
        self.conn.execute("""
            INSERT OR REPLACE INTO company_overview
            (symbol, name, sector, industry, market_cap, dividend_yield,
             ex_dividend_date, fifty_two_week_high, fifty_two_week_low, beta, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
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
        ])
        
        logger.debug(f"Saved company overview for {data.get('Symbol')}")
    
    def load_company_overview(
        self,
        symbol: str,
    ) -> Optional[Dict]:
        """
        Load company overview data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company data or None
        """
        result = self.conn.execute("""
            SELECT * FROM company_overview WHERE symbol = ?
        """, [symbol]).fetchdf()
        
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
            "time_published", "title", "summary", "source", "url", "ticker",
            "overall_sentiment_score", "overall_sentiment_label",
            "ticker_sentiment_score", "ticker_sentiment_label", "relevance_score"
        ]
        
        available_columns = [c for c in valid_columns if c in data.columns]
        insert_data = data[available_columns].copy()
        
        # Handle duplicates by using INSERT OR IGNORE
        self.conn.execute(f"""
            INSERT OR IGNORE INTO news_sentiment 
            ({', '.join(available_columns)})
            SELECT {', '.join(available_columns)}
            FROM insert_data
        """)
        
        logger.info(f"Saved {len(data)} news sentiment records")
        return len(data)
    
    def load_news_sentiment(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tickers: Optional[List[str]] = None,
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
        tickers: Optional[List[str]] = None,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
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
        tickers: Optional[List[str]] = None,
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
        tickers: Optional[List[str]] = None,
        before_date: Optional[datetime] = None,
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

