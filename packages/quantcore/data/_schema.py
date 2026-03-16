"""Schema initialization mixin for DataStore.

Provides DDL for all tables: OHLCV, intraday, options, news sentiment.
"""

import duckdb
from loguru import logger


class SchemaMixin:
    """Mixin that owns all CREATE TABLE / CREATE INDEX DDL."""

    def _init_schema(self) -> None:
        """Initialize database schema using a short-lived connection."""
        conn = self._open_connection()
        try:
            self._run_schema_ddl(conn)
        finally:
            conn.close()

    def _run_schema_ddl(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Execute all CREATE TABLE/INDEX DDL statements.

        Temporarily sets self._conn so sub-methods that reference self.conn
        work with the same short-lived connection, then clears it.
        """
        old_conn = self._conn
        self._conn = conn
        try:
            # Main OHLCV table
            conn.execute("""
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

            conn.execute("""
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

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
                ON ohlcv (symbol, timeframe, timestamp)
            """)

            # Initialize sub-schemas (these use self.conn which is now the passed conn)
            self._init_options_schema()
            self._init_news_sentiment_schema()
            self._init_intraday_schema()
            self._init_fundamentals_schema()
        finally:
            self._conn = old_conn

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
