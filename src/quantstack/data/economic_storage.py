"""PostgreSQL storage for economic indicators."""

from datetime import datetime

import pandas as pd
from loguru import logger

from quantstack.db import PgConnection, pg_conn


class EconomicStorage:
    """Storage manager for economic indicators."""

    def __init__(self):
        self._initialize_database()

    def _initialize_database(self):
        """Create database schema for economic indicators."""
        with pg_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    indicator VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    frequency VARCHAR NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (indicator, date)
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indicator_date
                ON economic_indicators (indicator, date)
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indicator_metadata (
                    indicator VARCHAR PRIMARY KEY,
                    description VARCHAR,
                    source VARCHAR DEFAULT 'Alpha Vantage',
                    frequency VARCHAR NOT NULL,
                    first_date DATE,
                    last_date DATE,
                    record_count INTEGER,
                    last_updated TIMESTAMP
                )
            """
            )

    def store_indicator(self, indicator_name: str, df: pd.DataFrame):
        """Store or update indicator data."""
        if df.empty:
            logger.warning("Empty DataFrame for indicator {}", indicator_name)
            return

        required_cols = ["date", "value", "frequency"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must have columns: {required_cols}, got {df.columns.tolist()}"
            )

        if "indicator" not in df.columns:
            df = df.copy()
            df["indicator"] = indicator_name

        df["date"] = pd.to_datetime(df["date"])

        with pg_conn() as conn:
            conn.execute(
                "DELETE FROM economic_indicators WHERE indicator = %s",
                [indicator_name],
            )

            rows = [
                (indicator_name, row["date"].date(), float(row["value"]), row["frequency"])
                for _, row in df.iterrows()
            ]
            conn.executemany(
                "INSERT INTO economic_indicators (indicator, date, value, frequency) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._update_metadata(conn, indicator_name)

        logger.info("Stored {} records for indicator {}", len(df), indicator_name)

    def store_all_indicators(self, indicators_data: dict[str, pd.DataFrame]):
        """Store multiple indicators at once."""
        for indicator_name, df in indicators_data.items():
            self.store_indicator(indicator_name, df)

    def _update_metadata(self, conn: PgConnection, indicator: str):
        """Update metadata for an indicator."""
        result = conn.execute(
            """
            SELECT
                frequency,
                MIN(date) as first_date,
                MAX(date) as last_date,
                COUNT(*) as record_count
            FROM economic_indicators
            WHERE indicator = ?
            GROUP BY indicator, frequency
        """,
            [indicator],
        ).fetchone()

        if result:
            frequency, first_date, last_date, record_count = result
            conn.execute(
                """
                INSERT INTO indicator_metadata
                    (indicator, frequency, first_date, last_date, record_count, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (indicator) DO UPDATE SET
                    frequency = EXCLUDED.frequency,
                    first_date = EXCLUDED.first_date,
                    last_date = EXCLUDED.last_date,
                    record_count = EXCLUDED.record_count,
                    last_updated = EXCLUDED.last_updated
            """,
                [indicator, frequency, first_date, last_date, record_count],
            )

    def get_indicator(
        self,
        indicator: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve indicator data."""
        with pg_conn() as conn:
            query = "SELECT date, value FROM economic_indicators WHERE indicator = ?"
            params = [indicator]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"
            return conn.execute(query, params).fetchdf()

    def get_all_indicators(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve all indicators as wide-format DataFrame."""
        with pg_conn() as conn:
            query = "SELECT indicator, date, value FROM economic_indicators WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"
            df = conn.execute(query, params).fetchdf()

            if df.empty:
                return pd.DataFrame()

            return df.pivot(index="date", columns="indicator", values="value")

    def list_indicators(self) -> pd.DataFrame:
        """List all available indicators with metadata."""
        with pg_conn() as conn:
            return conn.execute(
                """
                SELECT
                    indicator,
                    frequency,
                    first_date,
                    last_date,
                    record_count,
                    last_updated
                FROM indicator_metadata
                ORDER BY indicator
            """
            ).fetchdf()

    def get_latest_values(self) -> pd.Series:
        """Get latest value for each indicator."""
        with pg_conn() as conn:
            df = conn.execute(
                """
                SELECT indicator, value
                FROM economic_indicators ei
                WHERE date = (
                    SELECT MAX(date)
                    FROM economic_indicators ei2
                    WHERE ei2.indicator = ei.indicator
                )
            """
            ).fetchdf()

            if df.empty:
                return pd.Series(dtype=float)

            return df.set_index("indicator")["value"]

    def delete_indicator(self, indicator: str):
        """Delete all data for an indicator."""
        with pg_conn() as conn:
            conn.execute(
                "DELETE FROM economic_indicators WHERE indicator = ?", [indicator]
            )
            conn.execute(
                "DELETE FROM indicator_metadata WHERE indicator = ?", [indicator]
            )
