"""
DuckDB storage for economic indicators.

Separate database from market data to keep economic indicators isolated.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd
from loguru import logger

from quantcore.config.settings import get_settings


class EconomicStorage:
    """Storage manager for economic indicators using DuckDB."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize economic storage.

        Args:
            db_path: Path to DuckDB file. If None, uses settings.
        """
        self.settings = get_settings()
        self.db_path = db_path or (
            self.settings.data_dir / "economic_indicators.duckdb"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database and schema
        self._initialize_database()

    def _initialize_database(self):
        """Create database schema for economic indicators."""
        with duckdb.connect(str(self.db_path)) as conn:
            # Create main indicators table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    indicator VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    value DOUBLE NOT NULL,
                    frequency VARCHAR NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (indicator, date)
                )
            """
            )

            # Create index for efficient queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indicator_date
                ON economic_indicators (indicator, date)
            """
            )

            # Create metadata table
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

            logger.info("Initialized economic indicators database at {}", self.db_path)

    def store_indicator(self, indicator_name: str, df: pd.DataFrame):
        """Store or update indicator data.

        Args:
            indicator_name: Name of the indicator
            df: DataFrame with columns: date, value, frequency
        """
        if df.empty:
            logger.warning("Empty DataFrame for indicator {}", indicator_name)
            return

        # Ensure required columns
        required_cols = ["date", "value", "frequency"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must have columns: {required_cols}, got {df.columns.tolist()}"
            )

        # Add indicator column if not present
        if "indicator" not in df.columns:
            df = df.copy()
            df["indicator"] = indicator_name

        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])

        with duckdb.connect(str(self.db_path)) as conn:
            # Delete existing data for this indicator
            conn.execute(
                "DELETE FROM economic_indicators WHERE indicator = ?",
                [indicator_name],
            )

            # Insert new data
            conn.execute(
                """
                INSERT INTO economic_indicators (indicator, date, value, frequency)
                SELECT indicator, date, value, frequency
                FROM df
            """
            )

            # Update metadata
            self._update_metadata(conn, indicator_name)

            logger.info(
                "Stored {} records for indicator {}",
                len(df),
                indicator_name,
            )

    def store_all_indicators(self, indicators_data: Dict[str, pd.DataFrame]):
        """Store multiple indicators at once.

        Args:
            indicators_data: Dict mapping indicator name to DataFrame
        """
        for indicator_name, df in indicators_data.items():
            self.store_indicator(indicator_name, df)

    def _update_metadata(self, conn: duckdb.DuckDBPyConnection, indicator: str):
        """Update metadata for an indicator.

        Args:
            conn: Active database connection
            indicator: Indicator name
        """
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
                INSERT OR REPLACE INTO indicator_metadata
                (indicator, frequency, first_date, last_date, record_count, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                [indicator, frequency, first_date, last_date, record_count],
            )

    def get_indicator(
        self,
        indicator: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve indicator data.

        Args:
            indicator: Indicator name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with date and value columns
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = "SELECT date, value FROM economic_indicators WHERE indicator = ?"
            params = [indicator]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = conn.execute(query, params).df()

            if df.empty:
                logger.warning(
                    "No data found for indicator {} in date range", indicator
                )

            return df

    def get_all_indicators(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve all indicators as wide-format DataFrame.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with date index and indicator columns
        """
        with duckdb.connect(str(self.db_path)) as conn:
            query = "SELECT indicator, date, value FROM economic_indicators WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = conn.execute(query, params).df()

            if df.empty:
                return pd.DataFrame()

            # Pivot to wide format
            wide_df = df.pivot(index="date", columns="indicator", values="value")
            return wide_df

    def list_indicators(self) -> pd.DataFrame:
        """List all available indicators with metadata.

        Returns:
            DataFrame with indicator metadata
        """
        with duckdb.connect(str(self.db_path)) as conn:
            df = conn.execute(
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
            ).df()

            return df

    def get_latest_values(self) -> pd.Series:
        """Get latest value for each indicator.

        Returns:
            Series mapping indicator name to latest value
        """
        with duckdb.connect(str(self.db_path)) as conn:
            df = conn.execute(
                """
                SELECT
                    indicator,
                    value
                FROM economic_indicators
                WHERE (indicator, date) IN (
                    SELECT indicator, MAX(date)
                    FROM economic_indicators
                    GROUP BY indicator
                )
            """
            ).df()

            if df.empty:
                return pd.Series(dtype=float)

            return df.set_index("indicator")["value"]

    def delete_indicator(self, indicator: str):
        """Delete all data for an indicator.

        Args:
            indicator: Indicator name
        """
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(
                "DELETE FROM economic_indicators WHERE indicator = ?", [indicator]
            )
            conn.execute(
                "DELETE FROM indicator_metadata WHERE indicator = ?", [indicator]
            )

            logger.info("Deleted indicator {}", indicator)

    def vacuum(self):
        """Optimize database by reclaiming space."""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("VACUUM")
            logger.info("Vacuumed economic indicators database")
