# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PostgreSQL-backed market data store.

Provides the same public interface as DataStore (save_* / load_* / list_* /
get_* methods) but uses pg_conn() for all reads and writes.  Multiple MCP
server instances can call these methods concurrently without contention.

Design notes
------------
- No persistent connection — each method opens a pg_conn() context manager,
  executes, and returns.  This matches the pooled pg_conn() pattern used
  everywhere in the operational layer.
- Bulk upserts use psycopg2.extras.execute_values with ON CONFLICT DO UPDATE
  so repeated ingestion runs are idempotent.
- pd.read_sql_query() is used for loads because it avoids the psycopg2 cursor
  fetchall + column-name reconstruction dance.  It requires the raw psycopg2
  connection (conn._raw), not the PgConnection wrapper.
- psycopg2 uses %s placeholders.  The _translate() helper on PgConnection
  converts ? → %s, but we write %s directly here since we own the SQL.
"""

from __future__ import annotations

import json as _json
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg2.extras
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.db import pg_conn


def _safe_float(value: object) -> float | None:
    """Convert AV string values (e.g. "None", "N/A", "-") to float or None."""
    if value is None or value in ("None", "N/A", "-", ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _safe_str(value: object) -> str | None:
    """Convert AV string values (e.g. "None", "N/A") to str or None."""
    if value is None or value in ("None", "N/A", "-", ""):
        return None
    return str(value)


class PgDataStore:
    """PostgreSQL-backed replacement for DataStore.

    All save_* methods return the number of rows written.
    All load_* methods return a pandas DataFrame (or dict for overview).
    close() is a no-op — there is no persistent connection to release.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op.  Connections are released automatically by pg_conn()."""

    def __enter__(self) -> "PgDataStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prime_cursor(conn: Any) -> Any:
        """Ensure *conn* has an open cursor and return it.

        psycopg2.extras.execute_values requires a real psycopg2 cursor.
        PgConnection creates _cur lazily on the first execute() call.  This
        helper primes both the raw connection and the cursor without running
        a round-trip query.
        """
        raw = conn._ensure_raw()
        if conn._cur is None or conn._cur.closed:
            conn._cur = raw.cursor()
        return conn._cur

    @staticmethod
    def _df_from_query(
        query: str, params: list[Any], conn: Any
    ) -> pd.DataFrame:
        """Execute *query* and return a DataFrame.

        Accepts a PgConnection rather than conn._raw so that it can prime
        the underlying psycopg2 connection (conn._ensure_raw()) before
        calling pd.read_sql_query, which requires the raw connection object.
        """
        raw = conn._ensure_raw()
        return pd.read_sql_query(query, raw, params=params)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        replace: bool = False,
    ) -> int:
        """Upsert OHLCV bars for *symbol* / *timeframe*."""
        if df.empty:
            logger.warning(f"[PgDataStore] Empty DataFrame for {symbol} {timeframe.value}")
            return 0

        data = df.copy()
        # Keep only OHLCV — drop provider extras (vwap, trade_count, …)
        core_cols = {"open", "high", "low", "close", "volume"}
        extra = [c for c in data.columns if c.lower() not in core_cols]
        if extra:
            data = data.drop(columns=extra, errors="ignore")

        data = data.reset_index()
        data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        data["symbol"] = symbol
        data["timeframe"] = timeframe.value

        rows = list(
            data[["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]]
            .itertuples(index=False, name=None)
        )

        with pg_conn() as conn:
            if replace:
                conn.execute(
                    "DELETE FROM ohlcv WHERE symbol=%s AND timeframe=%s",
                    [symbol, timeframe.value],
                )

            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                """
                INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                    close=EXCLUDED.close, volume=EXCLUDED.volume
                """,
                rows,
            )

            # Refresh metadata in the same transaction
            conn.execute(
                """
                INSERT INTO data_metadata
                    (symbol, timeframe, first_timestamp, last_timestamp, row_count, updated_at)
                SELECT %s, %s, MIN(timestamp), MAX(timestamp), COUNT(*), NOW()
                FROM ohlcv
                WHERE symbol=%s AND timeframe=%s
                ON CONFLICT (symbol, timeframe) DO UPDATE SET
                    first_timestamp=EXCLUDED.first_timestamp,
                    last_timestamp=EXCLUDED.last_timestamp,
                    row_count=EXCLUDED.row_count,
                    updated_at=EXCLUDED.updated_at
                """,
                [symbol, timeframe.value, symbol, timeframe.value],
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} rows for {symbol} {timeframe.value}")
        return len(rows)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars; returns DataFrame with DatetimeIndex."""
        query = (
            "SELECT timestamp, open, high, low, close, volume "
            "FROM ohlcv WHERE symbol=%s AND timeframe=%s"
        )
        params: list[Any] = [symbol, timeframe.value]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with pg_conn() as conn:
            df = self._df_from_query(query, params, conn)

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    # ------------------------------------------------------------------
    # OHLCV 1-minute
    # ------------------------------------------------------------------

    def save_ohlcv_1m(
        self,
        df: pd.DataFrame,
        symbol: str,
        replace: bool = False,
    ) -> int:
        """Upsert 1-minute bars for *symbol*."""
        if df.empty:
            logger.warning(f"[PgDataStore] Empty 1m DataFrame for {symbol}")
            return 0

        data = df.copy().reset_index()
        data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)
        data["symbol"] = symbol

        if "vwap" not in data.columns:
            data["vwap"] = None
        if "trade_count" not in data.columns:
            data["trade_count"] = None

        rows = list(
            data[["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]
            .itertuples(index=False, name=None)
        )

        with pg_conn() as conn:
            if replace:
                conn.execute("DELETE FROM ohlcv_1m WHERE symbol=%s", [symbol])

            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                """
                INSERT INTO ohlcv_1m
                    (symbol, timestamp, open, high, low, close, volume, vwap, trade_count)
                VALUES %s
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                    close=EXCLUDED.close, volume=EXCLUDED.volume,
                    vwap=EXCLUDED.vwap, trade_count=EXCLUDED.trade_count
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} 1m bars for {symbol}")
        return len(rows)

    def load_ohlcv_1m(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load 1-minute bars; returns DataFrame with DatetimeIndex."""
        query = (
            "SELECT timestamp, open, high, low, close, volume, vwap, trade_count "
            "FROM ohlcv_1m WHERE symbol=%s"
        )
        params: list[Any] = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with pg_conn() as conn:
            df = self._df_from_query(query, params, conn)

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

        for col in ("vwap", "trade_count"):
            if col in df.columns and df[col].isna().all():
                df.drop(columns=[col], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    def save_financial_statements(self, df: pd.DataFrame) -> int:
        """Upsert financial statement rows."""
        if df.empty:
            return 0

        key_cols = {
            "ticker", "statement_type", "period_type", "report_period",
            "revenue", "net_income", "total_assets", "total_debt",
            "operating_income", "gross_profit", "eps_diluted",
        }
        data = df.copy()

        extra_cols = [c for c in data.columns if c not in key_cols and c != "data"]
        if extra_cols:
            data["data"] = data[extra_cols].apply(
                lambda row: _json.dumps(
                    {k: v for k, v in row.to_dict().items() if pd.notna(v)},
                    default=str,
                ),
                axis=1,
            )

        for col in key_cols:
            if col not in data.columns:
                data[col] = None

        insert_cols = sorted(key_cols) + (["data"] if "data" in data.columns else [])
        insert_df = data[insert_cols]

        rows = [
            tuple(
                _json.dumps(v, default=str) if isinstance(v, dict) else v
                for v in row
            )
            for row in insert_df.itertuples(index=False, name=None)
        ]

        col_list = ", ".join(insert_cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in insert_cols
            if c not in ("ticker", "report_period", "statement_type", "period_type")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO financial_statements ({col_list})
                VALUES %s
                ON CONFLICT (ticker, report_period, statement_type, period_type)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} financial statement rows")
        return len(rows)

    def load_financial_statements(
        self,
        ticker: str,
        statement_type: str | None = None,
        period_type: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Load financial statements for a ticker."""
        query = "SELECT * FROM financial_statements WHERE ticker=%s"
        params: list[Any] = [ticker]

        if statement_type:
            query += " AND statement_type=%s"
            params.append(statement_type)
        if period_type:
            query += " AND period_type=%s"
            params.append(period_type)

        query += " ORDER BY report_period DESC LIMIT %s"
        params.append(limit)

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    # ------------------------------------------------------------------
    # Financial metrics
    # ------------------------------------------------------------------

    def save_financial_metrics(self, df: pd.DataFrame) -> int:
        """Upsert financial metrics rows."""
        if df.empty:
            return 0

        key_cols = {
            "ticker", "date", "period_type",
            "market_cap", "pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda",
            "roe", "roa", "gross_margin", "operating_margin", "net_margin",
            "debt_to_equity", "current_ratio", "dividend_yield",
            "revenue_growth", "earnings_growth",
        }
        data = df.copy()

        extra_cols = [c for c in data.columns if c not in key_cols and c != "data"]
        if extra_cols:
            data["data"] = data[extra_cols].apply(
                lambda row: _json.dumps(
                    {k: v for k, v in row.to_dict().items() if pd.notna(v)},
                    default=str,
                ),
                axis=1,
            )

        for col in key_cols:
            if col not in data.columns:
                data[col] = None

        insert_cols = sorted(key_cols) + (["data"] if "data" in data.columns else [])
        insert_df = data[insert_cols]
        rows = list(insert_df.itertuples(index=False, name=None))

        col_list = ", ".join(insert_cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in insert_cols
            if c not in ("ticker", "date", "period_type")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO financial_metrics ({col_list})
                VALUES %s
                ON CONFLICT (ticker, date, period_type)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} financial metrics rows")
        return len(rows)

    def load_financial_metrics(
        self,
        ticker: str,
        period_type: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Load financial metrics for a ticker."""
        query = "SELECT * FROM financial_metrics WHERE ticker=%s"
        params: list[Any] = [ticker]

        if period_type:
            query += " AND period_type=%s"
            params.append(period_type)

        query += " ORDER BY date DESC LIMIT %s"
        params.append(limit)

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    # ------------------------------------------------------------------
    # Insider trades
    # ------------------------------------------------------------------

    def save_insider_trades(self, df: pd.DataFrame) -> int:
        """Upsert insider trade rows."""
        if df.empty:
            return 0

        cols = [
            "ticker", "transaction_date", "owner_name", "owner_title",
            "transaction_type", "shares", "price_per_share", "total_value",
            "shares_owned_after", "filing_date",
        ]
        data = df.copy()
        for c in cols:
            if c not in data.columns:
                data[c] = None

        data = data.dropna(subset=["transaction_date", "shares"])
        data["owner_name"] = data["owner_name"].fillna("Unknown")
        if data.empty:
            return 0

        rows = list(data[cols].itertuples(index=False, name=None))
        col_list = ", ".join(cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in cols
            if c not in ("ticker", "transaction_date", "owner_name", "shares")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO insider_trades ({col_list})
                VALUES %s
                ON CONFLICT (ticker, transaction_date, owner_name, shares)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} insider trade rows")
        return len(rows)

    def load_insider_trades(
        self,
        ticker: str,
        start_date: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Load insider trades for a ticker."""
        query = "SELECT * FROM insider_trades WHERE ticker=%s"
        params: list[Any] = [ticker]

        if start_date:
            query += " AND transaction_date >= %s"
            params.append(start_date.date() if hasattr(start_date, "date") else start_date)

        query += " ORDER BY transaction_date DESC LIMIT %s"
        params.append(limit)

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    # ------------------------------------------------------------------
    # Institutional ownership
    # ------------------------------------------------------------------

    def save_institutional_ownership(self, df: pd.DataFrame) -> int:
        """Upsert institutional ownership rows."""
        if df.empty:
            return 0

        cols = [
            "ticker", "investor_name", "report_date",
            "shares_held", "market_value", "portfolio_pct",
            "change_shares", "change_pct",
        ]
        data = df.copy()
        for c in cols:
            if c not in data.columns:
                data[c] = None

        data = data.dropna(subset=["report_date"])
        data["investor_name"] = data["investor_name"].fillna("Unknown")
        if data.empty:
            return 0

        rows = list(data[cols].itertuples(index=False, name=None))
        col_list = ", ".join(cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in cols
            if c not in ("ticker", "investor_name", "report_date")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO institutional_ownership ({col_list})
                VALUES %s
                ON CONFLICT (ticker, investor_name, report_date)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} institutional ownership rows")
        return len(rows)

    def load_institutional_ownership(
        self,
        ticker: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Load institutional ownership for a ticker."""
        query = (
            "SELECT * FROM institutional_ownership "
            "WHERE ticker=%s ORDER BY report_date DESC LIMIT %s"
        )
        with pg_conn() as conn:
            return self._df_from_query(query, [ticker, limit], conn)

    # ------------------------------------------------------------------
    # Macro indicators
    # ------------------------------------------------------------------

    def save_macro_indicators(self, indicator: str, df: pd.DataFrame) -> int:
        """Upsert macro time-series rows.

        Args:
            indicator: Indicator name (e.g. 'CPI', 'REAL_GDP').
            df: DataFrame with DatetimeIndex and a 'value' column.
        """
        if df.empty:
            return 0

        data = df.reset_index()
        data.columns = ["date", "value"]
        data["indicator"] = indicator

        rows = list(data[["indicator", "date", "value"]].itertuples(index=False, name=None))

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                """
                INSERT INTO macro_indicators (indicator, date, value)
                VALUES %s
                ON CONFLICT (indicator, date) DO UPDATE SET value=EXCLUDED.value
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} {indicator} rows")
        return len(rows)

    def load_macro_indicator(
        self,
        indicator: str,
        start_date: str | None = None,
    ) -> pd.DataFrame:
        """Load a macro indicator series."""
        query = "SELECT date, value FROM macro_indicators WHERE indicator=%s"
        params: list[Any] = [indicator]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)

        query += " ORDER BY date"

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    def list_macro_indicators(self) -> list[str]:
        """Return the distinct indicator names stored."""
        with pg_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT indicator FROM macro_indicators ORDER BY indicator"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Corporate actions
    # ------------------------------------------------------------------

    def save_corporate_actions(
        self,
        ticker: str,
        df: pd.DataFrame,
        action_type: str,
    ) -> int:
        """Upsert dividend or split records."""
        if df.empty:
            return 0

        data = df.copy()
        data["ticker"] = ticker
        data["action_type"] = action_type

        rename = {
            "ex_dividend_date": "effective_date",
            "split_factor": "amount",
        }
        data.rename(columns={k: v for k, v in rename.items() if k in data.columns}, inplace=True)

        for col in ("declaration_date", "record_date", "payment_date"):
            if col not in data.columns:
                data[col] = None

        cols = [
            "ticker", "action_type", "effective_date", "amount",
            "declaration_date", "record_date", "payment_date",
        ]
        # Replace pandas NaT/NaN with None so psycopg2 receives NULL.
        # Must convert to object dtype first — datetime64 columns keep NaT even after .where().
        rows = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in data[cols].itertuples(index=False, name=None)
        ]
        col_list = ", ".join(cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in cols
            if c not in ("ticker", "action_type", "effective_date")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO corporate_actions ({col_list})
                VALUES %s
                ON CONFLICT (ticker, action_type, effective_date)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} {action_type} records for {ticker}")
        return len(rows)

    def load_corporate_actions(
        self,
        ticker: str,
        action_type: str | None = None,
    ) -> pd.DataFrame:
        """Load corporate actions for a ticker."""
        query = "SELECT * FROM corporate_actions WHERE ticker=%s"
        params: list[Any] = [ticker]

        if action_type:
            query += " AND action_type=%s"
            params.append(action_type)

        query += " ORDER BY effective_date"

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    # ------------------------------------------------------------------
    # Analyst estimates
    # ------------------------------------------------------------------

    def save_analyst_estimates(self, df: pd.DataFrame) -> int:
        """Upsert analyst estimate rows."""
        if df.empty:
            return 0

        cols = [
            "ticker", "fiscal_date", "period_type", "metric",
            "consensus", "high", "low", "num_analysts",
        ]
        data = df.copy()
        for c in cols:
            if c not in data.columns:
                data[c] = None

        rows = list(data[cols].itertuples(index=False, name=None))
        col_list = ", ".join(cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in cols
            if c not in ("ticker", "fiscal_date", "period_type", "metric")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO analyst_estimates ({col_list})
                VALUES %s
                ON CONFLICT (ticker, fiscal_date, period_type, metric)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} analyst estimate rows")
        return len(rows)

    def load_analyst_estimates(self, ticker: str) -> pd.DataFrame:
        """Load analyst estimates for a ticker."""
        query = (
            "SELECT * FROM analyst_estimates "
            "WHERE ticker=%s ORDER BY fiscal_date DESC"
        )
        with pg_conn() as conn:
            return self._df_from_query(query, [ticker], conn)

    # ------------------------------------------------------------------
    # SEC filings
    # ------------------------------------------------------------------

    def save_sec_filings(self, df: pd.DataFrame) -> int:
        """Upsert SEC filing metadata rows."""
        if df.empty:
            return 0

        cols = [
            "ticker", "accession_number", "filing_type",
            "filed_date", "period_of_report", "url",
        ]
        data = df.copy()
        for c in cols:
            if c not in data.columns:
                data[c] = None

        rows = list(data[cols].itertuples(index=False, name=None))
        col_list = ", ".join(cols)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in cols
            if c not in ("ticker", "accession_number")
        )

        with pg_conn() as conn:
            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO sec_filings ({col_list})
                VALUES %s
                ON CONFLICT (ticker, accession_number)
                DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} SEC filing rows")
        return len(rows)

    def load_sec_filings(
        self,
        ticker: str,
        filing_type: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Load SEC filings for a ticker."""
        query = "SELECT * FROM sec_filings WHERE ticker=%s"
        params: list[Any] = [ticker]

        if filing_type:
            query += " AND filing_type=%s"
            params.append(filing_type)

        query += " ORDER BY filed_date DESC LIMIT %s"
        params.append(limit)

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    # ------------------------------------------------------------------
    # Options chains
    # ------------------------------------------------------------------

    def save_options_chain(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_date: datetime,
        replace: bool = True,
    ) -> int:
        """Upsert options chain rows for *symbol* / *data_date*."""
        if df.empty:
            logger.warning(f"[PgDataStore] Empty options DataFrame for {symbol}")
            return 0

        data = df.copy()
        if "underlying" not in data.columns:
            data["underlying"] = symbol
        if "data_date" not in data.columns:
            data["data_date"] = data_date

        if isinstance(data["data_date"].iloc[0], pd.Timestamp):
            data["data_date"] = data["data_date"].dt.date

        valid_columns = [
            "contract_id", "underlying", "data_date", "expiry", "strike",
            "option_type", "bid", "ask", "mid", "last", "volume",
            "open_interest", "iv", "delta", "gamma", "theta", "vega", "rho",
        ]
        available_columns = [c for c in valid_columns if c in data.columns]
        insert_df = data[available_columns]
        rows = list(insert_df.itertuples(index=False, name=None))

        date_val = data_date.date() if hasattr(data_date, "date") else data_date

        col_list = ", ".join(available_columns)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in available_columns
            if c not in ("contract_id", "data_date")
        )

        with pg_conn() as conn:
            if replace:
                conn.execute(
                    "DELETE FROM options_chains WHERE underlying=%s AND data_date=%s",
                    [symbol, date_val],
                )

            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO options_chains ({col_list})
                VALUES %s
                ON CONFLICT (contract_id, data_date) DO UPDATE SET {update_list}
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} options contracts for {symbol} on {data_date}")
        return len(rows)

    def load_options_chain(
        self,
        symbol: str,
        data_date: datetime | None = None,
        expiry_min: datetime | None = None,
        expiry_max: datetime | None = None,
        option_type: str | None = None,
    ) -> pd.DataFrame:
        """Load options chain data."""
        query = "SELECT * FROM options_chains WHERE underlying=%s"
        params: list[Any] = [symbol]

        if data_date:
            query += " AND data_date=%s"
            params.append(data_date.date() if hasattr(data_date, "date") else data_date)
        else:
            query += (
                " AND data_date=(SELECT MAX(data_date) "
                "FROM options_chains WHERE underlying=%s)"
            )
            params.append(symbol)

        if expiry_min:
            query += " AND expiry >= %s"
            params.append(expiry_min.date() if hasattr(expiry_min, "date") else expiry_min)
        if expiry_max:
            query += " AND expiry <= %s"
            params.append(expiry_max.date() if hasattr(expiry_max, "date") else expiry_max)
        if option_type:
            query += " AND LOWER(option_type)=%s"
            params.append(option_type.lower())

        query += " ORDER BY expiry, strike"

        with pg_conn() as conn:
            df = self._df_from_query(query, params, conn)

        if df.empty:
            return pd.DataFrame()

        for col in ("expiry", "data_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        return df

    # ------------------------------------------------------------------
    # Earnings calendar
    # ------------------------------------------------------------------

    def save_earnings_calendar(
        self,
        df: pd.DataFrame,
        replace: bool = False,
    ) -> int:
        """Upsert earnings calendar rows."""
        if df.empty:
            return 0

        data = df.copy()
        if "report_date" not in data.columns and "reportdate" in data.columns:
            data = data.rename(columns={"reportdate": "report_date"})

        valid_columns = [
            "symbol", "report_date", "fiscal_date_ending",
            "estimate", "reported_eps", "surprise", "surprise_pct",
        ]
        available_columns = [c for c in valid_columns if c in data.columns]
        rows = list(data[available_columns].itertuples(index=False, name=None))

        col_list = ", ".join(available_columns)
        update_list = ", ".join(
            f"{c}=EXCLUDED.{c}"
            for c in available_columns
            if c not in ("symbol", "report_date")
        )

        with pg_conn() as conn:
            if replace:
                conn.execute("DELETE FROM earnings_calendar")

            if update_list:
                conflict_clause = f"ON CONFLICT (symbol, report_date) DO UPDATE SET {update_list}"
            else:
                conflict_clause = "ON CONFLICT (symbol, report_date) DO NOTHING"

            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"INSERT INTO earnings_calendar ({col_list}) VALUES %s {conflict_clause}",
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} earnings records")
        return len(rows)

    # save_earnings_history is an alias used by the acquisition pipeline
    save_earnings_history = save_earnings_calendar

    def load_earnings_history(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load earnings calendar data."""
        return self.load_earnings_calendar(
            symbol=symbol, start_date=start_date, end_date=end_date
        )

    def load_earnings_calendar(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load earnings calendar data."""
        query = "SELECT * FROM earnings_calendar WHERE 1=1"
        params: list[Any] = []

        if symbol:
            query += " AND symbol=%s"
            params.append(symbol)
        if start_date:
            query += " AND report_date >= %s"
            params.append(start_date.date() if hasattr(start_date, "date") else start_date)
        if end_date:
            query += " AND report_date <= %s"
            params.append(end_date.date() if hasattr(end_date, "date") else end_date)

        query += " ORDER BY report_date"

        with pg_conn() as conn:
            df = self._df_from_query(query, params, conn)

        if not df.empty and "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])

        return df

    # ------------------------------------------------------------------
    # Company overview / fundamentals
    # ------------------------------------------------------------------

    def save_company_overview(self, data: dict) -> None:
        """Upsert company overview data."""
        if not data or "Symbol" not in data:
            return

        with pg_conn() as conn:
            conn.execute(
                """
                INSERT INTO company_overview
                    (symbol, name, sector, industry, market_cap, dividend_yield,
                     ex_dividend_date, fifty_two_week_high, fifty_two_week_low,
                     beta, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    name=EXCLUDED.name, sector=EXCLUDED.sector,
                    industry=EXCLUDED.industry, market_cap=EXCLUDED.market_cap,
                    dividend_yield=EXCLUDED.dividend_yield,
                    ex_dividend_date=EXCLUDED.ex_dividend_date,
                    fifty_two_week_high=EXCLUDED.fifty_two_week_high,
                    fifty_two_week_low=EXCLUDED.fifty_two_week_low,
                    beta=EXCLUDED.beta, updated_at=NOW()
                """,
                [
                    _safe_str(data.get("Symbol")),
                    _safe_str(data.get("Name")),
                    _safe_str(data.get("Sector")),
                    _safe_str(data.get("Industry")),
                    _safe_float(data.get("MarketCapitalization")),
                    _safe_float(data.get("DividendYield")),
                    _safe_str(data.get("ExDividendDate")),
                    _safe_float(data.get("52WeekHigh")),
                    _safe_float(data.get("52WeekLow")),
                    _safe_float(data.get("Beta")),
                ],
            )

        logger.debug(f"[PgDataStore] Saved company overview for {data.get('Symbol')}")

    # save_fundamentals is the alias used by the acquisition pipeline
    def save_fundamentals(self, data: dict) -> None:
        """Alias for save_company_overview (legacy pipeline compat)."""
        self.save_company_overview(data)

    def load_company_overview(self, symbol: str) -> dict | None:
        """Load company overview data."""
        query = "SELECT * FROM company_overview WHERE symbol=%s"
        with pg_conn() as conn:
            df = self._df_from_query(query, [symbol], conn)

        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def load_fundamentals(self, symbol: str) -> dict | None:
        """Alias for load_company_overview (legacy pipeline compat)."""
        return self.load_company_overview(symbol)

    # ------------------------------------------------------------------
    # News sentiment
    # ------------------------------------------------------------------

    def save_news_sentiment(
        self,
        df: pd.DataFrame,
        replace: bool = False,
    ) -> int:
        """Upsert news sentiment rows."""
        if df.empty:
            logger.warning("[PgDataStore] Empty news sentiment DataFrame")
            return 0

        data = df.copy()

        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            if "index" in data.columns:
                data = data.rename(columns={"index": "time_published"})

        if "time_published" not in data.columns:
            logger.error("[PgDataStore] No time_published column in news data")
            return 0

        if not pd.api.types.is_datetime64_any_dtype(data["time_published"]):
            data["time_published"] = pd.to_datetime(data["time_published"])

        valid_columns = [
            "time_published", "title", "summary", "source", "url", "ticker",
            "overall_sentiment_score", "overall_sentiment_label",
            "ticker_sentiment_score", "ticker_sentiment_label", "relevance_score",
        ]
        available_columns = [c for c in valid_columns if c in data.columns]
        rows = list(data[available_columns].itertuples(index=False, name=None))

        col_list = ", ".join(available_columns)

        with pg_conn() as conn:
            if replace:
                conn.execute("DELETE FROM news_sentiment")

            psycopg2.extras.execute_values(
                self._prime_cursor(conn),
                f"""
                INSERT INTO news_sentiment ({col_list})
                VALUES %s
                ON CONFLICT (time_published, title, ticker) DO NOTHING
                """,
                rows,
            )

        logger.info(f"[PgDataStore] Saved {len(rows)} news sentiment records")
        return len(rows)

    # save_news is an alias used by some collectors
    save_news = save_news_sentiment

    def load_news_sentiment(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load news sentiment data."""
        query = (
            "SELECT time_published, title, summary, source, url, ticker, "
            "overall_sentiment_score, overall_sentiment_label, "
            "ticker_sentiment_score, ticker_sentiment_label, relevance_score "
            "FROM news_sentiment WHERE 1=1"
        )
        params: list[Any] = []

        if start_date:
            query += " AND time_published >= %s"
            params.append(start_date)
        if end_date:
            query += " AND time_published <= %s"
            params.append(end_date)
        if tickers:
            placeholders = ", ".join(["%s"] * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        query += " ORDER BY time_published"

        with pg_conn() as conn:
            df = self._df_from_query(query, params, conn)

        if df.empty:
            return pd.DataFrame()

        df["time_published"] = pd.to_datetime(df["time_published"])
        return df.set_index("time_published")

    # ------------------------------------------------------------------
    # Metadata / introspection
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
    ) -> pd.DataFrame:
        """Return data_metadata rows."""
        query = "SELECT * FROM data_metadata WHERE 1=1"
        params: list[Any] = []

        if symbol:
            query += " AND symbol=%s"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe=%s"
            params.append(timeframe.value)

        with pg_conn() as conn:
            return self._df_from_query(query, params, conn)

    def get_available_symbols(self) -> list[str]:
        """Return distinct symbols in the ohlcv table."""
        with pg_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

    def get_date_range(
        self,
        symbol: str,
        timeframe: Timeframe,
    ) -> tuple[datetime | None, datetime | None]:
        """Return (first_ts, last_ts) for a symbol/timeframe pair."""
        with pg_conn() as conn:
            result = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) "
                "FROM ohlcv WHERE symbol=%s AND timeframe=%s",
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
        """Check if sufficient data exists for a symbol/timeframe/date range."""
        metadata = self.get_metadata(symbol=symbol, timeframe=timeframe)

        if metadata.empty:
            return (False, f"No data for {symbol} {timeframe.value}")

        meta_row = metadata.iloc[0]
        row_count = meta_row.get("row_count", 0)
        first_ts = meta_row.get("first_timestamp")
        last_ts = meta_row.get("last_timestamp")

        if row_count < min_bars:
            return (False, f"Too few bars: {row_count} < {min_bars}")

        if start_date and first_ts:
            first_ts = pd.Timestamp(first_ts)
            if first_ts > pd.Timestamp(start_date) + pd.Timedelta(days=7):
                return (False, f"Data starts too late: {first_ts} > {start_date}")

        if end_date and last_ts:
            last_ts = pd.Timestamp(last_ts)
            if last_ts < pd.Timestamp(end_date) - pd.Timedelta(days=7):
                return (False, f"Data ends too early: {last_ts} < {end_date}")

        if start_date and end_date:
            days = (end_date - start_date).days
            _bars_per_day = {
                Timeframe.W1: 5 / 7,
                Timeframe.D1: 5 / 7,
                Timeframe.H4: 5 / 7 * 2,
                Timeframe.H1: 5 / 7 * 6.5,
                Timeframe.M30: 5 / 7 * 13,
                Timeframe.M15: 5 / 7 * 26,
                Timeframe.M5: 5 / 7 * 78,
                Timeframe.M1: 5 / 7 * 390,
            }
            expected_bars = max(1, int(days * _bars_per_day.get(timeframe, 5 / 7)))
            completeness = row_count / expected_bars
            if completeness < completeness_threshold:
                return (
                    False,
                    f"Incomplete: {completeness:.1%} < {completeness_threshold:.1%}",
                )

        return (True, f"Valid: {row_count} bars")
