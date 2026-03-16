"""Fundamentals schema mixin for DataStore.

Provides DDL and save/load methods for fundamental data tables sourced from
FinancialDatasets.ai: financial statements, financial metrics, insider trades,
institutional ownership, analyst estimates, and SEC filings.

Existing tables reused (no DDL here):
  - ``earnings_calendar``   — populated by FundamentalsProvider.fetch_earnings()
  - ``news_sentiment``      — populated by FundamentalsProvider.fetch_company_news()
  - ``company_overview``    — populated by FundamentalsProvider.fetch_company_facts()
"""

from datetime import datetime

import pandas as pd
from loguru import logger


class FundamentalsSchemaMixin:
    """Mixin that owns DDL and CRUD for fundamentals tables."""

    # ── Schema initialisation ──────────────────────────────────────────────

    def _init_fundamentals_schema(self) -> None:
        """Create all fundamentals tables and indexes."""
        # Financial statements — one row per (ticker, period, statement_type).
        # Line items stored as a JSON blob because financial statement schemas
        # vary by company and FinancialDatasets.ai may add new fields.
        # Key numeric fields extracted as columns for direct SQL queries.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_statements (
                ticker            VARCHAR NOT NULL,
                statement_type    VARCHAR NOT NULL,
                period_type       VARCHAR NOT NULL,
                report_period     DATE    NOT NULL,
                revenue           DOUBLE,
                net_income        DOUBLE,
                total_assets      DOUBLE,
                total_debt        DOUBLE,
                operating_income  DOUBLE,
                gross_profit      DOUBLE,
                eps_diluted       DOUBLE,
                data              JSON,
                fetched_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, report_period, statement_type, period_type)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fin_stmt_ticker
            ON financial_statements (ticker, statement_type)
        """)

        # Financial metrics — valuation, profitability, leverage ratios.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                ticker           VARCHAR NOT NULL,
                date             DATE    NOT NULL,
                period_type      VARCHAR NOT NULL DEFAULT 'annual',
                market_cap       DOUBLE,
                pe_ratio         DOUBLE,
                pb_ratio         DOUBLE,
                ps_ratio         DOUBLE,
                ev_to_ebitda     DOUBLE,
                roe              DOUBLE,
                roa              DOUBLE,
                gross_margin     DOUBLE,
                operating_margin DOUBLE,
                net_margin       DOUBLE,
                debt_to_equity   DOUBLE,
                current_ratio    DOUBLE,
                dividend_yield   DOUBLE,
                revenue_growth   DOUBLE,
                earnings_growth  DOUBLE,
                data             JSON,
                fetched_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date, period_type)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fin_metrics_ticker
            ON financial_metrics (ticker, date)
        """)

        # Insider trades.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS insider_trades (
                ticker              VARCHAR NOT NULL,
                transaction_date    DATE    NOT NULL,
                owner_name          VARCHAR NOT NULL,
                owner_title         VARCHAR,
                transaction_type    VARCHAR,
                shares              DOUBLE,
                price_per_share     DOUBLE,
                total_value         DOUBLE,
                shares_owned_after  DOUBLE,
                filing_date         DATE,
                fetched_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, transaction_date, owner_name, shares)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
            ON insider_trades (ticker, transaction_date)
        """)

        # Institutional ownership.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS institutional_ownership (
                ticker          VARCHAR NOT NULL,
                investor_name   VARCHAR NOT NULL,
                report_date     DATE    NOT NULL,
                shares_held     DOUBLE,
                market_value    DOUBLE,
                portfolio_pct   DOUBLE,
                change_shares   DOUBLE,
                change_pct      DOUBLE,
                fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, investor_name, report_date)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_instit_ticker
            ON institutional_ownership (ticker, report_date)
        """)

        # Analyst estimates.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analyst_estimates (
                ticker       VARCHAR NOT NULL,
                fiscal_date  DATE    NOT NULL,
                period_type  VARCHAR NOT NULL DEFAULT 'annual',
                metric       VARCHAR NOT NULL DEFAULT 'eps',
                consensus    DOUBLE,
                high         DOUBLE,
                low          DOUBLE,
                num_analysts INTEGER,
                fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, fiscal_date, period_type, metric)
            )
        """)

        # SEC filings (metadata only — full text fetched on demand).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                ticker            VARCHAR NOT NULL,
                accession_number  VARCHAR NOT NULL,
                filing_type       VARCHAR,
                filed_date        DATE,
                period_of_report  DATE,
                url               VARCHAR,
                fetched_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, accession_number)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker
            ON sec_filings (ticker, filed_date)
        """)

        logger.debug("Fundamentals schema initialized")

    # ── Financial statements ───────────────────────────────────────────────

    def save_financial_statements(self, df: pd.DataFrame) -> int:
        """Save financial statement rows.

        Expects columns: ticker, statement_type, period_type, report_period,
        plus any of the extracted key fields.  Remaining columns are stored
        in the ``data`` JSON blob.
        """
        if df.empty:
            return 0
        import json as _json

        key_cols = {
            "ticker", "statement_type", "period_type", "report_period",
            "revenue", "net_income", "total_assets", "total_debt",
            "operating_income", "gross_profit", "eps_diluted",
        }
        data = df.copy()

        # Build JSON blob from non-key columns.
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

        insert_cols = list(key_cols) + (["data"] if "data" in data.columns else [])
        insert_df = data[insert_cols]

        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO financial_statements
            ({', '.join(insert_cols)})
            SELECT {', '.join(insert_cols)} FROM insert_df
            """
        )
        logger.info(f"Saved {len(insert_df)} financial statement rows")
        return len(insert_df)

    def load_financial_statements(
        self,
        ticker: str,
        statement_type: str | None = None,
        period_type: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Load financial statements for a ticker."""
        query = "SELECT * FROM financial_statements WHERE ticker = ?"
        params: list = [ticker]
        if statement_type:
            query += " AND statement_type = ?"
            params.append(statement_type)
        if period_type:
            query += " AND period_type = ?"
            params.append(period_type)
        query += " ORDER BY report_period DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).fetchdf()

    # ── Financial metrics ──────────────────────────────────────────────────

    def save_financial_metrics(self, df: pd.DataFrame) -> int:
        """Save financial metrics rows."""
        if df.empty:
            return 0
        import json as _json

        key_cols = {
            "ticker", "date", "period_type", "market_cap", "pe_ratio",
            "pb_ratio", "ps_ratio", "ev_to_ebitda", "roe", "roa",
            "gross_margin", "operating_margin", "net_margin",
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

        insert_cols = list(key_cols) + (["data"] if "data" in data.columns else [])
        insert_df = data[insert_cols]

        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO financial_metrics
            ({', '.join(insert_cols)})
            SELECT {', '.join(insert_cols)} FROM insert_df
            """
        )
        logger.info(f"Saved {len(insert_df)} financial metrics rows")
        return len(insert_df)

    def load_financial_metrics(
        self,
        ticker: str,
        period_type: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Load financial metrics for a ticker."""
        query = "SELECT * FROM financial_metrics WHERE ticker = ?"
        params: list = [ticker]
        if period_type:
            query += " AND period_type = ?"
            params.append(period_type)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).fetchdf()

    # ── Insider trades ─────────────────────────────────────────────────────

    def save_insider_trades(self, df: pd.DataFrame) -> int:
        """Save insider trade rows."""
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

        insert_df = data[cols]
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO insider_trades
            ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM insert_df
            """
        )
        logger.info(f"Saved {len(insert_df)} insider trade rows")
        return len(insert_df)

    def load_insider_trades(
        self,
        ticker: str,
        start_date: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Load insider trades for a ticker."""
        query = "SELECT * FROM insider_trades WHERE ticker = ?"
        params: list = [ticker]
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date.date() if hasattr(start_date, "date") else start_date)
        query += " ORDER BY transaction_date DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).fetchdf()

    # ── Institutional ownership ────────────────────────────────────────────

    def save_institutional_ownership(self, df: pd.DataFrame) -> int:
        """Save institutional ownership rows."""
        if df.empty:
            return 0

        cols = [
            "ticker", "investor_name", "report_date", "shares_held",
            "market_value", "portfolio_pct", "change_shares", "change_pct",
        ]
        data = df.copy()
        for c in cols:
            if c not in data.columns:
                data[c] = None

        insert_df = data[cols]
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO institutional_ownership
            ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM insert_df
            """
        )
        logger.info(f"Saved {len(insert_df)} institutional ownership rows")
        return len(insert_df)

    def load_institutional_ownership(
        self,
        ticker: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Load institutional ownership for a ticker."""
        query = """
            SELECT * FROM institutional_ownership
            WHERE ticker = ?
            ORDER BY report_date DESC
            LIMIT ?
        """
        return self.conn.execute(query, [ticker, limit]).fetchdf()

    # ── Analyst estimates ──────────────────────────────────────────────────

    def save_analyst_estimates(self, df: pd.DataFrame) -> int:
        """Save analyst estimate rows."""
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

        insert_df = data[cols]
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO analyst_estimates
            ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM insert_df
            """
        )
        logger.info(f"Saved {len(insert_df)} analyst estimate rows")
        return len(insert_df)

    def load_analyst_estimates(self, ticker: str) -> pd.DataFrame:
        """Load analyst estimates for a ticker."""
        return self.conn.execute(
            "SELECT * FROM analyst_estimates WHERE ticker = ? ORDER BY fiscal_date DESC",
            [ticker],
        ).fetchdf()

    # ── SEC filings ────────────────────────────────────────────────────────

    def save_sec_filings(self, df: pd.DataFrame) -> int:
        """Save SEC filing metadata rows."""
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

        insert_df = data[cols]
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO sec_filings
            ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM insert_df
            """
        )
        logger.info(f"Saved {len(insert_df)} SEC filing rows")
        return len(insert_df)

    def load_sec_filings(
        self,
        ticker: str,
        filing_type: str | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Load SEC filings for a ticker."""
        query = "SELECT * FROM sec_filings WHERE ticker = ?"
        params: list = [ticker]
        if filing_type:
            query += " AND filing_type = ?"
            params.append(filing_type)
        query += " ORDER BY filed_date DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).fetchdf()
