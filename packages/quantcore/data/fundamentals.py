"""
FundamentalsProvider — high-level access to fundamental data from FinancialDatasets.ai.

Wraps the low-level HTTP client and returns normalised DataFrames suitable for
DuckDB storage and feature engineering.  Each method handles JSON→DataFrame
conversion, column renaming, date parsing, and dtype coercion.

Usage::

    from quantcore.data.fundamentals import FundamentalsProvider

    fp = FundamentalsProvider(api_key="...")
    income = fp.fetch_income_statements("NVDA", period="annual", limit=5)
    metrics = fp.fetch_financial_metrics("NVDA")
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from quantcore.data.adapters.financial_datasets_client import FinancialDatasetsClient


def _to_snake_case(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case and strip leading/trailing whitespace."""
    df.columns = [_to_snake_case(c.strip()) for c in df.columns]
    return df


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse date columns, silently ignoring columns that don't exist."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _records_to_df(
    records: list[dict[str, Any]] | None,
    date_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert a list of JSON dicts to a normalised DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = _normalise_columns(df)
    if date_columns:
        df = _parse_dates(df, date_columns)
    return df


class FundamentalsProvider:
    """High-level fundamental data provider backed by FinancialDatasets.ai.

    Shares the same HTTP client (and rate limiter) when constructed from
    an existing ``FinancialDatasetsClient`` instance.

    Args:
        api_key: API key (used only if ``client`` is not provided).
        base_url: API base URL.
        rate_limit_rpm: Requests per minute.
        client: Optional pre-built client to share rate limiter with OHLCV adapter.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.financialdatasets.ai",
        rate_limit_rpm: int = 1000,
        client: FinancialDatasetsClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = FinancialDatasetsClient(
                api_key=api_key,
                base_url=base_url,
                rate_limit_rpm=rate_limit_rpm,
            )
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FundamentalsProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Financial statements ───────────────────────────────────────────────

    def fetch_income_statements(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch income statements.

        Returns DataFrame with columns like: ticker, report_period,
        period, revenue, cost_of_revenue, gross_profit, operating_income,
        net_income, eps_basic, eps_diluted, etc.
        """
        resp = self._client.get_income_statements(ticker, period, limit)
        records = (resp or {}).get("income_statements", [])
        df = _records_to_df(records, date_columns=["report_period", "period"])
        if not df.empty:
            df["ticker"] = ticker
            df["statement_type"] = "income"
        logger.debug(f"[Fundamentals] {ticker} income statements: {len(df)} rows")
        return df

    def fetch_balance_sheets(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch balance sheets."""
        resp = self._client.get_balance_sheets(ticker, period, limit)
        records = (resp or {}).get("balance_sheets", [])
        df = _records_to_df(records, date_columns=["report_period", "period"])
        if not df.empty:
            df["ticker"] = ticker
            df["statement_type"] = "balance"
        logger.debug(f"[Fundamentals] {ticker} balance sheets: {len(df)} rows")
        return df

    def fetch_cash_flows(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch cash flow statements."""
        resp = self._client.get_cash_flow_statements(ticker, period, limit)
        records = (resp or {}).get("cash_flow_statements", [])
        df = _records_to_df(records, date_columns=["report_period", "period"])
        if not df.empty:
            df["ticker"] = ticker
            df["statement_type"] = "cashflow"
        logger.debug(f"[Fundamentals] {ticker} cash flow statements: {len(df)} rows")
        return df

    # ── Financial metrics ──────────────────────────────────────────────────

    def fetch_financial_metrics(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch historical financial metrics (valuation, profitability, etc.)."""
        resp = self._client.get_financial_metrics(ticker, period, limit)
        records = (resp or {}).get("financial_metrics", [])
        df = _records_to_df(records, date_columns=["report_period", "period", "date"])
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} financial metrics: {len(df)} rows")
        return df

    def fetch_financial_metrics_snapshot(self, ticker: str) -> dict[str, Any]:
        """Fetch latest financial metrics snapshot as a dict."""
        resp = self._client.get_financial_metrics_snapshot(ticker)
        return (resp or {}).get("snapshot", {})

    # ── Earnings ───────────────────────────────────────────────────────────

    def fetch_earnings(
        self,
        ticker: str,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Fetch earnings data with estimates and surprises.

        Returns DataFrame with: ticker, report_date, fiscal_date_ending,
        estimate, reported_eps, surprise, surprise_pct.
        """
        resp = self._client.get_earnings(ticker, limit)
        records = (resp or {}).get("earnings", [])
        df = _records_to_df(
            records,
            date_columns=["report_date", "fiscal_date_ending", "date", "period_ending"],
        )
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} earnings: {len(df)} rows")
        return df

    # ── Insider trades ─────────────────────────────────────────────────────

    def fetch_insider_trades(
        self,
        ticker: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch insider trade transactions."""
        resp = self._client.get_insider_trades(ticker, limit)
        records = (resp or {}).get("insider_trades", [])
        df = _records_to_df(
            records,
            date_columns=["transaction_date", "filing_date"],
        )
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} insider trades: {len(df)} rows")
        return df

    # ── Institutional ownership ────────────────────────────────────────────

    def fetch_institutional_ownership(
        self,
        ticker: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch institutional ownership for a ticker."""
        resp = self._client.get_institutional_ownership_by_ticker(ticker, limit)
        records = (resp or {}).get("ownership", [])
        df = _records_to_df(records, date_columns=["report_date", "filing_date"])
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} institutional ownership: {len(df)} rows")
        return df

    # ── Analyst estimates ──────────────────────────────────────────────────

    def fetch_analyst_estimates(self, ticker: str) -> pd.DataFrame:
        """Fetch analyst consensus estimates."""
        resp = self._client.get_analyst_estimates(ticker)
        records = (resp or {}).get("analyst_estimates", [])
        df = _records_to_df(records, date_columns=["fiscal_date", "period_ending"])
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} analyst estimates: {len(df)} rows")
        return df

    # ── SEC filings ────────────────────────────────────────────────────────

    def fetch_sec_filings(
        self,
        ticker: str,
        limit: int = 20,
        filing_type: str | None = None,
    ) -> pd.DataFrame:
        """Fetch SEC filing metadata (not full text)."""
        resp = self._client.get_sec_filings(ticker, limit, filing_type)
        records = (resp or {}).get("filings", [])
        df = _records_to_df(
            records,
            date_columns=["filed_date", "period_of_report", "filing_date"],
        )
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} SEC filings: {len(df)} rows")
        return df

    # ── Company data ───────────────────────────────────────────────────────

    def fetch_company_facts(self, ticker: str) -> dict[str, Any]:
        """Fetch company facts (name, CIK, market cap, employees, etc.)."""
        resp = self._client.get_company_facts(ticker)
        return (resp or {}).get("company", {})

    def fetch_company_news(
        self,
        ticker: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch company news articles."""
        resp = self._client.get_company_news(ticker, limit)
        records = (resp or {}).get("news", [])
        df = _records_to_df(
            records,
            date_columns=["time_published", "published_at", "date"],
        )
        if not df.empty:
            df["ticker"] = ticker
        logger.debug(f"[Fundamentals] {ticker} company news: {len(df)} rows")
        return df

    # ── Macro ──────────────────────────────────────────────────────────────

    def fetch_interest_rates(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical interest rate data."""
        resp = self._client.get_interest_rates_historical(start_date, end_date)
        records = (resp or {}).get("interest_rates", [])
        df = _records_to_df(records, date_columns=["date"])
        logger.debug(f"[Fundamentals] interest rates: {len(df)} rows")
        return df

    # ── Search / screening ─────────────────────────────────────────────────

    def search_financials(self, filters: dict[str, Any]) -> pd.DataFrame:
        """Search financial statements by line items."""
        resp = self._client.search_financials(filters)
        records = (resp or {}).get("results", [])
        return _records_to_df(records)

    def screen_stocks(self, filters: dict[str, Any]) -> pd.DataFrame:
        """Screen stocks by financial criteria."""
        resp = self._client.stock_screener(filters)
        records = (resp or {}).get("results", [])
        return _records_to_df(records)
