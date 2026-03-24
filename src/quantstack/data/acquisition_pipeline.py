"""
Historical data acquisition pipeline — Alpha Vantage primary source.

All OHLCV is split/dividend-adjusted.  All enrichment (financials, macro,
insider, institutional, corporate actions, options, news) comes from Alpha
Vantage except FOMC dates (generated from macro_calendar.py).

## Phases

| Phase              | AV calls (50 syms)  | Frequency     |
|--------------------|---------------------|---------------|
| ohlcv_5min         | 50 × months = ~1200 | Daily delta   |
| ohlcv_daily        | 50                  | Daily delta   |
| financials         | 50 × 3 = 150        | Quarterly     |
| earnings_history   | 50                  | Quarterly     |
| macro              | 9 (global)          | Monthly       |
| insider            | 50                  | Weekly        |
| institutional      | 50                  | Quarterly     |
| corporate_actions  | 50 × 2 = 100        | Quarterly     |
| options            | 50                  | Daily         |
| news               | ~10 batches         | Daily         |
| fundamentals       | 50                  | Weekly        |

## Rate limits

Premium ($49.99/mo): set ALPHA_VANTAGE_RATE_LIMIT=75 in .env
All AV calls are serialised per phase to keep the built-in rate limiter correct.

## Idempotency

Every phase checks the DB before calling the API.  Safe to run daily.
"""

from __future__ import annotations

import asyncio
import calendar
import json as _json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe

if TYPE_CHECKING:
    from quantstack.data.adapters.alpaca import AlpacaAdapter
    from quantstack.data.fetcher import AlphaVantageClient
    from quantstack.data.storage import DataStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M5_LOOKBACK_MONTHS = 24  # AV intraday history depth (paid plan)

NEWS_BATCH_SIZE = 5  # AV NEWS_SENTIMENT: max tickers per call
NEWS_LOOKBACK_DAYS = 30

# Global macro series to fetch (not per-symbol)
MACRO_SERIES: list[tuple[str, str, str | None]] = [
    # (function,            interval,   maturity)
    ("REAL_GDP", "quarterly", None),
    ("FEDERAL_FUNDS_RATE", "monthly", None),
    ("TREASURY_YIELD", "daily", "10year"),
    ("CPI", "monthly", None),
    ("INFLATION", "annual", None),
    ("RETAIL_SALES", "monthly", None),
    ("UNEMPLOYMENT", "monthly", None),
    ("NONFARM_PAYROLL", "monthly", None),
    ("DURABLES", "monthly", None),
]

ALL_PHASES = [
    "ohlcv_5min",
    "ohlcv_1h",
    "ohlcv_daily",
    "financials",
    "earnings_history",
    "macro",
    "insider",
    "institutional",
    "corporate_actions",
    "options",
    "news",
    "fundamentals",
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class PhaseReport:
    phase: str
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.phase:<20}] {self.succeeded:>3}/{self.total:<3} ok  "
            f"{self.skipped:>3} skip  {self.failed:>3} fail  "
            f"{self.elapsed_seconds:>7.1f}s"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AcquisitionPipeline:
    """
    Full-stack data acquisition — OHLCV, financials, macro, insider,
    institutional, corporate actions, options, news, and fundamentals.

    Args:
        av_client: AlphaVantageClient (rate-limited AV wrapper).
        store:     DataStore (DuckDB persistence).
        alpaca:    Optional AlpacaAdapter (fallback OHLCV when AV returns empty).
    """

    def __init__(
        self,
        av_client: AlphaVantageClient,
        store: DataStore,
        alpaca: AlpacaAdapter | None = None,
    ) -> None:
        self._av = av_client
        self._store = store
        self._alpaca = alpaca

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    async def run(
        self,
        symbols: list[str],
        phases: list[str] | None = None,
        m5_lookback_months: int = M5_LOOKBACK_MONTHS,
    ) -> list[PhaseReport]:
        """Run selected phases sequentially.  Returns one PhaseReport per phase."""
        if phases is None:
            phases = ALL_PHASES

        reports: list[PhaseReport] = []
        for phase in phases:
            r = await self._dispatch(phase, symbols, m5_lookback_months)
            if r is not None:
                reports.append(r)
                logger.info(str(r))
        return reports

    async def _dispatch(
        self,
        phase: str,
        symbols: list[str],
        m5_lookback_months: int,
    ) -> PhaseReport | None:
        if phase == "ohlcv_5min":
            return await self.run_ohlcv_5min(symbols, m5_lookback_months)
        if phase == "ohlcv_1h":
            return await self.run_ohlcv_1h(symbols)
        if phase == "ohlcv_daily":
            return await self.run_ohlcv_daily(symbols)
        if phase == "financials":
            return await self.run_financials(symbols)
        if phase == "earnings_history":
            return await self.run_earnings_history(symbols)
        if phase == "macro":
            return await self.run_macro()
        if phase == "insider":
            return await self.run_insider(symbols)
        if phase == "institutional":
            return await self.run_institutional(symbols)
        if phase == "corporate_actions":
            return await self.run_corporate_actions(symbols)
        if phase == "options":
            return await self.run_options(symbols)
        if phase == "news":
            return await self.run_news(symbols)
        if phase == "fundamentals":
            return await self.run_fundamentals(symbols)
        logger.warning(f"Unknown phase '{phase}', skipping")
        return None

    # -----------------------------------------------------------------------
    # Phase: 5-min adjusted OHLCV
    # -----------------------------------------------------------------------

    async def run_ohlcv_5min(
        self,
        symbols: list[str],
        lookback_months: int = M5_LOOKBACK_MONTHS,
    ) -> PhaseReport:
        """TIME_SERIES_INTRADAY adjusted=true, one month per call, delta-only."""
        start = _now()
        report = PhaseReport(phase="ohlcv_5min", total=len(symbols))
        for symbol in symbols:
            try:
                bars = await asyncio.to_thread(
                    self._fetch_5min_delta, symbol, lookback_months
                )
                if bars == 0:
                    report.skipped += 1
                else:
                    report.succeeded += 1
                    logger.debug(f"[5min] {symbol}: +{bars} bars")
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[5min] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_5min_delta(self, symbol: str, lookback_months: int) -> int:
        last_dt = self._last_ohlcv_ts(symbol, Timeframe.M5)
        today = date.today()
        first_month = (
            _add_months(last_dt.date(), 0)
            if last_dt
            else _add_months(today, -lookback_months)
        )
        total = 0
        for month_str in _month_range(first_month, today):
            try:
                df = self._av.fetch_intraday_by_month(symbol, "5min", month_str, "full")
                if df.empty:
                    continue
                if last_dt:
                    df = df[df.index > last_dt]
                if df.empty:
                    continue
                total += self._store.save_ohlcv(df, symbol, Timeframe.M5)
            except Exception as exc:
                logger.debug(f"[5min] {symbol} {month_str}: {exc}")
        return total

    # -----------------------------------------------------------------------
    # Phase: 1-hour OHLCV (intraday extended, 60min interval)
    # -----------------------------------------------------------------------

    async def run_ohlcv_1h(self, symbols: list[str]) -> PhaseReport:
        """TIME_SERIES_INTRADAY_EXTENDED interval=60min, 2 years of hourly bars."""
        start = _now()
        report = PhaseReport(phase="ohlcv_1h", total=len(symbols))
        for symbol in symbols:
            try:
                bars = await asyncio.to_thread(self._fetch_1h_delta, symbol)
                if bars == 0:
                    report.skipped += 1
                else:
                    report.succeeded += 1
                    logger.debug(f"[1h] {symbol}: +{bars} bars")
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[1h] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_1h_delta(self, symbol: str) -> int:
        # Full fetch — save_ohlcv handles deduplication via INSERT OR REPLACE
        df = self._av.fetch_all_intraday_history(
            symbol, interval="60min", start_year=2022, end_year=2026,
        )
        if df is None or df.empty:
            return 0
        return self._store.save_ohlcv(df, symbol, Timeframe.H1)

    # -----------------------------------------------------------------------
    # Phase: daily adjusted OHLCV
    # -----------------------------------------------------------------------

    async def run_ohlcv_daily(self, symbols: list[str]) -> PhaseReport:
        """TIME_SERIES_DAILY_ADJUSTED outputsize=full, delta-only."""
        start = _now()
        report = PhaseReport(phase="ohlcv_daily", total=len(symbols))
        for symbol in symbols:
            try:
                bars = await asyncio.to_thread(self._fetch_daily_delta, symbol)
                if bars == 0:
                    report.skipped += 1
                else:
                    report.succeeded += 1
                    logger.debug(f"[daily] {symbol}: +{bars} bars")
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[daily] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_daily_delta(self, symbol: str) -> int:
        last_dt = self._last_ohlcv_ts(symbol, Timeframe.D1)
        if last_dt and (datetime.now(UTC) - last_dt).days < 1:
            return 0
        df = self._av.fetch_daily(symbol, outputsize="full")
        if df.empty:
            return 0
        if last_dt:
            last_date = last_dt.date() if hasattr(last_dt, "date") else last_dt
            df = df[df.index.date > last_date]  # type: ignore[attr-defined]
        if df.empty:
            return 0
        return self._store.save_ohlcv(df, symbol, Timeframe.D1)

    # -----------------------------------------------------------------------
    # Phase: financial statements (income, balance sheet, cash flow)
    # -----------------------------------------------------------------------

    async def run_financials(self, symbols: list[str]) -> PhaseReport:
        """INCOME_STATEMENT + BALANCE_SHEET + CASH_FLOW — 3 calls/symbol."""
        start = _now()
        report = PhaseReport(phase="financials", total=len(symbols))
        for symbol in symbols:
            try:
                rows = await asyncio.to_thread(self._fetch_and_store_financials, symbol)
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[financials] {symbol}: {rows} rows")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[financials] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_financials(self, symbol: str) -> int:
        # Skip if already fetched this quarter (statements change quarterly)
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT MAX(report_period) FROM financial_statements WHERE ticker = ?",
                    [symbol],
                ).fetchone()
                if row and row[0]:
                    last = row[0]
                    if isinstance(last, str):
                        last = datetime.fromisoformat(last).date()
                    if (date.today() - last).days < 80:  # ~1 quarter
                        return 0
        except Exception:
            pass

        total = 0
        for fetch_fn, stmt_type in [
            (self._av.fetch_income_statement, "income_statement"),
            (self._av.fetch_balance_sheet, "balance_sheet"),
            (self._av.fetch_cash_flow, "cash_flow"),
        ]:
            try:
                data = fetch_fn(symbol)
                if not data:
                    continue
                df = _av_statement_to_df(symbol, data, stmt_type)
                if not df.empty:
                    total += self._store.save_financial_statements(df)
            except Exception as exc:
                logger.debug(f"[financials] {symbol} {stmt_type}: {exc}")
        return total

    # -----------------------------------------------------------------------
    # Phase: earnings history (actual vs estimated EPS)
    # -----------------------------------------------------------------------

    async def run_earnings_history(self, symbols: list[str]) -> PhaseReport:
        """EARNINGS endpoint — actual + estimated EPS, annual + quarterly."""
        start = _now()
        report = PhaseReport(phase="earnings_history", total=len(symbols))
        for symbol in symbols:
            try:
                rows = await asyncio.to_thread(self._fetch_and_store_earnings, symbol)
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[earnings] {symbol}: {rows} rows")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[earnings] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_earnings(self, symbol: str) -> int:
        data = self._av.fetch_earnings_history(symbol)
        if not data:
            return 0

        rows = []
        for period_type, key in [
            ("annual", "annualEarnings"),
            ("quarterly", "quarterlyEarnings"),
        ]:
            for e in data.get(key, []):
                rows.append(
                    {
                        "symbol": symbol,
                        "report_date": e.get("reportedDate")
                        or e.get("fiscalDateEnding"),
                        "fiscal_date_ending": e.get("fiscalDateEnding"),
                        "estimate": _safe_float(e.get("estimatedEPS")),
                        "reported_eps": _safe_float(e.get("reportedEPS")),
                        "surprise": _safe_float(e.get("surprise")),
                        "surprise_pct": _safe_float(e.get("surprisePercentage")),
                    }
                )
        if not rows:
            return 0

        df = pd.DataFrame(rows)
        with self._store._use_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO earnings_calendar
                    (symbol, report_date, fiscal_date_ending,
                     estimate, reported_eps, surprise, surprise_pct)
                SELECT symbol, report_date, fiscal_date_ending,
                       estimate, reported_eps, surprise, surprise_pct
                FROM df
            """
            )
        return len(rows)

    # -----------------------------------------------------------------------
    # Phase: macro indicators (global — not per symbol)
    # -----------------------------------------------------------------------

    async def run_macro(self) -> PhaseReport:
        """
        Fetch 9 global macro series: GDP, Fed funds, Treasury 10yr, CPI,
        inflation, retail sales, unemployment, nonfarm payroll, durable goods.

        9 AV calls total regardless of universe size.
        Skips indicators whose last cached date is less than 28 days ago.
        """
        start = _now()
        report = PhaseReport(phase="macro", total=len(MACRO_SERIES))
        for function, interval, maturity in MACRO_SERIES:
            try:
                rows = await asyncio.to_thread(
                    self._fetch_and_store_macro, function, interval, maturity
                )
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[macro] {function}: +{rows} rows")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{function}: {exc}")
                logger.warning(f"[macro] {function} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_macro(
        self, function: str, interval: str, maturity: str | None
    ) -> int:
        # Skip if refreshed in the last 28 days
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT MAX(date) FROM macro_indicators WHERE indicator = ?",
                    [function],
                ).fetchone()
                if row and row[0]:
                    last = row[0]
                    if isinstance(last, str):
                        last = date.fromisoformat(last)
                    if (date.today() - last).days < 28:
                        return 0
        except Exception:
            pass

        kwargs: dict = {"interval": interval}
        if maturity:
            kwargs["maturity"] = maturity
        df = self._av.fetch_economic_indicator(function, **kwargs)
        if df.empty:
            return 0
        return self._store.save_macro_indicators(function, df)

    # -----------------------------------------------------------------------
    # Phase: insider transactions
    # -----------------------------------------------------------------------

    async def run_insider(self, symbols: list[str]) -> PhaseReport:
        """INSIDER_TRANSACTIONS — all reported buys/sells per symbol."""
        start = _now()
        report = PhaseReport(phase="insider", total=len(symbols))
        for symbol in symbols:
            try:
                rows = await asyncio.to_thread(self._fetch_and_store_insider, symbol)
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[insider] {symbol}: {rows} rows")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[insider] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_insider(self, symbol: str) -> int:
        # Skip if refreshed in the last 7 days
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT MAX(fetched_at) FROM insider_trades WHERE ticker = ?",
                    [symbol],
                ).fetchone()
                if row and row[0]:
                    last = row[0]
                    if isinstance(last, str):
                        last = datetime.fromisoformat(last)
                    if isinstance(last, datetime):
                        last = last.replace(tzinfo=UTC)
                    if (datetime.now(UTC) - last).days < 7:
                        return 0
        except Exception:
            pass

        df = self._av.fetch_insider_transactions(symbol)
        if df.empty:
            return 0

        df = df.copy()
        df["ticker"] = symbol
        # Map AV column names → schema column names
        rename = {
            "share_price": "price_per_share",
            "acquisition_or_disposition": "transaction_type",
        }
        df.rename(
            columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True
        )
        return self._store.save_insider_trades(df)

    # -----------------------------------------------------------------------
    # Phase: institutional holdings (13F)
    # -----------------------------------------------------------------------

    async def run_institutional(self, symbols: list[str]) -> PhaseReport:
        """INSTITUTIONAL_HOLDINGS — top 13F holders per symbol."""
        start = _now()
        report = PhaseReport(phase="institutional", total=len(symbols))
        for symbol in symbols:
            try:
                rows = await asyncio.to_thread(
                    self._fetch_and_store_institutional, symbol
                )
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[institutional] {symbol}: {rows} rows")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[institutional] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_institutional(self, symbol: str) -> int:
        # Skip if refreshed in the last 80 days (~1 quarter)
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT MAX(fetched_at) FROM institutional_ownership WHERE ticker = ?",
                    [symbol],
                ).fetchone()
                if row and row[0]:
                    last = row[0]
                    if isinstance(last, str):
                        last = datetime.fromisoformat(last)
                    if isinstance(last, datetime):
                        last = last.replace(tzinfo=UTC)
                    if (datetime.now(UTC) - last).days < 80:
                        return 0
        except Exception:
            pass

        df = self._av.fetch_institutional_holdings(symbol)
        if df.empty:
            return 0

        df = df.copy()
        df["ticker"] = symbol
        # Map AV column names → schema column names
        rename = {
            "investor": "investor_name",
            "date_reported": "report_date",
            "shares": "shares_held",
            "value": "market_value",
            "weight": "portfolio_pct",
            "change_in_shares": "change_shares",
        }
        df.rename(
            columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True
        )
        return self._store.save_institutional_ownership(df)

    # -----------------------------------------------------------------------
    # Phase: corporate actions (dividends + splits)
    # -----------------------------------------------------------------------

    async def run_corporate_actions(self, symbols: list[str]) -> PhaseReport:
        """DIVIDENDS + STOCK_SPLITS — full history per symbol, 2 calls each."""
        start = _now()
        report = PhaseReport(phase="corporate_actions", total=len(symbols))
        for symbol in symbols:
            try:
                rows = await asyncio.to_thread(
                    self._fetch_and_store_corp_actions, symbol
                )
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[corp_actions] {symbol}: {rows} rows")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[corp_actions] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_corp_actions(self, symbol: str) -> int:
        # Skip if refreshed in the last 28 days
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT MAX(fetched_at) FROM corporate_actions WHERE ticker = ?",
                    [symbol],
                ).fetchone()
                if row and row[0]:
                    last = row[0]
                    if isinstance(last, str):
                        last = datetime.fromisoformat(last)
                    if isinstance(last, datetime):
                        last = last.replace(tzinfo=UTC)
                    if (datetime.now(UTC) - last).days < 28:
                        return 0
        except Exception:
            pass

        total = 0
        divs = self._av.fetch_dividends(symbol)
        if not divs.empty:
            total += self._store.save_corporate_actions(symbol, divs, "dividend")

        splits = self._av.fetch_stock_splits(symbol)
        if not splits.empty:
            total += self._store.save_corporate_actions(symbol, splits, "split")

        return total

    # -----------------------------------------------------------------------
    # Phase: options chains
    # -----------------------------------------------------------------------

    async def run_options(self, symbols: list[str]) -> PhaseReport:
        """HISTORICAL_OPTIONS — today's full chain per symbol."""
        start = _now()
        today_str = date.today().isoformat()
        report = PhaseReport(phase="options", total=len(symbols))
        for symbol in symbols:
            try:
                rows = await asyncio.to_thread(
                    self._fetch_and_store_options, symbol, today_str
                )
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[options] {symbol}: {rows} contracts")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[options] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_options(self, symbol: str, date_str: str) -> int:
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM options_chains WHERE underlying = ? AND data_date = ?",
                    [symbol, date_str],
                ).fetchone()
                if row and row[0] > 0:
                    return 0
        except Exception:
            pass

        # fetch_historical_options returns a DataFrame directly
        df = self._av.fetch_historical_options(symbol, date=date_str)
        if df.empty:
            return 0
        self._store.save_options_chain(df, symbol, datetime.fromisoformat(date_str))
        return len(df)

    # -----------------------------------------------------------------------
    # Phase: news sentiment
    # -----------------------------------------------------------------------

    async def run_news(
        self,
        symbols: list[str],
        days_back: int = NEWS_LOOKBACK_DAYS,
        batch_size: int = NEWS_BATCH_SIZE,
    ) -> PhaseReport:
        """NEWS_SENTIMENT batched 5 tickers/call.  INSERT OR IGNORE prevents dupes."""
        start = _now()
        batches = _chunk(symbols, batch_size)
        report = PhaseReport(phase="news", total=len(batches))
        time_from = (datetime.now(UTC) - timedelta(days=days_back)).strftime(
            "%Y%m%dT%H%M"
        )
        for batch in batches:
            try:
                rows = await asyncio.to_thread(
                    self._fetch_and_store_news, batch, time_from
                )
                if rows > 0:
                    report.succeeded += 1
                    logger.debug(f"[news] {batch}: {rows} articles")
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{batch}: {exc}")
                logger.warning(f"[news] {batch} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_news(self, symbols: list[str], time_from: str) -> int:
        # fetch_news_sentiment returns a DataFrame with time_published as index
        df = self._av.fetch_news_sentiment(
            tickers=",".join(symbols),
            time_from=time_from,
            limit=200,
        )
        if df.empty:
            return 0
        logger.info(f"Fetched {len(df)} news articles")

        # Reset index to make time_published a regular column
        df = df.reset_index()

        # Ensure all required columns exist
        for col in ("ticker", "ticker_sentiment_score", "ticker_sentiment_label", "relevance_score"):
            if col not in df.columns:
                df[col] = None

        with self._store._use_conn() as conn:
            # Use ON CONFLICT instead of INSERT OR IGNORE (DuckDB prefers ON CONFLICT)
            conn.execute(
                """
                INSERT INTO news_sentiment
                    (time_published, title, summary, source, url, ticker,
                     overall_sentiment_score, overall_sentiment_label,
                     ticker_sentiment_score, ticker_sentiment_label, relevance_score)
                SELECT time_published, title, summary, source, url, ticker,
                       overall_sentiment_score, overall_sentiment_label,
                       ticker_sentiment_score, ticker_sentiment_label, relevance_score
                FROM df
                ON CONFLICT (time_published, title, ticker) DO NOTHING
            """
            )
        return len(df)

    # -----------------------------------------------------------------------
    # Phase: company overview + fundamentals
    # -----------------------------------------------------------------------

    async def run_fundamentals(self, symbols: list[str]) -> PhaseReport:
        """OVERVIEW — company profile, valuation ratios, sector.  1 call/symbol."""
        start = _now()
        report = PhaseReport(phase="fundamentals", total=len(symbols))
        for symbol in symbols:
            try:
                saved = await asyncio.to_thread(self._fetch_and_store_overview, symbol)
                if saved:
                    report.succeeded += 1
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{symbol}: {exc}")
                logger.warning(f"[fundamentals] {symbol} failed: {exc}")
        report.elapsed_seconds = _elapsed(start)
        return report

    def _fetch_and_store_overview(self, symbol: str) -> bool:
        # Skip if refreshed in the last 7 days
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT updated_at FROM company_overview WHERE symbol = ?",
                    [symbol],
                ).fetchone()
                if row and row[0]:
                    updated = row[0]
                    if isinstance(updated, str):
                        updated = datetime.fromisoformat(updated)
                    if isinstance(updated, datetime):
                        updated = updated.replace(tzinfo=UTC)
                    if (datetime.now(UTC) - updated).days < 7:
                        return False
        except Exception:
            pass

        overview = self._av.fetch_company_overview(symbol)
        if not overview or "Symbol" not in overview:
            return False

        # AV returns the string "None" for missing dates — convert to Python None
        ex_div = overview.get("ExDividendDate")
        if ex_div in (None, "None", "N/A", ""):
            ex_div = None

        with self._store._use_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO company_overview
                    (symbol, name, sector, industry, market_cap,
                     dividend_yield, ex_dividend_date,
                     fifty_two_week_high, fifty_two_week_low, beta, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                [
                    overview.get("Symbol"),
                    overview.get("Name"),
                    overview.get("Sector"),
                    overview.get("Industry"),
                    _safe_float(overview.get("MarketCapitalization")),
                    _safe_float(overview.get("DividendYield")),
                    ex_div,
                    _safe_float(overview.get("52WeekHigh")),
                    _safe_float(overview.get("52WeekLow")),
                    _safe_float(overview.get("Beta")),
                ],
            )
        logger.debug(f"[fundamentals] {symbol}: overview saved")
        return True

    # -----------------------------------------------------------------------
    # Helpers: last cached timestamp
    # -----------------------------------------------------------------------

    def _last_ohlcv_ts(self, symbol: str, tf: Timeframe) -> datetime | None:
        try:
            with self._store._use_conn() as conn:
                row = conn.execute(
                    "SELECT last_timestamp FROM data_metadata WHERE symbol = ? AND timeframe = ?",
                    [symbol, tf.value],
                ).fetchone()
                if row and row[0]:
                    ts = row[0]
                    if isinstance(ts, datetime):
                        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
                    if isinstance(ts, date):
                        return datetime(ts.year, ts.month, ts.day, tzinfo=UTC)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _elapsed(start: datetime) -> float:
    return (datetime.now(UTC) - start).total_seconds()


def _safe_float(value: object) -> float | None:
    if value is None or value in ("None", "", "-"):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _month_range(start: date, end: date) -> list[str]:
    months: list[str] = []
    cursor = date(start.year, start.month, 1)
    stop = date(end.year, end.month, 1)
    while cursor <= stop:
        months.append(cursor.strftime("%Y-%m"))
        cursor = _add_months(cursor, 1)
    return months


def _av_statement_to_df(symbol: str, data: dict, statement_type: str) -> pd.DataFrame:
    """
    Transform an AV INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW response
    into the format expected by DataStore.save_financial_statements().
    """
    # Field mappings: AV key → schema column
    KEY_MAP = {
        "totalRevenue": "revenue",
        "netIncome": "net_income",
        "grossProfit": "gross_profit",
        "operatingIncome": "operating_income",
        "totalAssets": "total_assets",
        "totalLiabilities": "total_debt",
    }

    rows = []
    for period_type, report_key in [
        ("annual", "annualReports"),
        ("quarterly", "quarterlyReports"),
    ]:
        for report in data.get(report_key, []):
            row: dict = {}
            row["ticker"] = symbol
            row["statement_type"] = statement_type
            row["period_type"] = period_type
            row["report_period"] = report.get("fiscalDateEnding")

            # Extract key numeric columns
            for av_key, col in KEY_MAP.items():
                if av_key in report:
                    row[col] = _safe_float(report[av_key])

            # Remaining fields go into the JSON blob (handled by save_financial_statements)
            for k, v in report.items():
                if k not in ("fiscalDateEnding",) and k not in KEY_MAP:
                    row[k] = v

            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
