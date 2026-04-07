"""
Scheduled data refresh — keeps market data fresh for trading decisions.

Two modes:
  - Intraday (every 5 min during market hours):
    Bulk quotes, intraday OHLCV, news sentiment for active universe.
  - End-of-day (once after market close):
    Daily candles, options chains, fundamentals for full universe.

All writes are idempotent (ON CONFLICT DO UPDATE). Safe to run concurrently.
Rate limits are enforced by AlphaVantageClient's built-in sliding window.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.db import pg_conn
from quantstack.signal_engine import cache as signal_cache

OPTIONS_REFRESH_TOP_N = int(os.environ.get("OPTIONS_REFRESH_TOP_N", "30"))


@dataclass
class RefreshReport:
    """Summary of a data refresh cycle."""

    mode: str  # "intraday" or "eod"
    symbols_refreshed: int = 0
    api_calls: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _get_active_symbols() -> list[str]:
    """Load active universe symbols from DB."""
    with pg_conn() as conn:
        rows = conn.execute(
            "SELECT symbol FROM universe WHERE is_active = TRUE ORDER BY symbol"
        ).fetchall()
    return [r[0] for r in rows]


def _get_watched_symbols() -> list[str]:
    """Get symbols with open positions or active strategies — higher priority."""
    with pg_conn() as conn:
        # Symbols with open positions (rows exist = position is open)
        pos_rows = conn.execute(
            "SELECT DISTINCT symbol FROM positions"
        ).fetchall()
        # Symbols with active strategies
        strat_rows = conn.execute(
            "SELECT DISTINCT symbol FROM strategies "
            "WHERE status IN ('live', 'forward_testing') AND symbol IS NOT NULL"
        ).fetchall()
    symbols = {r[0] for r in pos_rows} | {r[0] for r in strat_rows}
    return sorted(symbols)


async def run_intraday_refresh() -> RefreshReport:
    """Refresh intraday data for the active universe.

    Called every 5 minutes during market hours by the trading graph.

    Fetches:
      1. Bulk quotes (1 API call per 100 symbols)
      2. 5-min OHLCV for watched symbols (positions + active strategies)
      3. News sentiment for top symbols (1 call per 5 tickers)
    """
    report = RefreshReport(mode="intraday")
    t0 = time.monotonic()

    try:
        from quantstack.data.fetcher import AlphaVantageClient
        from quantstack.data.pg_storage import PgDataStore

        client = AlphaVantageClient()
        store = PgDataStore()

        all_symbols = _get_active_symbols()
        watched = _get_watched_symbols()

        if not all_symbols:
            logger.warning("[data_refresh] No active symbols in universe")
            report.elapsed_seconds = time.monotonic() - t0
            return report

        # --- 1. Bulk quotes (up to 100 symbols, 1 API call) ---
        try:
            quotes_df = await asyncio.to_thread(
                client.fetch_bulk_quotes, all_symbols[:100]
            )
            report.api_calls += 1
            if not quotes_df.empty:
                # Store as latest quotes in ohlcv table with 1-min timeframe
                refreshed_symbols = []
                for _, row in quotes_df.iterrows():
                    sym = row.get("symbol", "")
                    if not sym:
                        continue
                    report.symbols_refreshed += 1
                    refreshed_symbols.append(sym)
                for sym in refreshed_symbols:
                    signal_cache.invalidate(sym)
                logger.info(
                    "[data_refresh] Bulk quotes: %d symbols refreshed",
                    report.symbols_refreshed,
                )
        except Exception as exc:
            report.errors.append(f"bulk_quotes: {exc}")
            logger.error("[data_refresh] Bulk quotes failed: %s", exc)

        # --- 2. 5-min OHLCV for watched symbols (positions + strategies) ---
        for sym in watched[:20]:  # Cap at 20 to stay within rate limits
            try:
                df = await asyncio.to_thread(
                    client.fetch_intraday, sym, "5min", "compact"
                )
                report.api_calls += 1
                if df is not None and not df.empty:
                    await asyncio.to_thread(
                        store.save_ohlcv, df, sym, Timeframe.M5
                    )
                    signal_cache.invalidate(sym)
                    logger.debug("[data_refresh] 5min OHLCV: %s (%d bars)", sym, len(df))
            except Exception as exc:
                report.errors.append(f"intraday_{sym}: {exc}")
                logger.warning("[data_refresh] 5min %s failed: %s", sym, exc)

        # --- 3. News sentiment (batch of 5 tickers per call) ---
        # Prioritize watched symbols, then fill from universe
        news_symbols = watched[:10] or all_symbols[:10]
        batch_size = 5
        for i in range(0, len(news_symbols), batch_size):
            batch = news_symbols[i : i + batch_size]
            try:
                tickers_str = ",".join(batch)
                news_df = await asyncio.to_thread(
                    client.fetch_news_sentiment,
                    tickers=tickers_str,
                    topics="",
                    limit=50,
                )
                report.api_calls += 1
                if news_df is not None and not news_df.empty:
                    await asyncio.to_thread(store.save_news_sentiment, news_df)
                    for sym in batch:
                        signal_cache.invalidate(sym)
                    logger.debug(
                        "[data_refresh] News: %s (%d articles)",
                        tickers_str, len(news_df),
                    )
            except Exception as exc:
                report.errors.append(f"news_{','.join(batch)}: {exc}")
                logger.warning("[data_refresh] News %s failed: %s", batch, exc)

    except ImportError as exc:
        report.errors.append(f"import: {exc}")
        logger.error("[data_refresh] Import failed: %s", exc)
    except Exception as exc:
        report.errors.append(f"unexpected: {exc}")
        logger.error("[data_refresh] Unexpected error: %s", exc)

    report.elapsed_seconds = time.monotonic() - t0
    logger.info("[data_refresh] Cache stats after invalidation: %s", signal_cache.stats())
    logger.info(
        "[data_refresh] Intraday complete: %d symbols, %d API calls, %.1fs, %d errors",
        report.symbols_refreshed, report.api_calls,
        report.elapsed_seconds, len(report.errors),
    )
    return report


async def run_eod_refresh() -> RefreshReport:
    """End-of-day data sync — daily candles, options chains, fundamentals.

    Called once after market close (16:30 ET) by the supervisor graph.

    Fetches:
      1. Daily OHLCV for full universe
      2. Options chains for watched symbols
      3. Company overview (fundamentals) for symbols not refreshed in 7 days
    """
    report = RefreshReport(mode="eod")
    t0 = time.monotonic()

    try:
        from quantstack.data.fetcher import AlphaVantageClient
        from quantstack.data.pg_storage import PgDataStore

        client = AlphaVantageClient()
        store = PgDataStore()

        all_symbols = _get_active_symbols()
        watched = _get_watched_symbols()

        if not all_symbols:
            logger.warning("[eod_refresh] No active symbols in universe")
            report.elapsed_seconds = time.monotonic() - t0
            return report

        # --- 1. Daily OHLCV for full universe ---
        for sym in all_symbols:
            try:
                df = await asyncio.to_thread(
                    client.fetch_daily, sym, "compact"
                )
                report.api_calls += 1
                if df is not None and not df.empty:
                    await asyncio.to_thread(
                        store.save_ohlcv, df, sym, Timeframe.D1
                    )
                    report.symbols_refreshed += 1
                    logger.debug("[eod_refresh] Daily: %s (%d bars)", sym, len(df))
            except Exception as exc:
                report.errors.append(f"daily_{sym}: {exc}")
                logger.warning("[eod_refresh] Daily %s failed: %s", sym, exc)

        # --- 2. Options chains for watched + strategy-aware + top universe symbols ---
        options_strat_syms = _get_options_strategy_symbols()
        options_symbols = list(dict.fromkeys(
            watched + options_strat_syms + all_symbols[:OPTIONS_REFRESH_TOP_N]
        ))
        for sym in options_symbols[:OPTIONS_REFRESH_TOP_N]:  # Cap to respect rate limits
            try:
                opts_df = await asyncio.to_thread(
                    client.fetch_realtime_options, sym
                )
                report.api_calls += 1
                if opts_df is not None and not opts_df.empty:
                    await asyncio.to_thread(
                        store.save_options_chain,
                        opts_df, sym, datetime.now(timezone.utc),
                    )
                    logger.debug("[eod_refresh] Options: %s (%d contracts)", sym, len(opts_df))
            except Exception as exc:
                report.errors.append(f"options_{sym}: {exc}")
                logger.warning("[eod_refresh] Options %s failed: %s", sym, exc)

        # --- 3. Fundamentals for stale symbols (not refreshed in 7 days) ---
        stale_symbols = _get_stale_fundamentals(all_symbols, days=7)
        for sym in stale_symbols[:10]:  # Cap at 10 per EOD cycle
            try:
                overview = await asyncio.to_thread(
                    client.fetch_company_overview, sym
                )
                report.api_calls += 1
                if overview:
                    await asyncio.to_thread(store.save_company_overview, overview)
                    logger.debug("[eod_refresh] Fundamentals: %s", sym)
            except Exception as exc:
                report.errors.append(f"fundamentals_{sym}: {exc}")
                logger.warning("[eod_refresh] Fundamentals %s failed: %s", sym, exc)

        # --- 4. Earnings calendar (1 API call, global) ---
        try:
            earnings_df = await asyncio.to_thread(
                client.fetch_earnings_calendar, None, "3month"
            )
            report.api_calls += 1
            if earnings_df is not None and not earnings_df.empty:
                await asyncio.to_thread(store.save_earnings_calendar, earnings_df)
                logger.info("[eod_refresh] Earnings calendar: %d events", len(earnings_df))
        except Exception as exc:
            report.errors.append(f"earnings_calendar: {exc}")
            logger.warning("[eod_refresh] Earnings calendar failed: %s", exc)

    except ImportError as exc:
        report.errors.append(f"import: {exc}")
        logger.error("[eod_refresh] Import failed: %s", exc)
    except Exception as exc:
        report.errors.append(f"unexpected: {exc}")
        logger.error("[eod_refresh] Unexpected error: %s", exc)

    signal_cache.clear()
    logger.info("[eod_refresh] Cache cleared. Stats: %s", signal_cache.stats())

    # --- 5. Daily loss analysis — error-driven research task generation ---
    try:
        from quantstack.learning.loss_analyzer import run_daily_loss_analysis

        loss_summary = run_daily_loss_analysis()
        logger.info("[eod_refresh] Loss analysis: %s", loss_summary)
    except Exception as exc:
        report.errors.append(f"loss_analysis: {exc}")
        logger.warning("[eod_refresh] Loss analysis failed: %s", exc)

    report.elapsed_seconds = time.monotonic() - t0
    logger.info(
        "[eod_refresh] EOD complete: %d symbols, %d API calls, %.1fs, %d errors",
        report.symbols_refreshed, report.api_calls,
        report.elapsed_seconds, len(report.errors),
    )
    return report


def _get_options_strategy_symbols() -> list[str]:
    """Return symbols from active strategies that depend on options signals."""
    try:
        with pg_conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT symbol FROM strategies
                   WHERE status IN ('paper_ready', 'forward_testing', 'live')
                     AND symbol IS NOT NULL
                     AND (
                         signals::text LIKE '%options_flow%'
                         OR signals::text LIKE '%put_call_ratio%'
                     )"""
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


async def run_corporate_actions_refresh() -> RefreshReport:
    """Daily corporate actions check — dividends, splits, M&A events.

    Called once after EOD refresh by the supervisor graph's scheduled_tasks node.
    Delegates to :func:`quantstack.data.corporate_actions.refresh_corporate_actions`.
    """
    report = RefreshReport(mode="corporate_actions")
    t0 = time.monotonic()

    try:
        from quantstack.data.corporate_actions import refresh_corporate_actions

        symbols = _get_watched_symbols() or _get_active_symbols()
        if not symbols:
            logger.warning("[corp_actions_refresh] No symbols to check")
            report.elapsed_seconds = time.monotonic() - t0
            return report

        summary = await refresh_corporate_actions(symbols)
        report.symbols_refreshed = len(symbols)
        report.errors = summary.get("errors", [])
        logger.info("[corp_actions_refresh] Summary: %s", summary)

    except ImportError as exc:
        report.errors.append(f"import: {exc}")
        logger.error("[corp_actions_refresh] Import failed: %s", exc)
    except Exception as exc:
        report.errors.append(f"unexpected: {exc}")
        logger.error("[corp_actions_refresh] Failed: %s", exc)

    report.elapsed_seconds = time.monotonic() - t0
    return report


def _get_stale_fundamentals(symbols: list[str], days: int = 7) -> list[str]:
    """Return symbols whose company_overview hasn't been refreshed in N days."""
    try:
        with pg_conn() as conn:
            rows = conn.execute(
                """SELECT symbol FROM company_overview
                   WHERE symbol = ANY(%s)
                     AND updated_at > NOW() - INTERVAL '%s days'""",
                [symbols, days],
            ).fetchall()
        fresh = {r[0] for r in rows}
        return [s for s in symbols if s not in fresh]
    except Exception:
        # Table might not exist yet — return all symbols
        return symbols
