# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Batch OHLCV cache warming for the full universe.

Pre-fetches daily OHLCV bars into the DataStore (PostgreSQL) cache so that the
AutonomousScreener can score symbols without making per-symbol API calls.

Strategy:
  - For each symbol, query ``data_metadata`` for the last cached date.
  - Fetch only the delta (last_cached_date → today) via FinancialDatasetsClient.
  - Respect the rate limiter via the client's built-in sliding-window limiter.
  - Run with configurable concurrency (default max_concurrent=10).

Timing:
  - Daily delta for 700 symbols: ~70 seconds (1 req/symbol, rate limit 1000/min).
  - Full 1-year cold start: ~5-7 minutes (pagination for long histories).
  - Runs at 06:00 ET via scheduler, before the screener at 08:00 ET.

Failure mode: individual symbol fetch failures are logged and skipped.
The screener handles missing data by excluding those symbols from scoring.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe


@dataclass
class WarmReport:
    """Summary of a cache warming operation."""

    total_symbols: int = 0
    symbols_warmed: int = 0
    symbols_skipped: int = 0
    symbols_failed: int = 0
    bars_fetched: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class CacheWarmer:
    """
    Batch OHLCV cache warming for the universe.

    Args:
        store: DataStore instance for reading/writing OHLCV cache.
        client: FinancialDatasetsClient for fetching price data.
    """

    def __init__(self, store: Any, client: Any) -> None:
        self._store = store
        self._client = client

    async def warm(
        self,
        symbols: list[str],
        lookback_days: int = 252,
        max_concurrent: int = 10,
    ) -> WarmReport:
        """
        Fetch and cache OHLCV for all symbols.  Idempotent — fetches only
        the delta since last cached date.

        Args:
            symbols: List of ticker symbols to warm.
            lookback_days: How far back to fetch on cold start (default 1 year).
            max_concurrent: Max parallel HTTP requests (default 10).
                At 1000 req/min rate limit, 10 concurrent = ~600ms budget/req.

        Returns:
            WarmReport with counts and any errors.
        """
        report = WarmReport(total_symbols=len(symbols))
        start = datetime.now(timezone.utc)

        sem = asyncio.Semaphore(max_concurrent)

        async def _warm_one(symbol: str) -> None:
            async with sem:
                try:
                    bars = await asyncio.to_thread(
                        self._fetch_delta, symbol, lookback_days
                    )
                    if bars is None:
                        report.symbols_skipped += 1
                    elif bars == 0:
                        report.symbols_skipped += 1
                    else:
                        report.symbols_warmed += 1
                        report.bars_fetched += bars
                except Exception as exc:
                    report.symbols_failed += 1
                    report.errors.append(f"{symbol}: {exc}")
                    logger.debug(f"[CacheWarmer] Failed to warm {symbol}: {exc}")

        tasks = [asyncio.create_task(_warm_one(s)) for s in symbols]
        await asyncio.gather(*tasks)

        report.elapsed_seconds = (datetime.now(timezone.utc) - start).total_seconds()

        logger.info(
            f"[CacheWarmer] Done: {report.symbols_warmed} warmed, "
            f"{report.symbols_skipped} skipped, {report.symbols_failed} failed, "
            f"{report.bars_fetched} bars in {report.elapsed_seconds:.1f}s"
        )
        return report

    def _fetch_delta(self, symbol: str, lookback_days: int) -> int | None:
        """
        Fetch OHLCV delta for a single symbol synchronously.

        Returns the number of new bars fetched, or None if no fetch was needed.
        """
        last_date = self._get_last_cached_date(symbol)
        today = date.today()

        if last_date and last_date >= today - timedelta(days=1):
            # Already up to date (allow 1-day staleness for weekends)
            return 0

        start_date = (
            last_date + timedelta(days=1)
            if last_date
            else today - timedelta(days=lookback_days)
        )

        prices = self._client.get_all_historical_prices(
            ticker=symbol,
            interval="day",
            interval_multiplier=1,
            start_date=start_date.isoformat(),
            end_date=today.isoformat(),
        )

        if not prices:
            return None

        # Store in DataStore if it has an upsert method
        self._store_prices(symbol, prices)

        return len(prices)

    def _get_last_cached_date(self, symbol: str) -> date | None:
        """Query DataStore for the most recent cached daily bar date."""
        try:
            # Try data_metadata table first (tracks coverage per symbol)
            conn = self._store._get_read_connection()
            row = conn.execute(
                """
                SELECT last_timestamp FROM data_metadata
                WHERE symbol = ? AND timeframe = 'D1'
                """,
                [symbol],
            ).fetchone()
            if row and row[0]:
                ts = row[0]
                if isinstance(ts, datetime):
                    return ts.date()
                if isinstance(ts, date):
                    return ts
                return None
        except Exception:
            pass

        # Fallback: query ohlcv table directly
        try:
            conn = self._store._get_read_connection()
            row = conn.execute(
                """
                SELECT MAX(timestamp) FROM ohlcv
                WHERE symbol = ? AND timeframe = 'D1'
                """,
                [symbol],
            ).fetchone()
            if row and row[0]:
                ts = row[0]
                if isinstance(ts, datetime):
                    return ts.date()
                if isinstance(ts, date):
                    return ts
        except Exception:
            pass

        return None

    async def warm_incremental_from_stream(
        self,
        symbols: list[str],
        stream_manager: Any,
    ) -> int:
        """
        Persist streaming bars from LiveBarStore into PostgreSQL.

        Called periodically (e.g., every 5 minutes) to flush in-memory
        streaming bars to durable storage so they survive process restarts.

        Args:
            symbols: Symbols to persist.
            stream_manager: StreamManager instance with get_bars() method.

        Returns:
            Total bars persisted across all symbols.
        """
        if stream_manager is None or not getattr(stream_manager, "is_started", False):
            return 0

        total = 0
        for symbol in symbols:
            try:
                df = stream_manager.get_bars(symbol)
                if (
                    df is not None
                    and not df.empty
                    and hasattr(self._store, "save_ohlcv")
                ):
                    rows = self._store.save_ohlcv(
                        df, symbol, Timeframe.M1, replace=False
                    )
                    total += rows
            except Exception as exc:
                logger.debug(f"[CacheWarmer] Stream persist failed for {symbol}: {exc}")

        if total > 0:
            logger.debug(f"[CacheWarmer] Persisted {total} streaming bars to PostgreSQL")
        return total

    def _store_prices(self, symbol: str, prices: list[dict[str, Any]]) -> None:
        """Store fetched price bars in the DataStore."""
        try:
            # Use DataStore's built-in OHLCV storage if available
            if hasattr(self._store, "upsert_ohlcv"):
                df = pd.DataFrame(prices)
                # Normalize column names from API response
                col_map = {
                    "time": "timestamp",
                    "date": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
                df.rename(
                    columns={k: v for k, v in col_map.items() if k in df.columns},
                    inplace=True,
                )

                required = {"timestamp", "open", "high", "low", "close", "volume"}
                if not required.issubset(df.columns):
                    logger.debug(
                        f"[CacheWarmer] Missing columns for {symbol}: "
                        f"{required - set(df.columns)}"
                    )
                    return

                df["symbol"] = symbol
                df["timeframe"] = "D1"
                self._store.upsert_ohlcv(df)
            else:
                logger.debug(
                    f"[CacheWarmer] DataStore has no upsert_ohlcv — skipping store for {symbol}"
                )
        except Exception as exc:
            logger.debug(f"[CacheWarmer] Failed to store {symbol}: {exc}")
