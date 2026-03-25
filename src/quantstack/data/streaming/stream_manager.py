"""
StreamManager — lifecycle orchestrator for the real-time data pipeline.

Wires the existing streaming primitives into a single startup/shutdown unit:

    AlpacaStreamingAdapter (WebSocket)
        → BarPublisher (fan-out)
            → LiveBarStore (memory + PostgreSQL persistence)
            → IncrementalFeatureEngine (O(1) rolling features)

Provides a simple interface for the SignalEngine and AutonomousRunner:

    manager = StreamManager()
    await manager.start(symbols=["SPY", "AAPL", "MSFT"])

    # Get latest features (sub-millisecond, no I/O)
    features = manager.get_features("SPY")

    # Get recent bars as DataFrame (for SignalEngine collectors)
    df = manager.get_bars("SPY", n=200)

    await manager.stop()

Design constraints:
- One manager per process (singleton pattern via module-level instance).
- Alpaca is the only streaming provider wired here (Polygon/IBKR can be
  added by swapping the adapter; the downstream pipeline is provider-agnostic).
- If Alpaca credentials are missing, the manager starts in "passive" mode:
  no WebSocket, features computed from historical REST data on demand.
- Thread-safe reads: get_features() and get_bars() are safe from any thread.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.data.streaming.base import BarEvent
from quantstack.data.streaming.incremental_features import (
    IncrementalFeatureEngine,
    IncrementalFeatures,
)
from quantstack.data.streaming.live_store import LiveBarStore
from quantstack.data.streaming.alpaca_stream import AlpacaStreamingAdapter
from quantstack.data.streaming.publisher import BarPublisher


class StreamManager:
    """
    Lifecycle orchestrator for real-time market data.

    Coordinates streaming adapter → publisher → live store → feature engine
    as a single startup/shutdown unit.
    """

    def __init__(
        self,
        db_path: str | None = None,
        bar_window: int = 500,
    ) -> None:
        """
        Args:
            db_path: Ignored — DataStore uses PostgreSQL.
                     Kept for call-site compatibility.
            bar_window: Number of bars to keep in memory per symbol.
        """
        self._db_path = db_path
        self._bar_window = bar_window

        # Components (initialized in start())
        self._adapter: Any = (
            None  # StreamingAdapter — lazy to avoid import if not needed
        )
        self._publisher: BarPublisher | None = None
        self._live_store: LiveBarStore | None = None
        self._feature_engine: IncrementalFeatureEngine | None = None
        self._store: DataStore | None = None

        self._symbols: list[str] = []
        self._started = False
        self._passive_mode = False  # True if no streaming credentials

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_passive(self) -> bool:
        """True if running without a live WebSocket (REST-only fallback)."""
        return self._passive_mode

    async def start(self, symbols: list[str]) -> None:
        """
        Start the streaming pipeline for the given symbols.

        If Alpaca credentials are not configured, starts in passive mode
        (no WebSocket, features computed from cached historical data).
        """
        if self._started:
            logger.warning("[StreamManager] Already started — call stop() first")
            return

        self._symbols = [s.upper().strip() for s in symbols]
        logger.info(f"[StreamManager] Starting for {len(self._symbols)} symbols")

        # --- Initialize components ---
        self._store = DataStore(db_path=self._db_path)
        self._publisher = BarPublisher(max_queue_depth=1000)
        self._live_store = LiveBarStore(
            store=self._store,
            window=self._bar_window,
        )
        self._feature_engine = IncrementalFeatureEngine()

        # Wire publisher → live store + feature engine
        self._publisher.subscribe("live_store")
        self._publisher.subscribe("feature_engine")

        # Start background tasks
        await self._live_store.start()

        # Launch consumer tasks for live store and feature engine
        self._store_queue = self._publisher.subscribe("live_store_direct")
        self._feature_queue = self._publisher.subscribe("feature_engine_direct")
        self._consumer_tasks = [
            asyncio.create_task(self._consume_for_store(), name="stream_manager_store"),
            asyncio.create_task(
                self._consume_for_features(), name="stream_manager_features"
            ),
        ]

        # --- Try to connect streaming adapter ---
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if api_key and secret_key:
            try:
                self._adapter = AlpacaStreamingAdapter(
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1"),
                )
                self._adapter.add_callback(self._publisher.on_bar)
                await self._adapter.subscribe(self._symbols, Timeframe.M1)
                self._passive_mode = False
                logger.info(
                    f"[StreamManager] Alpaca streaming active for "
                    f"{len(self._symbols)} symbols"
                )
            except Exception as exc:
                logger.warning(
                    f"[StreamManager] Alpaca streaming failed: {exc} — "
                    "falling back to passive mode"
                )
                self._adapter = None
                self._passive_mode = True
        else:
            logger.info(
                "[StreamManager] No Alpaca credentials — starting in passive mode"
            )
            self._passive_mode = True

        # --- Warm feature engine from historical data ---
        await self._warm_features()

        self._started = True
        logger.info(
            f"[StreamManager] Started ({'streaming' if not self._passive_mode else 'passive'} mode)"
        )

    async def stop(self) -> None:
        """Gracefully shut down the streaming pipeline."""
        if not self._started:
            return

        logger.info("[StreamManager] Stopping...")

        # Stop adapter first (stops new data flowing in)
        if self._adapter is not None:
            try:
                await self._adapter.shutdown()
            except Exception as exc:
                logger.warning(f"[StreamManager] Adapter shutdown error: {exc}")

        # Shut down publisher (sends None to subscriber queues)
        if self._publisher is not None:
            await self._publisher.shutdown()

        # Cancel consumer tasks
        for task in getattr(self, "_consumer_tasks", []):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Flush and stop live store
        if self._live_store is not None:
            await self._live_store.stop()

        self._started = False
        logger.info("[StreamManager] Stopped")

    # ── Public data access ──────────────────────────────────────────────────

    def get_features(self, symbol: str) -> IncrementalFeatures | None:
        """
        Get latest incremental features for a symbol.

        Returns None if the feature engine hasn't received any bars for this symbol.
        Sub-millisecond: pure memory read, no I/O.
        """
        if self._feature_engine is None:
            return None
        return self._feature_engine.get_latest(symbol)

    def get_bars(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """
        Get recent bars as a standard OHLCV DataFrame.

        Uses the in-memory LiveBarStore window. Falls back to PostgreSQL if
        the live store is empty (e.g., passive mode or just started).

        Args:
            symbol: Ticker symbol.
            n: Number of bars to return. None → entire window.

        Returns:
            DataFrame with DatetimeIndex and columns: open, high, low, close, volume
        """
        # Try live store first (real-time data)
        if self._live_store is not None:
            df = self._live_store.as_dataframe(symbol, n=n)
            if df is not None and not df.empty:
                return df

        # Fall back to PostgreSQL historical data
        if self._store is not None:
            df = self._store.load_ohlcv(symbol, Timeframe.D1)
            if df is not None and not df.empty:
                return df.tail(n) if n else df

        return pd.DataFrame()

    def get_all_features(self) -> dict[str, IncrementalFeatures]:
        """Get latest features for all tracked symbols."""
        if self._feature_engine is None:
            return {}
        return {
            sym: feat
            for sym in self._symbols
            if (feat := self._feature_engine.get_latest(sym)) is not None
        }

    async def add_symbols(self, symbols: list[str]) -> None:
        """Subscribe to additional symbols at runtime."""
        new_symbols = [
            s.upper().strip() for s in symbols if s.upper().strip() not in self._symbols
        ]
        if not new_symbols:
            return

        self._symbols.extend(new_symbols)

        if self._adapter is not None and not self._passive_mode:
            try:
                await self._adapter.subscribe(new_symbols, Timeframe.M1)
            except Exception as exc:
                logger.warning(
                    f"[StreamManager] Failed to subscribe {new_symbols}: {exc}"
                )

        # Warm features for new symbols
        for sym in new_symbols:
            await self._warm_symbol(sym)

    async def remove_symbols(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols at runtime."""
        to_remove = [s.upper().strip() for s in symbols]

        if self._adapter is not None and not self._passive_mode:
            try:
                await self._adapter.unsubscribe(to_remove)
            except Exception as exc:
                logger.warning(
                    f"[StreamManager] Failed to unsubscribe {to_remove}: {exc}"
                )

        self._symbols = [s for s in self._symbols if s not in to_remove]

    # ── Internal consumers ──────────────────────────────────────────────────

    async def _consume_for_store(self) -> None:
        """Drain publisher queue and feed LiveBarStore."""
        queue = self._store_queue
        while True:
            bar = await queue.get()
            if bar is None:
                break
            try:
                await self._live_store.on_bar(bar)
            except Exception as exc:
                logger.debug(f"[StreamManager] LiveBarStore error: {exc}")

    async def _consume_for_features(self) -> None:
        """Drain publisher queue and feed IncrementalFeatureEngine."""
        queue = self._feature_queue
        while True:
            bar = await queue.get()
            if bar is None:
                break
            try:
                await self._feature_engine.on_bar(bar)
            except Exception as exc:
                logger.debug(f"[StreamManager] Feature engine error: {exc}")

    # ── Warmup ──────────────────────────────────────────────────────────────

    async def _warm_features(self) -> None:
        """
        Warm the feature engine from historical data so it's ready
        immediately (not waiting for 20+ bars to arrive via stream).
        """
        for symbol in self._symbols:
            await self._warm_symbol(symbol)

    async def _warm_symbol(self, symbol: str) -> None:
        """Warm a single symbol's feature state from PostgreSQL."""
        if self._store is None or self._feature_engine is None:
            return

        try:
            df = await asyncio.to_thread(self._store.load_ohlcv, symbol, Timeframe.D1)
            if df is None or df.empty:
                return

            # Feed last 50 bars to warm up EMA/RSI/ATR state
            warmup = df.tail(50)
            for _, row in warmup.iterrows():
                bar = BarEvent(
                    symbol=symbol,
                    timestamp=(
                        row.name
                        if hasattr(row.name, "tzinfo")
                        else datetime.now(timezone.utc)
                    ),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    timeframe=Timeframe.D1,
                    provider="warmup",
                )
                await self._feature_engine.on_bar(bar)

            logger.debug(f"[StreamManager] Warmed {symbol} with {len(warmup)} bars")

        except Exception as exc:
            logger.debug(f"[StreamManager] Warmup failed for {symbol}: {exc}")


# ── Module-level singleton ──────────────────────────────────────────────────

_instance: StreamManager | None = None


def get_stream_manager(db_path: str | None = None) -> StreamManager:
    """Get or create the module-level StreamManager singleton."""
    global _instance
    if _instance is None:
        _instance = StreamManager(db_path=db_path)
    return _instance
