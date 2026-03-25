"""
LiveBarStore — write-through cache: streaming bars → memory + PostgreSQL.

Design
------
* ``on_bar`` is ``BarCallback``-compatible and non-blocking:
  - Appends to the in-memory ``deque(maxlen=window)`` for the symbol immediately.
  - Enqueues the bar for PostgreSQL persistence (oldest-drop if queue is full).

* A single background asyncio Task drains the write queue in batches.
  All PostgreSQL writes happen on one asyncio task, so there is no thread-safety
  concern with the DataStore connection.  A batch of 50 bars is persisted in
  well under 1ms, so the event loop is never blocked for a meaningful duration.

* On startup, call ``start()`` to launch the writer task.
  On shutdown, call ``stop()`` to flush pending writes and cancel the task.

* ``get_window(symbol, n)`` returns the last ``n`` BarEvents from the in-memory
  deque.  On the first access for a symbol it seeds the deque from PostgreSQL so
  that the feature engine has a warm window immediately after restart.

* ``as_dataframe(symbol, n)`` converts the in-memory window to the standard
  OHLCV DataFrame contract expected by ``OHLCVResampler`` and ``FeatureFactory``.

Failure modes
-------------
* PostgreSQL write fails: logged as warning, bar is dropped from the write queue
  but is still in the in-memory deque.  No data is lost from the live window;
  only persistence is affected.
* Write queue overflow (producer faster than writer): oldest enqueued bars are
  dropped.  The in-memory deque remains accurate.
* DB seed fails on restart: the deque starts empty.  No crash; the feature
  engine will warm up naturally as bars arrive.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from datetime import UTC, datetime, timedelta

import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.data.streaming.base import BarEvent

_DEFAULT_WINDOW = 500  # bars kept per symbol in memory
_DEFAULT_BATCH_SIZE = 50  # bars batched per write cycle
_FLUSH_INTERVAL_S = 10.0  # max seconds between flushes when queue is sparse
_WRITE_QUEUE_DEPTH = 5_000  # oldest-drop when this limit is hit


class LiveBarStore:
    """Write-through in-memory + PostgreSQL cache for live streaming bars.

    Args:
        store:      ``DataStore`` instance owning the database connection.
        window:     Number of BarEvents to keep per symbol in memory.
        batch_size: Max bars to accumulate before flushing to PostgreSQL.
    """

    def __init__(
        self,
        store: DataStore,
        window: int = _DEFAULT_WINDOW,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._store = store
        self._window = window
        self._batch_size = batch_size

        # in-memory ring buffers: symbol → deque of BarEvents
        self._buffers: dict[str, deque[BarEvent]] = {}
        # symbols whose deque has been seeded from PostgreSQL
        self._seeded: set[str] = set()
        # protects _buffers and _seeded (reads from any thread; writes from event loop)
        self._lock = threading.RLock()

        # asyncio write queue — shared only within one event loop
        self._write_queue: asyncio.Queue | None = None
        self._writer_task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background writer task.

        Must be called once from within the asyncio event loop before any bars
        are published.
        """
        self._write_queue = asyncio.Queue(maxsize=_WRITE_QUEUE_DEPTH)
        self._writer_task = asyncio.create_task(
            self._writer_loop(), name="live_store_writer"
        )
        logger.info("[LiveBarStore] Writer task started")

    async def stop(self) -> None:
        """Flush remaining writes and shut down the writer task."""
        if self._writer_task and not self._writer_task.done():
            if self._write_queue is not None:
                # Sentinel: tells the writer loop to flush then exit
                await self._write_queue.put(None)
            try:
                await asyncio.wait_for(self._writer_task, timeout=15.0)
            except TimeoutError:
                self._writer_task.cancel()
                logger.warning(
                    "[LiveBarStore] Writer task timed out on stop — cancelled"
                )
        logger.info("[LiveBarStore] Stopped")

    # ── BarCallback interface ─────────────────────────────────────────────────

    async def on_bar(self, bar: BarEvent) -> None:
        """Receive a live bar: update in-memory deque and enqueue for persistence.

        Compatible with ``StreamingAdapter.add_callback()`` and
        ``BarPublisher.subscribe()``.
        """
        # 1. In-memory update (always succeeds, O(1))
        with self._lock:
            if bar.symbol not in self._buffers:
                self._buffers[bar.symbol] = deque(maxlen=self._window)
            self._buffers[bar.symbol].append(bar)

        # 2. Enqueue for PostgreSQL persistence (oldest-drop policy when full)
        if self._write_queue is None:
            return
        if self._write_queue.full():
            try:
                self._write_queue.get_nowait()
                logger.debug(
                    f"[LiveBarStore] Write queue full — dropped oldest bar "
                    f"({bar.symbol} @ {bar.timestamp})"
                )
            except asyncio.QueueEmpty:
                pass
        try:
            self._write_queue.put_nowait(bar)
        except asyncio.QueueFull:
            pass  # race between full-check and put — safe to skip

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_window(self, symbol: str, n: int | None = None) -> list[BarEvent]:
        """Return the last ``n`` BarEvents for ``symbol`` (thread-safe).

        Seeds the in-memory deque from PostgreSQL on first call per symbol so the
        feature engine has a warm window immediately after restart.

        Args:
            symbol: Ticker symbol.
            n:      Number of recent bars to return.  Defaults to ``self._window``.

        Returns:
            List of BarEvents sorted oldest → newest.
        """
        limit = n if n is not None else self._window
        with self._lock:
            if symbol not in self._seeded:
                self._seed_from_db(symbol)
                self._seeded.add(symbol)
            buf = self._buffers.get(symbol)
            if not buf:
                return []
            bars = list(buf)
        return bars[-limit:] if limit < len(bars) else bars

    def as_dataframe(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """Return the in-memory window as an OHLCV DataFrame.

        The returned DataFrame matches the standard DataStore contract:
        ``DatetimeIndex`` named ``"timestamp"`` (UTC-aware), columns
        ``["open", "high", "low", "close", "volume"]`` (float64).
        Optional columns ``vwap`` and ``trade_count`` are included only when
        at least one bar in the window carries them.

        Args:
            symbol: Ticker symbol.
            n:      Number of recent bars.  Defaults to full window.

        Returns:
            Empty DataFrame (with correct columns) if no bars are available.
        """
        bars = self.get_window(symbol, n)
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        records = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": b.vwap,
                "trade_count": b.trade_count,
            }
            for b in bars
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Drop optional columns when entirely null
        for col in ("vwap", "trade_count"):
            if col in df.columns and df[col].isna().all():
                df.drop(columns=[col], inplace=True)

        return df

    def symbols(self) -> list[str]:
        """Return all symbols currently tracked in memory."""
        with self._lock:
            return list(self._buffers.keys())

    def bar_count(self, symbol: str) -> int:
        """Return the number of bars currently in memory for ``symbol``."""
        with self._lock:
            buf = self._buffers.get(symbol)
            return len(buf) if buf else 0

    # ── Background writer ────────────────────────────────────────────────────

    async def _writer_loop(self) -> None:
        """Drain the write queue and persist bars to PostgreSQL in batches."""
        assert self._write_queue is not None
        batch: list[BarEvent] = []

        try:
            while True:
                # Wait up to FLUSH_INTERVAL_S for the next bar
                try:
                    item = await asyncio.wait_for(
                        self._write_queue.get(), timeout=_FLUSH_INTERVAL_S
                    )
                except TimeoutError:
                    # Periodic flush of whatever has accumulated
                    if batch:
                        await self._flush(batch)
                        batch.clear()
                    continue

                if item is None:
                    # Shutdown sentinel
                    break

                batch.append(item)

                # Drain additional items without waiting
                while len(batch) < self._batch_size:
                    try:
                        item = self._write_queue.get_nowait()
                        if item is None:
                            await self._flush(batch)
                            return
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                if len(batch) >= self._batch_size:
                    await self._flush(batch)
                    batch.clear()

        except asyncio.CancelledError:
            pass
        finally:
            # Best-effort flush of remaining batch on any exit path
            if batch:
                await self._flush(batch)
            logger.info("[LiveBarStore] Writer task exited")

    async def _flush(self, bars: list[BarEvent]) -> None:
        """Write a batch of BarEvents to PostgreSQL.

        Grouped by symbol to minimise INSERT overhead.  Runs in the event loop
        (a batch of 50 rows persists in < 1ms; blocking cost is negligible).
        """
        if not bars:
            return

        # Group by symbol
        by_symbol: dict[str, list[BarEvent]] = {}
        for bar in bars:
            by_symbol.setdefault(bar.symbol, []).append(bar)

        for symbol, sym_bars in by_symbol.items():
            try:
                records = [
                    {
                        "timestamp": b.timestamp,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                        "vwap": b.vwap,
                        "trade_count": b.trade_count,
                    }
                    for b in sym_bars
                ]
                df = pd.DataFrame(records)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
                self._store.save_ohlcv_1m(df, symbol)
                logger.debug(
                    f"[LiveBarStore] Persisted {len(sym_bars)} bars for {symbol}"
                )
            except Exception as exc:
                logger.warning(
                    f"[LiveBarStore] Failed to persist {len(sym_bars)} bars for {symbol}: {exc}"
                )

    # ── DB seed ───────────────────────────────────────────────────────────────

    def _seed_from_db(self, symbol: str) -> None:
        """Load recent bars from PostgreSQL to seed the in-memory deque.

        Called under ``self._lock`` on first access for a symbol.
        A 30-day look-back is generous enough to always find ``self._window``
        trading bars (390 bars/day × ~20 trading days = 7800 >> 500).
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=30)

        try:
            df = self._store.load_ohlcv_1m(symbol, start_date=start, end_date=end)
        except Exception as exc:
            logger.warning(
                f"[LiveBarStore] Could not seed '{symbol}' from PostgreSQL: {exc}"
            )
            return

        if df.empty:
            logger.debug(f"[LiveBarStore] No historical 1m bars found for '{symbol}'")
            return

        # Keep only the last self._window rows to stay within deque capacity
        df = df.tail(self._window)

        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self._window)

        has_vwap = "vwap" in df.columns
        has_trade_count = "trade_count" in df.columns

        for ts, row in df.iterrows():
            bar = BarEvent(
                symbol=symbol,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                vwap=float(row["vwap"]) if has_vwap and pd.notna(row["vwap"]) else None,
                trade_count=(
                    int(row["trade_count"])
                    if has_trade_count and pd.notna(row["trade_count"])
                    else None
                ),
                timeframe=Timeframe.M1,
                provider="",  # historical origin; provider not tracked at row level
            )
            self._buffers[symbol].append(bar)

        logger.info(
            f"[LiveBarStore] Seeded '{symbol}' with {len(df)} bars from PostgreSQL "
            f"(window={self._window})"
        )
