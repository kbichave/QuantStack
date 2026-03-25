# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tick executor — sub-second async execution hot path.

Architecture:
  - Consumes Tick events from MarketDataBus (WebSocket or polling).
  - Reads pre-computed signals from SignalCache (in-memory, no LLM, no DB).
  - Checks in-memory RiskState (no DB reads in hot path).
  - Submits orders to the broker adapter (PaperBroker or ETradeAdapter).
  - Enqueues fills to an async queue; a background writer persists to DB.

This executor is intentionally free of:
  - LLM calls
  - Database reads
  - File I/O
  - Blocking network calls in the hot path

The analysis plane (TradingDayFlow) runs concurrently and refreshes
SignalCache on its own schedule.  The executor acts on whatever the latest
valid signal is.  If the signal is stale (expired TTL), it holds.

Usage:
    executor = TickExecutor(
        signal_cache=cache,
        risk_state=risk_state,
        broker=paper_broker,
        kill_switch=kill_switch,
        fill_queue=asyncio.Queue(),
    )

    # Run alongside the analysis plane
    async with asyncio.TaskGroup() as tg:
        tg.create_task(executor.run(market_data_bus))
        tg.create_task(analysis_plane.run())
        tg.create_task(fill_writer.run(fill_queue, conn))
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from loguru import logger

from quantstack.execution.kill_switch import KillSwitch
from quantstack.execution.paper_broker import OrderRequest, PaperBroker
from quantstack.execution.risk_state import RiskState
from quantstack.execution.signal_cache import SignalCache
from quantstack.observability.metrics import record_fill, record_tick_latency

# ---------------------------------------------------------------------------
# Tick data model — canonical location is quantstack.shared.models
# ---------------------------------------------------------------------------

from quantstack.shared.models import Tick  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TickExecutor:
    """
    Async hot-path executor.

    One tick arrives → signal check → risk check → order submit →
    fill enqueued (no DB write in this path).

    The fill queue is drained by FillWriter which runs as a separate
    asyncio task and persists fills to the database after the hot path completes.
    """

    # Minimum interval between orders for the same symbol (anti-thrash)
    _MIN_ORDER_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        signal_cache: SignalCache,
        risk_state: RiskState,
        broker: PaperBroker,
        kill_switch: KillSwitch,
        fill_queue: asyncio.Queue,
        session_id: str = "",
    ):
        self._signal_cache = signal_cache
        self._risk_state = risk_state
        self._broker = broker
        self._kill_switch = kill_switch
        self._fill_queue = fill_queue
        self._session_id = session_id

        # Per-symbol last-order timestamp (anti-thrash guard)
        self._last_order_time: dict[str, float] = {}

        # Metrics counters (updated in hot path, read by Prometheus)
        self.ticks_processed: int = 0
        self.orders_submitted: int = 0
        self.orders_skipped_stale: int = 0
        self.orders_skipped_risk: int = 0
        self.orders_skipped_thrash: int = 0
        self.total_latency_ns: int = 0  # cumulative hot-path latency

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    async def run(self, tick_queue: asyncio.Queue) -> None:
        """
        Consume ticks from tick_queue until the kill switch fires or
        the queue produces a sentinel None value.

        Args:
            tick_queue: Populated by MarketDataBus.  Each item is a Tick
                        or None (sentinel = shutdown signal).
        """
        logger.info(f"[TickExecutor] Started (session={self._session_id})")

        while True:
            # Kill switch is checked in-memory — nanoseconds, no I/O
            if self._kill_switch.is_active():
                logger.critical("[TickExecutor] Kill switch active — stopping executor")
                break

            tick: Tick | None = await tick_queue.get()

            if tick is None:
                # Sentinel: clean shutdown
                logger.info("[TickExecutor] Received shutdown sentinel")
                break

            await self._process_tick(tick)
            tick_queue.task_done()

        logger.info(
            f"[TickExecutor] Stopped | ticks={self.ticks_processed} "
            f"orders={self.orders_submitted} "
            f"avg_latency={self._avg_latency_us:.1f}µs"
        )

    # -----------------------------------------------------------------------
    # Per-tick hot path
    # -----------------------------------------------------------------------

    async def _process_tick(self, tick: Tick) -> None:
        """
        Handle one tick.  The entire path below must be:
          - Non-blocking (no await on I/O)
          - In-memory only (no DB reads)
          - Sub-millisecond for common case

        The broker.execute() call IS blocking (paper: synchronous I/O;
        eTrade: async HTTP) but it's the last step, after all checks pass.
        We run it in a thread executor to avoid blocking the event loop.
        """
        start_ns = time.perf_counter_ns()
        self.ticks_processed += 1

        sym = tick.symbol.upper()

        # -- 1. Signal check (in-memory dict lookup, ~ns)
        signal = self._signal_cache.get(sym)
        if signal is None or signal.action == "HOLD":
            self.orders_skipped_stale += 1
            self._record_latency(start_ns)
            return

        # -- 2. Anti-thrash guard (per-symbol cooldown)
        last = self._last_order_time.get(sym, 0.0)
        if time.time() - last < self._MIN_ORDER_INTERVAL_SECONDS:
            self.orders_skipped_thrash += 1
            self._record_latency(start_ns)
            return

        # -- 3. Risk state check (in-memory, ~µs)
        verdict = self._risk_state.check(signal, tick.price)
        if not verdict.approved:
            self.orders_skipped_risk += 1
            logger.debug(f"[TickExecutor] {sym} risk rejected: {verdict.reason}")
            self._record_latency(start_ns)
            return

        qty = verdict.approved_quantity or 0
        if qty <= 0:
            self.orders_skipped_risk += 1
            self._record_latency(start_ns)
            return

        # -- 4. Build order request
        order = OrderRequest(
            symbol=sym,
            side=signal.action.lower(),  # "buy" or "sell"
            quantity=qty,
            order_type="market",
            current_price=tick.price,
            daily_volume=tick.volume or 1_000_000,
        )

        # -- 5. Submit to broker (run sync broker in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        fill = await loop.run_in_executor(None, self._broker.execute, order)

        if fill.rejected:
            logger.warning(f"[TickExecutor] {sym} fill rejected: {fill.reject_reason}")
            self._record_latency(start_ns)
            return

        # -- 6. Update in-memory risk state immediately (no DB)
        self._risk_state.apply_fill(
            symbol=sym,
            side=order.side,
            quantity=fill.filled_quantity,
            price=fill.fill_price,
        )

        # -- 7. Enqueue fill for async persistence (non-blocking)
        await self._fill_queue.put(fill)

        self._last_order_time[sym] = time.time()
        self.orders_submitted += 1

        latency_ns = time.perf_counter_ns() - start_ns
        latency_us = latency_ns / 1_000
        logger.info(
            f"[TickExecutor] FILLED {sym}: {order.side.upper()} {fill.filled_quantity} "
            f"@ ${fill.fill_price:.4f} | latency={latency_us:.1f}µs"
        )
        self._record_latency(start_ns)

        # Prometheus: record fill + hot-path latency
        try:
            record_fill(symbol=sym, side=order.side, speed="tick")
            record_tick_latency(latency_ns / 1e9)
        except Exception:
            pass  # Never let metrics instrumentation crash the hot path

    # -----------------------------------------------------------------------
    # Metrics helpers
    # -----------------------------------------------------------------------

    def _record_latency(self, start_ns: int) -> None:
        self.total_latency_ns += time.perf_counter_ns() - start_ns

    @property
    def _avg_latency_us(self) -> float:
        if self.ticks_processed == 0:
            return 0.0
        return (self.total_latency_ns / self.ticks_processed) / 1_000


# ---------------------------------------------------------------------------
# Fill writer — background task that drains the fill queue to the database
# ---------------------------------------------------------------------------


class FillWriter:
    """
    Async background task that persists fills to the database.

    Runs at low priority alongside the tick executor.  Fills in the queue
    are already recorded by PaperBroker (synchronously in execute()), so
    this writer is responsible for additional audit/session records.
    """

    def __init__(self, fill_queue: asyncio.Queue, session_id: str = ""):
        self._queue = fill_queue
        self._session_id = session_id
        self._fills_written: int = 0

    async def run(self) -> None:
        """Drain the fill queue until a None sentinel is received."""
        logger.info("[FillWriter] Started")
        while True:
            fill = await self._queue.get()
            if fill is None:
                logger.info(f"[FillWriter] Stopped — wrote {self._fills_written} fills")
                break
            await self._write(fill)
            self._queue.task_done()

    async def _write(self, fill) -> None:
        """Async-friendly wrapper — logging only for now; extend for audit records."""
        self._fills_written += 1
        logger.debug(
            f"[FillWriter] Persisted fill: {fill.symbol} {fill.side} "
            f"{fill.filled_quantity} @ {fill.fill_price:.4f}"
        )
