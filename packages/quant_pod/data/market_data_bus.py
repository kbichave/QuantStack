# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MarketDataBus — feed abstraction for the tick executor.

Decouples the executor from the data source so the same hot-path code
runs whether we're polling a REST API (minute-level) or streaming from
a WebSocket feed (tick-level).

Two operating modes:
  - REST polling: polls for quotes every N seconds.  Suitable for minute-level
    decisions.  Works with Alpha Vantage, eTrade quotes, or any sync provider.
    No extra dependencies.
  - WebSocket streaming: receives real-time ticks over a persistent WebSocket.
    Suitable for sub-second execution.  Requires a streaming data provider
    (Alpaca, Polygon.io, IEX Cloud).  See adapters/alpaca_ws.py.

The bus puts Tick objects onto an asyncio.Queue.  The TickExecutor reads
from that queue.  The bus manages connection lifecycle; the executor never
sees the underlying transport.

Usage — REST polling (minute-level):
    from quant_pod.data.market_data_bus import RestPollingBus

    bus = RestPollingBus(
        symbols=["SPY", "QQQ"],
        quote_fn=etrade_adapter.get_quote,  # Any sync callable
        poll_interval_seconds=30,
    )
    tick_queue = asyncio.Queue(maxsize=1000)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(bus.run(tick_queue))
        tg.create_task(executor.run(tick_queue))

Usage — WebSocket streaming (tick-level):
    from quant_pod.data.adapters.alpaca_ws import AlpacaWebSocketBus

    bus = AlpacaWebSocketBus(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        symbols=["SPY", "QQQ"],
    )
    # Same interface — just swap the bus
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from loguru import logger

from quant_pod.execution.tick_executor import Tick


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class MarketDataBus(ABC):
    """
    Base class for all market data feeds.

    Subclasses implement run() to continuously put Tick objects onto
    the tick_queue.  A None sentinel on the queue signals clean shutdown.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = [s.upper() for s in symbols]
        self._running = False

    @abstractmethod
    async def run(self, tick_queue: asyncio.Queue) -> None:
        """
        Push Tick events onto tick_queue until stopped.

        Implementors should:
        1. Set self._running = True on entry
        2. Put ticks onto tick_queue (non-blocking, drop if full)
        3. Honour stop() — check self._running each iteration
        4. Put None sentinel on tick_queue before returning
        """

    def stop(self) -> None:
        """Signal the bus to stop after the current operation completes."""
        self._running = False


# ---------------------------------------------------------------------------
# REST polling implementation (minute-level)
# ---------------------------------------------------------------------------


class RestPollingBus(MarketDataBus):
    """
    Polls a synchronous quote function on a fixed interval.

    Suitable for minute-level decisions.  Each poll cycle fetches all
    symbols and emits one Tick per symbol.

    Args:
        symbols: List of ticker symbols to poll
        quote_fn: Sync callable(symbol: str) → dict with keys:
                  "price" (float), "volume" (int), "bid" (float), "ask" (float)
                  Missing keys are treated as 0.
        poll_interval_seconds: How often to poll (default: 30s)
        error_sleep_seconds: How long to sleep after a failed poll (default: 5s)
    """

    def __init__(
        self,
        symbols: List[str],
        quote_fn: Callable[[str], dict],
        poll_interval_seconds: float = 30.0,
        error_sleep_seconds: float = 5.0,
    ):
        super().__init__(symbols)
        self._quote_fn = quote_fn
        self._interval = poll_interval_seconds
        self._error_sleep = error_sleep_seconds

    async def run(self, tick_queue: asyncio.Queue) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        logger.info(
            f"[RestPollingBus] Started: {self.symbols} "
            f"@ {self._interval:.0f}s interval"
        )

        while self._running:
            for symbol in self.symbols:
                if not self._running:
                    break
                try:
                    # Run sync quote function in thread pool to avoid blocking
                    data = await loop.run_in_executor(None, self._quote_fn, symbol)

                    tick = Tick(
                        symbol=symbol,
                        price=float(data.get("price", 0.0)),
                        volume=int(data.get("volume", 0)),
                        bid=data.get("bid"),
                        ask=data.get("ask"),
                    )

                    if tick.price <= 0:
                        logger.warning(
                            f"[RestPollingBus] {symbol}: invalid price {tick.price}, skipping"
                        )
                        continue

                    # Non-blocking put — drop if queue is full (backpressure)
                    try:
                        tick_queue.put_nowait(tick)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"[RestPollingBus] Tick queue full — dropping {symbol}"
                        )

                except Exception as e:
                    logger.warning(
                        f"[RestPollingBus] Failed to fetch {symbol}: {e}"
                    )
                    await asyncio.sleep(self._error_sleep)

            await asyncio.sleep(self._interval)

        # Sentinel: signal executor to shut down.
        # Must not block — if the queue is full, evict one stale tick to make
        # room rather than deadlocking here while nobody is draining the queue.
        try:
            tick_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                tick_queue.get_nowait()  # evict one stale tick
            except asyncio.QueueEmpty:
                pass
            try:
                tick_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass  # executor must handle missing sentinel via _running flag
        logger.info("[RestPollingBus] Stopped")
