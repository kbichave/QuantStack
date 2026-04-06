"""
Price feed abstraction for the ExecutionMonitor.

Three implementations:
  - BarPollingFeed: wraps StreamManager's in-memory bar cache (Phase 2)
  - AlpacaQuoteFeed: sub-second quotes via Alpaca WebSocket (Phase 3)
  - PaperPriceFeed: deterministic replay for testing

The ExecutionMonitor consumes prices via PriceFeedProtocol. It never
imports a specific broker's streaming library.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from loguru import logger

PriceCallback = Callable[[str, float, datetime], Awaitable[None]]
"""Signature: async callback(symbol, price, timestamp)."""


@runtime_checkable
class PriceFeedProtocol(Protocol):
    """Broker-agnostic price feed for the ExecutionMonitor."""

    async def subscribe(self, symbols: list[str], callback: PriceCallback) -> None: ...
    async def unsubscribe(self, symbols: list[str]) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


# =============================================================================
# BarPollingFeed — wraps StreamManager's in-memory bar/feature cache
# =============================================================================


class BarPollingFeed:
    """Price feed backed by StreamManager's in-memory bar/feature cache.

    Polls at a fixed interval. No new network connections — reads from
    the LiveBarStore that the existing streaming pipeline populates.
    """

    def __init__(
        self,
        stream_manager: Any = None,
        poll_interval_s: float | None = None,
    ) -> None:
        self._stream_manager = stream_manager
        self._poll_interval = poll_interval_s or float(
            os.getenv("EXEC_MONITOR_POLL_INTERVAL", "5")
        )
        self._symbols: set[str] = set()
        self._callback: PriceCallback | None = None
        self._poll_task: asyncio.Task | None = None
        self._running = False

    async def subscribe(self, symbols: list[str], callback: PriceCallback) -> None:
        self._symbols.update(symbols)
        self._callback = callback

    async def unsubscribe(self, symbols: list[str]) -> None:
        self._symbols -= set(symbols)

    async def start(self) -> None:
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[BarPollingFeed] Poll error: {exc}")

    async def _poll_once(self) -> None:
        if not self._callback or not self._stream_manager:
            return
        for symbol in list(self._symbols):
            try:
                features = self._stream_manager.get_features(symbol)
                if features is not None:
                    await self._callback(
                        symbol, features.close, features.timestamp
                    )
            except Exception as exc:
                logger.debug(f"[BarPollingFeed] Skip {symbol}: {exc}")


# =============================================================================
# AlpacaQuoteFeed — sub-second quotes via Alpaca WebSocket
# =============================================================================


class AlpacaQuoteFeed:
    """Sub-second price feed via Alpaca WebSocket quotes.

    Taps into the existing StockDataStream managed by AlpacaStreamingAdapter.
    Adds quote subscriptions alongside the existing bar subscriptions.
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._symbols: set[str] = set()
        self._callback: PriceCallback | None = None

    async def subscribe(self, symbols: list[str], callback: PriceCallback) -> None:
        self._symbols.update(symbols)
        self._callback = callback
        stream = getattr(self._adapter, "stream", None)
        if stream is None:
            raise RuntimeError(
                "AlpacaStreamingAdapter has no active stream — "
                "ensure streaming is started before subscribing quotes"
            )
        for symbol in symbols:
            stream.subscribe_quotes(self._on_quote, symbol)

    async def unsubscribe(self, symbols: list[str]) -> None:
        stream = getattr(self._adapter, "stream", None)
        if stream:
            for symbol in symbols:
                stream.unsubscribe_quotes(symbol)
        self._symbols -= set(symbols)

    async def start(self) -> None:
        # The adapter owns connection lifecycle — just verify it's running
        stream = getattr(self._adapter, "stream", None)
        if stream is None:
            raise RuntimeError("AlpacaStreamingAdapter stream not available")

    async def stop(self) -> None:
        if self._symbols:
            await self.unsubscribe(list(self._symbols))

    async def _on_quote(self, quote: Any) -> None:
        """Alpaca quote handler — compute mid-price and forward."""
        if not self._callback:
            return
        bid = getattr(quote, "bid_price", 0) or 0
        ask = getattr(quote, "ask_price", 0) or 0
        if bid <= 0 or ask <= 0:
            return  # Skip stale/invalid quotes
        mid = (bid + ask) / 2.0
        symbol = getattr(quote, "symbol", "")
        timestamp = getattr(quote, "timestamp", datetime.now())
        if symbol in self._symbols:
            await self._callback(symbol, mid, timestamp)


# =============================================================================
# PaperPriceFeed — deterministic replay for testing
# =============================================================================


class PaperPriceFeed:
    """Deterministic price feed for testing.

    Replays a pre-defined sequence of (symbol, price, timestamp) events.
    """

    def __init__(
        self,
        events: list[tuple[str, float, datetime]],
        replay_speed: float = 0.0,  # 0 = instant
    ) -> None:
        self._events = list(events)
        self._replay_speed = replay_speed
        self._symbols: set[str] = set()
        self._callback: PriceCallback | None = None
        self._replay_task: asyncio.Task | None = None
        self._running = False

    async def subscribe(self, symbols: list[str], callback: PriceCallback) -> None:
        self._symbols.update(symbols)
        self._callback = callback

    async def unsubscribe(self, symbols: list[str]) -> None:
        self._symbols -= set(symbols)

    async def start(self) -> None:
        self._running = True
        self._replay_task = asyncio.create_task(self._replay())

    async def stop(self) -> None:
        self._running = False
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass

    async def _replay(self) -> None:
        for symbol, price, timestamp in self._events:
            if not self._running:
                break
            if symbol in self._symbols and self._callback:
                await self._callback(symbol, price, timestamp)
            if self._replay_speed > 0:
                await asyncio.sleep(self._replay_speed)


# =============================================================================
# Factory
# =============================================================================


def get_price_feed(
    stream_manager: Any = None,
    adapter: Any = None,
) -> PriceFeedProtocol:
    """Select the best available price feed based on config.

    EXEC_MONITOR_QUOTE_FEED=quotes → AlpacaQuoteFeed (requires adapter)
    EXEC_MONITOR_QUOTE_FEED=bars   → BarPollingFeed
    EXEC_MONITOR_QUOTE_FEED=auto   → AlpacaQuoteFeed if adapter available, else BarPollingFeed
    """
    mode = os.getenv("EXEC_MONITOR_QUOTE_FEED", "auto").lower()

    if mode == "quotes":
        if adapter is None:
            raise ValueError(
                "EXEC_MONITOR_QUOTE_FEED=quotes requires an AlpacaStreamingAdapter"
            )
        return AlpacaQuoteFeed(adapter)

    if mode == "bars":
        return BarPollingFeed(stream_manager)

    # auto: prefer quotes if adapter available
    if adapter is not None and getattr(adapter, "stream", None) is not None:
        return AlpacaQuoteFeed(adapter)
    return BarPollingFeed(stream_manager)
