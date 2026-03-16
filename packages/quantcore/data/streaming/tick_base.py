"""
TickStreamingAdapter ABC — contract for real-time tick (trade/quote/L2) feeds.

Relationship to StreamingAdapter
---------------------------------
``StreamingAdapter`` (streaming/base.py) handles bar-level data and is used by
Alpaca, Polygon, and IBKR for 1-minute / 5-second bar feeds.

``TickStreamingAdapter`` handles *sub-bar* data: individual trades, quote updates,
and L2 order book changes.  It follows the same architectural patterns:

- Callbacks, not polling — three callback types (trade, quote, L2).
- Single normalised tick model — all providers normalise to TradeTick / QuoteTick /
  L2Update before calling callbacks.
- Reconnect is the adapter's responsibility — same exponential-backoff logic.

Usage
-----
    adapter = PolygonTickAdapter(api_key=...)

    async def on_trade(tick: TradeTick) -> None:
        print(tick.symbol, tick.price, tick.size)

    adapter.add_trade_callback(on_trade)
    await adapter.subscribe(["SPY", "AAPL"])
    # runs until cancelled / shutdown()
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional, Set

from loguru import logger

from quantcore.data.streaming.tick_models import L2Update, QuoteTick, TradeTick

# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

TradeCallback = Callable[[TradeTick], Awaitable[None]]
QuoteCallback = Callable[[QuoteTick], Awaitable[None]]
L2Callback    = Callable[[L2Update],  Awaitable[None]]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TickStreamingAdapter(ABC):
    """Base class for real-time tick streaming providers.

    Subclasses implement ``_connect``, ``_disconnect``, and
    ``_subscribe_symbols`` / ``_unsubscribe_symbols``.  The base handles
    callback registration, reconnect scheduling, and graceful shutdown.

    Args:
        reconnect_delay_s:      Initial backoff in seconds.
        max_reconnect_delay_s:  Maximum backoff cap.
        max_reconnect_attempts: 0 = infinite retries.
    """

    def __init__(
        self,
        reconnect_delay_s: float = 1.0,
        max_reconnect_delay_s: float = 60.0,
        max_reconnect_attempts: int = 0,
    ) -> None:
        self._trade_callbacks: List[TradeCallback] = []
        self._quote_callbacks: List[QuoteCallback] = []
        self._l2_callbacks:    List[L2Callback]    = []

        self._subscribed:  Set[str] = set()
        self._connected    = False
        self._shutdown     = False

        self._reconnect_delay     = reconnect_delay_s
        self._max_reconnect_delay = max_reconnect_delay_s
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_count     = 0

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name string (e.g. "polygon", "ibkr")."""
        ...

    # ── Connection lifecycle (subclass implements) ────────────────────────────

    @abstractmethod
    async def _connect(self) -> None:
        """Establish the WebSocket / gateway connection."""
        ...

    @abstractmethod
    async def _disconnect(self) -> None:
        """Tear down the connection cleanly."""
        ...

    @abstractmethod
    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        """Send trade + quote subscription messages for ``symbols``."""
        ...

    @abstractmethod
    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Send unsubscription messages for ``symbols``."""
        ...

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    def add_trade_callback(self, callback: TradeCallback) -> None:
        self._trade_callbacks.append(callback)

    def remove_trade_callback(self, callback: TradeCallback) -> None:
        self._trade_callbacks = [c for c in self._trade_callbacks if c is not callback]

    def add_quote_callback(self, callback: QuoteCallback) -> None:
        self._quote_callbacks.append(callback)

    def remove_quote_callback(self, callback: QuoteCallback) -> None:
        self._quote_callbacks = [c for c in self._quote_callbacks if c is not callback]

    def add_l2_callback(self, callback: L2Callback) -> None:
        self._l2_callbacks.append(callback)

    def remove_l2_callback(self, callback: L2Callback) -> None:
        self._l2_callbacks = [c for c in self._l2_callbacks if c is not callback]

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to tick events for ``symbols``.

        Connects if not already connected, then sends subscription requests.
        """
        if not self._connected:
            await self._connect_with_retry()

        new_symbols = [s for s in symbols if s not in self._subscribed]
        if new_symbols:
            await self._subscribe_symbols(new_symbols)
            self._subscribed.update(new_symbols)

    async def unsubscribe(self, symbols: List[str]) -> None:
        to_remove = [s for s in symbols if s in self._subscribed]
        if to_remove:
            await self._unsubscribe_symbols(to_remove)
            self._subscribed -= set(to_remove)

    async def shutdown(self) -> None:
        """Gracefully disconnect and stop reconnect attempts."""
        self._shutdown = True
        if self._connected:
            await self._disconnect()
        self._connected = False

    # ── Reconnect logic ───────────────────────────────────────────────────────

    async def _connect_with_retry(self) -> None:
        delay   = self._reconnect_delay
        attempt = 0

        while not self._shutdown:
            try:
                await self._connect()
                self._connected        = True
                self._reconnect_count  = 0
                return
            except Exception as exc:
                attempt += 1
                self._reconnect_count += 1
                logger.warning(
                    f"[{self.provider}] Tick connection attempt {attempt} failed: {exc}. "
                    f"Retrying in {delay:.1f}s"
                )
                if self._max_reconnect_attempts and attempt >= self._max_reconnect_attempts:
                    raise ConnectionError(
                        f"{self.provider} tick connection failed after {attempt} attempts"
                    ) from exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _handle_disconnect(self) -> None:
        self._connected = False
        if not self._shutdown:
            logger.warning(
                f"[{self.provider}] Unexpected tick disconnect. Reconnecting…"
            )
            await self._connect_with_retry()
            if self._subscribed:
                await self._subscribe_symbols(list(self._subscribed))

    # ── Dispatch helpers ──────────────────────────────────────────────────────

    async def _emit_trade(self, tick: TradeTick) -> None:
        if not self._trade_callbacks:
            return
        await asyncio.gather(
            *(cb(tick) for cb in self._trade_callbacks),
            return_exceptions=True,
        )

    async def _emit_quote(self, tick: QuoteTick) -> None:
        if not self._quote_callbacks:
            return
        await asyncio.gather(
            *(cb(tick) for cb in self._quote_callbacks),
            return_exceptions=True,
        )

    async def _emit_l2(self, update: L2Update) -> None:
        if not self._l2_callbacks:
            return
        await asyncio.gather(
            *(cb(update) for cb in self._l2_callbacks),
            return_exceptions=True,
        )
