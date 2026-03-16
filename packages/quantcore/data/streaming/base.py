"""
StreamingAdapter ABC — contract for real-time bar WebSocket feeds.

Design principles
-----------------
1. Callbacks, not polling.  Adapters call ``on_bar`` as bars arrive.
   Callers never block waiting for new data.

2. One bar type.  All adapters normalise to ``BarEvent`` before calling
   the callback.  Callers are provider-agnostic.

3. Reconnect is the adapter's responsibility.  The adapter must
   attempt re-connection with exponential backoff so the caller
   doesn't need retry logic.

4. Timeframe is fixed per subscription.  Alpaca streams 1-minute bars;
   IBKR streams 5-second bars that the adapter aggregates to 1-minute.
   If you need multiple granularities, register multiple adapters.

BarEvent contract
-----------------
- ``symbol`` : str     — ticker symbol
- ``timestamp`` : datetime (UTC) — bar close time
- ``open/high/low/close/volume`` : float
- ``vwap`` : Optional[float]  — included when provider supplies it
- ``trade_count`` : Optional[int]

Usage
-----
    adapter = AlpacaStreamingAdapter(api_key=..., secret_key=...)

    async def handle_bar(bar: BarEvent) -> None:
        print(bar.symbol, bar.close)

    adapter.add_callback(handle_bar)
    await adapter.subscribe(["SPY", "AAPL"], Timeframe.M1)
    # runs until cancelled
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from quantcore.config.timeframes import Timeframe
from quantcore.data.provider_enum import DataProvider

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BarEvent:
    """A completed OHLCV bar received from a real-time stream.

    Timestamp is the bar *close* time (i.e. a 09:31 bar covers 09:30–09:31).
    """

    symbol: str
    timestamp: datetime  # UTC bar close time
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None
    timeframe: Timeframe = Timeframe.M1
    provider: str = ""


# ---------------------------------------------------------------------------
# Callback type alias
# ---------------------------------------------------------------------------

BarCallback = Callable[[BarEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StreamingAdapter(ABC):
    """Base class for real-time bar streaming providers.

    Subclasses implement ``_connect``, ``_subscribe_symbols``,
    ``_unsubscribe_symbols``, and ``_disconnect``.  The base class handles
    callback registration, reconnect scheduling, and graceful shutdown.

    Args:
        reconnect_delay_s: Initial backoff (seconds) after a disconnect.
                           Doubles on each consecutive failure, capped at
                           ``max_reconnect_delay_s``.
        max_reconnect_delay_s: Maximum backoff cap.
        max_reconnect_attempts: Give up after this many consecutive failures
                                (0 = infinite).
    """

    def __init__(
        self,
        reconnect_delay_s: float = 1.0,
        max_reconnect_delay_s: float = 60.0,
        max_reconnect_attempts: int = 0,
    ) -> None:
        self._callbacks: list[BarCallback] = []
        self._subscribed: set[str] = set()
        self._timeframe: Timeframe | None = None
        self._connected = False
        self._reconnect_delay = reconnect_delay_s
        self._max_reconnect_delay = max_reconnect_delay_s
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_count = 0
        self._shutdown = False

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider(self) -> DataProvider:
        """Which streaming provider this adapter wraps."""
        ...

    @property
    @abstractmethod
    def supported_timeframes(self) -> list[Timeframe]:
        """Timeframes this adapter can stream natively."""
        ...

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def _connect(self) -> None:
        """Establish the WebSocket / gateway connection."""
        ...

    @abstractmethod
    async def _disconnect(self) -> None:
        """Tear down the connection cleanly."""
        ...

    @abstractmethod
    async def _subscribe_symbols(self, symbols: list[str]) -> None:
        """Send subscription messages for ``symbols``."""
        ...

    @abstractmethod
    async def _unsubscribe_symbols(self, symbols: list[str]) -> None:
        """Send unsubscription messages for ``symbols``."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Public API ────────────────────────────────────────────────────────────

    def add_callback(self, callback: BarCallback) -> None:
        """Register an async callback to receive BarEvents."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: BarCallback) -> None:
        self._callbacks = [c for c in self._callbacks if c is not callback]

    async def subscribe(self, symbols: list[str], timeframe: Timeframe) -> None:
        """Subscribe to bar events for ``symbols`` at ``timeframe``.

        Connects if not already connected, then sends subscription requests.

        Raises:
            ValueError: If ``timeframe`` is not in ``supported_timeframes``.
        """
        if timeframe not in self.supported_timeframes:
            raise ValueError(
                f"{self.__class__.__name__} does not support {timeframe}. "
                f"Supported: {[tf.name for tf in self.supported_timeframes]}"
            )
        self._timeframe = timeframe

        if not self._connected:
            await self._connect_with_retry()

        new_symbols = [s for s in symbols if s not in self._subscribed]
        if new_symbols:
            await self._subscribe_symbols(new_symbols)
            self._subscribed.update(new_symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from bar events for ``symbols``."""
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
        """Connect with exponential backoff.

        Retries indefinitely (or up to ``max_reconnect_attempts`` if set).
        """
        delay = self._reconnect_delay
        attempt = 0

        while not self._shutdown:
            try:
                await self._connect()
                self._connected = True
                self._reconnect_count = 0
                return
            except Exception as exc:
                attempt += 1
                self._reconnect_count += 1
                from loguru import logger

                logger.warning(
                    f"[{self.provider.value}] Connection attempt {attempt} failed: {exc}. "
                    f"Retrying in {delay:.1f}s"
                )
                if self._max_reconnect_attempts and attempt >= self._max_reconnect_attempts:
                    raise ConnectionError(
                        f"{self.provider.value} streaming connection failed after "
                        f"{attempt} attempts"
                    ) from exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _handle_disconnect(self) -> None:
        """Called by subclasses when an unexpected disconnect is detected."""
        self._connected = False
        if not self._shutdown:
            from loguru import logger

            logger.warning(f"[{self.provider.value}] Unexpected disconnect. Attempting reconnect…")
            await self._connect_with_retry()
            # Re-subscribe to all previously subscribed symbols
            if self._subscribed and self._timeframe:
                await self._subscribe_symbols(list(self._subscribed))

    # ── Callback dispatch ─────────────────────────────────────────────────────

    async def _emit(self, bar: BarEvent) -> None:
        """Dispatch ``bar`` to all registered callbacks concurrently."""
        if not self._callbacks:
            return
        await asyncio.gather(
            *(cb(bar) for cb in self._callbacks),
            return_exceptions=True,  # one slow callback must not block others
        )
