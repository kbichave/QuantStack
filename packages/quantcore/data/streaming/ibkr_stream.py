"""
IBKRStreamingAdapter — ib_insync reqRealTimeBars → 1-minute aggregation.

IB Gateway provides ``reqRealTimeBars()`` which delivers **5-second** bars.
Twelve consecutive 5-second bars are accumulated by ``_BarBuffer`` into a
single 1-minute bar that is emitted as a ``BarEvent``.

This is the only way to get real-time bars via IBKR — IB does not have a
WebSocket API for minute bars like Alpaca or Polygon.

Supported timeframes: M1 only (internally uses S5 raw feed + aggregation).

Connection management
---------------------
``IBKRStreamingAdapter`` uses the shared ``IBKRConnectionManager`` singleton
so it can co-exist with ``IBKRDataAdapter`` and ``ibkr_mcp`` on the same
IB Gateway socket without opening multiple connections.

Requires: ``ib_insync>=0.9.86``  (``uv pip install -e ".[ibkr]"``)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime

from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.provider_enum import DataProvider
from quantcore.data.streaming.base import BarEvent, StreamingAdapter

try:
    import ib_insync as ib

    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


def _require_ibkr() -> None:
    if not _IB_AVAILABLE:
        raise ImportError(
            "ib_insync is required for IBKRStreamingAdapter. Run: uv pip install -e '.[ibkr]'"
        )


# ---------------------------------------------------------------------------
# 5s → 1m bar aggregator
# ---------------------------------------------------------------------------


class _BarBuffer:
    """Accumulates twelve 5-second bars into one 1-minute bar.

    A new minute begins when ``bar.time`` crosses a minute boundary.
    The completed bar is returned from ``add()``; None while accumulating.
    """

    def __init__(self) -> None:
        self._bars: list = []
        self._minute: int | None = None  # minute epoch (floor of ts to minute)

    def add(self, bar) -> BarEvent | None:
        """Add a 5-second bar.  Returns a completed BarEvent or None."""
        # bar.time is a datetime from ib_insync
        ts: datetime = bar.time
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        minute_floor = ts.replace(second=0, microsecond=0)
        minute_epoch = int(minute_floor.timestamp())

        if self._minute is None:
            self._minute = minute_epoch

        if minute_epoch != self._minute:
            # Minute boundary crossed — emit completed bar
            completed = self._build(
                datetime.fromtimestamp(self._minute, tz=UTC)
                + __import__("datetime").timedelta(minutes=1)
            )
            # Reset accumulator with the new bar
            self._bars = [bar]
            self._minute = minute_epoch
            return completed

        self._bars.append(bar)

        # Also emit on the 12th bar (exactly 60s = 12 × 5s)
        if len(self._bars) == 12:
            completed = self._build(ts)
            self._bars = []
            self._minute = None
            return completed

        return None

    def _build(self, close_time: datetime) -> BarEvent | None:
        if not self._bars:
            return None
        return BarEvent(
            symbol="",  # set by caller
            timestamp=close_time,
            open=float(self._bars[0].open),
            high=max(float(b.high) for b in self._bars),
            low=min(float(b.low) for b in self._bars),
            close=float(self._bars[-1].close),
            volume=sum(float(b.volume) for b in self._bars),
            vwap=None,  # IB 5s bars do not include VWAP
            trade_count=None,
            timeframe=Timeframe.M1,
            provider=DataProvider.IBKR.value,
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class IBKRStreamingAdapter(StreamingAdapter):
    """Real-time 1-minute bar stream via IB Gateway reqRealTimeBars.

    Args:
        host:      IB Gateway host (default 127.0.0.1).
        port:      Gateway port (4001=IB Gateway, 7497=TWS).
        client_id: Unique integer per connection.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4001,
        client_id: int = 2,  # Use a different ID from IBKRDataAdapter (1)
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        _require_ibkr()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib: ib.IB | None = None
        self._contracts: dict[str, ib.Contract] = {}
        self._tickers: dict[str, object] = {}  # symbol → ib Ticker
        self._buffers: dict[str, _BarBuffer] = defaultdict(_BarBuffer)

    # ── StreamingAdapter identity ─────────────────────────────────────────────

    @property
    def provider(self) -> DataProvider:
        return DataProvider.IBKR

    @property
    def supported_timeframes(self) -> list[Timeframe]:
        return [Timeframe.M1, Timeframe.S5]

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def _connect(self) -> None:
        self._ib = ib.IB()
        await self._ib.connectAsync(
            host=self._host,
            port=self._port,
            clientId=self._client_id,
        )
        # Register disconnect handler
        self._ib.disconnectedEvent += self._on_disconnect
        logger.info(
            f"[IBKR] Streaming connected to {self._host}:{self._port} (clientId={self._client_id})"
        )

    async def _disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            # Cancel all real-time bar subscriptions
            for ticker in self._tickers.values():
                self._ib.cancelRealTimeBars(ticker)
            self._ib.disconnect()
        self._ib = None
        self._tickers.clear()
        self._contracts.clear()
        logger.info("[IBKR] Streaming disconnected")

    async def _subscribe_symbols(self, symbols: list[str]) -> None:
        for symbol in symbols:
            contract = ib.Stock(symbol, "SMART", "USD")
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.warning(f"[IBKR] Could not qualify contract for {symbol}, skipping")
                continue
            contract = qualified[0]
            self._contracts[symbol] = contract

            # reqRealTimeBars with barSize=5 always gives 5-second bars
            ticker = self._ib.reqRealTimeBars(contract, 5, "TRADES", False)
            # Register per-bar callback
            ticker.updateEvent += lambda t, s=symbol: self._on_rt_bar(t, s)
            self._tickers[symbol] = ticker

        logger.info(f"[IBKR] Subscribed to real-time bars: {symbols}")

    async def _unsubscribe_symbols(self, symbols: list[str]) -> None:
        for symbol in symbols:
            if symbol in self._tickers:
                self._ib.cancelRealTimeBars(self._tickers.pop(symbol))
                self._contracts.pop(symbol, None)
                self._buffers.pop(symbol, None)
        logger.info(f"[IBKR] Unsubscribed from real-time bars: {symbols}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_rt_bar(self, ticker, symbol: str) -> None:
        """ib_insync event callback — runs in ib_insync's thread."""
        if not ticker.realtimeBars:
            return
        bar = ticker.realtimeBars[-1]  # latest 5s bar

        if self._timeframe == Timeframe.S5:
            ts = bar.time
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            event = BarEvent(
                symbol=symbol,
                timestamp=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                timeframe=Timeframe.S5,
                provider=DataProvider.IBKR.value,
            )
        else:
            # M1: accumulate 12×5s → 1m
            event = self._buffers[symbol].add(bar)
            if event is None:
                return
            event.symbol = symbol

        # Schedule emission on the event loop (ib_insync callback is not async)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self._emit(event), loop)

    def _on_disconnect(self) -> None:
        """Called by ib_insync when the gateway connection drops."""
        self._connected = False
        if not self._shutdown:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self._handle_disconnect(), loop)
