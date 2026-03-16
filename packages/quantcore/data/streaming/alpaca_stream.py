"""
AlpacaStreamingAdapter — Alpaca StockDataStream WebSocket (alpaca-py).

Streams 1-minute bars for US equities.  Sub-minute bars (e.g. M5, M15) are
not available via Alpaca streaming; use PolygonStreamingAdapter for those.

Requires: ``alpaca-py>=0.20.0``  (``uv pip install -e ".[alpaca]"``)

Reconnect behaviour
-------------------
alpaca-py's StockDataStream handles WebSocket reconnection internally.
We wrap it in an ``asyncio.Task`` so it runs concurrently alongside the
rest of the event loop.  If the task crashes, ``_handle_disconnect`` is
called to re-create it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC

from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.provider_enum import DataProvider
from quantcore.data.streaming.base import BarEvent, StreamingAdapter

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models.bars import Bar as AlpacaBar
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


class AlpacaStreamingAdapter(StreamingAdapter):
    """Real-time 1-minute bar stream from Alpaca's WebSocket feed.

    Args:
        api_key:    Alpaca API key.
        secret_key: Alpaca secret key.
        paper:      If True, connect to paper data feed (same data, paper endpoint).
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for AlpacaStreamingAdapter. "
                "Run: uv pip install -e '.[alpaca]'"
            )
        import os
        self._api_key    = api_key    or os.getenv("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._paper      = paper
        self._stream: StockDataStream | None = None
        self._stream_task: asyncio.Task | None = None

    # ── StreamingAdapter identity ─────────────────────────────────────────────

    @property
    def provider(self) -> DataProvider:
        return DataProvider.ALPACA

    @property
    def supported_timeframes(self) -> list[Timeframe]:
        # Alpaca's real-time bar feed is 1-minute only.
        return [Timeframe.M1]

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def _connect(self) -> None:
        self._stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed="iex",  # "iex" for free; "sip" requires paid plan
        )
        logger.info("[Alpaca] Streaming connection initialised")

    async def _disconnect(self) -> None:
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self._stream:
            try:
                await self._stream.stop_ws()
            except Exception:
                pass
        self._stream = None
        self._stream_task = None
        logger.info("[Alpaca] Streaming disconnected")

    async def _subscribe_symbols(self, symbols: list[str]) -> None:
        if not self._stream:
            return
        for symbol in symbols:
            self._stream.subscribe_bars(self._on_bar, symbol)

        # Start the stream task if not already running
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(
                self._run_stream(), name="alpaca_stream"
            )
        logger.info(f"[Alpaca] Subscribed to bars: {symbols}")

    async def _unsubscribe_symbols(self, symbols: list[str]) -> None:
        if not self._stream:
            return
        for symbol in symbols:
            self._stream.unsubscribe_bars(symbol)
        logger.info(f"[Alpaca] Unsubscribed from bars: {symbols}")

    # ── Internal stream runner ────────────────────────────────────────────────

    async def _run_stream(self) -> None:
        """Run the alpaca-py WebSocket loop in the current event loop."""
        try:
            await self._stream._run_forever()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[Alpaca] Stream crashed: {exc}")
            await self._handle_disconnect()

    async def _on_bar(self, bar: AlpacaBar) -> None:
        """alpaca-py bar callback — normalise to BarEvent and emit."""
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        event = BarEvent(
            symbol=bar.symbol,
            timestamp=ts,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            vwap=float(bar.vwap) if bar.vwap is not None else None,
            trade_count=int(bar.trade_count) if bar.trade_count is not None else None,
            timeframe=Timeframe.M1,
            provider=DataProvider.ALPACA.value,
        )
        await self._emit(event)
