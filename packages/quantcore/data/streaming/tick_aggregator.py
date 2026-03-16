"""
TickAggregator — converts raw TradeTick stream into BarEvents.

Purpose
-------
When a TickStreamingAdapter is the data source (rather than a bar-streaming
adapter like PolygonStreamingAdapter), downstream consumers (BarPublisher,
LiveBarStore, IncrementalFeatureEngine) still expect BarEvents.
``TickAggregator`` bridges the two worlds: it acts as a ``TradeCallback``
registered with a ``TickStreamingAdapter`` and emits ``BarEvent``s as
``BarCallback``s to the rest of the bar pipeline.

Aggregation model
-----------------
Time-based: bars close on fixed calendar boundaries determined by the
configured ``Timeframe`` (e.g. every minute for M1, every 5 seconds for S5).

Boundary detection uses the bar's *close* time convention:
  A M1 bar starting at 09:30:00 closes at 09:31:00.  Any trade with
  timestamp 09:31:00.000... or later triggers emission of the 09:30 bar and
  opens the 09:31 bar.

Partial bars
------------
If no trade arrives in a bar period the bar is skipped entirely (no zero-
volume phantom bars).  This matches the convention used by Alpaca and Polygon
streaming adapters.

VWAP
----
Computed from accumulated dollar-volume and total volume within each bar:
    vwap = Σ(price × size) / Σ(size)

Usage
-----
    tick_adapter = PolygonTickAdapter(api_key=...)
    aggregator   = TickAggregator(timeframe=Timeframe.M1)

    # Pipe trades into aggregator
    tick_adapter.add_trade_callback(aggregator.on_trade)

    # Pipe completed bars into bar pipeline
    bar_publisher = BarPublisher()
    aggregator.add_callback(bar_publisher.on_bar)

    await tick_adapter.subscribe(["SPY"])
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from quantcore.config.timeframes import Timeframe
from quantcore.data.provider_enum import DataProvider
from quantcore.data.streaming.base import BarCallback, BarEvent
from quantcore.data.streaming.tick_models import TradeTick

# Seconds per timeframe bar (used for boundary arithmetic)
_TF_SECONDS: dict[Timeframe, int] = {
    Timeframe.S5:  5,
    Timeframe.M1:  60,
    Timeframe.M5:  300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1:  3600,
    Timeframe.H4:  14400,
    Timeframe.D1:  86400,
}


class _BarAccumulator:
    """Mutable state for one in-progress bar."""

    __slots__ = (
        "symbol", "bar_start_s", "open", "high", "low", "close",
        "volume", "dollar_volume", "trade_count",
    )

    def __init__(self, symbol: str, bar_start_s: int, first_trade: TradeTick) -> None:
        self.symbol       = symbol
        self.bar_start_s  = bar_start_s     # seconds epoch of bar open
        p, s              = first_trade.price, first_trade.size
        self.open         = p
        self.high         = p
        self.low          = p
        self.close        = p
        self.volume       = s
        self.dollar_volume = p * s
        self.trade_count  = 1

    def update(self, price: float, size: float) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close         = price
        self.volume       += size
        self.dollar_volume += price * size
        self.trade_count  += 1

    def to_bar_event(
        self, close_ts: datetime, timeframe: Timeframe, provider: str
    ) -> BarEvent:
        vwap = self.dollar_volume / self.volume if self.volume > 0 else None
        return BarEvent(
            symbol      = self.symbol,
            timestamp   = close_ts,          # close time (convention)
            open        = self.open,
            high        = self.high,
            low         = self.low,
            close       = self.close,
            volume      = self.volume,
            vwap        = vwap,
            trade_count = self.trade_count,
            timeframe   = timeframe,
            provider    = provider,
        )


class TickAggregator:
    """Aggregates TradeTick stream into BarEvents at a fixed timeframe.

    Args:
        timeframe:  Target bar timeframe (must be in ``_TF_SECONDS``).
        provider:   Provider label for emitted BarEvents.
    """

    def __init__(
        self,
        timeframe: Timeframe = Timeframe.M1,
        provider: str = DataProvider.POLYGON.value,
    ) -> None:
        if timeframe not in _TF_SECONDS:
            raise ValueError(
                f"TickAggregator does not support {timeframe}. "
                f"Supported: {list(_TF_SECONDS.keys())}"
            )
        self._tf        = timeframe
        self._bar_s     = _TF_SECONDS[timeframe]
        self._provider  = provider

        # symbol → in-progress accumulator
        self._accumulators: dict[str, _BarAccumulator] = {}

        # bar callbacks (compatible with BarPublisher.on_bar and LiveBarStore.on_bar)
        self._callbacks: list[BarCallback] = []

    # ── Callback registration ─────────────────────────────────────────────────

    def add_callback(self, callback: BarCallback) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: BarCallback) -> None:
        self._callbacks = [c for c in self._callbacks if c is not callback]

    # ── TradeCallback interface ───────────────────────────────────────────────

    async def on_trade(self, tick: TradeTick) -> None:
        """Receive a TradeTick and update or close the current bar."""
        ts_s      = tick.timestamp_ns // 1_000_000_000
        bar_start = (ts_s // self._bar_s) * self._bar_s  # floor to bar boundary
        symbol    = tick.symbol

        acc = self._accumulators.get(symbol)

        if acc is None:
            # First trade for this symbol
            self._accumulators[symbol] = _BarAccumulator(symbol, bar_start, tick)
            return

        if bar_start > acc.bar_start_s:
            # Crossed a bar boundary — emit the completed bar
            bar_close_ts = datetime.fromtimestamp(
                acc.bar_start_s + self._bar_s, tz=UTC
            )
            bar = acc.to_bar_event(bar_close_ts, self._tf, self._provider)
            await self._emit(bar)

            # Handle the case where multiple bar boundaries have been crossed
            # (e.g. gap in trade data).  Emit intermediate empty bars only if
            # needed; here we simply start a fresh accumulator.
            self._accumulators[symbol] = _BarAccumulator(symbol, bar_start, tick)
        else:
            acc.update(tick.price, tick.size)

    async def flush(self, symbol: str | None = None) -> None:
        """Emit all in-progress bars (e.g. at market close).

        Args:
            symbol: Flush only this symbol.  Flushes all symbols if None.
        """
        symbols = [symbol] if symbol else list(self._accumulators.keys())
        for sym in symbols:
            acc = self._accumulators.pop(sym, None)
            if acc and acc.volume > 0:
                bar_close_ts = datetime.fromtimestamp(
                    acc.bar_start_s + self._bar_s, tz=UTC
                )
                bar = acc.to_bar_event(bar_close_ts, self._tf, self._provider)
                await self._emit(bar)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _emit(self, bar: BarEvent) -> None:
        if not self._callbacks:
            return
        await asyncio.gather(
            *(cb(bar) for cb in self._callbacks),
            return_exceptions=True,
        )
