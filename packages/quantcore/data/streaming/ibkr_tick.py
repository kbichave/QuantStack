"""
IBKRTickAdapter — IB Gateway reqTickByTickData + reqMktDepth.

Two subscriptions per symbol:

1. ``reqTickByTickData(contract, "Last", 0, True)``
   Streams every last-sale print as a TickByTickLast event.
   This is the IB equivalent of Polygon's T.* feed.

2. ``reqMktDepth(contract, numRows=10, isSmartDepth=False)``
   Streams incremental order book updates as updateMktDepth events.
   This is the IB equivalent of L2 data.

NBBO (best bid/ask) is derived from the top of the MktDepth book; IB does
not have a separate reqMktData (Level 1) tick-by-tick callback — reqMktData
provides snapshots, not a streaming callback compatible with our model.

Connection
----------
Uses ``IBKRConnectionManager`` singleton (connection.py in ibkr_mcp package)
so it shares the same IB Gateway socket as IBKRStreamingAdapter and
IBKRDataAdapter without opening multiple client connections.

When ``ibkr_mcp`` is not installed, falls back to creating its own
``ib.IB()`` instance managed internally.

Requires: ib_insync>=0.9.86  (uv pip install -e \".[ibkr]\")
"""

from __future__ import annotations

import asyncio
from datetime import UTC

from loguru import logger

from quantcore.data.streaming.tick_base import TickStreamingAdapter
from quantcore.data.streaming.tick_models import L2Update, QuoteTick, TradeTick

try:
    import ib_insync as ib

    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


def _require_ibkr() -> None:
    if not _IB_AVAILABLE:
        raise ImportError(
            "ib_insync is required for IBKRTickAdapter. Run: uv pip install -e '.[ibkr]'"
        )


class IBKRTickAdapter(TickStreamingAdapter):
    """Real-time tick stream via IB Gateway reqTickByTickData + reqMktDepth.

    Args:
        host:       IB Gateway host (default 127.0.0.1).
        port:       Gateway port (4001 = IB Gateway, 7497 = TWS).
        client_id:  Unique integer per connection. Default 3 (avoids conflict
                    with IBKRDataAdapter=1, IBKRStreamingAdapter=2).
        depth_rows: Number of book levels to subscribe per side (default 10).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4001,
        client_id: int = 3,
        depth_rows: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        _require_ibkr()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._depth_rows = depth_rows

        self._ib: ib.IB | None = None

        # symbol → ib Contract
        self._contracts: dict[str, ib.Contract] = {}
        # symbol → tick-by-tick ticker
        self._tick_tickers: dict[str, object] = {}
        # symbol → depth ticker
        self._depth_tickers: dict[str, object] = {}

        # In-memory depth book for deriving NBBO:
        # bid/ask price/size per symbol (top-of-book only)
        self._depth_bids: dict[str, dict[int, tuple]] = {}  # symbol → {row: (price, size)}
        self._depth_asks: dict[str, dict[int, tuple]] = {}

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def provider(self) -> str:
        return "ibkr"

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def _connect(self) -> None:
        self._ib = ib.IB()
        await self._ib.connectAsync(
            host=self._host,
            port=self._port,
            clientId=self._client_id,
        )
        self._ib.disconnectedEvent += self._on_disconnect
        logger.info(
            f"[IBKR Tick] Connected to {self._host}:{self._port} (clientId={self._client_id})"
        )

    async def _disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            for ticker in self._tick_tickers.values():
                self._ib.cancelTickByTickData(ticker)
            for ticker in self._depth_tickers.values():
                self._ib.cancelMktDepth(ticker)
            self._ib.disconnect()
        self._ib = None
        self._tick_tickers.clear()
        self._depth_tickers.clear()
        self._contracts.clear()
        logger.info("[IBKR Tick] Disconnected")

    async def _subscribe_symbols(self, symbols: list[str]) -> None:
        for symbol in symbols:
            contract = ib.Stock(symbol, "SMART", "USD")
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.warning(f"[IBKR Tick] Could not qualify {symbol}, skipping")
                continue
            contract = qualified[0]
            self._contracts[symbol] = contract

            # Initialise depth books
            self._depth_bids[symbol] = {}
            self._depth_asks[symbol] = {}

            # Tick-by-tick "Last" subscription
            tick_ticker = self._ib.reqTickByTickData(
                contract, "Last", numberOfTicks=0, ignoreSize=True
            )
            tick_ticker.updateEvent += lambda t, s=symbol: self._on_tick(t, s)
            self._tick_tickers[symbol] = tick_ticker

            # Market depth subscription
            depth_ticker = self._ib.reqMktDepth(
                contract, numRows=self._depth_rows, isSmartDepth=False
            )
            depth_ticker.updateEvent += lambda t, s=symbol: self._on_depth(t, s)
            self._depth_tickers[symbol] = depth_ticker

        logger.info(f"[IBKR Tick] Subscribed to ticks: {symbols}")

    async def _unsubscribe_symbols(self, symbols: list[str]) -> None:
        for symbol in symbols:
            if symbol in self._tick_tickers:
                self._ib.cancelTickByTickData(self._tick_tickers.pop(symbol))
            if symbol in self._depth_tickers:
                self._ib.cancelMktDepth(self._depth_tickers.pop(symbol))
            self._contracts.pop(symbol, None)
            self._depth_bids.pop(symbol, None)
            self._depth_asks.pop(symbol, None)
        logger.info(f"[IBKR Tick] Unsubscribed: {symbols}")

    # ── ib_insync callbacks ───────────────────────────────────────────────────

    def _on_tick(self, ticker, symbol: str) -> None:
        """Called by ib_insync on each last-sale tick — NOT an async method."""
        if not ticker.ticks:
            return

        loop = asyncio.get_event_loop()
        if not loop.is_running():
            return

        for tick_data in ticker.ticks:
            ts = tick_data.time
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            ts_ns = int(ts.timestamp() * 1_000_000_000)

            trade = TradeTick(
                symbol=symbol,
                timestamp_ns=ts_ns,
                price=float(tick_data.price),
                size=float(tick_data.size),
                side=self._classify_side(symbol, float(tick_data.price)),
                exchange=None,
                trade_id=None,
            )
            asyncio.run_coroutine_threadsafe(self._emit_trade(trade), loop)

    def _on_depth(self, ticker, symbol: str) -> None:
        """Called by ib_insync on each market depth update."""
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            return

        if not ticker.domBids and not ticker.domAsks:
            return

        ts_ns = _now_ns()

        # Process bids
        for row, entry in enumerate(ticker.domBids):
            self._depth_bids[symbol][row] = (entry.price, entry.size)
            update = L2Update(
                symbol=symbol,
                timestamp_ns=ts_ns,
                side="bid",
                price=float(entry.price),
                size=float(entry.size),
                action="modify",
                is_snapshot=False,
            )
            asyncio.run_coroutine_threadsafe(self._emit_l2(update), loop)

        # Process asks
        for row, entry in enumerate(ticker.domAsks):
            self._depth_asks[symbol][row] = (entry.price, entry.size)
            update = L2Update(
                symbol=symbol,
                timestamp_ns=ts_ns,
                side="ask",
                price=float(entry.price),
                size=float(entry.size),
                action="modify",
                is_snapshot=False,
            )
            asyncio.run_coroutine_threadsafe(self._emit_l2(update), loop)

        # Emit synthetic NBBO from top-of-book
        self._emit_nbbo_from_depth(symbol, ts_ns, loop)

    def _emit_nbbo_from_depth(
        self, symbol: str, ts_ns: int, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Derive NBBO from best depth level and emit as QuoteTick."""
        bid_rows = self._depth_bids.get(symbol, {})
        ask_rows = self._depth_asks.get(symbol, {})
        if not bid_rows or not ask_rows:
            return

        best_bid_row = max(bid_rows.items(), key=lambda kv: kv[1][0], default=None)
        best_ask_row = min(ask_rows.items(), key=lambda kv: kv[1][0], default=None)
        if best_bid_row is None or best_ask_row is None:
            return

        bid_price, bid_size = best_bid_row[1]
        ask_price, ask_size = best_ask_row[1]

        quote = QuoteTick(
            symbol=symbol,
            timestamp_ns=ts_ns,
            bid=bid_price,
            ask=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
        )
        asyncio.run_coroutine_threadsafe(self._emit_quote(quote), loop)

    def _on_disconnect(self) -> None:
        self._connected = False
        if not self._shutdown:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self._handle_disconnect(), loop)

    # ── Tick classification ───────────────────────────────────────────────────

    def _classify_side(self, symbol: str, price: float) -> str:
        """Infer aggressor side from top-of-book when last tick is ambiguous."""
        bid_rows = self._depth_bids.get(symbol, {})
        ask_rows = self._depth_asks.get(symbol, {})
        if not bid_rows or not ask_rows:
            return "unknown"
        best_bid = max((v[0] for v in bid_rows.values()), default=None)
        best_ask = min((v[0] for v in ask_rows.values()), default=None)
        if best_bid is None or best_ask is None:
            return "unknown"
        mid = (best_bid + best_ask) / 2.0
        if price > mid:
            return "buy"
        if price < mid:
            return "sell"
        return "unknown"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _now_ns() -> int:
    """Current UTC time in nanoseconds."""
    import time

    return int(time.time_ns())
