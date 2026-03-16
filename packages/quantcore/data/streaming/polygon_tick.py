"""
PolygonTickAdapter — Polygon.io WebSocket trade (T.*) and quote (Q.*) feed.

Polygon WebSocket endpoint: wss://socket.polygon.io/stocks
Auth: {\"action\": \"auth\", \"params\": \"<POLYGON_API_KEY>\"}

Subscribe:
    {\"action\": \"subscribe\", \"params\": \"T.SPY,Q.SPY,T.AAPL,Q.AAPL\"}

Trade message (ev = \"T\"):
    {
        \"ev\": \"T\",
        \"sym\": \"AAPL\",
        \"i\":  \"55383\",        # trade ID
        \"x\":  4,               # exchange ID (see Polygon conditions API)
        \"p\":  135.22,          # price
        \"s\":  100,             # size
        \"t\":  1715774700123,   # timestamp ms epoch
        \"c\":  [14, 41],        # conditions
        \"z\":  1                # tape (1=NYSE, 2=AMEX, 3=NASDAQ)
    }

Quote message (ev = \"Q\"):
    {
        \"ev\": \"Q\",
        \"sym\": \"AAPL\",
        \"bx\":  4,              # bid exchange
        \"ax\":  11,             # ask exchange
        \"bp\":  135.20,         # bid price
        \"ap\":  135.25,         # ask price
        \"bs\":  1,              # bid size (lots of 100)
        \"as\":  2,              # ask size
        \"t\":   1715774700456,  # timestamp ms
        \"c\":   [1]             # conditions
    }

Aggressor side is inferred from the tick rule (Lee-Ready):
    price > mid_at_time → "buy"
    price < mid_at_time → "sell"
    price == mid        → "unknown"

We maintain the last-seen quote per symbol to compute the mid for tick-rule
classification.  This avoids the lookup overhead of a separate quote store.

Requires: aiohttp (already in core dependencies).
"""

from __future__ import annotations

import asyncio
import json
from datetime import timezone
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from quantcore.data.streaming.tick_base import TickStreamingAdapter
from quantcore.data.streaming.tick_models import L2Update, QuoteTick, TradeTick

_POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

# Polygon exchange ID → human-readable code (partial map for logging)
_EXCHANGE_MAP: Dict[int, str] = {
    1: "A", 2: "B", 4: "C", 7: "D", 8: "E", 10: "F",
    11: "G", 12: "H", 13: "I", 14: "J", 15: "K", 16: "L",
    17: "M", 18: "N", 20: "P", 21: "Q", 22: "S", 23: "T",
    24: "V", 25: "W", 26: "X", 27: "Y", 28: "Z",
}


class PolygonTickAdapter(TickStreamingAdapter):
    """Real-time trade and NBBO quote stream from Polygon.io WebSocket.

    L2 (order-book level) data is not available on Polygon's Starter plan;
    ``add_l2_callback`` can be registered but will never fire unless a plan
    with Level 2 data is active.

    Args:
        api_key: Polygon.io API key (``POLYGON_API_KEY`` env var as fallback).
        subscribe_quotes: If True (default), subscribe to Q.* quote events.
        subscribe_trades: If True (default), subscribe to T.* trade events.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        subscribe_quotes: bool = True,
        subscribe_trades: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        import os
        self._api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "POLYGON_API_KEY is required for PolygonTickAdapter. "
                "Add it to your .env file."
            )
        self._subscribe_quotes = subscribe_quotes
        self._subscribe_trades = subscribe_trades
        self._ws:      Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession]            = None
        self._recv_task: Optional[asyncio.Task]                   = None

        # last quote per symbol — used for Lee-Ready tick-rule classification
        self._last_quote: Dict[str, QuoteTick] = {}

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def provider(self) -> str:
        return "polygon"

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def _connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(_POLYGON_WS_URL)

        # Consume the initial "connected" message
        msg = await self._ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise ConnectionError(f"[Polygon Tick] Unexpected WS message: {msg}")

        # Authenticate
        await self._ws.send_json({"action": "auth", "params": self._api_key})
        auth_msg = await self._ws.receive_json()
        if not any(m.get("status") == "auth_success" for m in auth_msg):
            raise ConnectionError(f"[Polygon Tick] Auth failed: {auth_msg}")

        # Start receive loop
        self._recv_task = asyncio.create_task(
            self._recv_loop(), name="polygon_tick_recv"
        )
        logger.info("[Polygon Tick] Authenticated")

    async def _disconnect(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._ws = self._session = None
        logger.info("[Polygon Tick] Disconnected")

    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        params_parts = []
        if self._subscribe_trades:
            params_parts += [f"T.{s}" for s in symbols]
        if self._subscribe_quotes:
            params_parts += [f"Q.{s}" for s in symbols]
        if params_parts:
            await self._ws.send_json(
                {"action": "subscribe", "params": ",".join(params_parts)}
            )
        logger.info(f"[Polygon Tick] Subscribed to ticks: {symbols}")

    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        params_parts = []
        if self._subscribe_trades:
            params_parts += [f"T.{s}" for s in symbols]
        if self._subscribe_quotes:
            params_parts += [f"Q.{s}" for s in symbols]
        if params_parts:
            await self._ws.send_json(
                {"action": "unsubscribe", "params": ",".join(params_parts)}
            )
        logger.info(f"[Polygon Tick] Unsubscribed from ticks: {symbols}")

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"[Polygon Tick] WS closed/error: {msg}")
                    await self._handle_disconnect()
                    return
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[Polygon Tick] Receive loop crashed: {exc}")
            await self._handle_disconnect()

    async def _handle_message(self, raw: str) -> None:
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            return
        for m in messages:
            ev = m.get("ev")
            if ev == "T":
                await self._process_trade(m)
            elif ev == "Q":
                await self._process_quote(m)

    # ── Message processors ────────────────────────────────────────────────────

    async def _process_trade(self, m: dict) -> None:
        symbol = m.get("sym", "")
        ts_ms  = m.get("t", 0)
        ts_ns  = ts_ms * 1_000_000  # ms → ns

        price = float(m.get("p", 0))
        size  = float(m.get("s", 0))
        exch  = _EXCHANGE_MAP.get(m.get("x"), str(m.get("x", "")))

        # Lee-Ready tick rule for aggressor side
        side = self._classify_side(symbol, price)

        tick = TradeTick(
            symbol       = symbol,
            timestamp_ns = ts_ns,
            price        = price,
            size         = size,
            side         = side,
            exchange     = exch,
            trade_id     = str(m["i"]) if "i" in m else None,
            conditions   = m.get("c"),
        )
        await self._emit_trade(tick)

    async def _process_quote(self, m: dict) -> None:
        symbol = m.get("sym", "")
        ts_ms  = m.get("t", 0)
        ts_ns  = ts_ms * 1_000_000

        bid   = float(m.get("bp", 0))
        ask   = float(m.get("ap", 0))
        bsize = float(m.get("bs", 0)) * 100  # Polygon reports in round lots
        asize = float(m.get("as", 0)) * 100
        bexch = _EXCHANGE_MAP.get(m.get("bx"), None)
        aexch = _EXCHANGE_MAP.get(m.get("ax"), None)

        tick = QuoteTick(
            symbol       = symbol,
            timestamp_ns = ts_ns,
            bid          = bid,
            ask          = ask,
            bid_size     = bsize,
            ask_size     = asize,
            bid_exchange = bexch,
            ask_exchange = aexch,
        )
        self._last_quote[symbol] = tick
        await self._emit_quote(tick)

    # ── Tick rule ─────────────────────────────────────────────────────────────

    def _classify_side(self, symbol: str, price: float) -> str:
        """Lee-Ready tick rule: classify trade as buy/sell using last NBBO."""
        q = self._last_quote.get(symbol)
        if q is None or q.bid == 0.0 or q.ask == 0.0:
            return "unknown"
        mid = q.mid
        if price > mid:
            return "buy"
        if price < mid:
            return "sell"
        return "unknown"
