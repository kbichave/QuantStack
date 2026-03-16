"""
PolygonStreamingAdapter — Polygon.io WebSocket minute-aggregate (AM.*) feed.

The ``AM.*`` subscription delivers one message per symbol per minute containing
the completed bar for the previous minute.  This is the lowest-latency
real-time bar source available on Polygon's Starter plan.

Polygon endpoint: wss://socket.polygon.io/stocks
Auth:  ``{"action": "auth", "params": "<POLYGON_API_KEY>"}``
Subscribe: ``{"action": "subscribe", "params": "AM.SPY,AM.AAPL"}``

Message format (agg_per_second / AM):
    {
        "ev": "AM",
        "sym": "AAPL",
        "v": 4032,           # volume
        "o": 135.22,
        "c": 135.34,
        "h": 135.40,
        "l": 135.20,
        "vw": 135.28,        # VWAP
        "s": 1715774700000,  # start time ms epoch
        "e": 1715774760000,  # end time ms epoch
        "z": 1              # accumulated volume (free tier)
    }

Note on free-tier vs paid:
    Free tier:   15-minute delayed data.
    Starter ($29/mo): Real-time data, unlimited connections.
    We use the same WebSocket endpoint for both; only the auth key changes.

Requires: ``aiohttp>=3.9.0`` (already in core dependencies).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.provider_enum import DataProvider
from quantcore.data.streaming.base import BarEvent, StreamingAdapter

_POLYGON_WS_URL = "wss://socket.polygon.io/stocks"


class PolygonStreamingAdapter(StreamingAdapter):
    """Real-time 1-minute bar stream from Polygon.io WebSocket.

    Args:
        api_key: Polygon.io API key (``POLYGON_API_KEY`` env var as fallback).
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        import os
        self._api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "POLYGON_API_KEY is required for PolygonStreamingAdapter. "
                "Add it to your .env file."
            )
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._authenticated = False

    # ── StreamingAdapter identity ─────────────────────────────────────────────

    @property
    def provider(self) -> DataProvider:
        return DataProvider.POLYGON

    @property
    def supported_timeframes(self) -> List[Timeframe]:
        return [Timeframe.M1]

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def _connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(_POLYGON_WS_URL)

        # Consume the initial "connected" message
        msg = await self._ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise ConnectionError(f"[Polygon] Unexpected WS message on connect: {msg}")

        # Authenticate
        await self._ws.send_json({"action": "auth", "params": self._api_key})
        auth_msg = await self._ws.receive_json()
        # Response: [{"ev": "status", "status": "auth_success", ...}]
        if not any(m.get("status") == "auth_success" for m in auth_msg):
            raise ConnectionError(
                f"[Polygon] Authentication failed: {auth_msg}"
            )
        self._authenticated = True

        # Start receive loop
        self._recv_task = asyncio.create_task(
            self._recv_loop(), name="polygon_stream"
        )
        logger.info("[Polygon] Streaming connection authenticated")

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
        self._ws = None
        self._session = None
        self._authenticated = False
        logger.info("[Polygon] Streaming disconnected")

    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        params = ",".join(f"AM.{s}" for s in symbols)
        await self._ws.send_json({"action": "subscribe", "params": params})
        logger.info(f"[Polygon] Subscribed to AM bars: {symbols}")

    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        params = ",".join(f"AM.{s}" for s in symbols)
        await self._ws.send_json({"action": "unsubscribe", "params": params})
        logger.info(f"[Polygon] Unsubscribed from AM bars: {symbols}")

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        """Consume incoming WebSocket messages indefinitely."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"[Polygon] WS closed/error: {msg}")
                    await self._handle_disconnect()
                    return
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[Polygon] Receive loop crashed: {exc}")
            await self._handle_disconnect()

    async def _handle_message(self, raw: str) -> None:
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            return

        for m in messages:
            if m.get("ev") == "AM":
                await self._process_am_bar(m)

    async def _process_am_bar(self, m: dict) -> None:
        # "e" is the end timestamp in milliseconds (bar close time)
        close_ms = m.get("e") or m.get("s", 0)
        ts = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc)

        event = BarEvent(
            symbol=m.get("sym", ""),
            timestamp=ts,
            open=float(m.get("o", 0)),
            high=float(m.get("h", 0)),
            low=float(m.get("l", 0)),
            close=float(m.get("c", 0)),
            volume=float(m.get("v", 0)),
            vwap=float(m["vw"]) if "vw" in m else None,
            trade_count=int(m["z"]) if "z" in m else None,
            timeframe=Timeframe.M1,
            provider=DataProvider.POLYGON.value,
        )
        await self._emit(event)
