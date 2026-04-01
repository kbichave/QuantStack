# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Alpaca WebSocket market data adapter (tick-level).

Connects to Alpaca's real-time data stream via WebSocket and converts
incoming trade/quote messages into Tick objects for the TickExecutor.

Alpaca provides two streams:
  - iex (free): IEX Exchange data, slight delay, no cost
  - sip (paid): Consolidated tape (SIP), best quality, requires paid subscription

Docs: https://docs.alpaca.markets/reference/marketdatastreaming

Prerequisites:
    pip install "websockets>=12" alpaca-py

Configuration:
    ALPACA_API_KEY  — Alpaca API key
    ALPACA_API_SECRET — Alpaca secret key
    ALPACA_DATA_FEED — "iex" (default, free) or "sip" (paid)

Usage:
    bus = AlpacaWebSocketBus(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
        symbols=["SPY", "QQQ", "AAPL"],
        feed="iex",
    )
    tick_queue = asyncio.Queue(maxsize=10_000)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(bus.run(tick_queue))
        tg.create_task(executor.run(tick_queue))
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime

from loguru import logger

from quantstack.data.market_data_bus import MarketDataBus
from quantstack.shared.models import Tick

import websockets  # type: ignore[import]

# Alpaca streaming endpoints
_FEED_URLS = {
    "iex": "wss://stream.data.alpaca.markets/v2/iex",
    "sip": "wss://stream.data.alpaca.markets/v2/sip",
    "crypto": "wss://stream.data.alpaca.markets/v1beta3/crypto/us",
}


class AlpacaWebSocketBus(MarketDataBus):
    """
    Connects to Alpaca's real-time WebSocket feed and pushes Tick objects.

    Reconnects automatically with exponential backoff on disconnect.
    Subscribes to both trades (for price) and quotes (for bid/ask spread).
    The latest trade price is used as the tick price; bid/ask are attached
    from the most recent quote for the same symbol.

    Thread-safety: designed for single asyncio task ownership.
    """

    _RECONNECT_BASE_SECONDS = 1.0
    _RECONNECT_MAX_SECONDS = 60.0

    def __init__(
        self,
        symbols: list[str],
        api_key: str | None = None,
        api_secret: str | None = None,
        feed: str = "iex",
    ):
        super().__init__(symbols)
        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self._api_secret = api_secret or os.getenv("ALPACA_API_SECRET", "")
        self._feed = feed
        self._url = _FEED_URLS.get(feed, _FEED_URLS["iex"])

        # Latest quote per symbol for bid/ask enrichment of trade ticks
        self._latest_quotes: dict[str, dict] = {}

        if not self._api_key or not self._api_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set to use "
                "AlpacaWebSocketBus. Set them as environment variables."
            )

    async def run(self, tick_queue: asyncio.Queue) -> None:
        """
        Connect to Alpaca WebSocket, receive messages, and push Ticks.

        Reconnects automatically with exponential backoff.
        """
        self._running = True
        reconnect_wait = self._RECONNECT_BASE_SECONDS

        logger.info(
            f"[AlpacaWS] Starting stream: feed={self._feed} symbols={self.symbols}"
        )

        while self._running:
            try:
                await self._connect_and_stream(tick_queue)
                reconnect_wait = (
                    self._RECONNECT_BASE_SECONDS
                )  # reset on clean disconnect
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    f"[AlpacaWS] Disconnected: {e} — reconnecting in {reconnect_wait:.0f}s"
                )
                await asyncio.sleep(reconnect_wait)
                reconnect_wait = min(reconnect_wait * 2, self._RECONNECT_MAX_SECONDS)

        await tick_queue.put(None)  # sentinel
        logger.info("[AlpacaWS] Stopped")

    async def _connect_and_stream(self, tick_queue: asyncio.Queue) -> None:
        """Single connection lifecycle."""
        async with websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            # Auth
            await ws.send(
                json.dumps(
                    {"action": "auth", "key": self._api_key, "secret": self._api_secret}
                )
            )
            auth_resp = json.loads(await ws.recv())
            self._check_auth(auth_resp)

            # Subscribe to trades and quotes for all symbols
            await ws.send(
                json.dumps(
                    {
                        "action": "subscribe",
                        "trades": self.symbols,
                        "quotes": self.symbols,
                    }
                )
            )

            logger.info(f"[AlpacaWS] Subscribed to {self.symbols}")

            async for raw in ws:
                if not self._running:
                    break

                messages = json.loads(raw)
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    tick = self._parse_message(msg)
                    if tick is not None:
                        try:
                            tick_queue.put_nowait(tick)
                        except asyncio.QueueFull:
                            logger.warning(
                                f"[AlpacaWS] Queue full — dropping tick for {tick.symbol}"
                            )

    def _check_auth(self, response: list) -> None:
        """Raise if authentication failed."""
        for msg in response if isinstance(response, list) else [response]:
            if msg.get("T") == "error":
                raise ConnectionError(f"Alpaca auth error: {msg.get('msg')}")
            if msg.get("T") == "success" and msg.get("msg") == "authenticated":
                return
        # No explicit error and no explicit success — assume ok (some versions omit)

    def _parse_message(self, msg: dict) -> Tick | None:
        """Convert an Alpaca message dict to a Tick, or None if not a trade."""
        msg_type = msg.get("T")

        if msg_type == "q":
            # Quote — cache bid/ask for enrichment; don't emit a tick
            sym = msg.get("S", "").upper()
            self._latest_quotes[sym] = msg
            return None

        if msg_type == "t":
            # Trade — emit a tick with price and volume
            sym = msg.get("S", "").upper()
            price = float(msg.get("p", 0))
            volume = int(msg.get("s", 0))
            ts_str = msg.get("t", "")

            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(UTC)

            # Enrich with latest quote if available
            quote = self._latest_quotes.get(sym, {})
            bid = float(quote["bp"]) if "bp" in quote else None
            ask = float(quote["ap"]) if "ap" in quote else None

            if price <= 0:
                return None

            return Tick(
                symbol=sym,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                timestamp=ts,
            )

        return None
