# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Polygon.io data provider.

Why Polygon over Alpha Vantage for production:
  - Unlimited API calls ($29/month Starter plan)
  - 15-minute delayed quotes (free tier) or real-time (Starter+)
  - Clean REST API, well-maintained Python client
  - Used by institutional players; reliable uptime SLA

Requires:
    POLYGON_API_KEY environment variable

Usage:
    provider = PolygonProvider()
    bars = provider.get_bars("SPY", interval="1d", limit=252)
    quote = provider.get_quote("AAPL")
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import duckdb
import requests
from loguru import logger

from quantcore.data.provider import Bar, DataProvider, Quote, SymbolInfo
from quantcore.data.validator import DataValidator


# =============================================================================
# BAR CACHE
# =============================================================================


class _BarCache:
    """
    DuckDB-backed cache for bar responses with per-interval TTLs.

    Why: Polygon Starter = unlimited calls, but redundant fetches during a
    single trading session waste latency and count against rate limits.
    Daily bars don't change during the session; intraday bars only need
    refreshing every 15–30 minutes.

    TTLs:
      - daily / weekly / monthly: 6 hours (safe through a full session)
      - hourly / 4h:              30 minutes
      - minute bars (1m, 5m, 15m, 30m): 5 minutes
    """

    _TTL_SECONDS: Dict[str, int] = {
        "1d": 6 * 3600,
        "1w": 6 * 3600,
        "1mo": 6 * 3600,
        "1h": 30 * 60,
        "4h": 30 * 60,
        "1m": 5 * 60,
        "5m": 5 * 60,
        "15m": 5 * 60,
        "30m": 5 * 60,
    }
    _DEFAULT_TTL = 15 * 60  # 15 min for anything not listed

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.getenv("POLYGON_CACHE_DB_PATH", "~/.quant_pod/polygon_cache.duckdb")
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._init_schema()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self._db_path))
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS bar_cache (
                    cache_key   VARCHAR PRIMARY KEY,
                    bars_json   TEXT NOT NULL,
                    fetched_at  TIMESTAMP NOT NULL,
                    ttl_seconds INTEGER NOT NULL
                )
            """)

    def _cache_key(self, symbol: str, interval: str, start: date, end: date, limit: int) -> str:
        return f"{symbol}|{interval}|{start}|{end}|{limit}"

    def get(self, symbol: str, interval: str, start: date, end: date, limit: int) -> Optional[List[Bar]]:
        """Return cached bars if present and not expired, else None."""
        key = self._cache_key(symbol, interval, start, end, limit)
        with self._lock:
            row = self.conn.execute(
                "SELECT bars_json, fetched_at, ttl_seconds FROM bar_cache WHERE cache_key = ?",
                [key],
            ).fetchone()
        if row is None:
            return None
        bars_json, fetched_at, ttl_seconds = row
        age = (datetime.now() - fetched_at).total_seconds()
        if age > ttl_seconds:
            return None  # Expired
        try:
            raw = json.loads(bars_json)
            return [Bar(**b) for b in raw]
        except Exception:
            return None

    def set(self, symbol: str, interval: str, start: date, end: date, limit: int, bars: List[Bar]) -> None:
        """Store bars in the cache."""
        key = self._cache_key(symbol, interval, start, end, limit)
        ttl = self._TTL_SECONDS.get(interval, self._DEFAULT_TTL)
        bars_json = json.dumps([b.model_dump() for b in bars], default=str)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO bar_cache (cache_key, bars_json, fetched_at, ttl_seconds)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (cache_key) DO UPDATE SET
                    bars_json = excluded.bars_json,
                    fetched_at = excluded.fetched_at,
                    ttl_seconds = excluded.ttl_seconds
                """,
                [key, bars_json, datetime.now(), ttl],
            )


# =============================================================================
# POLYGON PROVIDER
# =============================================================================


class PolygonProvider(DataProvider):
    """
    Polygon.io REST API v2/v3 implementation.

    Rate limits:
      - Free tier: 5 calls/min (same as Alpha Vantage — do not use free)
      - Starter ($29/month): unlimited calls, 15-min delayed quotes
      - Developer ($79/month): unlimited, real-time

    Caches bar responses locally in DuckDB to avoid redundant calls.
    """

    BASE_URL = "https://api.polygon.io"

    # Interval → Polygon multiplier/timespan
    _INTERVAL_MAP = {
        "1m": ("1", "minute"),
        "5m": ("5", "minute"),
        "15m": ("15", "minute"),
        "30m": ("30", "minute"),
        "1h": ("1", "hour"),
        "4h": ("4", "hour"),
        "1d": ("1", "day"),
        "1w": ("1", "week"),
        "1mo": ("1", "month"),
    }

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self._api_key:
            raise ValueError(
                "POLYGON_API_KEY not set. Get a key at https://polygon.io"
            )
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})
        self._validator = DataValidator()
        self._cache = _BarCache()
        logger.info("PolygonProvider initialized")

    @property
    def name(self) -> str:
        return "polygon"

    # -------------------------------------------------------------------------
    # Bars
    # -------------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 252,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[Bar]:
        """Fetch OHLCV bars from Polygon /v2/aggs endpoint, with local DuckDB cache."""
        multiplier, timespan = self._INTERVAL_MAP.get(interval, ("1", "day"))
        end_date = end or date.today()
        if start is None:
            if timespan == "day":
                start_date = end_date - timedelta(days=int(limit * 1.4))
            elif timespan == "week":
                start_date = end_date - timedelta(weeks=limit)
            elif timespan == "month":
                start_date = end_date - timedelta(days=limit * 31)
            else:
                start_date = end_date - timedelta(days=30)
        else:
            start_date = start

        # Cache check — avoids redundant API calls within TTL window
        cached = self._cache.get(symbol, interval, start_date, end_date, limit)
        if cached is not None:
            logger.debug(f"[POLYGON] {symbol} {interval}: cache hit ({len(cached)} bars)")
            return cached

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol.upper()}/range"
            f"/{multiplier}/{timespan}"
            f"/{start_date.isoformat()}/{end_date.isoformat()}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": min(limit, 50000),
        }

        try:
            resp = self._get(url, params)
        except Exception as e:
            logger.error(f"[POLYGON] get_bars({symbol}) failed: {e}")
            return []

        results = resp.get("results") or []
        bars = []
        for r in results[-limit:]:
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.utcfromtimestamp(r["t"] / 1000),
                open=r["o"],
                high=r["h"],
                low=r["l"],
                close=r["c"],
                volume=int(r.get("v", 0)),
                vwap=r.get("vw"),
                interval=interval,
            )
            bars.append(bar)

        # Validate before returning
        valid_bars = self._validator.validate_bars(bars)
        logger.debug(f"[POLYGON] {symbol} {interval}: {len(valid_bars)}/{len(bars)} bars valid (API fetch)")

        # Store in cache for subsequent calls within TTL
        if valid_bars:
            self._cache.set(symbol, interval, start_date, end_date, limit, valid_bars)

        return valid_bars

    # -------------------------------------------------------------------------
    # Quote
    # -------------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        """Fetch latest quote from Polygon /v2/last/trade endpoint."""
        # Snapshot is most current
        url = f"{self.BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}"
        try:
            resp = self._get(url)
            ticker = resp.get("ticker") or {}
            day = ticker.get("day") or {}
            last_trade = ticker.get("lastTrade") or {}
            last_quote = ticker.get("lastQuote") or {}

            price = (
                last_trade.get("p")
                or day.get("c")
                or 0.0
            )
            return Quote(
                symbol=symbol,
                price=float(price),
                bid=last_quote.get("p"),
                ask=last_quote.get("P"),
                volume=int(day.get("v", 0)),
                timestamp=datetime.now(),
                delayed=True,  # Starter plan = 15-min delayed
            )
        except Exception as e:
            logger.error(f"[POLYGON] get_quote({symbol}) failed: {e}")
            # Fallback: last bar close
            bars = self.get_bars(symbol, interval="1d", limit=1)
            if bars:
                b = bars[-1]
                return Quote(
                    symbol=symbol,
                    price=b.close,
                    volume=b.volume,
                    timestamp=b.timestamp,
                    delayed=True,
                )
            raise

    # -------------------------------------------------------------------------
    # Symbol info
    # -------------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetch ticker details from Polygon /v3/reference/tickers."""
        url = f"{self.BASE_URL}/v3/reference/tickers/{symbol.upper()}"
        try:
            resp = self._get(url)
            result = resp.get("results") or {}
            return SymbolInfo(
                symbol=symbol,
                name=result.get("name", symbol),
                exchange=result.get("primary_exchange"),
                market_cap=result.get("market_cap"),
                currency=result.get("currency_name", "USD"),
            )
        except Exception as e:
            logger.warning(f"[POLYGON] get_symbol_info({symbol}) failed: {e}")
            return SymbolInfo(symbol=symbol, name=symbol)

    # -------------------------------------------------------------------------
    # HTTP helper
    # -------------------------------------------------------------------------

    def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """GET with retry and rate-limit handling."""
        for attempt in range(retries):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"[POLYGON] Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise
        raise RuntimeError(f"[POLYGON] All {retries} retries exhausted for {url}")


# =============================================================================
# WEBSOCKET STREAMING CLIENT
# =============================================================================


class PolygonStreamClient:
    """
    Real-time streaming client for Polygon.io WebSocket API.

    Streams minute-bar aggregates and trades into the local DuckDB cache
    so the rest of the system can read intraday bars without changing its
    interface — `PolygonProvider.get_bars()` will serve the cached data.

    Subscription types:
        AM.*  — per-minute OHLCV aggregates (most useful for strategy signals)
        A.*   — per-second aggregates (higher resolution, higher volume)
        T.*   — individual trades (tick-level)
        Q.*   — NBBO quotes

    Requires:
        pip install websockets   (already in most modern Python envs)

    Plan tier:
        Starter ($29/mo) — real-time; free tier does NOT support WebSocket.

    Usage::

        client = PolygonStreamClient(symbols=["SPY", "AAPL", "QQQ"])
        client.subscribe(on_bar=my_callback)   # optional live callback
        await client.connect()                 # blocks until stop() called

    Or run as a background thread::

        client = PolygonStreamClient(symbols=["SPY", "AAPL"])
        client.start()          # non-blocking — spawns daemon thread
        ...
        client.stop()
    """

    _WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(
        self,
        symbols: List[str],
        api_key: Optional[str] = None,
        subscription: str = "AM",  # AM=minute bars, A=second bars, T=trades, Q=quotes
        on_bar_callback=None,      # Optional[Callable[[Bar], None]]
        cache: Optional[_BarCache] = None,
    ):
        """
        Args:
            symbols: List of tickers to subscribe to.
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var).
            subscription: Message type prefix — "AM" (minute agg), "T" (trade), "Q" (quote).
            on_bar_callback: Optional function called with each Bar as it arrives.
            cache: DuckDB bar cache to write into (shared with PolygonProvider).
        """
        self._api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self._api_key:
            raise ValueError("POLYGON_API_KEY not set — WebSocket streaming requires a paid plan")

        self._symbols = [s.upper() for s in symbols]
        self._sub_prefix = subscription.upper()
        self._on_bar = on_bar_callback
        self._cache = cache or _BarCache()
        self._stop_event: Optional[Any] = None  # threading.Event, set in start()
        self._thread: Optional[Any] = None
        self._connected = False
        self._reconnect_delay = 5  # seconds, doubles on repeated failures
        self._max_reconnect_delay = 60

        # In-memory ring buffer of latest bars per symbol (most recent 500 per symbol)
        self._live_bars: Dict[str, List[Bar]] = {s: [] for s in self._symbols}
        self._bars_lock = Lock()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start streaming in a background daemon thread."""
        import threading
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="polygon-stream",
        )
        self._thread.start()
        logger.info(f"[POLYGON_WS] Streaming thread started for {self._symbols}")

    def stop(self) -> None:
        """Signal the streaming thread to stop and wait for it."""
        if self._stop_event:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("[POLYGON_WS] Stream stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_latest_bars(self, symbol: str, n: int = 60) -> List[Bar]:
        """
        Return the most recent N bars received from the live stream.

        These are in addition to (and more recent than) what's in the
        REST cache — useful for intraday signal computation.
        """
        sym = symbol.upper()
        with self._bars_lock:
            return list(self._live_bars.get(sym, [])[-n:])

    # -------------------------------------------------------------------------
    # Async core (run inside event loop thread)
    # -------------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        """Entry point for the background thread — owns an asyncio event loop."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._stream_with_reconnect())
        finally:
            loop.close()

    async def _stream_with_reconnect(self) -> None:
        """Connect, stream, and reconnect on failure until stop() is called."""
        import asyncio
        delay = self._reconnect_delay

        while not (self._stop_event and self._stop_event.is_set()):
            try:
                await self._connect_and_stream()
                delay = self._reconnect_delay  # Reset on clean disconnect
            except Exception as exc:
                self._connected = False
                logger.warning(
                    f"[POLYGON_WS] Stream disconnected: {exc}. "
                    f"Reconnecting in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _connect_and_stream(self) -> None:
        """Single WebSocket session: authenticate, subscribe, and receive messages."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for Polygon streaming. "
                "Install with: pip install websockets"
            )

        async with websockets.connect(
            self._WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # 1. Wait for connected event
            await self._expect_status(ws, "connected")

            # 2. Authenticate
            await ws.send(json.dumps({"action": "auth", "params": self._api_key}))
            await self._expect_status(ws, "auth_success")
            logger.info("[POLYGON_WS] Authenticated")
            self._connected = True

            # 3. Subscribe — e.g. "AM.SPY,AM.AAPL,AM.QQQ"
            subs = ",".join(f"{self._sub_prefix}.{sym}" for sym in self._symbols)
            await ws.send(json.dumps({"action": "subscribe", "params": subs}))
            logger.info(f"[POLYGON_WS] Subscribed to {subs}")

            # 4. Message loop
            async for raw in ws:
                if self._stop_event and self._stop_event.is_set():
                    break
                self._handle_message(raw)

    async def _expect_status(self, ws, expected_status: str) -> None:
        """Read messages until we see the expected status."""
        raw = await ws.recv()
        messages = json.loads(raw)
        for msg in messages:
            if msg.get("status") == expected_status:
                return
        raise ConnectionError(
            f"[POLYGON_WS] Expected status '{expected_status}', got: {messages}"
        )

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message."""
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug(f"[POLYGON_WS] Non-JSON message: {raw[:100]}")
            return

        for msg in messages:
            ev = msg.get("ev", "")
            if ev == "AM":
                self._handle_minute_agg(msg)
            elif ev == "A":
                self._handle_second_agg(msg)
            elif ev == "T":
                self._handle_trade(msg)

    def _handle_minute_agg(self, msg: dict) -> None:
        """Handle a per-minute aggregate (AM) message and store as a Bar."""
        try:
            symbol = msg.get("sym", "")
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.utcfromtimestamp(msg["s"] / 1000),  # start of minute
                open=float(msg["o"]),
                high=float(msg["h"]),
                low=float(msg["l"]),
                close=float(msg["c"]),
                volume=int(msg.get("v", 0)),
                vwap=msg.get("vw"),
                interval="1m",
            )
            self._store_bar(symbol, bar)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug(f"[POLYGON_WS] Could not parse AM message: {exc} — {msg}")

    def _handle_second_agg(self, msg: dict) -> None:
        """Handle a per-second aggregate (A) message."""
        try:
            symbol = msg.get("sym", "")
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.utcfromtimestamp(msg["s"] / 1000),
                open=float(msg["o"]),
                high=float(msg["h"]),
                low=float(msg["l"]),
                close=float(msg["c"]),
                volume=int(msg.get("v", 0)),
                interval="1s",
            )
            self._store_bar(symbol, bar)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug(f"[POLYGON_WS] Could not parse A message: {exc}")

    def _handle_trade(self, msg: dict) -> None:
        """Handle an individual trade tick (T) message."""
        # Trades are passed only to the callback — not stored as OHLCV bars
        if not self._on_bar:
            return
        try:
            symbol = msg.get("sym", "")
            price = float(msg.get("p", 0))
            size = int(msg.get("s", 0))
            ts = datetime.utcfromtimestamp(msg.get("t", 0) / 1e9)  # nanoseconds
            bar = Bar(
                symbol=symbol,
                timestamp=ts,
                open=price, high=price, low=price, close=price,
                volume=size,
                interval="tick",
            )
            self._on_bar(bar)
        except Exception:
            pass

    def _store_bar(self, symbol: str, bar: Bar) -> None:
        """Write a live bar to the ring buffer and fire the callback."""
        with self._bars_lock:
            ring = self._live_bars.setdefault(symbol, [])
            ring.append(bar)
            if len(ring) > 500:
                del ring[0]  # Keep last 500 bars per symbol

        if self._on_bar:
            try:
                self._on_bar(bar)
            except Exception as exc:
                logger.debug(f"[POLYGON_WS] on_bar callback error: {exc}")
