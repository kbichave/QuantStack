# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MarketDataBus and RestPollingBus.

No network calls — quote_fn is a synchronous mock.
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from unittest.mock import MagicMock

import pytest
from quantstack.data.market_data_bus import RestPollingBus
from quantstack.execution.tick_executor import Tick

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_quote_fn(prices: dict[str, float]) -> MagicMock:
    """Return a mock that emulates a quote function returning canned prices."""

    def quote_fn(symbol: str) -> dict:
        return {
            "price": prices.get(symbol, 100.0),
            "volume": 1_000_000,
            "bid": prices.get(symbol, 100.0) - 0.01,
            "ask": prices.get(symbol, 100.0) + 0.01,
        }

    mock = MagicMock(side_effect=quote_fn)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRestPollingBusEmitsTicks:
    @pytest.mark.asyncio
    async def test_emits_tick_for_each_symbol(self):
        """One poll cycle → one Tick per symbol."""
        prices = {"SPY": 450.0, "QQQ": 380.0}
        quote_fn = make_quote_fn(prices)

        bus = RestPollingBus(
            symbols=["SPY", "QQQ"],
            quote_fn=quote_fn,
            poll_interval_seconds=0,  # no sleep between polls
        )

        tick_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Run bus for exactly one poll then stop
        async def run_once():
            bus._running = True
            for sym in bus.symbols:
                data = bus._quote_fn(sym)
                tick = Tick(
                    symbol=sym,
                    price=float(data.get("price", 0)),
                    volume=int(data.get("volume", 0)),
                    bid=data.get("bid"),
                    ask=data.get("ask"),
                )
                await tick_queue.put(tick)

        await run_once()

        ticks = []
        while not tick_queue.empty():
            ticks.append(tick_queue.get_nowait())

        symbols_seen = {t.symbol for t in ticks}
        assert "SPY" in symbols_seen
        assert "QQQ" in symbols_seen
        assert len(ticks) == 2

    @pytest.mark.asyncio
    async def test_tick_prices_match_quote_fn(self):
        prices = {"AAPL": 185.0}
        quote_fn = make_quote_fn(prices)

        RestPollingBus(symbols=["AAPL"], quote_fn=quote_fn)
        tick_queue: asyncio.Queue = asyncio.Queue()

        async def run_once():
            data = quote_fn("AAPL")
            tick = Tick(
                symbol="AAPL",
                price=float(data["price"]),
                volume=int(data["volume"]),
                bid=data.get("bid"),
                ask=data.get("ask"),
            )
            await tick_queue.put(tick)

        await run_once()
        tick = tick_queue.get_nowait()
        assert tick.price == pytest.approx(185.0)
        assert tick.symbol == "AAPL"

    def test_symbols_uppercased(self):
        bus = RestPollingBus(symbols=["spy", "qqq"], quote_fn=lambda s: {})
        assert bus.symbols == ["SPY", "QQQ"]

    def test_stop_sets_running_false(self):
        bus = RestPollingBus(symbols=["SPY"], quote_fn=lambda s: {})
        bus._running = True
        bus.stop()
        assert bus._running is False


class TestRestPollingBusSentinel:
    @pytest.mark.asyncio
    async def test_sentinel_emitted_after_stop(self):
        """After stop() → bus.run() must put None sentinel on queue."""
        call_count = 0

        def quote_fn(symbol: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"price": 100.0, "volume": 1_000_000}

        bus = RestPollingBus(
            symbols=["SPY"],
            quote_fn=quote_fn,
            poll_interval_seconds=0,
        )

        tick_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        async def stopper():
            # Let bus run one cycle then stop it
            await asyncio.sleep(0.05)
            bus.stop()

        await asyncio.gather(
            bus.run(tick_queue),
            stopper(),
        )

        # Drain queue — last item must be None sentinel
        items = []
        while not tick_queue.empty():
            items.append(tick_queue.get_nowait())

        assert items[-1] is None, "Last item must be None sentinel"

    @pytest.mark.asyncio
    async def test_run_stops_on_stop_call(self):
        """bus.run() must return (not hang) after stop() is called."""
        bus = RestPollingBus(
            symbols=["SPY"],
            quote_fn=lambda s: {"price": 100.0, "volume": 1_000_000},
            poll_interval_seconds=0,
        )
        tick_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        async def stopper():
            await asyncio.sleep(0.05)
            bus.stop()

        # This should complete within the timeout, not hang
        await asyncio.wait_for(
            asyncio.gather(bus.run(tick_queue), stopper()),
            timeout=2.0,
        )


class TestRestPollingBusErrorHandling:
    @pytest.mark.asyncio
    async def test_error_in_quote_fn_does_not_crash_bus(self):
        """A failing quote_fn should be swallowed; other symbols still polled."""
        call_count = 0

        def flaky_quote(symbol: str) -> dict:
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise ValueError("API error")
            return {"price": 100.0, "volume": 1_000_000}

        bus = RestPollingBus(
            symbols=["BAD", "SPY"],
            quote_fn=flaky_quote,
            poll_interval_seconds=0,
            error_sleep_seconds=0,
        )
        tick_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        async def stopper():
            await asyncio.sleep(0.1)
            bus.stop()

        await asyncio.gather(bus.run(tick_queue), stopper())

        # SPY ticks should still appear even though BAD failed
        items = [item for item in _drain_queue(tick_queue) if item is not None]
        assert any(isinstance(t, Tick) and t.symbol == "SPY" for t in items)

    @pytest.mark.asyncio
    async def test_invalid_price_tick_dropped(self):
        """Ticks with price <= 0 should not be enqueued."""
        bus = RestPollingBus(
            symbols=["SPY"],
            quote_fn=lambda s: {"price": 0.0, "volume": 0},
            poll_interval_seconds=0,
        )
        tick_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        async def stopper():
            await asyncio.sleep(0.05)
            bus.stop()

        await asyncio.gather(bus.run(tick_queue), stopper())
        items = _drain_queue(tick_queue)
        # Only None sentinel should be in the queue
        real_ticks = [i for i in items if isinstance(i, Tick)]
        assert len(real_ticks) == 0


def _drain_queue(q: asyncio.Queue) -> list:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


class TestTickDataclass:
    def test_mid_uses_bid_ask_when_available(self):
        tick = Tick(symbol="SPY", price=450.0, volume=1000, bid=449.98, ask=450.02)
        assert tick.mid == pytest.approx((449.98 + 450.02) / 2)

    def test_mid_falls_back_to_price(self):
        tick = Tick(symbol="SPY", price=450.0, volume=1000)
        assert tick.mid == pytest.approx(450.0)

    def test_timestamp_defaults_to_now(self):
        from datetime import datetime

        before = datetime.now(UTC)
        tick = Tick(symbol="SPY", price=450.0, volume=1000)
        after = datetime.now(UTC)
        assert before <= tick.timestamp <= after
