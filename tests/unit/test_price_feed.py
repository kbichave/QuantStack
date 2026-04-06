"""Tests for PriceFeedProtocol implementations."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from quantstack.execution.price_feed import (
    BarPollingFeed,
    PaperPriceFeed,
    PriceFeedProtocol,
    get_price_feed,
)

ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:

    def test_bar_polling_feed_satisfies_protocol(self):
        """BarPollingFeed satisfies PriceFeedProtocol."""
        feed = BarPollingFeed()
        assert isinstance(feed, PriceFeedProtocol)

    def test_paper_price_feed_satisfies_protocol(self):
        """PaperPriceFeed satisfies PriceFeedProtocol."""
        feed = PaperPriceFeed(events=[])
        assert isinstance(feed, PriceFeedProtocol)


# ---------------------------------------------------------------------------
# BarPollingFeed
# ---------------------------------------------------------------------------


class TestBarPollingFeed:

    @pytest.mark.asyncio
    async def test_subscribe_stores_callback_and_symbols(self):
        """subscribe() stores callback and symbol list."""
        feed = BarPollingFeed()
        callback = AsyncMock()
        await feed.subscribe(["SPY", "AAPL"], callback)
        assert feed._symbols == {"SPY", "AAPL"}
        assert feed._callback is callback

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_symbol(self):
        """unsubscribe removes symbol from poll set."""
        feed = BarPollingFeed()
        callback = AsyncMock()
        await feed.subscribe(["SPY", "AAPL"], callback)
        await feed.unsubscribe(["SPY"])
        assert feed._symbols == {"AAPL"}

    @pytest.mark.asyncio
    async def test_poll_invokes_callback(self):
        """Poll cycle invokes callback with (symbol, price, timestamp)."""
        features = MagicMock()
        features.close = 450.0
        features.timestamp = datetime(2026, 4, 6, 10, 0, tzinfo=ET)

        stream_mgr = MagicMock()
        stream_mgr.get_features.return_value = features

        callback = AsyncMock()
        feed = BarPollingFeed(stream_manager=stream_mgr)
        await feed.subscribe(["SPY"], callback)
        await feed._poll_once()

        callback.assert_called_once_with("SPY", 450.0, features.timestamp)

    @pytest.mark.asyncio
    async def test_poll_skips_when_no_data(self):
        """When StreamManager has no data, callback is NOT invoked."""
        stream_mgr = MagicMock()
        stream_mgr.get_features.return_value = None

        callback = AsyncMock()
        feed = BarPollingFeed(stream_manager=stream_mgr)
        await feed.subscribe(["SPY"], callback)
        await feed._poll_once()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop() cancels the polling task cleanly."""
        feed = BarPollingFeed(poll_interval_s=100.0)  # Long interval so it doesn't fire
        await feed.start()
        assert feed._poll_task is not None
        await feed.stop()
        assert feed._poll_task is None


# ---------------------------------------------------------------------------
# PaperPriceFeed
# ---------------------------------------------------------------------------


class TestPaperPriceFeed:

    @pytest.mark.asyncio
    async def test_replays_events(self):
        """PaperPriceFeed replays events through callback."""
        events = [
            ("SPY", 450.0, datetime(2026, 4, 6, 10, 0, tzinfo=ET)),
            ("SPY", 451.0, datetime(2026, 4, 6, 10, 1, tzinfo=ET)),
            ("AAPL", 175.0, datetime(2026, 4, 6, 10, 0, tzinfo=ET)),
        ]
        callback = AsyncMock()
        feed = PaperPriceFeed(events=events, replay_speed=0.0)
        await feed.subscribe(["SPY", "AAPL"], callback)
        await feed.start()
        # Wait for replay to complete
        await feed._replay_task
        assert callback.call_count == 3

    @pytest.mark.asyncio
    async def test_filters_by_subscribed_symbols(self):
        """Only subscribed symbols are delivered to callback."""
        events = [
            ("SPY", 450.0, datetime(2026, 4, 6, 10, 0, tzinfo=ET)),
            ("AAPL", 175.0, datetime(2026, 4, 6, 10, 0, tzinfo=ET)),
        ]
        callback = AsyncMock()
        feed = PaperPriceFeed(events=events, replay_speed=0.0)
        await feed.subscribe(["SPY"], callback)  # Only SPY
        await feed.start()
        await feed._replay_task
        assert callback.call_count == 1
        callback.assert_called_once_with("SPY", 450.0, events[0][2])

    @pytest.mark.asyncio
    async def test_stop_halts_replay(self):
        """stop() halts replay without error."""
        from datetime import timedelta
        base = datetime(2026, 4, 6, 10, 0, tzinfo=ET)
        events = [("SPY", 450.0 + i, base + timedelta(seconds=i)) for i in range(100)]
        callback = AsyncMock()
        feed = PaperPriceFeed(events=events, replay_speed=0.01)
        await feed.subscribe(["SPY"], callback)
        await feed.start()
        await feed.stop()
        # Should have processed fewer than all 100 events
        assert callback.call_count < 100


# ---------------------------------------------------------------------------
# Factory: get_price_feed
# ---------------------------------------------------------------------------


class TestGetPriceFeed:

    def test_bars_mode_returns_bar_feed(self, monkeypatch):
        """EXEC_MONITOR_QUOTE_FEED=bars → BarPollingFeed."""
        monkeypatch.setenv("EXEC_MONITOR_QUOTE_FEED", "bars")
        feed = get_price_feed(stream_manager=MagicMock())
        assert isinstance(feed, BarPollingFeed)

    def test_auto_no_adapter_returns_bar_feed(self, monkeypatch):
        """EXEC_MONITOR_QUOTE_FEED=auto without adapter → BarPollingFeed."""
        monkeypatch.setenv("EXEC_MONITOR_QUOTE_FEED", "auto")
        feed = get_price_feed(stream_manager=MagicMock(), adapter=None)
        assert isinstance(feed, BarPollingFeed)

    def test_quotes_without_adapter_raises(self, monkeypatch):
        """EXEC_MONITOR_QUOTE_FEED=quotes without adapter → ValueError."""
        monkeypatch.setenv("EXEC_MONITOR_QUOTE_FEED", "quotes")
        with pytest.raises(ValueError, match="requires"):
            get_price_feed(adapter=None)
