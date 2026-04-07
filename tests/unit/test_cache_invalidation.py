"""Tests for signal cache auto-invalidation after intraday/EOD refresh."""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_brief(symbol: str):
    """Create a minimal valid SignalBrief for cache testing."""
    from quantstack.signal_engine.brief import SignalBrief

    return SignalBrief(
        date=date.today(),
        market_overview="test",
        market_bias="neutral",
        risk_environment="normal",
    )


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the signal cache module state between tests."""
    from quantstack.signal_engine import cache as signal_cache

    signal_cache.clear()
    signal_cache.hits = 0
    signal_cache.misses = 0
    yield
    signal_cache.clear()


def _mock_client():
    """Return a mock AlphaVantageClient with sensible defaults."""
    client = MagicMock()
    client.fetch_bulk_quotes.return_value = pd.DataFrame(
        [{"symbol": "AAPL", "price": 180.0}, {"symbol": "MSFT", "price": 420.0}]
    )
    client.fetch_intraday.return_value = pd.DataFrame(
        {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]}
    )
    client.fetch_news_sentiment.return_value = pd.DataFrame()
    return client


def _mock_store():
    store = MagicMock()
    store.save_ohlcv.return_value = None
    store.save_news_sentiment.return_value = 0
    return store


# Patch paths: deferred imports resolve from source modules
_PATCHES = {
    "active": "quantstack.data.scheduled_refresh._get_active_symbols",
    "watched": "quantstack.data.scheduled_refresh._get_watched_symbols",
    "client": "quantstack.data.fetcher.AlphaVantageClient",
    "store": "quantstack.data.pg_storage.PgDataStore",
}


class TestIntradayCacheInvalidation:
    """Verify cache.invalidate(symbol) is called after intraday data writes."""

    @pytest.mark.asyncio
    async def test_intraday_refresh_invalidates_refreshed_symbols(self):
        """After intraday refresh writes data for symbols, those are invalidated."""
        from quantstack.signal_engine import cache as signal_cache

        signal_cache.put("AAPL", _make_brief("AAPL"))
        assert signal_cache.get("AAPL") is not None

        with (
            patch(_PATCHES["active"], return_value=["AAPL", "MSFT"]),
            patch(_PATCHES["watched"], return_value=[]),
            patch(_PATCHES["client"], return_value=_mock_client()),
            patch(_PATCHES["store"], return_value=_mock_store()),
        ):
            from quantstack.data.scheduled_refresh import run_intraday_refresh

            await run_intraday_refresh()

        assert signal_cache.get("AAPL") is None

    @pytest.mark.asyncio
    async def test_invalidation_is_per_symbol(self):
        """Only symbols that received fresh data are invalidated."""
        from quantstack.signal_engine import cache as signal_cache

        for sym in ("AAPL", "MSFT", "GOOG"):
            signal_cache.put(sym, _make_brief(sym))

        client = _mock_client()
        client.fetch_bulk_quotes.return_value = pd.DataFrame(
            [{"symbol": "AAPL", "price": 180.0}, {"symbol": "MSFT", "price": 420.0}]
        )

        with (
            patch(_PATCHES["active"], return_value=["AAPL", "MSFT", "GOOG"]),
            patch(_PATCHES["watched"], return_value=[]),
            patch(_PATCHES["client"], return_value=client),
            patch(_PATCHES["store"], return_value=_mock_store()),
        ):
            from quantstack.data.scheduled_refresh import run_intraday_refresh

            await run_intraday_refresh()

        assert signal_cache.get("AAPL") is None
        assert signal_cache.get("MSFT") is None
        assert signal_cache.get("GOOG") is not None

    @pytest.mark.asyncio
    async def test_cache_stats_logged_after_refresh(self):
        """Cache stats are logged after each intraday refresh cycle."""
        with (
            patch(_PATCHES["active"], return_value=["AAPL"]),
            patch(_PATCHES["watched"], return_value=[]),
            patch(_PATCHES["client"], return_value=_mock_client()),
            patch(_PATCHES["store"], return_value=_mock_store()),
            patch("quantstack.data.scheduled_refresh.logger") as mock_logger,
        ):
            from quantstack.data.scheduled_refresh import run_intraday_refresh

            await run_intraday_refresh()

            cache_stat_logged = any(
                "Cache" in str(call) and "stats" in str(call).lower()
                for call in mock_logger.info.call_args_list
            )
            assert cache_stat_logged, (
                "Expected a log line with cache stats after intraday refresh"
            )


class TestEodCacheInvalidation:
    """Verify EOD refresh calls cache.clear()."""

    @pytest.mark.asyncio
    async def test_eod_refresh_clears_cache(self):
        """EOD refresh calls signal_cache.clear()."""
        from quantstack.signal_engine import cache as signal_cache

        signal_cache.put("AAPL", _make_brief("AAPL"))

        mock_client = MagicMock()
        mock_client.fetch_daily.return_value = pd.DataFrame()
        mock_client.fetch_realtime_options.return_value = pd.DataFrame()
        mock_client.fetch_company_overview.return_value = None
        mock_client.fetch_earnings_calendar.return_value = pd.DataFrame()

        with (
            patch(_PATCHES["active"], return_value=["AAPL"]),
            patch(_PATCHES["watched"], return_value=[]),
            patch(_PATCHES["client"], return_value=mock_client),
            patch(_PATCHES["store"], return_value=_mock_store()),
            patch(
                "quantstack.data.scheduled_refresh._get_stale_fundamentals",
                return_value=[],
            ),
        ):
            from quantstack.data.scheduled_refresh import run_eod_refresh

            await run_eod_refresh()

        assert signal_cache.get("AAPL") is None
        assert signal_cache.stats()["size"] == 0
