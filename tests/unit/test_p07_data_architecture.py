# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for P07: Data Architecture Evolution.

Covers: Yahoo adapter, FMP adapter, ProviderChain, PIT helper,
staleness tiering, market session detection.
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")


# ─── Market Session & Staleness Tiering ────────────────────────────────

class TestMarketSession:
    """Test get_market_session() and get_stale_threshold()."""

    def test_market_hours_monday_1030(self):
        from quantstack.data.validator import get_market_session
        now = datetime(2025, 4, 7, 10, 30, tzinfo=ET)
        assert get_market_session(now) == "market_hours"

    def test_market_hours_at_open(self):
        from quantstack.data.validator import get_market_session
        now = datetime(2025, 4, 7, 9, 30, tzinfo=ET)
        assert get_market_session(now) == "market_hours"

    def test_extended_hours_at_close(self):
        from quantstack.data.validator import get_market_session
        # 16:00 is extended hours, not market hours
        now = datetime(2025, 4, 7, 16, 0, tzinfo=ET)
        assert get_market_session(now) == "extended_hours"

    def test_extended_hours_premarket(self):
        from quantstack.data.validator import get_market_session
        now = datetime(2025, 4, 7, 7, 0, tzinfo=ET)
        assert get_market_session(now) == "extended_hours"

    def test_after_hours_overnight(self):
        from quantstack.data.validator import get_market_session
        now = datetime(2025, 4, 7, 2, 0, tzinfo=ET)
        assert get_market_session(now) == "after_hours"

    def test_weekend_saturday(self):
        from quantstack.data.validator import get_market_session
        now = datetime(2025, 4, 5, 14, 0, tzinfo=ET)
        assert get_market_session(now) == "after_hours"

    def test_weekend_sunday(self):
        from quantstack.data.validator import get_market_session
        now = datetime(2025, 4, 6, 10, 0, tzinfo=ET)
        assert get_market_session(now) == "after_hours"


class TestStaleThreshold:
    """Test tiered staleness thresholds."""

    def test_market_hours_30min(self):
        from quantstack.data.validator import get_stale_threshold
        now = datetime(2025, 4, 7, 10, 0, tzinfo=ET)
        assert get_stale_threshold(now) == timedelta(minutes=30)

    def test_extended_hours_8h(self):
        from quantstack.data.validator import get_stale_threshold
        now = datetime(2025, 4, 7, 7, 0, tzinfo=ET)
        assert get_stale_threshold(now) == timedelta(hours=8)

    def test_after_hours_24h(self):
        from quantstack.data.validator import get_stale_threshold
        now = datetime(2025, 4, 5, 14, 0, tzinfo=ET)
        assert get_stale_threshold(now) == timedelta(hours=24)


# ─── Yahoo Provider ───────────────────────────────────────────────────

class TestYahooProvider:
    """Test Yahoo Finance adapter."""

    def test_missing_yfinance_raises_config_error(self, monkeypatch):
        from quantstack.data.providers.base import ConfigurationError

        # Temporarily hide yfinance
        monkeypatch.setitem(sys.modules, "yfinance", None)
        # Also need to clear any cached import in the yahoo module
        monkeypatch.delitem(sys.modules, "quantstack.data.providers.yahoo", raising=False)

        with pytest.raises(ConfigurationError, match="yfinance"):
            # Re-import to trigger __init__ check
            from importlib import reload
            import quantstack.data.providers.yahoo as ym
            reload(ym)
            ym.YahooProvider()

    def test_fetch_daily_returns_normalized_df(self):
        """YahooProvider.fetch_ohlcv_daily returns properly normalized DataFrame."""
        mock_yf = MagicMock()
        raw_df = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000000, 1100000],
            "Adj Close": [101.0, 102.0],
        }, index=pd.DatetimeIndex(["2025-04-01", "2025-04-02"], name="Date"))

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            mock_yf.download.return_value = raw_df
            from quantstack.data.providers.yahoo import YahooProvider
            provider = YahooProvider()
            df = provider.fetch_ohlcv_daily("AAPL")

        assert df is not None
        expected_cols = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_cols
        assert "adj_close" not in df.columns
        assert len(df) == 2
        assert df["symbol"].iloc[0] == "AAPL"

    def test_cache_hit_avoids_second_download(self):
        """Second call should return cached result without calling yfinance again."""
        mock_yf = MagicMock()
        raw_df = pd.DataFrame({
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000000],
        }, index=pd.DatetimeIndex(["2025-04-01"], name="Date"))

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            mock_yf.download.return_value = raw_df
            from quantstack.data.providers.yahoo import YahooProvider
            provider = YahooProvider()
            df1 = provider.fetch_ohlcv_daily("AAPL")
            df2 = provider.fetch_ohlcv_daily("AAPL")

        assert mock_yf.download.call_count == 1
        assert df1 is not None
        assert df2 is not None

    def test_empty_download_returns_none(self):
        """Empty yfinance result returns None."""
        mock_yf = MagicMock()
        mock_yf.download.return_value = pd.DataFrame()

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from quantstack.data.providers.yahoo import YahooProvider
            provider = YahooProvider()
            result = provider.fetch_ohlcv_daily("BADTICKER")

        assert result is None

    def test_name_is_yahoo(self):
        mock_yf = MagicMock()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from quantstack.data.providers.yahoo import YahooProvider
            provider = YahooProvider()
        assert provider.name() == "yahoo"


# ─── FMP Provider ─────────────────────────────────────────────────────

class TestFMPProvider:
    """Test Financial Modeling Prep adapter."""

    def test_missing_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        from quantstack.data.providers.base import ConfigurationError
        from quantstack.data.providers.fmp import FMPProvider
        with pytest.raises(ConfigurationError, match="FMP_API_KEY"):
            FMPProvider()

    def test_name_is_fmp(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        from quantstack.data.providers.fmp import FMPProvider
        provider = FMPProvider()
        assert provider.name() == "fmp"

    @patch("quantstack.data.providers.fmp.requests.Session")
    def test_fetch_fundamentals_returns_dict(self, mock_session_cls, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"symbol": "AAPL", "companyName": "Apple"}]
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        from quantstack.data.providers.fmp import FMPProvider
        provider = FMPProvider()
        result = provider.fetch_fundamentals("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"


# ─── Provider Chain ───────────────────────────────────────────────────

class TestProviderChain:
    """Test ProviderChain fallback logic."""

    def test_empty_providers_raises(self):
        from quantstack.data.provider_chain import ProviderChain
        with pytest.raises(ValueError, match="at least one"):
            ProviderChain([])

    def test_first_provider_succeeds(self):
        from quantstack.data.provider import Bar
        from quantstack.data.provider_chain import ProviderChain

        provider1 = MagicMock()
        provider1.name = "mock1"
        provider1.get_bars.return_value = [
            Bar(symbol="AAPL", timestamp=datetime.now(), open=100, high=102,
                low=99, close=101, volume=1000000)
        ]

        chain = ProviderChain([provider1])
        bars = chain.get_bars("AAPL")
        assert len(bars) == 1
        provider1.get_bars.assert_called_once()

    def test_fallback_on_first_failure(self):
        from quantstack.data.provider import Bar
        from quantstack.data.provider_chain import ProviderChain

        provider1 = MagicMock()
        provider1.name = "failing"
        provider1.get_bars.side_effect = RuntimeError("API down")

        provider2 = MagicMock()
        provider2.name = "backup"
        provider2.get_bars.return_value = [
            Bar(symbol="AAPL", timestamp=datetime.now(), open=100, high=102,
                low=99, close=101, volume=1000000)
        ]

        chain = ProviderChain([provider1, provider2])
        bars = chain.get_bars("AAPL")
        assert len(bars) == 1
        assert provider2.get_bars.called

    def test_circuit_breaker_opens_after_5_failures(self):
        from quantstack.data.provider_chain import ProviderChain

        provider1 = MagicMock()
        provider1.name = "flaky"
        provider1.get_bars.side_effect = RuntimeError("down")

        provider2 = MagicMock()
        provider2.name = "backup"
        provider2.get_bars.return_value = []

        chain = ProviderChain([provider1, provider2])

        # Exhaust circuit breaker (5 consecutive failures)
        for _ in range(5):
            chain.get_bars("AAPL")

        # After 5 failures, circuit is open — provider1 should be skipped
        stats = chain.get_stats()
        assert stats["flaky"]["circuit_open"] is True
        assert stats["flaky"]["consecutive_failures"] >= 5

    def test_stats_track_success_and_latency(self):
        from quantstack.data.provider_chain import ProviderChain

        provider = MagicMock()
        provider.name = "fast"
        provider.get_bars.return_value = []

        chain = ProviderChain([provider])
        chain.get_bars("AAPL")
        chain.get_bars("MSFT")

        stats = chain.get_stats()
        assert stats["fast"]["successes"] == 2
        assert stats["fast"]["failures"] == 0
        assert stats["fast"]["success_rate"] == 1.0

    def test_reset_circuit(self):
        from quantstack.data.provider_chain import ProviderChain

        provider = MagicMock()
        provider.name = "flaky"
        provider.get_bars.side_effect = RuntimeError("down")

        backup = MagicMock()
        backup.name = "backup"
        backup.get_bars.return_value = []

        chain = ProviderChain([provider, backup])

        for _ in range(5):
            chain.get_bars("AAPL")

        assert chain.get_stats()["flaky"]["circuit_open"] is True

        chain.reset_circuit("flaky")
        assert chain.get_stats()["flaky"]["circuit_open"] is False


# ─── PIT Query ────────────────────────────────────────────────────────

class TestPITQuery:
    """Test point-in-time query helper (table whitelist only — no DB needed)."""

    def test_rejects_unknown_table(self):
        from quantstack.data.pit import pit_query
        with pytest.raises(ValueError, match="not in allowed list"):
            pit_query("users; DROP TABLE positions", "AAPL", as_of=date(2025, 1, 1))

    def test_rejects_arbitrary_table(self):
        from quantstack.data.pit import pit_query
        with pytest.raises(ValueError, match="not in allowed list"):
            pit_query("positions", "AAPL", as_of=date(2025, 1, 1))

    def test_allows_valid_tables(self):
        """Allowed tables should not raise ValueError (they'll fail on DB connect)."""
        from quantstack.data.pit import pit_query
        for table in ("financial_statements", "earnings_calendar",
                       "insider_trades", "institutional_holdings"):
            # Should raise a DB connection error, NOT a ValueError
            with pytest.raises(Exception) as exc_info:
                pit_query(table, "AAPL", as_of=date(2025, 1, 1))
            assert "not in allowed list" not in str(exc_info.value)


# ─── Routing Table ────────────────────────────────────────────────────

class TestRoutingTable:
    """Verify the updated routing table includes new providers."""

    def test_ohlcv_daily_has_fmp_and_yahoo_fallbacks(self):
        from quantstack.data.providers.registry import _ROUTING_TABLE
        chain = _ROUTING_TABLE["ohlcv_daily"]
        assert chain == ["alpha_vantage", "fmp", "yahoo"]

    def test_ohlcv_intraday_has_yahoo_fallback(self):
        from quantstack.data.providers.registry import _ROUTING_TABLE
        chain = _ROUTING_TABLE["ohlcv_intraday"]
        assert "yahoo" in chain

    def test_fundamentals_has_fmp_fallback(self):
        from quantstack.data.providers.registry import _ROUTING_TABLE
        chain = _ROUTING_TABLE["fundamentals"]
        assert "fmp" in chain
        # FMP should be before edgar
        assert chain.index("fmp") < chain.index("edgar")
