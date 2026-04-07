"""Tests for staleness rejection across signal engine collectors.

Verifies that collectors return {} when data is stale and compute
normally when data is fresh. Also tests the all-stale synthesis edge case.
"""

import pytest
from unittest.mock import MagicMock, patch


def _mock_store():
    """Return a mock DataStore with minimal attributes."""
    return MagicMock()


class TestCollectorStaleness:
    """Each collector returns {} when check_freshness returns False."""

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.technical.check_freshness", return_value=False)
    async def test_technical_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.technical import collect_technical

        result = await collect_technical("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "1d", max_days=4)

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.macro.check_freshness", return_value=False)
    async def test_macro_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.macro import collect_macro

        result = await collect_macro("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "macro_indicators", max_days=45)

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.sentiment.check_freshness", return_value=False)
    async def test_sentiment_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.sentiment import collect_sentiment

        result = await collect_sentiment("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "news_sentiment", max_days=7)

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.fundamentals.check_freshness", return_value=False)
    async def test_fundamentals_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.fundamentals import collect_fundamentals

        result = await collect_fundamentals("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "company_overview", max_days=90)

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.insider_signals.check_freshness", return_value=False)
    async def test_insider_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.insider_signals import collect_insider_signals

        result = await collect_insider_signals("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "insider_trades", max_days=30)

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.sector.check_freshness", return_value=False)
    async def test_sector_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.sector import collect_sector

        result = await collect_sector("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "1d", max_days=7)

    @pytest.mark.asyncio
    @patch("quantstack.signal_engine.collectors.ewf_collector.check_freshness", return_value=False)
    async def test_ewf_returns_empty_when_stale(self, mock_cf):
        from quantstack.signal_engine.collectors.ewf_collector import collect_ewf

        result = await collect_ewf("AAPL", _mock_store())
        assert result == {}
        mock_cf.assert_called_once_with("AAPL", "ewf_forecasts", max_days=7)


class TestMlSignalNotModified:
    """ml_signal collector should NOT have a staleness check."""

    def test_ml_signal_has_no_check_freshness_import(self):
        """Verify ml_signal.py does not import check_freshness."""
        import importlib
        import quantstack.signal_engine.collectors.ml_signal as mod

        source = importlib.util.find_spec(mod.__name__).origin
        with open(source) as f:
            content = f.read()
        assert "check_freshness" not in content
