"""Tests for src/quantstack/signal_engine/staleness.py"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock


class TestCheckFreshness:
    """Tests for the check_freshness() function."""

    def _make_conn(self, last_timestamp):
        """Create a mock db_conn context manager that returns a row with last_timestamp."""
        conn = MagicMock()
        if last_timestamp is None:
            conn.execute.return_value.fetchone.return_value = None
        else:
            conn.execute.return_value.fetchone.return_value = {
                "last_timestamp": last_timestamp,
            }
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_returns_true_when_data_within_max_days(self, mock_db_conn):
        from quantstack.signal_engine.staleness import check_freshness

        last_ts = datetime.now(timezone.utc) - timedelta(days=1)
        mock_db_conn.return_value = self._make_conn(last_ts)

        assert check_freshness("AAPL", "ohlcv", max_days=4) is True

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_returns_false_when_data_exceeds_max_days(self, mock_db_conn):
        from quantstack.signal_engine.staleness import check_freshness

        last_ts = datetime.now(timezone.utc) - timedelta(days=5)
        mock_db_conn.return_value = self._make_conn(last_ts)

        assert check_freshness("AAPL", "ohlcv", max_days=4) is False

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_returns_false_when_no_metadata_row_exists(self, mock_db_conn):
        from quantstack.signal_engine.staleness import check_freshness

        mock_db_conn.return_value = self._make_conn(None)

        assert check_freshness("AAPL", "ohlcv", max_days=4) is False

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_uses_calendar_days_not_trading_days(self, mock_db_conn):
        """4 calendar days covers a 3-day weekend (Fri close -> Tue open)."""
        from quantstack.signal_engine.staleness import check_freshness

        # 3 days 23 hours ago — within the 4-day calendar window
        last_ts = datetime.now(timezone.utc) - timedelta(days=3, hours=23)
        mock_db_conn.return_value = self._make_conn(last_ts)

        assert check_freshness("AAPL", "ohlcv", max_days=4) is True

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_logs_warning_when_stale(self, mock_db_conn):
        from quantstack.signal_engine.staleness import check_freshness

        last_ts = datetime.now(timezone.utc) - timedelta(days=10)
        mock_db_conn.return_value = self._make_conn(last_ts)

        with patch("quantstack.signal_engine.staleness.logger") as mock_logger:
            check_freshness("AAPL", "ohlcv", max_days=4)
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "AAPL" in call_args
            assert "ohlcv" in call_args

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_logs_warning_when_missing(self, mock_db_conn):
        from quantstack.signal_engine.staleness import check_freshness

        mock_db_conn.return_value = self._make_conn(None)

        with patch("quantstack.signal_engine.staleness.logger") as mock_logger:
            check_freshness("AAPL", "ohlcv", max_days=4)
            mock_logger.warning.assert_called_once()

    @patch("quantstack.signal_engine.staleness.db_conn")
    def test_does_not_log_when_fresh(self, mock_db_conn):
        from quantstack.signal_engine.staleness import check_freshness

        last_ts = datetime.now(timezone.utc) - timedelta(days=1)
        mock_db_conn.return_value = self._make_conn(last_ts)

        with patch("quantstack.signal_engine.staleness.logger") as mock_logger:
            check_freshness("AAPL", "ohlcv", max_days=4)
            mock_logger.warning.assert_not_called()


class TestMetadataCoverage:
    """Verify STALENESS_THRESHOLDS keys match acquisition pipeline sources."""

    def test_all_threshold_keys_have_acquisition_coverage(self):
        from quantstack.signal_engine.staleness import STALENESS_THRESHOLDS

        expected_sources = {
            "ohlcv",
            "options_chains",
            "news_sentiment",
            "company_overview",
            "macro_indicators",
            "insider_trades",
            "short_interest",
            "sector",
            "events",
            "ewf",
        }
        assert set(STALENESS_THRESHOLDS.keys()) == expected_sources
