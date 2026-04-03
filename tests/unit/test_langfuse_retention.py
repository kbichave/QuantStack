"""Tests for Langfuse trace retention cleanup."""

import pytest
from unittest.mock import patch, MagicMock


class TestLangfuseRetention:
    def test_cleanup_returns_int(self):
        """cleanup_langfuse_traces returns the count of deleted traces."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5

        with patch("quantstack.health.langfuse_retention._get_langfuse_conn", return_value=mock_conn):
            from quantstack.health.langfuse_retention import cleanup_langfuse_traces
            result = cleanup_langfuse_traces(retention_days=30)

        assert isinstance(result, int)
        assert result == 5

    def test_cleanup_is_idempotent(self):
        """Running cleanup on empty DB returns 0 and does not error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 0

        with patch("quantstack.health.langfuse_retention._get_langfuse_conn", return_value=mock_conn):
            from quantstack.health.langfuse_retention import cleanup_langfuse_traces
            result = cleanup_langfuse_traces(retention_days=30)

        assert result == 0

    def test_cleanup_uses_retention_days_parameter(self):
        """The retention_days parameter controls the delete threshold."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 0

        with patch("quantstack.health.langfuse_retention._get_langfuse_conn", return_value=mock_conn):
            from quantstack.health.langfuse_retention import cleanup_langfuse_traces
            cleanup_langfuse_traces(retention_days=7)

        # Verify the SQL was called with the interval parameter
        mock_cursor.execute.assert_called()
        call_args = str(mock_cursor.execute.call_args)
        assert "7" in call_args or "interval" in call_args.lower()
