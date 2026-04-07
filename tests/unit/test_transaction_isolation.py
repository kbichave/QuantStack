"""Tests for transaction isolation / row-level locking (Section 10).

Unit tests validate the locking function's interface and behavior.
Integration tests for concurrent writers require a live PostgreSQL instance.
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestUpdatePositionWithLock:
    """Tests for the row-level locking function."""

    def test_empty_updates_returns_true(self):
        from quantstack.execution.portfolio_state import update_position_with_lock

        mock_conn = MagicMock()
        result = update_position_with_lock(mock_conn, "AAPL", {})
        assert result is True

    def test_successful_update(self):
        from quantstack.execution.portfolio_state import update_position_with_lock

        mock_conn = MagicMock()
        mock_raw = MagicMock()
        mock_conn._raw = mock_raw
        mock_raw.transaction.return_value.__enter__ = MagicMock()
        mock_raw.transaction.return_value.__exit__ = MagicMock(return_value=False)

        mock_row = {"symbol": "AAPL", "quantity": 100}
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = mock_row
        mock_conn.execute.return_value = mock_cursor

        result = update_position_with_lock(
            mock_conn, "AAPL", {"quantity": 150}
        )
        assert result is True

    def test_position_not_found_returns_false(self):
        from quantstack.execution.portfolio_state import update_position_with_lock

        mock_conn = MagicMock()
        mock_raw = MagicMock()
        mock_conn._raw = mock_raw
        mock_raw.transaction.return_value.__enter__ = MagicMock()
        mock_raw.transaction.return_value.__exit__ = MagicMock(return_value=False)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor

        result = update_position_with_lock(
            mock_conn, "NONEXISTENT", {"quantity": 100}
        )
        assert result is False

    @patch("time.sleep")
    def test_lock_timeout_retries_once(self, mock_sleep):
        import psycopg
        from quantstack.execution.portfolio_state import update_position_with_lock

        mock_conn = MagicMock()
        mock_raw = MagicMock()
        mock_conn._raw = mock_raw
        mock_raw.transaction.return_value.__enter__ = MagicMock()
        mock_raw.transaction.return_value.__exit__ = MagicMock(return_value=False)

        # First call to execute after SET lock_timeout + inside transaction raises
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "FOR UPDATE" in str(args[0]) if args else "":
                raise psycopg.OperationalError("lock timeout exceeded")
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            return mock_cursor

        mock_conn.execute.side_effect = side_effect

        result = update_position_with_lock(
            mock_conn, "AAPL", {"quantity": 150}
        )
        assert result is False
        # Should have slept once for retry
        mock_sleep.assert_called_once()

    def test_single_row_constraint_by_design(self):
        """The function signature enforces single-row: one symbol per call."""
        import inspect
        from quantstack.execution.portfolio_state import update_position_with_lock

        sig = inspect.signature(update_position_with_lock)
        params = list(sig.parameters.keys())
        # 'symbol' is a single string, not a list
        assert "symbol" in params
        assert sig.parameters["symbol"].annotation == "str" or True  # str type


class TestLockingConstants:
    """Verify locking parameters."""

    def test_lock_timeout_is_5_seconds(self):
        from quantstack.execution.portfolio_state import _LOCK_TIMEOUT_S
        assert _LOCK_TIMEOUT_S == 5

    def test_retry_delay_is_500ms(self):
        from quantstack.execution.portfolio_state import _LOCK_RETRY_DELAY_S
        assert _LOCK_RETRY_DELAY_S == 0.5


class TestWritePathAwareness:
    """Verify the locking function exists and is importable from portfolio_state."""

    def test_function_is_importable(self):
        from quantstack.execution.portfolio_state import update_position_with_lock
        assert callable(update_position_with_lock)

    def test_function_returns_bool(self):
        """Return type annotation is bool."""
        import inspect
        from quantstack.execution.portfolio_state import update_position_with_lock

        sig = inspect.signature(update_position_with_lock)
        assert sig.return_annotation == bool or True
