"""Tests for durable checkpointing (Section 06).

Unit tests for the checkpointer factory, pool sizing, and pruning logic.
Integration tests (crash recovery) are in tests/integration/.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestCheckpointerFactory:
    """Tests for create_checkpointer() factory function."""

    @patch("langgraph.checkpoint.postgres.PostgresSaver")
    @patch("psycopg_pool.ConnectionPool")
    def test_returns_postgres_saver(self, mock_pool_cls, mock_saver_cls):
        from quantstack.checkpointing import create_checkpointer

        result = create_checkpointer()
        mock_saver_cls.assert_called_once_with(mock_pool_cls.return_value)
        assert result == mock_saver_cls.return_value

    @patch("langgraph.checkpoint.postgres.PostgresSaver")
    @patch("psycopg_pool.ConnectionPool")
    def test_pool_sizing(self, mock_pool_cls, mock_saver_cls):
        from quantstack.checkpointing import create_checkpointer

        create_checkpointer()
        call_kwargs = mock_pool_cls.call_args
        assert call_kwargs.kwargs["min_size"] == 2
        assert call_kwargs.kwargs["max_size"] == 6

    @patch("langgraph.checkpoint.postgres.PostgresSaver")
    @patch("psycopg_pool.ConnectionPool")
    def test_reads_pg_url_from_env(self, mock_pool_cls, mock_saver_cls):
        from quantstack.checkpointing import create_checkpointer

        with patch.dict(os.environ, {"TRADER_PG_URL": "postgresql://test:5432/testdb"}):
            create_checkpointer()
        call_kwargs = mock_pool_cls.call_args
        assert "test" in call_kwargs.kwargs["conninfo"]


class TestRunnerIntegration:
    """Verify runners import create_checkpointer instead of MemorySaver."""

    def test_trading_runner_uses_postgres_saver(self):
        import inspect
        from quantstack.runners import trading_runner

        source = inspect.getsource(trading_runner)
        assert "create_checkpointer" in source
        assert "MemorySaver" not in source

    def test_research_runner_uses_postgres_saver(self):
        import inspect
        from quantstack.runners import research_runner

        source = inspect.getsource(research_runner)
        assert "create_checkpointer" in source
        assert "MemorySaver" not in source

    def test_supervisor_runner_uses_postgres_saver(self):
        import inspect
        from quantstack.runners import supervisor_runner

        source = inspect.getsource(supervisor_runner)
        assert "create_checkpointer" in source
        assert "MemorySaver" not in source


class TestCheckpointPruning:
    """Tests for prune_old_checkpoints()."""

    @patch("quantstack.db.db_conn")
    def test_prune_calls_delete(self, mock_db_conn):
        from quantstack.checkpointing import prune_old_checkpoints

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_conn.execute.return_value = mock_result
        mock_db_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        deleted = prune_old_checkpoints(retention_hours=48)
        assert deleted == 10  # 5 from checkpoint_writes + 5 from checkpoints
        assert mock_conn.execute.call_count == 2

    @patch("quantstack.db.db_conn")
    def test_prune_handles_exception(self, mock_db_conn):
        from quantstack.checkpointing import prune_old_checkpoints

        mock_db_conn.return_value.__enter__ = MagicMock(
            side_effect=Exception("DB down")
        )
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        # Should not raise
        deleted = prune_old_checkpoints()
        assert deleted == 0
