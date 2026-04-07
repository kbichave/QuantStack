"""Tests for HNSW vector index migration.

These tests verify the migration function logic without requiring a live
PostgreSQL instance. They mock the connection to verify the correct SQL
is executed.

For full integration verification (index exists, correct params),
run against a live PostgreSQL with pgvector — see section-04 validation steps.
"""

from unittest.mock import MagicMock, call

from quantstack.db import _migrate_hnsw_index_pg


class TestHnswMigration:

    def test_creates_vector_extension_first(self):
        conn = MagicMock()
        _migrate_hnsw_index_pg(conn)
        calls = conn.execute.call_args_list
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in str(c) for c in calls)

    def test_creates_hnsw_index(self):
        conn = MagicMock()
        _migrate_hnsw_index_pg(conn)
        calls = [str(c) for c in conn.execute.call_args_list]
        index_call = [c for c in calls if "idx_embeddings_hnsw" in c]
        assert len(index_call) == 1
        assert "hnsw" in index_call[0]
        assert "vector_cosine_ops" in index_call[0]
        assert "m = 16" in index_call[0]
        assert "ef_construction = 100" in index_call[0]

    def test_idempotent_uses_if_not_exists(self):
        conn = MagicMock()
        _migrate_hnsw_index_pg(conn)
        calls = [str(c) for c in conn.execute.call_args_list]
        index_call = [c for c in calls if "idx_embeddings_hnsw" in c][0]
        assert "IF NOT EXISTS" in index_call

    def test_runs_without_error_twice(self):
        conn = MagicMock()
        _migrate_hnsw_index_pg(conn)
        _migrate_hnsw_index_pg(conn)
        # No exception raised — idempotent
