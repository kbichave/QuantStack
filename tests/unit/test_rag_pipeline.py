"""Tests for RAG Pipeline (pgvector backend).

Unit tests mock psycopg2 at the connection level. No live database required.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class FakeEmbeddingFunction:
    """Deterministic embedding function for testing.

    Returns 1024-dimensional vectors based on text content.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in input]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        base = [float(ord(c)) for c in text[:10]]
        return base + [0.0] * (1024 - len(base))


@pytest.fixture()
def embedding_fn():
    return FakeEmbeddingFunction()


@pytest.fixture()
def mock_conn():
    """Mock psycopg2 connection with cursor that tracks executed SQL."""
    conn = MagicMock()
    conn.closed = False
    cursor = MagicMock()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# OllamaEmbeddingFunction
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingFunction:
    """Tests for embeddings.py."""

    def test_calls_ollama_embed_with_correct_model(self):
        import sys
        mock_ollama_mod = MagicMock()
        mock_client = MagicMock()
        mock_client.embed.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
        mock_ollama_mod.Client.return_value = mock_client
        with patch.dict(sys.modules, {"ollama": mock_ollama_mod}):
            from quantstack.rag.embeddings import OllamaEmbeddingFunction
            ef = OllamaEmbeddingFunction(model_name="mxbai-embed-large")
            result = ef(["hello", "world"])
            mock_client.embed.assert_called_once_with(
                model="mxbai-embed-large", input=["hello", "world"]
            )
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_returns_list_of_float_lists(self):
        import sys
        mock_ollama_mod = MagicMock()
        mock_client = MagicMock()
        mock_client.embed.return_value = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]]
        }
        mock_ollama_mod.Client.return_value = mock_client
        with patch.dict(sys.modules, {"ollama": mock_ollama_mod}):
            from quantstack.rag.embeddings import OllamaEmbeddingFunction
            ef = OllamaEmbeddingFunction()
            result = ef(["a", "b"])
            assert isinstance(result, list)
            assert all(isinstance(v, list) for v in result)
            assert all(isinstance(x, float) for v in result for x in v)

    def test_raises_connection_error_on_failure(self):
        import sys
        mock_ollama_mod = MagicMock()
        mock_ollama_mod.Client.return_value.embed.side_effect = Exception("unreachable")
        with patch.dict(sys.modules, {"ollama": mock_ollama_mod}):
            from quantstack.rag.embeddings import OllamaEmbeddingFunction
            ef = OllamaEmbeddingFunction()
            with pytest.raises(ConnectionError, match="Failed to get embeddings"):
                ef(["test"])

    def test_embedding_dimension_constant(self):
        from quantstack.rag.embeddings import EMBEDDING_DIMENSION
        assert EMBEDDING_DIMENSION == 1024


# ---------------------------------------------------------------------------
# chunk_markdown (pure function — unchanged)
# ---------------------------------------------------------------------------

class TestChunkMarkdown:
    """Tests for ingest.py :: chunk_markdown."""

    def test_chunks_text_with_correct_size(self):
        from quantstack.rag.ingest import chunk_markdown
        text = "## Section\n\n" + ("word " * 500)
        chunks = chunk_markdown(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "section" in chunk

    def test_preserves_section_headings(self):
        from quantstack.rag.ingest import chunk_markdown
        text = "## My Section\n\nSome content here.\n\n## Another\n\nMore content."
        chunks = chunk_markdown(text, chunk_size=5000)
        assert any("Another" in c["section"] or "My Section" in c["section"] for c in chunks)

    def test_empty_text_returns_empty(self):
        from quantstack.rag.ingest import chunk_markdown
        assert chunk_markdown("") == []
        assert chunk_markdown("   ") == []


# ---------------------------------------------------------------------------
# file_to_collection (pure function — unchanged)
# ---------------------------------------------------------------------------

class TestFileToCollection:
    """Tests for ingest.py :: file_to_collection."""

    def test_strategy_registry_maps_to_strategy_knowledge(self):
        from quantstack.rag.ingest import file_to_collection
        assert file_to_collection("strategy_registry.md") == "strategy_knowledge"

    def test_trade_journal_maps_to_trade_outcomes(self):
        from quantstack.rag.ingest import file_to_collection
        assert file_to_collection("trade_journal.md") == "trade_outcomes"

    def test_session_handoff_maps_to_market_research(self):
        from quantstack.rag.ingest import file_to_collection
        assert file_to_collection("session_handoff_2026_04_01.md") == "market_research"

    def test_unknown_defaults_to_market_research(self):
        from quantstack.rag.ingest import file_to_collection
        assert file_to_collection("random_file.md") == "market_research"


# ---------------------------------------------------------------------------
# store_embedding
# ---------------------------------------------------------------------------

class TestStoreEmbedding:
    """Tests for query.py :: store_embedding."""

    def test_inserts_with_correct_params(self, mock_conn):
        from quantstack.rag.query import store_embedding
        embedding = [0.1] * 1024
        store_embedding("doc-1", "trade_outcomes", "some text", embedding,
                        {"ticker": "SPY"}, conn=mock_conn)
        cursor = mock_conn.cursor.return_value
        assert cursor.execute.called
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO embeddings" in sql
        assert "ON CONFLICT" in sql

    def test_upserts_on_conflict(self, mock_conn):
        from quantstack.rag.query import store_embedding
        embedding = [0.1] * 1024
        store_embedding("doc-1", "trade_outcomes", "text v1", embedding, conn=mock_conn)
        store_embedding("doc-1", "trade_outcomes", "text v2", embedding, conn=mock_conn)
        assert mock_conn.cursor.return_value.execute.call_count == 2


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------

class TestSearchSimilar:
    """Tests for query.py :: search_similar."""

    def test_queries_with_collection_filter(self, mock_conn):
        from quantstack.rag.query import search_similar
        import psycopg2.extras
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = [
            {"content": "result text", "metadata": {"ticker": "SPY"},
             "distance": 0.1, "collection": "strategy_knowledge"},
        ]
        mock_conn.cursor.return_value = dict_cursor

        results = search_similar([0.1] * 1024, collection="strategy_knowledge",
                                 n_results=5, conn=mock_conn)
        assert len(results) == 1
        assert results[0]["text"] == "result text"
        assert results[0]["distance"] == 0.1

    def test_queries_with_metadata_filter(self, mock_conn):
        from quantstack.rag.query import search_similar
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = dict_cursor

        search_similar([0.1] * 1024, collection="trade_outcomes",
                       metadata_filter={"ticker": "AAPL"}, conn=mock_conn)
        sql = dict_cursor.execute.call_args[0][0]
        assert "ticker" in sql

    def test_returns_empty_for_no_matches(self, mock_conn):
        from quantstack.rag.query import search_similar
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = dict_cursor

        results = search_similar([0.1] * 1024, conn=mock_conn)
        assert results == []


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------

class TestDeleteCollection:
    """Tests for query.py :: delete_collection."""

    def test_deletes_by_collection_name(self, mock_conn):
        from quantstack.rag.query import delete_collection
        mock_conn.cursor.return_value.rowcount = 5
        count = delete_collection("trade_outcomes", conn=mock_conn)
        assert count == 5
        sql = mock_conn.cursor.return_value.execute.call_args[0][0]
        assert "DELETE FROM embeddings" in sql

    def test_returns_zero_when_empty(self, mock_conn):
        from quantstack.rag.query import delete_collection
        mock_conn.cursor.return_value.rowcount = 0
        count = delete_collection("trade_outcomes", conn=mock_conn)
        assert count == 0


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------

class TestSearchKnowledgeBase:
    """Tests for query.py :: search_knowledge_base."""

    def test_returns_results_with_correct_shape(self, mock_conn, embedding_fn):
        from quantstack.rag.query import search_knowledge_base
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = [
            {"content": "momentum works", "metadata": {"ticker": "SPY"},
             "distance": 0.05, "collection": "strategy_knowledge"},
        ]
        mock_conn.cursor.return_value = dict_cursor

        results = search_knowledge_base(
            "momentum", collection="strategy_knowledge",
            conn=mock_conn, embedding_fn=embedding_fn,
        )
        assert len(results) >= 1
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]
        assert "collection" in results[0]

    def test_filters_by_ticker(self, mock_conn, embedding_fn):
        from quantstack.rag.query import search_knowledge_base
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = [
            {"content": "AAPL trade", "metadata": {"ticker": "AAPL"},
             "distance": 0.1, "collection": "trade_outcomes"},
        ]
        mock_conn.cursor.return_value = dict_cursor

        results = search_knowledge_base(
            "trade", collection="trade_outcomes", ticker="AAPL",
            conn=mock_conn, embedding_fn=embedding_fn,
        )
        sql = dict_cursor.execute.call_args[0][0]
        assert "ticker" in sql

    def test_empty_results(self, mock_conn, embedding_fn):
        from quantstack.rag.query import search_knowledge_base
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = dict_cursor

        results = search_knowledge_base(
            "anything", collection="strategy_knowledge",
            conn=mock_conn, embedding_fn=embedding_fn,
        )
        assert results == []

    def test_searches_all_collections_when_none(self, mock_conn, embedding_fn):
        from quantstack.rag.query import search_knowledge_base, COLLECTIONS
        dict_cursor = MagicMock()
        dict_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = dict_cursor

        search_knowledge_base("data", conn=mock_conn, embedding_fn=embedding_fn, n_results=10)
        # Should have queried once per collection
        assert dict_cursor.execute.call_count == len(COLLECTIONS)

    def test_returns_empty_when_embedding_fails(self, mock_conn):
        from quantstack.rag.query import search_knowledge_base
        broken_fn = MagicMock(side_effect=ConnectionError("no ollama"))
        results = search_knowledge_base(
            "test", conn=mock_conn, embedding_fn=broken_fn,
        )
        assert results == []


# ---------------------------------------------------------------------------
# remember_knowledge
# ---------------------------------------------------------------------------

class TestRememberKnowledge:
    """Tests for query.py :: remember_knowledge."""

    def test_writes_document_returns_id(self, mock_conn, embedding_fn):
        from quantstack.rag.query import remember_knowledge
        doc_id = remember_knowledge(
            "lesson learned", "trade_outcomes",
            metadata={"ticker": "SPY", "outcome": "win"},
            conn=mock_conn, embedding_fn=embedding_fn,
        )
        assert doc_id.startswith("trade_outcomes::")
        assert mock_conn.cursor.return_value.execute.called

    def test_rejects_unknown_collection(self, mock_conn, embedding_fn):
        from quantstack.rag.query import remember_knowledge
        with pytest.raises(ValueError, match="Unknown collection"):
            remember_knowledge("text", "fake_collection", conn=mock_conn, embedding_fn=embedding_fn)

    def test_returns_empty_when_embedding_fails(self, mock_conn):
        from quantstack.rag.query import remember_knowledge
        broken_fn = MagicMock(side_effect=ConnectionError("no ollama"))
        doc_id = remember_knowledge(
            "text", "trade_outcomes",
            conn=mock_conn, embedding_fn=broken_fn,
        )
        assert doc_id == ""


# ---------------------------------------------------------------------------
# ingest_memory_files
# ---------------------------------------------------------------------------

class TestIngestMemoryFiles:
    """Tests for ingest.py :: ingest_memory_files."""

    def test_ingests_md_files(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.ingest import ingest_memory_files
        (tmp_path / "strategy_registry.md").write_text("## Strategy\n\nSome strategy info")
        (tmp_path / "trade_journal.md").write_text("## Trade\n\nTrade outcome data")

        with patch("quantstack.rag.query.store_embedding"):
            counts = ingest_memory_files(str(tmp_path), embedding_fn=embedding_fn, conn=mock_conn)
        assert counts.get("strategy_knowledge", 0) > 0
        assert counts.get("trade_outcomes", 0) > 0

    def test_idempotent_skips_when_populated(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.ingest import ingest_memory_files
        (tmp_path / "strategy_registry.md").write_text("## Strategy\n\nData here")

        # Simulate non-empty collections
        mock_conn.cursor.return_value.fetchone.return_value = (5,)

        counts = ingest_memory_files(str(tmp_path), embedding_fn=embedding_fn, conn=mock_conn)
        assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# No chromadb imports (static check)
# ---------------------------------------------------------------------------

class TestNoChromadbImports:
    """Verify no code imports chromadb after migration."""

    def test_no_chromadb_in_rag_modules(self):
        import quantstack.rag.query as q
        import quantstack.rag.ingest as ing
        import quantstack.rag.embeddings as emb
        import inspect

        for mod in [q, ing, emb]:
            source = inspect.getsource(mod)
            assert "import chromadb" not in source, f"{mod.__name__} still imports chromadb"
            assert "from chromadb" not in source, f"{mod.__name__} still imports from chromadb"


# ---------------------------------------------------------------------------
# route_file (pure function from migrate_memory.py)
# ---------------------------------------------------------------------------

class TestRouteFile:
    """Tests for migrate_memory.py :: route_file."""

    def test_routes_strategy_registry(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        f = tmp_path / "strategy_registry.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result is not None
        assert result[0] == "strategy_knowledge"

    def test_routes_ticker_files(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        ticker_dir = tmp_path / "tickers"
        ticker_dir.mkdir()
        f = ticker_dir / "AAPL.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result is not None
        assert result[0] == "market_research"
        assert result[1]["ticker"] == "AAPL"

    def test_skips_templates(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        f = tpl_dir / "template.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result is None

    def test_default_routing(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        f = tmp_path / "random_notes.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result is not None
        assert result[0] == "market_research"


# ---------------------------------------------------------------------------
# _chunk_text (pure function from migrate_memory.py)
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for migrate_memory.py :: _chunk_text."""

    def test_splits_long_text(self):
        from quantstack.rag.migrate_memory import _chunk_text
        text = "word " * 500
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1

    def test_empty_text_returns_empty(self):
        from quantstack.rag.migrate_memory import _chunk_text
        assert _chunk_text("") == []
        assert _chunk_text("  ") == []

    def test_short_text_single_chunk(self):
        from quantstack.rag.migrate_memory import _chunk_text
        chunks = _chunk_text("short text", chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == "short text"


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------

class TestEnsureSchema:
    """Tests for query.py :: ensure_schema."""

    def test_executes_init_sql(self, mock_conn):
        from quantstack.rag.query import ensure_schema
        ensure_schema(conn=mock_conn)
        sql = mock_conn.cursor.return_value.execute.call_args[0][0]
        assert "CREATE EXTENSION" in sql
        assert "CREATE TABLE" in sql
        assert "CREATE INDEX" in sql
