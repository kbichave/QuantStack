"""Tests for Memory Migration (pgvector backend)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class FakeEmbeddingFunction:
    """Deterministic embedding function for testing (1024-dim)."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in input]

    @staticmethod
    def _embed(text: str) -> list[float]:
        base = [float(ord(c)) for c in text[:10]]
        return base + [0.0] * (1024 - len(base))


@pytest.fixture()
def mock_conn():
    """Mock psycopg2 connection."""
    conn = MagicMock()
    conn.closed = False
    cursor = MagicMock()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture()
def embedding_fn():
    return FakeEmbeddingFunction()


class TestRouteFile:
    """Test file-to-collection routing logic."""

    def test_strategy_registry_routes_to_strategy_knowledge(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        f = tmp_path / "strategy_registry.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result is not None
        assert result[0] == "strategy_knowledge"
        assert result[1]["content_type"] == "strategy_definition"

    def test_workshop_lessons_routes_to_negative_result(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        f = tmp_path / "workshop_lessons.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result[0] == "strategy_knowledge"
        assert result[1]["content_type"] == "negative_result"

    def test_trade_journal_routes_to_trade_outcomes(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        f = tmp_path / "trade_journal.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result[0] == "trade_outcomes"
        assert result[1]["content_type"] == "trade_outcome"

    def test_ticker_file_routes_with_ticker_metadata(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        tickers_dir = tmp_path / "tickers"
        tickers_dir.mkdir()
        f = tickers_dir / "AAPL.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result[0] == "market_research"
        assert result[1]["ticker"] == "AAPL"
        assert result[1]["content_type"] == "ticker_research"

    def test_templates_are_skipped(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        f = templates_dir / "empty_template.md"
        f.touch()
        assert route_file(f, tmp_path) is None

    def test_risk_desk_report_routes_correctly(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        f = tmp_path / "risk_desk_report_2026_04_02.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result[0] == "market_research"
        assert result[1]["content_type"] == "risk_report"

    def test_session_handoff_subdir_routes_correctly(self, tmp_path):
        from quantstack.rag.migrate_memory import route_file
        subdir = tmp_path / "session_handoffs"
        subdir.mkdir()
        f = subdir / "2026_04_01.md"
        f.touch()
        result = route_file(f, tmp_path)
        assert result[0] == "market_research"
        assert result[1]["content_type"] == "session_handoff"


class TestMigrateMemory:
    """Test the full migration flow with mocked psycopg2."""

    def test_ingests_strategy_registry(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        (tmp_path / "strategy_registry.md").write_text(
            "## Momentum\n\nRSI crossover strategy for trending regimes.\n\n"
            "## Mean Reversion\n\nBollinger band mean reversion for ranging markets."
        )
        with patch("quantstack.rag.query.store_embedding"):
            counts = migrate_memory(tmp_path, embedding_fn, force=True, conn=mock_conn)
        assert counts["strategy_knowledge"] > 0

    def test_ingests_workshop_lessons_as_negative_result(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        (tmp_path / "workshop_lessons.md").write_text(
            "## Lesson 1\n\nLate entries in neutral regime fail 70% of the time."
        )
        with patch("quantstack.rag.query.store_embedding") as mock_store:
            counts = migrate_memory(tmp_path, embedding_fn, force=True, conn=mock_conn)
        assert counts["strategy_knowledge"] > 0
        # Verify metadata includes content_type
        call_args = mock_store.call_args_list[0]
        metadata = call_args[1].get("metadata") or call_args[0][4]
        assert metadata["content_type"] == "negative_result"

    def test_ingests_ticker_files_with_metadata(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        tickers_dir = tmp_path / "tickers"
        tickers_dir.mkdir()
        (tickers_dir / "AAPL.md").write_text("## AAPL Analysis\n\nApple momentum setup.")
        (tickers_dir / "SPY.md").write_text("## SPY Analysis\n\nSPY index tracking.")
        with patch("quantstack.rag.query.store_embedding") as mock_store:
            counts = migrate_memory(tmp_path, embedding_fn, force=True, conn=mock_conn)
        assert counts["market_research"] >= 2
        # Check ticker metadata was passed
        tickers = set()
        for c in mock_store.call_args_list:
            meta = c[1].get("metadata") or c[0][4]
            if "ticker" in meta:
                tickers.add(meta["ticker"])
        assert "AAPL" in tickers
        assert "SPY" in tickers

    def test_skips_if_collections_populated(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        (tmp_path / "strategy_registry.md").write_text("## Strategy\n\nContent here.")
        # Simulate all collections non-empty
        mock_conn.cursor.return_value.fetchone.return_value = (10,)
        counts = migrate_memory(tmp_path, embedding_fn, conn=mock_conn)
        assert all(v == 0 for v in counts.values())

    def test_force_flag_overrides_skip(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        (tmp_path / "strategy_registry.md").write_text("## Strategy\n\nContent here.")
        # Even with non-empty collections, force=True should ingest
        mock_conn.cursor.return_value.fetchone.return_value = (10,)
        with patch("quantstack.rag.query.store_embedding"):
            counts = migrate_memory(tmp_path, embedding_fn, force=True, conn=mock_conn)
        assert sum(counts.values()) > 0

    def test_empty_files_produce_no_chunks(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        (tmp_path / "empty_file.md").write_text("")
        (tmp_path / "whitespace.md").write_text("   \n\n   ")
        counts = migrate_memory(tmp_path, embedding_fn, conn=mock_conn)
        assert sum(counts.values()) == 0

    def test_ingested_content_is_retrievable(self, mock_conn, embedding_fn, tmp_path):
        from quantstack.rag.migrate_memory import migrate_memory
        (tmp_path / "strategy_registry.md").write_text(
            "## Momentum Crossover\n\nThis strategy uses RSI and MACD crossover signals."
        )
        stored_docs = []
        def capture_store(doc_id, collection, content, embedding, metadata, conn=None):
            stored_docs.append({"id": doc_id, "collection": collection, "content": content})

        with patch("quantstack.rag.query.store_embedding", side_effect=capture_store):
            migrate_memory(tmp_path, embedding_fn, force=True, conn=mock_conn)
        assert len(stored_docs) > 0
        all_text = " ".join(d["content"] for d in stored_docs).lower()
        assert "momentum" in all_text or "rsi" in all_text
