"""RAG degradation tests: verify search/remember survive DB being unavailable."""

from unittest.mock import MagicMock, patch


class TestRagDegradation:
    def test_search_returns_empty_when_embedding_fails(self):
        """search_knowledge_base returns empty list when embedding fails."""
        broken_fn = MagicMock(side_effect=ConnectionError("Ollama down"))
        from quantstack.rag.query import search_knowledge_base
        result = search_knowledge_base("test query", embedding_fn=broken_fn)
        assert result == []

    def test_remember_returns_empty_when_embedding_fails(self):
        """remember_knowledge returns empty string when embedding fails."""
        broken_fn = MagicMock(side_effect=ConnectionError("Ollama down"))
        from quantstack.rag.query import remember_knowledge
        mock_conn = MagicMock()
        result = remember_knowledge(
            "test content", "trade_outcomes",
            conn=mock_conn, embedding_fn=broken_fn,
        )
        assert result == ""

    def test_search_returns_empty_when_db_fails(self):
        """search_knowledge_base returns empty when DB query fails."""
        embedding_fn = MagicMock(return_value=[[0.1] * 1024])
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.execute.side_effect = Exception("DB down")

        from quantstack.rag.query import search_knowledge_base
        result = search_knowledge_base(
            "test", collection="trade_outcomes",
            conn=mock_conn, embedding_fn=embedding_fn,
        )
        assert result == []
