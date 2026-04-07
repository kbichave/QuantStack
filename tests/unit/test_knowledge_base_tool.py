"""Tests for search_knowledge_base tool using RAG semantic search.

Verifies the tool delegates to the RAG module, maps the output schema
correctly, and handles Ollama unavailability gracefully.
"""

import json
from unittest.mock import patch, AsyncMock

import pytest


@pytest.fixture
def _mock_rag_search():
    """Patch the RAG search function used by the tool."""
    with patch("quantstack.tools.langchain.learning_tools.rag_search") as mock:
        yield mock


class TestSearchKnowledgeBaseTool:

    @pytest.mark.asyncio
    async def test_calls_rag_search_with_query(self, _mock_rag_search):
        _mock_rag_search.return_value = []

        from quantstack.tools.langchain.learning_tools import search_knowledge_base

        await search_knowledge_base.ainvoke({"query": "momentum strategy", "top_k": 3})

        _mock_rag_search.assert_called_once_with(query="momentum strategy", n_results=3)

    @pytest.mark.asyncio
    async def test_maps_rag_output_to_tool_contract(self, _mock_rag_search):
        _mock_rag_search.return_value = [
            {
                "text": "momentum lesson learned",
                "metadata": {"id": "abc123", "created_at": "2026-01-01"},
                "distance": 0.12,
                "collection": "strategy_knowledge",
            }
        ]

        from quantstack.tools.langchain.learning_tools import search_knowledge_base

        result = json.loads(
            await search_knowledge_base.ainvoke({"query": "momentum", "top_k": 5})
        )

        assert result["count"] == 1
        entry = result["results"][0]
        assert entry["content"] == "momentum lesson learned"
        assert entry["category"] == "strategy_knowledge"
        assert entry["id"] == "abc123"
        assert entry["created_at"] == "2026-01-01"
        assert entry["distance"] == 0.12
        assert entry["metadata"] == {"id": "abc123", "created_at": "2026-01-01"}

    @pytest.mark.asyncio
    async def test_empty_results_return_empty_list(self, _mock_rag_search):
        _mock_rag_search.return_value = []

        from quantstack.tools.langchain.learning_tools import search_knowledge_base

        result = json.loads(
            await search_knowledge_base.ainvoke({"query": "nonexistent topic", "top_k": 5})
        )

        assert result["results"] == []
        assert result["count"] == 0
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_handles_ollama_unavailability(self, _mock_rag_search):
        _mock_rag_search.side_effect = ConnectionError("Connection refused")

        from quantstack.tools.langchain.learning_tools import search_knowledge_base

        result = json.loads(
            await search_knowledge_base.ainvoke({"query": "test query", "top_k": 5})
        )

        assert "error" in result
        assert "embedding" in result["error"].lower() or "unavailable" in result["error"].lower()
