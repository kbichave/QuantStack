# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Alpha Knowledge Graph.

All tests run WITHOUT a database — db_conn() is mocked everywhere.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Model tests (pure, no DB)
# ---------------------------------------------------------------------------

from quantstack.knowledge.kg_models import (
    HypothesisResult,
    NoveltyResult,
    OverlapResult,
    SimilarHypothesis,
)


class TestKGModels:
    """Pydantic model validation tests."""

    def test_similar_hypothesis_creation(self):
        sh = SimilarHypothesis(
            hypothesis_id="h1", name="momentum works", cosine_similarity=0.92,
        )
        assert sh.hypothesis_id == "h1"
        assert sh.cosine_similarity == 0.92
        assert sh.regime is None
        assert sh.outcome is None

    def test_similar_hypothesis_with_optional_fields(self):
        sh = SimilarHypothesis(
            hypothesis_id="h2", name="mean reversion", cosine_similarity=0.75,
            regime="ranging", outcome="positive",
        )
        assert sh.regime == "ranging"
        assert sh.outcome == "positive"

    def test_novelty_result_novel(self):
        nr = NoveltyResult(is_novel=True, similar_hypotheses=[], recommendation="novel")
        assert nr.is_novel is True
        assert nr.recommendation == "novel"
        assert len(nr.similar_hypotheses) == 0

    def test_novelty_result_redundant(self):
        sh = SimilarHypothesis(hypothesis_id="h1", name="x", cosine_similarity=0.95)
        nr = NoveltyResult(
            is_novel=False, similar_hypotheses=[sh], recommendation="redundant",
        )
        assert nr.is_novel is False
        assert nr.recommendation == "redundant"

    def test_novelty_result_similar_but_different_regime(self):
        nr = NoveltyResult(
            is_novel=True, similar_hypotheses=[],
            recommendation="similar_but_different_regime",
        )
        assert nr.recommendation == "similar_but_different_regime"

    def test_overlap_result_not_crowded(self):
        result = OverlapResult(
            is_crowded=False, shared_factor_count=1,
            shared_factors=["momentum"], affected_strategies=["strat_b"],
        )
        assert result.is_crowded is False
        assert result.shared_factor_count == 1

    def test_overlap_result_crowded(self):
        result = OverlapResult(
            is_crowded=True, shared_factor_count=4,
            shared_factors=["momentum", "value", "size", "quality"],
            affected_strategies=["strat_a", "strat_b"],
        )
        assert result.is_crowded is True
        assert len(result.shared_factors) == 4

    def test_hypothesis_result_minimal(self):
        hr = HypothesisResult(
            hypothesis_id="h1", hypothesis_text="test", test_date="2026-01-01",
            outcome="positive",
        )
        assert hr.result_sharpe is None
        assert hr.result_ic is None
        assert hr.regime_at_test is None

    def test_hypothesis_result_full(self):
        hr = HypothesisResult(
            hypothesis_id="h1", hypothesis_text="test", test_date="2026-01-01",
            outcome="positive", result_sharpe=1.5, result_ic=0.04,
            regime_at_test="trending_up",
        )
        assert hr.result_sharpe == 1.5
        assert hr.result_ic == 0.04

    def test_model_json_roundtrip(self):
        nr = NoveltyResult(
            is_novel=True,
            similar_hypotheses=[
                SimilarHypothesis(hypothesis_id="h1", name="x", cosine_similarity=0.8),
            ],
            recommendation="novel",
        )
        data = json.loads(nr.model_dump_json())
        nr2 = NoveltyResult(**data)
        assert nr2.is_novel is True
        assert len(nr2.similar_hypotheses) == 1


# ---------------------------------------------------------------------------
# Embedding tests (pure, no DB)
# ---------------------------------------------------------------------------

from quantstack.knowledge.embeddings import EMBEDDING_DIM, generate_embedding


class TestEmbeddings:
    """Tests for the deterministic hash-based embedding function."""

    def test_returns_correct_dimension(self):
        emb = generate_embedding("test hypothesis")
        assert len(emb) == EMBEDDING_DIM

    def test_returns_floats(self):
        emb = generate_embedding("test")
        assert all(isinstance(x, float) for x in emb)

    def test_deterministic(self):
        a = generate_embedding("momentum factor in equities")
        b = generate_embedding("momentum factor in equities")
        assert a == b

    def test_different_inputs_differ(self):
        a = generate_embedding("momentum factor")
        b = generate_embedding("mean reversion factor")
        assert a != b

    def test_case_insensitive(self):
        a = generate_embedding("Momentum Factor")
        b = generate_embedding("momentum factor")
        assert a == b

    def test_whitespace_normalised(self):
        a = generate_embedding("  test  ")
        b = generate_embedding("test")
        assert a == b

    def test_unit_vector(self):
        emb = generate_embedding("normalisation check")
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Helper: mock db_conn context manager
# ---------------------------------------------------------------------------

def _make_mock_conn(fetchone_val=None, fetchall_val=None):
    """Build a mock PgConnection that works as a context manager."""
    mock_conn = MagicMock()
    mock_conn.execute = MagicMock()
    mock_conn.fetchone = MagicMock(return_value=fetchone_val)
    mock_conn.fetchall = MagicMock(return_value=fetchall_val or [])
    return mock_conn


@contextmanager
def _mock_db_cm(mock_conn):
    """Wrap a mock conn as a context manager matching db_conn() protocol."""
    yield mock_conn


# ---------------------------------------------------------------------------
# KnowledgeGraph tests (DB mocked)
# ---------------------------------------------------------------------------

from quantstack.knowledge.graph import KnowledgeGraph


class TestKnowledgeGraphCreateNode:
    """create_node tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_create_node_returns_node_id(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchone_val={"node_id": "abc-123"})
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        node_id = kg.create_node("hypothesis", "test hyp", {"key": "val"})
        assert node_id == "abc-123"
        mock_conn.execute.assert_called_once()

    @patch("quantstack.knowledge.graph.db_conn")
    def test_create_node_with_embedding(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchone_val={"node_id": "emb-id"})
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        emb = generate_embedding("test")
        node_id = kg.create_node("hypothesis", "with embedding", {}, embedding=emb)
        assert node_id == "emb-id"
        # Verify the SQL contains the embedding vector cast
        call_args = mock_conn.execute.call_args
        assert "::vector" in call_args[0][0]


class TestKnowledgeGraphCreateEdge:
    """create_edge tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_create_edge_success(self, mock_db_conn):
        mock_conn = _make_mock_conn(
            fetchall_val=[{"node_id": "src"}, {"node_id": "tgt"}],
        )
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        edge_id = kg.create_edge("uses", "src", "tgt", weight=0.8)
        assert isinstance(edge_id, str)
        assert len(edge_id) == 36  # uuid4 format

    @patch("quantstack.knowledge.graph.db_conn")
    def test_create_edge_missing_node_raises(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[{"node_id": "src"}])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        with pytest.raises(ValueError, match="Node.*not found"):
            kg.create_edge("uses", "src", "missing")

    @patch("quantstack.knowledge.graph.db_conn")
    def test_create_edge_with_temporal_bounds(self, mock_db_conn):
        from datetime import date

        mock_conn = _make_mock_conn(
            fetchall_val=[{"node_id": "s"}, {"node_id": "t"}],
        )
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        edge_id = kg.create_edge(
            "uses", "s", "t",
            valid_from=date(2026, 1, 1), valid_to=date(2026, 12, 31),
        )
        assert isinstance(edge_id, str)
        # Verify valid_from/valid_to were passed to execute
        insert_call = mock_conn.execute.call_args_list[-1]
        params = insert_call[0][1]
        assert date(2026, 1, 1) in params
        assert date(2026, 12, 31) in params


class TestHypothesisNovelty:
    """check_hypothesis_novelty tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_novel_when_no_similar(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result = kg.check_hypothesis_novelty("completely new idea")
        assert isinstance(result, NoveltyResult)
        assert result.is_novel is True
        assert result.recommendation == "novel"

    @patch("quantstack.knowledge.graph.db_conn")
    def test_redundant_when_high_similarity_same_regime(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {
                "node_id": "h1", "name": "existing hyp",
                "properties": {"regime": "trending_up", "outcome": "positive"},
                "cosine_similarity": 0.92,
            },
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result = kg.check_hypothesis_novelty("similar hyp", regime="trending_up")
        assert result.is_novel is False
        assert result.recommendation == "redundant"

    @patch("quantstack.knowledge.graph.db_conn")
    def test_similar_but_different_regime(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {
                "node_id": "h1", "name": "existing hyp",
                "properties": {"regime": "ranging", "outcome": "positive"},
                "cosine_similarity": 0.90,
            },
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result = kg.check_hypothesis_novelty("similar hyp", regime="trending_up")
        assert result.is_novel is True
        assert result.recommendation == "similar_but_different_regime"

    @patch("quantstack.knowledge.graph.db_conn")
    def test_returns_correct_similar_hypotheses(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {"node_id": "h1", "name": "hyp1", "properties": {}, "cosine_similarity": 0.70},
            {"node_id": "h2", "name": "hyp2", "properties": {}, "cosine_similarity": 0.60},
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result = kg.check_hypothesis_novelty("test")
        assert len(result.similar_hypotheses) == 2
        assert result.similar_hypotheses[0].cosine_similarity == 0.70


class TestFactorOverlap:
    """check_factor_overlap tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_not_crowded_when_few_shared(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {"factor_name": "momentum", "other_strategy_name": "strat_b"},
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result = kg.check_factor_overlap("strat_a_id")
        assert isinstance(result, OverlapResult)
        assert result.is_crowded is False
        assert result.shared_factor_count == 1

    @patch("quantstack.knowledge.graph.db_conn")
    def test_crowded_when_many_shared(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {"factor_name": "momentum", "other_strategy_name": "strat_b"},
            {"factor_name": "value", "other_strategy_name": "strat_b"},
            {"factor_name": "size", "other_strategy_name": "strat_c"},
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result = kg.check_factor_overlap("strat_a_id")
        assert result.is_crowded is True
        assert result.shared_factor_count == 3


class TestRecordExperiment:
    """record_experiment tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_creates_all_nodes_and_edges(self, mock_db_conn):
        call_count = {"execute": 0}
        node_ids = iter(["hyp-id", "res-id", "fac-1", "fac-2"])

        def next_fetchone():
            try:
                return {"node_id": next(node_ids)}
            except StopIteration:
                return {"node_id": "fallback"}

        mock_conn = _make_mock_conn()
        mock_conn.fetchone = MagicMock(side_effect=next_fetchone)
        # Edge validation: return both source and target as found.
        # Since create_edge queries both IDs, we return a superset.
        mock_conn.fetchall = MagicMock(return_value=[
            {"node_id": "hyp-id"}, {"node_id": "res-id"},
            {"node_id": "fac-1"}, {"node_id": "fac-2"},
        ])

        original_execute = mock_conn.execute

        def counting_execute(*args, **kwargs):
            call_count["execute"] += 1
            return original_execute(*args, **kwargs)

        mock_conn.execute = counting_execute
        # Return a fresh context manager each time db_conn() is called
        mock_db_conn.side_effect = lambda: _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        result_id = kg.record_experiment(
            hypothesis="momentum works in trending markets",
            result={"outcome": "positive", "sharpe": 1.5, "ic": 0.05},
            factors_used=["momentum", "volume"],
            regime="trending_up",
        )
        assert isinstance(result_id, str)
        # Should have multiple execute calls:
        # 1 hypothesis node, 1 result node, 1 tested_by edge (validate + insert),
        # 2 factor nodes, 2 uses edges (validate + insert each)
        assert call_count["execute"] > 5


class TestResearchHistory:
    """get_research_history tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_returns_hypothesis_results(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {
                "node_id": "h1", "name": "momentum hyp",
                "properties": {
                    "test_date": "2026-01-15", "outcome": "positive",
                    "sharpe": 1.2, "ic": 0.03, "regime": "trending_up",
                },
                "cosine_similarity": 0.85,
            },
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        results = kg.get_research_history("momentum")
        assert len(results) == 1
        assert results[0].hypothesis_text == "momentum hyp"
        assert results[0].result_sharpe == 1.2

    @patch("quantstack.knowledge.graph.db_conn")
    def test_filters_by_regime(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[
            {
                "node_id": "h1", "name": "hyp1",
                "properties": {"test_date": "2026-01-15", "outcome": "positive", "regime": "trending_up"},
                "cosine_similarity": 0.85,
            },
            {
                "node_id": "h2", "name": "hyp2",
                "properties": {"test_date": "2026-01-16", "outcome": "negative", "regime": "ranging"},
                "cosine_similarity": 0.80,
            },
        ])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        kg = KnowledgeGraph()
        results = kg.get_research_history("test", regime="trending_up")
        assert len(results) == 1
        assert results[0].regime_at_test == "trending_up"


# ---------------------------------------------------------------------------
# Population / backfill tests
# ---------------------------------------------------------------------------

from quantstack.knowledge.population import (
    backfill_from_autoresearch,
    backfill_from_ic_observations,
    backfill_from_ml_experiments,
    backfill_from_strategies,
    run_full_backfill,
)


class TestPopulationBackfill:
    """Backfill pipeline tests — all handle missing tables gracefully."""

    @patch("quantstack.knowledge.population._table_exists", return_value=False)
    def test_strategies_missing_table_returns_zero(self, _mock):
        assert backfill_from_strategies() == 0

    @patch("quantstack.knowledge.population._table_exists", return_value=False)
    def test_ml_experiments_missing_table_returns_zero(self, _mock):
        assert backfill_from_ml_experiments() == 0

    @patch("quantstack.knowledge.population._table_exists", return_value=False)
    def test_autoresearch_missing_table_returns_zero(self, _mock):
        assert backfill_from_autoresearch() == 0

    @patch("quantstack.knowledge.population._table_exists", return_value=False)
    def test_ic_observations_missing_table_returns_zero(self, _mock):
        assert backfill_from_ic_observations() == 0

    @patch("quantstack.knowledge.population._table_exists", return_value=False)
    def test_full_backfill_all_missing(self, _mock):
        result = run_full_backfill()
        assert result == {
            "strategies": 0,
            "ml_experiments": 0,
            "autoresearch": 0,
            "ic_observations": 0,
        }


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestKnowledgeGraphTools:
    """LLM-facing tool wrapper tests."""

    @patch("quantstack.knowledge.graph.db_conn")
    def test_check_hypothesis_novelty_returns_json(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        from quantstack.tools.langchain.knowledge_graph_tools import (
            check_hypothesis_novelty,
        )
        result = check_hypothesis_novelty.invoke({"hypothesis_text": "new idea"})
        parsed = json.loads(result)
        assert "is_novel" in parsed
        assert "recommendation" in parsed

    @patch("quantstack.knowledge.graph.db_conn")
    def test_check_factor_overlap_returns_json(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        from quantstack.tools.langchain.knowledge_graph_tools import (
            check_factor_overlap,
        )
        result = check_factor_overlap.invoke({"strategy_id": "strat-123"})
        parsed = json.loads(result)
        assert "is_crowded" in parsed
        assert "shared_factors" in parsed

    @patch("quantstack.knowledge.graph.db_conn")
    def test_get_research_history_returns_json(self, mock_db_conn):
        mock_conn = _make_mock_conn(fetchall_val=[])
        mock_db_conn.return_value = _mock_db_cm(mock_conn)

        from quantstack.tools.langchain.knowledge_graph_tools import (
            get_research_history,
        )
        result = get_research_history.invoke({"topic": "momentum"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @patch("quantstack.knowledge.graph.db_conn")
    def test_record_experiment_returns_json(self, mock_db_conn):
        node_ids = iter(["hyp-id", "res-id", "fac-1", "fac-2"])

        def next_fetchone():
            try:
                return {"node_id": next(node_ids)}
            except StopIteration:
                return {"node_id": "fallback"}

        mock_conn = _make_mock_conn()
        mock_conn.fetchone = MagicMock(side_effect=next_fetchone)
        mock_conn.fetchall = MagicMock(return_value=[
            {"node_id": "hyp-id"}, {"node_id": "res-id"},
            {"node_id": "fac-1"}, {"node_id": "fac-2"},
        ])
        mock_db_conn.side_effect = lambda: _mock_db_cm(mock_conn)

        from quantstack.tools.langchain.knowledge_graph_tools import record_experiment
        result = record_experiment.invoke({
            "hypothesis": "test hypothesis",
            "result_json": '{"outcome": "positive", "sharpe": 1.2}',
            "factors_used": "momentum, volume",
            "regime": "trending_up",
        })
        parsed = json.loads(result)
        assert "hypothesis_id" in parsed
        assert parsed["status"] == "recorded"
