# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for OPRO weekly prompt evolution loop."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from quantstack.db import run_migrations
from quantstack.optimization.opro_loop import (
    PROMOTION_THRESHOLD,
    PromptCandidate,
    OPROLoop,
)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    run_migrations(c)
    return c


@pytest.fixture
def loop(conn):
    return OPROLoop(conn)


def _make_candidate(
    node_name: str = "regime_classification",
    fitness: float = 0.5,
    status: str = "candidate",
    generation: int = 1,
    promoted_at: str | None = None,
) -> PromptCandidate:
    import uuid
    return PromptCandidate(
        candidate_id=str(uuid.uuid4()),
        node_name=node_name,
        prompt_text=f"Test prompt gen={generation}",
        generation=generation,
        fitness=fitness,
        status=status,
        created_at=datetime.now().isoformat(),
        promoted_at=promoted_at,
    )


class TestScoring:
    def test_no_data_returns_zero(self, loop):
        candidate = _make_candidate()
        score = loop.score_candidate(candidate, window_days=7)
        assert score == 0.0


class TestPromotion:
    def test_five_percent_improvement_promotes(self, loop):
        champion = _make_candidate(fitness=1.0, status="champion")
        candidate = _make_candidate(fitness=1.06)  # 6% better
        result = loop.promote_if_better(candidate, champion)
        assert result is not None
        assert result.status == "testing"

    def test_below_threshold_no_promotion(self, loop):
        champion = _make_candidate(fitness=1.0, status="champion")
        candidate = _make_candidate(fitness=1.04)  # 4% better — not enough
        result = loop.promote_if_better(candidate, champion)
        assert result is None

    def test_no_champion_promotes_first(self, loop):
        candidate = _make_candidate(fitness=0.3)
        result = loop.promote_if_better(candidate, None)
        assert result is not None
        assert result.status == "testing"

    def test_zero_fitness_champion_promotes_positive(self, loop):
        champion = _make_candidate(fitness=0.0, status="champion")
        candidate = _make_candidate(fitness=0.1)
        result = loop.promote_if_better(candidate, champion)
        assert result is not None


class TestTestingPeriod:
    def test_fresh_testing_not_complete(self, loop):
        candidate = _make_candidate(
            status="testing",
            promoted_at=datetime.now().isoformat(),
        )
        assert not loop._is_testing_period_complete(candidate)

    def test_old_testing_is_complete(self, loop):
        candidate = _make_candidate(
            status="testing",
            promoted_at=(datetime.now() - timedelta(days=8)).isoformat(),
        )
        assert loop._is_testing_period_complete(candidate)

    def test_no_promoted_at_not_complete(self, loop):
        candidate = _make_candidate(status="testing", promoted_at=None)
        assert not loop._is_testing_period_complete(candidate)


class TestRunWeekly:
    def test_no_nodes_returns_empty(self, loop):
        result = loop.run_weekly()
        assert result == []

    def test_with_critiques_runs(self, loop, conn):
        # Insert a critique so there's an active node
        conn.execute(
            "INSERT INTO prompt_critiques (id, trade_id, node_name, critique) "
            "VALUES (nextval('prompt_critiques_seq'), 1, 'regime_classification', 'Too slow to detect regime shift')"
        )
        # This will try LLM and likely fall back, but should not crash
        result = loop.run_weekly()
        assert isinstance(result, list)
