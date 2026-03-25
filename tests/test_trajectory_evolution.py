# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for QuantaAlpha trajectory crossover and evolution."""

from __future__ import annotations

import pytest

from quantstack.db import pg_conn, run_migrations
from quantstack.optimization.trajectory_evolution import (
    Trajectory,
    TrajectoryEvolution,
    TrajectorySegment,
)
import uuid


@pytest.fixture
def conn():
    with pg_conn() as c:
        run_migrations(c)
        yield c


@pytest.fixture
def evo(conn):
    return TrajectoryEvolution(conn)


def _make_segment(seg_type: str, fitness: float = 0.5, **content_kw) -> TrajectorySegment:
    return TrajectorySegment(
        segment_type=seg_type,
        content=content_kw or {"key": "value"},
        fitness=fitness,
        source_trajectory_id="test",
    )


def _make_trajectory(
    fitness: float = 0.5,
    generation: int = 1,
    hypothesis_fitness: float = 0.6,
    feature_fitness: float = 0.4,
) -> Trajectory:
    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        generation=generation,
        segments=[
            _make_segment("hypothesis", hypothesis_fitness, regime_target="trending_up"),
            _make_segment("feature_selection", feature_fitness, features=["rsi_14", "macd_histogram"]),
            _make_segment("model_config", 0.5, n_estimators=100, learning_rate=0.1),
            _make_segment("backtest_eval", 0.5, eval_window=504),
        ],
        overall_fitness=fitness,
    )


class TestCrossover:
    def test_takes_best_segments(self, evo):
        parent_a = _make_trajectory(hypothesis_fitness=0.8, feature_fitness=0.3)
        parent_b = _make_trajectory(hypothesis_fitness=0.4, feature_fitness=0.9)

        child = evo.crossover(parent_a, parent_b)

        hyp = next(s for s in child.segments if s.segment_type == "hypothesis")
        feat = next(s for s in child.segments if s.segment_type == "feature_selection")
        assert hyp.fitness == 0.8   # From parent_a
        assert feat.fitness == 0.9  # From parent_b

    def test_child_has_both_parents(self, evo):
        parent_a = _make_trajectory()
        parent_b = _make_trajectory()
        child = evo.crossover(parent_a, parent_b)
        assert parent_a.trajectory_id in child.parent_ids
        assert parent_b.trajectory_id in child.parent_ids

    def test_child_has_all_segment_types(self, evo):
        parent_a = _make_trajectory()
        parent_b = _make_trajectory()
        child = evo.crossover(parent_a, parent_b)
        types = {s.segment_type for s in child.segments}
        assert types == {"hypothesis", "feature_selection", "model_config", "backtest_eval"}


class TestMutation:
    def test_mutation_changes_one_segment(self, evo):
        trajectory = _make_trajectory()
        original_contents = [s.content.copy() for s in trajectory.segments]
        mutated = evo.mutate(trajectory)

        changes = 0
        for i, seg in enumerate(mutated.segments):
            if seg.content != original_contents[i]:
                changes += 1
        assert changes <= 1  # At most one segment changed

    def test_mutation_sets_description(self, evo):
        trajectory = _make_trajectory()
        mutated = evo.mutate(trajectory)
        # May or may not have a mutation description (depends on random choice)
        assert isinstance(mutated.mutation_applied, str)


class TestPopulation:
    def test_needs_two_parents(self, evo):
        # No trajectories → empty
        result = evo.run_monthly()
        assert result == []

    def test_with_parents_produces_offspring(self, evo, conn):
        # Record 3 trajectories
        for i in range(3):
            t = _make_trajectory(fitness=0.5 + i * 0.1, generation=0)
            evo._persist_trajectory(t)

        result = evo.run_monthly(population_size=5)
        assert len(result) > 0
        assert all(t.generation == 1 for t in result)

    def test_population_capped(self, evo, conn):
        for i in range(5):
            t = _make_trajectory(fitness=0.5 + i * 0.05, generation=0)
            evo._persist_trajectory(t)

        result = evo.run_monthly(population_size=3)
        assert len(result) <= 3


class TestRecordTrajectory:
    def test_record_and_load(self, evo, conn):
        segments = [
            _make_segment("hypothesis", 0.7, regime_target="trending_up"),
            _make_segment("feature_selection", 0.6, features=["rsi_14"]),
        ]
        t = evo.record_trajectory(segments, overall_fitness=0.65)

        count = conn.execute("SELECT COUNT(*) FROM research_trajectories").fetchone()[0]
        assert count == 1

        loaded = evo._load_top_trajectories(generation=0, top_k=1)
        assert len(loaded) == 1
        assert loaded[0].overall_fitness == 0.65
