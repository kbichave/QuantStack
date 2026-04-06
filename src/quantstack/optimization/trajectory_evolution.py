# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trajectory-level evolutionary optimization for research runs.

Each nightly research run is a "trajectory" composed of segments
(hypothesis generation, feature selection, model config, backtest
evaluation). Monthly, we crossover the best segments from different
trajectories and mutate to discover new alpha.

Paper: QuantaAlpha (2026) — https://arxiv.org/abs/2602.07085

Schedule: Monthly (1st Saturday), via ResearchOrchestrator.run_monthly().
STATUS: CONDITIONAL. Activates when closed trade count >= 200 AND
research trajectories >= 10 (lowered from 500 trades).
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

SEGMENT_TYPES = ("hypothesis", "feature_selection", "model_config", "backtest_eval")


@dataclass
class TrajectorySegment:
    """One step in a research trajectory."""
    segment_type: str       # One of SEGMENT_TYPES
    content: dict           # Serialized configuration for this step
    fitness: float = 0.0    # Walk-forward Sharpe of the strategy this contributed to
    source_trajectory_id: str = ""


@dataclass
class Trajectory:
    """A complete research run composed of segments."""
    trajectory_id: str
    generation: int
    segments: list[TrajectorySegment] = field(default_factory=list)
    overall_fitness: float = 0.0
    parent_ids: list[str] = field(default_factory=list)
    mutation_applied: str = ""
    created_at: str = ""


TRAJECTORIES_TABLE = "research_trajectories"


# ---------------------------------------------------------------------------
# TrajectoryEvolution
# ---------------------------------------------------------------------------

class TrajectoryEvolution:
    """Monthly evolutionary optimization over research trajectories.

    Usage:
        evo = TrajectoryEvolution(conn)
        new_gen = evo.run_monthly(population_size=20)
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Monthly run
    # ------------------------------------------------------------------

    def run_monthly(self, population_size: int = 20) -> list[Trajectory]:
        """Execute one generation of trajectory evolution.

        1. Load all trajectories from last month
        2. Rank by fitness (walk-forward Sharpe)
        3. Select top-5 as parents
        4. Crossover pairs to produce offspring
        5. Mutate offspring
        6. Persist new generation
        """
        current_gen = self._get_current_generation()
        parents = self._load_top_trajectories(generation=current_gen, top_k=5)

        if len(parents) < 2:
            logger.info(
                f"[TrajectoryEvolution] Only {len(parents)} trajectories in "
                f"gen {current_gen}, need >=2 for crossover. Recording current runs."
            )
            return []

        new_gen = current_gen + 1
        offspring = []

        # Crossover all pairs of top parents
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                child = self.crossover(parents[i], parents[j])
                child.generation = new_gen
                offspring.append(child)

                if len(offspring) >= population_size:
                    break
            if len(offspring) >= population_size:
                break

        # Mutate a fraction of offspring
        for child in offspring:
            if random.random() < 0.3:  # 30% mutation rate
                child = self.mutate(child)

        # Persist
        for child in offspring:
            self._persist_trajectory(child)

        logger.info(
            f"[TrajectoryEvolution] Gen {new_gen}: {len(offspring)} offspring "
            f"from {len(parents)} parents"
        )
        return offspring

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(self, parent_a: Trajectory, parent_b: Trajectory) -> Trajectory:
        """Create offspring by taking the best segment from each parent by type.

        For each segment type, picks the segment with higher fitness.
        """
        child_segments = []

        for seg_type in SEGMENT_TYPES:
            seg_a = next((s for s in parent_a.segments if s.segment_type == seg_type), None)
            seg_b = next((s for s in parent_b.segments if s.segment_type == seg_type), None)

            if seg_a and seg_b:
                winner = seg_a if seg_a.fitness >= seg_b.fitness else seg_b
                child_segments.append(TrajectorySegment(
                    segment_type=seg_type,
                    content=copy.deepcopy(winner.content),
                    fitness=winner.fitness,
                    source_trajectory_id=winner.source_trajectory_id,
                ))
            elif seg_a:
                child_segments.append(copy.deepcopy(seg_a))
            elif seg_b:
                child_segments.append(copy.deepcopy(seg_b))

        child = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            generation=0,  # Set by caller
            segments=child_segments,
            overall_fitness=0.0,
            parent_ids=[parent_a.trajectory_id, parent_b.trajectory_id],
            mutation_applied="",
            created_at=datetime.now().isoformat(),
        )

        # Estimate fitness as average of segment fitnesses
        if child_segments:
            child.overall_fitness = sum(s.fitness for s in child_segments) / len(child_segments)

        return child

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, trajectory: Trajectory, mutation_rate: float = 0.2) -> Trajectory:
        """Randomly perturb one segment in the trajectory.

        Mutations:
        - hypothesis: change regime target
        - feature_selection: add or remove a feature
        - model_config: adjust a hyperparameter
        - backtest_eval: change evaluation window
        """
        if not trajectory.segments:
            return trajectory

        # Pick a random segment to mutate
        idx = random.randint(0, len(trajectory.segments) - 1)
        segment = trajectory.segments[idx]
        content = copy.deepcopy(segment.content)

        mutation_desc = ""

        if segment.segment_type == "hypothesis":
            regimes = ["trending_up", "trending_down", "ranging"]
            current = content.get("regime_target", "trending_up")
            options = [r for r in regimes if r != current]
            if options:
                content["regime_target"] = random.choice(options)
                mutation_desc = f"hypothesis:regime_target={content['regime_target']}"

        elif segment.segment_type == "feature_selection":
            features = content.get("features", [])
            candidate_features = [
                "rsi_14", "macd_histogram", "bb_width", "adx_14",
                "natr", "obv", "vwap_ratio", "atr_14", "cci_20",
            ]
            if features and random.random() < 0.5:
                # Remove a random feature
                removed = features.pop(random.randint(0, len(features) - 1))
                mutation_desc = f"feature_selection:removed={removed}"
            else:
                # Add a random feature
                new_feat = random.choice(candidate_features)
                if new_feat not in features:
                    features.append(new_feat)
                    mutation_desc = f"feature_selection:added={new_feat}"
            content["features"] = features

        elif segment.segment_type == "model_config":
            # Perturb a numeric hyperparameter by ±20%
            for key, val in content.items():
                if isinstance(val, (int, float)) and val > 0:
                    factor = 1.0 + random.uniform(-0.2, 0.2)
                    content[key] = type(val)(val * factor)
                    mutation_desc = f"model_config:{key}={content[key]}"
                    break

        elif segment.segment_type == "backtest_eval":
            # Change evaluation window
            windows = [252, 504, 756, 1260]
            current = content.get("eval_window", 504)
            options = [w for w in windows if w != current]
            if options:
                content["eval_window"] = random.choice(options)
                mutation_desc = f"backtest_eval:window={content['eval_window']}"

        trajectory.segments[idx] = TrajectorySegment(
            segment_type=segment.segment_type,
            content=content,
            fitness=segment.fitness * 0.9,  # Discount mutated fitness
            source_trajectory_id=segment.source_trajectory_id,
        )
        trajectory.mutation_applied = mutation_desc

        return trajectory

    # ------------------------------------------------------------------
    # Record a research run as a trajectory
    # ------------------------------------------------------------------

    def record_trajectory(
        self,
        segments: list[TrajectorySegment],
        overall_fitness: float = 0.0,
    ) -> Trajectory:
        """Record a nightly research run as a trajectory for future evolution."""
        generation = self._get_current_generation()
        trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            generation=generation,
            segments=segments,
            overall_fitness=overall_fitness,
            created_at=datetime.now().isoformat(),
        )
        self._persist_trajectory(trajectory)
        return trajectory

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_current_generation(self) -> int:
        try:
            row = self._conn.execute(
                f"SELECT MAX(generation) FROM {TRAJECTORIES_TABLE}"
            ).fetchone()
            return (row[0] or 0) if row else 0
        except Exception as exc:
            logger.warning(f"[TrajectoryEvolution] Failed to get current generation: {exc}")
            return 0

    def _load_top_trajectories(self, generation: int, top_k: int = 5) -> list[Trajectory]:
        try:
            rows = self._conn.execute(
                f"SELECT trajectory_id, generation, segments, overall_fitness, "
                f"parent_ids, mutation_applied, created_at "
                f"FROM {TRAJECTORIES_TABLE} "
                f"WHERE generation = ? "
                f"ORDER BY overall_fitness DESC "
                f"LIMIT ?",
                [generation, top_k],
            ).fetchall()
            trajectories = []
            for r in rows:
                segments_raw = json.loads(r[2]) if r[2] else []
                segments = [
                    TrajectorySegment(
                        segment_type=s.get("segment_type", ""),
                        content=s.get("content", {}),
                        fitness=s.get("fitness", 0.0),
                        source_trajectory_id=s.get("source_trajectory_id", ""),
                    )
                    for s in segments_raw
                ]
                trajectories.append(Trajectory(
                    trajectory_id=r[0],
                    generation=r[1],
                    segments=segments,
                    overall_fitness=r[3] or 0.0,
                    parent_ids=json.loads(r[4]) if r[4] else [],
                    mutation_applied=r[5] or "",
                    created_at=str(r[6] or ""),
                ))
            return trajectories
        except Exception as exc:
            logger.warning(f"[TrajectoryEvolution] Load failed: {exc}")
            return []

    def _persist_trajectory(self, trajectory: Trajectory) -> None:
        try:
            segments_json = json.dumps([
                {
                    "segment_type": s.segment_type,
                    "content": s.content,
                    "fitness": s.fitness,
                    "source_trajectory_id": s.source_trajectory_id,
                }
                for s in trajectory.segments
            ])
            self._conn.execute(
                f"INSERT INTO {TRAJECTORIES_TABLE} "
                f"(trajectory_id, generation, segments, overall_fitness, "
                f"parent_ids, mutation_applied, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    trajectory.trajectory_id, trajectory.generation,
                    segments_json, trajectory.overall_fitness,
                    json.dumps(trajectory.parent_ids),
                    trajectory.mutation_applied,
                    trajectory.created_at or datetime.now().isoformat(),
                ],
            )
        except Exception as exc:
            logger.warning(f"[TrajectoryEvolution] Persist failed: {exc}")
