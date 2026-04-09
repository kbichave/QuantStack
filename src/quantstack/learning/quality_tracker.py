# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Agent Quality Tracking — DB-persisted quality scoring per agent per cycle.

Records direction accuracy, magnitude, and timing scores for each agent
decision cycle. Provides rolling quality windows, degradation alerts,
and a leaderboard for agent confidence weighting.

Design:
  - One row per (agent_name, cycle_id) in ``agent_quality_scores``.
  - Composite score = 0.50 * direction_correct + 0.30 * magnitude + 0.20 * timing.
  - Rolling window defaults to 21 cycles (one trading month).
  - Degradation alert fires when win_rate < 0.40 for 5+ consecutive cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from quantstack.db import pg_conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIRECTION_WEIGHT = 0.50
_MAGNITUDE_WEIGHT = 0.30
_TIMING_WEIGHT = 0.20

_DEGRADATION_WIN_RATE = 0.40
_DEGRADATION_MIN_CYCLES = 5

_DEFAULT_ROLLING_WINDOW = 21

TABLE = "agent_quality_scores"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              SERIAL PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    cycle_id        TEXT NOT NULL,
    direction_correct BOOLEAN NOT NULL,
    magnitude_score FLOAT NOT NULL,
    timing_score    FLOAT NOT NULL,
    composite_score FLOAT NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (agent_name, cycle_id)
);
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentQualityScore:
    """Single quality observation for one agent in one cycle."""

    agent_name: str
    cycle_id: str
    direction_correct: bool
    magnitude_score: float  # 0.0 – 1.0
    timing_score: float  # 0.0 – 1.0
    composite_score: float  # weighted average

    def __post_init__(self) -> None:
        if not 0.0 <= self.magnitude_score <= 1.0:
            raise ValueError(f"magnitude_score must be in [0, 1], got {self.magnitude_score}")
        if not 0.0 <= self.timing_score <= 1.0:
            raise ValueError(f"timing_score must be in [0, 1], got {self.timing_score}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_composite(
    direction_correct: bool,
    magnitude_score: float,
    timing_score: float,
) -> float:
    direction_val = 1.0 if direction_correct else 0.0
    return round(
        _DIRECTION_WEIGHT * direction_val
        + _MAGNITUDE_WEIGHT * magnitude_score
        + _TIMING_WEIGHT * timing_score,
        4,
    )


def _ensure_table() -> None:
    """Create the quality scores table if it doesn't exist."""
    with pg_conn() as conn:
        conn.execute(_CREATE_TABLE_SQL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_agent_quality(
    agent_name: str,
    cycle_id: str,
    direction_correct: bool,
    magnitude_score: float,
    timing_score: float,
) -> None:
    """Persist a quality observation for an agent cycle.

    Upserts on (agent_name, cycle_id) so replays are safe.
    """
    composite = _compute_composite(direction_correct, magnitude_score, timing_score)
    score = AgentQualityScore(
        agent_name=agent_name,
        cycle_id=cycle_id,
        direction_correct=direction_correct,
        magnitude_score=magnitude_score,
        timing_score=timing_score,
        composite_score=composite,
    )

    _ensure_table()
    with pg_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE}
                (agent_name, cycle_id, direction_correct, magnitude_score,
                 timing_score, composite_score, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (agent_name, cycle_id) DO UPDATE SET
                direction_correct = EXCLUDED.direction_correct,
                magnitude_score   = EXCLUDED.magnitude_score,
                timing_score      = EXCLUDED.timing_score,
                composite_score   = EXCLUDED.composite_score,
                recorded_at       = EXCLUDED.recorded_at
            """,
            (
                score.agent_name,
                score.cycle_id,
                score.direction_correct,
                score.magnitude_score,
                score.timing_score,
                score.composite_score,
                datetime.now(timezone.utc),
            ),
        )
    logger.info(
        "Recorded quality for {agent}@{cycle}: composite={comp:.3f}",
        agent=agent_name,
        cycle=cycle_id,
        comp=composite,
    )


def get_rolling_quality(
    agent_name: str,
    window: int = _DEFAULT_ROLLING_WINDOW,
) -> dict:
    """Return rolling quality stats for an agent over the last *window* cycles.

    Returns:
        dict with keys: win_rate, avg_composite, trend.
        ``trend`` is one of "improving", "stable", "degrading" based on
        comparing the first-half vs second-half composite averages.
    """
    _ensure_table()
    with pg_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT direction_correct, composite_score
            FROM {TABLE}
            WHERE agent_name = %s
            ORDER BY recorded_at DESC
            LIMIT %s
            """,
            (agent_name, window),
        ).fetchall()

    if not rows:
        return {"win_rate": 0.0, "avg_composite": 0.0, "trend": "stable"}

    wins = sum(1 for r in rows if r[0])
    win_rate = round(wins / len(rows), 4)
    avg_composite = round(sum(r[1] for r in rows) / len(rows), 4)

    # Trend: compare older half vs newer half of the window.
    # rows are ordered newest-first, so first half = recent, second half = older.
    mid = len(rows) // 2
    if mid < 2:
        trend = "stable"
    else:
        recent_avg = sum(r[1] for r in rows[:mid]) / mid
        older_avg = sum(r[1] for r in rows[mid:]) / (len(rows) - mid)
        delta = recent_avg - older_avg
        if delta > 0.05:
            trend = "improving"
        elif delta < -0.05:
            trend = "degrading"
        else:
            trend = "stable"

    return {"win_rate": win_rate, "avg_composite": avg_composite, "trend": trend}


def check_degradation_alerts() -> list[dict]:
    """Return agents with win_rate < 0.40 over their last 5+ consecutive cycles.

    Each dict contains: agent_name, win_rate, consecutive_cycles, avg_composite.
    """
    _ensure_table()
    with pg_conn() as conn:
        agents = conn.execute(
            f"SELECT DISTINCT agent_name FROM {TABLE}"
        ).fetchall()

    alerts: list[dict] = []
    for (agent_name,) in agents:
        with pg_conn() as conn:
            recent = conn.execute(
                f"""
                SELECT direction_correct, composite_score
                FROM {TABLE}
                WHERE agent_name = %s
                ORDER BY recorded_at DESC
                LIMIT %s
                """,
                (agent_name, _DEGRADATION_MIN_CYCLES),
            ).fetchall()

        if len(recent) < _DEGRADATION_MIN_CYCLES:
            continue

        wins = sum(1 for r in recent if r[0])
        win_rate = wins / len(recent)

        if win_rate < _DEGRADATION_WIN_RATE:
            avg_composite = sum(r[1] for r in recent) / len(recent)
            alerts.append({
                "agent_name": agent_name,
                "win_rate": round(win_rate, 4),
                "consecutive_cycles": len(recent),
                "avg_composite": round(avg_composite, 4),
            })
            logger.warning(
                "Degradation alert: {agent} win_rate={wr:.1%} over {n} cycles",
                agent=agent_name,
                wr=win_rate,
                n=len(recent),
            )

    return alerts


def get_quality_leaderboard() -> list[dict]:
    """All agents ranked by rolling composite score (descending).

    Returns list of dicts with: agent_name, avg_composite, win_rate, sample_count.
    """
    _ensure_table()
    with pg_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                agent_name,
                AVG(composite_score) AS avg_composite,
                AVG(direction_correct::int) AS win_rate,
                COUNT(*) AS sample_count
            FROM (
                SELECT agent_name, composite_score, direction_correct,
                       ROW_NUMBER() OVER (
                           PARTITION BY agent_name ORDER BY recorded_at DESC
                       ) AS rn
                FROM {TABLE}
            ) sub
            WHERE rn <= {_DEFAULT_ROLLING_WINDOW}
            GROUP BY agent_name
            ORDER BY avg_composite DESC
            """
        ).fetchall()

    return [
        {
            "agent_name": r[0],
            "avg_composite": round(r[1], 4),
            "win_rate": round(r[2], 4),
            "sample_count": r[3],
        }
        for r in rows
    ]
