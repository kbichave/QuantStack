# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Research Prioritization — scored ranking to replace FIFO queue.

Scores research candidates by a weighted formula:
    priority = alpha_uplift * 0.35
             + portfolio_gap * 0.25
             + failure_freq  * 0.20
             + staleness     * 0.20

Higher scores surface first. This ensures the research graph works on
the highest-impact hypotheses rather than whatever was queued last.

Table: ``research_candidates`` stores hypothesis definitions and outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from quantstack.db import pg_conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W_ALPHA_UPLIFT = 0.35
_W_PORTFOLIO_GAP = 0.25
_W_FAILURE_FREQ = 0.20
_W_STALENESS = 0.20

TABLE = "research_candidates"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    hypothesis_id     TEXT PRIMARY KEY,
    description       TEXT NOT NULL,
    alpha_uplift_est  FLOAT NOT NULL DEFAULT 0.0,
    portfolio_gap_score FLOAT NOT NULL DEFAULT 0.0,
    failure_frequency FLOAT NOT NULL DEFAULT 0.0,
    staleness_days    FLOAT NOT NULL DEFAULT 0.0,
    priority_score    FLOAT NOT NULL DEFAULT 0.0,
    status            TEXT NOT NULL DEFAULT 'pending',
    outcome           TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at      TIMESTAMPTZ
);
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResearchCandidate:
    """A scored research hypothesis."""

    hypothesis_id: str
    description: str
    alpha_uplift_est: float = 0.0
    portfolio_gap_score: float = 0.0
    failure_frequency: float = 0.0
    staleness_days: float = 0.0
    priority_score: float = field(default=0.0, init=False)

    def compute_priority(self) -> float:
        """Compute and store the weighted priority score."""
        self.priority_score = round(
            _W_ALPHA_UPLIFT * self.alpha_uplift_est
            + _W_PORTFOLIO_GAP * self.portfolio_gap_score
            + _W_FAILURE_FREQ * self.failure_frequency
            + _W_STALENESS * self.staleness_days,
            4,
        )
        return self.priority_score


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ensure_table() -> None:
    with pg_conn() as conn:
        conn.execute(_CREATE_TABLE_SQL)


def _row_to_candidate(row: tuple) -> ResearchCandidate:
    """Map a DB row to a ResearchCandidate."""
    candidate = ResearchCandidate(
        hypothesis_id=row[0],
        description=row[1],
        alpha_uplift_est=row[2],
        portfolio_gap_score=row[3],
        failure_frequency=row[4],
        staleness_days=row[5],
    )
    candidate.priority_score = row[6]
    return candidate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_research_candidates(
    candidates: list[ResearchCandidate],
) -> list[ResearchCandidate]:
    """Score and sort candidates by priority (descending).

    Mutates each candidate's ``priority_score`` in place and returns
    the list sorted highest-first.
    """
    for c in candidates:
        c.compute_priority()
    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return candidates


def enqueue_candidate(candidate: ResearchCandidate) -> None:
    """Insert or update a research candidate in the DB.

    Computes priority before persisting.
    """
    candidate.compute_priority()
    _ensure_table()
    with pg_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE}
                (hypothesis_id, description, alpha_uplift_est, portfolio_gap_score,
                 failure_frequency, staleness_days, priority_score, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s)
            ON CONFLICT (hypothesis_id) DO UPDATE SET
                description       = EXCLUDED.description,
                alpha_uplift_est  = EXCLUDED.alpha_uplift_est,
                portfolio_gap_score = EXCLUDED.portfolio_gap_score,
                failure_frequency = EXCLUDED.failure_frequency,
                staleness_days    = EXCLUDED.staleness_days,
                priority_score    = EXCLUDED.priority_score
            """,
            (
                candidate.hypothesis_id,
                candidate.description,
                candidate.alpha_uplift_est,
                candidate.portfolio_gap_score,
                candidate.failure_frequency,
                candidate.staleness_days,
                candidate.priority_score,
                datetime.now(timezone.utc),
            ),
        )
    logger.info(
        "Enqueued research candidate {hid} with priority={p:.4f}",
        hid=candidate.hypothesis_id,
        p=candidate.priority_score,
    )


def get_next_research_task() -> ResearchCandidate | None:
    """Query DB for the highest-priority unstarted candidate.

    Re-scores all pending candidates before selecting, so staleness
    and other dynamic factors are up to date.

    Returns None if no pending candidates exist.
    """
    _ensure_table()
    with pg_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT hypothesis_id, description, alpha_uplift_est,
                   portfolio_gap_score, failure_frequency, staleness_days,
                   priority_score
            FROM {TABLE}
            WHERE status = 'pending'
            ORDER BY priority_score DESC
            """,
        ).fetchall()

    if not rows:
        return None

    # Re-score in memory and pick the best
    candidates = [_row_to_candidate(r) for r in rows]
    scored = score_research_candidates(candidates)
    winner = scored[0]

    # Mark as in-progress
    with pg_conn() as conn:
        conn.execute(
            f"UPDATE {TABLE} SET status = 'in_progress' WHERE hypothesis_id = %s",
            (winner.hypothesis_id,),
        )

    logger.info(
        "Next research task: {hid} (priority={p:.4f})",
        hid=winner.hypothesis_id,
        p=winner.priority_score,
    )
    return winner


def mark_research_complete(hypothesis_id: str, outcome: str) -> None:
    """Mark a research candidate as completed with its outcome.

    Parameters
    ----------
    hypothesis_id:
        The candidate to mark.
    outcome:
        Free-text outcome description (e.g., "validated — IC=0.04, added to pipeline"
        or "rejected — no significant alpha after purged CV").
    """
    _ensure_table()
    with pg_conn() as conn:
        conn.execute(
            f"""
            UPDATE {TABLE}
            SET status = 'completed', outcome = %s, completed_at = %s
            WHERE hypothesis_id = %s
            """,
            (outcome, datetime.now(timezone.utc), hypothesis_id),
        )
    logger.info(
        "Research {hid} completed: {outcome}",
        hid=hypothesis_id,
        outcome=outcome[:80],
    )
