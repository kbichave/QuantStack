# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Prompt A/B Testing — statistical comparison of prompt variants per agent.

Supports running multiple prompt variants for each agent simultaneously,
recording per-variant outcome scores, and promoting winners via t-test
once sufficient samples are collected.

Lifecycle:
  1. Configure variants for an agent (one marked ``is_control``).
  2. Route each cycle to a random variant, record the outcome score.
  3. Call ``evaluate_ab_test`` periodically — returns a winner once
     ``min_samples`` are met and p-value < ``significance_level``.
  4. ``promote_winning_variant`` archives the old control and marks
     the winner as the new baseline.

Tables:
  - ``prompt_ab_variants``: variant definitions
  - ``prompt_ab_results``: per-observation scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger
from scipy.stats import ttest_ind

from quantstack.db import pg_conn

# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

VARIANTS_TABLE = "prompt_ab_variants"
RESULTS_TABLE = "prompt_ab_results"

_CREATE_VARIANTS_SQL = f"""
CREATE TABLE IF NOT EXISTS {VARIANTS_TABLE} (
    variant_id   TEXT NOT NULL,
    agent_name   TEXT NOT NULL,
    prompt_template TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_control   BOOLEAN NOT NULL DEFAULT FALSE,
    archived     BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (agent_name, variant_id)
);
"""

_CREATE_RESULTS_SQL = f"""
CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
    id           SERIAL PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    variant_id   TEXT NOT NULL,
    score        FLOAT NOT NULL,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptVariant:
    """A single prompt variant for an agent."""

    variant_id: str
    agent_name: str
    prompt_template: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_control: bool = False


@dataclass
class ABTestConfig:
    """Configuration for an A/B test on a single agent."""

    agent_name: str
    variants: list[PromptVariant]
    min_samples: int = 50
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        controls = [v for v in self.variants if v.is_control]
        if len(controls) != 1:
            raise ValueError(
                f"Exactly one variant must be marked is_control, got {len(controls)}"
            )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ensure_tables() -> None:
    with pg_conn() as conn:
        conn.execute(_CREATE_VARIANTS_SQL)
        conn.execute(_CREATE_RESULTS_SQL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_variants(config: ABTestConfig) -> None:
    """Persist variant definitions for an A/B test.

    Existing non-archived variants for the agent are left untouched;
    new variant_ids are inserted.
    """
    _ensure_tables()
    with pg_conn() as conn:
        for v in config.variants:
            conn.execute(
                f"""
                INSERT INTO {VARIANTS_TABLE}
                    (variant_id, agent_name, prompt_template, created_at, is_control)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (agent_name, variant_id) DO UPDATE SET
                    prompt_template = EXCLUDED.prompt_template,
                    is_control      = EXCLUDED.is_control
                """,
                (v.variant_id, v.agent_name, v.prompt_template, v.created_at, v.is_control),
            )
    logger.info(
        "Registered {n} variants for agent {agent}",
        n=len(config.variants),
        agent=config.agent_name,
    )


def record_ab_result(agent_name: str, variant_id: str, score: float) -> None:
    """Store an outcome score for a specific variant of an agent."""
    _ensure_tables()
    with pg_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {RESULTS_TABLE} (agent_name, variant_id, score, recorded_at)
            VALUES (%s, %s, %s, %s)
            """,
            (agent_name, variant_id, score, datetime.now(timezone.utc)),
        )
    logger.debug(
        "AB result: {agent}/{variant} score={score:.4f}",
        agent=agent_name,
        variant=variant_id,
        score=score,
    )


def evaluate_ab_test(agent_name: str, min_samples: int = 50, significance_level: float = 0.05) -> dict | None:
    """Evaluate A/B test results for an agent.

    Returns None if insufficient samples. Otherwise returns:
        {
            "winner": variant_id,
            "p_value": float,
            "improvement_pct": float,
            "sufficient_samples": bool,
            "control_mean": float,
            "treatment_mean": float,
        }
    """
    _ensure_tables()

    # Find the control variant
    with pg_conn() as conn:
        control_row = conn.execute(
            f"""
            SELECT variant_id FROM {VARIANTS_TABLE}
            WHERE agent_name = %s AND is_control = TRUE AND archived = FALSE
            """,
            (agent_name,),
        ).fetchone()

    if not control_row:
        logger.warning("No control variant found for agent {agent}", agent=agent_name)
        return None

    control_id = control_row[0]

    # Get all non-archived treatment variants
    with pg_conn() as conn:
        treatment_rows = conn.execute(
            f"""
            SELECT DISTINCT variant_id FROM {VARIANTS_TABLE}
            WHERE agent_name = %s AND is_control = FALSE AND archived = FALSE
            """,
            (agent_name,),
        ).fetchall()

    if not treatment_rows:
        return None

    # Fetch control scores
    with pg_conn() as conn:
        control_scores = [
            r[0]
            for r in conn.execute(
                f"SELECT score FROM {RESULTS_TABLE} WHERE agent_name = %s AND variant_id = %s",
                (agent_name, control_id),
            ).fetchall()
        ]

    if len(control_scores) < min_samples:
        return None

    # Evaluate each treatment against control, pick the best
    best_result: dict | None = None
    for (treatment_id,) in treatment_rows:
        with pg_conn() as conn:
            treatment_scores = [
                r[0]
                for r in conn.execute(
                    f"SELECT score FROM {RESULTS_TABLE} WHERE agent_name = %s AND variant_id = %s",
                    (agent_name, treatment_id),
                ).fetchall()
            ]

        if len(treatment_scores) < min_samples:
            continue

        t_stat, p_value = ttest_ind(treatment_scores, control_scores)

        control_mean = sum(control_scores) / len(control_scores)
        treatment_mean = sum(treatment_scores) / len(treatment_scores)
        improvement_pct = (
            ((treatment_mean - control_mean) / control_mean * 100.0)
            if control_mean != 0
            else 0.0
        )

        is_significant = p_value < significance_level and treatment_mean > control_mean

        result = {
            "winner": treatment_id if is_significant else control_id,
            "p_value": round(float(p_value), 6),
            "improvement_pct": round(improvement_pct, 2),
            "sufficient_samples": True,
            "control_mean": round(control_mean, 4),
            "treatment_mean": round(treatment_mean, 4),
        }

        if best_result is None or (is_significant and improvement_pct > best_result["improvement_pct"]):
            best_result = result

    return best_result


def promote_winning_variant(agent_name: str, variant_id: str) -> None:
    """Mark *variant_id* as the new control and archive the old control.

    Steps:
      1. Archive the current control (set archived=TRUE, is_control=FALSE).
      2. Set the winner as is_control=TRUE.
    """
    _ensure_tables()
    with pg_conn() as conn:
        # Archive old control
        conn.execute(
            f"""
            UPDATE {VARIANTS_TABLE}
            SET archived = TRUE, is_control = FALSE
            WHERE agent_name = %s AND is_control = TRUE AND archived = FALSE
            """,
            (agent_name,),
        )
        # Promote winner
        conn.execute(
            f"""
            UPDATE {VARIANTS_TABLE}
            SET is_control = TRUE
            WHERE agent_name = %s AND variant_id = %s
            """,
            (agent_name, variant_id),
        )
    logger.info(
        "Promoted variant {variant} as new control for {agent}",
        variant=variant_id,
        agent=agent_name,
    )
