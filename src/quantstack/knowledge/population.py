# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backfill pipeline for the Alpha Knowledge Graph.

Reads existing tables (strategies, autoresearch_experiments, ml_experiments,
ic_observations) and creates corresponding nodes and edges.  Each function
is idempotent — safe to re-run after schema changes or data additions.

All functions handle missing tables gracefully (return 0) so the pipeline
works at any deployment stage.
"""

from __future__ import annotations

from loguru import logger

from quantstack.db import db_conn
from quantstack.knowledge.graph import KnowledgeGraph


def _table_exists(table_name: str) -> bool:
    """Check whether *table_name* exists in the public schema."""
    with db_conn() as conn:
        conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            [table_name],
        )
        row = conn.fetchone()
        return bool(row and row.get("exists", False))


def backfill_from_strategies() -> int:
    """Create strategy + factor nodes and uses edges from the strategies table.

    Returns the number of strategy nodes created/updated.
    """
    if not _table_exists("strategies"):
        logger.info("[KG-Backfill] strategies table not found — skipping")
        return 0

    kg = KnowledgeGraph()
    count = 0

    with db_conn() as conn:
        conn.execute(
            "SELECT strategy_id, name, factors, regime, status FROM strategies"
        )
        rows = conn.fetchall()

    for row in rows:
        strategy_name = row.get("name") or row.get("strategy_id", "unknown")
        props = {
            "strategy_id": row.get("strategy_id"),
            "regime": row.get("regime"),
            "status": row.get("status"),
        }
        strategy_node_id = kg.create_node("strategy", strategy_name, props)

        # Parse factors — stored as JSONB array or comma-separated string
        factors_raw = row.get("factors")
        if isinstance(factors_raw, list):
            factors = factors_raw
        elif isinstance(factors_raw, str):
            factors = [f.strip() for f in factors_raw.split(",") if f.strip()]
        else:
            factors = []

        for factor_name in factors:
            factor_id = kg.create_node("factor", factor_name)
            try:
                kg.create_edge("uses", strategy_node_id, factor_id)
            except ValueError:
                logger.warning("[KG-Backfill] Edge creation failed for %s → %s", strategy_name, factor_name)

        count += 1

    logger.info("[KG-Backfill] Backfilled %d strategies", count)
    return count


def backfill_from_ml_experiments() -> int:
    """Create result nodes and tested_by edges from ml_experiments table.

    Returns the number of experiment nodes created.
    """
    if not _table_exists("ml_experiments"):
        logger.info("[KG-Backfill] ml_experiments table not found — skipping")
        return 0

    kg = KnowledgeGraph()
    count = 0

    with db_conn() as conn:
        conn.execute(
            "SELECT experiment_id, model_type, metrics, created_at FROM ml_experiments"
        )
        rows = conn.fetchall()

    for row in rows:
        exp_name = f"ml_{row.get('model_type', 'unknown')}_{row.get('experiment_id', '')[:8]}"
        props = {
            "model_type": row.get("model_type"),
            "metrics": row.get("metrics"),
        }
        kg.create_node("result", exp_name, props)
        count += 1

    logger.info("[KG-Backfill] Backfilled %d ML experiments", count)
    return count


def backfill_from_autoresearch() -> int:
    """Create hypothesis + result nodes from autoresearch_experiments.

    Returns the number of hypothesis nodes created.
    """
    if not _table_exists("autoresearch_experiments"):
        logger.info("[KG-Backfill] autoresearch_experiments table not found — skipping")
        return 0

    kg = KnowledgeGraph()
    count = 0

    with db_conn() as conn:
        conn.execute(
            """
            SELECT experiment_id, hypothesis, hypothesis_source,
                   oos_ic, sharpe, status, created_at
            FROM autoresearch_experiments
            """
        )
        rows = conn.fetchall()

    for row in rows:
        hyp_data = row.get("hypothesis") or {}
        if isinstance(hyp_data, str):
            import json
            hyp_data = json.loads(hyp_data)

        hyp_text = hyp_data.get("text", hyp_data.get("name", f"autoresearch_{row['experiment_id'][:8]}"))
        props = {
            "source": row.get("hypothesis_source"),
            "outcome": row.get("status"),
            "sharpe": row.get("sharpe"),
            "ic": row.get("oos_ic"),
            "test_date": str(row.get("created_at", "")),
        }

        from quantstack.knowledge.embeddings import generate_embedding
        emb = generate_embedding(hyp_text)
        kg.create_node("hypothesis", hyp_text, props, embedding=emb)

        # Result node
        result_name = f"autoresearch_result_{row['experiment_id'][:8]}"
        result_props = {
            "sharpe": row.get("sharpe"),
            "oos_ic": row.get("oos_ic"),
            "status": row.get("status"),
        }
        kg.create_node("result", result_name, result_props)
        count += 1

    logger.info("[KG-Backfill] Backfilled %d autoresearch experiments", count)
    return count


def backfill_from_ic_observations() -> int:
    """Create evidence nodes from ic_observations table.

    Returns the number of evidence nodes created.
    """
    if not _table_exists("ic_observations"):
        logger.info("[KG-Backfill] ic_observations table not found — skipping")
        return 0

    kg = KnowledgeGraph()
    count = 0

    with db_conn() as conn:
        conn.execute(
            "SELECT id, factor_name, ic_value, period, created_at FROM ic_observations"
        )
        rows = conn.fetchall()

    for row in rows:
        evidence_name = f"ic_{row.get('factor_name', 'unknown')}_{row.get('period', '')}"
        props = {
            "factor_name": row.get("factor_name"),
            "ic_value": row.get("ic_value"),
            "period": row.get("period"),
        }
        kg.create_node("evidence", evidence_name, props)
        count += 1

    logger.info("[KG-Backfill] Backfilled %d IC observations", count)
    return count


def run_full_backfill() -> dict[str, int]:
    """Run all backfill pipelines and return counts per source."""
    results = {
        "strategies": backfill_from_strategies(),
        "ml_experiments": backfill_from_ml_experiments(),
        "autoresearch": backfill_from_autoresearch(),
        "ic_observations": backfill_from_ic_observations(),
    }
    total = sum(results.values())
    logger.info("[KG-Backfill] Full backfill complete: %d total nodes created/updated", total)
    return results
