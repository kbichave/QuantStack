"""Health metrics collector for the supervisor graph.

Queries operational health metrics from PostgreSQL and exposes them
via Prometheus gauges. Called by the health_check node after its
LLM-based introspection.
"""

from __future__ import annotations

import logging

from quantstack.db import db_conn

logger = logging.getLogger(__name__)


async def collect_health_metrics() -> dict[str, float | int]:
    """Query operational health metrics from PostgreSQL.

    Returns dict with keys:
        - trading_cycle_success_rate (float, 0.0-1.0)
        - research_cycle_success_rate (float, 0.0-1.0)
        - trading_cycle_error_count (int)
        - research_cycle_error_count (int)
        - strategy_generation_7d (int)
        - research_queue_depth (int)
    """
    result: dict[str, float | int] = {}

    # Cycle success rate + error count per graph
    for graph_name in ("trading", "research"):
        prefix = f"{graph_name}_graph" if graph_name != "trading" else "trading-graph"
        graph_key = f"{graph_name}-graph"

        # Success rate (last 10 cycles)
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'success') AS successes,
                    COUNT(*) AS total
                FROM (
                    SELECT status FROM graph_checkpoints
                    WHERE graph_name = %s
                    ORDER BY started_at DESC
                    LIMIT 10
                ) sub
                """,
                (graph_key,),
            ).fetchone()

        successes, total = row if row else (0, 0)
        rate = successes / total if total > 0 else 0.0
        result[f"{graph_name}_cycle_success_rate"] = rate

        # Error count (most recent cycle)
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS error_count
                FROM graph_checkpoints
                WHERE graph_name = %s
                  AND status = 'error'
                  AND cycle_id = (
                      SELECT cycle_id FROM graph_checkpoints
                      WHERE graph_name = %s
                      ORDER BY started_at DESC LIMIT 1
                  )
                """,
                (graph_key, graph_key),
            ).fetchone()

        error_count = row[0] if row else 0
        result[f"{graph_name}_cycle_error_count"] = error_count

    # Strategy generation velocity (last 7 days)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE created_at >= NOW() - INTERVAL '7 days'"
        ).fetchone()
    result["strategy_generation_7d"] = row[0] if row else 0

    # Research queue depth
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM research_tasks WHERE status = 'pending'"
        ).fetchone()
    result["research_queue_depth"] = row[0] if row else 0

    return result
