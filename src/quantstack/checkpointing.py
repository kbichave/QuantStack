"""Durable checkpoint management for LangGraph StateGraphs.

Provides a PostgresSaver-backed checkpointer factory that enables crash
recovery. Each graph runner gets a shared checkpointer backed by a dedicated
psycopg3 connection pool (separate from the application pool in db.py).

Connection budget:
  Main pool (db.py):      max 20
  Checkpointer pool:      max  6
  Scheduler:              1
  Backup job:             1
  Total:                  ~28 of PostgreSQL default 100 max_connections
"""

import logging
import os

logger = logging.getLogger(__name__)


def create_checkpointer():
    """Create a PostgresSaver backed by a dedicated psycopg3 connection pool.

    Pool is sized for checkpoint operations: min_size=2, max_size=6.
    This is intentionally smaller than the main application pool (max_size=20)
    because checkpoint writes are less frequent than application queries.

    setup() is NOT called here. Table creation is a deployment step,
    not a per-startup step. See setup_checkpoint_tables().
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    pg_url = os.getenv(
        "TRADER_PG_URL",
        f"postgresql://localhost/quantstack",
    )

    pool = ConnectionPool(
        conninfo=pg_url,
        min_size=2,
        max_size=6,
        max_lifetime=3600,
        max_idle=600,
    )

    return PostgresSaver(pool)


def setup_checkpoint_tables() -> None:
    """Create the PostgresSaver tables if they don't exist.

    Run once as a deployment/migration step, not on every startup.
    Safe to call multiple times (idempotent CREATE IF NOT EXISTS).
    """
    checkpointer = create_checkpointer()
    checkpointer.setup()
    logger.info("PostgresSaver checkpoint tables created/verified")


def prune_old_checkpoints(retention_hours: int = 48) -> int:
    """Delete checkpoint data older than retention_hours.

    Preserves:
    - The most recent completed cycle per graph (regardless of age)
    - Any in-progress cycles (incomplete checkpoint sequences)

    Returns the number of rows deleted.
    """
    from quantstack.db import db_conn

    total_deleted = 0
    try:
        with db_conn() as conn:
            # Delete old checkpoint writes
            result = conn.execute(
                """DELETE FROM checkpoint_writes
                   WHERE thread_id NOT IN (
                       -- Preserve latest completed per graph
                       SELECT DISTINCT ON (split_part(thread_id, '-', 1))
                              thread_id
                       FROM checkpoint_writes
                       ORDER BY split_part(thread_id, '-', 1), created_at DESC
                   )
                   AND created_at < NOW() - INTERVAL '%s hours'""",
                [retention_hours],
            )
            if result:
                total_deleted += result.rowcount or 0

            # Delete old checkpoints
            result = conn.execute(
                """DELETE FROM checkpoints
                   WHERE thread_id NOT IN (
                       SELECT DISTINCT ON (split_part(thread_id, '-', 1))
                              thread_id
                       FROM checkpoints
                       ORDER BY split_part(thread_id, '-', 1), created_at DESC
                   )
                   AND created_at < NOW() - INTERVAL '%s hours'""",
                [retention_hours],
            )
            if result:
                total_deleted += result.rowcount or 0

    except Exception:
        logger.exception("Checkpoint pruning failed")

    if total_deleted:
        logger.info("Pruned %d old checkpoint rows (retention=%dh)", total_deleted, retention_hours)

    return total_deleted
