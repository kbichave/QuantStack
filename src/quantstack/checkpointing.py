"""Durable checkpoint management for LangGraph StateGraphs.

Provides an AsyncPostgresSaver-backed checkpointer factory that enables crash
recovery. Each graph runner gets a shared checkpointer backed by a dedicated
psycopg3 async connection pool (separate from the application pool in db.py).

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


def _get_pg_url() -> str:
    return os.getenv("TRADER_PG_URL", "postgresql://localhost/quantstack")


def _run_checkpoint_setup() -> None:
    """Create checkpoint tables using a sync autocommit connection.

    Uses the sync PostgresSaver.MIGRATIONS list directly since
    CREATE INDEX CONCURRENTLY requires autocommit mode.
    """
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    pg_url = _get_pg_url()
    migrations = PostgresSaver.MIGRATIONS

    with psycopg.connect(pg_url, autocommit=True, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(migrations[0])
            results = cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = results.fetchone()
            version = -1 if row is None else row["v"]
            for v, migration in zip(
                range(version + 1, len(migrations)),
                migrations[version + 1:],
                strict=False,
            ):
                cur.execute(migration)
                cur.execute(
                    "INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,)
                )
    logger.info("PostgresSaver checkpoint tables ready")


async def create_checkpointer():
    """Create an AsyncPostgresSaver backed by a dedicated async connection pool.

    Pool is sized for checkpoint operations: min_size=2, max_size=6.
    Tables are created synchronously before the async pool is opened.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    pg_url = _get_pg_url()

    # Ensure tables exist (sync, autocommit — safe for CREATE INDEX CONCURRENTLY)
    try:
        _run_checkpoint_setup()
    except Exception as exc:
        logger.warning("Checkpoint table setup failed (may already exist): %s", exc)

    pool = AsyncConnectionPool(
        conninfo=pg_url,
        min_size=2,
        max_size=6,
        max_lifetime=3600,
        max_idle=600,
        open=False,
    )
    await pool.open()

    return AsyncPostgresSaver(pool)


def setup_checkpoint_tables() -> None:
    """Create the PostgresSaver tables if they don't exist.

    Run once as a deployment/migration step, not on every startup.
    Safe to call multiple times (idempotent CREATE IF NOT EXISTS).
    """
    _run_checkpoint_setup()


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
