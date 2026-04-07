"""Langfuse trace retention cleanup.

Deletes traces older than a configurable retention period from the
Langfuse PostgreSQL database. Called daily by the supervisor crew.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _get_langfuse_conn():
    """Get a psycopg3 connection to the Langfuse database."""
    import psycopg
    url = os.environ.get(
        "LANGFUSE_DATABASE_URL",
        "postgresql://langfuse:langfuse@langfuse-db:5432/langfuse",
    )
    return psycopg.connect(url)


def cleanup_langfuse_traces(retention_days: int = 30) -> int:
    """Delete Langfuse traces older than retention_days.

    Connects to the Langfuse Postgres instance and removes old trace data
    to prevent unbounded storage growth. Returns the number of traces deleted.
    """
    conn = _get_langfuse_conn()
    try:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM traces WHERE created_at < NOW() - INTERVAL '%s days'",
            [retention_days],
        )
        deleted = cur.rowcount
        logger.info(
            "Langfuse retention cleanup: deleted %d traces older than %d days",
            deleted, retention_days,
        )
        return deleted
    finally:
        conn.close()
