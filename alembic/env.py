"""Alembic migration environment for QuantStack.

Reads TRADER_PG_URL from the environment and acquires PostgreSQL advisory
lock 5145534154 (same key as db.py) before running migrations.
"""

import logging
import os
from contextlib import contextmanager

from alembic import context
from sqlalchemy import create_engine, pool, text

logger = logging.getLogger("alembic.env")

config = context.config

_MIGRATION_ADVISORY_LOCK = 5145534154


def _get_url() -> str:
    url = os.environ.get("TRADER_PG_URL", "")
    if not url:
        raise RuntimeError("TRADER_PG_URL is not set")
    return url


@contextmanager
def _advisory_lock(connection):
    """Acquire a blocking advisory lock, release on exit."""
    connection.execute(text(f"SELECT pg_advisory_lock({_MIGRATION_ADVISORY_LOCK})"))
    try:
        yield
    finally:
        connection.execute(text(f"SELECT pg_advisory_unlock({_MIGRATION_ADVISORY_LOCK})"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live connection)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with a live database connection."""
    connectable = create_engine(
        _get_url(),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        with _advisory_lock(connection):
            context.configure(connection=connection, target_metadata=None)
            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
