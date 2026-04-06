# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Single consolidated database for all QuantStack state — PostgreSQL only.

## Architecture

All tables (operational and analytics) live in **PostgreSQL** (TRADER_PG_URL).
PostgreSQL provides true MVCC: unlimited concurrent readers, writers never block
readers, no file locks.

Operational tables: positions, signals, strategies, fills, audit, etc.
Analytics tables: ML experiments, research programs, reflexion episodes, etc.

## Connection Model

``pg_conn()`` gets a connection from a thread-safe pool (psycopg2
ThreadedConnectionPool, default size 2–10).  Multiple processes and threads
can hold connections simultaneously — no file locks, no contention.

``db_conn()`` is an alias for ``pg_conn()``.

Usage::

    with pg_conn() as conn:
        conn.execute("INSERT INTO strategies ...", [params])

## Migrations

``run_migrations(conn)`` creates all tables.  Called once at service startup.
Idempotent — uses ``CREATE TABLE IF NOT EXISTS`` throughout.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Iterator

import pandas as pd
import psycopg2
import psycopg2.extensions
import psycopg2.extras
import psycopg2.pool
from loguru import logger

# ---------------------------------------------------------------------------
# psycopg2 JSON behaviour — keep JSON/JSONB columns as raw strings
# ---------------------------------------------------------------------------
# Legacy: DuckDB-era code calls json.loads() explicitly on JSON fields.
# psycopg2 by default parses JSON/JSONB into Python objects, which causes
# json.loads(list) → TypeError in those code paths.
#
# Migration path: new code should use coerce_json() from
# quantstack.tools._shared instead of calling json.loads() directly.
# Once all explicit json.loads() calls on DB columns are removed, delete
# these two lines and let psycopg2 parse JSON normally.
psycopg2.extras.register_default_json(loads=lambda x: x)
psycopg2.extras.register_default_jsonb(loads=lambda x: x)

# ---------------------------------------------------------------------------
# Path / URL resolution
# ---------------------------------------------------------------------------


def _resolve_pg_url() -> str:
    return os.getenv("TRADER_PG_URL", "postgresql://localhost/quantstack")


# ---------------------------------------------------------------------------
# PostgreSQL connection pool
# ---------------------------------------------------------------------------

_pg_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pg_pool_lock = Lock()


def _get_pg_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is None or _pg_pool.closed:
            url = _resolve_pg_url()
            # Pool size: 20 default handles concurrent data acquisition,
            # trading loop, research loop, and scheduler.  Override with
            # PG_POOL_MAX env var if running multiple processes.
            maxconn = int(os.getenv("PG_POOL_MAX", "20"))
            _pg_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=maxconn, dsn=url)
            logger.debug(f"[DB] PostgreSQL pool created → {url}")
    return _pg_pool


def reset_pg_pool() -> None:
    """Close and discard the pool.  Used by tests and graceful shutdown."""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is not None and not _pg_pool.closed:
            _pg_pool.closeall()
            _pg_pool = None
            logger.debug("[DB] PostgreSQL pool closed")


# ---------------------------------------------------------------------------
# PgConnection — psycopg2 wrapper
# ---------------------------------------------------------------------------


class PgConnection:
    """Thread-safe psycopg2 connection wrapper.

    Provides ``.execute()``, ``.fetchone()``, ``.fetchall()``, ``.fetchdf()``
    on top of a pooled psycopg2 connection, so all services share a consistent
    call surface.

    Lifecycle:
        The pool connection is lazily acquired on first ``execute()`` and
        returned to the pool on ``release()``.  Unlike the old
        ``ManagedConnection`` design, there is no exclusive file lock to release
        between tool calls — the pool manages allocation transparently.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, pool: psycopg2.pool.ThreadedConnectionPool):
        """Create a pool-backed PgConnection.

        Args:
            pool: PostgreSQL connection pool.
        """
        self._pool = pool
        self._raw: psycopg2.extensions.connection | None = None
        self._cur: psycopg2.extensions.cursor | None = None
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_raw(self) -> psycopg2.extensions.connection:
        """Lazily acquire a connection from the pool."""
        if self._raw is None or self._raw.closed:
            self._raw = self._pool.getconn()
            # Set session-level timeout before entering any transaction.
            # If this connection goes idle inside a transaction for > 30s
            # (e.g. the process is killed mid-write), PostgreSQL auto-terminates
            # it and releases locks instead of blocking migrations forever.
            self._raw.autocommit = True
            _cur = self._raw.cursor()
            _cur.execute("SET idle_in_transaction_session_timeout = '30s'")
            _cur.close()
            self._raw.autocommit = False
            logger.debug("[DB] PostgreSQL connection acquired from pool")
        return self._raw

    @staticmethod
    def _translate(query: str) -> str:
        """Translate ``?`` placeholders → psycopg2 ``%s``."""
        return query.replace("?", "%s")

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def execute(self, query: str, params: object = None) -> "PgConnection":
        with self._lock:
            q_upper = query.strip().upper()

            # Map explicit transaction keywords to psycopg2 Python API
            if q_upper == "BEGIN":
                # psycopg2 is already in a transaction (autocommit=False).
                # No-op — a transaction is implicitly open on the first statement.
                return self
            if q_upper == "COMMIT":
                raw = self._ensure_raw()
                raw.commit()
                return self
            if q_upper == "ROLLBACK":
                raw = self._ensure_raw()
                raw.rollback()
                return self

            raw = self._ensure_raw()
            if self._cur is None or self._cur.closed:
                self._cur = raw.cursor()
            translated = self._translate(query)
            try:
                if params is not None:
                    self._cur.execute(translated, params)
                else:
                    self._cur.execute(translated)
            except psycopg2.OperationalError as op_err:
                # The server may have killed an idle connection (TCP timeout,
                # idle_in_transaction_session_timeout, admin terminate, etc.).
                # Discard the broken connection, acquire a fresh one, and retry
                # once before surfacing the error.
                err_msg = str(op_err).lower()
                is_broken = (
                    "server closed" in err_msg
                    or "connection" in err_msg
                    or raw.closed != 0
                )
                if is_broken:
                    try:
                        self._pool.putconn(self._raw, close=True)
                    except Exception as exc:
                        logger.debug("[DB] putconn(close=True) for broken conn failed: %s", exc)
                    self._raw = None
                    self._cur = None
                    logger.warning(
                        "[DB] broken connection detected — reacquiring and retrying"
                    )
                    try:
                        raw = self._ensure_raw()
                        self._cur = raw.cursor()
                        if params is not None:
                            self._cur.execute(translated, params)
                        else:
                            self._cur.execute(translated)
                    except Exception:
                        try:
                            self._raw.rollback()
                        except Exception as rb_err:
                            logger.debug(f"[DB] rollback after retry also failed: {rb_err}")
                        raise
                else:
                    try:
                        raw.rollback()
                    except Exception as rb_err:
                        logger.debug(f"[DB] rollback after failed execute also failed: {rb_err}")
                    raise
            except Exception:
                # Roll back immediately so the connection is clean for the next
                # caller.  ctx.db is a long-lived shared connection — without
                # this, any failed statement leaves psycopg2 in "aborted
                # transaction" state and every subsequent query on the same
                # connection fails with "current transaction is aborted".
                try:
                    raw.rollback()
                except Exception as rb_err:
                    logger.debug(f"[DB] rollback after failed execute also failed: {rb_err}")
                raise
        return self

    def executemany(self, query: str, params: object = None) -> "PgConnection":
        with self._lock:
            raw = self._ensure_raw()
            if self._cur is None or self._cur.closed:
                self._cur = raw.cursor()
            try:
                self._cur.executemany(self._translate(query), params)
            except Exception:
                try:
                    raw.rollback()
                except Exception as rb_err:
                    logger.debug(f"[DB] rollback after failed executemany also failed: {rb_err}")
                raise
        return self

    def fetchone(self):
        with self._lock:
            return self._cur.fetchone() if self._cur else None

    def fetchall(self):
        with self._lock:
            return self._cur.fetchall() if self._cur else []

    def fetchdf(self):
        rows = self.fetchall()
        cols = (
            [d[0] for d in self._cur.description]
            if self._cur and self._cur.description
            else []
        )
        return pd.DataFrame(rows, columns=cols)

    def fetchnumpy(self):
        df = self.fetchdf()
        return {col: df[col].values for col in df.columns}

    def fetch_arrow_table(self):
        import pyarrow as pa  # optional dep — only needed for Arrow export
        return pa.Table.from_pandas(self.fetchdf())

    @property
    def description(self):
        return self._cur.description if self._cur else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Explicitly commit the current transaction.

        Provided for compatibility with DuckDB-era code that called
        ``conn.commit()`` directly.  Prefer letting ``release()`` commit
        automatically, but explicit commits are safe here too.
        """
        with self._lock:
            if self._raw is not None and not self._raw.closed:
                self._raw.commit()

    def rollback(self) -> None:
        """Explicitly roll back the current transaction.

        Provided for compatibility with DuckDB-era code that called
        ``conn.rollback()`` directly.
        """
        with self._lock:
            if self._raw is not None and not self._raw.closed:
                self._raw.rollback()

    def release(self) -> None:
        """Return the connection to the pool.

        Safe to call multiple times.  Commits any open transaction before
        returning — services that don't explicitly COMMIT rely on this.
        """
        with self._lock:
            if self._raw is not None and not self._raw.closed:
                try:
                    self._raw.commit()
                except Exception as commit_err:
                    logger.warning(f"[DB] commit failed on release, attempting rollback: {commit_err}")
                    try:
                        self._raw.rollback()
                    except Exception as rb_err:
                        logger.warning(f"[DB] rollback also failed on release: {rb_err}")
                try:
                    self._pool.putconn(self._raw)
                except Exception as putconn_err:
                    logger.warning(f"[DB] putconn failed, connection may be leaked: {putconn_err}")
                self._raw = None
                self._cur = None
                logger.debug("[DB] PostgreSQL connection returned to pool")

    def close(self) -> None:
        """Alias for ``release()``."""
        self.release()

    @property
    def is_open(self) -> bool:
        return self._raw is not None and not self._raw.closed

    def __repr__(self) -> str:
        state = "open" if self.is_open else "released"
        return f"PgConnection({state})"


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def pg_conn() -> Iterator[PgConnection]:
    """Get a PostgreSQL connection from the pool as a context manager.

    Commits on clean exit, rolls back on exception, returns to pool always.
    Multiple processes can call this concurrently — no file lock, no waiting.

    Usage::

        with pg_conn() as conn:
            conn.execute("INSERT INTO strategies ...", [params])
        # committed and returned to pool here
    """
    pool = _get_pg_pool()
    conn = PgConnection(pool)
    try:
        yield conn
        # Commit any open transaction that wasn't explicitly committed
        if conn._raw is not None and not conn._raw.closed:
            conn._raw.commit()
    except Exception:
        if conn._raw is not None and not conn._raw.closed:
            try:
                conn._raw.rollback()
            except Exception as rb_exc:
                logger.debug("[DB] rollback during exception cleanup failed: %s", rb_exc)
        raise
    finally:
        conn.release()


# db_conn is the canonical alias used throughout the codebase.
db_conn = pg_conn


def open_db(path: str = "") -> PgConnection:
    """Return a PgConnection for use by services that hold a long-lived reference.

    The caller is responsible for calling ``.release()`` when done
    (``create_trading_context`` does this after migrations).
    """
    return PgConnection(_get_pg_pool())


def reset_connection() -> None:
    """Close the PostgreSQL pool.  Used in tests for clean teardown."""
    reset_pg_pool()


# ---------------------------------------------------------------------------
# Legacy compat shims (no-op — file locks no longer exist)
# ---------------------------------------------------------------------------


def open_db_readonly(path: str = "") -> PgConnection:
    """Compat shim — all connections are concurrent-read-safe in PostgreSQL."""
    return open_db(path)


def reset_connection_readonly() -> None:
    """Compat shim — no-op."""


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------


def _to_pg(sql: str) -> str:
    """Normalise DDL for PostgreSQL.

    Replaces bare DOUBLE (not followed by PRECISION) with DOUBLE PRECISION,
    and JSON column types with JSONB.
    """
    import re
    # Replace DOUBLE not already followed by ' PRECISION'
    sql = re.sub(r'\bDOUBLE(?! PRECISION)', 'DOUBLE PRECISION', sql)
    return (
        sql
        .replace(" JSON,", " JSONB,")
        .replace(" JSON\n", " JSONB\n")
        .replace(" JSON)", " JSONB)")
        .replace(" JSON NOT NULL", " JSONB NOT NULL")
        .replace(" JSON DEFAULT", " JSONB DEFAULT")
    )


def _alter_safe(conn: PgConnection, table: str, column: str, col_type: str) -> None:
    """Add a column if it doesn't already exist (``ADD COLUMN IF NOT EXISTS``)."""
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}")


# ---------------------------------------------------------------------------
# Migrations — PostgreSQL (operational tables)
# ---------------------------------------------------------------------------


_MIGRATION_ADVISORY_LOCK = 5145534154  # "QUANT" ASCII → unique lock key
_migrations_done: bool = False  # module-level flag — migrations run once per process


def run_migrations_pg(conn: PgConnection) -> None:
    """Create all tables in PostgreSQL.

    Idempotent — uses CREATE TABLE IF NOT EXISTS and ADD COLUMN IF NOT EXISTS.

    Uses a PostgreSQL advisory lock so that when 10 services start
    simultaneously, only one runs migrations; the others skip (tables already
    exist).  Each DDL runs in autocommit mode so that:
      - No long-held transaction blocks concurrent tool calls
      - A server killed mid-migration leaves no orphaned locks
      - lock_timeout = '8s' makes any blocked DDL fail fast rather than hang

    Also uses a module-level ``_migrations_done`` flag so that within a single
    process (e.g. the test suite) migrations only run once — subsequent calls
    return immediately, avoiding DDL lock contention with concurrent test
    transactions.
    """
    global _migrations_done
    if _migrations_done:
        logger.debug("[DB] Migrations already run this process — skipping")
        return
    raw = conn._ensure_raw()
    prev_autocommit = raw.autocommit
    raw.autocommit = True
    cur = raw.cursor()
    try:
        # Advisory lock: only one process runs migrations at a time.
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_MIGRATION_ADVISORY_LOCK,))
        result = cur.fetchone()
        locked = result[0] if result else False
        if not locked:
            logger.info("[DB] Migration lock held by another server — skipping (tables already exist)")
            return

        try:
            # Fail fast if any DDL is blocked by an existing lock (e.g. stale
            # idle-in-transaction session).  8s is long enough to be safe on
            # a loaded dev machine, short enough to not hang startup.
            cur.execute("SET lock_timeout = '8s'")

            # Each _migrate_* call runs in autocommit mode — every DDL
            # statement commits immediately, no transaction to orphan.
            _migrate_portfolio_pg(conn)
            _migrate_broker_pg(conn)
            _migrate_audit_pg(conn)
            _migrate_learning_pg(conn)
            _migrate_memory_pg(conn)
            _migrate_signals_pg(conn)
            _migrate_system_pg(conn)
            _migrate_strategies_pg(conn)
            _migrate_regime_matrix_pg(conn)
            _migrate_strategy_outcomes_pg(conn)
            _migrate_universe_pg(conn)
            _migrate_screener_pg(conn)
            _migrate_coordination_pg(conn)
            _migrate_research_wip_pg(conn)
            _migrate_conversations_pg(conn)
            _migrate_attribution_pg(conn)
            _migrate_equity_alerts_pg(conn)
            _migrate_market_data_pg(conn)
            _migrate_analytics_pg(conn)
            _migrate_research_queue_pg(conn)
            _migrate_loop_context_pg(conn)
            _migrate_bugs_pg(conn)
            _migrate_ml_pipeline_pg(conn)
            _migrate_capital_allocation_pg(conn)
            _migrate_risk_monitoring_pg(conn)
            _migrate_stat_arb_pg(conn)
            _migrate_tool_search_metrics_pg(conn)
            _migrate_trade_quality_pg(conn)
            _migrate_market_holidays_pg(conn)
            _migrate_signal_history_pg(conn)
            _migrate_signal_ic_pg(conn)
            _migrate_pnl_attribution_pg(conn)
            _migrate_regime_state_pg(conn)
            _migrate_institutional_gaps_pg(conn)
            _migrate_ewf_pg(conn)

            logger.info("[DB] PostgreSQL migrations complete")
            _migrations_done = True
        except Exception:
            logger.exception("[DB] Migration failed")
            raise
        finally:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_ADVISORY_LOCK,))
    finally:
        cur.close()
        raw.autocommit = prev_autocommit


def _migrate_portfolio_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol          TEXT PRIMARY KEY,
            quantity        INTEGER NOT NULL,
            avg_cost        DOUBLE PRECISION NOT NULL,
            side            TEXT DEFAULT 'long',
            opened_at       TIMESTAMPTZ DEFAULT NOW(),
            last_updated    TIMESTAMPTZ DEFAULT NOW(),
            unrealized_pnl  DOUBLE PRECISION DEFAULT 0.0,
            current_price   DOUBLE PRECISION DEFAULT 0.0,
            strategy_id     TEXT DEFAULT '',
            regime_at_entry TEXT DEFAULT 'unknown',
            instrument_type TEXT DEFAULT 'equity',
            time_horizon    TEXT DEFAULT 'swing',
            stop_price      DOUBLE PRECISION,
            target_price    DOUBLE PRECISION,
            trailing_stop   DOUBLE PRECISION,
            entry_atr       DOUBLE PRECISION DEFAULT 0.0,
            option_expiry   TEXT,
            option_strike   DOUBLE PRECISION,
            option_type     TEXT,
            strike          DOUBLE PRECISION,
            expiry          TEXT,
            premium_at_entry DOUBLE PRECISION
        )
    """))
    # v3 — execution monitor bookkeeping columns
    conn.execute(_to_pg("""
        ALTER TABLE positions ADD COLUMN IF NOT EXISTS monitor_last_check TIMESTAMPTZ
    """))
    conn.execute(_to_pg("""
        ALTER TABLE positions ADD COLUMN IF NOT EXISTS monitor_hwm DOUBLE PRECISION
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS cash_balance (
            id          INTEGER PRIMARY KEY DEFAULT 1,
            cash        DOUBLE PRECISION NOT NULL,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(_to_pg("""
        CREATE SEQUENCE IF NOT EXISTS closed_trades_seq START 1
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS closed_trades (
            id           BIGINT PRIMARY KEY DEFAULT nextval('closed_trades_seq'),
            symbol       TEXT NOT NULL,
            side         TEXT NOT NULL,
            quantity     INTEGER NOT NULL,
            entry_price  DOUBLE PRECISION NOT NULL,
            exit_price   DOUBLE PRECISION NOT NULL,
            realized_pnl DOUBLE PRECISION NOT NULL,
            opened_at    TIMESTAMPTZ,
            closed_at    TIMESTAMPTZ DEFAULT NOW(),
            holding_days INTEGER DEFAULT 0,
            session_id   TEXT DEFAULT '',
            strategy_id       TEXT DEFAULT '',
            regime_at_entry   TEXT DEFAULT 'unknown',
            regime_at_exit    TEXT DEFAULT 'unknown',
            exit_reason       TEXT DEFAULT '',
            instrument_type   TEXT DEFAULT 'equity'
        )
    """))


def _migrate_broker_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS fills (
            order_id            TEXT PRIMARY KEY,
            symbol              TEXT NOT NULL,
            side                TEXT NOT NULL,
            requested_quantity  INTEGER,
            filled_quantity     INTEGER,
            fill_price          DOUBLE PRECISION,
            slippage_bps        DOUBLE PRECISION,
            commission          DOUBLE PRECISION DEFAULT 0.0,
            partial             BOOLEAN DEFAULT FALSE,
            rejected            BOOLEAN DEFAULT FALSE,
            reject_reason       TEXT,
            filled_at           TIMESTAMPTZ DEFAULT NOW(),
            session_id          TEXT DEFAULT ''
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS fills_symbol_idx ON fills (symbol, filled_at)
    """)


def _migrate_audit_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS decision_events (
            event_id                TEXT PRIMARY KEY,
            session_id              TEXT NOT NULL,
            event_type              TEXT NOT NULL,
            agent_name              TEXT NOT NULL,
            agent_role              TEXT DEFAULT '',
            symbol                  TEXT DEFAULT '',
            action                  TEXT DEFAULT '',
            confidence              DOUBLE PRECISION DEFAULT 0.0,
            input_context_hash      TEXT DEFAULT '',
            market_data_snapshot    JSONB,
            portfolio_snapshot      JSONB,
            tool_calls              JSONB,
            output_summary          TEXT DEFAULT '',
            output_structured       JSONB,
            risk_approved           BOOLEAN,
            risk_violations         JSONB,
            created_at              TIMESTAMPTZ DEFAULT NOW(),
            decision_latency_ms     INTEGER,
            parent_event_ids        JSONB
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS decisions_session_idx ON decision_events (session_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS decisions_symbol_idx ON decision_events (symbol, created_at)
    """)


def _migrate_learning_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS agent_skills (
            agent_id            TEXT PRIMARY KEY,
            prediction_count    INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            signal_count        INTEGER DEFAULT 0,
            winning_signals     INTEGER DEFAULT 0,
            total_signal_pnl    DOUBLE PRECISION DEFAULT 0.0,
            last_updated        TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute("CREATE SEQUENCE IF NOT EXISTS calibration_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS calibration_records (
            id                  BIGINT PRIMARY KEY DEFAULT nextval('calibration_seq'),
            agent_name          TEXT NOT NULL,
            stated_confidence   DOUBLE PRECISION NOT NULL,
            was_correct         BOOLEAN NOT NULL,
            symbol              TEXT DEFAULT '',
            action              TEXT DEFAULT '',
            pnl                 DOUBLE PRECISION DEFAULT 0.0,
            session_id          TEXT DEFAULT '',
            recorded_at         TIMESTAMPTZ DEFAULT NOW()
        )
    """))


def _migrate_memory_pg(conn: PgConnection) -> None:
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_memory_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id          BIGINT PRIMARY KEY DEFAULT nextval('agent_memory_seq'),
            session_id  TEXT NOT NULL,
            sim_date    DATE,
            agent       TEXT NOT NULL,
            symbol      TEXT DEFAULT '',
            category    TEXT DEFAULT 'general',
            content_json TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS memory_symbol_idx
        ON agent_memory (symbol, created_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS memory_session_idx
        ON agent_memory (session_id, created_at DESC)
    """)


def _migrate_signals_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS signal_state (
            symbol              TEXT PRIMARY KEY,
            action              TEXT NOT NULL,
            confidence          DOUBLE PRECISION NOT NULL,
            position_size_pct   DOUBLE PRECISION NOT NULL,
            stop_loss           DOUBLE PRECISION,
            take_profit         DOUBLE PRECISION,
            generated_at        TIMESTAMPTZ NOT NULL,
            expires_at          TIMESTAMPTZ NOT NULL,
            session_id          TEXT NOT NULL
        )
    """))


def _migrate_system_pg(conn: PgConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_state (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    # Seed the daily risk-free rate used by attribution_engine.
    # ON CONFLICT DO NOTHING — never overwrites a value set by eod_data_sync.
    # Initial value: ~5% annual ÷ 252 trading days.
    conn.execute("""
        INSERT INTO system_state (key, value, updated_at)
        VALUES ('risk_free_rate_daily', '0.000198', NOW())
        ON CONFLICT (key) DO NOTHING
    """)


def _migrate_strategies_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id         TEXT PRIMARY KEY,
            name                TEXT NOT NULL UNIQUE,
            description         TEXT DEFAULT '',
            asset_class         TEXT DEFAULT 'equities',
            regime_affinity     JSONB,
            parameters          JSONB NOT NULL,
            entry_rules         JSONB NOT NULL,
            exit_rules          JSONB NOT NULL,
            risk_params         JSONB,
            backtest_summary    JSONB,
            walkforward_summary JSONB,
            status              TEXT DEFAULT 'draft',
            source              TEXT DEFAULT 'manual',
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            updated_at          TIMESTAMPTZ DEFAULT NOW(),
            created_by          TEXT DEFAULT 'claude_code',
            instrument_type     TEXT DEFAULT 'equity',
            time_horizon        TEXT DEFAULT 'swing',
            holding_period_days INTEGER DEFAULT 5
        )
    """))
    # Backfill NULL created_at / updated_at on existing rows.
    # Note: ALTER TABLE ... SET DEFAULT was removed — it requires AccessExclusiveLock
    # which blocks concurrent SELECTs.  CREATE TABLE IF NOT EXISTS already sets all
    # defaults correctly for new installs; existing rows are backfilled below.
    conn.execute(
        "UPDATE strategies SET created_at = NOW() WHERE created_at IS NULL"
    )
    conn.execute(
        "UPDATE strategies SET updated_at = NOW() WHERE updated_at IS NULL"
    )
    conn.execute("""
        CREATE INDEX IF NOT EXISTS strategies_status_idx ON strategies (status)
    """)
    # Add symbol column for direct filtering without JSON extraction.
    conn.execute("""
        ALTER TABLE strategies ADD COLUMN IF NOT EXISTS symbol TEXT
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS strategies_symbol_idx ON strategies (symbol)
    """)
    # Backfill symbol from parameters JSON where available,
    # otherwise extract the uppercase ticker prefix from the name.
    conn.execute("""
        UPDATE strategies
        SET symbol = COALESCE(
            (parameters::jsonb)->>'symbol',
            CASE
                WHEN name ~ '^[A-Z]{3,5}_' THEN split_part(name, '_', 1)
                WHEN name ~ '^[a-z]{3,5}_' THEN UPPER(split_part(name, '_', 1))
                ELSE NULL
            END
        )
        WHERE symbol IS NULL
    """)


def _migrate_regime_matrix_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS regime_strategy_matrix (
            regime          TEXT NOT NULL,
            strategy_id     TEXT NOT NULL,
            allocation_pct  DOUBLE PRECISION NOT NULL,
            confidence      DOUBLE PRECISION DEFAULT 0.5,
            last_updated    TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (regime, strategy_id)
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS rsm_regime_idx ON regime_strategy_matrix (regime)
    """)


def _migrate_strategy_outcomes_pg(conn: PgConnection) -> None:
    conn.execute("CREATE SEQUENCE IF NOT EXISTS strategy_outcomes_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS strategy_outcomes (
            id               BIGINT PRIMARY KEY DEFAULT nextval('strategy_outcomes_seq'),
            strategy_id      TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            regime_at_entry  TEXT NOT NULL DEFAULT 'unknown',
            action           TEXT NOT NULL,
            entry_price      DOUBLE PRECISION NOT NULL,
            exit_price       DOUBLE PRECISION,
            realized_pnl_pct DOUBLE PRECISION,
            outcome          TEXT,
            opened_at        TIMESTAMPTZ DEFAULT NOW(),
            closed_at        TIMESTAMPTZ,
            session_id       TEXT DEFAULT ''
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS so_strategy_idx ON strategy_outcomes (strategy_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS so_open_idx
        ON strategy_outcomes (strategy_id, symbol, closed_at)
    """)
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS strategy_daily_pnl (
            date                DATE NOT NULL,
            strategy_id         TEXT NOT NULL,
            realized_pnl        DOUBLE PRECISION DEFAULT 0,
            unrealized_pnl      DOUBLE PRECISION DEFAULT 0,
            num_trades          INTEGER DEFAULT 0,
            win_count           INTEGER DEFAULT 0,
            loss_count          INTEGER DEFAULT 0,
            PRIMARY KEY (date, strategy_id)
        )
    """))


def _migrate_universe_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS universe (
            symbol              TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            sector              TEXT DEFAULT 'Unknown',
            source              TEXT NOT NULL,
            market_cap          DOUBLE PRECISION,
            avg_daily_volume    DOUBLE PRECISION,
            is_active           BOOLEAN DEFAULT TRUE,
            added_at            TIMESTAMPTZ DEFAULT NOW(),
            last_refreshed      TIMESTAMPTZ DEFAULT NOW(),
            deactivated_reason  TEXT
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS universe_source_idx ON universe (source, is_active)
    """)


def _migrate_screener_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS screener_results (
            symbol              TEXT NOT NULL,
            screened_at         TIMESTAMPTZ NOT NULL,
            regime_used         TEXT,
            tier                INTEGER NOT NULL,
            composite_score     DOUBLE PRECISION NOT NULL,
            momentum_score      DOUBLE PRECISION,
            volatility_rank     DOUBLE PRECISION,
            volume_surge        DOUBLE PRECISION,
            regime_fit          DOUBLE PRECISION,
            catalyst_proximity  DOUBLE PRECISION,
            PRIMARY KEY (symbol, screened_at)
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS screener_latest_idx
        ON screener_results (screened_at DESC, tier)
    """)


def _migrate_coordination_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS loop_events (
            event_id    TEXT PRIMARY KEY,
            event_type  TEXT NOT NULL,
            source_loop TEXT NOT NULL,
            payload     JSONB,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS loop_events_type_idx ON loop_events (event_type, created_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS loop_events_created_idx ON loop_events (created_at)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loop_cursors (
            consumer_id     TEXT PRIMARY KEY,
            last_event_id   TEXT,
            last_polled_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    # Create loop_heartbeats if it doesn't exist (preserves data across restarts)
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS loop_heartbeats (
            loop_name           TEXT NOT NULL,
            iteration           INTEGER NOT NULL,
            started_at          TIMESTAMPTZ NOT NULL,
            finished_at         TIMESTAMPTZ,
            symbols_processed   INTEGER DEFAULT 0,
            errors              INTEGER DEFAULT 0,
            status              TEXT DEFAULT 'running',
            PRIMARY KEY (loop_name, iteration)
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS heartbeats_loop_idx
        ON loop_heartbeats (loop_name, started_at DESC)
    """)
    # Backfill: mark any pre-existing orphaned rows so the supervisor and
    # counter queries exclude them. Idempotent — safe to run on every startup.
    conn.execute("""
        UPDATE loop_heartbeats
           SET status = 'orphaned'
         WHERE status = 'running'
           AND finished_at IS NULL
           AND started_at < NOW() - INTERVAL '30 minutes'
    """)


def _migrate_conversations_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS agent_conversations (
            conversation_id  TEXT PRIMARY KEY,
            session_id       TEXT NOT NULL,
            loop_name        TEXT DEFAULT 'trading_operator',
            iteration        INTEGER,
            agent_name       TEXT NOT NULL,
            role             TEXT NOT NULL,
            symbol           TEXT,
            strategy_id      TEXT,
            content          TEXT NOT NULL,
            summary          TEXT,
            created_at       TIMESTAMPTZ DEFAULT NOW(),
            metadata         JSONB
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS conv_agent_idx
        ON agent_conversations (agent_name, created_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS conv_symbol_idx
        ON agent_conversations (symbol, created_at DESC)
    """)
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS signal_snapshots (
            snapshot_id          TEXT PRIMARY KEY,
            symbol               TEXT NOT NULL,
            created_at           TIMESTAMPTZ DEFAULT NOW(),
            technical            JSONB,
            regime               JSONB,
            volume               JSONB,
            risk                 JSONB,
            sentiment            JSONB,
            fundamentals         JSONB,
            events               JSONB,
            consensus_bias       TEXT,
            consensus_conviction DOUBLE PRECISION,
            collector_failures   JSONB
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS snap_symbol_idx
        ON signal_snapshots (symbol, created_at DESC)
    """)


def _migrate_attribution_pg(conn: PgConnection) -> None:
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS daily_equity (
            date                DATE PRIMARY KEY,
            cash                DOUBLE PRECISION NOT NULL,
            positions_value     DOUBLE PRECISION NOT NULL,
            total_equity        DOUBLE PRECISION NOT NULL,
            daily_pnl           DOUBLE PRECISION NOT NULL,
            cumulative_pnl      DOUBLE PRECISION NOT NULL,
            daily_return_pct    DOUBLE PRECISION NOT NULL,
            high_water_mark     DOUBLE PRECISION NOT NULL,
            drawdown_pct        DOUBLE PRECISION NOT NULL,
            open_positions      INTEGER NOT NULL,
            created_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS benchmark_daily (
            date                DATE NOT NULL,
            benchmark           TEXT NOT NULL,
            close_price         DOUBLE PRECISION NOT NULL,
            daily_return_pct    DOUBLE PRECISION,
            cumulative_return   DOUBLE PRECISION,
            PRIMARY KEY (date, benchmark)
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS benchmark_comparison (
            date                DATE NOT NULL,
            benchmark           TEXT NOT NULL,
            window_days         INTEGER NOT NULL,
            portfolio_sharpe    DOUBLE PRECISION,
            benchmark_sharpe    DOUBLE PRECISION,
            portfolio_sortino   DOUBLE PRECISION,
            alpha               DOUBLE PRECISION,
            beta                DOUBLE PRECISION,
            PRIMARY KEY (date, benchmark, window_days)
        )
    """))


def _migrate_equity_alerts_pg(conn: PgConnection) -> None:
    for seq in ("equity_alerts_seq", "alert_exit_signals_seq", "alert_updates_seq"):
        conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq} START 1")

    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS equity_alerts (
            id                  BIGINT PRIMARY KEY DEFAULT nextval('equity_alerts_seq'),
            symbol              TEXT NOT NULL,
            action              TEXT NOT NULL,
            time_horizon        TEXT NOT NULL,
            instrument_type     TEXT DEFAULT 'equity',
            strategy_id         TEXT DEFAULT '',
            strategy_name       TEXT DEFAULT '',
            confidence          DOUBLE PRECISION DEFAULT 0,
            debate_verdict      TEXT DEFAULT '',
            debate_summary      TEXT DEFAULT '',
            current_price       DOUBLE PRECISION,
            suggested_entry     DOUBLE PRECISION,
            stop_price          DOUBLE PRECISION,
            target_price        DOUBLE PRECISION,
            trailing_stop_pct   DOUBLE PRECISION,
            risk_reward_ratio   DOUBLE PRECISION,
            regime              TEXT DEFAULT 'unknown',
            sector              TEXT DEFAULT '',
            catalyst            TEXT DEFAULT '',
            thesis              TEXT DEFAULT '',
            key_risks           TEXT DEFAULT '',
            piotroski_f_score   INTEGER,
            fcf_yield_pct       DOUBLE PRECISION,
            pe_ratio            DOUBLE PRECISION,
            analyst_consensus   TEXT DEFAULT '',
            status              TEXT DEFAULT 'pending',
            status_reason       TEXT DEFAULT '',
            urgency             TEXT DEFAULT 'today',
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            acted_at            TIMESTAMPTZ,
            expired_at          TIMESTAMPTZ
        )
    """))
    # Fix column defaults if table was created before sequences/defaults existed
    _alert_defaults = [
        ("equity_alerts", "id", "nextval('equity_alerts_seq')"),
        ("equity_alerts", "created_at", "NOW()"),
        ("equity_alerts", "status", "'pending'"),
        ("equity_alerts", "urgency", "'today'"),
        ("alert_exit_signals", "id", "nextval('alert_exit_signals_seq')"),
        ("alert_exit_signals", "created_at", "NOW()"),
        ("alert_updates", "id", "nextval('alert_updates_seq')"),
        ("alert_updates", "created_at", "NOW()"),
    ]
    for tbl, col, default in _alert_defaults:
        conn.execute(f"ALTER TABLE IF EXISTS {tbl} ALTER COLUMN {col} SET DEFAULT {default}")

    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS alert_exit_signals (
            id                  BIGINT PRIMARY KEY DEFAULT nextval('alert_exit_signals_seq'),
            alert_id            BIGINT NOT NULL,
            signal_type         TEXT NOT NULL,
            severity            TEXT DEFAULT 'warning',
            exit_price          DOUBLE PRECISION,
            pnl_pct             DOUBLE PRECISION,
            headline            TEXT NOT NULL,
            commentary          TEXT DEFAULT '',
            what_changed        TEXT DEFAULT '',
            lesson              TEXT DEFAULT '',
            recommended_action  TEXT DEFAULT 'hold',
            recommended_reason  TEXT DEFAULT '',
            acknowledged        BOOLEAN DEFAULT false,
            action_taken        TEXT DEFAULT '',
            created_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS alert_updates (
            id                  BIGINT PRIMARY KEY DEFAULT nextval('alert_updates_seq'),
            alert_id            BIGINT NOT NULL,
            update_type         TEXT NOT NULL,
            commentary          TEXT NOT NULL,
            data_snapshot       TEXT DEFAULT '',
            thesis_status       TEXT DEFAULT 'intact',
            created_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """))


# ---------------------------------------------------------------------------
# Migrations — Market data / analytics tables (PostgreSQL)
# ---------------------------------------------------------------------------


def _migrate_market_data_pg(conn: PgConnection) -> None:
    """Create market data tables in PostgreSQL."""

    # -- OHLCV ----------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol    VARCHAR          NOT NULL,
            timeframe VARCHAR          NOT NULL,
            timestamp TIMESTAMPTZ      NOT NULL,
            open      DOUBLE PRECISION NOT NULL,
            high      DOUBLE PRECISION NOT NULL,
            low       DOUBLE PRECISION NOT NULL,
            close     DOUBLE PRECISION NOT NULL,
            volume    DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
        ON ohlcv (symbol, timeframe, timestamp)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_metadata (
            symbol          VARCHAR     NOT NULL,
            timeframe       VARCHAR     NOT NULL,
            first_timestamp TIMESTAMPTZ,
            last_timestamp  TIMESTAMPTZ,
            row_count       INTEGER,
            updated_at      TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, timeframe)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_1m (
            symbol      VARCHAR          NOT NULL,
            timestamp   TIMESTAMPTZ      NOT NULL,
            open        DOUBLE PRECISION NOT NULL,
            high        DOUBLE PRECISION NOT NULL,
            low         DOUBLE PRECISION NOT NULL,
            close       DOUBLE PRECISION NOT NULL,
            volume      DOUBLE PRECISION NOT NULL,
            vwap        DOUBLE PRECISION,
            trade_count INTEGER,
            PRIMARY KEY (symbol, timestamp)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_ts
        ON ohlcv_1m (symbol, timestamp)
    """)

    # -- Fundamentals ---------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS financial_statements (
            ticker           VARCHAR NOT NULL,
            statement_type   VARCHAR NOT NULL,
            period_type      VARCHAR NOT NULL,
            report_period    DATE    NOT NULL,
            revenue          DOUBLE PRECISION,
            net_income       DOUBLE PRECISION,
            total_assets     DOUBLE PRECISION,
            total_debt       DOUBLE PRECISION,
            operating_income DOUBLE PRECISION,
            gross_profit     DOUBLE PRECISION,
            eps_diluted      DOUBLE PRECISION,
            data             JSONB,
            fetched_at       TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, report_period, statement_type, period_type)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fin_stmt_ticker
        ON financial_statements (ticker, statement_type)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS financial_metrics (
            ticker           VARCHAR NOT NULL,
            date             DATE    NOT NULL,
            period_type      VARCHAR NOT NULL DEFAULT 'annual',
            market_cap       DOUBLE PRECISION,
            pe_ratio         DOUBLE PRECISION,
            pb_ratio         DOUBLE PRECISION,
            ps_ratio         DOUBLE PRECISION,
            ev_to_ebitda     DOUBLE PRECISION,
            roe              DOUBLE PRECISION,
            roa              DOUBLE PRECISION,
            gross_margin     DOUBLE PRECISION,
            operating_margin DOUBLE PRECISION,
            net_margin       DOUBLE PRECISION,
            debt_to_equity   DOUBLE PRECISION,
            current_ratio    DOUBLE PRECISION,
            dividend_yield   DOUBLE PRECISION,
            revenue_growth   DOUBLE PRECISION,
            earnings_growth  DOUBLE PRECISION,
            data             JSONB,
            fetched_at       TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, date, period_type)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fin_metrics_ticker
        ON financial_metrics (ticker, date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS insider_trades (
            ticker             VARCHAR NOT NULL,
            transaction_date   DATE    NOT NULL,
            owner_name         VARCHAR NOT NULL,
            owner_title        VARCHAR,
            transaction_type   VARCHAR,
            shares             DOUBLE PRECISION,
            price_per_share    DOUBLE PRECISION,
            total_value        DOUBLE PRECISION,
            shares_owned_after DOUBLE PRECISION,
            filing_date        DATE,
            fetched_at         TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, transaction_date, owner_name, shares)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
        ON insider_trades (ticker, transaction_date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS institutional_ownership (
            ticker        VARCHAR NOT NULL,
            investor_name VARCHAR NOT NULL,
            report_date   DATE    NOT NULL,
            shares_held   DOUBLE PRECISION,
            market_value  DOUBLE PRECISION,
            portfolio_pct DOUBLE PRECISION,
            change_shares DOUBLE PRECISION,
            change_pct    DOUBLE PRECISION,
            fetched_at    TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, investor_name, report_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_instit_ticker
        ON institutional_ownership (ticker, report_date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_indicators (
            indicator  VARCHAR          NOT NULL,
            date       DATE             NOT NULL,
            value      DOUBLE PRECISION,
            fetched_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (indicator, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_macro_indicator_date
        ON macro_indicators (indicator, date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS corporate_actions (
            ticker           VARCHAR NOT NULL,
            action_type      VARCHAR NOT NULL,
            effective_date   DATE    NOT NULL,
            amount           DOUBLE PRECISION,
            declaration_date DATE,
            record_date      DATE,
            payment_date     DATE,
            fetched_at       TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, action_type, effective_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_corp_actions_ticker
        ON corporate_actions (ticker, effective_date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_estimates (
            ticker       VARCHAR NOT NULL,
            fiscal_date  DATE    NOT NULL,
            period_type  VARCHAR NOT NULL DEFAULT 'annual',
            metric       VARCHAR NOT NULL DEFAULT 'eps',
            consensus    DOUBLE PRECISION,
            high         DOUBLE PRECISION,
            low          DOUBLE PRECISION,
            num_analysts INTEGER,
            fetched_at   TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, fiscal_date, period_type, metric)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            ticker           VARCHAR NOT NULL,
            accession_number VARCHAR NOT NULL,
            filing_type      VARCHAR,
            filed_date       DATE,
            period_of_report DATE,
            url              VARCHAR,
            fetched_at       TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ticker, accession_number)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker
        ON sec_filings (ticker, filed_date)
    """)

    # -- Options / earnings / overview / news ---------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS options_chains (
            contract_id   VARCHAR          NOT NULL,
            underlying    VARCHAR          NOT NULL,
            data_date     DATE             NOT NULL,
            expiry        DATE             NOT NULL,
            strike        DOUBLE PRECISION NOT NULL,
            option_type   VARCHAR          NOT NULL,
            bid           DOUBLE PRECISION,
            ask           DOUBLE PRECISION,
            mid           DOUBLE PRECISION,
            last          DOUBLE PRECISION,
            volume        INTEGER,
            open_interest INTEGER,
            iv            DOUBLE PRECISION,
            delta         DOUBLE PRECISION,
            gamma         DOUBLE PRECISION,
            theta         DOUBLE PRECISION,
            vega          DOUBLE PRECISION,
            rho           DOUBLE PRECISION,
            PRIMARY KEY (contract_id, data_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_options_underlying_date
        ON options_chains (underlying, data_date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_options_expiry
        ON options_chains (expiry)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            symbol             VARCHAR NOT NULL,
            report_date        DATE    NOT NULL,
            fiscal_date_ending DATE,
            estimate           DOUBLE PRECISION,
            reported_eps       DOUBLE PRECISION,
            surprise           DOUBLE PRECISION,
            surprise_pct       DOUBLE PRECISION,
            PRIMARY KEY (symbol, report_date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_earnings_date
        ON earnings_calendar (report_date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS company_overview (
            symbol              VARCHAR NOT NULL PRIMARY KEY,
            name                VARCHAR,
            sector              VARCHAR,
            industry            VARCHAR,
            description         TEXT    DEFAULT '',
            market_cap          DOUBLE PRECISION,
            dividend_yield      DOUBLE PRECISION,
            ex_dividend_date    DATE,
            fifty_two_week_high DOUBLE PRECISION,
            fifty_two_week_low  DOUBLE PRECISION,
            beta                DOUBLE PRECISION,
            updated_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    # Idempotent migration for existing DBs that predate this column.
    conn.execute("""
        ALTER TABLE company_overview
        ADD COLUMN IF NOT EXISTS description TEXT DEFAULT ''
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            time_published          TIMESTAMPTZ NOT NULL,
            title                   VARCHAR     NOT NULL,
            summary                 VARCHAR,
            source                  VARCHAR,
            url                     VARCHAR,
            ticker                  VARCHAR     NOT NULL,
            overall_sentiment_score DOUBLE PRECISION,
            overall_sentiment_label VARCHAR,
            ticker_sentiment_score  DOUBLE PRECISION,
            ticker_sentiment_label  VARCHAR,
            relevance_score         DOUBLE PRECISION,
            PRIMARY KEY (time_published, title, ticker)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_time
        ON news_sentiment (time_published)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_ticker
        ON news_sentiment (ticker)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_ticker_time
        ON news_sentiment (ticker, time_published)
    """)

    # -- Put-Call Ratio (AV Data Expansion) --------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS put_call_ratio (
            symbol      VARCHAR          NOT NULL,
            date        DATE             NOT NULL,
            put_volume  DOUBLE PRECISION,
            call_volume DOUBLE PRECISION,
            pcr         DOUBLE PRECISION,
            source      VARCHAR          NOT NULL DEFAULT 'computed',
            fetched_at  TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, date, source)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pcr_symbol_date
        ON put_call_ratio (symbol, date)
    """)

    # -- Delisting status on company_overview ------------------------------
    conn.execute("""
        ALTER TABLE company_overview
        ADD COLUMN IF NOT EXISTS delisted_at DATE
    """)

    logger.debug("[DB] Market data tables migrated")


# ---------------------------------------------------------------------------
# Migrations — Analytics tables (PostgreSQL)
# ---------------------------------------------------------------------------


def _migrate_research_wip_pg(conn: PgConnection) -> None:
    """Research work-in-progress tracking (prevents duplicate research by parallel agents)."""
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS research_wip (
            symbol              TEXT NOT NULL,
            domain              TEXT NOT NULL CHECK (domain IN ('investment', 'swing', 'options')),
            agent_id            TEXT NOT NULL,
            started_at          TIMESTAMPTZ DEFAULT NOW(),
            heartbeat_at        TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, domain)
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_research_wip_heartbeat
        ON research_wip(heartbeat_at)
    """)
    logger.debug("[DB] Research WIP table migrated")


def _migrate_analytics_pg(conn: PgConnection) -> None:
    """Create analytics tables in PostgreSQL (ML experiments, research programs, reflexion)."""
    # -- ML / research --------------------------------------------------
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS ml_experiments (
            experiment_id       TEXT PRIMARY KEY,
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            symbol              TEXT NOT NULL,
            model_type          TEXT NOT NULL,
            feature_tiers       JSONB,
            label_method        TEXT DEFAULT 'event',
            n_features_raw      INTEGER,
            n_features_filtered INTEGER,
            test_auc            DOUBLE PRECISION,
            test_accuracy       DOUBLE PRECISION,
            cv_auc_mean         DOUBLE PRECISION,
            cv_auc_std          DOUBLE PRECISION,
            top_features        JSONB,
            causal_dropped      JSONB,
            hyperparams         JSONB,
            training_duration_s DOUBLE PRECISION,
            hypothesis_id       TEXT,
            verdict             TEXT DEFAULT 'pending',
            failure_analysis    TEXT,
            notes               TEXT
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS mlexp_symbol_idx
        ON ml_experiments (symbol, created_at DESC)
    """)
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS alpha_research_program (
            investigation_id    TEXT PRIMARY KEY,
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            updated_at          TIMESTAMPTZ DEFAULT NOW(),
            thesis              TEXT NOT NULL,
            status              TEXT DEFAULT 'active',
            priority            INTEGER DEFAULT 5,
            source              TEXT,
            experiments_run     INTEGER DEFAULT 0,
            best_oos_sharpe     DOUBLE PRECISION,
            last_result_summary TEXT,
            next_steps          TEXT,
            dead_end_reason     TEXT,
            target_regimes      JSONB,
            target_symbols      JSONB
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS ml_research_program (
            program_id          TEXT PRIMARY KEY,
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            updated_at          TIMESTAMPTZ DEFAULT NOW(),
            focus_area          TEXT NOT NULL,
            status              TEXT DEFAULT 'active',
            hypothesis          TEXT,
            experiments_run     INTEGER DEFAULT 0,
            best_metric         DOUBLE PRECISION,
            best_config         JSONB,
            lessons_learned     TEXT,
            next_experiment     TEXT
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS research_plans (
            plan_id             TEXT PRIMARY KEY,
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            pod_name            TEXT NOT NULL,
            plan_type           TEXT NOT NULL,
            plan_json           JSONB NOT NULL,
            context_summary     TEXT,
            executed            BOOLEAN DEFAULT FALSE,
            execution_results   JSONB
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS breakthrough_features (
            feature_name        TEXT PRIMARY KEY,
            first_seen          TIMESTAMPTZ DEFAULT NOW(),
            last_seen           TIMESTAMPTZ DEFAULT NOW(),
            occurrence_count    INTEGER DEFAULT 1,
            avg_shap_importance DOUBLE PRECISION,
            winning_strategies  JSONB,
            regimes_effective   JSONB
        )
    """))

    # -- Reflections ----------------------------------------------------
    conn.execute("CREATE SEQUENCE IF NOT EXISTS trade_reflections_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS trade_reflections (
            id               BIGINT PRIMARY KEY DEFAULT nextval('trade_reflections_seq'),
            symbol           TEXT NOT NULL,
            strategy_id      TEXT,
            action           TEXT,
            entry_price      DOUBLE PRECISION,
            exit_price       DOUBLE PRECISION,
            realized_pnl_pct DOUBLE PRECISION,
            holding_days     INTEGER,
            regime_at_entry  TEXT,
            regime_at_exit   TEXT,
            conviction       DOUBLE PRECISION,
            signals_entry    TEXT,
            signals_exit     TEXT,
            lesson           TEXT,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_reflections_regime_symbol "
        "ON trade_reflections (regime_at_entry, symbol)"
    )

    # -- Optimization / reflexion ---------------------------------------
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS reflexion_episodes (
            episode_id           TEXT PRIMARY KEY,
            trade_id             BIGINT,
            regime               TEXT NOT NULL,
            strategy_id          TEXT NOT NULL,
            symbol               TEXT NOT NULL,
            pnl_pct              DOUBLE PRECISION NOT NULL,
            root_cause           TEXT NOT NULL,
            verbal_reinforcement TEXT NOT NULL,
            counterfactual       TEXT,
            tags                 JSONB,
            created_at           TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(
        "CREATE INDEX IF NOT EXISTS reflex_regime_strat_idx "
        "ON reflexion_episodes (regime, strategy_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS reflex_symbol_idx "
        "ON reflexion_episodes (symbol, created_at DESC)"
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS step_credits_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS step_credits (
            id                  BIGINT PRIMARY KEY DEFAULT nextval('step_credits_seq'),
            trade_id            BIGINT NOT NULL,
            step_type           TEXT NOT NULL,
            step_output         TEXT,
            credit_score        DOUBLE PRECISION NOT NULL,
            attribution_method  TEXT NOT NULL,
            evidence            TEXT,
            created_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute("CREATE INDEX IF NOT EXISTS sc_trade_idx ON step_credits (trade_id)")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS prompt_versions_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id             BIGINT PRIMARY KEY DEFAULT nextval('prompt_versions_seq'),
            node_name      TEXT NOT NULL,
            version        INTEGER NOT NULL,
            prompt_text    TEXT NOT NULL,
            source         TEXT NOT NULL,
            parent_version INTEGER,
            status         TEXT DEFAULT 'proposed',
            avg_pnl_since  DOUBLE PRECISION,
            trades_since   INTEGER DEFAULT 0,
            created_at     TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (node_name, version)
        )
    """))
    conn.execute("CREATE SEQUENCE IF NOT EXISTS prompt_critiques_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS prompt_critiques (
            id           BIGINT PRIMARY KEY DEFAULT nextval('prompt_critiques_seq'),
            trade_id     BIGINT NOT NULL,
            node_name    TEXT NOT NULL,
            critique     TEXT NOT NULL,
            proposed_edit TEXT,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS judge_verdicts (
            verdict_id       TEXT PRIMARY KEY,
            hypothesis_id    TEXT NOT NULL,
            approved         BOOLEAN NOT NULL,
            score            DOUBLE PRECISION NOT NULL,
            flags            JSONB,
            reasoning        TEXT,
            similar_failures JSONB,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS prompt_candidates (
            candidate_id     TEXT PRIMARY KEY,
            node_name        TEXT NOT NULL,
            prompt_text      TEXT NOT NULL,
            generation       INTEGER NOT NULL,
            fitness          DOUBLE PRECISION,
            trades_evaluated INTEGER DEFAULT 0,
            status           TEXT DEFAULT 'candidate',
            source           TEXT DEFAULT 'opro',
            parent_ids       JSONB,
            created_at       TIMESTAMPTZ DEFAULT NOW(),
            promoted_at      TIMESTAMPTZ,
            retired_at       TIMESTAMPTZ
        )
    """))
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pc_node_status_idx "
        "ON prompt_candidates (node_name, status)"
    )
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS research_trajectories (
            trajectory_id    TEXT PRIMARY KEY,
            generation       INTEGER NOT NULL,
            segments         JSONB NOT NULL,
            overall_fitness  DOUBLE PRECISION,
            parent_ids       JSONB,
            mutation_applied TEXT,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn.execute(
        "CREATE INDEX IF NOT EXISTS rt_gen_idx "
        "ON research_trajectories (generation, overall_fitness DESC)"
    )
    logger.debug("[DB] Analytics tables migrated")


# ---------------------------------------------------------------------------
# Research queue — AutoResearchClaw task inbox
# ---------------------------------------------------------------------------


def _migrate_research_queue_pg(conn: PgConnection) -> None:
    """research_queue — AutoResearchClaw task inbox.

    Populated by: trade-reflector (bug_fix), DriftDetector (ml_arch_search),
    research loop (strategy_hypothesis), and manual inserts.
    Consumed by: scripts/autoresclaw_runner.py (Sunday 20:00 scheduler job).
    """
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS research_queue (
            task_id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
            task_type        TEXT NOT NULL
                             CHECK (task_type IN (
                                 'ml_arch_search', 'rl_env_design',
                                 'bug_fix', 'strategy_hypothesis'
                             )),
            priority         INTEGER NOT NULL DEFAULT 5,
            topic            TEXT,
            context_json     JSONB NOT NULL DEFAULT '{}',
            status           TEXT NOT NULL DEFAULT 'pending'
                             CHECK (status IN ('pending', 'running', 'done', 'failed')),
            source           TEXT,
            result_path      TEXT,
            error_message    TEXT,
            created_at       TIMESTAMPTZ DEFAULT NOW(),
            started_at       TIMESTAMPTZ,
            completed_at     TIMESTAMPTZ
        )
    """))
    # Idempotent column addition for tables created before topic was added.
    conn.execute(
        "ALTER TABLE research_queue ADD COLUMN IF NOT EXISTS topic TEXT"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS rq_status_priority_idx "
        "ON research_queue (status, priority DESC, created_at)"
    )
    logger.debug("[DB] research_queue table migrated")


def _migrate_loop_context_pg(conn: PgConnection) -> None:
    """loop_iteration_context — stateless loop iteration state.

    Replaces in-session state[] dict in trading/research loop prompts.
    Each loop writes its per-iteration context here so that `claude` (no
    --continue) can start fresh each iteration and read prior state from DB.

    Keys used by trading_loop: market_intel, stale_symbols,
        closes_since_review, last_weekly_review_at
    Keys used by research_loop: last_domain, domain_history,
        last_execution_audit_at, cross_domain_transfers
    """
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS loop_iteration_context (
            loop_name    TEXT        NOT NULL,
            context_key  TEXT        NOT NULL,
            context_json JSONB       NOT NULL DEFAULT '{}',
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (loop_name, context_key)
        )
    """))
    conn.execute(
        "CREATE INDEX IF NOT EXISTS lic_loop_name_idx "
        "ON loop_iteration_context (loop_name)"
    )
    logger.debug("[DB] loop_iteration_context table migrated")


def _migrate_bugs_pg(conn: PgConnection) -> None:
    """bugs — persistent bug tracker with full lifecycle.

    Every tool failure recorded by record_tool_error() lands here.
    Status lifecycle: open → in_progress → fixed | reverted | wont_fix

    Deduplication key: (tool_name, loop_name, error_fingerprint).
    error_fingerprint is the first 120 chars of the error message, which
    is stable enough to group repeated identical failures without being so
    broad that different bugs collapse into one row.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bugs (
            bug_id              TEXT        NOT NULL PRIMARY KEY,
            tool_name           TEXT        NOT NULL,
            loop_name           TEXT        NOT NULL DEFAULT 'trading_loop',
            error_message       TEXT        NOT NULL,
            error_fingerprint   TEXT        NOT NULL,
            stack_trace         TEXT,
            status              TEXT        NOT NULL DEFAULT 'open'
                                    CHECK (status IN ('open','in_progress','fixed','reverted','wont_fix')),
            priority            INTEGER     NOT NULL DEFAULT 5,
            consecutive_errors  INTEGER     NOT NULL DEFAULT 1,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            fixed_at            TIMESTAMPTZ,
            fix_commit          TEXT,
            fix_summary         TEXT,
            arc_task_id         TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS bugs_status_priority_idx "
        "ON bugs (status, priority DESC, created_at ASC)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS bugs_dedup_idx "
        "ON bugs (tool_name, loop_name, error_fingerprint) "
        "WHERE status IN ('open', 'in_progress')"
    )
    logger.debug("[DB] bugs table migrated")

    # -- Graph runner checkpoints (LangGraph cycle tracking) --
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_checkpoints (
            graph_name          TEXT        NOT NULL,
            cycle_number        INTEGER     NOT NULL,
            duration_seconds    REAL        NOT NULL,
            status              TEXT        NOT NULL DEFAULT 'success',
            error_message       TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (graph_name, cycle_number)
        )
    """)
    logger.debug("[DB] graph_checkpoints table migrated")

    # -- Agent events (real-time dashboard feed) --
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_events (
            id              BIGSERIAL   PRIMARY KEY,
            graph_name      TEXT        NOT NULL,
            node_name       TEXT        NOT NULL,
            agent_name      TEXT        NOT NULL DEFAULT '',
            event_type      TEXT        NOT NULL,
            content         TEXT        NOT NULL DEFAULT '',
            metadata        JSONB       NOT NULL DEFAULT '{}',
            cycle_number    INTEGER     NOT NULL DEFAULT 0,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_events_created "
        "ON agent_events (created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_events_graph "
        "ON agent_events (graph_name, created_at DESC)"
    )
    logger.debug("[DB] agent_events table migrated")


def _migrate_risk_monitoring_pg(conn: PgConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_snapshots (
            id                  BIGSERIAL       PRIMARY KEY,
            snapshot_time       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            total_equity        DOUBLE PRECISION NOT NULL,
            gross_exposure      DOUBLE PRECISION NOT NULL,
            net_exposure        DOUBLE PRECISION NOT NULL,
            largest_position_pct DOUBLE PRECISION NOT NULL,
            position_count      INTEGER         NOT NULL,
            daily_pnl           DOUBLE PRECISION NOT NULL,
            var_95              DOUBLE PRECISION,
            var_99              DOUBLE PRECISION,
            cvar_99             DOUBLE PRECISION,
            portfolio_dd_pct    DOUBLE PRECISION,
            market_beta         DOUBLE PRECISION,
            momentum_beta       DOUBLE PRECISION,
            value_beta          DOUBLE PRECISION,
            avg_pairwise_corr   DOUBLE PRECISION,
            escalation_level    TEXT,
            metadata            JSONB           DEFAULT '{}'
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_risk_snapshots_time "
        "ON risk_snapshots (snapshot_time DESC)"
    )
    logger.debug("[DB] risk_snapshots table migrated")


def _migrate_stat_arb_pg(conn: PgConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stat_arb_pairs (
            id                  BIGSERIAL       PRIMARY KEY,
            symbol_a            TEXT            NOT NULL,
            symbol_b            TEXT            NOT NULL,
            sector              TEXT,
            p_value             DOUBLE PRECISION NOT NULL,
            half_life_days      DOUBLE PRECISION,
            current_z_score     DOUBLE PRECISION,
            beta                DOUBLE PRECISION,
            is_active           BOOLEAN         DEFAULT TRUE,
            discovered_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            UNIQUE (symbol_a, symbol_b)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stat_arb_pairs_active "
        "ON stat_arb_pairs (is_active) WHERE is_active = TRUE"
    )
    logger.debug("[DB] stat_arb_pairs table migrated")


def _migrate_capital_allocation_pg(conn: PgConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_capital_allocations (
            id                  BIGSERIAL       PRIMARY KEY,
            strategy_id         TEXT            NOT NULL,
            allocated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            score               DOUBLE PRECISION NOT NULL,
            budget_pct          DOUBLE PRECISION NOT NULL,
            budget_dollars      DOUBLE PRECISION NOT NULL,
            sharpe_component    DOUBLE PRECISION,
            capacity_component  DOUBLE PRECISION,
            correlation_penalty DOUBLE PRECISION,
            regime_fit          DOUBLE PRECISION
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategy_alloc_latest "
        "ON strategy_capital_allocations (strategy_id, allocated_at DESC)"
    )
    logger.debug("[DB] strategy_capital_allocations table migrated")


def _migrate_ml_pipeline_pg(conn: PgConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            id                  BIGSERIAL       PRIMARY KEY,
            symbol              TEXT            NOT NULL,
            bar_type            TEXT            NOT NULL,
            timestamp           TIMESTAMPTZ     NOT NULL,
            open                DOUBLE PRECISION NOT NULL,
            high                DOUBLE PRECISION NOT NULL,
            low                 DOUBLE PRECISION NOT NULL,
            close               DOUBLE PRECISION NOT NULL,
            volume              BIGINT          NOT NULL,
            dollar_volume       DOUBLE PRECISION NOT NULL,
            tick_count          INTEGER         NOT NULL,
            vwap                DOUBLE PRECISION NOT NULL,
            bar_duration_seconds INTEGER        NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bars_symbol_type_ts "
        "ON bars (symbol, bar_type, timestamp)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_params (
            id          BIGSERIAL       PRIMARY KEY,
            symbol      TEXT            NOT NULL,
            param_name  TEXT            NOT NULL,
            param_value DOUBLE PRECISION NOT NULL,
            updated_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        )
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_params_symbol_param "
        "ON feature_params (symbol, param_name)"
    )
    logger.debug("[DB] ML pipeline tables (bars, feature_params) migrated")


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def _migrate_tool_search_metrics_pg(conn: PgConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_search_metrics (
            id                  SERIAL PRIMARY KEY,
            computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            time_window_h       INTEGER NOT NULL,
            search_hit_rate     DOUBLE PRECISION NOT NULL,
            discovery_accuracy  DOUBLE PRECISION NOT NULL,
            fallback_rate       DOUBLE PRECISION NOT NULL,
            top_missed_tools    TEXT[],
            total_searches      INTEGER NOT NULL,
            total_discoveries   INTEGER NOT NULL,
            total_misses        INTEGER NOT NULL,
            total_fallbacks     INTEGER NOT NULL
        )
    """)
    logger.debug("[DB] tool_search_metrics table migrated")


def _migrate_trade_quality_pg(conn: PgConnection) -> None:
    # Repair closed_trades if it was created without a PRIMARY KEY
    # (legacy schema from DuckDB era). The FK below requires a unique constraint.
    row = conn.execute("""
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'closed_trades'::regclass AND contype = 'p'
    """).fetchone()
    if row is None:
        # Upgrade id column to BIGINT with PK + default sequence
        conn.execute("CREATE SEQUENCE IF NOT EXISTS closed_trades_seq START 1")
        conn.execute("""
            ALTER TABLE closed_trades
                ALTER COLUMN id SET DATA TYPE BIGINT,
                ALTER COLUMN id SET DEFAULT nextval('closed_trades_seq'),
                ALTER COLUMN id SET NOT NULL,
                ADD PRIMARY KEY (id)
        """)
        logger.info("[DB] Repaired closed_trades: added PRIMARY KEY on id")

    conn.execute("CREATE SEQUENCE IF NOT EXISTS trade_quality_scores_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS trade_quality_scores (
            id                BIGINT PRIMARY KEY DEFAULT nextval('trade_quality_scores_seq'),
            trade_id          BIGINT REFERENCES closed_trades(id),
            cycle_number      INTEGER NOT NULL,
            execution_quality DOUBLE PRECISION NOT NULL,
            thesis_accuracy   DOUBLE PRECISION NOT NULL,
            risk_management   DOUBLE PRECISION NOT NULL,
            timing_quality    DOUBLE PRECISION NOT NULL,
            sizing_quality    DOUBLE PRECISION NOT NULL,
            overall_score     DOUBLE PRECISION NOT NULL,
            justification     TEXT NOT NULL,
            scored_at         TIMESTAMPTZ DEFAULT NOW(),
            model_used        TEXT NOT NULL
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS tqs_trade_idx ON trade_quality_scores (trade_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS tqs_cycle_idx ON trade_quality_scores (cycle_number)
    """)


def _compute_us_holidays(year: int) -> list[tuple]:
    """Return US market holidays for a given year.

    Each tuple: (date, name, market_status, close_time_str_or_None).
    """
    import calendar as cal
    from datetime import date as d
    from datetime import timedelta

    from dateutil.easter import easter

    def _observe(dt: d) -> d:
        """Apply weekend observance rules."""
        if dt.weekday() == 5:  # Saturday -> Friday
            return dt - timedelta(days=1)
        if dt.weekday() == 6:  # Sunday -> Monday
            return dt + timedelta(days=1)
        return dt

    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> d:
        """Return the nth occurrence of weekday in month/year (1-indexed)."""
        first = d(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))

    def _last_weekday(year: int, month: int, weekday: int) -> d:
        """Return the last occurrence of weekday in month/year."""
        last_day = d(year, month, cal.monthrange(year, month)[1])
        offset = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=offset)

    holidays = []

    # Full closures
    holidays.append((_observe(d(year, 1, 1)), "New Year's Day", "closed", None))
    holidays.append((_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day", "closed", None))  # 3rd Monday Jan
    holidays.append((_nth_weekday(year, 2, 0, 3), "Presidents' Day", "closed", None))  # 3rd Monday Feb
    good_friday = easter(year) - timedelta(days=2)
    holidays.append((good_friday, "Good Friday", "closed", None))
    holidays.append((_last_weekday(year, 5, 0), "Memorial Day", "closed", None))  # Last Monday May
    holidays.append((_observe(d(year, 6, 19)), "Juneteenth", "closed", None))
    holidays.append((_observe(d(year, 7, 4)), "Independence Day", "closed", None))
    holidays.append((_nth_weekday(year, 9, 0, 1), "Labor Day", "closed", None))  # 1st Monday Sep
    thanksgiving = _nth_weekday(year, 11, 3, 4)  # 4th Thursday Nov
    holidays.append((thanksgiving, "Thanksgiving", "closed", None))
    holidays.append((_observe(d(year, 12, 25)), "Christmas", "closed", None))

    # Early closures (13:00 ET)
    early_time = "13:00:00"
    # Day before Independence Day (only if Jul 4 not Monday)
    jul4 = d(year, 7, 4)
    if jul4.weekday() != 0:  # not Monday
        jul3_observed = _observe(d(year, 7, 3))
        if jul3_observed.weekday() < 5:  # weekday
            holidays.append((jul3_observed, "Day Before Independence Day", "early_close", early_time))
    # Black Friday
    holidays.append((thanksgiving + timedelta(days=1), "Black Friday", "early_close", early_time))
    # Christmas Eve (only if Dec 25 not Monday)
    dec25 = d(year, 12, 25)
    if dec25.weekday() != 0:  # not Monday
        dec24 = d(year, 12, 24)
        if dec24.weekday() < 5:  # weekday
            holidays.append((dec24, "Christmas Eve", "early_close", early_time))

    return holidays


def _seed_market_holidays(conn: PgConnection, year: int) -> None:
    """Seed market_holidays for a given year. Idempotent via ON CONFLICT."""
    holidays = _compute_us_holidays(year)
    for dt, name, status, close_time in holidays:
        conn.execute(
            "INSERT INTO market_holidays (date, name, market_status, close_time, exchange) "
            "VALUES (%s, %s, %s, %s, 'NYSE') ON CONFLICT DO NOTHING",
            (dt, name, status, close_time),
        )


def _migrate_market_holidays_pg(conn: PgConnection) -> None:
    """Create market_holidays table and seed US holidays."""
    import datetime

    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_holidays (
            date            DATE NOT NULL,
            name            TEXT NOT NULL,
            market_status   TEXT NOT NULL DEFAULT 'closed',
            close_time      TIME,
            exchange        TEXT NOT NULL DEFAULT 'NYSE',
            PRIMARY KEY (date, exchange)
        )
    """)
    current_year = datetime.date.today().year
    _seed_market_holidays(conn, current_year)
    _seed_market_holidays(conn, current_year + 1)


def _migrate_signal_history_pg(conn: PgConnection) -> None:
    """Create the signals table: authoritative per-strategy, per-symbol signal record."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_date   DATE        NOT NULL,
            strategy_id   TEXT        NOT NULL,
            symbol        TEXT        NOT NULL,
            signal_value  FLOAT       NOT NULL,
            confidence    FLOAT       NOT NULL,
            regime        TEXT,
            metadata      JSONB,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (signal_date, strategy_id, symbol)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS signals_strategy_symbol_date_idx
        ON signals (strategy_id, symbol, signal_date DESC)
    """)


def _migrate_signal_ic_pg(conn: PgConnection) -> None:
    """Create signal_ic: nightly cross-sectional IC metrics per strategy (no symbol column)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_ic (
            date              DATE    NOT NULL,
            strategy_id       TEXT    NOT NULL,
            horizon_days      INTEGER NOT NULL,
            rank_ic           FLOAT,
            ic_positive_rate  FLOAT,
            icir_21d          FLOAT,
            icir_63d          FLOAT,
            ic_tstat          FLOAT,
            n_symbols         INTEGER,
            updated_at        TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (date, strategy_id, horizon_days)
        )
    """)


def _migrate_pnl_attribution_pg(conn: PgConnection) -> None:
    """Create pnl_attribution: daily 4-component P&L decomposition per position."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pnl_attribution (
            date             DATE             NOT NULL,
            symbol           TEXT             NOT NULL,
            strategy_id      TEXT             NOT NULL,
            total_pnl        NUMERIC          NOT NULL,
            market_pnl       NUMERIC          NOT NULL DEFAULT 0,
            sector_pnl       NUMERIC          NOT NULL DEFAULT 0,
            alpha_pnl        NUMERIC          NOT NULL DEFAULT 0,
            residual_pnl     NUMERIC          NOT NULL DEFAULT 0,
            beta_market      NUMERIC,
            beta_sector      NUMERIC,
            sector_etf       TEXT,
            holding_day      INTEGER,
            PRIMARY KEY (date, symbol, strategy_id)
        )
    """)


def _migrate_regime_state_pg(conn: PgConnection) -> None:
    """Create regime_state: historical regime classifications; current = most recent row."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_state (
            detected_at      TIMESTAMPTZ PRIMARY KEY,
            regime           TEXT        NOT NULL,
            adx              FLOAT,
            vix_level        FLOAT,
            spy_20d_return   FLOAT,
            breadth_score    FLOAT,
            confidence       FLOAT,
            previous_regime  TEXT,
            regime_change    BOOLEAN     DEFAULT FALSE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS regime_state_detected_at_idx
        ON regime_state (detected_at DESC)
    """)


def _migrate_institutional_gaps_pg(conn: PgConnection) -> None:
    """Schema changes for the five institutional gap fixes.

    M1: ic_gate_grandfathered_until on strategies
    M2: ac_expected_cost_bps + forecast_error_bps on tca_results
    M3: tca_coefficients table
    M4: symbol_execution_quality table
    M5: strategy_mmc table
    M6: alt_data_ic table
    """
    # M1 — Grandfathered IC gate column
    _alter_safe(conn, "strategies", "ic_gate_grandfathered_until", "TIMESTAMPTZ")
    conn.execute("""
        UPDATE strategies
        SET ic_gate_grandfathered_until = NOW() + INTERVAL '90 days'
        WHERE status = 'live' AND ic_gate_grandfathered_until IS NULL
    """)
    logger.debug("[DB] M1: ic_gate_grandfathered_until column added")

    # M2 — TCA forecast tracking columns
    _alter_safe(conn, "tca_results", "ac_expected_cost_bps", "FLOAT")
    _alter_safe(conn, "tca_results", "forecast_error_bps", "FLOAT")
    logger.debug("[DB] M2: tca_results forecast columns added")

    # M3 — TCA coefficients table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tca_coefficients (
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol_group    TEXT NOT NULL,
            eta             FLOAT NOT NULL,
            gamma           FLOAT NOT NULL,
            beta            FLOAT NOT NULL DEFAULT 0.6,
            n_trades_in_fit INTEGER,
            r_squared       FLOAT,
            PRIMARY KEY (updated_at, symbol_group)
        )
    """)
    logger.debug("[DB] M3: tca_coefficients table created")

    # M4 — Symbol execution quality table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS symbol_execution_quality (
            symbol          TEXT NOT NULL,
            week_ending     DATE NOT NULL,
            mean_abs_error_bps FLOAT,
            quality_scalar  FLOAT,
            n_trades        INTEGER,
            PRIMARY KEY (symbol, week_ending)
        )
    """)
    logger.debug("[DB] M4: symbol_execution_quality table created")

    # M5 — Strategy MMC table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_mmc (
            date                             DATE NOT NULL,
            strategy_id                      TEXT NOT NULL,
            mmc_score                        FLOAT,
            signal_correlation_to_portfolio  FLOAT,
            capital_weight_scalar            FLOAT,
            n_days_in_window                 INTEGER,
            computed_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (date, strategy_id)
        )
    """)
    logger.debug("[DB] M5: strategy_mmc table created")

    # M6 — Alt data IC table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alt_data_ic (
            date            DATE NOT NULL,
            signal_source   TEXT NOT NULL,
            symbol          TEXT NOT NULL DEFAULT '',
            rank_ic         FLOAT,
            icir_21d        FLOAT,
            n_observations  INTEGER,
            PRIMARY KEY (date, signal_source, symbol)
        )
    """)
    logger.debug("[DB] M6: alt_data_ic table created")

    logger.info("[DB] Institutional gap migrations complete")


def _migrate_ewf_pg(conn: PgConnection) -> None:
    """Create ewf_chart_analyses table for EWF Elliott Wave signal integration."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ewf_chart_analyses (
            id                         BIGSERIAL PRIMARY KEY,
            symbol                     VARCHAR(10)  NOT NULL,
            timeframe                  VARCHAR(20)  NOT NULL,
            fetched_at                 TIMESTAMPTZ  NOT NULL,
            analyzed_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            image_path                 TEXT,
            bias                       VARCHAR(10),
            wave_position              TEXT,
            wave_degree                VARCHAR(20),
            current_wave_label         VARCHAR(10),
            key_levels                 JSONB,
            blue_box_active            BOOLEAN      NOT NULL DEFAULT FALSE,
            blue_box_zone              JSONB,
            confidence                 FLOAT,
            invalidation_rule_violated BOOLEAN      NOT NULL DEFAULT FALSE,
            analyst_notes              TEXT,
            summary                    TEXT,
            raw_analysis               TEXT,
            model_used                 VARCHAR(50),
            CONSTRAINT ewf_chart_analyses_unique UNIQUE (symbol, timeframe, fetched_at)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ewf_symbol_timeframe_fetched
        ON ewf_chart_analyses (symbol, timeframe, fetched_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ewf_blue_box_active
        ON ewf_chart_analyses (blue_box_active, fetched_at DESC)
        WHERE blue_box_active = TRUE
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ewf_symbol_fetched
        ON ewf_chart_analyses (symbol, fetched_at DESC)
    """)
    # v2: add columns for improved prompt extraction (turning_signal, reasoning, etc.)
    for col_sql in [
        "ALTER TABLE ewf_chart_analyses ADD COLUMN IF NOT EXISTS turning_signal VARCHAR(20)",
        "ALTER TABLE ewf_chart_analyses ADD COLUMN IF NOT EXISTS completed_wave_sequence TEXT",
        "ALTER TABLE ewf_chart_analyses ADD COLUMN IF NOT EXISTS projected_path TEXT",
        "ALTER TABLE ewf_chart_analyses ADD COLUMN IF NOT EXISTS reasoning TEXT",
    ]:
        conn.execute(col_sql)
    logger.debug("[DB] ewf_chart_analyses table migrated")


def run_migrations(conn: PgConnection) -> None:
    """Run all migrations.  Called once at startup.  Idempotent."""
    run_migrations_pg(conn)


# ---------------------------------------------------------------------------
# Compat aliases (old names used by tests)
# ---------------------------------------------------------------------------

_migrate_coordination = _migrate_coordination_pg
_migrate_screener = _migrate_screener_pg
_migrate_universe = _migrate_universe_pg
