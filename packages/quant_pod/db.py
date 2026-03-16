# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Single consolidated DuckDB database for all QuantPod state.

Every service (portfolio, broker, audit, learning, memory, signals) shares
one database file so trade execution can be expressed as a single ACID
transaction.  Separate DB files made cross-service consistency impossible —
a crash between two writes would leave permanent partial state.

Schema ownership lives here.  Services receive an injected connection;
they do NOT open their own files.

## Connection Model

DuckDB allows only ONE write connection per file across OS processes.
The MCP server is the canonical write owner — it holds the connection for
its entire lifetime.  Other processes (FastAPI, scripts) must connect
read-only via open_db_readonly().

### Lock conflict handling

open_db() wraps duckdb.connect() with a lock guard (_connect_with_lock_guard):

  - Stale lock (owning process is dead): retries for up to
    _STALE_LOCK_RETRY_SECS (10 s) until the OS releases the lock.

  - Live conflict (owning process is alive): raises RuntimeError immediately
    with the PID and the exact `kill PID` command to resolve it.

The MCP server's lifespan() catches this RuntimeError and falls back to an
in-memory context (degraded mode) so the Claude session is never crashed by
a lock conflict.  Analysis tools keep working; tools that need persistent
state return {"success": False, "degraded_mode": True}.

Usage:
    # Production (MCP server — write owner)
    from quant_pod.db import open_db, run_migrations
    conn = open_db("~/.quant_pod/trader.duckdb")
    run_migrations(conn)

    # FastAPI / scripts — read-only, no lock competition
    from quant_pod.db import open_db_readonly
    conn = open_db_readonly()   # FileNotFoundError if MCP server not started

    # Tests — fully isolated in-memory DB
    conn = open_db(":memory:")
    run_migrations(conn)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from threading import Lock

import duckdb
from loguru import logger

# How long (seconds) to keep retrying after detecting a stale lock.
# A stale lock means the previous owner process is dead; the OS will release
# the file lock shortly.  Retry window must exceed typical OS cleanup latency.
_STALE_LOCK_RETRY_SECS = 10
_STALE_LOCK_POLL_INTERVAL = 0.5

# Pattern DuckDB embeds in its lock error: "... (PID 12345) ..."
_LOCK_PID_RE = re.compile(r"\(PID\s+(\d+)\)")

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_conn: duckdb.DuckDBPyConnection | None = None
_conn_lock = Lock()


def _is_process_alive(pid: int) -> bool:
    """Return True if a process with this PID is running on this machine."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False  # PID does not exist
    except PermissionError:
        return True  # PID exists but owned by another user — still alive


def _connect_with_lock_guard(path: str) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection, handling lock conflicts intelligently.

    Two cases:
      1. Stale lock (owning process is dead) — the OS will release the file
         lock shortly after process exit.  Retry for up to
         _STALE_LOCK_RETRY_SECS before giving up.
      2. Live duplicate — another server process owns the lock.  Fail
         immediately with an actionable error that includes the PID and
         the exact kill command.
    """
    deadline = time.monotonic() + _STALE_LOCK_RETRY_SECS
    last_exc: Exception | None = None

    while True:
        try:
            return duckdb.connect(path)
        except duckdb.IOException as exc:
            last_exc = exc
            msg = str(exc)
            if "Conflicting lock" not in msg:
                raise  # unrelated I/O error, don't mask it

            match = _LOCK_PID_RE.search(msg)
            if not match:
                raise  # can't determine owner, surface the original error

            owner_pid = int(match.group(1))

            if _is_process_alive(owner_pid):
                raise RuntimeError(
                    f"QuantPod DB is locked by a running process (PID {owner_pid}).\n"
                    f"  → Kill it:   kill {owner_pid}\n"
                    f"  → Or check:  ps -p {owner_pid}\n"
                    f"  DB path: {path}"
                ) from exc

            # Stale lock — process is dead.  Retry until the OS cleans up.
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Stale lock on {path} (dead PID {owner_pid}) did not clear "
                    f"after {_STALE_LOCK_RETRY_SECS}s. "
                    f"Try: rm -f '{path}.wal' and restart."
                ) from last_exc

            logger.warning(
                f"[DB] Stale lock detected (dead PID {owner_pid}), "
                f"retrying in {_STALE_LOCK_POLL_INTERVAL}s..."
            )
            time.sleep(_STALE_LOCK_POLL_INTERVAL)


def open_db(path: str = "") -> duckdb.DuckDBPyConnection:
    """
    Open (or return a cached) DuckDB connection.

    Args:
        path: File path or ":memory:" for in-memory.  Defaults to the value of
              TRADER_DB_PATH env var, then ~/.quant_pod/trader.duckdb.
    """
    global _conn

    if not path:
        path = os.getenv("TRADER_DB_PATH", "~/.quant_pod/trader.duckdb")

    if path != ":memory:":
        resolved = Path(path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        path = str(resolved)

    with _conn_lock:
        if _conn is None:
            _conn = _connect_with_lock_guard(path)
            logger.info(f"[DB] Opened consolidated database at {path}")
        return _conn


def reset_connection() -> None:
    """
    Close and clear the cached connection.

    Call this in tests between test cases to get a fresh in-memory DB.
    """
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None


# ---------------------------------------------------------------------------
# Read-only connection — for processes that must not compete for the write lock
# ---------------------------------------------------------------------------

_conn_ro: duckdb.DuckDBPyConnection | None = None
_conn_ro_lock = Lock()


def open_db_readonly(path: str = "") -> duckdb.DuckDBPyConnection:
    """
    Open (or return a cached) read-only DuckDB connection.

    Multiple processes can hold read-only connections simultaneously without
    conflicting with the write owner (MCP server).  Does NOT run migrations —
    the write owner is always responsible for schema.

    For ':memory:' paths (test compat) falls back to the regular write
    connection so test code that calls this function works identically to
    production code that calls open_db().

    Raises:
        FileNotFoundError: if the DB file does not exist.  The write owner
                           (MCP server) must be started first.
    """
    global _conn_ro

    if not path:
        path = os.getenv("TRADER_DB_PATH", "~/.quant_pod/trader.duckdb")

    if path == ":memory:":
        # Tests use :memory: — read-only has no meaning there; reuse write conn.
        return open_db(path)

    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            f"DB not found at {resolved}. "
            "The MCP server must be started (and have run migrations) before "
            "read-only consumers can connect."
        )

    path = str(resolved)
    with _conn_ro_lock:
        if _conn_ro is None:
            _conn_ro = duckdb.connect(path, read_only=True)
            logger.info(f"[DB] Opened read-only connection at {path}")
        return _conn_ro


def reset_connection_readonly() -> None:
    """
    Close and clear the cached read-only connection.

    Call this in tests after any test that opens a file-backed read-only
    connection so the cached singleton doesn't bleed across tests.
    """
    global _conn_ro
    with _conn_ro_lock:
        if _conn_ro is not None:
            try:
                _conn_ro.close()
            except Exception:
                pass
            _conn_ro = None


# ---------------------------------------------------------------------------
# Migrations — idempotent, append-only schema upgrades
# ---------------------------------------------------------------------------


def run_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create all tables if they do not exist.

    Safe to call multiple times — all statements are CREATE IF NOT EXISTS.
    Add new tables or columns here; never modify existing column definitions
    (use ALTER TABLE in a new migration block guarded by a version check).
    """
    conn.execute("BEGIN")
    try:
        _migrate_portfolio(conn)
        _migrate_broker(conn)
        _migrate_audit(conn)
        _migrate_learning(conn)
        _migrate_memory(conn)
        _migrate_signals(conn)
        _migrate_system(conn)
        _migrate_strategies(conn)
        _migrate_regime_matrix(conn)
        conn.execute("COMMIT")
        logger.info("[DB] Migrations complete")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _migrate_portfolio(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol          VARCHAR PRIMARY KEY,
            quantity        INTEGER NOT NULL,
            avg_cost        DOUBLE NOT NULL,
            side            VARCHAR DEFAULT 'long',
            opened_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            unrealized_pnl  DOUBLE DEFAULT 0.0,
            current_price   DOUBLE DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cash_balance (
            id          INTEGER PRIMARY KEY DEFAULT 1,
            cash        DOUBLE NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS closed_trades (
            id           INTEGER PRIMARY KEY,
            symbol       VARCHAR NOT NULL,
            side         VARCHAR NOT NULL,
            quantity     INTEGER NOT NULL,
            entry_price  DOUBLE NOT NULL,
            exit_price   DOUBLE NOT NULL,
            realized_pnl DOUBLE NOT NULL,
            opened_at    TIMESTAMP,
            closed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            holding_days INTEGER DEFAULT 0,
            session_id   VARCHAR DEFAULT ''
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS closed_trades_seq START 1")


def _migrate_broker(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            order_id            VARCHAR PRIMARY KEY,
            symbol              VARCHAR NOT NULL,
            side                VARCHAR NOT NULL,
            requested_quantity  INTEGER,
            filled_quantity     INTEGER,
            fill_price          DOUBLE,
            slippage_bps        DOUBLE,
            commission          DOUBLE DEFAULT 0.0,
            partial             BOOLEAN DEFAULT FALSE,
            rejected            BOOLEAN DEFAULT FALSE,
            reject_reason       VARCHAR,
            filled_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id          VARCHAR DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS fills_symbol_idx ON fills (symbol, filled_at)
    """)


def _migrate_audit(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_events (
            event_id                VARCHAR PRIMARY KEY,
            session_id              VARCHAR NOT NULL,
            event_type              VARCHAR NOT NULL,
            agent_name              VARCHAR NOT NULL,
            agent_role              VARCHAR DEFAULT '',
            symbol                  VARCHAR DEFAULT '',
            action                  VARCHAR DEFAULT '',
            confidence              DOUBLE DEFAULT 0.0,
            input_context_hash      VARCHAR DEFAULT '',
            market_data_snapshot    JSON,
            portfolio_snapshot      JSON,
            tool_calls              JSON,
            output_summary          TEXT DEFAULT '',
            output_structured       JSON,
            risk_approved           BOOLEAN,
            risk_violations         JSON,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            decision_latency_ms     INTEGER,
            parent_event_ids        JSON
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS decisions_session_idx
        ON decision_events (session_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS decisions_symbol_idx
        ON decision_events (symbol, created_at)
    """)


def _migrate_learning(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_skills (
            agent_id            VARCHAR PRIMARY KEY,
            prediction_count    INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            signal_count        INTEGER DEFAULT 0,
            winning_signals     INTEGER DEFAULT 0,
            total_signal_pnl    DOUBLE DEFAULT 0.0,
            last_updated        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_records (
            id                  INTEGER PRIMARY KEY,
            agent_name          VARCHAR NOT NULL,
            stated_confidence   DOUBLE NOT NULL,
            was_correct         BOOLEAN NOT NULL,
            symbol              VARCHAR DEFAULT '',
            action              VARCHAR DEFAULT '',
            pnl                 DOUBLE DEFAULT 0.0,
            session_id          VARCHAR DEFAULT '',
            recorded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS calibration_seq START 1")


def _migrate_memory(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Replaces the markdown blackboard file.

    Structured storage allows indexed queries by symbol/agent/session,
    avoids full-file reads, and prevents prompt injection through
    freeform text concatenation.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id          INTEGER PRIMARY KEY,
            session_id  VARCHAR NOT NULL,
            sim_date    DATE,
            agent       VARCHAR NOT NULL,
            symbol      VARCHAR DEFAULT '',
            category    VARCHAR DEFAULT 'general',
            content_json VARCHAR NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_memory_seq START 1")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS memory_symbol_idx
        ON agent_memory (symbol, created_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS memory_session_idx
        ON agent_memory (session_id, created_at DESC)
    """)


def _migrate_signals(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Signal state shared between the minute analyst and tick executor.

    The tick executor reads ONLY from in-memory SignalCache; this table
    is the persistence layer for crash recovery.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_state (
            symbol              VARCHAR PRIMARY KEY,
            action              VARCHAR NOT NULL,
            confidence          DOUBLE NOT NULL,
            position_size_pct   DOUBLE NOT NULL,
            stop_loss           DOUBLE,
            take_profit         DOUBLE,
            generated_at        TIMESTAMP NOT NULL,
            expires_at          TIMESTAMP NOT NULL,
            session_id          VARCHAR NOT NULL
        )
    """)


def _migrate_system(conn: duckdb.DuckDBPyConnection) -> None:
    """
    System-level flags (kill switch, daily halt) in the DB.

    These complement the sentinel files — the DB is the authoritative source,
    sentinel files are the fast cross-process signal.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_state (
            key         VARCHAR PRIMARY KEY,
            value       VARCHAR NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _migrate_strategies(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Strategy registry — Claude Code's persistent strategy catalog.

    Every strategy hypothesis (manual, decoded, or generated) is registered
    here with its rules, parameters, backtest summary, and lifecycle status.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id         VARCHAR PRIMARY KEY,
            name                VARCHAR NOT NULL UNIQUE,
            description         TEXT DEFAULT '',
            asset_class         VARCHAR DEFAULT 'equities',
            regime_affinity     JSON,
            parameters          JSON NOT NULL,
            entry_rules         JSON NOT NULL,
            exit_rules          JSON NOT NULL,
            risk_params         JSON,
            backtest_summary    JSON,
            walkforward_summary JSON,
            status              VARCHAR DEFAULT 'draft',
            source              VARCHAR DEFAULT 'manual',
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by          VARCHAR DEFAULT 'claude_code'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS strategies_status_idx
        ON strategies (status)
    """)


def _migrate_regime_matrix(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Regime-strategy allocation matrix.

    Maps market regimes to strategies with capital allocation weights.
    Updated by /reflect sessions as performance data accumulates.
    Seeded with the initial matrix from CLAUDE.md section 6.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_strategy_matrix (
            regime          VARCHAR NOT NULL,
            strategy_id     VARCHAR NOT NULL,
            allocation_pct  DOUBLE NOT NULL,
            confidence      DOUBLE DEFAULT 0.5,
            last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (regime, strategy_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS rsm_regime_idx
        ON regime_strategy_matrix (regime)
    """)
