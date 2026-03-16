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

Usage:
    # Production
    from quant_pod.db import open_db, run_migrations
    conn = open_db("~/.quant_pod/trader.duckdb")
    run_migrations(conn)

    # Tests — fully isolated in-memory DB
    conn = open_db(":memory:")
    run_migrations(conn)
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Optional

import duckdb
from loguru import logger


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_conn: Optional[duckdb.DuckDBPyConnection] = None
_conn_lock = Lock()


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
            _conn = duckdb.connect(path)
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
