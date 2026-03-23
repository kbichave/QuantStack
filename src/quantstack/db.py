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

DuckDB allows only ONE connection per file across OS processes.  Even
read_only=True fails when another process holds a write connection (DuckDB
1.5.0 enforces exclusive file locks).

To avoid blocking every other process for the MCP server's lifetime, we use
SHORT-LIVED connections: open → operate → close.  The write lock is held for
milliseconds per operation instead of forever.

### open_db()

Returns a fresh connection each call (no caching).  Caller MUST close it
when done — preferably via the ``db_conn()`` context manager::

    with db_conn() as conn:
        conn.execute("INSERT INTO ...")

For in-memory databases (tests), a cached singleton is used so all services
share the same in-memory DB within a test.

### Migrations

``run_migrations()`` is idempotent (CREATE IF NOT EXISTS).  The MCP server
calls it once at startup.  Subsequent ``open_db()`` calls skip migrations.

Usage:
    # Production — short-lived connections
    from quantstack.db import db_conn
    with db_conn() as conn:
        conn.execute("INSERT INTO ...")

    # MCP server startup — run migrations once
    from quantstack.db import open_db, run_migrations
    conn = open_db()
    run_migrations(conn)
    conn.close()

    # Legacy / compat — open_db still works but caller must close
    conn = open_db()
    # ... use conn ...
    conn.close()

    # Tests — fully isolated in-memory DB (cached singleton)
    conn = open_db(":memory:")
    run_migrations(conn)
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from threading import Lock, RLock

import duckdb
from loguru import logger

from quantstack.shared.duckdb_lock import (
    connect_with_lock_guard as _connect_with_lock_guard,
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_path(path: str = "") -> str:
    """Resolve the DB file path."""
    if path == ":memory:":
        return path
    if not path:
        path = os.getenv("TRADER_DB_PATH", "~/.quant_pod/trader.duckdb")
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


# ---------------------------------------------------------------------------
# ManagedConnection — drop-in proxy with auto-reconnect
# ---------------------------------------------------------------------------


class ManagedConnection:
    """Drop-in replacement for ``duckdb.DuckDBPyConnection``.

    Opens a real DuckDB connection on first ``execute()`` and holds it until
    ``release()`` is called.  After ``release()``, the next ``execute()``
    transparently reconnects.

    This means the file lock is held only while operations are in flight.
    Between MCP tool calls the lock can be released so other processes
    (scripts, autonomous runner) can access the DB.

    For ``:memory:`` paths, wraps a persistent singleton (no release).
    """

    def __init__(self, path: str):
        self._path = path
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._lock = RLock()
        self._is_memory = path == ":memory:"

    def _ensure_open(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            if self._is_memory:
                self._conn = duckdb.connect(":memory:")
            else:
                self._conn = _connect_with_lock_guard(self._path)
                logger.debug(f"[DB] Connection opened → {self._path}")
        return self._conn

    # -- DuckDB API surface used by services --------------------------------

    def execute(self, query: str, params: object = None) -> ManagedConnection:
        with self._lock:
            conn = self._ensure_open()
            if params is not None:
                conn.execute(query, params)
            else:
                conn.execute(query)
        return self  # support chaining: conn.execute(...).fetchone()

    def executemany(self, query: str, params: object = None) -> ManagedConnection:
        with self._lock:
            conn = self._ensure_open()
            conn.executemany(query, params)
        return self

    def fetchone(self):
        with self._lock:
            return self._ensure_open().fetchone()

    def fetchall(self):
        with self._lock:
            return self._ensure_open().fetchall()

    def fetchdf(self):
        with self._lock:
            return self._ensure_open().fetchdf()

    def fetchnumpy(self):
        with self._lock:
            return self._ensure_open().fetchnumpy()

    def fetch_arrow_table(self):
        with self._lock:
            return self._ensure_open().fetch_arrow_table()

    def description(self):
        with self._lock:
            return self._ensure_open().description

    # -- Lifecycle ----------------------------------------------------------

    def release(self) -> None:
        """Close the underlying connection, releasing the file lock.

        The next ``execute()`` call will transparently reconnect.
        Safe to call multiple times.  No-op for ``:memory:``.
        """
        if self._is_memory:
            return
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                logger.debug("[DB] Connection released")

    def close(self) -> None:
        """Alias for ``release()``."""
        self.release()

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    def __repr__(self) -> str:
        state = "open" if self._conn else "released"
        return f"ManagedConnection({self._path!r}, {state})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_managed: ManagedConnection | None = None
_managed_lock = Lock()


def open_db(path: str = "") -> ManagedConnection:
    """
    Return the module-level ``ManagedConnection``.

    All callers in the same process share one proxy.  The proxy auto-opens
    on first ``execute()`` and can be ``release()``d between operations so
    other processes can access the DB file.

    For ``:memory:`` (tests): each call to ``open_db(":memory:")`` with a
    fresh ``reset_connection()`` creates a new in-memory DB.
    """
    global _managed
    resolved = _resolve_path(path)

    with _managed_lock:
        if _managed is None or _managed._path != resolved:
            _managed = ManagedConnection(resolved)
        return _managed


@contextlib.contextmanager
def db_conn(path: str = ""):
    """Context manager: opens a connection, yields it, releases on exit.

    ::

        with db_conn() as conn:
            conn.execute("INSERT INTO strategies ...")
        # lock released here — other processes can connect
    """
    conn = open_db(path)
    try:
        yield conn
    finally:
        conn.release()


def reset_connection() -> None:
    """Close and discard the singleton.  Used by tests."""
    global _managed
    with _managed_lock:
        if _managed is not None:
            _managed.release()
            _managed = None


# ---------------------------------------------------------------------------
# Read-only compat (identical to open_db — DuckDB 1.5 has no concurrent RO)
# ---------------------------------------------------------------------------


def open_db_readonly(path: str = "") -> ManagedConnection:
    """Same as ``open_db()`` — DuckDB 1.5 doesn't support concurrent connections."""
    return open_db(path)


def reset_connection_readonly() -> None:
    """Legacy compat."""
    reset_connection()


# ---------------------------------------------------------------------------
# Migrations — idempotent, append-only schema upgrades
# ---------------------------------------------------------------------------


def run_migrations(conn: ManagedConnection | duckdb.DuckDBPyConnection) -> None:
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
        _migrate_strategy_outcomes(conn)
        _migrate_universe(conn)
        _migrate_screener(conn)
        _migrate_coordination(conn)
        _migrate_conversations(conn)
        _migrate_attribution(conn)
        _migrate_research(conn)
        _migrate_reflection(conn)
        _migrate_optimization_v2(conn)
        conn.execute("COMMIT")
        logger.info("[DB] Migrations complete")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _alter_safe(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    col_type: str,
) -> None:
    """Add a column if it doesn't already exist. Swallows 'already exists' errors."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except duckdb.CatalogException:
        pass  # column already exists


def _migrate_portfolio(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
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
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_balance (
            id          INTEGER PRIMARY KEY DEFAULT 1,
            cash        DOUBLE NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
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
    """
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS closed_trades_seq START 1")

    # v2 — position metadata for autonomous trading loop (strategy context + exit levels)
    _alter_safe(conn, "positions", "strategy_id", "VARCHAR DEFAULT ''")
    _alter_safe(conn, "positions", "regime_at_entry", "VARCHAR DEFAULT 'unknown'")
    _alter_safe(conn, "positions", "instrument_type", "VARCHAR DEFAULT 'equity'")
    _alter_safe(conn, "positions", "time_horizon", "VARCHAR DEFAULT 'swing'")
    _alter_safe(conn, "positions", "stop_price", "DOUBLE")
    _alter_safe(conn, "positions", "target_price", "DOUBLE")
    _alter_safe(conn, "positions", "trailing_stop", "DOUBLE")
    _alter_safe(conn, "positions", "entry_atr", "DOUBLE")
    _alter_safe(conn, "positions", "option_expiry", "VARCHAR")
    _alter_safe(conn, "positions", "option_strike", "DOUBLE")
    _alter_safe(conn, "positions", "option_type", "VARCHAR")

    # v2 — closed trade attribution + exit reasoning
    _alter_safe(conn, "closed_trades", "strategy_id", "VARCHAR DEFAULT ''")
    _alter_safe(conn, "closed_trades", "regime_at_entry", "VARCHAR DEFAULT 'unknown'")
    _alter_safe(conn, "closed_trades", "regime_at_exit", "VARCHAR DEFAULT 'unknown'")
    _alter_safe(conn, "closed_trades", "exit_reason", "VARCHAR DEFAULT ''")
    _alter_safe(conn, "closed_trades", "instrument_type", "VARCHAR DEFAULT 'equity'")


def _migrate_broker(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
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
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS fills_symbol_idx ON fills (symbol, filled_at)
    """
    )


def _migrate_audit(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
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
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS decisions_session_idx
        ON decision_events (session_id)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS decisions_symbol_idx
        ON decision_events (symbol, created_at)
    """
    )


def _migrate_learning(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_skills (
            agent_id            VARCHAR PRIMARY KEY,
            prediction_count    INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            signal_count        INTEGER DEFAULT 0,
            winning_signals     INTEGER DEFAULT 0,
            total_signal_pnl    DOUBLE DEFAULT 0.0,
            last_updated        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
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
    """
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS calibration_seq START 1")


def _migrate_memory(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Replaces the markdown blackboard file.

    Structured storage allows indexed queries by symbol/agent/session,
    avoids full-file reads, and prevents prompt injection through
    freeform text concatenation.
    """
    conn.execute(
        """
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
    """
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_memory_seq START 1")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS memory_symbol_idx
        ON agent_memory (symbol, created_at DESC)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS memory_session_idx
        ON agent_memory (session_id, created_at DESC)
    """
    )


def _migrate_signals(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Signal state shared between the minute analyst and tick executor.

    The tick executor reads ONLY from in-memory SignalCache; this table
    is the persistence layer for crash recovery.
    """
    conn.execute(
        """
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
    """
    )


def _migrate_system(conn: duckdb.DuckDBPyConnection) -> None:
    """
    System-level flags (kill switch, daily halt) in the DB.

    These complement the sentinel files — the DB is the authoritative source,
    sentinel files are the fast cross-process signal.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_state (
            key         VARCHAR PRIMARY KEY,
            value       VARCHAR NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )


def _migrate_strategies(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Strategy registry — Claude Code's persistent strategy catalog.

    Every strategy hypothesis (manual, decoded, or generated) is registered
    here with its rules, parameters, backtest summary, and lifecycle status.
    """
    conn.execute(
        """
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
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS strategies_status_idx
        ON strategies (status)
    """
    )
    # v0.5.0 additive columns — instrument type and time horizon classification
    conn.execute(
        """
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS instrument_type VARCHAR DEFAULT 'equity'
    """
    )
    conn.execute(
        """
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS time_horizon VARCHAR DEFAULT 'swing'
    """
    )
    conn.execute(
        """
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS holding_period_days INTEGER DEFAULT 5
    """
    )


def _migrate_regime_matrix(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Regime-strategy allocation matrix.

    Maps market regimes to strategies with capital allocation weights.
    Updated by /reflect sessions as performance data accumulates.
    Seeded with the initial matrix from CLAUDE.md section 6.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regime_strategy_matrix (
            regime          VARCHAR NOT NULL,
            strategy_id     VARCHAR NOT NULL,
            allocation_pct  DOUBLE NOT NULL,
            confidence      DOUBLE DEFAULT 0.5,
            last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (regime, strategy_id)
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS rsm_regime_idx
        ON regime_strategy_matrix (regime)
    """
    )


def _migrate_strategy_outcomes(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Strategy outcome attribution — the learning loop's ground truth.

    Written at entry (buy fill) and closed at exit (sell fill) by execute_trade.
    OutcomeTracker reads this table to update regime_affinity weights.

    Why a separate table (not a column on closed_trades):
      closed_trades is owned by PortfolioState, which has no concept of strategies.
      This table belongs to the learning layer, injected by the MCP execution path.
      Decoupled by design — a closed_trades failure never corrupts outcome data.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_outcomes (
            id               INTEGER PRIMARY KEY,
            strategy_id      VARCHAR NOT NULL,
            symbol           VARCHAR NOT NULL,
            regime_at_entry  VARCHAR NOT NULL DEFAULT 'unknown',
            action           VARCHAR NOT NULL,
            entry_price      DOUBLE NOT NULL,
            exit_price       DOUBLE,
            realized_pnl_pct DOUBLE,
            outcome          VARCHAR,
            opened_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at        TIMESTAMP,
            session_id       VARCHAR DEFAULT ''
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS so_strategy_idx
        ON strategy_outcomes (strategy_id)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS so_open_idx
        ON strategy_outcomes (strategy_id, symbol, closed_at)
    """
    )


def _migrate_universe(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Symbol universe — SP500, NASDAQ-100, and liquid ETF constituents.

    Populated by UniverseRegistry.refresh_constituents() (weekly).
    Read by AutonomousScreener to score and rank tradeable symbols.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS universe (
            symbol              VARCHAR PRIMARY KEY,
            name                VARCHAR NOT NULL,
            sector              VARCHAR DEFAULT 'Unknown',
            source              VARCHAR NOT NULL,
            market_cap          DOUBLE,
            avg_daily_volume    DOUBLE,
            is_active           BOOLEAN DEFAULT TRUE,
            added_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_refreshed      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deactivated_reason  VARCHAR
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS universe_source_idx
        ON universe (source, is_active)
    """
    )


def _migrate_screener(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Screener results — daily scored and tiered watchlist.

    Written by AutonomousScreener.screen() each morning.
    Read by WatchlistLoader.load_tiered() to feed the runner.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS screener_results (
            symbol              VARCHAR NOT NULL,
            screened_at         TIMESTAMP NOT NULL,
            regime_used         VARCHAR,
            tier                INTEGER NOT NULL,
            composite_score     DOUBLE NOT NULL,
            momentum_score      DOUBLE,
            volatility_rank     DOUBLE,
            volume_surge        DOUBLE,
            regime_fit          DOUBLE,
            catalyst_proximity  DOUBLE,
            PRIMARY KEY (symbol, screened_at)
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS screener_latest_idx
        ON screener_results (screened_at DESC, tier)
    """
    )


def _migrate_coordination(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Inter-loop coordination tables — event bus, cursors, and heartbeats.

    The event bus is an append-only log that loops poll at iteration start.
    Cursors track each consumer's high-water mark.
    Heartbeats track loop health for the supervisor.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_events (
            event_id    VARCHAR PRIMARY KEY,
            event_type  VARCHAR NOT NULL,
            source_loop VARCHAR NOT NULL,
            payload     JSON,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS loop_events_type_idx
        ON loop_events (event_type, created_at)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS loop_events_created_idx
        ON loop_events (created_at)
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_cursors (
            consumer_id     VARCHAR PRIMARY KEY,
            last_event_id   VARCHAR,
            last_polled_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_heartbeats (
            loop_name           VARCHAR NOT NULL,
            iteration           INTEGER NOT NULL,
            started_at          TIMESTAMP NOT NULL,
            finished_at         TIMESTAMP,
            symbols_processed   INTEGER DEFAULT 0,
            errors              INTEGER DEFAULT 0,
            status              VARCHAR DEFAULT 'running',
            PRIMARY KEY (loop_name, iteration)
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS heartbeats_loop_idx
        ON loop_heartbeats (loop_name, started_at DESC)
    """
    )


def _migrate_conversations(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Agent conversation log and signal snapshots.

    agent_conversations: Full desk agent reports with reasoning, persisted for
    Slack posting, debugging, and prompt optimization during /reflect sessions.

    signal_snapshots: Raw SignalEngine collector outputs per symbol, enabling
    analysis of which collectors drove which decisions.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_conversations (
            conversation_id  VARCHAR PRIMARY KEY,
            session_id       VARCHAR NOT NULL,
            loop_name        VARCHAR DEFAULT 'trading_operator',
            iteration        INTEGER,
            agent_name       VARCHAR NOT NULL,
            role             VARCHAR NOT NULL,
            symbol           VARCHAR,
            strategy_id      VARCHAR,
            content          TEXT NOT NULL,
            summary          VARCHAR,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata         JSON
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS conv_agent_idx
        ON agent_conversations (agent_name, created_at DESC)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS conv_symbol_idx
        ON agent_conversations (symbol, created_at DESC)
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_snapshots (
            snapshot_id          VARCHAR PRIMARY KEY,
            symbol               VARCHAR NOT NULL,
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            technical            JSON,
            regime               JSON,
            volume               JSON,
            risk                 JSON,
            sentiment            JSON,
            fundamentals         JSON,
            events               JSON,
            consensus_bias       VARCHAR,
            consensus_conviction DOUBLE,
            collector_failures   JSON
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS snap_symbol_idx
        ON signal_snapshots (symbol, created_at DESC)
    """
    )


def _migrate_attribution(conn: duckdb.DuckDBPyConnection) -> None:
    """Phase A: P&L attribution tables — daily equity curve + strategy attribution."""

    # A.1: Daily equity curve — immutable daily snapshots for track record.
    # Source of truth for NAV history. INSERT only, never UPDATE.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_equity (
            date                DATE PRIMARY KEY,
            cash                DOUBLE NOT NULL,
            positions_value     DOUBLE NOT NULL,
            total_equity        DOUBLE NOT NULL,
            daily_pnl           DOUBLE NOT NULL,
            cumulative_pnl      DOUBLE NOT NULL,
            daily_return_pct    DOUBLE NOT NULL,
            high_water_mark     DOUBLE NOT NULL,
            drawdown_pct        DOUBLE NOT NULL,
            open_positions      INTEGER NOT NULL,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # A.2: Per-strategy daily P&L rollup.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_daily_pnl (
            date                DATE NOT NULL,
            strategy_id         VARCHAR NOT NULL,
            realized_pnl        DOUBLE DEFAULT 0,
            unrealized_pnl      DOUBLE DEFAULT 0,
            num_trades          INTEGER DEFAULT 0,
            win_count           INTEGER DEFAULT 0,
            loss_count          INTEGER DEFAULT 0,
            PRIMARY KEY (date, strategy_id)
        )
    """
    )

    # A.2: Add strategy_id and regime_at_entry to closed_trades if missing.
    # ALTER TABLE ADD COLUMN is idempotent in DuckDB (errors if column exists,
    # so we catch and ignore).
    for col, typedef in [
        ("strategy_id", "VARCHAR DEFAULT ''"),
        ("regime_at_entry", "VARCHAR DEFAULT 'unknown'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE closed_trades ADD COLUMN {col} {typedef}")
            logger.info(f"[DB] Added {col} to closed_trades")
        except duckdb.CatalogException:
            pass  # Column already exists — expected on subsequent runs

    # A.3: Benchmark comparison table (daily benchmark returns + rolling metrics).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_daily (
            date                DATE NOT NULL,
            benchmark           VARCHAR NOT NULL,
            close_price         DOUBLE NOT NULL,
            daily_return_pct    DOUBLE,
            cumulative_return   DOUBLE,
            PRIMARY KEY (date, benchmark)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_comparison (
            date                DATE NOT NULL,
            benchmark           VARCHAR NOT NULL,
            window_days         INTEGER NOT NULL,
            portfolio_sharpe    DOUBLE,
            benchmark_sharpe    DOUBLE,
            portfolio_sortino   DOUBLE,
            alpha               DOUBLE,
            beta                DOUBLE,
            PRIMARY KEY (date, benchmark, window_days)
        )
    """
    )


def _migrate_research(conn: duckdb.DuckDBPyConnection) -> None:
    """Research pod tables — experiment tracking, research programs, pod state."""

    # ML experiment log — every training run, HPO trial, or model evaluation
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_experiments (
            experiment_id       VARCHAR PRIMARY KEY,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol              VARCHAR NOT NULL,
            model_type          VARCHAR NOT NULL,
            feature_tiers       JSON,
            label_method        VARCHAR DEFAULT 'event',
            n_features_raw      INTEGER,
            n_features_filtered INTEGER,
            test_auc            DOUBLE,
            test_accuracy       DOUBLE,
            cv_auc_mean         DOUBLE,
            cv_auc_std          DOUBLE,
            top_features        JSON,
            causal_dropped      JSON,
            hyperparams         JSON,
            training_duration_s DOUBLE,
            hypothesis_id       VARCHAR,
            verdict             VARCHAR DEFAULT 'pending',
            failure_analysis    TEXT,
            notes               TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS mlexp_symbol_idx ON ml_experiments (symbol, created_at DESC)
    """
    )

    # Alpha research program — persistent multi-week research agenda
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alpha_research_program (
            investigation_id    VARCHAR PRIMARY KEY,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            thesis              TEXT NOT NULL,
            status              VARCHAR DEFAULT 'active',
            priority            INTEGER DEFAULT 5,
            source              VARCHAR,
            experiments_run     INTEGER DEFAULT 0,
            best_oos_sharpe     DOUBLE,
            last_result_summary TEXT,
            next_steps          TEXT,
            dead_end_reason     TEXT,
            target_regimes      JSON,
            target_symbols      JSON
        )
    """
    )

    # ML research program — model-level research state
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_research_program (
            program_id          VARCHAR PRIMARY KEY,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            focus_area          VARCHAR NOT NULL,
            status              VARCHAR DEFAULT 'active',
            hypothesis          TEXT,
            experiments_run     INTEGER DEFAULT 0,
            best_metric         DOUBLE,
            best_config         JSON,
            lessons_learned     TEXT,
            next_experiment     TEXT
        )
    """
    )

    # Research plans — structured output from each pod run
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_plans (
            plan_id             VARCHAR PRIMARY KEY,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pod_name            VARCHAR NOT NULL,
            plan_type           VARCHAR NOT NULL,
            plan_json           JSON NOT NULL,
            context_summary     TEXT,
            executed            BOOLEAN DEFAULT FALSE,
            execution_results   JSON
        )
    """
    )

    # Breakthrough features — features that appear in 3+ winning strategies
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS breakthrough_features (
            feature_name        VARCHAR PRIMARY KEY,
            first_seen          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            occurrence_count    INTEGER DEFAULT 1,
            avg_shap_importance DOUBLE,
            winning_strategies  JSON,
            regimes_effective   JSON
        )
    """
    )


def _migrate_reflection(conn) -> None:
    """Trade reflections — automatic post-trade analysis with SQL-based retrieval."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_reflections (
            id              INTEGER PRIMARY KEY,
            symbol          VARCHAR NOT NULL,
            strategy_id     VARCHAR,
            action          VARCHAR,
            entry_price     DOUBLE,
            exit_price      DOUBLE,
            realized_pnl_pct DOUBLE,
            holding_days    INTEGER,
            regime_at_entry VARCHAR,
            regime_at_exit  VARCHAR,
            conviction      DOUBLE,
            signals_entry   VARCHAR,
            signals_exit    VARCHAR,
            lesson          VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_reflections_regime_symbol "
        "ON trade_reflections (regime_at_entry, symbol)"
    )


def _migrate_optimization_v2(conn) -> None:
    """Prompt optimization & per-trade reward learning tables.

    Papers: Reflexion (NeurIPS 2023), TextGrad (Nature 2024), OPRO (DeepMind 2023),
    QuantAgent (2024), QuantaAlpha (2026), AgentPRM (2025).
    """
    # Phase 1: Reflexion episodic memory
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reflexion_episodes (
            episode_id          VARCHAR PRIMARY KEY,
            trade_id            INTEGER,
            regime              VARCHAR NOT NULL,
            strategy_id         VARCHAR NOT NULL,
            symbol              VARCHAR NOT NULL,
            pnl_pct             DOUBLE NOT NULL,
            root_cause          VARCHAR NOT NULL,
            verbal_reinforcement TEXT NOT NULL,
            counterfactual      TEXT,
            tags                JSON,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS reflex_regime_strat_idx "
        "ON reflexion_episodes (regime, strategy_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS reflex_symbol_idx "
        "ON reflexion_episodes (symbol, created_at DESC)"
    )

    # Phase 2: Step-level credit assignment (AgentPRM)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS step_credits_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS step_credits (
            id                  INTEGER PRIMARY KEY DEFAULT nextval('step_credits_seq'),
            trade_id            INTEGER NOT NULL,
            step_type           VARCHAR NOT NULL,
            step_output         TEXT,
            credit_score        DOUBLE NOT NULL,
            attribution_method  VARCHAR NOT NULL,
            evidence            TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS sc_trade_idx ON step_credits (trade_id)"
    )

    # Phase 3: TextGrad prompt versions and critiques
    conn.execute("CREATE SEQUENCE IF NOT EXISTS prompt_versions_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id                  INTEGER PRIMARY KEY DEFAULT nextval('prompt_versions_seq'),
            node_name           VARCHAR NOT NULL,
            version             INTEGER NOT NULL,
            prompt_text         TEXT NOT NULL,
            source              VARCHAR NOT NULL,
            parent_version      INTEGER,
            status              VARCHAR DEFAULT 'proposed',
            avg_pnl_since       DOUBLE,
            trades_since        INTEGER DEFAULT 0,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (node_name, version)
        )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS prompt_critiques_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_critiques (
            id                  INTEGER PRIMARY KEY DEFAULT nextval('prompt_critiques_seq'),
            trade_id            INTEGER NOT NULL,
            node_name           VARCHAR NOT NULL,
            critique            TEXT NOT NULL,
            proposed_edit       TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Phase 4: Hypothesis judge verdicts (QuantAgent)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS judge_verdicts (
            verdict_id          VARCHAR PRIMARY KEY,
            hypothesis_id       VARCHAR NOT NULL,
            approved            BOOLEAN NOT NULL,
            score               DOUBLE NOT NULL,
            flags               JSON,
            reasoning           TEXT,
            similar_failures    JSON,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Phase 5: OPRO prompt candidates
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_candidates (
            candidate_id        VARCHAR PRIMARY KEY,
            node_name           VARCHAR NOT NULL,
            prompt_text         TEXT NOT NULL,
            generation          INTEGER NOT NULL,
            fitness             DOUBLE,
            trades_evaluated    INTEGER DEFAULT 0,
            status              VARCHAR DEFAULT 'candidate',
            source              VARCHAR DEFAULT 'opro',
            parent_ids          JSON,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            promoted_at         TIMESTAMP,
            retired_at          TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pc_node_status_idx "
        "ON prompt_candidates (node_name, status)"
    )

    # Phase 6: QuantaAlpha research trajectories
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_trajectories (
            trajectory_id       VARCHAR PRIMARY KEY,
            generation          INTEGER NOT NULL,
            segments            JSON NOT NULL,
            overall_fitness     DOUBLE,
            parent_ids          JSON,
            mutation_applied    VARCHAR,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS rt_gen_idx "
        "ON research_trajectories (generation, overall_fitness DESC)"
    )

    # Options columns on positions table (added for options trading support)
    for col, typedef in [
        ("instrument_type", "VARCHAR DEFAULT 'equity'"),
        ("strike", "DOUBLE"),
        ("expiry", "VARCHAR"),
        ("option_type", "VARCHAR"),
        ("premium_at_entry", "DOUBLE"),
    ]:
        try:
            conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # Column already exists
