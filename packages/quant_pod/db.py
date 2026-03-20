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
from pathlib import Path
from threading import Lock

import duckdb
from loguru import logger

from shared.duckdb_lock import connect_with_lock_guard as _connect_with_lock_guard

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_conn: duckdb.DuckDBPyConnection | None = None
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
        _migrate_strategy_outcomes(conn)
        _migrate_universe(conn)
        _migrate_screener(conn)
        _migrate_coordination(conn)
        _migrate_conversations(conn)
        _migrate_attribution(conn)
        _migrate_research(conn)
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
    # v0.5.0 additive columns — instrument type and time horizon classification
    conn.execute("""
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS instrument_type VARCHAR DEFAULT 'equity'
    """)
    conn.execute("""
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS time_horizon VARCHAR DEFAULT 'swing'
    """)
    conn.execute("""
        ALTER TABLE strategies
        ADD COLUMN IF NOT EXISTS holding_period_days INTEGER DEFAULT 5
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
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS so_strategy_idx
        ON strategy_outcomes (strategy_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS so_open_idx
        ON strategy_outcomes (strategy_id, symbol, closed_at)
    """)


def _migrate_universe(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Symbol universe — SP500, NASDAQ-100, and liquid ETF constituents.

    Populated by UniverseRegistry.refresh_constituents() (weekly).
    Read by AutonomousScreener to score and rank tradeable symbols.
    """
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS universe_source_idx
        ON universe (source, is_active)
    """)


def _migrate_screener(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Screener results — daily scored and tiered watchlist.

    Written by AutonomousScreener.screen() each morning.
    Read by WatchlistLoader.load_tiered() to feed the runner.
    """
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS screener_latest_idx
        ON screener_results (screened_at DESC, tier)
    """)


def _migrate_coordination(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Inter-loop coordination tables — event bus, cursors, and heartbeats.

    The event bus is an append-only log that loops poll at iteration start.
    Cursors track each consumer's high-water mark.
    Heartbeats track loop health for the supervisor.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loop_events (
            event_id    VARCHAR PRIMARY KEY,
            event_type  VARCHAR NOT NULL,
            source_loop VARCHAR NOT NULL,
            payload     JSON,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS loop_events_type_idx
        ON loop_events (event_type, created_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS loop_events_created_idx
        ON loop_events (created_at)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS loop_cursors (
            consumer_id     VARCHAR PRIMARY KEY,
            last_event_id   VARCHAR,
            last_polled_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS heartbeats_loop_idx
        ON loop_heartbeats (loop_name, started_at DESC)
    """)


def _migrate_conversations(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Agent conversation log and signal snapshots.

    agent_conversations: Full desk agent reports with reasoning, persisted for
    Slack posting, debugging, and prompt optimization during /reflect sessions.

    signal_snapshots: Raw SignalEngine collector outputs per symbol, enabling
    analysis of which collectors drove which decisions.
    """
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS conv_agent_idx
        ON agent_conversations (agent_name, created_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS conv_symbol_idx
        ON agent_conversations (symbol, created_at DESC)
    """)

    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS snap_symbol_idx
        ON signal_snapshots (symbol, created_at DESC)
    """)


def _migrate_attribution(conn: duckdb.DuckDBPyConnection) -> None:
    """Phase A: P&L attribution tables — daily equity curve + strategy attribution."""

    # A.1: Daily equity curve — immutable daily snapshots for track record.
    # Source of truth for NAV history. INSERT only, never UPDATE.
    conn.execute("""
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
    """)

    # A.2: Per-strategy daily P&L rollup.
    conn.execute("""
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
    """)

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_daily (
            date                DATE NOT NULL,
            benchmark           VARCHAR NOT NULL,
            close_price         DOUBLE NOT NULL,
            daily_return_pct    DOUBLE,
            cumulative_return   DOUBLE,
            PRIMARY KEY (date, benchmark)
        )
    """)
    conn.execute("""
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
    """)


def _migrate_research(conn: duckdb.DuckDBPyConnection) -> None:
    """Research pod tables — experiment tracking, research programs, pod state."""

    # ML experiment log — every training run, HPO trial, or model evaluation
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS mlexp_symbol_idx ON ml_experiments (symbol, created_at DESC)
    """)

    # Alpha research program — persistent multi-week research agenda
    conn.execute("""
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
    """)

    # ML research program — model-level research state
    conn.execute("""
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
    """)

    # Research plans — structured output from each pod run
    conn.execute("""
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
    """)

    # Breakthrough features — features that appear in 3+ winning strategies
    conn.execute("""
        CREATE TABLE IF NOT EXISTS breakthrough_features (
            feature_name        VARCHAR PRIMARY KEY,
            first_seen          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            occurrence_count    INTEGER DEFAULT 1,
            avg_shap_importance DOUBLE,
            winning_strategies  JSON,
            regimes_effective   JSON
        )
    """)
