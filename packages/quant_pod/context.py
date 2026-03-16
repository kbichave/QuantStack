# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
TradingContext — single dependency-injection container for the trading system.

All services share one DuckDB connection and are wired together here.
The only place a connection is opened; every downstream service receives it
as a constructor argument.

Why this matters:
  - Tests call create_trading_context(":memory:") and get a fully-wired,
    completely isolated system with zero file-system side-effects.
  - Production calls create_trading_context() and gets the persistent DB.
  - No module-level singletons are needed. Each service is instantiated once
    per context, and the context is passed explicitly.

Usage — production:
    ctx = create_trading_context()
    flow = TradingDayFlow(
        portfolio=ctx.portfolio,
        risk_gate=ctx.risk_gate,
        audit=ctx.audit,
        signal_cache=ctx.signal_cache,
    )
    executor = TickExecutor(
        signal_cache=ctx.signal_cache,
        risk_state=ctx.risk_state,
        broker=ctx.broker,
        kill_switch=ctx.kill_switch,
        fill_queue=fill_queue,
        session_id=ctx.session_id,
    )

Usage — tests:
    @pytest.fixture
    def ctx():
        return create_trading_context(db_path=":memory:")

    def test_something(ctx):
        ctx.portfolio.upsert_position(...)
        # Each test gets a fresh in-memory DB — no shared state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import duckdb
from loguru import logger

from quant_pod.audit.decision_log import DecisionLog
from quant_pod.db import open_db, run_migrations
from quant_pod.execution.kill_switch import KillSwitch
from quant_pod.execution.paper_broker import PaperBroker
from quant_pod.execution.portfolio_state import PortfolioState
from quant_pod.execution.risk_gate import RiskGate, RiskLimits
from quant_pod.execution.risk_state import RiskState
from quant_pod.execution.signal_cache import SignalCache
from quant_pod.memory.blackboard import Blackboard


@dataclass
class TradingContext:
    """
    Fully-wired runtime container for the trading system.

    All fields are set at construction time; the object is effectively
    immutable after create_trading_context() returns.  Services share a
    single DuckDB connection to enable cross-service ACID transactions.

    Field ownership:
      db           — raw DuckDB connection; use only for migrations / ad-hoc queries.
      portfolio    — open positions and cash; the source of truth for the system.
      risk_gate    — slow-path rule engine called by TradingDayFlow.
      risk_state   — fast in-memory mirror consumed by TickExecutor hot path.
      signal_cache — TTL-gated signal store written by analyst, read by executor.
      kill_switch  — emergency halt; checked on every tick.
      broker       — PaperBroker (paper mode) or EtradeBroker (live mode).
      audit        — append-only decision log.
      blackboard   — structured agent memory (replaces markdown file).
      session_id   — UUID for this trading session; threaded through all logs.
    """

    db: duckdb.DuckDBPyConnection
    portfolio: PortfolioState
    risk_gate: RiskGate
    risk_state: RiskState
    signal_cache: SignalCache
    kill_switch: KillSwitch
    broker: PaperBroker
    audit: DecisionLog
    blackboard: Blackboard
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def create_trading_context(
    db_path: str = "",
    initial_cash: float = 100_000.0,
    risk_limits: RiskLimits | None = None,
    session_id: str | None = None,
) -> TradingContext:
    """
    Build and return a fully-wired TradingContext.

    All services receive the same DuckDB connection so they can participate
    in the same transactions.  The database schema is created (idempotently)
    before any service is initialised.

    Args:
        db_path:      File path or ":memory:".  Defaults to TRADER_DB_PATH
                      env var, then ~/.quant_pod/trader.duckdb.
        initial_cash: Starting cash balance for new portfolios.
        risk_limits:  Override default risk limits.  None = load from env.
        session_id:   Session UUID.  Auto-generated if not provided.

    Returns:
        A fully-wired TradingContext.  Every field is ready to use.

    Raises:
        duckdb.Error: If the database cannot be opened or migrations fail.
    """
    sid = session_id or str(uuid.uuid4())

    # 1. Open a single connection for the whole context.
    #    open_db() caches connections by path, so ":memory:" creates a fresh
    #    in-memory database for each test (after reset_connection() is called).
    #
    #    For tests: always pass ":memory:" explicitly so each fixture gets an
    #    independent DB rather than the global cached production connection.
    conn = duckdb.connect(db_path) if db_path == ":memory:" else open_db(db_path)

    # 2. Run migrations — idempotent, all CREATE IF NOT EXISTS.
    run_migrations(conn)

    # 3. Build services in dependency order.
    portfolio = PortfolioState(conn=conn, initial_cash=initial_cash)

    # Resolve limits once — shared between RiskGate and RiskState so they
    # enforce the same thresholds.
    limits = risk_limits or RiskLimits.from_env()

    risk_gate = RiskGate(limits=limits, portfolio=portfolio)

    # RiskState is the in-memory hot-path mirror of portfolio + risk limits.
    risk_state = RiskState.from_portfolio(portfolio=portfolio, limits=limits)

    signal_cache = SignalCache(conn=conn)

    kill_switch = KillSwitch()

    broker = PaperBroker(conn=conn, portfolio=portfolio)

    audit = DecisionLog(conn=conn)

    blackboard = Blackboard(conn=conn, session_id=sid)

    ctx = TradingContext(
        db=conn,
        portfolio=portfolio,
        risk_gate=risk_gate,
        risk_state=risk_state,
        signal_cache=signal_cache,
        kill_switch=kill_switch,
        broker=broker,
        audit=audit,
        blackboard=blackboard,
        session_id=sid,
    )

    logger.info(
        f"[TradingContext] Created | session={sid} | "
        f"db={'memory' if db_path == ':memory:' else db_path or 'default'}"
    )
    return ctx
