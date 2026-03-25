# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
TradingContext — single dependency-injection container for the trading system.

All services share one ``PgConnection`` (PostgreSQL pool-backed) and are
wired together here.  The only place a connection is opened; every downstream
service receives it as a constructor argument.

Why this matters:
  - Tests call create_trading_context(":memory:") and get a fully-wired,
    completely isolated system backed by DuckDB in-memory — zero file-system
    side-effects, no PostgreSQL required in unit tests.
  - Production calls create_trading_context() and gets PostgreSQL (TRADER_PG_URL).
  - No module-level singletons are needed. Each service is instantiated once
    per context, and the context is passed explicitly.
  - Multiple MCP server instances can each have their own TradingContext and
    connection — no exclusive file lock, no contention.

Usage — production:
    ctx = create_trading_context()

Usage — tests:
    @pytest.fixture
    def ctx():
        return create_trading_context(db_path=":memory:")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from loguru import logger

from quantstack.audit.decision_log import DecisionLog
from quantstack.db import PgConnection, open_db, run_migrations
from quantstack.execution.kill_switch import KillSwitch
from quantstack.execution.paper_broker import PaperBroker
from quantstack.execution.portfolio_state import PortfolioState
from quantstack.execution.risk_gate import RiskGate, RiskLimits
from quantstack.execution.risk_state import RiskState
from quantstack.execution.signal_cache import SignalCache
from quantstack.memory.blackboard import Blackboard


@dataclass
class TradingContext:
    """
    Fully-wired runtime container for the trading system.

    All fields are set at construction time; the object is effectively
    immutable after create_trading_context() returns.  Services share a
    single PgConnection (PostgreSQL pool connection) enabling cross-service
    ACID transactions.  Multiple TradingContexts can coexist in separate
    processes without lock contention.

    Field ownership:
      db           — PgConnection; use only for migrations / ad-hoc queries.
      portfolio    — open positions and cash; the source of truth for the system.
      risk_gate    — slow-path rule engine called by TradingDayFlow.
      risk_state   — fast in-memory mirror consumed by TickExecutor hot path.
      signal_cache — TTL-gated signal store written by analyst, read by executor.
      kill_switch  — emergency halt; checked on every tick.
      broker       — PaperBroker (paper mode) or EtradeBroker (live mode).
      audit        — append-only decision log.
      blackboard   — structured agent memory.
      session_id   — UUID for this trading session; threaded through all logs.
    """

    db: PgConnection
    portfolio: PortfolioState
    risk_gate: RiskGate
    risk_state: RiskState
    signal_cache: SignalCache
    kill_switch: KillSwitch
    broker: PaperBroker
    audit: DecisionLog

    @property
    def conn(self) -> PgConnection:
        """Alias for ``db`` — used by options_execution and other tools."""
        return self.db

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

    All services receive the same PgConnection so they can participate in the
    same transactions.  The database schema is created (idempotently) before
    any service is initialised.

    Args:
        db_path:      Pass ``":memory:"`` for an in-memory DuckDB instance
                      (unit tests).  Pass ``""`` or omit to use PostgreSQL
                      (TRADER_PG_URL env var, default postgresql://localhost/quantpod).
        initial_cash: Starting cash balance for new portfolios.
        risk_limits:  Override default risk limits.  None = load from env.
        session_id:   Session UUID.  Auto-generated if not provided.

    Returns:
        A fully-wired TradingContext.  Every field is ready to use.
    """
    sid = session_id or str(uuid.uuid4())

    # 1. Open connection — in-memory DuckDB for tests, PostgreSQL for production.
    conn = open_db(db_path)

    # 2. Run migrations — idempotent, all CREATE TABLE/INDEX IF NOT EXISTS.
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
        f"db={'memory' if db_path == ':memory:' else 'postgres'}"
    )
    return ctx
