# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Persistent portfolio state — survives process restarts.

Stores positions, cash balance, and daily P&L in DuckDB so the agent
always knows what it already holds before making new decisions.

Usage:
    state = PortfolioState()

    # On startup: load what we own
    positions = state.get_positions()

    # After a fill: record the new position
    state.upsert_position(Position(symbol="SPY", quantity=100, avg_cost=450.0, side="long"))

    # Inject into crew context as a read-only summary
    summary = state.as_context_string()

    # On close: update P&L and mark closed
    state.close_position("SPY", exit_price=460.0, quantity=100)

Reconciliation:
    # At startup compare DB state vs broker state and flag mismatches
    mismatches = state.reconcile(broker_positions=[...])
"""

from __future__ import annotations

import os
from datetime import date, datetime
from threading import RLock
from typing import Any

import duckdb
from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# DATA MODELS
# =============================================================================


class Position(BaseModel):
    """A single open position."""

    symbol: str
    quantity: int
    avg_cost: float
    side: str = "long"  # "long" or "short"
    opened_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    unrealized_pnl: float = 0.0
    current_price: float = 0.0

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_cost


class ClosedTrade(BaseModel):
    """Record of a completed trade for P&L tracking."""

    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    realized_pnl: float
    opened_at: datetime
    closed_at: datetime = Field(default_factory=datetime.now)
    holding_days: int = 0


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio snapshot."""

    snapshot_at: datetime = Field(default_factory=datetime.now)
    cash: float
    positions_value: float
    total_equity: float
    daily_pnl: float
    total_realized_pnl: float
    position_count: int
    largest_position_pct: float = 0.0


# =============================================================================
# PORTFOLIO STATE
# =============================================================================


class PortfolioState:
    """
    DuckDB-backed portfolio state.

    Thread-safe. Survives process restarts. Injected into every crew run
    as immutable context so agents know what they currently hold.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection | None = None,
        initial_cash: float = 100_000.0,
        # Legacy parameter kept for backward compatibility — ignored when conn is provided
        db_path: str | None = None,
        read_only: bool = False,
    ):
        # RLock (reentrant) because upsert_position() calls get_position() under the lock
        self._lock = RLock()
        self._initial_cash = initial_cash
        self._read_only = read_only

        if conn is not None:
            # Injected connection (preferred — supports consolidated DB and in-memory tests)
            self._conn = conn
        else:
            # Fall back to own file for backward compatibility
            from quant_pod.db import open_db, run_migrations

            if db_path is None:
                db_path = os.getenv("PORTFOLIO_DB_PATH", "~/.quant_pod/trader.duckdb")
            self._conn = open_db(db_path)
            run_migrations(self._conn)

        # Skip seeding on read-only connections — the write owner seeds on first run,
        # and DDL/INSERT would raise duckdb.InvalidInputException on a read-only conn.
        if not self._read_only:
            self._seed_cash()
        logger.info("PortfolioState initialized")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def _seed_cash(self) -> None:
        """Insert initial cash row if the table is empty (first run)."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM cash_balance").fetchone()
            existing = row[0] if row is not None else 0
            if existing == 0:
                self._conn.execute(
                    "INSERT INTO cash_balance (id, cash) VALUES (1, ?)",
                    [self._initial_cash],
                )

    # -------------------------------------------------------------------------
    # Positions
    # -------------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        """Return all open positions."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT symbol, quantity, avg_cost, side, opened_at, last_updated, "
                "unrealized_pnl, current_price FROM positions"
            ).fetchall()
        return [
            Position(
                symbol=r[0],
                quantity=r[1],
                avg_cost=r[2],
                side=r[3],
                opened_at=r[4],
                last_updated=r[5],
                unrealized_pnl=r[6],
                current_price=r[7],
            )
            for r in rows
        ]

    def get_position(self, symbol: str) -> Position | None:
        """Return a single position or None."""
        with self._lock:
            row = self.conn.execute(
                "SELECT symbol, quantity, avg_cost, side, opened_at, last_updated, "
                "unrealized_pnl, current_price FROM positions WHERE symbol = ?",
                [symbol],
            ).fetchone()
        if row is None:
            return None
        return Position(
            symbol=row[0],
            quantity=row[1],
            avg_cost=row[2],
            side=row[3],
            opened_at=row[4],
            last_updated=row[5],
            unrealized_pnl=row[6],
            current_price=row[7],
        )

    def upsert_position(self, pos: Position) -> None:
        """
        Insert or update a position after a fill.

        Side-flip handling: if the incoming position has the opposite side to the
        existing one, we first close the existing position (recording realized P&L at
        the new fill price), then insert the new reversed position.  Overwriting
        in-place would silently discard the realized P&L on the closed inventory.
        """
        with self._lock:
            existing = self.get_position(pos.symbol)
            if existing is None:
                self.conn.execute(
                    """
                    INSERT INTO positions
                        (symbol, quantity, avg_cost, side, opened_at, last_updated,
                         unrealized_pnl, current_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        pos.symbol,
                        pos.quantity,
                        pos.avg_cost,
                        pos.side,
                        pos.opened_at,
                        pos.last_updated,
                        pos.unrealized_pnl,
                        pos.current_price,
                    ],
                )
            elif existing.side == pos.side:
                # Same direction — weighted average cost
                total_qty = existing.quantity + pos.quantity
                if total_qty == 0:
                    # Nets to zero: treat as full close (caller should use close_position instead)
                    self.conn.execute("DELETE FROM positions WHERE symbol = ?", [pos.symbol])
                    logger.info(f"[PORTFOLIO] Position {pos.symbol} netted to zero")
                    return
                new_avg = (
                    (existing.avg_cost * existing.quantity) + (pos.avg_cost * pos.quantity)
                ) / total_qty
                self.conn.execute(
                    """
                    UPDATE positions
                    SET quantity = ?, avg_cost = ?, side = ?,
                        last_updated = ?, unrealized_pnl = ?, current_price = ?
                    WHERE symbol = ?
                    """,
                    [
                        total_qty,
                        new_avg,
                        pos.side,
                        datetime.now(),
                        pos.unrealized_pnl,
                        pos.current_price,
                        pos.symbol,
                    ],
                )
            else:
                # Side flip: close the existing position first, then open the reverse.
                # Using pos.avg_cost as the exit price since it represents the fill price
                # of the reversing trade.
                close_qty = abs(existing.quantity)
                mult = 1 if existing.side == "long" else -1
                realized = mult * (pos.avg_cost - existing.avg_cost) * close_qty
                holding_days = (datetime.now() - existing.opened_at).days

                self.conn.execute(
                    """
                    INSERT INTO closed_trades
                        (id, symbol, side, quantity, entry_price, exit_price,
                         realized_pnl, opened_at, closed_at, holding_days)
                    VALUES (nextval('closed_trades_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        existing.symbol,
                        existing.side,
                        close_qty,
                        existing.avg_cost,
                        pos.avg_cost,
                        realized,
                        existing.opened_at,
                        datetime.now(),
                        holding_days,
                    ],
                )

                # Delete old position and insert the new reversed one
                self.conn.execute("DELETE FROM positions WHERE symbol = ?", [pos.symbol])
                self.conn.execute(
                    """
                    INSERT INTO positions
                        (symbol, quantity, avg_cost, side, opened_at, last_updated,
                         unrealized_pnl, current_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        pos.symbol,
                        pos.quantity,
                        pos.avg_cost,
                        pos.side,
                        pos.opened_at,
                        pos.last_updated,
                        pos.unrealized_pnl,
                        pos.current_price,
                    ],
                )
                logger.info(
                    f"[PORTFOLIO] Side flip {pos.symbol}: closed {existing.side} {close_qty} "
                    f"@ {pos.avg_cost:.2f} (P&L: {realized:+.2f}), "
                    f"opened {pos.side} {pos.quantity}"
                )
                return

        logger.info(f"[PORTFOLIO] Upserted {pos.symbol}: {pos.quantity} @ {pos.avg_cost:.2f}")

    def update_prices(self, prices: dict[str, float]) -> None:
        """Mark-to-market: update current_price and unrealized_pnl for all positions."""
        with self._lock:
            for symbol, price in prices.items():
                pos = self.get_position(symbol)
                if pos is None:
                    continue
                mult = 1 if pos.side == "long" else -1
                unrealized = mult * (price - pos.avg_cost) * abs(pos.quantity)
                self.conn.execute(
                    "UPDATE positions SET current_price = ?, unrealized_pnl = ?, "
                    "last_updated = ? WHERE symbol = ?",
                    [price, unrealized, datetime.now(), symbol],
                )

    def close_position(
        self, symbol: str, exit_price: float, quantity: int | None = None
    ) -> ClosedTrade | None:
        """
        Close all or part of a position and record realized P&L.

        Returns the ClosedTrade record, or None if no position existed.
        """
        with self._lock:
            pos = self.get_position(symbol)
            if pos is None:
                logger.warning(f"[PORTFOLIO] No open position for {symbol}")
                return None

            close_qty = quantity or abs(pos.quantity)
            mult = 1 if pos.side == "long" else -1
            realized = mult * (exit_price - pos.avg_cost) * close_qty

            holding_days = (datetime.now() - pos.opened_at).days

            closed = ClosedTrade(
                symbol=symbol,
                side=pos.side,
                quantity=close_qty,
                entry_price=pos.avg_cost,
                exit_price=exit_price,
                realized_pnl=realized,
                opened_at=pos.opened_at,
                holding_days=holding_days,
            )

            self.conn.execute(
                """
                INSERT INTO closed_trades
                    (id, symbol, side, quantity, entry_price, exit_price,
                     realized_pnl, opened_at, closed_at, holding_days)
                VALUES (nextval('closed_trades_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol,
                    pos.side,
                    close_qty,
                    pos.avg_cost,
                    exit_price,
                    realized,
                    pos.opened_at,
                    datetime.now(),
                    holding_days,
                ],
            )

            remaining = abs(pos.quantity) - close_qty
            if remaining <= 0:
                self.conn.execute("DELETE FROM positions WHERE symbol = ?", [symbol])
            else:
                self.conn.execute(
                    "UPDATE positions SET quantity = ?, last_updated = ? WHERE symbol = ?",
                    [remaining * (1 if pos.side == "long" else -1), datetime.now(), symbol],
                )

            logger.info(
                f"[PORTFOLIO] Closed {close_qty} {symbol} @ {exit_price:.2f} | P&L: {realized:+.2f}"
            )
            return closed

    # -------------------------------------------------------------------------
    # Cash
    # -------------------------------------------------------------------------

    def get_cash(self) -> float:
        """Return current cash balance."""
        with self._lock:
            row = self.conn.execute("SELECT cash FROM cash_balance WHERE id = 1").fetchone()
        return float(row[0]) if row else self._initial_cash

    def adjust_cash(self, delta: float) -> float:
        """Adjust cash by delta (positive = add, negative = spend)."""
        with self._lock:
            current = self.get_cash()  # RLock allows re-entry here
            new_cash = current + delta
            self.conn.execute(
                "UPDATE cash_balance SET cash = ?, updated_at = ? WHERE id = 1",
                [new_cash, datetime.now()],
            )
            return new_cash

    # -------------------------------------------------------------------------
    # P&L
    # -------------------------------------------------------------------------

    def get_total_realized_pnl(self) -> float:
        """Sum of all realized P&L from closed trades."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM closed_trades"
            ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def get_daily_pnl(self, for_date: date | None = None) -> float:
        """Realized P&L for a specific date (defaults to today)."""
        d = for_date or date.today()
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM closed_trades "
                "WHERE closed_at::DATE = ?",
                [str(d)],
            ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def get_snapshot(self) -> PortfolioSnapshot:
        """
        Atomic portfolio snapshot for monitoring/alerting.

        Acquires the lock once across all reads so the snapshot is
        internally consistent — no other write can interleave.
        """
        with self._lock:
            positions = self.get_positions()
            cash = self.get_cash()
            daily_pnl = self.get_daily_pnl()
            total_realized = self.get_total_realized_pnl()

        positions_value = sum(abs(p.quantity) * p.current_price for p in positions)
        total_equity = cash + positions_value
        largest_pct = 0.0
        if total_equity > 0 and positions:
            largest_pct = max(abs(p.quantity) * p.current_price / total_equity for p in positions)
        return PortfolioSnapshot(
            cash=cash,
            positions_value=positions_value,
            total_equity=total_equity,
            daily_pnl=daily_pnl,
            total_realized_pnl=total_realized,
            position_count=len(positions),
            largest_position_pct=largest_pct,
        )

    # -------------------------------------------------------------------------
    # Context string (injected into every crew run)
    # -------------------------------------------------------------------------

    def as_context_string(self) -> str:
        """
        Format current portfolio as a markdown string for agent context.

        This is injected at the TOP of every crew run so agents always
        know what they already hold before making new decisions.
        """
        positions = self.get_positions()
        snapshot = self.get_snapshot()

        lines = [
            "## CURRENT PORTFOLIO STATE (READ-ONLY — DO NOT CONTRADICT)",
            "",
            f"**Cash:** ${snapshot.cash:,.2f}",
            f"**Positions Value:** ${snapshot.positions_value:,.2f}",
            f"**Total Equity:** ${snapshot.total_equity:,.2f}",
            f"**Daily P&L:** ${snapshot.daily_pnl:+,.2f}",
            f"**Total Realized P&L:** ${snapshot.total_realized_pnl:+,.2f}",
            "",
        ]

        if positions:
            lines.append("### Open Positions")
            lines.append("")
            lines.append("| Symbol | Side | Qty | Avg Cost | Current | Unrealized P&L |")
            lines.append("|--------|------|-----|----------|---------|---------------|")
            for p in positions:
                lines.append(
                    f"| {p.symbol} | {p.side.upper()} | {p.quantity:,} "
                    f"| ${p.avg_cost:.2f} | ${p.current_price:.2f} "
                    f"| ${p.unrealized_pnl:+,.2f} |"
                )
        else:
            lines.append("### Open Positions")
            lines.append("")
            lines.append("*No open positions.*")

        lines += [
            "",
            "**RULE:** Before recommending any trade, verify it does not duplicate "
            "an existing position above. Adding to an existing position requires "
            "explicit justification.",
        ]

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    def reconcile(self, broker_positions: list[dict[str, Any]]) -> list[str]:
        """
        Compare DB state to broker-reported positions.

        Returns a list of mismatch descriptions (empty = clean).

        Args:
            broker_positions: List of dicts with keys:
                symbol, quantity, side, avg_cost
        """
        mismatches = []
        db_positions = {p.symbol: p for p in self.get_positions()}
        broker_map = {p["symbol"]: p for p in broker_positions}

        # DB positions not in broker
        for symbol, db_pos in db_positions.items():
            if symbol not in broker_map:
                mismatches.append(f"DB has {db_pos.quantity} {symbol} but broker shows none")
            else:
                bp = broker_map[symbol]
                if abs(db_pos.quantity - bp["quantity"]) > 0:
                    mismatches.append(
                        f"{symbol}: DB qty={db_pos.quantity}, broker qty={bp['quantity']}"
                    )

        # Broker positions not in DB
        for symbol, bp in broker_map.items():
            if symbol not in db_positions:
                mismatches.append(f"Broker has {bp['quantity']} {symbol} but DB shows none")

        if mismatches:
            logger.warning(f"[PORTFOLIO] Reconciliation found {len(mismatches)} mismatches")
            for m in mismatches:
                logger.warning(f"  - {m}")
        else:
            logger.info("[PORTFOLIO] Reconciliation clean")

        return mismatches

    def reset(self, initial_cash: float | None = None) -> None:
        """Reset portfolio to zero positions (use for paper trading resets)."""
        with self._lock:
            self.conn.execute("DELETE FROM positions")
            cash = initial_cash or self._initial_cash
            self.conn.execute(
                "UPDATE cash_balance SET cash = ?, updated_at = ? WHERE id = 1",
                [cash, datetime.now()],
            )
        logger.info(f"[PORTFOLIO] Reset to ${cash:,.2f} cash, no positions")


# Singleton — used only when no TradingContext is available.
# Prefer injecting a PortfolioState through TradingContext in all new code.
_portfolio_state: PortfolioState | None = None


def get_portfolio_state(
    conn: duckdb.DuckDBPyConnection | None = None,
    initial_cash: float = 100_000.0,
    # Legacy parameter — ignored when conn is provided
    db_path: str | None = None,
) -> PortfolioState:
    """Get the singleton PortfolioState instance (write connection)."""
    global _portfolio_state
    if _portfolio_state is None:
        if conn is None:
            from quant_pod.db import open_db, run_migrations

            conn = open_db(db_path or "")
            run_migrations(conn)
        _portfolio_state = PortfolioState(conn=conn, initial_cash=initial_cash)
    return _portfolio_state


# Read-only singleton — for processes that must not compete for the write lock
# (e.g. the FastAPI server's GET endpoints).
_portfolio_state_ro: PortfolioState | None = None


def get_portfolio_state_readonly() -> PortfolioState:
    """
    Get a read-only PortfolioState singleton.

    Use this in processes (FastAPI, scripts) that run alongside the MCP server
    and only need to READ portfolio data.  The returned instance cannot execute
    writes (upsert_position, adjust_cash, etc.) — those calls will raise a
    duckdb.InvalidInputException at runtime.

    Raises:
        FileNotFoundError: if trader.duckdb doesn't exist yet (MCP server not started).
    """
    global _portfolio_state_ro
    if _portfolio_state_ro is None:
        from quant_pod.db import open_db_readonly

        _portfolio_state_ro = PortfolioState(conn=open_db_readonly(), read_only=True)
    return _portfolio_state_ro
