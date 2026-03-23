# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
In-memory risk state for the tick executor hot path.

The RiskGate (risk_gate.py) is the authoritative rule engine — it handles
the full rule set and has disk-backed persistence.  RiskState is its
read-optimised shadow that the tick executor checks in sub-microseconds.

Invariants:
  - Loaded from DuckDB (portfolio + risk_gate state) on startup.
  - Updated in-memory after every fill and every mark-to-market.
  - Written back to DuckDB asynchronously by a background flusher.
  - The tick executor NEVER reads from DuckDB — only from this object.

The split allows the tick executor to run at market-data frequency without
being blocked by DB I/O or the RiskGate's full rule evaluation.  When
RiskState says HALT, the executor stops immediately; the RiskGate catches
all the edge cases on the slower analysis plane.

Usage:
    state = RiskState.from_portfolio(portfolio, limits)

    # Tick executor hot path — purely in-memory
    verdict = state.check(signal, tick_price)
    if not verdict.approved:
        continue

    # After a fill, update state (still in-memory, no DB)
    state.apply_fill(symbol="SPY", side="buy", quantity=100, price=450.0)

    # Periodically sync back to DB (done by background flusher, not hot path)
    state.flush_to_db(conn)
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from threading import RLock

import duckdb
from loguru import logger

from quantstack.execution.risk_gate import RiskLimits, RiskVerdict, RiskViolation
from quantstack.execution.signal_cache import TradeSignal

# ---------------------------------------------------------------------------
# In-memory position snapshot (lighter than the full Position model)
# ---------------------------------------------------------------------------


@dataclass
class PositionSlot:
    """Minimal position info needed for risk checks in the hot path."""

    symbol: str
    side: str  # "long" or "short"
    quantity: int  # absolute value (sign encoded in side)
    avg_cost: float
    current_price: float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.current_price


# ---------------------------------------------------------------------------
# RiskState
# ---------------------------------------------------------------------------


@dataclass
class RiskState:
    """
    Lightweight in-memory risk snapshot.

    All fields are plain Python — no DB access, no I/O.
    The tick executor checks this on every tick.
    """

    # Current portfolio metrics
    cash: float = 100_000.0
    positions: dict[str, PositionSlot] = field(default_factory=dict)
    daily_realized_pnl: float = 0.0

    # Flags
    kill_switch_active: bool = False
    daily_halted: bool = False

    # Limits (loaded from RiskLimits / env on startup)
    limits: RiskLimits = field(default_factory=RiskLimits)

    # Thread safety for concurrent tick/fill operations
    _lock: RLock = field(default_factory=RLock, compare=False, repr=False)

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_portfolio(
        cls,
        portfolio: PortfolioState,  # type: ignore[name-defined]  # noqa: F821
        limits: RiskLimits | None = None,
    ) -> RiskState:
        """
        Initialise from a live PortfolioState snapshot.

        Called once at session start; subsequent updates via apply_fill().
        """
        snapshot = portfolio.get_snapshot()
        db_positions = portfolio.get_positions()

        slots: dict[str, PositionSlot] = {}
        for p in db_positions:
            slots[p.symbol] = PositionSlot(
                symbol=p.symbol,
                side=p.side,
                quantity=abs(p.quantity),
                avg_cost=p.avg_cost,
                current_price=p.current_price,
            )

        state = cls(
            cash=snapshot.cash,
            positions=slots,
            daily_realized_pnl=snapshot.daily_pnl,
            limits=limits or RiskLimits.from_env(),
        )
        logger.info(
            f"[RiskState] Loaded: cash=${state.cash:,.0f} "
            f"positions={len(state.positions)} "
            f"daily_pnl={state.daily_realized_pnl:+.2f}"
        )
        return state

    # ---------------------------------------------------------------------------
    # Hot-path check — called on every tick (no I/O)
    # ---------------------------------------------------------------------------

    def check(self, signal: TradeSignal, tick_price: float) -> RiskVerdict:
        """
        Fast in-memory risk check for the tick executor.

        Checks only the subset of rules that can be evaluated from the
        in-memory snapshot.  The full RiskGate runs slower checks on the
        analysis plane.

        Returns RiskVerdict with approved_quantity set to the final order size.
        """
        with self._lock:
            # 1. Kill switch
            if self.kill_switch_active:
                return RiskVerdict(
                    approved=False,
                    violations=[
                        RiskViolation(
                            rule="kill_switch",
                            limit=0,
                            actual=0,
                            description="Kill switch is active — trading halted",
                        )
                    ],
                )

            # 2. Daily halt
            if self.daily_halted:
                return RiskVerdict(
                    approved=False,
                    violations=[
                        RiskViolation(
                            rule="daily_loss_halt",
                            limit=0,
                            actual=0,
                            description="Daily halt active — trading halted for today",
                        )
                    ],
                )

            # 3. Daily loss limit (fast path using in-memory P&L)
            equity = self._total_equity(tick_price)
            if equity > 0:
                loss_pct = abs(min(0.0, self.daily_realized_pnl)) / equity
                if loss_pct >= self.limits.daily_loss_limit_pct:
                    self.daily_halted = True
                    return RiskVerdict(
                        approved=False,
                        violations=[
                            RiskViolation(
                                rule="daily_loss_limit",
                                limit=self.limits.daily_loss_limit_pct,
                                actual=loss_pct,
                                description=(
                                    f"Daily loss {loss_pct:.1%} >= "
                                    f"limit {self.limits.daily_loss_limit_pct:.1%} — HALT"
                                ),
                            )
                        ],
                    )

            # 4. Per-symbol position size
            quantity = max(1, int(equity * signal.position_size_pct / tick_price))
            existing = self.positions.get(signal.symbol.upper())
            existing_notional = existing.notional if existing else 0.0
            order_notional = quantity * tick_price
            new_notional = existing_notional + order_notional

            max_notional = min(
                equity * self.limits.max_position_pct,
                self.limits.max_position_notional,
            )
            if new_notional > max_notional:
                allowed = max(0.0, max_notional - existing_notional)
                quantity = int(allowed / tick_price) if tick_price > 0 else 0
                if quantity <= 0:
                    return RiskVerdict(
                        approved=False,
                        violations=[
                            RiskViolation(
                                rule="max_position_size",
                                limit=max_notional,
                                actual=new_notional,
                                description=(
                                    f"{signal.symbol} position limit reached "
                                    f"(${new_notional:,.0f} > ${max_notional:,.0f})"
                                ),
                            )
                        ],
                    )

            # 5. Gross exposure
            current_gross = sum(p.notional for p in self.positions.values())
            if (
                current_gross + order_notional
                > equity * self.limits.max_gross_exposure_pct
            ):
                return RiskVerdict(
                    approved=False,
                    violations=[
                        RiskViolation(
                            rule="max_gross_exposure",
                            limit=equity * self.limits.max_gross_exposure_pct,
                            actual=current_gross + order_notional,
                            description="Gross exposure limit reached",
                        )
                    ],
                )

            return RiskVerdict(approved=True, approved_quantity=quantity)

    # ---------------------------------------------------------------------------
    # State updates (called after fills, mark-to-market)
    # ---------------------------------------------------------------------------

    def apply_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        realized_pnl: float = 0.0,
    ) -> None:
        """Update in-memory state after a fill. No I/O."""
        with self._lock:
            sym = symbol.upper()
            cost = quantity * price

            if side == "buy":
                self.cash -= cost
                existing = self.positions.get(sym)
                if existing and existing.side == "long":
                    new_qty = existing.quantity + quantity
                    new_avg = (
                        existing.avg_cost * existing.quantity + price * quantity
                    ) / new_qty
                    existing.quantity = new_qty
                    existing.avg_cost = new_avg
                    existing.current_price = price
                else:
                    self.positions[sym] = PositionSlot(
                        symbol=sym,
                        side="long",
                        quantity=quantity,
                        avg_cost=price,
                        current_price=price,
                    )
            elif side == "sell":
                self.cash += cost - abs(realized_pnl)  # rough cash adjustment
                existing = self.positions.get(sym)
                if existing:
                    remaining = existing.quantity - quantity
                    if remaining <= 0:
                        del self.positions[sym]
                    else:
                        existing.quantity = remaining
                        existing.current_price = price
                self.daily_realized_pnl += realized_pnl

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update current prices for open positions. No I/O."""
        with self._lock:
            for sym, price in prices.items():
                slot = self.positions.get(sym.upper())
                if slot is not None:
                    slot.current_price = price

    def set_kill_switch(self, active: bool) -> None:
        """Mirror kill switch state — called by the KillSwitch on trigger/reset."""
        with self._lock:
            self.kill_switch_active = active

    def reset_daily(self) -> None:
        """Reset per-day state at session start."""
        with self._lock:
            self.daily_realized_pnl = 0.0
            self.daily_halted = False

    # ---------------------------------------------------------------------------
    # Persistence (async background flusher — NOT called in hot path)
    # ---------------------------------------------------------------------------

    def flush_to_db(self, conn: duckdb.DuckDBPyConnection) -> None:  # type: ignore
        """
        Persist current risk state to the system_state table.

        Called periodically by the background flusher, never in the hot path.
        """
        with self._lock:
            conn.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES ('risk_state', ?, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                                updated_at = excluded.updated_at
                """,
                [
                    _json.dumps(
                        {
                            "cash": self.cash,
                            "daily_realized_pnl": self.daily_realized_pnl,
                            "kill_switch_active": self.kill_switch_active,
                            "daily_halted": self.daily_halted,
                        }
                    )
                ],
            )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _total_equity(self, current_price: float = 0.0) -> float:
        """Approximate total equity from in-memory state."""
        positions_value = sum(p.notional for p in self.positions.values())
        return self.cash + positions_value
