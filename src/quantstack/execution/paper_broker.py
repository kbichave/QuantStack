# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Paper trading broker — live data, simulated fills.

Agents cannot tell whether they are in paper or live mode.
The broker interface is identical; only this file changes.

Slippage model:
  - Market orders: half-spread + sqrt(volume_impact) impact
  - Limit orders: fill only if price crosses limit; no slippage
  - Partial fills: modelled when order size > 1% of daily volume

Usage:
    broker = PaperBroker()

    fill = broker.execute(
        symbol="SPY",
        side="buy",
        quantity=100,
        order_type="market",
        current_price=450.0,
        daily_volume=80_000_000,
    )

    print(fill.fill_price, fill.slippage_bps)
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime
from threading import RLock

from loguru import logger
from pydantic import BaseModel, Field

from quantstack.db import PgConnection, open_db, run_migrations
from quantstack.execution.portfolio_state import (
    PortfolioState,
    Position,
    get_portfolio_state,
)

# =============================================================================
# DATA MODELS
# =============================================================================


class OrderRequest(BaseModel):
    """A trade order to be executed."""

    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str = "market"  # "market" or "limit"
    limit_price: float | None = None
    current_price: float = 0.0
    daily_volume: int = 1_000_000
    requested_at: datetime = Field(default_factory=datetime.now)


class Fill(BaseModel):
    """Result of executing an order."""

    order_id: str
    symbol: str
    side: str
    requested_quantity: int
    filled_quantity: int
    fill_price: float
    slippage_bps: float  # basis points of slippage applied
    commission: float = 0.0
    partial: bool = False
    rejected: bool = False
    reject_reason: str | None = None
    filled_at: datetime = Field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        return self.filled_quantity * self.fill_price + self.commission


# =============================================================================
# PAPER BROKER
# =============================================================================


class PaperBroker:
    """
    Simulated broker with realistic fill model.

    Fill model:
      1. Validate order (reject if violates hard checks)
      2. Apply half-spread slippage on market orders
      3. Apply square-root price impact for large orders
      4. Model partial fills when size > 1% of daily volume
      5. Flat $0.005/share commission (Alpaca-like)
      6. Record fill and update PortfolioState
    """

    # Default assumptions when no volume data available
    DEFAULT_DAILY_VOLUME = 5_000_000
    COMMISSION_PER_SHARE = 0.005  # $0.005/share
    HALF_SPREAD_BPS = 2  # 0.02% half-spread assumption

    def __init__(
        self,
        conn: PgConnection | None = None,
        portfolio: PortfolioState | None = None,
        # Legacy parameter — ignored when conn is provided
        db_path: str | None = None,
    ):
        self._lock = RLock()

        if conn is not None:
            self._conn = conn
        else:

            if db_path is None:
                db_path = os.getenv("PAPER_BROKER_DB_PATH", "")
            self._conn = open_db(db_path)
            run_migrations(self._conn)

        self._portfolio = (
            portfolio if portfolio is not None else get_portfolio_state(self._conn)
        )
        logger.info("PaperBroker initialized")

    @property
    def conn(self) -> PgConnection:
        return self._conn

    # -------------------------------------------------------------------------
    # Core execution
    # -------------------------------------------------------------------------

    def execute(self, req: OrderRequest) -> Fill:
        """
        Execute an order with realistic fill simulation.

        This is the single entry point. Risk gate must be called BEFORE this.
        """
        logger.info(
            f"[PAPER] {req.side.upper()} {req.quantity} {req.symbol} "
            f"@ {req.order_type} (ref ${req.current_price:.2f})"
        )

        if req.current_price <= 0:
            return self._reject(req, "current_price must be > 0")

        if req.quantity <= 0:
            return self._reject(req, "quantity must be > 0")

        # Risk gate should reject volume=0 before we reach here.
        # If it somehow slips through, use the default rather than dividing by zero.
        daily_vol = (
            req.daily_volume if req.daily_volume > 0 else self.DEFAULT_DAILY_VOLUME
        )

        # Partial fill modelling: cap at 2% of daily volume per order
        max_fill = max(1, int(daily_vol * 0.02))
        filled_qty = min(req.quantity, max_fill)
        partial = filled_qty < req.quantity

        # Slippage calculation
        fill_price = self._calc_fill_price(
            side=req.side,
            ref_price=req.current_price,
            quantity=filled_qty,
            daily_volume=daily_vol,
            order_type=req.order_type,
            limit_price=req.limit_price,
        )

        # Limit order not filled (price didn't cross)
        if fill_price is None:
            return Fill(
                order_id=req.order_id,
                symbol=req.symbol,
                side=req.side,
                requested_quantity=req.quantity,
                filled_quantity=0,
                fill_price=0.0,
                slippage_bps=0.0,
                rejected=True,
                reject_reason="Limit price not reached",
            )

        slippage_bps = abs(fill_price - req.current_price) / req.current_price * 10_000
        commission = filled_qty * self.COMMISSION_PER_SHARE

        fill = Fill(
            order_id=req.order_id,
            symbol=req.symbol,
            side=req.side,
            requested_quantity=req.quantity,
            filled_quantity=filled_qty,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            commission=commission,
            partial=partial,
        )

        self._record_fill(fill)
        self._update_portfolio(fill, req)

        logger.info(
            f"[PAPER] FILLED {fill.filled_quantity} {fill.symbol} @ "
            f"${fill.fill_price:.4f} ({fill.slippage_bps:.1f} bps slippage)"
        )
        return fill

    def _calc_fill_price(
        self,
        side: str,
        ref_price: float,
        quantity: int,
        daily_volume: int,
        order_type: str,
        limit_price: float | None,
    ) -> float | None:
        """Return simulated fill price, or None if limit order won't fill."""
        if order_type == "limit":
            if limit_price is None:
                return ref_price
            # Buy limit: only fills if ref_price <= limit_price
            # Sell limit: only fills if ref_price >= limit_price
            if side == "buy" and ref_price > limit_price:
                return None
            if side == "sell" and ref_price < limit_price:
                return None
            return limit_price  # Fill at limit, no slippage

        # Market order: half-spread + square-root impact
        direction = 1 if side == "buy" else -1

        # Half-spread
        spread_cost = ref_price * self.HALF_SPREAD_BPS / 10_000

        # Square-root market impact: k * sqrt(qty / (0.01 * daily_vol))
        # k=5 calibrated so an order at 1% of ADV costs ~5 bps of impact
        impact_bps = 5 * math.sqrt(quantity / max(1, daily_volume * 0.01))
        impact_cost = ref_price * impact_bps / 10_000

        fill_price = ref_price + direction * (spread_cost + impact_cost)
        return round(fill_price, 4)

    def _reject(self, req: OrderRequest, reason: str) -> Fill:
        fill = Fill(
            order_id=req.order_id,
            symbol=req.symbol,
            side=req.side,
            requested_quantity=req.quantity,
            filled_quantity=0,
            fill_price=0.0,
            slippage_bps=0.0,
            rejected=True,
            reject_reason=reason,
        )
        self._record_fill(fill)
        logger.warning(f"[PAPER] REJECTED {req.symbol}: {reason}")
        return fill

    def _record_fill(self, fill: Fill) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO fills
                    (order_id, symbol, side, requested_quantity, filled_quantity,
                     fill_price, slippage_bps, commission, partial, rejected,
                     reject_reason, filled_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    fill.order_id,
                    fill.symbol,
                    fill.side,
                    fill.requested_quantity,
                    fill.filled_quantity,
                    fill.fill_price,
                    fill.slippage_bps,
                    fill.commission,
                    fill.partial,
                    fill.rejected,
                    fill.reject_reason,
                    fill.filled_at,
                    getattr(fill, "session_id", ""),
                ],
            )

    def _update_portfolio(self, fill: Fill, req: OrderRequest) -> None:
        """Reflect fill in PortfolioState and adjust cash."""
        if fill.rejected or fill.filled_quantity == 0:
            return

        total_cost = fill.fill_price * fill.filled_quantity + fill.commission

        if fill.side == "buy":
            pos = Position(
                symbol=fill.symbol,
                quantity=fill.filled_quantity,
                avg_cost=fill.fill_price,
                side="long",
                current_price=fill.fill_price,
            )
            self._portfolio.upsert_position(pos)
            self._portfolio.adjust_cash(-total_cost)

        elif fill.side == "sell":
            existing = self._portfolio.get_position(fill.symbol)
            if existing and existing.side == "long":
                self._portfolio.close_position(
                    fill.symbol,
                    exit_price=fill.fill_price,
                    quantity=fill.filled_quantity,
                )
                self._portfolio.adjust_cash(
                    fill.fill_price * fill.filled_quantity - fill.commission
                )
            else:
                # Short sale
                pos = Position(
                    symbol=fill.symbol,
                    quantity=-fill.filled_quantity,
                    avg_cost=fill.fill_price,
                    side="short",
                    current_price=fill.fill_price,
                )
                self._portfolio.upsert_position(pos)
                self._portfolio.adjust_cash(
                    fill.fill_price * fill.filled_quantity - fill.commission
                )

    # -------------------------------------------------------------------------
    # History / reporting
    # -------------------------------------------------------------------------

    def get_fills(self, symbol: str | None = None, limit: int = 100) -> list[Fill]:
        """Return recent fills, optionally filtered by symbol."""
        query = "SELECT * FROM fills"
        params: list[str | int] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY filled_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [
            Fill(
                order_id=r[0],
                symbol=r[1],
                side=r[2],
                requested_quantity=r[3],
                filled_quantity=r[4],
                fill_price=r[5],
                slippage_bps=r[6],
                commission=r[7],
                partial=r[8],
                rejected=r[9],
                reject_reason=r[10],
                filled_at=r[11],
            )
            for r in rows
        ]

    def get_total_commission(self) -> float:
        """Total commission paid in paper trading session."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(commission), 0) FROM fills WHERE rejected = FALSE"
        ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def get_avg_slippage_bps(self) -> float:
        """Average slippage in basis points across all fills."""
        row = self.conn.execute(
            "SELECT COALESCE(AVG(slippage_bps), 0) FROM fills "
            "WHERE rejected = FALSE AND filled_quantity > 0"
        ).fetchone()
        return float(row[0]) if row is not None else 0.0


# Singleton — prefer injecting via TradingContext in new code.
_paper_broker: PaperBroker | None = None


def get_paper_broker(
    conn: PgConnection | None = None,
    db_path: str | None = None,
) -> PaperBroker:
    """Get the singleton PaperBroker instance."""
    global _paper_broker
    if _paper_broker is None:
        if conn is None:

            conn = open_db(db_path or "")
            run_migrations(conn)
        _paper_broker = PaperBroker(conn=conn)
    return _paper_broker
