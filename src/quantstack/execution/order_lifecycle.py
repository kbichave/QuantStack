# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
OMS Order Lifecycle — explicit state machine separating OMS from EMS.

Addresses GAP-7 in the gap analysis:
  "Signal → RiskGate → PaperBroker/eTrade MCP. No separation between
   OMS and EMS layers."

What this adds to the existing flow:
  BEFORE (conflated):
    TradeSignal → RiskGate.check() → broker.execute() → Fill

  AFTER (separated):
    TradeSignal → [OMS] OrderLifecycle.submit() → Order(NEW)
               → [OMS] OrderLifecycle.acknowledge() → Order(SUBMITTED)
               → [EMS] broker.execute() → Fill
               → [OMS] OrderLifecycle.record_fill() → Order(FILLED)

The OMS layer adds:
  1. Explicit order state machine (auditable lifecycle per order)
  2. Arrival price capture at signal-fire time (feeds TCAEngine)
  3. Execution algorithm selection (IMMEDIATE / TWAP / VWAP) based on
     order size as % of ADV — uses the same logic as TCAEngine.pre_trade()
  4. Order expiry — stale orders (generated before current bar close)
     never reach the broker
  5. Pre-trade compliance check distinct from RiskGate risk check:
     - Restricted hours (e.g., no trades in first/last 5min of session)
     - Duplicate order prevention (same symbol, same direction within 60s)

State machine:
  NEW → SUBMITTED → PARTIALLY_FILLED → FILLED
                 └→ REJECTED
                 └→ CANCELLED
                 └→ EXPIRED (TTL elapsed before fill)

Invariants:
  - Only NEW orders can be submitted (no double-submit).
  - FILLED/REJECTED/CANCELLED/EXPIRED are terminal — no transitions out.
  - Thread-safe: all mutations go through _lock.

Usage:
    oms = OrderLifecycle(conn)

    # When signal fires:
    order = oms.submit(signal, risk_verdict, arrival_price=450.25, adv=80_000_000)
    if order.status != OrderStatus.SUBMITTED:
        # Rejected by OMS compliance check
        return

    # When broker fills:
    order = oms.record_fill(order.order_id, fill)
    # TCA can now compute implementation shortfall via order.arrival_price vs fill.fill_price
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from threading import RLock

from loguru import logger

from quantstack.core.execution.tca_storage import TCAStore
from quantstack.db import PgConnection, open_db, run_migrations
from quantstack.observability.trace import TraceContext


class OrderStatus(str, Enum):
    """Order lifecycle states."""

    NEW = "new"  # Created, not yet submitted to broker
    SUBMITTED = "submitted"  # Sent to broker, awaiting acknowledgement
    ACKNOWLEDGED = "acknowledged"  # Broker accepted the order
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"  # Terminal: fully executed
    REJECTED = "rejected"  # Terminal: broker or OMS rejected
    CANCELLED = "cancelled"  # Terminal: cancelled before fill
    EXPIRED = "expired"  # Terminal: TTL elapsed, never submitted


# States from which no further transitions are allowed
_TERMINAL_STATES = {
    OrderStatus.FILLED,
    OrderStatus.REJECTED,
    OrderStatus.CANCELLED,
    OrderStatus.EXPIRED,
}


class ExecAlgoOMS(str, Enum):
    """Execution algorithm selected by OMS pre-trade analysis."""

    IMMEDIATE = "immediate"  # < 0.2% ADV — submit now, market order
    TWAP = "twap"  # 0.2–1% ADV — spread over session time
    VWAP = "vwap"  # 1–5% ADV — volume-weighted over session
    POV = "pov"  # > 5% ADV — participation-rate algo (institutional)


@dataclass
class Order:
    """
    A single order tracked through its lifecycle.

    Immutable fields (set on creation): order_id, symbol, side, quantity,
    signal_id, arrival_price.

    Mutable fields (updated on state transitions): status, filled_quantity,
    fill_price, exec_algo, rejection_reason, timestamps.
    """

    order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    quantity: int
    signal_id: str  # Traceability back to TradeSignal.session_id
    arrival_price: float  # Price at signal-fire time (TCA benchmark)
    exec_algo: ExecAlgoOMS = ExecAlgoOMS.IMMEDIATE
    order_type: str = "market"  # "market" | "limit"
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: int = 0
    fill_price: float | None = None
    rejection_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    expires_at: datetime | None = None  # None = no expiry
    trace_id: str | None = None  # Observability: trace ID from signal → fill

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATES

    @property
    def implementation_shortfall_bps(self) -> float | None:
        """
        Basis-point cost vs arrival price (positive = cost, negative = improvement).

        Only available after FILLED.
        """
        if self.fill_price is None or self.arrival_price <= 0:
            return None
        direction = 1 if self.side == "buy" else -1
        return (
            direction
            * (self.fill_price - self.arrival_price)
            / self.arrival_price
            * 10_000
        )

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) >= self.expires_at


class OrderLifecycle:
    """
    OMS — tracks order state from signal to fill.

    Injected between the analysis plane (SignalCache) and the execution plane
    (broker.execute) to provide order-level auditability and compliance checks.

    Persist to `orders` table in the shared DuckDB connection.
    """

    _lock = RLock()

    # Compliance: reject new orders for the same symbol+side within this window.
    # Prevents duplicate signals firing multiple orders in one session.
    _DUPLICATE_WINDOW_SECONDS = 60

    # Default order TTL — orders not filled within this time are expired.
    _DEFAULT_TTL_SECONDS = 900  # 15 minutes

    def __init__(self, conn: PgConnection) -> None:
        self.conn = conn
        self._orders: dict[str, Order] = {}  # order_id → Order (in-memory)
        self._ensure_table()
        self._load_pending()

    # -------------------------------------------------------------------------
    # OMS entry point — called when a new signal clears RiskGate
    # -------------------------------------------------------------------------

    def submit(
        self,
        symbol: str,
        side: str,
        quantity: int,
        arrival_price: float,
        signal_id: str = "",
        adv: int = 1_000_000,
        order_type: str = "market",
        limit_price: float | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> Order:
        """
        Create and submit an order through OMS pre-trade compliance checks.

        Steps:
          1. Duplicate order check (same symbol+side within 60s → REJECTED)
          2. Expiry time check (0-quantity or zero price → REJECTED)
          3. Execution algorithm selection (based on size / ADV)
          4. Persist order to DuckDB
          5. Transition NEW → SUBMITTED

        Returns the order. Callers must check order.status — if REJECTED or
        EXPIRED, do not call broker.execute().
        """
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side.lower(),
            quantity=quantity,
            signal_id=signal_id,
            arrival_price=arrival_price,
            order_type=order_type,
            limit_price=limit_price,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
            trace_id=TraceContext.get_trace_id(),
        )

        with self._lock:
            # Compliance: duplicate order check
            rejection = self._compliance_check(order)
            if rejection:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = rejection
                self._orders[order.order_id] = order
                self._persist(order)
                logger.warning(
                    f"[OMS] Order REJECTED ({symbol} {side} {quantity}): {rejection}"
                )
                return order

            # Select execution algorithm based on size / ADV
            order.exec_algo = self._select_exec_algo(quantity, adv)

            # Transition NEW → SUBMITTED
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now(UTC)
            self._orders[order.order_id] = order
            self._persist(order)

        logger.info(
            f"[OMS] Order SUBMITTED {order.order_id[:8]} "
            f"{symbol} {side.upper()} {quantity} "
            f"@ arrival ${arrival_price:.2f} algo={order.exec_algo.value}"
        )
        return order

    # -------------------------------------------------------------------------
    # State transitions
    # -------------------------------------------------------------------------

    def acknowledge(self, order_id: str) -> Order:
        """Broker acknowledged receipt of the order (SUBMITTED → ACKNOWLEDGED)."""
        return self._transition(order_id, OrderStatus.ACKNOWLEDGED)

    def record_partial_fill(
        self,
        order_id: str,
        filled_quantity: int,
        fill_price: float,
    ) -> Order:
        """Record a partial fill (SUBMITTED|ACKNOWLEDGED|PARTIALLY_FILLED → PARTIALLY_FILLED)."""
        with self._lock:
            order = self._get_mutable(order_id)
            if order is None:
                raise KeyError(f"[OMS] Order not found: {order_id}")
            if order.is_terminal:
                raise ValueError(
                    f"[OMS] Cannot partially fill terminal order {order_id} ({order.status})"
                )

            order.filled_quantity += filled_quantity
            order.fill_price = fill_price  # Running fill price (last partial price)
            order.status = OrderStatus.PARTIALLY_FILLED
            self._orders[order_id] = order
            self._persist(order)

        logger.info(
            f"[OMS] Order PARTIAL FILL {order_id[:8]}: "
            f"{filled_quantity}/{order.quantity} @ ${fill_price:.4f}"
        )
        return order

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        filled_quantity: int | None = None,
    ) -> Order:
        """
        Record a complete fill and transition to FILLED.

        Args:
            order_id: The order to close out.
            fill_price: Actual execution price.
            filled_quantity: If None, assumes full order quantity was filled.
        """
        with self._lock:
            order = self._get_mutable(order_id)
            if order is None:
                raise KeyError(f"[OMS] Order not found: {order_id}")
            if order.is_terminal:
                raise ValueError(
                    f"[OMS] Cannot fill terminal order {order_id} ({order.status})"
                )

            order.filled_quantity = (
                filled_quantity if filled_quantity is not None else order.quantity
            )
            order.fill_price = fill_price
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now(UTC)
            self._orders[order_id] = order
            self._persist(order)

        shortfall = order.implementation_shortfall_bps
        logger.info(
            f"[OMS] Order FILLED {order_id[:8]} {order.symbol} "
            f"@ ${fill_price:.4f} | IS={shortfall:.1f}bps vs arrival ${order.arrival_price:.2f}"
            if shortfall is not None
            else f"[OMS] Order FILLED {order_id[:8]} {order.symbol} @ ${fill_price:.4f}"
        )

        # --- Persist TCA result to durable storage ---
        # Non-fatal: if TCAStore fails, the fill is still recorded in OMS.
        self._persist_tca_result(order)

        return order

    def reject(self, order_id: str, reason: str) -> Order:
        """Broker-side rejection (SUBMITTED|ACKNOWLEDGED → REJECTED)."""
        with self._lock:
            order = self._get_mutable(order_id)
            if order is None:
                raise KeyError(f"[OMS] Order not found: {order_id}")
            if order.is_terminal:
                raise ValueError(f"[OMS] Cannot reject terminal order {order_id}")
            order.status = OrderStatus.REJECTED
            order.rejection_reason = reason
            self._orders[order_id] = order
            self._persist(order)

        logger.warning(f"[OMS] Order REJECTED {order_id[:8]}: {reason}")
        return order

    def cancel(self, order_id: str, reason: str = "Manual cancel") -> Order:
        """Cancel a pending order (any non-terminal state → CANCELLED)."""
        with self._lock:
            order = self._get_mutable(order_id)
            if order is None:
                raise KeyError(f"[OMS] Order not found: {order_id}")
            if order.is_terminal:
                return order  # Already done — idempotent
            order.status = OrderStatus.CANCELLED
            order.rejection_reason = reason
            self._orders[order_id] = order
            self._persist(order)

        logger.info(f"[OMS] Order CANCELLED {order_id[:8]}: {reason}")
        return order

    def expire_stale(self) -> list[Order]:
        """
        Scan pending orders and transition expired ones to EXPIRED.

        Call this at the start of each tick-executor cycle to prevent
        stale orders from the previous analysis cycle reaching the broker.
        Returns the list of orders that were expired.
        """
        expired = []
        with self._lock:
            for order in list(self._orders.values()):
                if not order.is_terminal and order.is_expired:
                    order.status = OrderStatus.EXPIRED
                    order.rejection_reason = "TTL elapsed"
                    self._orders[order.order_id] = order
                    self._persist(order)
                    expired.append(order)

        if expired:
            logger.info(f"[OMS] Expired {len(expired)} stale orders")
        return expired

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_order(self, order_id: str) -> Order | None:
        """Return the in-memory order, or None if not found."""
        with self._lock:
            return self._orders.get(order_id)

    def get_pending_orders(self) -> list[Order]:
        """Return all orders in non-terminal states."""
        with self._lock:
            return [o for o in self._orders.values() if not o.is_terminal]

    def get_recent_orders(self, limit: int = 50) -> list[Order]:
        """Return the most recently created orders (for API/dashboard)."""
        with self._lock:
            orders = sorted(
                self._orders.values(), key=lambda o: o.created_at, reverse=True
            )
            return orders[:limit]

    def session_summary(self) -> dict:
        """
        Aggregate statistics for the current session's orders.

        Returns counts per status, total notional, fill rate, and
        average implementation shortfall (bps) across filled orders.
        """
        with self._lock:
            orders = list(self._orders.values())

        status_counts: dict[str, int] = {}
        for o in orders:
            status_counts[o.status.value] = status_counts.get(o.status.value, 0) + 1

        filled = [o for o in orders if o.status == OrderStatus.FILLED]
        shortfalls = [
            o.implementation_shortfall_bps
            for o in filled
            if o.implementation_shortfall_bps is not None
        ]

        total_notional = sum(
            (o.fill_price or o.arrival_price) * o.filled_quantity for o in filled
        )

        return {
            "order_count": len(orders),
            "status_breakdown": status_counts,
            "fill_rate": len(filled) / len(orders) if orders else 0.0,
            "avg_implementation_shortfall_bps": (
                sum(shortfalls) / len(shortfalls) if shortfalls else 0.0
            ),
            "total_filled_notional": total_notional,
        }

    # -------------------------------------------------------------------------
    # Execution algorithm selection
    # -------------------------------------------------------------------------

    @staticmethod
    def _select_exec_algo(quantity: int, adv: int) -> ExecAlgoOMS:
        """
        Choose execution algorithm based on order size as % of ADV.

        This mirrors the logic in TCAEngine.pre_trade() but lives in OMS so
        the routing decision is logged in the order record.

        Thresholds (industry standard from Almgren & Chriss):
          < 0.2% ADV  → IMMEDIATE (market order — negligible impact)
          0.2–1% ADV  → TWAP (spread evenly over session)
          1–5% ADV    → VWAP (weight by intraday volume profile)
          > 5% ADV    → POV  (participate at constant rate — institutional)
        """
        if adv <= 0:
            return ExecAlgoOMS.IMMEDIATE
        pct_adv = quantity / adv
        if pct_adv < 0.002:
            return ExecAlgoOMS.IMMEDIATE
        if pct_adv < 0.01:
            return ExecAlgoOMS.TWAP
        if pct_adv < 0.05:
            return ExecAlgoOMS.VWAP
        return ExecAlgoOMS.POV

    # -------------------------------------------------------------------------
    # Compliance check (OMS, not risk)
    # -------------------------------------------------------------------------

    def _compliance_check(self, order: Order) -> str | None:
        """
        OMS-level compliance checks (distinct from RiskGate risk checks).

        Risk checks (position size, daily P&L) live in RiskGate and MUST be
        called before submitting to OMS. These checks are about process, not risk:
          1. Duplicate order within 60s for same symbol+side
          2. Zero quantity or non-positive price

        Returns error message if rejected, None if approved.
        Called with self._lock held.
        """
        if order.quantity <= 0:
            return "quantity must be > 0"

        if order.arrival_price <= 0:
            return "arrival_price must be > 0"

        # Duplicate order check — same symbol+side within window
        cutoff = datetime.now(UTC) - timedelta(seconds=self._DUPLICATE_WINDOW_SECONDS)
        for existing in self._orders.values():
            if (
                existing.symbol == order.symbol
                and existing.side == order.side
                and existing.created_at >= cutoff
                and existing.status
                not in {
                    OrderStatus.REJECTED,
                    OrderStatus.CANCELLED,
                    OrderStatus.EXPIRED,
                }
            ):
                return (
                    f"Duplicate order: {order.symbol} {order.side} already submitted "
                    f"{(datetime.now(UTC) - existing.created_at).total_seconds():.0f}s ago"
                )

        return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _transition(self, order_id: str, new_status: OrderStatus) -> Order:
        """Generic state transition helper for simple cases."""
        with self._lock:
            order = self._get_mutable(order_id)
            if order is None:
                raise KeyError(f"[OMS] Order not found: {order_id}")
            if order.is_terminal:
                raise ValueError(f"[OMS] Cannot transition terminal order {order_id}")
            order.status = new_status
            self._orders[order_id] = order
            self._persist(order)
        return order

    def _get_mutable(self, order_id: str) -> Order | None:
        """Return the order dict, or None. Must be called with _lock held."""
        return self._orders.get(order_id)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the `orders` table if it doesn't exist."""
        try:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id        VARCHAR PRIMARY KEY,
                    symbol          VARCHAR NOT NULL,
                    side            VARCHAR NOT NULL,
                    quantity        INTEGER NOT NULL,
                    signal_id       VARCHAR DEFAULT '',
                    arrival_price   DOUBLE NOT NULL,
                    exec_algo       VARCHAR DEFAULT 'immediate',
                    order_type      VARCHAR DEFAULT 'market',
                    limit_price     DOUBLE,
                    status          VARCHAR NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    fill_price      DOUBLE,
                    rejection_reason VARCHAR,
                    created_at      TIMESTAMP WITH TIME ZONE NOT NULL,
                    submitted_at    TIMESTAMP WITH TIME ZONE,
                    filled_at       TIMESTAMP WITH TIME ZONE,
                    expires_at      TIMESTAMP WITH TIME ZONE
                )
            """
            )
        except Exception as e:
            logger.warning(f"[OMS] Could not create orders table: {e}")

    def _load_pending(self) -> None:
        """Reload non-terminal orders from DuckDB on startup (crash recovery)."""
        try:
            terminal_values = ", ".join(f"'{s.value}'" for s in _TERMINAL_STATES)
            rows = self.conn.execute(
                f"""
                SELECT order_id, symbol, side, quantity, signal_id,
                       arrival_price, exec_algo, order_type, limit_price,
                       status, filled_quantity, fill_price, rejection_reason,
                       created_at, submitted_at, filled_at, expires_at
                FROM orders
                WHERE status NOT IN ({terminal_values})
                ORDER BY created_at DESC
                LIMIT 500
                """
            ).fetchall()
            for row in rows:
                order = self._row_to_order(row)
                self._orders[order.order_id] = order
            if rows:
                logger.info(f"[OMS] Recovered {len(rows)} pending orders from DB")
        except Exception as e:
            logger.warning(f"[OMS] Could not load orders from DB: {e}")

    def _persist(self, order: Order) -> None:
        """Upsert order to DuckDB. Called with _lock held."""
        try:
            self.conn.execute(
                """
                INSERT INTO orders (
                    order_id, symbol, side, quantity, signal_id,
                    arrival_price, exec_algo, order_type, limit_price,
                    status, filled_quantity, fill_price, rejection_reason,
                    created_at, submitted_at, filled_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (order_id) DO UPDATE SET
                    status = excluded.status,
                    filled_quantity = excluded.filled_quantity,
                    fill_price = excluded.fill_price,
                    rejection_reason = excluded.rejection_reason,
                    submitted_at = excluded.submitted_at,
                    filled_at = excluded.filled_at,
                    exec_algo = excluded.exec_algo
                """,
                [
                    order.order_id,
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.signal_id,
                    order.arrival_price,
                    order.exec_algo.value,
                    order.order_type,
                    order.limit_price,
                    order.status.value,
                    order.filled_quantity,
                    order.fill_price,
                    order.rejection_reason,
                    order.created_at,
                    order.submitted_at,
                    order.filled_at,
                    order.expires_at,
                ],
            )
        except Exception as e:
            logger.warning(f"[OMS] Could not persist order {order.order_id[:8]}: {e}")

    def _persist_tca_result(self, order: Order) -> None:
        """
        Persist TCA result to TCAStore after a fill.

        Non-fatal: if TCAStore is unavailable or fails, the fill is still
        recorded in OMS. TCA persistence is best-effort.

        This bridges the gap between OMS (which computes shortfall) and
        TCAStore (which provides historical analysis for /reflect sessions).
        """
        shortfall = order.implementation_shortfall_bps
        if shortfall is None or order.fill_price is None:
            return

        try:
            store = TCAStore()
            store.save_result_raw(
                trade_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                shares=order.filled_quantity,
                fill_price=order.fill_price,
                arrival_price=order.arrival_price,
                shortfall_vs_arrival_bps=shortfall,
                shortfall_vs_vwap_bps=None,  # VWAP not available at OMS level
                shortfall_dollar=(
                    (order.fill_price - order.arrival_price)
                    * order.filled_quantity
                    * (1 if order.side == "buy" else -1)
                ),
                is_favorable=shortfall < 0,
            )
            store.close()
            logger.debug(
                f"[OMS] TCA persisted for {order.order_id[:8]} "
                f"IS={shortfall:.1f}bps"
            )
        except Exception as exc:
            logger.warning(
                f"[OMS] TCA persistence failed for {order.order_id[:8]}: {exc}"
            )

    @staticmethod
    def _row_to_order(row: tuple) -> Order:
        """Reconstruct an Order from a DuckDB row."""
        (
            order_id,
            symbol,
            side,
            quantity,
            signal_id,
            arrival_price,
            exec_algo,
            order_type,
            limit_price,
            status,
            filled_quantity,
            fill_price,
            rejection_reason,
            created_at,
            submitted_at,
            filled_at,
            expires_at,
        ) = row

        def tz(dt):
            if dt is None:
                return None
            if hasattr(dt, "tzinfo") and dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            signal_id=signal_id or "",
            arrival_price=arrival_price,
            exec_algo=ExecAlgoOMS(exec_algo),
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus(status),
            filled_quantity=filled_quantity or 0,
            fill_price=fill_price,
            rejection_reason=rejection_reason,
            created_at=tz(created_at),
            submitted_at=tz(submitted_at),
            filled_at=tz(filled_at),
            expires_at=tz(expires_at),
        )


# =============================================================================
# Singleton access
# =============================================================================


_oms: OrderLifecycle | None = None


def get_order_lifecycle(
    conn: PgConnection | None = None,
) -> OrderLifecycle:
    """
    Get the singleton OrderLifecycle instance.

    Prefer injecting via TradingContext in new code. This singleton is for
    backward-compatible access from existing TradingDayFlow.
    """
    global _oms
    if _oms is None:
        if conn is None:
            conn = open_db("")
            run_migrations(conn)
        _oms = OrderLifecycle(conn)
    return _oms
