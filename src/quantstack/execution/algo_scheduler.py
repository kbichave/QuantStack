# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Algo Scheduler Core / EMS — parent-child order lifecycle management.

Manages TWAP/VWAP parent orders that are sliced into time-scheduled child
orders.  Each parent tracks aggregate fill state recomputed from its children
(parent-child invariant).  State transitions are validated via explicit
finite-state machines for both parent and child orders.

DB tables: ``algo_parent_orders``, ``algo_child_orders``, ``algo_performance``
(created by ``_migrate_execution_layer_pg`` in ``db.py``).

The actual TWAP/VWAP scheduling logic (child generation, timing, participation
checks) lives in section-08.  This module provides the order model, state
machines, persistence, crash recovery, and the ``AlgoScheduler`` async
orchestrator shell.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from loguru import logger

from quantstack.db import PgConnection


# ---------------------------------------------------------------------------
# Broker protocol — minimal interface needed by the scheduler
# ---------------------------------------------------------------------------


class Broker(Protocol):
    """Minimal broker interface for order cancellation during recovery."""

    async def cancel_order(self, broker_order_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlgoParentOrder:
    """Parent algo order representing the full intent (e.g. buy 1000 SPY via TWAP)."""

    parent_order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    total_quantity: int
    algo_type: str  # "twap" | "vwap"
    start_time: datetime
    end_time: datetime
    arrival_price: float
    max_participation_rate: float = 0.02
    status: str = "pending"
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    cancel_reason: str | None = None


@dataclass
class ChildOrder:
    """A single time-slice of a parent algo order."""

    child_id: str  # "{parent_id}-C{seq}"
    parent_id: str
    scheduled_time: datetime
    target_quantity: int
    filled_quantity: int = 0
    fill_price: float = 0.0
    status: str = "pending"
    attempts: int = 0
    broker_order_id: str | None = None


# ---------------------------------------------------------------------------
# State machines
# ---------------------------------------------------------------------------

_PARENT_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"active"},
    "active": {"completing", "cancelling"},
    "completing": {"completed"},
    "cancelling": {"cancelled"},
}

_CHILD_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"submitted"},
    "submitted": {"partially_filled", "filled", "cancelled", "expired", "rejected"},
    "partially_filled": {"filled"},
}

# Terminal states — no further transitions allowed
_PARENT_TERMINAL = {"completed", "cancelled"}
_CHILD_TERMINAL = {"filled", "cancelled", "expired", "rejected"}


def transition_parent(parent: AlgoParentOrder, new_status: str) -> None:
    """Validate and apply a parent state transition.

    Raises:
        ValueError: If the transition is not allowed by the state machine.
    """
    allowed = _PARENT_TRANSITIONS.get(parent.status, set())
    if new_status not in allowed:
        raise ValueError(
            f"Invalid parent transition: {parent.status!r} -> {new_status!r} "
            f"(allowed: {sorted(allowed) if allowed else 'none — terminal state'})"
        )
    parent.status = new_status


def transition_child(child: ChildOrder, new_status: str) -> None:
    """Validate and apply a child state transition.

    Raises:
        ValueError: If the transition is not allowed by the state machine.
    """
    allowed = _CHILD_TRANSITIONS.get(child.status, set())
    if new_status not in allowed:
        raise ValueError(
            f"Invalid child transition: {child.status!r} -> {new_status!r} "
            f"(allowed: {sorted(allowed) if allowed else 'none — terminal state'})"
        )
    child.status = new_status


# ---------------------------------------------------------------------------
# POV fallback
# ---------------------------------------------------------------------------

_POV_MAX_PARTICIPATION = 0.05


def normalize_algo_type(
    algo_type: str, max_participation_rate: float
) -> tuple[str, float]:
    """Normalize algo type, dispatching POV as VWAP with capped participation.

    Returns:
        (effective_algo_type, effective_max_participation_rate)
    """
    if algo_type.lower() == "pov":
        return "vwap", min(max_participation_rate, _POV_MAX_PARTICIPATION)
    return algo_type.lower(), max_participation_rate


# ---------------------------------------------------------------------------
# Parent-child invariant
# ---------------------------------------------------------------------------


def update_parent_from_children(
    parent: AlgoParentOrder, children: list[ChildOrder]
) -> None:
    """Recompute parent.filled_quantity and parent.avg_fill_price from children.

    avg_fill_price is the VWAP across all filled children:
        sum(child.filled_quantity * child.fill_price) / sum(child.filled_quantity)

    This enforces the parent-child invariant: the parent's fill state is always
    the exact aggregate of its children's fills.
    """
    total_filled = 0
    notional = 0.0
    for child in children:
        if child.filled_quantity > 0:
            total_filled += child.filled_quantity
            notional += child.filled_quantity * child.fill_price

    parent.filled_quantity = total_filled
    parent.avg_fill_price = (notional / total_filled) if total_filled > 0 else 0.0


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------


def persist_parent(conn: PgConnection, parent: AlgoParentOrder) -> None:
    """UPSERT parent order state to algo_parent_orders."""
    conn.execute(
        """
        INSERT INTO algo_parent_orders
            (parent_order_id, symbol, side, total_quantity, algo_type,
             start_time, end_time, arrival_price, max_participation_rate,
             status, filled_quantity, avg_fill_price)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (parent_order_id) DO UPDATE SET
            status = EXCLUDED.status,
            filled_quantity = EXCLUDED.filled_quantity,
            avg_fill_price = EXCLUDED.avg_fill_price,
            max_participation_rate = EXCLUDED.max_participation_rate
        """,
        (
            parent.parent_order_id,
            parent.symbol,
            parent.side,
            parent.total_quantity,
            parent.algo_type,
            parent.start_time,
            parent.end_time,
            parent.arrival_price,
            parent.max_participation_rate,
            parent.status,
            parent.filled_quantity,
            parent.avg_fill_price,
        ),
    )


def persist_child(conn: PgConnection, child: ChildOrder) -> None:
    """UPSERT child order state to algo_child_orders."""
    conn.execute(
        """
        INSERT INTO algo_child_orders
            (child_id, parent_id, scheduled_time, target_quantity,
             filled_quantity, fill_price, status, attempts, broker_order_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (child_id) DO UPDATE SET
            filled_quantity = EXCLUDED.filled_quantity,
            fill_price = EXCLUDED.fill_price,
            status = EXCLUDED.status,
            attempts = EXCLUDED.attempts,
            broker_order_id = EXCLUDED.broker_order_id
        """,
        (
            child.child_id,
            child.parent_id,
            child.scheduled_time,
            child.target_quantity,
            child.filled_quantity,
            child.fill_price,
            child.status,
            child.attempts,
            child.broker_order_id,
        ),
    )


def persist_performance(
    conn: PgConnection,
    parent: AlgoParentOrder,
    children: list[ChildOrder],
) -> None:
    """Insert completion metrics to algo_performance.

    Called once after a parent reaches COMPLETED or CANCELLED.
    Computes implementation shortfall vs. arrival price and child statistics.
    """
    filled_children = [c for c in children if c.status == "filled"]
    failed_children = [c for c in children if c.status in ("cancelled", "expired", "rejected")]

    # Implementation shortfall in bps: (avg_fill - arrival) / arrival * 10000
    # For sells, the sign is inverted (selling lower than arrival is adverse).
    if parent.arrival_price > 0 and parent.avg_fill_price > 0:
        raw_is = (parent.avg_fill_price - parent.arrival_price) / parent.arrival_price
        if parent.side == "sell":
            raw_is = -raw_is
        implementation_shortfall_bps = raw_is * 10_000
    else:
        implementation_shortfall_bps = 0.0

    # First and last fill times from children
    fill_times = [
        c.scheduled_time for c in filled_children if c.scheduled_time is not None
    ]
    first_fill = min(fill_times) if fill_times else None
    last_fill = max(fill_times) if fill_times else None

    conn.execute(
        """
        INSERT INTO algo_performance
            (parent_order_id, symbol, side, algo_type, total_qty, filled_qty,
             arrival_price, avg_fill_price, implementation_shortfall_bps,
             num_children, num_children_filled, num_children_failed,
             max_participation_rate, decision_time, first_fill_time,
             last_fill_time, scheduled_end_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (parent_order_id) DO NOTHING
        """,
        (
            parent.parent_order_id,
            parent.symbol,
            parent.side,
            parent.algo_type,
            parent.total_quantity,
            parent.filled_quantity,
            parent.arrival_price,
            parent.avg_fill_price,
            implementation_shortfall_bps,
            len(children),
            len(filled_children),
            len(failed_children),
            parent.max_participation_rate,
            parent.start_time,
            first_fill,
            last_fill,
            parent.end_time,
        ),
    )


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def startup_recovery(conn: PgConnection, broker: Broker | None = None) -> int:
    """Find ACTIVE/COMPLETING parents and cancel them for safe restart.

    Called synchronously at service startup before the async event loop.
    Returns the number of parents recovered.

    If a broker is provided, attempts to cancel any non-terminal child orders
    that have a broker_order_id.  Broker cancel failures are logged but do not
    prevent the state cleanup from proceeding.
    """
    rows = conn.execute(
        "SELECT parent_order_id FROM algo_parent_orders "
        "WHERE status IN ('active', 'completing')"
    ).fetchall()

    if not rows:
        return 0

    parent_ids = [r[0] for r in rows]
    logger.warning(
        "Crash recovery: cancelling {} in-flight parent orders: {}",
        len(parent_ids),
        parent_ids,
    )

    for pid in parent_ids:
        # Cancel non-terminal children
        child_rows = conn.execute(
            "SELECT child_id, broker_order_id FROM algo_child_orders "
            "WHERE parent_id = %s AND status NOT IN ('filled', 'cancelled', 'expired', 'rejected')",
            (pid,),
        ).fetchall()

        for child_row in child_rows:
            child_id = child_row[0]
            broker_order_id = child_row[1]

            # Attempt broker cancellation if possible
            if broker is not None and broker_order_id is not None:
                try:
                    # startup_recovery is sync; broker.cancel_order is async.
                    # Use a new event loop if none is running, or schedule on the
                    # existing one.  At startup, typically no loop is running yet.
                    import asyncio as _asyncio

                    try:
                        loop = _asyncio.get_running_loop()
                        loop.create_task(broker.cancel_order(broker_order_id))
                    except RuntimeError:
                        _asyncio.run(broker.cancel_order(broker_order_id))
                except Exception:
                    logger.opt(exception=True).warning(
                        "Crash recovery: failed to cancel broker order {} for child {}",
                        broker_order_id,
                        child_id,
                    )

            conn.execute(
                "UPDATE algo_child_orders SET status = 'cancelled' "
                "WHERE child_id = %s",
                (child_id,),
            )

        # Cancel the parent
        conn.execute(
            "UPDATE algo_parent_orders SET status = 'cancelled' "
            "WHERE parent_order_id = %s",
            (pid,),
        )

    return len(parent_ids)


# ---------------------------------------------------------------------------
# AlgoScheduler — async orchestrator
# ---------------------------------------------------------------------------


class AlgoScheduler:
    """Async EMS orchestrating parent algo orders and their child slices.

    The ``run()`` loop is a placeholder — section-08 fills in the actual
    TWAP/VWAP child scheduling, submission, and fill polling.
    """

    def __init__(self, broker: Broker, conn: PgConnection) -> None:
        self._broker = broker
        self._conn = conn
        self._active_parents: dict[str, AlgoParentOrder] = {}
        self._children: dict[str, list[ChildOrder]] = {}

    async def startup_recovery(self) -> int:
        """Run crash recovery (delegates to module-level function)."""
        return startup_recovery(self._conn, self._broker)

    async def submit_parent(
        self, parent: AlgoParentOrder, children: list[ChildOrder]
    ) -> None:
        """Register a parent and its pre-generated children for execution.

        Normalizes POV → VWAP, persists to DB, and adds to active tracking.
        """
        parent.algo_type, parent.max_participation_rate = normalize_algo_type(
            parent.algo_type, parent.max_participation_rate
        )

        persist_parent(self._conn, parent)
        for child in children:
            persist_child(self._conn, child)

        self._active_parents[parent.parent_order_id] = parent
        self._children[parent.parent_order_id] = children

        logger.info(
            "Submitted parent order {} ({} {} {} via {}, {} children)",
            parent.parent_order_id,
            parent.side,
            parent.total_quantity,
            parent.symbol,
            parent.algo_type,
            len(children),
        )

    async def cancel_parent(self, parent_order_id: str, reason: str) -> None:
        """Request cancellation of a single parent order."""
        parent = self._active_parents.get(parent_order_id)
        if parent is None:
            logger.warning("cancel_parent: {} not found in active set", parent_order_id)
            return

        parent.cancel_reason = reason
        transition_parent(parent, "cancelling")
        persist_parent(self._conn, parent)

        children = self._children.get(parent_order_id, [])
        all_terminal = True
        for child in children:
            if child.status not in _CHILD_TERMINAL:
                if child.broker_order_id is not None:
                    try:
                        await self._broker.cancel_order(child.broker_order_id)
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Failed to cancel broker order {} for child {}",
                            child.broker_order_id,
                            child.child_id,
                        )
                transition_child(child, "cancelled")
                persist_child(self._conn, child)
            if child.status not in _CHILD_TERMINAL:
                all_terminal = False

        if all_terminal:
            transition_parent(parent, "cancelled")
            persist_parent(self._conn, parent)
            update_parent_from_children(parent, children)
            persist_performance(self._conn, parent, children)
            del self._active_parents[parent_order_id]
            del self._children[parent_order_id]

    async def cancel_all(self, reason: str) -> None:
        """Cancel all active parent orders (e.g. kill switch, EOD)."""
        parent_ids = list(self._active_parents.keys())
        for pid in parent_ids:
            await self.cancel_parent(pid, reason)

    async def run(self) -> None:
        """Main scheduling loop — placeholder for section-08.

        Section-08 will implement:
        - Periodic tick checking scheduled_time vs now
        - Child submission to broker
        - Fill polling and state updates
        - Completion detection
        """
        logger.info("AlgoScheduler.run() started (placeholder — section-08 implements scheduling)")
        # Placeholder: yield control so tests can drive the scheduler
        # without blocking.  Real implementation will loop with sleep intervals.
        await asyncio.sleep(0)
