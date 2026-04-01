# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Atomic strategy status transitions.

The strategy lifecycle FSM:

    draft ──> forward_testing ──> live ──> retired
      │              │              │
      └──> retired   └──> retired   └──> forward_testing (demotion)

Multiple processes (Factory loop, /review skill, AutoPromoter) can attempt
status changes concurrently.  Since PostgreSQL uses MVCC, the risk is TOCTOU
within the async MCP handler: between reading the current status and writing
the new one, another coroutine may interleave.

Solution: conditional UPDATE (compare-and-swap).  The UPDATE's WHERE clause
checks ``status = expected_status``.  If another coroutine already changed
the status, the WHERE fails and zero rows are affected.  The caller sees
``False`` and can retry or log.

Usage:
    from quantstack.coordination.strategy_lock import StrategyStatusLock

    lock = StrategyStatusLock(conn, event_bus)
    ok = lock.transition("strat_abc", expected_status="draft",
                         new_status="forward_testing",
                         reason="OOS Sharpe 0.72, overfit ratio 1.4")
    if not ok:
        logger.warning("Status already changed by another process")
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from quantstack.db import PgConnection

from quantstack.coordination.event_bus import Event, EventBus, EventType


# Valid FSM transitions
VALID_TRANSITIONS: set[tuple[str, str]] = {
    ("draft", "forward_testing"),
    ("draft", "retired"),
    ("forward_testing", "live"),
    ("forward_testing", "retired"),
    ("live", "retired"),
    ("live", "forward_testing"),  # demotion on CRITICAL degradation
}


class StrategyStatusLock:
    """
    Atomic strategy status transitions with event publishing.

    All transitions use a conditional UPDATE (CAS) to prevent race conditions.
    On success, an event is published to the event bus so other loops are notified.
    """

    def __init__(
        self,
        conn: PgConnection,
        event_bus: EventBus | None = None,
    ) -> None:
        self._conn = conn
        self._bus = event_bus

    def transition(
        self,
        strategy_id: str,
        expected_status: str,
        new_status: str,
        reason: str,
    ) -> bool:
        """
        Atomically transition a strategy's status.

        Args:
            strategy_id: The strategy to transition.
            expected_status: The status we expect the strategy to be in.
                If the actual status differs, the transition fails (returns False).
            new_status: The target status.
            reason: Human-readable reason for the transition (logged + event payload).

        Returns:
            True if the transition succeeded, False if the status had already changed.

        Raises:
            ValueError: If the transition is not in VALID_TRANSITIONS.
        """
        if (expected_status, new_status) not in VALID_TRANSITIONS:
            raise ValueError(
                f"Invalid transition: {expected_status!r} -> {new_status!r}. "
                f"Valid transitions: {VALID_TRANSITIONS}"
            )

        # Read current status first to detect CAS failure
        row = self._conn.execute(
            "SELECT status FROM strategies WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()

        if row is None:
            logger.warning(f"[StrategyLock] Strategy {strategy_id} not found")
            return False

        actual_status = row[0]
        if actual_status != expected_status:
            logger.info(
                f"[StrategyLock] CAS failed for {strategy_id}: "
                f"expected {expected_status!r}, found {actual_status!r}, "
                f"wanted {new_status!r}"
            )
            return False

        # Status matches — safe to update
        self._conn.execute(
            """
            UPDATE strategies
            SET status = ?, updated_at = ?
            WHERE strategy_id = ? AND status = ?
            """,
            [new_status, datetime.now(timezone.utc), strategy_id, expected_status],
        )

        logger.info(
            f"[StrategyLock] {strategy_id}: {expected_status} -> {new_status} "
            f"(reason: {reason})"
        )

        # Publish event
        if self._bus:
            if new_status == "retired":
                event_type = EventType.STRATEGY_RETIRED
            elif new_status == "forward_testing" and expected_status == "live":
                event_type = EventType.STRATEGY_DEMOTED
            else:
                event_type = EventType.STRATEGY_PROMOTED

            self._bus.publish(
                Event(
                    event_type=event_type,
                    source_loop="strategy_lock",
                    payload={
                        "strategy_id": strategy_id,
                        "from_status": expected_status,
                        "to_status": new_status,
                        "reason": reason,
                    },
                )
            )

        return True

    def get_status(self, strategy_id: str) -> str | None:
        """Read the current status of a strategy.  Returns None if not found."""
        row = self._conn.execute(
            "SELECT status FROM strategies WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()
        return row[0] if row else None
