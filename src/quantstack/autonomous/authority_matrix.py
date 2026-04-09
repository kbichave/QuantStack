# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Decision authority matrix (P15) — ceilings and rate limits for autonomous actions.

Every autonomous decision passes through the AuthorityGate before execution.
The gate enforces:
  - Maximum value per decision (e.g., max 5% of portfolio for a new position)
  - Maximum decisions per day per type (rate limiting)
  - Confirmation requirements for high-impact actions

This is a *separate* layer from the risk gate. The risk gate validates that a
trade is safe to execute. The authority gate validates that the *system* is
authorized to make this class of decision at this frequency.

Design choice: ceilings are hardcoded defaults, not DB-driven. Changing
authority levels is a code change that goes through review — not something
the autonomous system can modify on its own. This is intentional: the system
should not be able to grant itself more authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from quantstack.db import pg_conn


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class DecisionType(str, Enum):
    """Categories of autonomous decisions subject to authority ceilings."""

    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"
    PROMOTE_STRATEGY = "promote_strategy"
    DEMOTE_STRATEGY = "demote_strategy"
    ADJUST_WEIGHT = "adjust_weight"
    EMERGENCY_HALT = "emergency_halt"


@dataclass(frozen=True)
class AuthorityCeiling:
    """Ceiling for a single decision type."""

    decision_type: DecisionType
    max_value: float  # max value as fraction of portfolio (for position sizing)
    max_per_day: int
    requires_confirmation: bool


# ---------------------------------------------------------------------------
# Default ceilings — change these via code review, not runtime config
# ---------------------------------------------------------------------------

DEFAULT_CEILINGS: dict[DecisionType, AuthorityCeiling] = {
    DecisionType.OPEN_POSITION: AuthorityCeiling(
        decision_type=DecisionType.OPEN_POSITION,
        max_value=0.05,  # 5% of portfolio
        max_per_day=3,
        requires_confirmation=False,
    ),
    DecisionType.CLOSE_POSITION: AuthorityCeiling(
        decision_type=DecisionType.CLOSE_POSITION,
        max_value=1.0,  # 100% — can close any position
        max_per_day=10,
        requires_confirmation=False,
    ),
    DecisionType.PROMOTE_STRATEGY: AuthorityCeiling(
        decision_type=DecisionType.PROMOTE_STRATEGY,
        max_value=0.0,  # not value-based
        max_per_day=1,
        requires_confirmation=False,
    ),
    DecisionType.DEMOTE_STRATEGY: AuthorityCeiling(
        decision_type=DecisionType.DEMOTE_STRATEGY,
        max_value=0.0,
        max_per_day=2,
        requires_confirmation=False,
    ),
    DecisionType.ADJUST_WEIGHT: AuthorityCeiling(
        decision_type=DecisionType.ADJUST_WEIGHT,
        max_value=0.0,
        max_per_day=5,
        requires_confirmation=False,
    ),
    DecisionType.EMERGENCY_HALT: AuthorityCeiling(
        decision_type=DecisionType.EMERGENCY_HALT,
        max_value=0.0,
        max_per_day=1,
        requires_confirmation=True,
    ),
}


# ---------------------------------------------------------------------------
# Authority gate
# ---------------------------------------------------------------------------


class AuthorityGate:
    """Enforces decision authority ceilings for the autonomous system.

    Usage::

        gate = AuthorityGate()
        allowed, reason = gate.check_authority(
            DecisionType.OPEN_POSITION, value=0.03, agent_name="entry_scanner"
        )
        if allowed:
            gate.record_decision(DecisionType.OPEN_POSITION, 0.03, "entry_scanner")
            # proceed with trade
        else:
            logger.warning(f"Authority denied: {reason}")
    """

    def __init__(
        self,
        ceilings: dict[DecisionType, AuthorityCeiling] | None = None,
    ) -> None:
        self._ceilings = ceilings or DEFAULT_CEILINGS

    def check_authority(
        self,
        decision_type: DecisionType,
        value: float = 0.0,
        agent_name: str = "unknown",
    ) -> tuple[bool, str]:
        """Check whether a decision is within the authority ceiling.

        Returns:
            (allowed, reason) — allowed is True if the decision may proceed,
            reason explains the denial if allowed is False.
        """
        ceiling = self._ceilings.get(decision_type)
        if ceiling is None:
            return False, f"no ceiling defined for {decision_type.value}"

        # Value check (for position-sizing decisions)
        if ceiling.max_value > 0 and value > ceiling.max_value:
            return (
                False,
                f"{decision_type.value}: value {value:.2%} exceeds ceiling "
                f"{ceiling.max_value:.2%}",
            )

        # Rate limit check
        usage = self.daily_usage(decision_type)
        if usage >= ceiling.max_per_day:
            return (
                False,
                f"{decision_type.value}: daily limit reached "
                f"({usage}/{ceiling.max_per_day})",
            )

        # Confirmation check
        if ceiling.requires_confirmation:
            return (
                False,
                f"{decision_type.value}: requires human confirmation "
                f"(agent={agent_name})",
            )

        return True, "authorized"

    def record_decision(
        self,
        decision_type: DecisionType,
        value: float = 0.0,
        agent_name: str = "unknown",
    ) -> None:
        """Log a decision to the authority_decisions table for rate limiting."""
        try:
            with pg_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO authority_decisions
                        (decision_type, value, agent_name, decided_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    [decision_type.value, value, agent_name],
                )
            logger.info(
                f"[AUTHORITY] Recorded {decision_type.value} "
                f"(value={value:.4f}, agent={agent_name})"
            )
        except Exception as exc:
            # Recording failure should not block the decision — log and continue.
            # The next check_authority call will still work; it just won't see
            # this decision in the count, which is a conservative failure mode
            # (allows one extra decision rather than blocking).
            logger.error(f"[AUTHORITY] Failed to record decision: {exc}")

    def daily_usage(self, decision_type: DecisionType) -> int:
        """Count how many decisions of this type have been made today (UTC)."""
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM authority_decisions
                    WHERE decision_type = %s
                      AND decided_at::date = CURRENT_DATE
                    """,
                    [decision_type.value],
                ).fetchone()
                return int(row["cnt"]) if row else 0
        except Exception as exc:
            # If we can't read usage, return a high number to be conservative
            # (deny rather than allow when uncertain).
            logger.error(f"[AUTHORITY] Failed to read daily usage: {exc}")
            return 999
