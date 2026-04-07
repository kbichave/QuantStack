# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Mandate enforcement gate — sits BEFORE risk_gate in the trading pipeline.

Every proposed trade must pass mandate_check() before reaching the RiskGate.
The mandate is a hard gate: if the CIO mandate says no, the trade is rejected
regardless of what the risk gate would say.

Enforcement order:
  1. mandate_check()  <-- this module (CIO-level governance)
  2. risk_gate.check()  (position-level risk controls)

If no mandate exists (pre-09:30 ET window), the gate approves by default
to avoid blocking early-morning operations.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from quantstack.governance.mandate import get_active_mandate


@dataclass
class MandateVerdict:
    """Result of the mandate enforcement check."""

    approved: bool
    rejection_reason: str | None = None


def _count_todays_entries() -> int:
    """Count new position entries opened today.

    Placeholder: returns 0.  In production, this would query the fills/positions
    table for entries opened on the current date.  Can be patched in tests.
    """
    return 0


def _sum_todays_notional() -> float:
    """Sum notional value of all entries opened today.

    Placeholder: returns 0.0.  In production, this would query the fills table
    for cumulative notional deployed today.  Can be patched in tests.
    """
    return 0.0


def mandate_check(
    symbol: str,
    sector: str,
    side: str,
    notional: float,
    strategy_id: str,
) -> MandateVerdict:
    """Enforce the active daily mandate as a hard gate.

    Checks (in order):
      1. If no mandate (pre-09:30 window) -> approve.
      2. Sector in blocked_sectors -> reject.
      3. Today's entry count >= max_new_positions -> reject.
      4. Today's cumulative notional + proposed > max_daily_notional -> reject.
      5. Strategy directive is "pause" or "exit" -> reject.
      6. All checks pass -> approve.

    Args:
        symbol: Ticker symbol for the proposed trade.
        sector: Sector classification of the symbol.
        side: "buy" or "sell".
        notional: Dollar notional of the proposed trade.
        strategy_id: Strategy identifier to check against directives.

    Returns:
        MandateVerdict with approved=True or approved=False + rejection_reason.
    """
    mandate = get_active_mandate()

    # Pre-mandate window: no mandate yet, allow trades through
    if mandate is None:
        return MandateVerdict(approved=True)

    # 1. Blocked sectors
    sector_lower = sector.lower()
    blocked = [s.lower() for s in mandate.blocked_sectors]
    if "all" in blocked or sector_lower in blocked:
        reason = (
            f"Sector '{sector}' is blocked by today's mandate "
            f"(blocked: {mandate.blocked_sectors})"
        )
        logger.warning("[MANDATE] REJECTED %s %s: %s", side, symbol, reason)
        return MandateVerdict(approved=False, rejection_reason=reason)

    # 2. Max new positions
    todays_entries = _count_todays_entries()
    if todays_entries >= mandate.max_new_positions:
        reason = (
            f"Max new positions reached: {todays_entries} >= "
            f"{mandate.max_new_positions}"
        )
        logger.warning("[MANDATE] REJECTED %s %s: %s", side, symbol, reason)
        return MandateVerdict(approved=False, rejection_reason=reason)

    # 3. Max daily notional
    todays_notional = _sum_todays_notional()
    if todays_notional + notional > mandate.max_daily_notional:
        reason = (
            f"Daily notional would exceed limit: "
            f"${todays_notional:,.0f} + ${notional:,.0f} = "
            f"${todays_notional + notional:,.0f} > "
            f"${mandate.max_daily_notional:,.0f}"
        )
        logger.warning("[MANDATE] REJECTED %s %s: %s", side, symbol, reason)
        return MandateVerdict(approved=False, rejection_reason=reason)

    # 4. Strategy directives
    directive = mandate.strategy_directives.get(
        strategy_id,
        mandate.strategy_directives.get("_all", "active"),
    )
    if directive in ("pause", "exit"):
        reason = (
            f"Strategy '{strategy_id}' directive is '{directive}' — "
            f"no new entries permitted"
        )
        logger.warning("[MANDATE] REJECTED %s %s: %s", side, symbol, reason)
        return MandateVerdict(approved=False, rejection_reason=reason)

    # All checks passed
    logger.debug(
        "[MANDATE] APPROVED %s %s %s ($%.0f) under mandate %s",
        side,
        symbol,
        strategy_id,
        notional,
        mandate.mandate_id,
    )
    return MandateVerdict(approved=True)
