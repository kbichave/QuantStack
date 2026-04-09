# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Position reconciliation (P15) — broker vs. system position comparison.

The broker (Alpaca) is the source of truth for what we actually hold. The system
DB tracks what we *think* we hold. Discrepancies happen due to partial fills,
race conditions between order submission and DB writes, or manual interventions
in the broker UI.

Reconciliation runs periodically (typically once per market close) and:
  1. Fetches positions from Alpaca and from the system DB
  2. Matches by symbol, compares qty and avg_cost
  3. Auto-fixes small discrepancies (<$100) — broker wins
  4. Logs alerts for large discrepancies — do NOT auto-fix

Why $100 tolerance: rounding differences in avg_cost across partial fills
routinely produce sub-$100 mismatches. These are noise, not signal. Anything
larger indicates a real problem (missed fill, stale position, manual trade).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import pg_conn


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class _PositionSnapshot:
    """Normalized position for comparison."""

    symbol: str
    quantity: float
    avg_cost: float
    market_value: float


@dataclass
class ReconciliationResult:
    """Outcome of a broker-vs-system position reconciliation."""

    matches: list[dict[str, Any]] = field(default_factory=list)
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    broker_only: list[dict[str, Any]] = field(default_factory=list)
    system_only: list[dict[str, Any]] = field(default_factory=list)
    total_discrepancy_usd: float = 0.0

    @property
    def is_clean(self) -> bool:
        """True if there are no mismatches and no orphaned positions."""
        return (
            len(self.mismatches) == 0
            and len(self.broker_only) == 0
            and len(self.system_only) == 0
        )


# ---------------------------------------------------------------------------
# Broker position fetch
# ---------------------------------------------------------------------------


def _fetch_broker_positions() -> list[_PositionSnapshot]:
    """Fetch current positions from Alpaca.

    Uses the alpaca-py TradingClient directly rather than going through our
    broker adapter — reconciliation needs to read the broker state independently
    of the execution path to detect drift between the two.
    """
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        logger.warning(
            "[RECONCILE] alpaca-py not installed — returning empty broker positions"
        )
        return []

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")

    if not api_key or not secret_key:
        logger.warning("[RECONCILE] Alpaca credentials not set — skipping broker fetch")
        return []

    try:
        client = TradingClient(api_key, secret_key, paper=paper)
        raw_positions = client.get_all_positions()
        return [
            _PositionSnapshot(
                symbol=str(pos.symbol),
                quantity=float(pos.qty or 0),
                avg_cost=float(pos.avg_entry_price or 0),
                market_value=float(pos.market_value or 0),
            )
            for pos in raw_positions
        ]
    except Exception as exc:
        logger.error(f"[RECONCILE] Failed to fetch broker positions: {exc}")
        return []


def _fetch_system_positions() -> list[_PositionSnapshot]:
    """Fetch current positions from the system database."""
    try:
        with pg_conn() as conn:
            rows = conn.execute(
                """
                SELECT symbol, quantity, avg_cost,
                       (quantity * avg_cost) AS market_value
                FROM positions
                WHERE quantity != 0
                """,
            ).fetchall()
            return [
                _PositionSnapshot(
                    symbol=row["symbol"],
                    quantity=float(row["quantity"]),
                    avg_cost=float(row["avg_cost"]),
                    market_value=float(row["market_value"]),
                )
                for row in rows
            ]
    except Exception as exc:
        logger.error(f"[RECONCILE] Failed to fetch system positions: {exc}")
        return []


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

# Tolerance for matching: positions within this USD difference are considered
# matched (covers rounding in avg_cost across partial fills).
_TOLERANCE_USD = 100.0


def reconcile_positions() -> ReconciliationResult:
    """Compare broker positions against system DB and return a detailed result.

    Matching logic:
      - Match by symbol (case-insensitive)
      - If qty matches and abs(broker_value - system_value) < tolerance: match
      - Otherwise: mismatch with details
      - Positions only in broker or only in system: orphans
    """
    result = ReconciliationResult()

    broker_positions = _fetch_broker_positions()
    system_positions = _fetch_system_positions()

    broker_by_symbol = {p.symbol.upper(): p for p in broker_positions}
    system_by_symbol = {p.symbol.upper(): p for p in system_positions}

    all_symbols = set(broker_by_symbol.keys()) | set(system_by_symbol.keys())

    for symbol in sorted(all_symbols):
        broker_pos = broker_by_symbol.get(symbol)
        system_pos = system_by_symbol.get(symbol)

        if broker_pos and system_pos:
            value_diff = abs(broker_pos.market_value - system_pos.market_value)
            qty_match = abs(broker_pos.quantity - system_pos.quantity) < 0.01

            if qty_match and value_diff < _TOLERANCE_USD:
                result.matches.append(
                    {
                        "symbol": symbol,
                        "broker_qty": broker_pos.quantity,
                        "system_qty": system_pos.quantity,
                        "value_diff_usd": value_diff,
                    }
                )
            else:
                result.mismatches.append(
                    {
                        "symbol": symbol,
                        "broker_qty": broker_pos.quantity,
                        "system_qty": system_pos.quantity,
                        "broker_avg_cost": broker_pos.avg_cost,
                        "system_avg_cost": system_pos.avg_cost,
                        "value_diff_usd": value_diff,
                    }
                )
                result.total_discrepancy_usd += value_diff

        elif broker_pos and not system_pos:
            result.broker_only.append(
                {
                    "symbol": symbol,
                    "broker_qty": broker_pos.quantity,
                    "broker_avg_cost": broker_pos.avg_cost,
                    "market_value": broker_pos.market_value,
                }
            )
            result.total_discrepancy_usd += abs(broker_pos.market_value)

        elif system_pos and not broker_pos:
            result.system_only.append(
                {
                    "symbol": symbol,
                    "system_qty": system_pos.quantity,
                    "system_avg_cost": system_pos.avg_cost,
                    "market_value": system_pos.market_value,
                }
            )
            result.total_discrepancy_usd += abs(system_pos.market_value)

    # Log mismatches as system events
    if not result.is_clean:
        try:
            with pg_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO system_events (event_type, payload, created_at)
                    VALUES ('reconciliation_mismatch', %s, NOW())
                    """,
                    [
                        f'{{"mismatches": {len(result.mismatches)}, '
                        f'"broker_only": {len(result.broker_only)}, '
                        f'"system_only": {len(result.system_only)}, '
                        f'"total_discrepancy_usd": {result.total_discrepancy_usd:.2f}}}'
                    ],
                )
        except Exception as exc:
            logger.error(f"[RECONCILE] Failed to log mismatch event: {exc}")

    logger.info(
        f"[RECONCILE] Result: {len(result.matches)} match, "
        f"{len(result.mismatches)} mismatch, "
        f"{len(result.broker_only)} broker-only, "
        f"{len(result.system_only)} system-only, "
        f"discrepancy=${result.total_discrepancy_usd:.2f}"
    )

    return result


def auto_fix_discrepancies(result: ReconciliationResult) -> list[str]:
    """Auto-fix small discrepancies by trusting the broker as source of truth.

    Rules:
      - Small mismatches (value_diff < $100): update system to match broker
      - Broker-only positions (< $100): insert into system DB
      - Large discrepancies or system-only positions: log alert, do NOT fix

    Returns a list of actions taken (human-readable strings).
    """
    actions: list[str] = []

    # Fix small mismatches
    for mm in result.mismatches:
        if mm["value_diff_usd"] < _TOLERANCE_USD:
            try:
                with pg_conn() as conn:
                    conn.execute(
                        """
                        UPDATE positions
                        SET quantity = %s, avg_cost = %s, updated_at = NOW()
                        WHERE UPPER(symbol) = %s
                        """,
                        [mm["broker_qty"], mm["broker_avg_cost"], mm["symbol"]],
                    )
                action = (
                    f"auto-fixed {mm['symbol']}: "
                    f"qty {mm['system_qty']}→{mm['broker_qty']}, "
                    f"cost ${mm['system_avg_cost']:.2f}→${mm['broker_avg_cost']:.2f}"
                )
                actions.append(action)
                logger.info(f"[RECONCILE] {action}")
            except Exception as exc:
                logger.error(f"[RECONCILE] Failed to auto-fix {mm['symbol']}: {exc}")
        else:
            action = (
                f"ALERT: {mm['symbol']} large mismatch "
                f"(${mm['value_diff_usd']:.2f}) — NOT auto-fixed"
            )
            actions.append(action)
            logger.warning(f"[RECONCILE] {action}")

    # Handle broker-only positions
    for bp in result.broker_only:
        if abs(bp["market_value"]) < _TOLERANCE_USD:
            try:
                with pg_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO positions (symbol, quantity, avg_cost, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (symbol) DO UPDATE
                        SET quantity = EXCLUDED.quantity,
                            avg_cost = EXCLUDED.avg_cost,
                            updated_at = NOW()
                        """,
                        [bp["symbol"], bp["broker_qty"], bp["broker_avg_cost"]],
                    )
                action = (
                    f"auto-inserted broker-only {bp['symbol']}: "
                    f"qty={bp['broker_qty']}"
                )
                actions.append(action)
                logger.info(f"[RECONCILE] {action}")
            except Exception as exc:
                logger.error(
                    f"[RECONCILE] Failed to insert broker-only {bp['symbol']}: {exc}"
                )
        else:
            action = (
                f"ALERT: broker-only {bp['symbol']} "
                f"(${bp['market_value']:.2f}) — NOT auto-inserted"
            )
            actions.append(action)
            logger.warning(f"[RECONCILE] {action}")

    # System-only positions: never auto-fix (could be a pending order)
    for sp in result.system_only:
        action = (
            f"ALERT: system-only {sp['symbol']} "
            f"(qty={sp['system_qty']}) — requires manual review"
        )
        actions.append(action)
        logger.warning(f"[RECONCILE] {action}")

    return actions
