# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
IC-to-Retirement sweep — retire forward_testing strategies with persistently weak IC.

Strategies demoted to forward_testing after IC decay can linger indefinitely.
This sweep runs periodically (e.g. nightly via supervisor) and retires any
forward_testing strategy that has spent >= 30 days in that status AND whose
21-day ICIR has been consistently below 0.3 for the last 30 days of signal_ic
data.

Usage:
    with pg_conn() as conn:
        retired = run_ic_retirement_sweep(conn)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from loguru import logger

from quantstack.db import PgConnection

# A strategy must be in forward_testing for at least this many days before
# the sweep considers retiring it. This gives the strategy time to accumulate
# enough IC data for a fair evaluation.
_MIN_FORWARD_TEST_DAYS = 30

# ICIR threshold: if ALL signal_ic rows in the evaluation window have
# icir_21d below this value, the strategy is retired.
_ICIR_RETIREMENT_THRESHOLD = 0.3

# How many days of signal_ic history to examine.
_IC_LOOKBACK_DAYS = 30

# Horizon filter for signal_ic rows.
_IC_HORIZON_DAYS = 21


def run_ic_retirement_sweep(conn: PgConnection) -> list[str]:
    """Retire forward_testing strategies with persistently weak IC.

    Returns a list of strategy_ids that were retired.

    Algorithm:
        1. Query all strategies with status = 'forward_testing'.
        2. For each, compute days in forward_testing from updated_at.
        3. Skip if < 30 days (not enough time to judge).
        4. Query signal_ic for last 30 days where horizon_days = 21.
        5. Skip if no rows (no IC data to judge).
        6. If ALL rows have icir_21d < 0.3: retire.
        7. Log lifecycle event to loop_events.
    """
    retired: list[str] = []

    rows = conn.execute(
        "SELECT strategy_id, updated_at FROM strategies WHERE status = 'forward_testing'"
    ).fetchall()

    if not rows:
        logger.debug("[IC-Sweep] No forward_testing strategies to evaluate")
        return retired

    now = datetime.now(timezone.utc)

    for strategy_id, updated_at in rows:
        # Compute days in forward_testing.
        # updated_at is set when a strategy transitions to forward_testing.
        if updated_at is None:
            continue

        # Handle both timezone-aware and naive datetimes from the DB.
        if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
            days_in_ft = (now - updated_at).days
        else:
            days_in_ft = (now.replace(tzinfo=None) - updated_at).days

        if days_in_ft < _MIN_FORWARD_TEST_DAYS:
            logger.debug(
                "[IC-Sweep] %s: %d days in forward_testing (< %d) — skipping",
                strategy_id,
                days_in_ft,
                _MIN_FORWARD_TEST_DAYS,
            )
            continue

        # Query signal_ic for the last 30 days, horizon_days = 21.
        ic_rows = conn.execute(
            """
            SELECT icir_21d FROM signal_ic
            WHERE strategy_id = %s
              AND horizon_days = %s
              AND date >= (CURRENT_DATE - %s)
            ORDER BY date DESC
            """,
            [strategy_id, _IC_HORIZON_DAYS, _IC_LOOKBACK_DAYS],
        ).fetchall()

        if not ic_rows:
            logger.debug(
                "[IC-Sweep] %s: no signal_ic rows in last %d days — skipping",
                strategy_id,
                _IC_LOOKBACK_DAYS,
            )
            continue

        # Check if ALL rows have icir_21d < threshold.
        all_below = all(
            row[0] is not None and row[0] < _ICIR_RETIREMENT_THRESHOLD
            for row in ic_rows
        )

        if not all_below:
            logger.debug(
                "[IC-Sweep] %s: %d IC rows, not all below %.2f — keeping",
                strategy_id,
                len(ic_rows),
                _ICIR_RETIREMENT_THRESHOLD,
            )
            continue

        # Retire the strategy.
        conn.execute(
            "UPDATE strategies SET status = 'retired', updated_at = NOW() "
            "WHERE strategy_id = %s AND status = 'forward_testing'",
            [strategy_id],
        )

        # Log lifecycle event.
        event_id = str(uuid.uuid4())
        icir_values = [row[0] for row in ic_rows]
        payload = json.dumps({
            "strategy_id": strategy_id,
            "reason": "ic_decay_retirement",
            "days_in_forward_testing": days_in_ft,
            "ic_rows_evaluated": len(ic_rows),
            "icir_values": icir_values,
            "threshold": _ICIR_RETIREMENT_THRESHOLD,
        })
        conn.execute(
            """
            INSERT INTO loop_events (event_id, event_type, source_loop, payload, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            """,
            [event_id, "strategy_retired", "ic_retirement_sweep", payload],
        )

        retired.append(strategy_id)
        logger.warning(
            "[IC-Sweep] RETIRED %s — %d days in forward_testing, "
            "%d IC rows all below %.2f (values: %s)",
            strategy_id,
            days_in_ft,
            len(ic_rows),
            _ICIR_RETIREMENT_THRESHOLD,
            icir_values,
        )

    conn.commit()

    if retired:
        logger.info("[IC-Sweep] Retired %d strategies: %s", len(retired), retired)
    else:
        logger.debug("[IC-Sweep] No strategies retired this sweep")

    return retired
