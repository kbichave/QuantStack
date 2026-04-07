# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Daily mandate data model and persistence.

The CIO agent produces a DailyMandate once per day before market open.
The mandate controls which sectors, strategies, and position limits are
active for the session.  If no mandate exists by 09:30 ET, a conservative
default is applied (zero new positions, all strategies paused).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from loguru import logger

from quantstack.db import db_conn

# 09:30 ET expressed as UTC offset.  ET = UTC-4 (EDT) or UTC-5 (EST).
# We use 13:30 UTC as a safe default (covers EDT; during EST this is 14:30
# which is still before market open — conservative direction).
_MANDATE_CUTOFF_UTC_HOUR = 13
_MANDATE_CUTOFF_UTC_MINUTE = 30


@dataclass
class DailyMandate:
    """CIO-level daily directive that governs all trading activity."""

    mandate_id: str
    date: str  # ISO format YYYY-MM-DD
    regime_assessment: str
    allowed_sectors: list[str] = field(default_factory=list)
    blocked_sectors: list[str] = field(default_factory=list)
    max_new_positions: int = 0
    max_daily_notional: float = 0.0
    strategy_directives: dict[str, str] = field(default_factory=dict)
    risk_overrides: dict = field(default_factory=dict)
    focus_areas: list[str] = field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _default_mandate(today: str) -> DailyMandate:
    """Conservative fallback mandate when CIO agent is unavailable.

    Zero new positions, all strategies paused, all sectors blocked.
    This ensures no capital is deployed without explicit CIO approval.
    Does NOT liquidate existing positions (directives are "pause", not "exit").
    """
    return DailyMandate(
        mandate_id=f"default-{today}",
        date=today,
        regime_assessment="unknown",
        allowed_sectors=[],
        blocked_sectors=["all"],
        max_new_positions=0,
        max_daily_notional=0.0,
        strategy_directives={"_all": "pause"},
        risk_overrides={},
        focus_areas=[],
        reasoning="Conservative default — CIO unavailable",
        created_at=datetime.now(timezone.utc),
    )


def persist_mandate(mandate: DailyMandate) -> None:
    """Write a mandate to the daily_mandates table.

    Uses ON CONFLICT DO UPDATE so re-running the CIO agent for the same
    date overwrites the previous mandate (last writer wins).
    """
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO daily_mandates
                (mandate_id, date, regime_assessment, allowed_sectors,
                 blocked_sectors, max_new_positions, max_daily_notional,
                 strategy_directives, risk_overrides, focus_areas,
                 reasoning, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                mandate_id = EXCLUDED.mandate_id,
                regime_assessment = EXCLUDED.regime_assessment,
                allowed_sectors = EXCLUDED.allowed_sectors,
                blocked_sectors = EXCLUDED.blocked_sectors,
                max_new_positions = EXCLUDED.max_new_positions,
                max_daily_notional = EXCLUDED.max_daily_notional,
                strategy_directives = EXCLUDED.strategy_directives,
                risk_overrides = EXCLUDED.risk_overrides,
                focus_areas = EXCLUDED.focus_areas,
                reasoning = EXCLUDED.reasoning,
                created_at = EXCLUDED.created_at
            """,
            [
                mandate.mandate_id,
                mandate.date,
                mandate.regime_assessment,
                json.dumps(mandate.allowed_sectors),
                json.dumps(mandate.blocked_sectors),
                mandate.max_new_positions,
                mandate.max_daily_notional,
                json.dumps(mandate.strategy_directives),
                json.dumps(mandate.risk_overrides),
                json.dumps(mandate.focus_areas),
                mandate.reasoning,
                mandate.created_at,
            ],
        )
    logger.info(
        "[GOVERNANCE] Persisted mandate %s for %s (max_pos=%d, notional=%.0f)",
        mandate.mandate_id,
        mandate.date,
        mandate.max_new_positions,
        mandate.max_daily_notional,
    )


def get_active_mandate(today: str | None = None) -> DailyMandate | None:
    """Load today's mandate from the database.

    Behaviour:
      - Row exists for today -> return it.
      - No row AND current time is past 09:30 ET -> return conservative default.
        (CIO should have run by now; absence means something is wrong.)
      - No row AND current time is before 09:30 ET -> return None.
        (Pre-mandate window; callers should allow trades through.)

    Args:
        today: ISO date string (YYYY-MM-DD).  Defaults to current UTC date.

    Returns:
        DailyMandate if one is active, None during pre-mandate window.
    """
    if today is None:
        today = date.today().isoformat()

    try:
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT mandate_id, date, regime_assessment, allowed_sectors,
                       blocked_sectors, max_new_positions, max_daily_notional,
                       strategy_directives, risk_overrides, focus_areas,
                       reasoning, created_at
                FROM daily_mandates
                WHERE date = %s
                """,
                [today],
            ).fetchone()
    except Exception as exc:
        logger.warning("[GOVERNANCE] Failed to load mandate: %s", exc)
        row = None

    if row is not None:
        return _row_to_mandate(row)

    # No row — check whether we're past the mandate cutoff
    now_utc = datetime.now(timezone.utc)
    past_cutoff = (
        now_utc.hour > _MANDATE_CUTOFF_UTC_HOUR
        or (
            now_utc.hour == _MANDATE_CUTOFF_UTC_HOUR
            and now_utc.minute >= _MANDATE_CUTOFF_UTC_MINUTE
        )
    )

    if past_cutoff:
        logger.warning(
            "[GOVERNANCE] No mandate for %s and past 09:30 ET — using conservative default",
            today,
        )
        return _default_mandate(today)

    # Pre-mandate window: no mandate yet, callers should allow trades
    return None


def _row_to_mandate(row) -> DailyMandate:
    """Convert a database row to a DailyMandate dataclass."""

    def _parse_json(val, default):
        if val is None:
            return default
        if isinstance(val, str):
            return json.loads(val)
        return val  # already parsed (JSONB driver)

    return DailyMandate(
        mandate_id=row[0],
        date=row[1],
        regime_assessment=row[2] or "unknown",
        allowed_sectors=_parse_json(row[3], []),
        blocked_sectors=_parse_json(row[4], []),
        max_new_positions=row[5] or 0,
        max_daily_notional=float(row[6] or 0.0),
        strategy_directives=_parse_json(row[7], {}),
        risk_overrides=_parse_json(row[8], {}),
        focus_areas=_parse_json(row[9], []),
        reasoning=row[10] or "",
        created_at=row[11] or datetime.now(timezone.utc),
    )
