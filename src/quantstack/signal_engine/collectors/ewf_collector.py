"""
EWF Elliott Wave signal collector.

Reads the latest EWF chart analysis from the ``ewf_chart_analyses`` table
and returns a flat dict of ``ewf_``-prefixed fields for the SignalBrief.

Design invariants:
- Never raises. Returns ``{}`` on any failure (timeout, DB error, no data).
- Returns ``{}`` when no fresh analysis exists within the TTL window —
  this is normal behavior (before first scraper run, over weekends, after TTL).
- The ``store`` parameter is accepted for API compatibility but unused;
  this collector queries PostgreSQL directly via ``pg_conn()``.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import pg_conn
from quantstack.signal_engine.staleness import check_freshness

_COLLECTOR_TIMEOUT = 10.0  # seconds

_TTL_SQL = """
SELECT
    id, symbol, timeframe, analyzed_at,
    bias, turning_signal, wave_position, wave_degree, current_wave_label,
    key_levels, blue_box_active, blue_box_zone,
    confidence, summary, projected_path
FROM ewf_chart_analyses
WHERE symbol = %s
  AND (
       (timeframe IN ('1h_premarket','1h_midday') AND analyzed_at > NOW() - INTERVAL '4 hours')
    OR (timeframe = '4h'        AND analyzed_at > NOW() - INTERVAL '6 hours')
    OR (timeframe = 'daily'     AND analyzed_at > NOW() - INTERVAL '26 hours')
    OR (timeframe = 'weekly'    AND analyzed_at > NOW() - INTERVAL '8 days')
    OR (timeframe = 'blue_box'  AND analyzed_at > NOW() - INTERVAL '24 hours')
  )
ORDER BY analyzed_at DESC
LIMIT 1
"""


async def collect_ewf(symbol: str, store: Any) -> dict[str, Any]:
    """Fetch the latest EWF Elliott Wave analysis for a symbol.

    Returns a flat dict of EWF signal fields if a fresh analysis exists
    within the TTL window. Returns {} (neutral) if no fresh data exists,
    if EWF has not published an analysis for this symbol yet, or if any
    error occurs. Never raises.
    """
    if not check_freshness(symbol, "ewf_forecasts", max_days=7):
        return {}
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_ewf_sync, symbol),
            timeout=_COLLECTOR_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("[ewf_collector] %s: timed out after %.0fs", symbol, _COLLECTOR_TIMEOUT)
        return {}
    except Exception as exc:
        logger.warning("[ewf_collector] %s: %s", symbol, exc)
        return {}


def _collect_ewf_sync(symbol: str) -> dict[str, Any]:
    """Synchronous collector body — called via asyncio.to_thread."""
    with pg_conn() as conn:
        conn.execute(_TTL_SQL, (symbol,))
        row = conn.fetchone()

    if row is None:
        return {}

    # Unpack positional columns matching the SELECT order
    (
        _id, _symbol, timeframe, analyzed_at,
        bias, turning_signal, wave_position, wave_degree, current_wave_label,
        key_levels_raw, blue_box_active, blue_box_zone_raw,
        confidence, summary, projected_path,
    ) = row

    # Parse JSONB columns (psycopg2 may return str or dict depending on config)
    key_levels = _parse_jsonb(key_levels_raw)
    blue_box_zone = _parse_jsonb(blue_box_zone_raw)

    # Compute age in hours
    now = datetime.now(timezone.utc)
    if analyzed_at.tzinfo is None:
        analyzed_at = analyzed_at.replace(tzinfo=timezone.utc)
    age_hours = (now - analyzed_at).total_seconds() / 3600

    return {
        "ewf_bias": bias,
        "ewf_turning_signal": turning_signal or "none",
        "ewf_wave_position": wave_position,
        "ewf_wave_degree": wave_degree,
        "ewf_current_wave_label": current_wave_label,
        "ewf_confidence": confidence,
        "ewf_key_support": key_levels.get("support", []) if key_levels else [],
        "ewf_key_resistance": key_levels.get("resistance", []) if key_levels else [],
        "ewf_invalidation_level": key_levels.get("invalidation") if key_levels else None,
        "ewf_target": key_levels.get("target") if key_levels else None,
        "ewf_blue_box_active": bool(blue_box_active),
        "ewf_blue_box_low": blue_box_zone.get("low") if blue_box_zone else None,
        "ewf_blue_box_high": blue_box_zone.get("high") if blue_box_zone else None,
        "ewf_summary": summary,
        "ewf_projected_path": projected_path,
        "ewf_timeframe_used": timeframe,
        "ewf_age_hours": round(age_hours, 2),
    }


def _parse_jsonb(val: Any) -> dict | None:
    """Parse a JSONB column value that may be str, dict, or None."""
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
    return None
