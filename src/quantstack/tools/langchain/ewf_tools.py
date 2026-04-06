"""EWF Elliott Wave Forecast analysis tools for LangGraph agents."""

import json
from datetime import date, datetime, timezone
from typing import Annotated, Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import Field

from quantstack.db import pg_conn

_TTL_MAP: dict[str, str] = {
    "1h_premarket": "4 hours",
    "1h_midday": "4 hours",
    "4h": "6 hours",
    "daily": "26 hours",
    "weekly": "8 days",
    "blue_box": "24 hours",
}

_MULTI_TTL_SQL = """
SELECT symbol, timeframe, fetched_at, analyzed_at,
       bias, wave_position, wave_degree, current_wave_label,
       key_levels, blue_box_active, blue_box_zone,
       confidence, invalidation_rule_violated, analyst_notes, summary
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
"""

_SINGLE_TF_SQL = """
SELECT symbol, timeframe, fetched_at, analyzed_at,
       bias, wave_position, wave_degree, current_wave_label,
       key_levels, blue_box_active, blue_box_zone,
       confidence, invalidation_rule_violated, analyst_notes, summary
FROM ewf_chart_analyses
WHERE symbol = %s AND timeframe = %s
  AND analyzed_at > NOW() - INTERVAL '{ttl}'
ORDER BY analyzed_at DESC
LIMIT 1
"""


def _parse_jsonb(val: Any) -> dict | None:
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


def _row_to_dict(row: tuple) -> dict:
    (
        symbol, timeframe, fetched_at, analyzed_at,
        bias, wave_position, wave_degree, current_wave_label,
        key_levels_raw, blue_box_active, blue_box_zone_raw,
        confidence, invalidation_rule_violated, analyst_notes, summary,
    ) = row

    key_levels = _parse_jsonb(key_levels_raw)
    blue_box_zone = _parse_jsonb(blue_box_zone_raw)

    now = datetime.now(timezone.utc)
    aa = analyzed_at
    if aa and aa.tzinfo is None:
        aa = aa.replace(tzinfo=timezone.utc)
    staleness_hours = round((now - aa).total_seconds() / 3600, 2) if aa else None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "fetched_at": fetched_at.isoformat() if fetched_at else None,
        "analyzed_at": aa.isoformat() if aa else None,
        "bias": bias,
        "wave_position": wave_position,
        "wave_degree": wave_degree,
        "current_wave_label": current_wave_label,
        "key_levels": key_levels,
        "blue_box_active": bool(blue_box_active),
        "blue_box_zone": blue_box_zone,
        "confidence": confidence,
        "invalidation_rule_violated": bool(invalidation_rule_violated),
        "analyst_notes": analyst_notes,
        "summary": summary,
        "staleness_hours": staleness_hours,
    }


@tool
async def get_ewf_analysis(
    symbol: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL', 'SPY'")],
    timeframe: Annotated[
        str | None,
        Field(
            description=(
                "Optional. One of: '1h_premarket', '1h_midday', '4h', 'daily', 'weekly'. "
                "If omitted, returns the most recent analysis across all timeframes within the freshness window."
            )
        ),
    ] = None,
) -> str:
    """Retrieve Elliott Wave Forecast analysis for a symbol.

    Returns the latest EWF chart analysis within the freshness window.
    Includes wave position, directional bias, key price levels (support,
    resistance, invalidation, target), Blue Box zone if active, and
    analyst confidence score.

    Returns {"ewf_available": false, "reason": "..."} if no fresh data exists.
    EWF charts are updated several times per day — data may be up to 6 hours
    old for 4H timeframe, 26 hours for daily.

    Synonyms: elliott wave, EWF chart, wave count, EWF signal, wave analysis.
    """
    try:
        sym = symbol.strip().upper()
        with pg_conn() as conn:
            if timeframe:
                tf = timeframe.strip()
                ttl = _TTL_MAP.get(tf, "24 hours")
                sql = _SINGLE_TF_SQL.format(ttl=ttl)
                conn.execute(sql, (sym, tf))
            else:
                conn.execute(_MULTI_TTL_SQL, (sym,))
            rows = conn.fetchall()

        if not rows:
            return json.dumps({
                "ewf_available": False,
                "reason": "No fresh EWF analysis found for this symbol/timeframe.",
            })

        results = [_row_to_dict(r) for r in rows]
        return json.dumps({"ewf_available": True, "results": results}, default=str)

    except Exception as exc:
        logger.warning("[ewf_tools] get_ewf_analysis failed: %s", exc)
        return json.dumps({
            "ewf_available": False,
            "reason": "EWF data temporarily unavailable.",
        })


@tool
async def get_ewf_blue_box_setups(
    date: Annotated[
        str | None,
        Field(description="Date as YYYY-MM-DD. Defaults to today if omitted.")
    ] = None,
) -> str:
    """Retrieve active EWF Blue Box reversal setups for a given date.

    The Blue Box is EWF's highest-conviction setup: a price zone where a
    corrective wave sequence is expected to complete and reverse. When price
    enters the Blue Box, it is a high-probability entry zone.

    Returns list of active setups with: symbol, direction (bullish/bearish),
    zone boundaries (low/high), confidence, and analyst summary.
    Returns empty list if no Blue Box setups exist for the date.

    Synonyms: blue box, EWF reversal zone, Elliott Wave setup, reversal zone, high probability zone.
    """
    try:
        if date:
            try:
                datetime.strptime(date, "%Y-%m-%d")
                date_str = date
            except ValueError:
                return json.dumps({
                    "date": date,
                    "setups": [],
                    "count": 0,
                    "error": f"Invalid date format: {date}. Expected YYYY-MM-DD.",
                })
        else:
            from datetime import date as date_cls
            date_str = date_cls.today().isoformat()

        with pg_conn() as conn:
            conn.execute(
                "SELECT symbol, bias, blue_box_zone, confidence, summary, "
                "       analyst_notes, timeframe, analyzed_at "
                "FROM ewf_chart_analyses "
                "WHERE timeframe = 'blue_box' "
                "  AND blue_box_active = TRUE "
                "  AND DATE(analyzed_at AT TIME ZONE 'UTC') = %s "
                "ORDER BY confidence DESC",
                (date_str,),
            )
            rows = conn.fetchall()

        setups = []
        for symbol, bias, bb_zone_raw, confidence, summary, notes, tf, analyzed_at in rows:
            bb_zone = _parse_jsonb(bb_zone_raw)
            setups.append({
                "symbol": symbol,
                "direction": bias,
                "zone_low": bb_zone.get("low") if bb_zone else None,
                "zone_high": bb_zone.get("high") if bb_zone else None,
                "confidence": confidence,
                "summary": summary,
                "analyst_notes": notes,
                "analyzed_at": analyzed_at.isoformat() if analyzed_at else None,
            })

        return json.dumps({
            "date": date_str,
            "setups": setups,
            "count": len(setups),
        }, default=str)

    except Exception as exc:
        logger.warning("[ewf_tools] get_ewf_blue_box_setups failed: %s", exc)
        date_str = date or "unknown"
        return json.dumps({
            "date": date_str,
            "setups": [],
            "count": 0,
            "error": "EWF data temporarily unavailable.",
        })
