"""Trading calendar tools for LangGraph agents.

Gives agents awareness of market holidays, long weekends, upcoming events,
and their impact on options theta decay. Critical for:
- Options agents deciding whether to hold through weekends/holidays
- Position monitors flagging accelerated theta exposure
- Daily planners incorporating event risk into plans
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def _build_calendar_context(days_ahead: int = 14) -> dict:
    """Build forward-looking trading calendar context.

    Returns a dict with market closure schedule, theta decay multipliers,
    upcoming macro events, and options expiration dates.
    """
    from quantstack.core.core.calendar import TradingCalendar
    from quantstack.data.macro_calendar import MacroCalendarGenerator
    from quantstack.shared.event_calendar import EventCalendar, EventType

    today = date.today()
    end = today + timedelta(days=days_ahead)
    now_et = datetime.now(ET)

    cal = TradingCalendar("NYSE")

    # Trading days in the window
    trading_days = cal.get_trading_days(today, end)
    trading_day_set = set(trading_days)

    # Build day-by-day schedule
    schedule = []
    for offset in range(days_ahead + 1):
        d = today + timedelta(days=offset)
        is_trading = d in trading_day_set
        day_info = {
            "date": d.isoformat(),
            "weekday": d.strftime("%A"),
            "is_trading_day": is_trading,
        }
        if not is_trading:
            if d.weekday() >= 5:
                day_info["reason"] = "weekend"
            else:
                day_info["reason"] = "holiday"
        schedule.append(day_info)

    # Find non-trading stretches (weekends, long weekends, holiday closures)
    closures = []
    i = 0
    while i < len(schedule):
        if not schedule[i]["is_trading_day"]:
            start_idx = i
            while i < len(schedule) and not schedule[i]["is_trading_day"]:
                i += 1
            closure_days = schedule[start_idx:i]
            calendar_days = len(closure_days)
            has_holiday = any(d.get("reason") == "holiday" for d in closure_days)
            closures.append({
                "start": closure_days[0]["date"],
                "end": closure_days[-1]["date"],
                "calendar_days": calendar_days,
                "type": "long_weekend" if has_holiday else "weekend",
                "theta_multiplier": calendar_days,
                "note": (
                    f"{calendar_days} calendar days of theta decay with no trading. "
                    f"Options positions lose {calendar_days}x daily theta over this period."
                ),
            })
        else:
            i += 1

    # Next trading day info
    if not cal.is_trading_day(today):
        next_trading = cal.next_trading_day(today)
        days_until_open = (next_trading - today).days
    else:
        next_trading = today
        days_until_open = 0

    # Upcoming events from macro calendar + event calendar
    events = []
    try:
        ecal = EventCalendar()
        macro = MacroCalendarGenerator()
        built = macro.build_calendar(today.year, today.year + 1)
        for evt in built.get_upcoming_events(now_et, hours_ahead=days_ahead * 24):
            events.append({
                "type": evt.event_type.value,
                "date": evt.timestamp.strftime("%Y-%m-%d %H:%M ET"),
                "impact": evt.impact,
                "blackout_hours_before": evt.blackout_hours_before,
                "blackout_hours_after": evt.blackout_hours_after,
            })
    except Exception as exc:
        logger.debug("Could not load macro events: %s", exc)

    # OPEX detection (third Friday of each month)
    opex_dates = []
    for month_offset in range(2):
        m = today.month + month_offset
        y = today.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        first_day = date(y, m, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        if third_friday >= today:
            is_quad_witch = m in (3, 6, 9, 12)
            opex_dates.append({
                "date": third_friday.isoformat(),
                "days_away": (third_friday - today).days,
                "type": "quad_witch" if is_quad_witch else "monthly_opex",
            })

    total_calendar_days = days_ahead
    total_trading_days = len([d for d in trading_days if d >= today])

    return {
        "as_of": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "today_is_trading_day": cal.is_trading_day(today),
        "next_trading_day": next_trading.isoformat(),
        "days_until_market_open": days_until_open,
        "trading_days_in_window": total_trading_days,
        "calendar_days_in_window": total_calendar_days,
        "theta_efficiency": (
            round(total_trading_days / total_calendar_days, 2)
            if total_calendar_days > 0 else 1.0
        ),
        "market_closures": closures,
        "upcoming_events": events[:15],
        "opex_dates": opex_dates,
        "schedule": schedule,
    }


@tool
async def get_trading_calendar(
    days_ahead: Annotated[int, Field(
        description="Number of days to look ahead for calendar context (default 14, max 60)"
    )] = 14,
) -> str:
    """Returns the forward-looking trading calendar including market holidays, weekends, long weekends, upcoming macro events (FOMC, CPI, NFP), and options expiration dates (OPEX/quad witch).

    Critical for options trading decisions:
    - **Theta decay multiplier**: Shows how many calendar days of theta decay occur over each market closure. A 3-day weekend = 3x daily theta with zero trading opportunity.
    - **Theta efficiency**: Ratio of trading days to calendar days. Lower efficiency means more theta bleed per trading day.
    - **Long weekends/holidays**: Identifies periods where holding short gamma is especially risky.
    - **OPEX dates**: Monthly options expiration and quarterly quad witching dates with days-away countdown.
    - **Macro events**: FOMC, CPI, NFP with blackout windows that affect entry timing.

    Use BEFORE entering any options position to assess calendar risk. Use in position monitoring to flag upcoming theta-heavy periods.
    """
    days_ahead = min(max(days_ahead, 1), 60)
    try:
        context = _build_calendar_context(days_ahead)
        return json.dumps(context, default=str)
    except Exception as e:
        logger.error("get_trading_calendar failed: %s", e)
        return json.dumps({"error": str(e)})
