# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Events collector — replaces calendar_events_ic.

Fetches upcoming earnings / FOMC / economic events via the data layer.
Uses a short timeout; returns safe defaults on miss so the rest of the
analysis is never blocked by a network timeout.
"""

import asyncio
import os
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.data.earnings import EarningsManager
from quantstack.data.macro_calendar import MacroCalendarGenerator
from quantstack.signal_engine.staleness import check_freshness


_TIMEOUT_SECONDS = 6.0


async def collect_events(symbol: str, _store: Any) -> dict[str, Any]:
    """
    Fetch upcoming calendar events for *symbol*.

    Returns a dict with keys:
        has_earnings_24h  : bool
        has_earnings_7d   : bool
        has_fomc_24h      : bool
        has_macro_event   : bool — CPI, NFP, GDP in next 24h
        next_event_desc   : str  — human-readable description of nearest event
        events_7d         : list[dict] — raw event list for the next 7 days
    """
    if not check_freshness(symbol, "earnings_calendar", max_days=30):
        return {}
    try:
        return await asyncio.wait_for(_fetch_events(symbol), timeout=_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug(
            f"[events] {symbol}: {type(exc).__name__} — returning safe defaults"
        )
        return _safe_defaults()


async def _fetch_events(symbol: str) -> dict[str, Any]:
    raw = await asyncio.to_thread(_events_from_data_layer, symbol)
    return _parse_events(raw)


def _events_from_data_layer(symbol: str) -> list[dict]:
    """Best-effort events from local earnings data + Alpha Vantage + macro calendar."""
    events: list[dict] = []

    # 1. Local earnings via EarningsManager
    mgr = EarningsManager()
    upcoming_earnings = mgr.get_all_upcoming(days_ahead=7)
    for evt in upcoming_earnings:
        if evt.symbol == symbol or evt.symbol is None:
            events.append({
                "event_type": "earnings",
                "symbol": evt.symbol,
                "date": str(evt.report_date),
                "description": f"Earnings: {evt.symbol}",
            })

    # 2. Alpha Vantage earnings (if local is empty)
    if not events:
        av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if av_key:
            adapter = AlphaVantageAdapter(api_key=av_key)
            df = adapter.fetch_earnings(symbol=symbol, horizon="3month")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    report_date = row.get("report_date")
                    if report_date is not None:
                        events.append(
                            {
                                "event_type": "earnings",
                                "symbol": symbol,
                                "date": str(report_date)[:10],
                                "description": f"Earnings: {symbol} (est EPS: {row.get('estimate', '?')})",
                            }
                        )

    # 3. Macro calendar (FOMC, CPI, NFP) — affects all symbols
    gen = MacroCalendarGenerator()
    calendar = gen.build_calendar()
    upcoming = calendar.get_upcoming_events(datetime.now(), hours_ahead=168)  # 7 days

    for evt in upcoming:
        events.append(
            {
                "event_type": evt.event_type.value.lower(),
                "symbol": None,  # affects all
                "date": str(evt.timestamp.date()),
                "description": f"{evt.event_type.value} — {evt.timestamp.strftime('%b %d %I:%M %p ET')}",
            }
        )

    return events


def _parse_events(raw: Any) -> dict[str, Any]:
    """Normalize raw event list into the events collector output schema."""
    today = date.today()
    cutoff_24h = today + timedelta(days=1)
    cutoff_7d = today + timedelta(days=7)

    events = []
    if isinstance(raw, dict):
        events = raw.get("upcoming_events", raw.get("events", []))
    elif isinstance(raw, list):
        events = raw

    has_earnings_24h = False
    has_earnings_7d = False
    has_fomc_24h = False
    has_macro_24h = False
    next_event_desc = "None in next 7 days"
    events_7d = []

    macro_keywords = {"cpi", "nfp", "nonfarm", "gdp", "pce", "fomc", "fed"}

    for event in events:
        if not isinstance(event, dict):
            continue
        event_date_raw = event.get("date") or event.get("event_date")
        try:
            if isinstance(event_date_raw, str):
                event_date = date.fromisoformat(event_date_raw[:10])
            elif isinstance(event_date_raw, date):
                event_date = event_date_raw
            else:
                continue
        except ValueError:
            continue

        if event_date > cutoff_7d:
            continue

        event_type = str(event.get("type", event.get("event_type", ""))).lower()
        event_name = str(event.get("name", event.get("description", ""))).lower()
        events_7d.append(event)

        if event_date <= cutoff_24h:
            if "earnings" in event_type or "earnings" in event_name:
                has_earnings_24h = True
            if "fomc" in event_type or "fomc" in event_name or "fed" in event_name:
                has_fomc_24h = True
            if any(kw in event_name for kw in macro_keywords):
                has_macro_24h = True

        if "earnings" in event_type or "earnings" in event_name:
            has_earnings_7d = True

    if events_7d:
        first = events_7d[0]
        next_event_desc = (
            f"{first.get('name', first.get('type', 'Event'))} "
            f"on {first.get('date', first.get('event_date', 'unknown'))}"
        )

    return {
        "has_earnings_24h": has_earnings_24h,
        "has_earnings_7d": has_earnings_7d,
        "has_fomc_24h": has_fomc_24h,
        "has_macro_event": has_macro_24h,
        "next_event_desc": next_event_desc,
        "events_7d": events_7d[:10],  # cap to keep output small
    }


def _safe_defaults() -> dict[str, Any]:
    return {
        "has_earnings_24h": False,
        "has_earnings_7d": False,
        "has_fomc_24h": False,
        "has_macro_event": False,
        "next_event_desc": "unknown (events fetch failed)",
        "events_7d": [],
    }
