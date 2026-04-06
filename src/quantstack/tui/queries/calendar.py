"""Calendar queries: earnings events."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class EarningsEvent:
    symbol: str
    report_date: date
    estimate: float | None
    reported_eps: float | None
    surprise_pct: float | None


def fetch_earnings_calendar(conn: PgConnection, days_ahead: int = 90) -> list[EarningsEvent]:
    """Return upcoming earnings events for universe symbols."""
    try:
        conn.execute(
            "SELECT symbol, report_date, estimate, reported_eps, surprise_pct "
            "FROM earnings_calendar "
            "WHERE report_date >= CURRENT_DATE "
            "AND report_date <= CURRENT_DATE + make_interval(days => %s) "
            "ORDER BY report_date",
            (days_ahead,),
        )
        return [
            EarningsEvent(
                symbol=r[0], report_date=r[1],
                estimate=float(r[2]) if r[2] is not None else None,
                reported_eps=float(r[3]) if r[3] is not None else None,
                surprise_pct=float(r[4]) if r[4] is not None else None,
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_earnings_calendar failed", exc_info=True)
        return []
