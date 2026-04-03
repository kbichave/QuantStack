"""Shared runner utilities: market hours detection and cycle interval selection."""

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# NYSE holidays for 2025-2027 (dates when markets are fully closed)
NYSE_HOLIDAYS: set[tuple[int, int, int]] = {
    # 2025
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18),
    (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1),
    (2025, 11, 27), (2025, 12, 25),
    # 2026
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3),
    (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25),
    # 2027
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26),
    (2027, 5, 31), (2027, 6, 18), (2027, 7, 5), (2027, 9, 6),
    (2027, 11, 25), (2027, 12, 24),
}

INTERVALS: dict[str, dict[str, int | None]] = {
    # Bootstrap phase: aggressive research, trading only during market hours.
    # After strategy library is built (30+ strategies), scale back research
    # to after_hours=1800, weekend=3600.
    "trading":    {"market": 300,  "after_hours": None,  "weekend": None},
    "research":   {"market": 120,  "after_hours": 180,   "weekend": 300},
    "supervisor": {"market": 300,  "after_hours": 300,   "weekend": 300},
}


def is_market_hours(dt: datetime | None = None) -> bool:
    """Check if the given datetime falls within NYSE regular trading hours.

    Naive datetimes are treated as US/Eastern.
    """
    if dt is None:
        dt = datetime.now(ET)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)

    # Weekend check
    if dt.weekday() >= 5:
        return False

    # Holiday check
    if (dt.year, dt.month, dt.day) in NYSE_HOLIDAYS:
        return False

    # Time check: [9:30, 16:00)
    t = dt.time()
    return MARKET_OPEN <= t < MARKET_CLOSE


def _is_weekend(dt: datetime | None = None) -> bool:
    """Check if the given datetime is a weekend day."""
    if dt is None:
        dt = datetime.now(ET)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)
    return dt.weekday() >= 5


def get_cycle_interval(graph_name: str, dt: datetime | None = None) -> int | None:
    """Return the sleep interval in seconds for the given graph, or None to pause."""
    config = INTERVALS[graph_name]
    if is_market_hours(dt):
        return config["market"]
    if _is_weekend(dt):
        return config["weekend"]
    return config["after_hours"]
