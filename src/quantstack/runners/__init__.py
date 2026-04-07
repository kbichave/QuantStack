"""Shared runner utilities: market hours detection and cycle interval selection."""

from datetime import datetime, time as dtime
from enum import Enum
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
EXTENDED_OPEN = dtime(4, 0)
EXTENDED_CLOSE = dtime(20, 0)

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


class OperatingMode(str, Enum):
    """Four-mode operating system for 24/7 operation."""

    MARKET = "market"           # 9:30-16:00 ET Mon-Fri
    EXTENDED = "extended"       # 04:00-09:30 ET, 16:00-20:00 ET Mon-Fri
    OVERNIGHT = "overnight"     # 20:00-04:00 ET Mon-Fri
    WEEKEND = "weekend"         # Sat-Sun all day, NYSE holidays


def get_operating_mode(dt: datetime | None = None) -> OperatingMode:
    """Determine current operating mode based on ET time and calendar.

    Args:
        dt: Datetime to check. If None, uses current time.
            Naive datetimes are treated as US/Eastern.

    Returns:
        The current OperatingMode.
    """
    if dt is None:
        dt = datetime.now(ET)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)

    # Weekend/holiday check
    if dt.weekday() >= 5:
        return OperatingMode.WEEKEND
    if (dt.year, dt.month, dt.day) in NYSE_HOLIDAYS:
        return OperatingMode.WEEKEND

    t = dt.time()

    # Market hours: [9:30, 16:00)
    if MARKET_OPEN <= t < MARKET_CLOSE:
        return OperatingMode.MARKET

    # Extended hours: [4:00, 9:30) and [16:00, 20:00)
    if EXTENDED_OPEN <= t < MARKET_OPEN:
        return OperatingMode.EXTENDED
    if MARKET_CLOSE <= t < EXTENDED_CLOSE:
        return OperatingMode.EXTENDED

    # Everything else: overnight
    return OperatingMode.OVERNIGHT


# Mode-aware interval table
INTERVALS: dict[str, dict[OperatingMode, int | None]] = {
    "trading": {
        OperatingMode.MARKET: 300,
        OperatingMode.EXTENDED: 300,
        OperatingMode.OVERNIGHT: None,
        OperatingMode.WEEKEND: None,
    },
    "research": {
        OperatingMode.MARKET: 120,
        OperatingMode.EXTENDED: 180,
        OperatingMode.OVERNIGHT: 120,
        OperatingMode.WEEKEND: 300,
    },
    "supervisor": {
        OperatingMode.MARKET: 300,
        OperatingMode.EXTENDED: 300,
        OperatingMode.OVERNIGHT: 300,
        OperatingMode.WEEKEND: 300,
    },
}


def is_market_hours(dt: datetime | None = None) -> bool:
    """Check if the given datetime falls within NYSE regular trading hours.

    Naive datetimes are treated as US/Eastern.
    """
    return get_operating_mode(dt) == OperatingMode.MARKET


def get_cycle_interval(graph_name: str, dt: datetime | None = None) -> int | None:
    """Return the sleep interval in seconds for the given graph, or None to pause."""
    mode = get_operating_mode(dt)
    return INTERVALS[graph_name][mode]
