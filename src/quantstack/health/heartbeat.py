"""File-based heartbeat for Docker health checks.

Each crew runner writes a timestamp file after every successful cycle.
Docker health checks read it to determine container health.
"""

import sys
import time
from pathlib import Path

HEARTBEAT_DIR = Path("/tmp")

DEFAULT_MAX_AGE: dict[str, int] = {
    "trading": 120,
    "research": 600,
    "supervisor": 360,
}


def write_heartbeat(crew_name: str) -> None:
    """Write current unix timestamp to {HEARTBEAT_DIR}/{crew_name}-heartbeat."""
    path = HEARTBEAT_DIR / f"{crew_name}-heartbeat"
    path.write_text(str(time.time()))


def check_health(crew_name: str, max_age_seconds: int | None = None) -> bool:
    """Return True if heartbeat file exists and is fresher than max_age_seconds.

    If max_age_seconds is None, uses DEFAULT_MAX_AGE for the crew.
    Returns False if the file is missing or stale.
    """
    if max_age_seconds is None:
        max_age_seconds = DEFAULT_MAX_AGE.get(crew_name, 300)

    path = HEARTBEAT_DIR / f"{crew_name}-heartbeat"
    if not path.is_file():
        return False

    try:
        ts = float(path.read_text().strip())
    except (ValueError, OSError):
        return False

    return (time.time() - ts) < max_age_seconds


def check(crew_name: str) -> None:
    """CLI entry point for Docker health check.

    Called as: python -c "from quantstack.health.heartbeat import check_health; check_health('trading')"
    Exits 0 if healthy, 1 if unhealthy.
    """
    sys.exit(0 if check_health(crew_name) else 1)
