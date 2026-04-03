"""Timer-based watchdog for detecting stuck crew cycles."""

import threading
from typing import Callable


class AgentWatchdog:
    """Detects stuck crew cycles via a background timer.

    Usage:
        watchdog = AgentWatchdog(timeout_seconds=600, on_timeout=handle_stuck)
        watchdog.start_cycle()
        crew.kickoff(...)
        watchdog.end_cycle()

    If end_cycle() is not called within timeout_seconds, on_timeout fires.
    """

    def __init__(self, timeout_seconds: int | float, on_timeout: Callable[[], None]) -> None:
        self._timeout = timeout_seconds
        self._on_timeout = on_timeout
        self._timer: threading.Timer | None = None

    def start_cycle(self) -> None:
        """Start or reset the watchdog timer for a new cycle."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def end_cycle(self) -> None:
        """Cancel the current timer. Call after cycle completes normally."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
