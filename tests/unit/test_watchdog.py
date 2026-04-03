"""Tests for AgentWatchdog timer-based stuck cycle detection."""

import threading
import time

from quantstack.health.watchdog import AgentWatchdog


class TestWatchdog:
    def test_triggers_after_timeout(self):
        event = threading.Event()
        watchdog = AgentWatchdog(timeout_seconds=0.5, on_timeout=event.set)
        watchdog.start_cycle()
        # Wait up to 2s for the timeout callback
        triggered = event.wait(timeout=2.0)
        assert triggered, "Watchdog callback should have fired after 0.5s"

    def test_does_not_trigger_if_cycle_completes(self):
        event = threading.Event()
        watchdog = AgentWatchdog(timeout_seconds=1.0, on_timeout=event.set)
        watchdog.start_cycle()
        time.sleep(0.2)
        watchdog.end_cycle()
        # Wait past the original timeout
        triggered = event.wait(timeout=1.5)
        assert not triggered, "Watchdog callback should NOT fire after end_cycle()"

    def test_end_cycle_resets_for_next_cycle(self):
        event = threading.Event()
        watchdog = AgentWatchdog(timeout_seconds=0.5, on_timeout=event.set)

        # Cycle 1: complete before timeout
        watchdog.start_cycle()
        time.sleep(0.1)
        watchdog.end_cycle()

        # Cycle 2: complete before timeout
        watchdog.start_cycle()
        time.sleep(0.1)
        watchdog.end_cycle()

        # Wait past the timeout
        triggered = event.wait(timeout=1.0)
        assert not triggered, "Watchdog should not fire across two completed cycles"

    def test_start_cycle_resets_existing_timer(self):
        event = threading.Event()
        watchdog = AgentWatchdog(timeout_seconds=0.5, on_timeout=event.set)

        # Start, then restart before timeout
        watchdog.start_cycle()
        time.sleep(0.3)
        watchdog.start_cycle()  # reset — timer restarts
        time.sleep(0.3)
        # 0.3s into second timer, should not have fired yet (0.5s timeout)
        assert not event.is_set()
        watchdog.end_cycle()
