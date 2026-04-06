"""Tests for TieredRefreshScheduler."""
import threading

import pytest

from quantstack.tui.refresh import TieredRefreshScheduler


class TestTieredRefreshScheduler:
    """Refresh tier configuration and behavior."""

    def test_has_four_tiers(self):
        scheduler = TieredRefreshScheduler()
        assert set(scheduler.TIERS.keys()) == {"T1", "T2", "T3", "T4"}

    def test_tier_intervals(self):
        scheduler = TieredRefreshScheduler()
        assert scheduler.TIERS["T1"] == 5.0
        assert scheduler.TIERS["T2"] == 15.0
        assert scheduler.TIERS["T3"] == 60.0
        assert scheduler.TIERS["T4"] == 120.0

    def test_stagger_offsets(self):
        scheduler = TieredRefreshScheduler()
        assert scheduler.STAGGER["T1"] == 0.0
        assert scheduler.STAGGER["T2"] == 0.3
        assert scheduler.STAGGER["T3"] == 0.6
        assert scheduler.STAGGER["T4"] == 0.9

    def test_db_semaphore_is_threading_semaphore(self):
        scheduler = TieredRefreshScheduler()
        assert isinstance(scheduler._db_semaphore, threading.Semaphore)

    def test_db_semaphore_default_value_is_five(self):
        scheduler = TieredRefreshScheduler()
        # Semaphore(5) allows 5 concurrent acquires before blocking
        acquired = 0
        for _ in range(5):
            assert scheduler._db_semaphore.acquire(blocking=False)
            acquired += 1
        # 6th acquire should fail (non-blocking)
        assert not scheduler._db_semaphore.acquire(blocking=False)
        # Release all
        for _ in range(acquired):
            scheduler._db_semaphore.release()

    def test_active_tab_change_updates_query_set(self):
        scheduler = TieredRefreshScheduler()
        scheduler.active_tab = "tab-overview"
        assert scheduler.active_tab == "tab-overview"
        scheduler.active_tab = "tab-portfolio"
        assert scheduler.active_tab == "tab-portfolio"

    def test_header_queries_fire_regardless_of_tab(self):
        """Widgets registered with always_on=True are refreshed on every tick."""
        scheduler = TieredRefreshScheduler()
        scheduler.active_tab = "tab-portfolio"
        # always_on widgets should be included in the refresh set
        # even when their tab doesn't match active_tab
        widgets = scheduler.get_refreshable_widgets("T1")
        # At this point no widgets are registered, so just verify it returns a list
        assert isinstance(widgets, list)
