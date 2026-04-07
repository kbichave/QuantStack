"""Unit tests for OperatingMode enum and get_operating_mode()."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from quantstack.runners import (
    ET,
    OperatingMode,
    get_cycle_interval,
    get_operating_mode,
)

ET_TZ = ZoneInfo("America/New_York")


class TestGetOperatingMode:
    def test_market_hours_10am_monday(self):
        dt = datetime(2026, 4, 6, 10, 0, tzinfo=ET_TZ)  # Monday 10:00 ET
        assert get_operating_mode(dt) == OperatingMode.MARKET

    def test_extended_5pm_tuesday(self):
        dt = datetime(2026, 4, 7, 17, 0, tzinfo=ET_TZ)  # Tuesday 17:00 ET
        assert get_operating_mode(dt) == OperatingMode.EXTENDED

    def test_extended_5am_wednesday(self):
        dt = datetime(2026, 4, 8, 5, 0, tzinfo=ET_TZ)  # Wednesday 05:00 ET
        assert get_operating_mode(dt) == OperatingMode.EXTENDED

    def test_overnight_10pm_thursday(self):
        dt = datetime(2026, 4, 9, 22, 0, tzinfo=ET_TZ)  # Thursday 22:00 ET
        assert get_operating_mode(dt) == OperatingMode.OVERNIGHT

    def test_weekend_saturday(self):
        dt = datetime(2026, 4, 11, 14, 0, tzinfo=ET_TZ)  # Saturday 14:00 ET
        assert get_operating_mode(dt) == OperatingMode.WEEKEND

    def test_nyse_holiday_returns_weekend(self):
        # 2026-01-19 is MLK Day (Monday), NYSE closed
        dt = datetime(2026, 1, 19, 11, 0, tzinfo=ET_TZ)
        assert get_operating_mode(dt) == OperatingMode.WEEKEND

    def test_market_open_boundary(self):
        dt = datetime(2026, 4, 6, 9, 30, tzinfo=ET_TZ)
        assert get_operating_mode(dt) == OperatingMode.MARKET

    def test_market_close_boundary(self):
        dt = datetime(2026, 4, 6, 16, 0, tzinfo=ET_TZ)
        assert get_operating_mode(dt) == OperatingMode.EXTENDED

    def test_extended_open_boundary(self):
        dt = datetime(2026, 4, 6, 4, 0, tzinfo=ET_TZ)
        assert get_operating_mode(dt) == OperatingMode.EXTENDED

    def test_overnight_before_extended(self):
        dt = datetime(2026, 4, 6, 3, 59, tzinfo=ET_TZ)
        assert get_operating_mode(dt) == OperatingMode.OVERNIGHT


class TestGetCycleInterval:
    def test_trading_market_300(self):
        dt = datetime(2026, 4, 6, 10, 0, tzinfo=ET_TZ)
        assert get_cycle_interval("trading", dt) == 300

    def test_trading_extended_300(self):
        dt = datetime(2026, 4, 6, 17, 0, tzinfo=ET_TZ)
        assert get_cycle_interval("trading", dt) == 300

    def test_trading_overnight_none(self):
        dt = datetime(2026, 4, 6, 22, 0, tzinfo=ET_TZ)
        assert get_cycle_interval("trading", dt) is None

    def test_research_overnight_120(self):
        dt = datetime(2026, 4, 6, 22, 0, tzinfo=ET_TZ)
        assert get_cycle_interval("research", dt) == 120

    def test_research_weekend_300(self):
        dt = datetime(2026, 4, 11, 14, 0, tzinfo=ET_TZ)
        assert get_cycle_interval("research", dt) == 300

    def test_supervisor_always_300(self):
        for dt in [
            datetime(2026, 4, 6, 10, 0, tzinfo=ET_TZ),   # market
            datetime(2026, 4, 6, 17, 0, tzinfo=ET_TZ),   # extended
            datetime(2026, 4, 6, 22, 0, tzinfo=ET_TZ),   # overnight
            datetime(2026, 4, 11, 14, 0, tzinfo=ET_TZ),  # weekend
        ]:
            assert get_cycle_interval("supervisor", dt) == 300
