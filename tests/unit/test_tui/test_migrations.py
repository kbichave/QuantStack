"""Tests for market_holidays migration and holiday date computation."""
from datetime import date, timedelta
from unittest.mock import MagicMock, call

import pytest

from quantstack.db import PgConnection, _compute_us_holidays, _seed_market_holidays


class TestComputeUsHolidays:

    def test_returns_at_least_10_holidays(self):
        holidays = _compute_us_holidays(2026)
        assert len(holidays) >= 10

    def test_2026_new_years(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["New Year's Day"] == date(2026, 1, 1)

    def test_2026_mlk_day(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Martin Luther King Jr. Day"] == date(2026, 1, 19)

    def test_2026_presidents_day(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Presidents' Day"] == date(2026, 2, 16)

    def test_2026_good_friday(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Good Friday"] == date(2026, 4, 3)

    def test_2026_memorial_day(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Memorial Day"] == date(2026, 5, 25)

    def test_2026_juneteenth(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Juneteenth"] == date(2026, 6, 19)

    def test_2026_independence_day(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        # Jul 4, 2026 is Saturday -> observed Friday Jul 3
        assert names["Independence Day"] == date(2026, 7, 3)

    def test_2026_labor_day(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Labor Day"] == date(2026, 9, 7)

    def test_2026_thanksgiving(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Thanksgiving"] == date(2026, 11, 26)

    def test_2026_christmas(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Christmas"] == date(2026, 12, 25)

    def test_early_closes_have_correct_status(self):
        holidays = _compute_us_holidays(2026)
        early = [h for h in holidays if h[2] == "early_close"]
        assert len(early) >= 1
        for h in early:
            assert h[3] == "13:00:00"

    def test_2026_black_friday(self):
        holidays = _compute_us_holidays(2026)
        names = {h[1]: h[0] for h in holidays}
        assert names["Black Friday"] == date(2026, 11, 27)

    def test_weekend_observance_sunday_to_monday(self):
        # 2027: Jul 4 is Sunday -> observed Monday Jul 5
        holidays = _compute_us_holidays(2027)
        names = {h[1]: h[0] for h in holidays}
        assert names["Independence Day"] == date(2027, 7, 5)

    def test_all_holidays_are_weekdays(self):
        for year in (2026, 2027):
            holidays = _compute_us_holidays(year)
            for dt, name, status, _ in holidays:
                assert dt.weekday() < 5, f"{name} on {dt} is not a weekday"


class TestSeedMarketHolidays:

    def test_uses_on_conflict_do_nothing(self):
        conn = MagicMock(spec=PgConnection)
        _seed_market_holidays(conn, 2026)
        for c in conn.execute.call_args_list:
            sql = c[0][0]
            assert "ON CONFLICT DO NOTHING" in sql

    def test_idempotent_no_error_on_double_call(self):
        conn = MagicMock(spec=PgConnection)
        _seed_market_holidays(conn, 2026)
        _seed_market_holidays(conn, 2026)
        # No exception raised


class TestBenchmarkDailyExists:

    def test_benchmark_daily_in_attribution_migration(self):
        """benchmark_daily is created by _migrate_attribution_pg, not this section."""
        from quantstack.db import _migrate_attribution_pg
        conn = MagicMock(spec=PgConnection)
        _migrate_attribution_pg(conn)
        sql_calls = [c[0][0] for c in conn.execute.call_args_list]
        assert any("benchmark_daily" in sql for sql in sql_calls)
