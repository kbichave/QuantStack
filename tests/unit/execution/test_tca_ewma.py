"""Tests for TCA EWMA feedback loop.

Verifies:
  - resolve_time_bucket maps ET times to correct buckets
  - First fill creates a tca_parameters row with realized values
  - EWMA update formula: new = 0.9 * old + 0.1 * realized
  - sample_count increments on each fill
  - Different time buckets produce separate rows
  - conservative_multiplier decays from 2.0 to 1.0 over 50 samples
  - get_ewma_forecast returns None when no data, correct dict otherwise
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from quantstack.db import db_conn, _migrate_execution_layer_pg
from quantstack.execution.tca_ewma import (
    conservative_multiplier,
    get_ewma_forecast,
    resolve_time_bucket,
    update_ewma_after_fill,
)

_ET = ZoneInfo("America/New_York")


@pytest.fixture()
def conn():
    """Yield a live DB connection inside a savepoint that rolls back on teardown."""
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


class TestResolveTimeBucket:
    def test_morning(self):
        # 10:00 ET = morning
        ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "morning"

    def test_morning_boundary_start(self):
        # 09:30 ET exactly = morning
        ts = datetime(2026, 4, 6, 9, 30, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "morning"

    def test_midday(self):
        # 12:00 ET = midday
        ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "midday"

    def test_midday_boundary_start(self):
        # 11:00 ET exactly = midday
        ts = datetime(2026, 4, 6, 11, 0, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "midday"

    def test_afternoon(self):
        # 14:30 ET = afternoon
        ts = datetime(2026, 4, 6, 14, 30, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "afternoon"

    def test_afternoon_boundary_start(self):
        # 14:00 ET exactly = afternoon
        ts = datetime(2026, 4, 6, 14, 0, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "afternoon"

    def test_close(self):
        # 15:45 ET = close
        ts = datetime(2026, 4, 6, 15, 45, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "close"

    def test_close_boundary_start(self):
        # 15:30 ET exactly = close
        ts = datetime(2026, 4, 6, 15, 30, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "close"

    def test_outside_market_hours_premarket(self):
        # 08:00 ET = outside → close
        ts = datetime(2026, 4, 6, 8, 0, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "close"

    def test_outside_market_hours_afterhours(self):
        # 18:00 ET = outside → close
        ts = datetime(2026, 4, 6, 18, 0, tzinfo=_ET)
        assert resolve_time_bucket(ts) == "close"

    def test_naive_timestamp_assumed_utc(self):
        # 14:30 UTC naive → 10:30 ET (EDT, April) → morning
        ts = datetime(2026, 4, 6, 14, 30)
        assert resolve_time_bucket(ts) == "morning"


class TestConservativeMultiplier:
    def test_zero_samples(self):
        assert conservative_multiplier(0) == pytest.approx(2.0)

    def test_one_sample(self):
        # 2.0 - 1/50 = 1.98
        assert conservative_multiplier(1) == pytest.approx(1.98)

    def test_25_samples(self):
        assert conservative_multiplier(25) == pytest.approx(1.5)

    def test_50_samples(self):
        assert conservative_multiplier(50) == pytest.approx(1.0)

    def test_100_samples_clamped(self):
        assert conservative_multiplier(100) == pytest.approx(1.0)


class TestUpdateEwmaAfterFill:
    def test_first_fill_creates_row(self, conn):
        # Arrival=100, fill=100.50 → 50 bps total
        ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)  # morning
        update_ewma_after_fill(
            conn,
            order_id="ord-001",
            symbol="AAPL",
            fill_timestamp=ts,
            arrival_price=100.0,
            fill_price=100.50,
            fill_quantity=100,
            adv=1_000_000.0,
        )

        row = conn.execute(
            "SELECT ewma_spread_bps, ewma_impact_bps, ewma_total_bps, "
            "sample_count FROM tca_parameters "
            "WHERE symbol = ? AND time_bucket = ?",
            ["AAPL", "morning"],
        ).fetchone()

        assert row is not None
        assert float(row[2]) == pytest.approx(50.0)  # total = 50 bps
        assert float(row[0]) == pytest.approx(20.0)  # spread = 40% of 50
        assert float(row[1]) == pytest.approx(30.0)  # impact = 60% of 50
        assert int(row[3]) == 1

    def test_ewma_update_formula(self, conn):
        ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)

        # First fill: 50 bps total
        update_ewma_after_fill(
            conn, "ord-001", "AAPL", ts,
            arrival_price=100.0, fill_price=100.50,
            fill_quantity=100, adv=1_000_000.0,
        )

        # Second fill: 100 bps total
        update_ewma_after_fill(
            conn, "ord-002", "AAPL", ts,
            arrival_price=100.0, fill_price=101.00,
            fill_quantity=100, adv=1_000_000.0,
        )

        row = conn.execute(
            "SELECT ewma_total_bps, sample_count FROM tca_parameters "
            "WHERE symbol = ? AND time_bucket = ?",
            ["AAPL", "morning"],
        ).fetchone()

        # EWMA: 0.1 * 100 + 0.9 * 50 = 55
        assert float(row[0]) == pytest.approx(55.0)
        assert int(row[1]) == 2

    def test_sample_count_increments(self, conn):
        ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)  # midday

        for i in range(5):
            update_ewma_after_fill(
                conn, f"ord-{i}", "MSFT", ts,
                arrival_price=200.0, fill_price=200.10,
                fill_quantity=50, adv=5_000_000.0,
            )

        row = conn.execute(
            "SELECT sample_count FROM tca_parameters "
            "WHERE symbol = ? AND time_bucket = ?",
            ["MSFT", "midday"],
        ).fetchone()
        assert int(row[0]) == 5

    def test_different_buckets_separate_rows(self, conn):
        morning_ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        afternoon_ts = datetime(2026, 4, 6, 14, 30, tzinfo=_ET)

        update_ewma_after_fill(
            conn, "ord-am", "TSLA", morning_ts,
            arrival_price=150.0, fill_price=150.30,
            fill_quantity=100, adv=10_000_000.0,
        )
        update_ewma_after_fill(
            conn, "ord-pm", "TSLA", afternoon_ts,
            arrival_price=150.0, fill_price=150.60,
            fill_quantity=100, adv=10_000_000.0,
        )

        rows = conn.execute(
            "SELECT time_bucket, ewma_total_bps FROM tca_parameters "
            "WHERE symbol = ? ORDER BY time_bucket",
            ["TSLA"],
        ).fetchall()

        assert len(rows) == 2
        buckets = {row[0]: float(row[1]) for row in rows}
        assert "morning" in buckets
        assert "afternoon" in buckets
        # morning: 20 bps, afternoon: 40 bps
        assert buckets["morning"] == pytest.approx(20.0)
        assert buckets["afternoon"] == pytest.approx(40.0)

    def test_zero_arrival_price_skipped(self, conn):
        """Arrival price of 0 would cause division by zero; should be a no-op."""
        ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        update_ewma_after_fill(
            conn, "ord-bad", "BAD", ts,
            arrival_price=0.0, fill_price=10.0,
            fill_quantity=100, adv=1_000_000.0,
        )
        row = conn.execute(
            "SELECT COUNT(*) FROM tca_parameters WHERE symbol = ?",
            ["BAD"],
        ).fetchone()
        assert int(row[0]) == 0


class TestGetEwmaForecast:
    def test_returns_none_when_no_data(self, conn):
        result = get_ewma_forecast(conn, "UNKNOWN", "morning")
        assert result is None

    def test_returns_correct_dict(self, conn):
        ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        update_ewma_after_fill(
            conn, "ord-001", "AAPL", ts,
            arrival_price=100.0, fill_price=100.50,
            fill_quantity=100, adv=1_000_000.0,
        )

        result = get_ewma_forecast(conn, "AAPL", "morning")
        assert result is not None
        assert result["ewma_total_bps"] == pytest.approx(50.0)
        assert result["ewma_spread_bps"] == pytest.approx(20.0)
        assert result["ewma_impact_bps"] == pytest.approx(30.0)
        assert result["sample_count"] == 1
        # multiplier at 1 sample: max(1.0, 2.0 - 1/50) = 1.98
        assert result["multiplier"] == pytest.approx(1.98)

    def test_multiplier_decays_with_samples(self, conn):
        ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)

        # Insert 50 fills to get multiplier to 1.0
        for i in range(50):
            update_ewma_after_fill(
                conn, f"ord-{i}", "GOOG", ts,
                arrival_price=100.0, fill_price=100.10,
                fill_quantity=100, adv=5_000_000.0,
            )

        result = get_ewma_forecast(conn, "GOOG", "midday")
        assert result is not None
        assert result["sample_count"] == 50
        assert result["multiplier"] == pytest.approx(1.0)
