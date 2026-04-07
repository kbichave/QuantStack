"""Tests for ewf_chart_analyses table schema (Section 01)."""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg.errors
import pytest


@pytest.fixture(autouse=True)
def _clean_ewf_table(trading_ctx):
    """Delete all rows from ewf_chart_analyses before each test."""
    trading_ctx.db.execute("DELETE FROM ewf_chart_analyses")
    trading_ctx.db.commit()
    yield


class TestEwfChartAnalysesTable:
    """Verify the ewf_chart_analyses migration creates the correct schema."""

    def test_table_exists(self, trading_ctx):
        """_run_migrations creates ewf_chart_analyses table."""
        trading_ctx.db.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'ewf_chart_analyses'"
        )
        assert trading_ctx.db.fetchone() is not None

    def test_migration_idempotent(self, trading_ctx):
        """Running the EWF migration function twice raises no error."""
        from quantstack.db import _migrate_ewf_pg

        _migrate_ewf_pg(trading_ctx.db)

    def test_has_all_required_columns(self, trading_ctx):
        """ewf_chart_analyses has all expected columns."""
        trading_ctx.db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'ewf_chart_analyses' "
            "ORDER BY ordinal_position"
        )
        columns = {row[0] for row in trading_ctx.db.fetchall()}
        expected = {
            "id", "symbol", "timeframe", "fetched_at", "analyzed_at",
            "image_path", "bias", "turning_signal", "wave_position",
            "wave_degree", "current_wave_label", "completed_wave_sequence",
            "projected_path", "key_levels", "blue_box_active",
            "blue_box_zone", "confidence", "invalidation_rule_violated",
            "analyst_notes", "summary", "reasoning", "raw_analysis",
            "model_used",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

    def test_unique_constraint_enforced(self, trading_ctx):
        """UNIQUE(symbol, timeframe, fetched_at) prevents duplicate rows."""
        ts = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
        trading_ctx.db.execute(
            "INSERT INTO ewf_chart_analyses (symbol, timeframe, fetched_at) "
            "VALUES (%s, %s, %s)",
            ("AAPL", "4h", ts),
        )
        trading_ctx.db.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            trading_ctx.db.execute(
                "INSERT INTO ewf_chart_analyses (symbol, timeframe, fetched_at) "
                "VALUES (%s, %s, %s)",
                ("AAPL", "4h", ts),
            )

    def test_blue_box_active_defaults_false(self, trading_ctx):
        """blue_box_active defaults to FALSE on insert."""
        ts = datetime(2026, 4, 4, 13, 0, 0, tzinfo=timezone.utc)
        trading_ctx.db.execute(
            "INSERT INTO ewf_chart_analyses (symbol, timeframe, fetched_at) "
            "VALUES (%s, %s, %s)",
            ("MSFT", "daily", ts),
        )
        trading_ctx.db.commit()
        trading_ctx.db.execute(
            "SELECT blue_box_active FROM ewf_chart_analyses "
            "WHERE symbol = 'MSFT' AND timeframe = 'daily'"
        )
        row = trading_ctx.db.fetchone()
        assert row is not None
        assert row[0] is False

    def test_analyzed_at_defaults_to_now(self, trading_ctx):
        """analyzed_at defaults to approximately current timestamp."""
        ts = datetime(2026, 4, 4, 14, 0, 0, tzinfo=timezone.utc)
        trading_ctx.db.execute(
            "INSERT INTO ewf_chart_analyses (symbol, timeframe, fetched_at) "
            "VALUES (%s, %s, %s)",
            ("GOOG", "weekly", ts),
        )
        trading_ctx.db.commit()

        trading_ctx.db.execute(
            "SELECT EXTRACT(EPOCH FROM (NOW() - analyzed_at)) "
            "FROM ewf_chart_analyses WHERE symbol = 'GOOG'"
        )
        row = trading_ctx.db.fetchone()
        assert row is not None
        assert row[0] < 30, f"analyzed_at too far from NOW(): {row[0]}s"

    def test_nullable_optional_fields(self, trading_ctx):
        """Optional fields accept NULL values."""
        ts = datetime(2026, 4, 4, 15, 0, 0, tzinfo=timezone.utc)
        trading_ctx.db.execute(
            "INSERT INTO ewf_chart_analyses (symbol, timeframe, fetched_at) "
            "VALUES (%s, %s, %s)",
            ("TSLA", "4h", ts),
        )
        trading_ctx.db.commit()
        trading_ctx.db.execute(
            "SELECT bias, wave_position, blue_box_zone, confidence, summary "
            "FROM ewf_chart_analyses WHERE symbol = 'TSLA'"
        )
        row = trading_ctx.db.fetchone()
        assert row is not None
        assert all(v is None for v in row)
