"""Tests for OHLCV partitioning migration and startup hook.

DB-requiring tests are marked with @pytest.mark.requires_db.
Pure logic tests run without a database.
"""

import pytest
from datetime import date


class TestGenerateMonthRanges:
    """Test the month range generation logic used by the migration script."""

    def test_single_month_range(self):
        """When start and end are in the same month (beyond today), only one range."""
        from scripts.migrations.partition_ohlcv import generate_month_ranges

        # Use dates beyond today so max(end, today) = end
        future = date(2030, 6, 1)
        ranges = generate_month_ranges(future, date(2030, 6, 20), future_months=0)
        assert len(ranges) == 1
        assert ranges[0] == (date(2030, 6, 1), date(2030, 7, 1))

    def test_spans_year_boundary(self):
        from scripts.migrations.partition_ohlcv import generate_month_ranges

        ranges = generate_month_ranges(
            date(2025, 11, 1), date(2026, 2, 15), future_months=0
        )
        assert len(ranges) >= 4
        assert ranges[0][0] == date(2025, 11, 1)
        assert ranges[-1][1] >= date(2026, 3, 1)

    def test_includes_future_months(self):
        from scripts.migrations.partition_ohlcv import generate_month_ranges

        ranges = generate_month_ranges(
            date(2025, 1, 1), date(2025, 1, 31), future_months=4
        )
        # Should cover Jan 2025 through at least May 2025 (4 months after today or end)
        last_start = ranges[-1][0]
        assert last_start >= date(2025, 5, 1)

    def test_no_duplicate_ranges(self):
        from scripts.migrations.partition_ohlcv import generate_month_ranges

        ranges = generate_month_ranges(date(2024, 1, 1), date(2026, 4, 30))
        starts = [r[0] for r in ranges]
        assert len(starts) == len(set(starts)), "Duplicate month ranges found"

    def test_contiguous_ranges(self):
        """Each range's end == next range's start."""
        from scripts.migrations.partition_ohlcv import generate_month_ranges

        ranges = generate_month_ranges(date(2024, 1, 1), date(2025, 12, 31))
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0], (
                f"Gap between {ranges[i]} and {ranges[i+1]}"
            )


class TestEnsureOhlcvPartitions:
    """Test the startup hook that creates future partitions."""

    def test_skips_silently_when_table_not_partitioned(self):
        """When ohlcv is not partitioned (relkind != 'p'), function returns without error."""
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()
        # relkind = 'r' (regular table, not partitioned)
        mock_conn.execute.return_value.fetchone.return_value = ("r",)

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.db.pg_conn", return_value=cm):
            from quantstack.db import ensure_ohlcv_partitions

            ensure_ohlcv_partitions()  # should not raise

        # Only one execute call (the relkind check)
        assert mock_conn.execute.call_count == 1

    def test_skips_when_table_does_not_exist(self):
        """When ohlcv table doesn't exist, function returns without error."""
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.db.pg_conn", return_value=cm):
            from quantstack.db import ensure_ohlcv_partitions

            ensure_ohlcv_partitions()  # should not raise
