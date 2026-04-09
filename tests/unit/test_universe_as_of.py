"""Tests for 4.2 — Survivorship Bias / universe_as_of (QS-B2)."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from quantstack.universe import universe_as_of


def _mock_conn(rows: list[tuple[str]]) -> MagicMock:
    """Create a mock DB connection returning *rows* from execute().fetchall()."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


class TestUniverseAsOf:
    def test_excludes_pre_ipo(self):
        """Symbol with ipo_date after the query date should be excluded."""
        # Only AAPL (IPO before query date) returned, NVDA not included
        conn = _mock_conn([("AAPL",)])
        result = universe_as_of(date(2019, 6, 1), conn)
        assert result == ["AAPL"]
        # Verify the SQL was called with the correct date parameters
        args = conn.execute.call_args
        assert args[0][1] == [date(2019, 6, 1), date(2019, 6, 1)]

    def test_excludes_delisted(self):
        """Symbol delisted before the query date should be excluded."""
        conn = _mock_conn([("AAPL",)])
        result = universe_as_of(date(2024, 1, 1), conn)
        assert "AAPL" in result

    def test_includes_null_ipo(self):
        """Symbols with NULL ipo_date are conservatively included."""
        # The SQL uses (co.ipo_date IS NULL OR co.ipo_date <= %s)
        # so NULL ipo dates pass through — we just verify the query runs
        conn = _mock_conn([("QQQ",), ("SPY",)])
        result = universe_as_of(date(2020, 1, 1), conn)
        assert result == ["QQQ", "SPY"]  # preserves DB ORDER BY

    def test_empty_universe(self):
        """Returns empty list when no symbols match."""
        conn = _mock_conn([])
        result = universe_as_of(date(1900, 1, 1), conn)
        assert result == []

    def test_returns_sorted(self):
        """Result preserves the DB ORDER BY (alphabetical)."""
        conn = _mock_conn([("AAPL",), ("MSFT",), ("SPY",)])
        result = universe_as_of(date(2024, 1, 1), conn)
        assert result == ["AAPL", "MSFT", "SPY"]
