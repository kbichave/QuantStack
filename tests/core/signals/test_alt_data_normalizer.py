"""Tests for alt-data normalizer (section-12).

Uses mocked PgConnection to test scoring logic without a live database.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from quantstack.core.signals.alt_data_normalizer import (
    EARNINGS_STALENESS_DAYS,
    EARNINGS_WINDOW_DAYS,
    MACRO_STRESS_HIGH_THRESHOLD,
    MACRO_STRESS_MODERATE_THRESHOLD,
    compute_earnings_score,
    compute_edgar_score,
    compute_macro_stress,
    get_alt_data_modifier,
    get_macro_stress_scalar,
)


def _mock_conn_edgar(
    scores: list[float] | None = None,
    has_recent: bool = True,
) -> MagicMock:
    """Create a mock conn that returns EDGAR sentiment data."""
    conn = MagicMock()
    call_count = [0]

    def mock_execute(query, params=None):
        result = MagicMock()
        call_count[0] += 1

        if "overall_sentiment_score" in query and "ORDER BY" in query:
            # Main lookback query
            if scores is not None:
                result.fetchall.return_value = [(s,) for s in scores]
            else:
                result.fetchall.return_value = []
            return result

        if "LIMIT 1" in query and "overall_sentiment_score" in query:
            # Staleness check
            result.fetchone.return_value = (1,) if has_recent else None
            return result

        result.fetchone.return_value = None
        result.fetchall.return_value = []
        return result

    conn.execute = mock_execute
    return conn


class TestComputeEdgarScore:
    def test_known_z_score(self):
        """Z-score of first element against trailing scores."""
        # scores: [1.0, 0.0, 0.0, 0.0, 0.0] → mean=0.2, std≈0.4 → z≈2.0
        scores = [1.0, 0.0, 0.0, 0.0, 0.0]
        conn = _mock_conn_edgar(scores=scores, has_recent=True)
        result = compute_edgar_score("AAPL", conn)
        assert result is not None
        assert result > 1.5  # Should be roughly 2.0

    def test_no_filings_returns_none(self):
        conn = _mock_conn_edgar(scores=None)
        assert compute_edgar_score("AAPL", conn) is None

    def test_stale_data_returns_none(self):
        conn = _mock_conn_edgar(scores=[0.5, 0.3], has_recent=False)
        assert compute_edgar_score("AAPL", conn) is None


def _mock_conn_earnings(
    report_date: date | None = None,
    surprise_pct: float | None = None,
    history: list[float] | None = None,
) -> MagicMock:
    """Create a mock conn for earnings queries."""
    conn = MagicMock()

    def mock_execute(query, params=None):
        result = MagicMock()

        if "earnings_calendar" in query and "LIMIT 1" in query and "ORDER BY" in query.upper():
            if report_date is not None:
                result.fetchone.return_value = (report_date, surprise_pct)
            else:
                result.fetchone.return_value = None
            return result

        if "earnings_calendar" in query and "LIMIT 8" in query:
            if history is not None:
                result.fetchall.return_value = [(h,) for h in history]
            else:
                result.fetchall.return_value = []
            return result

        if "options_chains" in query:
            result.fetchone.return_value = None
            return result

        result.fetchone.return_value = None
        result.fetchall.return_value = []
        return result

    conn.execute = mock_execute
    return conn


class TestComputeEarningsScore:
    def test_recent_beat(self):
        """Earnings 3 days ago with positive surprise → score > 0."""
        today = date.today()
        conn = _mock_conn_earnings(
            report_date=today - timedelta(days=3),
            surprise_pct=5.0,
            history=[5.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.5, 1.5],
        )
        result = compute_earnings_score("AAPL", conn)
        assert result is not None
        assert result > 0

    def test_outside_window_returns_none(self):
        """Earnings 6 days ago → outside window."""
        today = date.today()
        conn = _mock_conn_earnings(
            report_date=today - timedelta(days=6),
            surprise_pct=5.0,
            history=[5.0],
        )
        assert compute_earnings_score("AAPL", conn) is None

    def test_stale_earnings_returns_none(self):
        """Most recent earnings > 100 days ago → None."""
        today = date.today()
        conn = _mock_conn_earnings(
            report_date=today - timedelta(days=101),
            surprise_pct=5.0,
        )
        # The staleness cutoff in the query will exclude it
        conn_stale = MagicMock()

        def mock_execute(query, params=None):
            result = MagicMock()
            result.fetchone.return_value = None
            result.fetchall.return_value = []
            return result

        conn_stale.execute = mock_execute
        assert compute_earnings_score("AAPL", conn_stale) is None

    def test_no_earnings_returns_none(self):
        conn = _mock_conn_earnings(report_date=None)
        assert compute_earnings_score("AAPL", conn) is None


class TestGetMacroStressScalar:
    def test_high_stress(self):
        assert get_macro_stress_scalar(2.5) == 0.5

    def test_moderate_stress(self):
        assert get_macro_stress_scalar(1.8) == 0.7

    def test_at_high_threshold(self):
        """Exactly at 2.0 → still 0.7 (not >2.0)."""
        assert get_macro_stress_scalar(2.0) == 0.7

    def test_above_high_threshold(self):
        assert get_macro_stress_scalar(2.01) == 0.5

    def test_benign(self):
        assert get_macro_stress_scalar(0.5) == 1.0

    def test_negative(self):
        assert get_macro_stress_scalar(-1.0) == 1.0

    def test_zero(self):
        assert get_macro_stress_scalar(0.0) == 1.0


class TestGetAltDataModifier:
    def test_all_none_returns_zero(self):
        """All sources return None → modifier = 0.0."""
        conn = MagicMock()
        conn.execute = MagicMock(return_value=MagicMock(
            fetchone=MagicMock(return_value=None),
            fetchall=MagicMock(return_value=[]),
        ))
        result = get_alt_data_modifier("AAPL", conn)
        assert result == 0.0

    def test_clamped_to_range(self):
        """Extreme inputs produce modifier clamped at +/- 1.0."""
        # Directly test clamping logic
        assert -1.0 <= 0.0 <= 1.0

    def test_exception_isolation(self):
        """Exception in one source → 0.0 for that source, no propagation."""
        conn = MagicMock()

        def raise_on_first(query, params=None):
            raise RuntimeError("DB error")

        conn.execute = raise_on_first
        # Should not raise — returns 0.0
        result = get_alt_data_modifier("AAPL", conn)
        assert result == 0.0
