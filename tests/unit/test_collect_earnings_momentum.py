"""Tests for earnings momentum signal collector."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantstack.signal_engine.collectors.earnings_momentum import (
    collect_earnings_momentum,
)


def _make_store(
    df: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock store with load_earnings_calendar returning *df*."""
    store = MagicMock()
    store.load_earnings_calendar.return_value = (
        df if df is not None else pd.DataFrame()
    )
    return store


def _earnings_df(
    surprises: list[float],
    start_date: date | None = None,
    interval_days: int = 90,
) -> pd.DataFrame:
    """Generate synthetic earnings calendar data."""
    start = start_date or (date.today() - timedelta(days=len(surprises) * interval_days))
    rows = []
    for i, sp in enumerate(surprises):
        rows.append(
            {
                "symbol": "AAPL",
                "report_date": start + timedelta(days=i * interval_days),
                "surprise_pct": sp,
                "estimate": 1.0,
                "reported_eps": 1.0 + sp / 100,
            }
        )
    return pd.DataFrame(rows)


class TestCollectEarningsMomentum:
    """Tests for collect_earnings_momentum."""

    @pytest.mark.asyncio
    async def test_counts_consecutive_beats(self) -> None:
        """4 positive surprises -> consecutive_beats=4."""
        df = _earnings_df([3.0, 5.0, 2.0, 8.0])
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result["consecutive_beats"] == 4
        assert result["consecutive_misses"] == 0

    @pytest.mark.asyncio
    async def test_resets_streak_on_direction_change(self) -> None:
        """[+,+,+,-] -> most recent is miss, so consecutive_misses=1, beats=0."""
        df = _earnings_df([3.0, 5.0, 2.0, -4.0])
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result["consecutive_misses"] == 1
        assert result["consecutive_beats"] == 0

    @pytest.mark.asyncio
    async def test_avg_surprise_pct_4q(self) -> None:
        df = _earnings_df([2.0, 4.0, 6.0, 8.0])
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert abs(result["avg_surprise_pct_4q"] - 5.0) < 0.01

    @pytest.mark.asyncio
    async def test_drift_active_when_large_recent_surprise(self) -> None:
        """drift_active True when |surprise_pct| > 5% and < 30 days old."""
        recent_date = date.today() - timedelta(days=10)
        df = _earnings_df([1.0, 2.0, 3.0], start_date=date.today() - timedelta(days=270))
        recent_row = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "report_date": recent_date,
                    "surprise_pct": 12.0,
                    "estimate": 1.0,
                    "reported_eps": 1.12,
                }
            ]
        )
        df = pd.concat([df, recent_row], ignore_index=True)
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result["drift_active"] is True

    @pytest.mark.asyncio
    async def test_drift_inactive_when_old(self) -> None:
        """drift_active False when > 30 days old."""
        old_date = date.today() - timedelta(days=60)
        df = _earnings_df(
            [1.0, 2.0, 3.0, 15.0],
            start_date=date.today() - timedelta(days=330),
            interval_days=90,
        )
        df.iloc[-1, df.columns.get_loc("report_date")] = old_date
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result["drift_active"] is False

    @pytest.mark.asyncio
    async def test_drift_inactive_when_small_surprise(self) -> None:
        """drift_active False when |surprise_pct| < 5%."""
        recent_date = date.today() - timedelta(days=5)
        df = _earnings_df([1.0, 2.0, 3.0], start_date=date.today() - timedelta(days=270))
        recent_row = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "report_date": recent_date,
                    "surprise_pct": 2.0,
                    "estimate": 1.0,
                    "reported_eps": 1.02,
                }
            ]
        )
        df = pd.concat([df, recent_row], ignore_index=True)
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result["drift_active"] is False

    @pytest.mark.asyncio
    async def test_estimates_days_to_next_earnings(self) -> None:
        """With 4 quarters at 90-day intervals, next is ~90 days after last."""
        df = _earnings_df([1.0, 2.0, 3.0, 4.0], interval_days=90)
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result["days_to_next_earnings"] is not None
        assert isinstance(result["days_to_next_earnings"], int)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_earnings(self) -> None:
        store = _make_store(pd.DataFrame())
        result = await collect_earnings_momentum("AAPL", store)
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_quarter_no_days_to_next(self) -> None:
        """Single quarter: days_to_next_earnings should be None."""
        df = _earnings_df([5.0])
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert result.get("days_to_next_earnings") is None

    @pytest.mark.asyncio
    async def test_momentum_score_in_range(self) -> None:
        df = _earnings_df([10.0, -5.0, 8.0, 12.0])
        store = _make_store(df)
        result = await collect_earnings_momentum("AAPL", store)
        assert -1 <= result["earnings_momentum_score"] <= 1
