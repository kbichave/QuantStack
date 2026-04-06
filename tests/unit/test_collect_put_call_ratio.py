"""Tests for put-call ratio signal collector."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantstack.signal_engine.collectors.put_call_ratio import (
    collect_put_call_ratio,
)


def _make_store(df: pd.DataFrame) -> MagicMock:
    """Build a mock store with load_options_volume_summary returning *df*."""
    store = MagicMock()
    store.load_options_volume_summary.return_value = df
    return store


def _volume_df(
    days: int = 30,
    base_put: int = 1000,
    base_call: int = 1200,
    start_date: date | None = None,
) -> pd.DataFrame:
    """Generate a synthetic volume summary DataFrame."""
    start = start_date or (date.today() - timedelta(days=days))
    dates = [start + timedelta(days=i) for i in range(days)]
    return pd.DataFrame(
        {
            "date": dates,
            "put_volume": [base_put + i * 10 for i in range(days)],
            "call_volume": [base_call + i * 5 for i in range(days)],
        }
    )


class TestCollectPutCallRatio:
    """Tests for collect_put_call_ratio."""

    @pytest.mark.asyncio
    async def test_computes_pcr_from_known_volumes(self) -> None:
        df = _volume_df(days=30, base_put=600, base_call=1000)
        store = _make_store(df)
        result = await collect_put_call_ratio("AAPL", store)
        assert "pcr_raw" in result
        expected_pcr = 890 / 1145
        assert abs(result["pcr_raw"] - expected_pcr) < 0.01

    @pytest.mark.asyncio
    async def test_computes_10d_sma(self) -> None:
        df = _volume_df(days=30, base_put=500, base_call=1000)
        store = _make_store(df)
        result = await collect_put_call_ratio("AAPL", store)
        assert "pcr_10d_sma" in result
        assert isinstance(result["pcr_10d_sma"], float)

    @pytest.mark.asyncio
    async def test_percentile_signal_high_pcr_bullish(self) -> None:
        """PCR at >80th percentile should give contrarian bullish (+1)."""
        days = 60
        start = date.today() - timedelta(days=days)
        dates = [start + timedelta(days=i) for i in range(days)]
        put_vol = [500] * 50 + [3000] * 10
        call_vol = [1000] * 60
        df = pd.DataFrame(
            {"date": dates, "put_volume": put_vol, "call_volume": call_vol}
        )
        store = _make_store(df)
        result = await collect_put_call_ratio("SPY", store)
        assert result["pcr_signal"] == 1

    @pytest.mark.asyncio
    async def test_percentile_signal_low_pcr_bearish(self) -> None:
        """PCR at <20th percentile should give contrarian bearish (-1)."""
        days = 60
        start = date.today() - timedelta(days=days)
        dates = [start + timedelta(days=i) for i in range(days)]
        put_vol = [2000] * 50 + [100] * 10
        call_vol = [1000] * 60
        df = pd.DataFrame(
            {"date": dates, "put_volume": put_vol, "call_volume": call_vol}
        )
        store = _make_store(df)
        result = await collect_put_call_ratio("SPY", store)
        assert result["pcr_signal"] == -1

    @pytest.mark.asyncio
    async def test_returns_empty_when_total_volume_below_threshold(self) -> None:
        df = _volume_df(days=30, base_put=100, base_call=100)
        df.iloc[-1, df.columns.get_loc("put_volume")] = 50
        df.iloc[-1, df.columns.get_loc("call_volume")] = 50
        store = _make_store(df)
        result = await collect_put_call_ratio("AAPL", store)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_options_data(self) -> None:
        empty_df = pd.DataFrame(columns=["date", "put_volume", "call_volume"])
        store = _make_store(empty_df)
        result = await collect_put_call_ratio("AAPL", store)
        assert result == {}

    @pytest.mark.asyncio
    async def test_includes_pcr_percentile_30d(self) -> None:
        df = _volume_df(days=40, base_put=800, base_call=1000)
        store = _make_store(df)
        result = await collect_put_call_ratio("AAPL", store)
        assert "pcr_percentile_30d" in result
        assert 0 <= result["pcr_percentile_30d"] <= 1

    @pytest.mark.asyncio
    async def test_handles_call_volume_zero(self) -> None:
        """Division by zero guard: rows with call_volume==0 are excluded."""
        days = 30
        start = date.today() - timedelta(days=days)
        dates = [start + timedelta(days=i) for i in range(days)]
        put_vol = [1000] * days
        call_vol = [0] * days
        df = pd.DataFrame(
            {"date": dates, "put_volume": put_vol, "call_volume": call_vol}
        )
        store = _make_store(df)
        result = await collect_put_call_ratio("AAPL", store)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_neutral_when_history_short(self) -> None:
        """With <20 days of history, signal should be 0 (neutral)."""
        df = _volume_df(days=10, base_put=800, base_call=1000)
        store = _make_store(df)
        result = await collect_put_call_ratio("AAPL", store)
        assert result.get("pcr_signal") == 0
