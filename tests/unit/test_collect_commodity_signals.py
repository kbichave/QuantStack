"""Tests for commodity signals collector."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantstack.signal_engine.collectors.commodity import (
    collect_commodity_signals,
    _classify_rotation,
    _compute_usd_strength,
    _classify_regime,
)


def _indicator_df(
    values: list[float],
    days: int | None = None,
    start_offset_days: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic macro indicator DataFrame."""
    n = days or len(values)
    start = date.today() - timedelta(days=start_offset_days or n)
    dates = [start + timedelta(days=i) for i in range(n)]
    if len(values) < n:
        values = values + [values[-1]] * (n - len(values))
    return pd.DataFrame({"date": dates[:n], "value": values[:n]})


def _make_store(
    gold: pd.DataFrame | None = None,
    silver: pd.DataFrame | None = None,
    copper: pd.DataFrame | None = None,
    eurusd: pd.DataFrame | None = None,
    usdjpy: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock store with load_macro_indicator returning per-indicator data."""
    indicator_map = {
        "GOLD": gold if gold is not None else pd.DataFrame(),
        "SILVER": silver if silver is not None else pd.DataFrame(),
        "COPPER": copper if copper is not None else pd.DataFrame(),
        "EURUSD": eurusd if eurusd is not None else pd.DataFrame(),
        "USDJPY": usdjpy if usdjpy is not None else pd.DataFrame(),
    }

    store = MagicMock()
    store.load_macro_indicator.side_effect = (
        lambda ind, start_date=None: indicator_map.get(ind, pd.DataFrame())
    )
    return store


class TestCollectCommoditySignals:
    """Tests for collect_commodity_signals."""

    @pytest.mark.asyncio
    async def test_computes_gold_silver_ratio(self) -> None:
        gold = _indicator_df([2000.0] * 30, days=30)
        silver = _indicator_df([25.0] * 30, days=30)
        store = _make_store(gold=gold, silver=silver)
        result = await collect_commodity_signals("SPY", store)
        assert abs(result["gold_silver_ratio"] - 80.0) < 0.01

    @pytest.mark.asyncio
    async def test_computes_copper_gold_ratio(self) -> None:
        gold = _indicator_df([2000.0] * 30, days=30)
        copper = _indicator_df([4.5] * 30, days=30)
        store = _make_store(gold=gold, copper=copper)
        result = await collect_commodity_signals("SPY", store)
        expected = 4.5 / 2000.0
        assert abs(result["copper_gold_ratio"] - expected) < 0.0001

    def test_sector_rotation_favor_cyclicals(self) -> None:
        """Copper up, gold down -> favor_cyclicals."""
        assert _classify_rotation(gold_ret=-1.0, copper_ret=2.0) == "favor_cyclicals"

    def test_sector_rotation_favor_defensives(self) -> None:
        """Gold up, copper down -> favor_defensives."""
        assert _classify_rotation(gold_ret=2.0, copper_ret=-1.0) == "favor_defensives"

    def test_sector_rotation_inflationary(self) -> None:
        """Both up -> inflationary."""
        assert _classify_rotation(gold_ret=2.0, copper_ret=1.0) == "inflationary"

    def test_sector_rotation_neutral(self) -> None:
        """Both down -> neutral."""
        assert _classify_rotation(gold_ret=-1.0, copper_ret=-2.0) == "neutral"

    @pytest.mark.asyncio
    async def test_returns_empty_when_data_stale(self) -> None:
        """Gold data > 2 days old -> empty dict."""
        gold = _indicator_df([2000.0] * 10, days=10, start_offset_days=15)
        store = _make_store(gold=gold)
        result = await collect_commodity_signals("SPY", store)
        assert result == {}

    @pytest.mark.asyncio
    async def test_handles_missing_forex_data(self) -> None:
        """Missing forex -> usd_strength_proxy is None."""
        gold = _indicator_df([2000.0] * 30, days=30)
        store = _make_store(gold=gold, eurusd=pd.DataFrame(), usdjpy=pd.DataFrame())
        result = await collect_commodity_signals("SPY", store)
        assert result["usd_strength_proxy"] is None

    def test_computes_usd_strength_proxy(self) -> None:
        """USD strength from EUR/USD and USD/JPY."""
        eurusd = _indicator_df(
            [1.10, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04], days=7
        )
        usdjpy = _indicator_df(
            [140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0], days=7
        )
        result = _compute_usd_strength(eurusd, usdjpy)
        assert result is not None
        assert result > 0

    def test_commodity_regime_risk_off(self) -> None:
        assert _classify_regime(0.75) == "risk_off"

    def test_commodity_regime_risk_on(self) -> None:
        assert _classify_regime(0.25) == "risk_on"

    def test_commodity_regime_neutral(self) -> None:
        assert _classify_regime(0.50) == "neutral"

    def test_commodity_regime_unknown_when_none(self) -> None:
        assert _classify_regime(None) == "unknown"
