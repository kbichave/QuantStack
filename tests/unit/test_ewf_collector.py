"""Tests for EWF signal collector (Section 05)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.signal_engine.collectors.ewf_collector import (
    collect_ewf,
    _collect_ewf_sync,
)


@pytest.fixture(autouse=True)
def _clean_ewf_table(trading_ctx):
    """Delete all rows from ewf_chart_analyses before each test."""
    trading_ctx.db.execute("DELETE FROM ewf_chart_analyses")
    trading_ctx.db.commit()
    yield


def _insert_ewf_row(trading_ctx, *, symbol="AAPL", timeframe="4h",
                     bias="bullish", confidence=0.85, hours_ago=1.0,
                     blue_box_active=False, blue_box_zone=None,
                     key_levels=None, wave_position="wave 3 of 5",
                     wave_degree="minor", current_wave_label="3",
                     summary="Bullish impulse in progress"):
    """Insert a test row into ewf_chart_analyses with analyzed_at = NOW() - hours_ago."""
    kl = json.dumps(key_levels) if key_levels else None
    bbz = json.dumps(blue_box_zone) if blue_box_zone else None
    trading_ctx.db.execute(
        "INSERT INTO ewf_chart_analyses "
        "(symbol, timeframe, fetched_at, analyzed_at, bias, confidence, "
        " blue_box_active, blue_box_zone, key_levels, wave_position, "
        " wave_degree, current_wave_label, summary) "
        "VALUES (%s, %s, NOW() - INTERVAL '%s hours', "
        "        NOW() - INTERVAL '%s hours', "
        "        %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)",
        (symbol, timeframe, hours_ago, hours_ago,
         bias, confidence, blue_box_active, bbz, kl,
         wave_position, wave_degree, current_wave_label, summary),
    )
    trading_ctx.db.commit()


class TestCollectEwf:
    """Unit tests for collect_ewf."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_data(self, trading_ctx):
        result = await collect_ewf("AAPL", MagicMock())
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_outside_ttl(self, trading_ctx):
        """4h TTL is 6 hours; a 7-hour-old row should not be returned."""
        _insert_ewf_row(trading_ctx, timeframe="4h", hours_ago=7.0)
        result = await collect_ewf("AAPL", MagicMock())
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_data_within_ttl(self, trading_ctx):
        _insert_ewf_row(trading_ctx, bias="bullish", confidence=0.85, hours_ago=1.0)
        result = await collect_ewf("AAPL", MagicMock())
        assert result["ewf_bias"] == "bullish"
        assert result["ewf_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self):
        """DB exception is caught and {} returned."""
        with patch(
            "quantstack.signal_engine.collectors.ewf_collector.pg_conn",
            side_effect=Exception("DB down"),
        ):
            result = await collect_ewf("AAPL", MagicMock())
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self):
        """Timeout returns {} without raising."""
        def slow_sync(symbol):
            import time
            time.sleep(15)
            return {}

        with patch(
            "quantstack.signal_engine.collectors.ewf_collector._collect_ewf_sync",
            side_effect=slow_sync,
        ):
            result = await collect_ewf("AAPL", MagicMock())
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_most_recent_fresh_row(self, trading_ctx):
        """When multiple timeframes are fresh, most recent wins."""
        _insert_ewf_row(trading_ctx, timeframe="blue_box", hours_ago=2.0,
                         bias="bearish", confidence=0.7)
        _insert_ewf_row(trading_ctx, timeframe="4h", hours_ago=0.5,
                         bias="bullish", confidence=0.9)
        result = await collect_ewf("AAPL", MagicMock())
        assert result["ewf_timeframe_used"] == "4h"
        assert result["ewf_bias"] == "bullish"

    @pytest.mark.asyncio
    async def test_store_parameter_compatibility(self, trading_ctx):
        """Accepts both MagicMock and None as store."""
        result1 = await collect_ewf("AAPL", MagicMock())
        result2 = await collect_ewf("AAPL", None)
        assert result1 == result2 == {}

    @pytest.mark.asyncio
    async def test_return_dict_has_all_expected_keys(self, trading_ctx):
        _insert_ewf_row(
            trading_ctx,
            key_levels={"support": [180.0], "resistance": [200.0],
                        "invalidation": 175.0, "target": 210.0},
            blue_box_active=True,
            blue_box_zone={"low": 185.0, "high": 195.0},
        )
        result = await collect_ewf("AAPL", MagicMock())
        expected_keys = {
            "ewf_bias", "ewf_turning_signal", "ewf_wave_position",
            "ewf_wave_degree", "ewf_current_wave_label", "ewf_confidence",
            "ewf_key_support", "ewf_key_resistance",
            "ewf_invalidation_level", "ewf_target",
            "ewf_blue_box_active", "ewf_blue_box_low", "ewf_blue_box_high",
            "ewf_summary", "ewf_projected_path",
            "ewf_timeframe_used", "ewf_age_hours",
        }
        assert expected_keys == set(result.keys())
        assert result["ewf_key_support"] == [180.0]
        assert result["ewf_invalidation_level"] == 175.0
        assert result["ewf_blue_box_active"] is True
        assert result["ewf_blue_box_low"] == 185.0

    @pytest.mark.asyncio
    async def test_age_hours_reflects_actual_age(self, trading_ctx):
        _insert_ewf_row(trading_ctx, hours_ago=1.5)
        result = await collect_ewf("AAPL", MagicMock())
        assert 1.3 < result["ewf_age_hours"] < 1.7
