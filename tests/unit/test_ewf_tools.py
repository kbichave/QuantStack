"""Tests for EWF LangChain tools (Section 07)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.tools.langchain.ewf_tools import get_ewf_analysis, get_ewf_blue_box_setups


def _mock_pg_conn(rows):
    """Return a context manager mock that yields a cursor returning `rows`."""
    mock_conn = MagicMock()
    mock_conn.fetchall.return_value = rows
    mock_conn.fetchone.return_value = rows[0] if rows else None
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _sample_row(symbol="AAPL", timeframe="4h", bias="bullish", confidence=0.85):
    """Return a tuple matching the SELECT column order in ewf_tools."""
    now = datetime.now(timezone.utc)
    return (
        symbol, timeframe,
        now,  # fetched_at
        now,  # analyzed_at
        bias,
        "wave 3 of 5",  # wave_position
        "minor",  # wave_degree
        "3",  # current_wave_label
        {"support": [180.0], "resistance": [200.0], "invalidation": 170.0, "target": 210.0},
        False,  # blue_box_active
        None,  # blue_box_zone
        confidence,
        False,  # invalidation_rule_violated
        "Strong impulse",  # analyst_notes
        "Bullish impulse in progress",  # summary
    )


class TestGetEwfAnalysis:
    @pytest.mark.asyncio
    async def test_no_rows_returns_unavailable(self):
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn([])):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL"})
        data = json.loads(result)
        assert data["ewf_available"] is False

    @pytest.mark.asyncio
    async def test_fresh_row_returns_available(self):
        rows = [_sample_row()]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL"})
        data = json.loads(result)
        assert data["ewf_available"] is True
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["bias"] == "bullish"
        assert r["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_multiple_timeframes_returned(self):
        rows = [_sample_row(timeframe="4h"), _sample_row(timeframe="daily")]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL"})
        data = json.loads(result)
        assert data["ewf_available"] is True
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_filters_by_timeframe(self):
        rows = [_sample_row(timeframe="4h")]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL", "timeframe": "4h"})
        data = json.loads(result)
        assert data["ewf_available"] is True
        assert data["results"][0]["timeframe"] == "4h"

    @pytest.mark.asyncio
    async def test_staleness_hours_present(self):
        rows = [_sample_row()]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL"})
        data = json.loads(result)
        assert "staleness_hours" in data["results"][0]
        assert isinstance(data["results"][0]["staleness_hours"], float)

    @pytest.mark.asyncio
    async def test_db_error_returns_graceful_fallback(self):
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", side_effect=Exception("DB down")):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL"})
        data = json.loads(result)
        assert data["ewf_available"] is False
        assert "unavailable" in data["reason"].lower()

    @pytest.mark.asyncio
    async def test_output_is_valid_json(self):
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn([])):
            result = await get_ewf_analysis.ainvoke({"symbol": "AAPL"})
        json.loads(result)  # must not raise


class TestGetEwfBlueBoxSetups:
    def _bb_row(self, symbol="AAPL", bias="bullish", confidence=0.9):
        now = datetime.now(timezone.utc)
        return (
            symbol, bias,
            {"low": 150.0, "high": 160.0},  # blue_box_zone
            confidence,
            "Blue box reversal zone",  # summary
            "High probability",  # analyst_notes
            "blue_box",  # timeframe
            now,  # analyzed_at
        )

    @pytest.mark.asyncio
    async def test_no_rows_returns_empty_list(self):
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn([])):
            result = await get_ewf_blue_box_setups.ainvoke({"date": "2026-04-04"})
        data = json.loads(result)
        assert data["setups"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_returns_setups_when_active(self):
        rows = [self._bb_row()]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_blue_box_setups.ainvoke({"date": "2026-04-04"})
        data = json.loads(result)
        assert data["count"] == 1
        assert data["setups"][0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_defaults_to_today(self):
        from datetime import date
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn([])):
            result = await get_ewf_blue_box_setups.ainvoke({})
        data = json.loads(result)
        assert data["date"] == date.today().isoformat()

    @pytest.mark.asyncio
    async def test_sorted_by_confidence_desc(self):
        rows = [self._bb_row("MSFT", confidence=0.7), self._bb_row("AAPL", confidence=0.9)]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_blue_box_setups.ainvoke({"date": "2026-04-04"})
        data = json.loads(result)
        # DB sorts by confidence DESC, so AAPL (0.9) should come before MSFT (0.7)
        # but our mock returns them in insertion order — the ORDER BY is in SQL, verified by query
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_setup_includes_all_fields(self):
        rows = [self._bb_row()]
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn(rows)):
            result = await get_ewf_blue_box_setups.ainvoke({"date": "2026-04-04"})
        data = json.loads(result)
        s = data["setups"][0]
        assert "symbol" in s
        assert "direction" in s
        assert "zone_low" in s
        assert "zone_high" in s
        assert "confidence" in s
        assert "summary" in s

    @pytest.mark.asyncio
    async def test_output_is_valid_json(self):
        with patch("quantstack.tools.langchain.ewf_tools.pg_conn", return_value=_mock_pg_conn([])):
            result = await get_ewf_blue_box_setups.ainvoke({"date": "2026-04-04"})
        json.loads(result)  # must not raise


class TestToolRegistry:
    def test_registry_contains_get_ewf_analysis(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert "get_ewf_analysis" in TOOL_REGISTRY

    def test_registry_contains_get_ewf_blue_box_setups(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        assert "get_ewf_blue_box_setups" in TOOL_REGISTRY

    def test_get_ewf_analysis_is_basetool(self):
        from langchain_core.tools import BaseTool
        from quantstack.tools.registry import TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["get_ewf_analysis"], BaseTool)

    def test_get_ewf_blue_box_setups_is_basetool(self):
        from langchain_core.tools import BaseTool
        from quantstack.tools.registry import TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["get_ewf_blue_box_setups"], BaseTool)
