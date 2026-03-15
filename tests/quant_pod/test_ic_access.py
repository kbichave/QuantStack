# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Enhancement 1 (Granular IC Access) and Enhancement 5 (Execution Feedback Loop).

Tests are structured to work without a live MCP server by exercising the
helper functions and MCP tool logic with mocked dependencies.
"""

from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# IC Output Cache helpers
# =============================================================================


def test_ic_cache_set_and_get():
    from quant_pod.mcp.server import _ic_cache_set, _ic_cache_get, _ic_output_cache

    symbol, ic_name = "TEST", "regime_detector_ic"
    _ic_cache_set(symbol, ic_name, "trending_up, confidence 0.85")

    result = _ic_cache_get(symbol, ic_name)
    assert result == "trending_up, confidence 0.85"


def test_ic_cache_miss_for_unknown():
    from quant_pod.mcp.server import _ic_cache_get

    result = _ic_cache_get("NOMATCH", "nonexistent_ic")
    assert result is None


def test_ic_cache_ttl_expiry(monkeypatch):
    """Cache entries expire after TTL."""
    from quant_pod.mcp import server as srv

    srv._ic_output_cache.clear()
    srv._ic_cache_set("SPY", "volatility_ic", "ATR contracting")

    # Fake that the entry is old by manipulating its timestamp
    key = "SPY::volatility_ic"
    srv._ic_output_cache[key]["ts"] = time.monotonic() - (srv._IC_CACHE_TTL_SECS + 1)

    result = srv._ic_cache_get("SPY", "volatility_ic")
    assert result is None
    assert key not in srv._ic_output_cache  # expired entry cleaned up


def test_populate_ic_cache_from_result():
    """_populate_ic_cache_from_result extracts and caches IC outputs from crew result."""
    from quant_pod.mcp import server as srv

    srv._ic_output_cache.clear()
    symbol = "AAPL"

    # Mock a crew result with tasks_output
    mock_task_output = MagicMock()
    mock_task_output.raw = "IC raw output text"

    mock_result = MagicMock()
    mock_result.tasks_output = [mock_task_output] * 3  # 3 task outputs

    srv._populate_ic_cache_from_result(symbol, mock_result)

    # First 3 ICs in IC_AGENT_ORDER should be cached
    from quant_pod.crews.trading_crew import IC_AGENT_ORDER
    for i in range(3):
        ic_name = IC_AGENT_ORDER[i]
        cached = srv._ic_cache_get(symbol, ic_name)
        assert cached == "IC raw output text", f"IC {ic_name} not cached"


def test_populate_ic_cache_handles_missing_tasks_output():
    """Gracefully handles crew results without tasks_output."""
    from quant_pod.mcp import server as srv

    mock_result = MagicMock(spec=[])  # No tasks_output attribute
    srv._populate_ic_cache_from_result("SPY", mock_result)  # Should not raise


def test_populate_ic_cache_handles_empty_tasks_output():
    """Gracefully handles crew results with empty tasks_output."""
    from quant_pod.mcp import server as srv

    mock_result = MagicMock()
    mock_result.tasks_output = []
    srv._populate_ic_cache_from_result("SPY", mock_result)  # Should not raise


# =============================================================================
# list_ics
# =============================================================================


@pytest.mark.asyncio
async def test_list_ics_returns_catalog():
    from quant_pod.mcp.server import list_ics

    # FastMCP wraps tool functions in FunctionTool; call via .fn
    result = await list_ics.fn()

    assert result["success"] is True
    assert result["total_ics"] == 13
    ics = result["ics"]
    assert any(ic["name"] == "regime_detector_ic" for ic in ics)
    assert any(ic["name"] == "data_ingestion_ic" for ic in ics)

    # Verify structure of each IC entry
    for ic in ics:
        assert "name" in ic
        assert "description" in ic
        assert "pod" in ic
        assert "capabilities" in ic
        assert "asset_classes" in ic


@pytest.mark.asyncio
async def test_list_ics_returns_pods():
    from quant_pod.mcp.server import list_ics

    result = await list_ics.fn()

    assert "pods" in result
    pods = result["pods"]
    pod_names = [p["name"] for p in pods]
    assert "technicals_pod_manager" in pod_names
    assert "risk_pod_manager" in pod_names

    for pod in pods:
        assert "constituent_ics" in pod


# =============================================================================
# get_last_ic_output
# =============================================================================


@pytest.mark.asyncio
async def test_get_last_ic_output_cache_miss():
    from quant_pod.mcp import server as srv
    from quant_pod.mcp.server import get_last_ic_output

    # Ensure no cached entry exists
    key = "FRESHSYM::trend_momentum_ic"
    srv._ic_output_cache.pop(key, None)

    result = await get_last_ic_output.fn("FRESHSYM", "trend_momentum_ic")

    assert result["success"] is True
    assert result["cache_miss"] is True
    assert "note" in result


@pytest.mark.asyncio
async def test_get_last_ic_output_cache_hit():
    from quant_pod.mcp import server as srv
    from quant_pod.mcp.server import get_last_ic_output

    srv._ic_cache_set("MSFT", "volatility_ic", "vol contracting, ATR declining")

    result = await get_last_ic_output.fn("MSFT", "volatility_ic")

    assert result["success"] is True
    assert result["cache_miss"] is False
    assert result["raw_output"] == "vol contracting, ATR declining"
    assert result["ic_name"] == "volatility_ic"
    assert result["symbol"] == "MSFT"


# =============================================================================
# run_ic input validation
# =============================================================================


@pytest.mark.asyncio
async def test_run_ic_invalid_name_returns_error():
    from quant_pod.mcp.server import run_ic

    result = await run_ic.fn("nonexistent_ic", "SPY")

    assert result["success"] is False
    assert "Unknown IC" in result["error"]


# =============================================================================
# run_pod input validation
# =============================================================================


@pytest.mark.asyncio
async def test_run_pod_invalid_name_returns_error():
    from quant_pod.mcp.server import run_pod

    result = await run_pod.fn("nonexistent_pod", "SPY")

    assert result["success"] is False
    assert "Unknown pod" in result["error"]


# =============================================================================
# run_crew_subset input validation
# =============================================================================


@pytest.mark.asyncio
async def test_run_crew_subset_invalid_ic_returns_error():
    from quant_pod.mcp.server import run_crew_subset

    result = await run_crew_subset.fn(["bad_ic_name", "regime_detector_ic"], "SPY")

    assert result["success"] is False
    assert "Unknown ICs" in result["error"]


# =============================================================================
# get_fill_quality
# =============================================================================


@pytest.mark.asyncio
async def test_get_fill_quality_order_not_found():
    """Returns error when order_id doesn't exist in fills table."""
    mock_ctx = MagicMock()
    mock_ctx.db.execute.return_value.fetchone.return_value = None

    with patch("quant_pod.mcp.server._require_ctx", return_value=mock_ctx):
        from quant_pod.mcp.server import get_fill_quality
        result = await get_fill_quality.fn("nonexistent_order_id")

    assert result["success"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_get_fill_quality_returns_analysis():
    """Returns quality analysis for a known fill."""
    mock_ctx = MagicMock()
    mock_ctx.db.execute.return_value.fetchone.return_value = (
        "order_123", "SPY", "buy", 450.25, 10, 2.5, 1.0, "2026-03-15 09:30:00"
    )

    with patch("quant_pod.mcp.server._require_ctx", return_value=mock_ctx):
        # Don't mock DataStore — let it fail gracefully (no VWAP data in test env)
        from quant_pod.mcp.server import get_fill_quality
        result = await get_fill_quality.fn("order_123")

    assert result["success"] is True
    assert result["order_id"] == "order_123"
    assert result["symbol"] == "SPY"
    assert result["fill_price"] == 450.25
    assert result["slippage_bps"] == 2.5
    assert "quality_note" in result


# =============================================================================
# get_position_monitor
# =============================================================================


@pytest.mark.asyncio
async def test_get_position_monitor_no_position():
    """Returns has_position=False when no open position exists."""
    mock_ctx = MagicMock()
    mock_ctx.portfolio.get_position.return_value = None

    with patch("quant_pod.mcp.server._require_ctx", return_value=mock_ctx):
        from quant_pod.mcp.server import get_position_monitor
        result = await get_position_monitor.fn("MISSING")

    assert result["success"] is True
    assert result["has_position"] is False


@pytest.mark.asyncio
async def test_get_position_monitor_open_position():
    """Returns comprehensive position status for an open position."""
    mock_pos = MagicMock()
    mock_pos.current_price = 510.0
    mock_pos.avg_cost = 500.0
    mock_pos.quantity = 20
    mock_pos.unrealized_pnl = 200.0

    mock_ctx = MagicMock()
    mock_ctx.portfolio.get_position.return_value = mock_pos
    # positions table query
    mock_ctx.db.execute.return_value.fetchone.return_value = ("2026-03-10 09:30:00",)

    with (
        patch("quant_pod.mcp.server._require_ctx", return_value=mock_ctx),
        patch("quant_pod.mcp.server._detect_regime_for_symbol", new=AsyncMock(
            return_value={"trend": "trending_up", "volatility": "normal", "confidence": 0.82}
        )),
    ):
        from quant_pod.mcp.server import get_position_monitor
        result = await get_position_monitor.fn("SPY")

    assert result["success"] is True
    assert result["has_position"] is True
    assert result["symbol"] == "SPY"
    assert result["pnl_pct"] == pytest.approx(2.0, rel=0.01)
    assert result["unrealized_pnl"] == 200.0
    assert "flags" in result
    assert "recommended_action" in result


# =============================================================================
# Scheduler script
# =============================================================================


def test_scheduler_schedule_definition():
    """Scheduler defines 4 jobs covering key market times."""
    from scripts.scheduler import SCHEDULE

    assert len(SCHEDULE) == 4
    labels = [j["label"] for j in SCHEDULE]
    assert "morning_routine" in labels
    assert "midday_review" in labels
    assert "preclose_review" in labels
    assert "weekly_reflect" in labels

    # Weekly reflect must be Friday only
    weekly = next(j for j in SCHEDULE if j["label"] == "weekly_reflect")
    assert weekly["weekdays"] == "fri"

    # Morning routine must be Mon-Fri
    morning = next(j for j in SCHEDULE if j["label"] == "morning_routine")
    assert morning["weekdays"] == "mon-fri"
    assert morning["hour"] == 9
    assert morning["minute"] == 15


def test_scheduler_dry_run(capsys, monkeypatch):
    """Dry run prints session information without launching Claude Code."""
    from scripts.scheduler import run_session

    run_session("test prompt", "test_label", dry_run=True)

    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "test_label" in captured.out
