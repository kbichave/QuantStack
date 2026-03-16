# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Phase 5 MCP tools: regime matrix CRUD, multi-analysis,
conflict resolution.
"""

from __future__ import annotations

import uuid

import pytest

from quant_pod.context import create_trading_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:", session_id=str(uuid.uuid4()))
    yield context
    context.db.close()


@pytest.fixture
def _inject_ctx(ctx):
    import quant_pod.mcp.server as srv
    original = srv._ctx
    srv._ctx = ctx
    yield ctx
    srv._ctx = original


def _fn(tool_obj):
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


# ---------------------------------------------------------------------------
# get_regime_strategies
# ---------------------------------------------------------------------------


class TestGetRegimeStrategies:
    @pytest.mark.asyncio
    async def test_empty_initially(self, _inject_ctx):
        from quant_pod.mcp.server import get_regime_strategies

        result = await _fn(get_regime_strategies)("trending_up")
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_allocations_after_set(self, _inject_ctx):
        from quant_pod.mcp.server import set_regime_allocation, get_regime_strategies

        await _fn(set_regime_allocation)(
            regime="trending_up",
            allocations=[
                {"strategy_id": "s1", "allocation_pct": 0.30, "confidence": 0.8},
                {"strategy_id": "s2", "allocation_pct": 0.20, "confidence": 0.6},
            ],
        )

        result = await _fn(get_regime_strategies)("trending_up")
        assert result["success"] is True
        assert result["total"] == 2
        ids = [a["strategy_id"] for a in result["allocations"]]
        assert "s1" in ids
        assert "s2" in ids


# ---------------------------------------------------------------------------
# set_regime_allocation
# ---------------------------------------------------------------------------


class TestSetRegimeAllocation:
    @pytest.mark.asyncio
    async def test_inserts_allocations(self, _inject_ctx):
        from quant_pod.mcp.server import set_regime_allocation

        result = await _fn(set_regime_allocation)(
            regime="ranging",
            allocations=[{"strategy_id": "s1", "allocation_pct": 0.25}],
        )
        assert result["success"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_upserts_existing(self, _inject_ctx):
        from quant_pod.mcp.server import set_regime_allocation, get_regime_strategies

        await _fn(set_regime_allocation)(
            regime="ranging",
            allocations=[{"strategy_id": "s1", "allocation_pct": 0.25}],
        )
        # Update
        await _fn(set_regime_allocation)(
            regime="ranging",
            allocations=[{"strategy_id": "s1", "allocation_pct": 0.40}],
        )

        result = await _fn(get_regime_strategies)("ranging")
        assert result["allocations"][0]["allocation_pct"] == 0.40

    @pytest.mark.asyncio
    async def test_rejects_over_100_pct(self, _inject_ctx):
        from quant_pod.mcp.server import set_regime_allocation

        result = await _fn(set_regime_allocation)(
            regime="trending_up",
            allocations=[
                {"strategy_id": "s1", "allocation_pct": 0.60},
                {"strategy_id": "s2", "allocation_pct": 0.50},
            ],
        )
        assert result["success"] is False
        assert "exceeds" in result["error"]


# ---------------------------------------------------------------------------
# resolve_portfolio_conflicts
# ---------------------------------------------------------------------------


class TestResolvePortfolioConflicts:
    @pytest.mark.asyncio
    async def test_no_conflicts(self, _inject_ctx):
        from quant_pod.mcp.server import resolve_portfolio_conflicts

        result = await _fn(resolve_portfolio_conflicts)(
            proposed_trades=[
                {"symbol": "SPY", "action": "buy", "confidence": 0.8, "strategy_id": "s1", "capital_pct": 0.05},
                {"symbol": "QQQ", "action": "sell", "confidence": 0.7, "strategy_id": "s2", "capital_pct": 0.05},
            ]
        )
        assert result["success"] is True
        assert len(result["resolved_trades"]) == 2
        assert result["conflicts_count"] == 0

    @pytest.mark.asyncio
    async def test_conflict_skip(self, _inject_ctx):
        from quant_pod.mcp.server import resolve_portfolio_conflicts

        result = await _fn(resolve_portfolio_conflicts)(
            proposed_trades=[
                {"symbol": "SPY", "action": "buy", "confidence": 0.85, "strategy_id": "s1", "capital_pct": 0.05},
                {"symbol": "SPY", "action": "sell", "confidence": 0.80, "strategy_id": "s2", "capital_pct": 0.05},
            ]
        )
        assert result["success"] is True
        assert len(result["resolved_trades"]) == 0
        assert result["conflicts_count"] == 1
