# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Phase 5 MCP tools: regime matrix CRUD, multi-analysis,
conflict resolution.
"""

from __future__ import annotations

import uuid

import pytest
from quantstack.context import create_trading_context
from quantstack.db import pg_conn
from quantstack.mcp.tools.meta import get_regime_strategies, resolve_portfolio_conflicts, set_regime_allocation
import quantstack.mcp._state as _mcp_state


def _clean_meta_tables() -> None:
    """Commit-delete tables written by pg_conn-based MCP tools."""
    with pg_conn() as conn:
        conn.execute("DELETE FROM regime_strategy_matrix")
        conn.execute("DELETE FROM strategies")


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
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    # Committed deletes for tables that MCP tools write via pg_conn()
    _clean_meta_tables()
    ctx.portfolio.reset()
    ctx.db.execute("DELETE FROM fills")
    ctx.db.execute("DELETE FROM decision_events")
    ctx.db.execute("DELETE FROM closed_trades")
    yield ctx
    _mcp_state._ctx = original
    ctx.db.execute("ROLLBACK")
    _clean_meta_tables()


def _fn(tool_obj):
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


# ---------------------------------------------------------------------------
# get_regime_strategies
# ---------------------------------------------------------------------------


class TestGetRegimeStrategies:
    @pytest.mark.asyncio
    async def test_empty_initially(self, _inject_ctx):
        result = await _fn(get_regime_strategies)("trending_up")
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_allocations_after_set(self, _inject_ctx):
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
        result = await _fn(set_regime_allocation)(
            regime="ranging",
            allocations=[{"strategy_id": "s1", "allocation_pct": 0.25}],
        )
        assert result["success"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_upserts_existing(self, _inject_ctx):
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
        result = await _fn(resolve_portfolio_conflicts)(
            proposed_trades=[
                {
                    "symbol": "SPY",
                    "action": "buy",
                    "confidence": 0.8,
                    "strategy_id": "s1",
                    "capital_pct": 0.05,
                },
                {
                    "symbol": "QQQ",
                    "action": "sell",
                    "confidence": 0.7,
                    "strategy_id": "s2",
                    "capital_pct": 0.05,
                },
            ]
        )
        assert result["success"] is True
        assert len(result["resolved_trades"]) == 2
        assert result["conflicts_count"] == 0

    @pytest.mark.asyncio
    async def test_conflict_skip(self, _inject_ctx):
        result = await _fn(resolve_portfolio_conflicts)(
            proposed_trades=[
                {
                    "symbol": "SPY",
                    "action": "buy",
                    "confidence": 0.85,
                    "strategy_id": "s1",
                    "capital_pct": 0.05,
                },
                {
                    "symbol": "SPY",
                    "action": "sell",
                    "confidence": 0.80,
                    "strategy_id": "s2",
                    "capital_pct": 0.05,
                },
            ]
        )
        assert result["success"] is True
        assert len(result["resolved_trades"]) == 0
        assert result["conflicts_count"] == 1
