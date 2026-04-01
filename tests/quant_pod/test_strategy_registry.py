# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the strategy registry CRUD operations (MCP tools).

All tests use an in-memory TradingContext — no file I/O, no LLM calls.
"""

from __future__ import annotations

import uuid

import pytest
from quantstack.context import create_trading_context
from quantstack.db import pg_conn
from quantstack.mcp.tools.strategy import get_strategy, list_strategies, register_strategy, update_strategy
import quantstack.mcp._state as _mcp_state


def _clean_strategies() -> None:
    """Commit a hard DELETE — needed because register_strategy uses pg_conn() (committed writes)."""
    with pg_conn() as conn:
        conn.execute("DELETE FROM strategies")
        conn.execute("DELETE FROM regime_strategy_matrix")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    _clean_strategies()
    context = create_trading_context(db_path=":memory:", session_id=str(uuid.uuid4()))
    yield context
    context.db.execute("ROLLBACK")
    context.db.close()
    _clean_strategies()


@pytest.fixture
def _inject_ctx(ctx):
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    yield ctx
    _mcp_state._ctx = original


def _fn(tool_obj):
    """Extract underlying async function from FunctionTool."""
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


# ---------------------------------------------------------------------------
# register_strategy
# ---------------------------------------------------------------------------


class TestRegisterStrategy:
    @pytest.mark.asyncio
    async def test_register_returns_success(self, _inject_ctx):
        result = await _fn(register_strategy)(
            name="test_momentum",
            parameters={"rsi_period": 14},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "crosses_below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[
                {"indicator": "rsi", "condition": "crosses_above", "value": 70}
            ],
        )
        assert result["success"] is True
        assert result["strategy_id"].startswith("strat_")
        assert result["status"] == "draft"

    @pytest.mark.asyncio
    async def test_register_with_all_fields(self, _inject_ctx):
        result = await _fn(register_strategy)(
            name="full_strategy",
            description="A comprehensive test strategy",
            asset_class="equities",
            parameters={"rsi_period": 14, "sma_fast": 10, "sma_slow": 50},
            entry_rules=[
                {
                    "indicator": "sma_crossover",
                    "condition": "crosses_above",
                    "direction": "long",
                }
            ],
            exit_rules=[
                {"indicator": "rsi", "condition": "crosses_above", "value": 80}
            ],
            regime_affinity={"trending_up": 0.9, "ranging": 0.2},
            risk_params={"stop_loss_atr": 2.0, "position_pct": 0.05},
            source="workshop",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_duplicate_name_fails(self, _inject_ctx):
        await _fn(register_strategy)(
            name="unique_name",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        result = await _fn(register_strategy)(
            name="unique_name",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 20,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# list_strategies
# ---------------------------------------------------------------------------


class TestListStrategies:
    @pytest.mark.asyncio
    async def test_empty_initially(self, _inject_ctx):
        result = await _fn(list_strategies)()
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_registered_strategies(self, _inject_ctx):
        await _fn(register_strategy)(
            name="s1",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        await _fn(register_strategy)(
            name="s2",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 20,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )

        result = await _fn(list_strategies)()
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_status(self, _inject_ctx):
        r = await _fn(register_strategy)(
            name="s_draft",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        await _fn(update_strategy)(strategy_id=r["strategy_id"], status="backtested")

        await _fn(register_strategy)(
            name="s_still_draft",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 20,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )

        drafts = await _fn(list_strategies)(status="draft")
        assert drafts["total"] == 1
        assert drafts["strategies"][0]["name"] == "s_still_draft"

        backtested = await _fn(list_strategies)(status="backtested")
        assert backtested["total"] == 1


# ---------------------------------------------------------------------------
# get_strategy
# ---------------------------------------------------------------------------


class TestGetStrategy:
    @pytest.mark.asyncio
    async def test_get_by_id(self, _inject_ctx):
        r = await _fn(register_strategy)(
            name="get_test",
            parameters={"rsi_period": 14},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
            description="test desc",
        )
        result = await _fn(get_strategy)(strategy_id=r["strategy_id"])
        assert result["success"] is True
        assert result["strategy"]["name"] == "get_test"
        assert result["strategy"]["parameters"]["rsi_period"] == 14
        assert result["strategy"]["description"] == "test desc"

    @pytest.mark.asyncio
    async def test_get_by_name(self, _inject_ctx):
        await _fn(register_strategy)(
            name="by_name_test",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        result = await _fn(get_strategy)(name="by_name_test")
        assert result["success"] is True
        assert result["strategy"]["name"] == "by_name_test"

    @pytest.mark.asyncio
    async def test_not_found(self, _inject_ctx):
        result = await _fn(get_strategy)(strategy_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# update_strategy
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    @pytest.mark.asyncio
    async def test_update_status(self, _inject_ctx):
        r = await _fn(register_strategy)(
            name="update_test",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        await _fn(update_strategy)(strategy_id=r["strategy_id"], status="backtested")

        check = await _fn(get_strategy)(strategy_id=r["strategy_id"])
        assert check["strategy"]["status"] == "backtested"

    @pytest.mark.asyncio
    async def test_update_backtest_summary(self, _inject_ctx):
        r = await _fn(register_strategy)(
            name="bt_summary_test",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        summary = {"sharpe_ratio": 1.5, "max_drawdown": 8.2, "total_trades": 75}
        await _fn(update_strategy)(
            strategy_id=r["strategy_id"],
            backtest_summary=summary,
        )

        check = await _fn(get_strategy)(strategy_id=r["strategy_id"])
        assert check["strategy"]["backtest_summary"]["sharpe_ratio"] == 1.5

    @pytest.mark.asyncio
    async def test_update_no_fields_fails(self, _inject_ctx):
        r = await _fn(register_strategy)(
            name="no_fields",
            parameters={},
            entry_rules=[
                {
                    "indicator": "rsi",
                    "condition": "below",
                    "value": 30,
                    "direction": "long",
                }
            ],
            exit_rules=[],
        )
        result = await _fn(update_strategy)(strategy_id=r["strategy_id"])
        assert result["success"] is False
