# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Phase 2 strategy registry MCP tools — full CRUD lifecycle.

Tools: register_strategy, list_strategies, get_strategy, update_strategy.

All tools use live_db_or_error() + pg_conn() directly, so we need inject_ctx
(for the live_db_or_error guard) and a real pg_conn (for actual writes).
"""

from __future__ import annotations

import pytest

from quantstack.mcp.tools.strategy import (
    register_strategy,
    list_strategies,
    get_strategy,
    update_strategy,
)
import quantstack.mcp._state as _mcp_state
from tests.quantstack.mcp.conftest import _fn, assert_standard_response, assert_error_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_strategies():
    """Clean strategy tables before and after each test."""
    from quantstack.db import pg_conn

    def _clean():
        with pg_conn() as conn:
            conn.execute("DELETE FROM regime_strategy_matrix")
            conn.execute("DELETE FROM strategies")

    _clean()
    yield
    _clean()


def _register_kwargs(name: str = "test_rsi_bounce", **overrides) -> dict:
    """Build a valid register_strategy kwargs dict with sensible defaults."""
    base = {
        "name": name,
        "parameters": {"rsi_period": 14, "threshold": 30},
        "entry_rules": [{"indicator": "rsi_14", "condition": "crosses_below", "value": 30}],
        "exit_rules": [{"indicator": "rsi_14", "condition": "crosses_above", "value": 70}],
        "description": "Buy when RSI crosses below 30, sell above 70",
        "asset_class": "equities",
        "source": "manual",
        "instrument_type": "equity",
        "time_horizon": "swing",
        "holding_period_days": 5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# register_strategy
# ---------------------------------------------------------------------------


class TestRegisterStrategy:
    @pytest.mark.asyncio
    async def test_register_returns_strategy_id(self, inject_ctx, _clean_strategies):
        result = await _fn(register_strategy)(**_register_kwargs())
        assert_standard_response(result)
        assert result["success"] is True
        assert result["strategy_id"].startswith("strat_")
        assert result["status"] == "draft"

    @pytest.mark.asyncio
    async def test_register_duplicate_name_fails(self, inject_ctx, _clean_strategies):
        first = await _fn(register_strategy)(**_register_kwargs("duplicate_name"))
        assert first["success"] is True

        second = await _fn(register_strategy)(**_register_kwargs("duplicate_name"))
        assert second["success"] is False
        assert "already exists" in second["error"]
        assert first["strategy_id"] in second["error"]


# ---------------------------------------------------------------------------
# get_strategy
# ---------------------------------------------------------------------------


class TestGetStrategy:
    @pytest.mark.asyncio
    async def test_get_by_id(self, inject_ctx, _clean_strategies):
        reg = await _fn(register_strategy)(**_register_kwargs("get_by_id_strat"))
        assert reg["success"] is True
        sid = reg["strategy_id"]

        result = await _fn(get_strategy)(strategy_id=sid)
        assert_standard_response(result)
        assert result["success"] is True
        assert result["strategy"]["strategy_id"] == sid
        assert result["strategy"]["name"] == "get_by_id_strat"
        assert result["strategy"]["status"] == "draft"

    @pytest.mark.asyncio
    async def test_get_by_name(self, inject_ctx, _clean_strategies):
        reg = await _fn(register_strategy)(**_register_kwargs("get_by_name_strat"))
        assert reg["success"] is True

        result = await _fn(get_strategy)(name="get_by_name_strat")
        assert_standard_response(result)
        assert result["success"] is True
        assert result["strategy"]["name"] == "get_by_name_strat"
        assert result["strategy"]["strategy_id"] == reg["strategy_id"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_by_id(self, inject_ctx, _clean_strategies):
        result = await _fn(get_strategy)(strategy_id="strat_doesnotexist")
        assert_error_response(result)
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_nonexistent_by_name(self, inject_ctx, _clean_strategies):
        result = await _fn(get_strategy)(name="no_such_strategy")
        assert_error_response(result)
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_no_id_no_name_returns_error(self, inject_ctx, _clean_strategies):
        """Calling get_strategy with neither ID nor name should fail gracefully."""
        result = await _fn(get_strategy)()
        assert_error_response(result)
        assert "provide" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_deserializes_json_fields(self, inject_ctx, _clean_strategies):
        """JSON columns (parameters, entry_rules, etc.) should be deserialized to dicts/lists."""
        reg = await _fn(register_strategy)(**_register_kwargs("json_deser_strat"))
        result = await _fn(get_strategy)(strategy_id=reg["strategy_id"])
        strat = result["strategy"]

        # These fields are stored as JSON strings but should come back as Python objects
        assert isinstance(strat["parameters"], dict), "parameters should be deserialized"
        assert isinstance(strat["entry_rules"], list), "entry_rules should be deserialized"
        assert isinstance(strat["exit_rules"], list), "exit_rules should be deserialized"
        assert isinstance(strat["risk_params"], dict), "risk_params should be deserialized"


# ---------------------------------------------------------------------------
# list_strategies
# ---------------------------------------------------------------------------


class TestListStrategies:
    @pytest.mark.asyncio
    async def test_list_empty(self, inject_ctx, _clean_strategies):
        result = await _fn(list_strategies)()
        assert_standard_response(result)
        assert result["success"] is True
        assert result["total"] == 0
        assert result["strategies"] == []

    @pytest.mark.asyncio
    async def test_list_returns_registered(self, inject_ctx, _clean_strategies):
        await _fn(register_strategy)(**_register_kwargs("list_strat_a"))
        await _fn(register_strategy)(**_register_kwargs("list_strat_b"))

        result = await _fn(list_strategies)()
        assert result["success"] is True
        assert result["total"] == 2
        names = {s["name"] for s in result["strategies"]}
        assert names == {"list_strat_a", "list_strat_b"}

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, inject_ctx, _clean_strategies):
        """All newly registered strategies are 'draft'; filtering by 'live' should return 0."""
        await _fn(register_strategy)(**_register_kwargs("filter_status_strat"))

        draft_result = await _fn(list_strategies)(status="draft")
        assert draft_result["success"] is True
        assert draft_result["total"] >= 1

        live_result = await _fn(list_strategies)(status="live")
        assert live_result["success"] is True
        assert live_result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_filter_by_asset_class(self, inject_ctx, _clean_strategies):
        await _fn(register_strategy)(
            **_register_kwargs("options_strat", asset_class="options")
        )
        await _fn(register_strategy)(
            **_register_kwargs("equities_strat", asset_class="equities")
        )

        result = await _fn(list_strategies)(asset_class="options")
        assert result["success"] is True
        assert result["total"] == 1
        assert result["strategies"][0]["name"] == "options_strat"


# ---------------------------------------------------------------------------
# update_strategy
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    @pytest.mark.asyncio
    async def test_update_status_and_description(self, inject_ctx, _clean_strategies):
        reg = await _fn(register_strategy)(**_register_kwargs("update_strat"))
        sid = reg["strategy_id"]

        result = await _fn(update_strategy)(
            strategy_id=sid,
            status="backtested",
            description="Updated after backtest run",
        )
        assert_standard_response(result)
        assert result["success"] is True
        assert "status" in result["updated_fields"]
        assert "description" in result["updated_fields"]

        # Verify the update persisted
        fetched = await _fn(get_strategy)(strategy_id=sid)
        assert fetched["strategy"]["status"] == "backtested"
        assert fetched["strategy"]["description"] == "Updated after backtest run"

    @pytest.mark.asyncio
    async def test_update_no_fields_returns_error(self, inject_ctx, _clean_strategies):
        reg = await _fn(register_strategy)(**_register_kwargs("no_update_strat"))

        result = await _fn(update_strategy)(strategy_id=reg["strategy_id"])
        assert_error_response(result)
        assert "no fields" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_json_fields(self, inject_ctx, _clean_strategies):
        """Updating JSON fields (parameters, risk_params) should serialize correctly."""
        reg = await _fn(register_strategy)(**_register_kwargs("json_update_strat"))
        sid = reg["strategy_id"]

        new_params = {"rsi_period": 21, "threshold": 25, "atr_multiplier": 1.5}
        new_risk = {"stop_loss_atr": 3.0, "max_position_pct": 0.05}

        result = await _fn(update_strategy)(
            strategy_id=sid,
            parameters=new_params,
            risk_params=new_risk,
        )
        assert result["success"] is True

        fetched = await _fn(get_strategy)(strategy_id=sid)
        assert fetched["strategy"]["parameters"] == new_params
        assert fetched["strategy"]["risk_params"] == new_risk

    @pytest.mark.asyncio
    async def test_update_nonexistent_strategy_succeeds_silently(
        self, inject_ctx, _clean_strategies
    ):
        """BUG DETECTOR: update_strategy does not check if the strategy exists before
        running the UPDATE. If the strategy_id doesn't match any row, the UPDATE
        succeeds with 0 rows affected but still returns success=True.
        This test documents that behavior."""
        result = await _fn(update_strategy)(
            strategy_id="strat_doesnotexist",
            status="live",
        )
        # The tool returns success even though no row was updated — this is a bug.
        # It should verify rowcount > 0 and return an error if the strategy doesn't exist.
        assert_standard_response(result)
        assert result["success"] is True  # documents the bug: should be False


# ---------------------------------------------------------------------------
# Missing context
# ---------------------------------------------------------------------------


class TestMissingContext:
    @pytest.mark.asyncio
    async def test_register_without_context(self):
        """When MCP state is not initialized, tools should return an error dict."""
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(register_strategy)(**_register_kwargs("orphan"))
            assert_error_response(result)
            assert "not initialized" in result["error"].lower()
        finally:
            _mcp_state._ctx = original

    @pytest.mark.asyncio
    async def test_list_without_context(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(list_strategies)()
            assert_error_response(result)
        finally:
            _mcp_state._ctx = original

    @pytest.mark.asyncio
    async def test_get_without_context(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(get_strategy)(strategy_id="strat_anything")
            assert_error_response(result)
        finally:
            _mcp_state._ctx = original

    @pytest.mark.asyncio
    async def test_update_without_context(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(update_strategy)(
                strategy_id="strat_anything", status="live"
            )
            assert_error_response(result)
        finally:
            _mcp_state._ctx = original
