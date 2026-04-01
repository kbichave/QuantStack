# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for _impl.py — shared business logic used by multiple MCP tools.

Tests get_strategy_impl, register_strategy_impl, and run_backtest_impl
directly, without any MCP layer.  Only external boundaries (DB, price data)
are mocked.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import quantstack.mcp._state as _mcp_state
from quantstack.context import create_trading_context
from quantstack.mcp.tools._impl import (
    get_strategy_impl,
    register_strategy_impl,
    run_backtest_impl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """In-memory TradingContext with clean state."""
    context = create_trading_context(
        db_path=":memory:",
        initial_cash=100_000.0,
        session_id=str(uuid.uuid4()),
    )
    yield context
    context.db.close()


@pytest.fixture
def inject_ctx(ctx):
    """Inject test context into MCP state so live_db_or_error succeeds."""
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    ctx.db.execute("DELETE FROM strategies")
    yield ctx
    _mcp_state._ctx = original


@contextmanager
def _mock_pg_context(ctx):
    yield ctx.db


@pytest.fixture
def mock_pg(ctx, inject_ctx):
    """Patch pg_conn() to use the in-memory test DB."""
    with patch("quantstack.db.pg_conn", return_value=_mock_pg_context(ctx)):
        yield ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n_days: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV for backtest tests."""
    rng = np.random.default_rng(42)
    close = start_price + np.cumsum(rng.normal(0, 0.5, n_days))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n_days),
            "high": close + np.abs(rng.normal(0, 0.3, n_days)),
            "low": close - np.abs(rng.normal(0, 0.3, n_days)),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n_days),
        },
        index=pd.date_range("2023-01-01", periods=n_days, freq="1D"),
    )


# ===========================================================================
# get_strategy_impl
# ===========================================================================


class TestGetStrategyImpl:
    async def test_no_context_returns_error(self):
        """Without a TradingContext, live_db_or_error fails gracefully."""
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await get_strategy_impl(strategy_id="strat_abc123")
            assert result["success"] is False
            assert "error" in result
        finally:
            _mcp_state._ctx = original

    async def test_no_args_returns_error(self, mock_pg):
        result = await get_strategy_impl()
        assert result["success"] is False
        assert "Provide strategy_id or name" in result["error"]

    async def test_not_found_by_id(self, mock_pg):
        result = await get_strategy_impl(strategy_id="strat_nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_not_found_by_name(self, mock_pg):
        result = await get_strategy_impl(name="no_such_strategy")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_found_by_id(self, mock_pg):
        # Register first
        reg = await register_strategy_impl(
            name="test_get_by_id",
            parameters={"rsi": 14},
            entry_rules=[{"indicator": "rsi", "condition": "lt", "value": 30}],
            exit_rules=[{"indicator": "rsi", "condition": "gt", "value": 70}],
        )
        assert reg["success"] is True
        strategy_id = reg["strategy_id"]

        # Now fetch
        result = await get_strategy_impl(strategy_id=strategy_id)
        assert result["success"] is True
        strat = result["strategy"]
        assert strat["name"] == "test_get_by_id"
        assert strat["status"] == "draft"
        assert strat["parameters"] == {"rsi": 14}

    async def test_found_by_name(self, mock_pg):
        await register_strategy_impl(
            name="findme_by_name",
            parameters={},
            entry_rules=[{"rule": "x"}],
            exit_rules=[{"rule": "y"}],
        )
        result = await get_strategy_impl(name="findme_by_name")
        assert result["success"] is True
        assert result["strategy"]["name"] == "findme_by_name"

    async def test_json_fields_deserialized(self, mock_pg):
        """JSON columns (parameters, entry_rules, etc.) should be dicts/lists, not raw strings."""
        reg = await register_strategy_impl(
            name="json_deser_test",
            parameters={"ema_fast": 8, "ema_slow": 21},
            entry_rules=[{"indicator": "ema_cross", "condition": "bullish"}],
            exit_rules=[{"indicator": "trailing_stop", "atr_mult": 2.0}],
            regime_affinity={"trending_up": 0.9, "ranging": 0.3},
            risk_params={"stop_loss_atr": 1.5},
        )
        result = await get_strategy_impl(strategy_id=reg["strategy_id"])
        strat = result["strategy"]

        assert isinstance(strat["parameters"], dict)
        assert strat["parameters"]["ema_fast"] == 8
        assert isinstance(strat["entry_rules"], list)
        assert isinstance(strat["exit_rules"], list)
        assert isinstance(strat["regime_affinity"], dict)
        assert strat["regime_affinity"]["trending_up"] == 0.9
        assert isinstance(strat["risk_params"], dict)


# ===========================================================================
# register_strategy_impl
# ===========================================================================


class TestRegisterStrategyImpl:
    async def test_register_happy_path(self, mock_pg):
        result = await register_strategy_impl(
            name="momentum_rsi",
            parameters={"rsi_period": 14},
            entry_rules=[{"indicator": "rsi", "condition": "lt", "value": 30}],
            exit_rules=[{"indicator": "rsi", "condition": "gt", "value": 70}],
            description="RSI mean reversion",
            asset_class="equities",
            source="workshop",
        )
        assert result["success"] is True
        assert result["status"] == "draft"
        assert result["strategy_id"].startswith("strat_")

    async def test_register_returns_unique_ids(self, mock_pg):
        ids = set()
        for i in range(5):
            r = await register_strategy_impl(
                name=f"unique_{i}",
                parameters={},
                entry_rules=[{"r": i}],
                exit_rules=[{"r": i}],
            )
            assert r["success"] is True
            ids.add(r["strategy_id"])
        assert len(ids) == 5

    async def test_duplicate_name_rejected(self, mock_pg):
        await register_strategy_impl(
            name="dup_test",
            parameters={},
            entry_rules=[{"r": 1}],
            exit_rules=[{"r": 1}],
        )
        result = await register_strategy_impl(
            name="dup_test",
            parameters={},
            entry_rules=[{"r": 2}],
            exit_rules=[{"r": 2}],
        )
        assert result["success"] is False
        assert "already exists" in result["error"]

    async def test_register_persists_all_fields(self, mock_pg):
        reg = await register_strategy_impl(
            name="full_fields",
            parameters={"a": 1},
            entry_rules=[{"b": 2}],
            exit_rules=[{"c": 3}],
            description="desc",
            asset_class="options",
            regime_affinity={"ranging": 0.8},
            risk_params={"stop": 2.0},
            source="decoded",
            instrument_type="options",
            time_horizon="intraday",
            holding_period_days=1,
        )
        fetched = await get_strategy_impl(strategy_id=reg["strategy_id"])
        strat = fetched["strategy"]

        assert strat["description"] == "desc"
        assert strat["asset_class"] == "options"
        assert strat["source"] == "decoded"
        assert strat["instrument_type"] == "options"
        assert strat["time_horizon"] == "intraday"
        assert strat["holding_period_days"] == 1
        assert strat["regime_affinity"]["ranging"] == 0.8
        assert strat["risk_params"]["stop"] == 2.0

    async def test_no_context_returns_error(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await register_strategy_impl(
                name="noctx",
                parameters={},
                entry_rules=[{}],
                exit_rules=[{}],
            )
            assert result["success"] is False
        finally:
            _mcp_state._ctx = original


# ===========================================================================
# run_backtest_impl
# ===========================================================================


class TestRunBacktestImpl:
    async def test_no_context_returns_error(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await run_backtest_impl(
                strategy_id="strat_abc", symbol="SPY"
            )
            assert result["success"] is False
        finally:
            _mcp_state._ctx = original

    async def test_strategy_not_found(self, mock_pg):
        result = await run_backtest_impl(
            strategy_id="strat_nonexistent", symbol="SPY"
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_no_entry_rules(self, mock_pg):
        """Strategy with empty entry_rules should fail gracefully."""
        reg = await register_strategy_impl(
            name="empty_rules",
            parameters={},
            entry_rules=[],
            exit_rules=[],
        )
        with patch(
            "quantstack.mcp.tools._impl._fetch_price_data",
            return_value=_make_ohlcv(),
        ):
            result = await run_backtest_impl(
                strategy_id=reg["strategy_id"], symbol="SPY"
            )
        assert result["success"] is False
        assert "no entry_rules" in result["error"].lower()

    async def test_no_price_data(self, mock_pg):
        reg = await register_strategy_impl(
            name="no_data",
            parameters={},
            entry_rules=[{"indicator": "rsi", "condition": "lt", "value": 30}],
            exit_rules=[{"indicator": "rsi", "condition": "gt", "value": 70}],
        )
        with patch(
            "quantstack.mcp.tools._impl._fetch_price_data",
            return_value=None,
        ):
            result = await run_backtest_impl(
                strategy_id=reg["strategy_id"], symbol="FAKE"
            )
        assert result["success"] is False
        assert "no price data" in result["error"].lower()

    async def test_successful_backtest_returns_metrics(self, mock_pg):
        """Happy path: register strategy, mock price data + engine result, verify metrics."""
        reg = await register_strategy_impl(
            name="bt_happy",
            parameters={"rsi_period": 14},
            entry_rules=[{"indicator": "rsi_14", "condition": "crosses_below", "value": 30}],
            exit_rules=[{"indicator": "rsi_14", "condition": "crosses_above", "value": 70}],
        )
        ohlcv = _make_ohlcv(200)

        # Mock the entire backtest pipeline to avoid engine internals
        mock_engine_result = MagicMock()
        mock_engine_result.total_trades = 4
        mock_engine_result.win_rate = 60.0
        mock_engine_result.sharpe_ratio = 1.25
        mock_engine_result.max_drawdown = 8.5
        mock_engine_result.total_return = 12.3
        mock_engine_result.profit_factor = 1.8
        mock_engine_result.trades = [
            {"pnl": 500.0, "entry": "2023-01-20", "exit": "2023-02-20"},
            {"pnl": -200.0, "entry": "2023-03-20", "exit": "2023-04-30"},
            {"pnl": 800.0, "entry": "2023-05-15", "exit": "2023-06-20"},
            {"pnl": 100.0, "entry": "2023-07-01", "exit": "2023-08-01"},
        ]

        mock_signals = pd.Series(0, index=ohlcv.index)

        with patch(
            "quantstack.mcp.tools._impl._fetch_price_data",
            return_value=ohlcv,
        ), patch(
            "quantstack.mcp.tools._impl._generate_signals_from_rules",
            return_value=mock_signals,
        ), patch(
            "quantstack.mcp.tools._impl.BacktestEngine"
        ) as MockEngine:
            MockEngine.return_value.run.return_value = mock_engine_result
            result = await run_backtest_impl(
                strategy_id=reg["strategy_id"],
                symbol="SPY",
                initial_capital=100_000.0,
            )

        assert result["success"] is True
        assert result["strategy_id"] == reg["strategy_id"]
        assert result["symbol"] == "SPY"

        # Verify metric keys exist and are numeric
        for key in [
            "total_trades", "win_rate", "sharpe_ratio", "max_drawdown",
            "total_return_pct", "profit_factor", "calmar_ratio", "avg_trade_pnl",
            "bars_tested",
        ]:
            assert key in result, f"Missing metric: {key}"
            assert isinstance(result[key], (int, float)), f"{key} should be numeric"

        assert result["bars_tested"] == 200
        assert result["total_trades"] == 4
        assert result["sharpe_ratio"] == 1.25
        assert "trades" not in result  # trades list is omitted from response

    async def test_backtest_persists_summary(self, mock_pg):
        """After backtest, strategy record should have backtest_summary and status updated."""
        reg = await register_strategy_impl(
            name="bt_persist",
            parameters={},
            entry_rules=[{"indicator": "rsi_14", "condition": "lt", "value": 30}],
            exit_rules=[{"indicator": "rsi_14", "condition": "gt", "value": 70}],
        )
        ohlcv = _make_ohlcv(100)
        mock_signals = pd.Series(0, index=ohlcv.index)

        mock_engine_result = MagicMock()
        mock_engine_result.total_trades = 1
        mock_engine_result.win_rate = 100.0
        mock_engine_result.sharpe_ratio = 2.0
        mock_engine_result.max_drawdown = 3.0
        mock_engine_result.total_return = 5.0
        mock_engine_result.profit_factor = 3.0
        mock_engine_result.trades = [{"pnl": 300.0}]

        with patch(
            "quantstack.mcp.tools._impl._fetch_price_data",
            return_value=ohlcv,
        ), patch(
            "quantstack.mcp.tools._impl._generate_signals_from_rules",
            return_value=mock_signals,
        ), patch(
            "quantstack.mcp.tools._impl.BacktestEngine"
        ) as MockEngine:
            MockEngine.return_value.run.return_value = mock_engine_result
            await run_backtest_impl(
                strategy_id=reg["strategy_id"], symbol="SPY"
            )

        # Fetch the strategy and verify backtest_summary was persisted
        fetched = await get_strategy_impl(strategy_id=reg["strategy_id"])
        strat = fetched["strategy"]
        assert strat["status"] == "backtested"
        assert strat["backtest_summary"] is not None
        assert isinstance(strat["backtest_summary"], dict)
        assert strat["backtest_summary"]["symbol"] == "SPY"
        assert strat["backtest_summary"]["total_trades"] == 1
