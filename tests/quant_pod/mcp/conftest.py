# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for MCP tool tests.

Provides:
  - ctx / inject_ctx — in-memory TradingContext with clean state
  - mock_pg_conn — patches pg_conn() to use the test DB
  - synthetic_ohlcv — factory for realistic OHLCV DataFrames
  - _fn — extracts async callable from FunctionTool wrapper
  - assert_standard_response / assert_error_response — response shape validators
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantstack.context import create_trading_context
from quantstack.db import reset_pg_pool
from quantstack.execution.portfolio_state import Position
import quantstack.mcp._state as _mcp_state


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """In-memory TradingContext for isolated testing."""
    reset_pg_pool()  # reclaim connections leaked by prior tests
    context = create_trading_context(
        db_path=":memory:",
        initial_cash=100_000.0,
        session_id=str(uuid.uuid4()),
    )
    yield context
    context.db.close()
    reset_pg_pool()


@pytest.fixture
def inject_ctx(ctx):
    """Inject the test context into MCP state with clean DB tables."""
    original = _mcp_state._ctx
    _mcp_state._ctx = ctx
    ctx.portfolio.reset()
    ctx.db.execute("DELETE FROM fills")
    ctx.db.execute("DELETE FROM decision_events")
    ctx.db.execute("DELETE FROM strategies")
    ctx.db.execute("DELETE FROM regime_strategy_matrix")
    ctx.db.execute("DELETE FROM closed_trades")
    yield ctx
    _mcp_state._ctx = original
    try:
        ctx.db.execute("ROLLBACK")
    except Exception:
        pass  # Connection may have been closed during long-running tests


@pytest.fixture
def ctx_with_position(ctx):
    """Inject a small position (5 shares @ $100) for execution tests."""
    ctx.portfolio.upsert_position(
        Position(
            symbol="SPY", quantity=5, avg_cost=100.0,
            current_price=100.0, side="long",
        )
    )
    return ctx


# ---------------------------------------------------------------------------
# pg_conn mock — routes pg_conn() calls through test DB
# ---------------------------------------------------------------------------


@contextmanager
def _mock_pg_context(ctx):
    """Context manager that makes pg_conn() use the test TradingContext's connection."""
    yield ctx.db


@pytest.fixture
def mock_pg_conn(ctx):
    """Patch pg_conn() to use the test TradingContext's DB connection.

    This is critical for tools that use pg_conn() directly (alerts, strategy,
    learning, coordination) instead of require_ctx().
    """
    with patch("quantstack.db.pg_conn", return_value=_mock_pg_context(ctx)):
        yield ctx


# ---------------------------------------------------------------------------
# FunctionTool helper (DEPRECATED — tools are now plain callables)
# ---------------------------------------------------------------------------


def _fn(tool_obj):
    """Identity function. Tools are plain callables after the tool_def migration."""
    return tool_obj


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def synthetic_ohlcv(
    symbol: str = "SPY",
    n_days: int = 252,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic OHLCV DataFrame with trend + noise.

    Args:
        symbol: Ticker symbol (used in naming only).
        n_days: Number of bars.
        start_price: Starting close price.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with open/high/low/close/volume columns and DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    trend = start_price + t * 0.03
    noise = rng.normal(0, 0.5, n_days)
    oscillation = 2 * np.sin(t * 2 * np.pi / 40)
    close = trend + noise + oscillation

    high = close + np.abs(rng.normal(0, 0.3, n_days))
    low = close - np.abs(rng.normal(0, 0.3, n_days))
    open_ = close + rng.normal(0, 0.15, n_days)
    volume = rng.integers(100_000, 1_000_000, n_days)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2023-01-01", periods=n_days, freq="1D"),
    )


# ---------------------------------------------------------------------------
# Response shape validators
# ---------------------------------------------------------------------------


def assert_standard_response(result: dict[str, Any]) -> None:
    """Assert the tool returned a well-formed response with 'success' key."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "success" in result, f"Missing 'success' key in response: {list(result.keys())}"


def assert_error_response(result: dict[str, Any]) -> None:
    """Assert the tool returned a well-formed error response."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("success") is False, f"Expected success=False, got {result.get('success')}"
    assert "error" in result, f"Missing 'error' key in error response: {list(result.keys())}"
