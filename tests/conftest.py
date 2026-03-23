# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Root pytest configuration for QuantCore tests.

This conftest.py is automatically loaded by pytest and provides:
- Common fixtures available to all test modules
- pytest hooks for test collection and reporting
- Markers for test categorization
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import quantstack.db as _db
import quantstack.execution.portfolio_state as _ps
import quantstack.execution.risk_gate as _rg
import quantstack.mcp._state as _mcp_state
from quantstack.context import create_trading_context
from quantstack.execution.tick_executor import TickExecutor


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API access"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers",
        "regression: strategy regression tests — re-run backtests and assert metrics stability",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Modify test collection based on markers."""
    # Skip slow tests by default unless explicitly requested
    if config.getoption("-m") is None:
        skip_slow = pytest.mark.skip(reason="slow test - use -m slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# =============================================================================
# Common Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def temp_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary DuckDB database for testing."""
    db_path = tmp_path / "test.duckdb"
    yield db_path
    # Cleanup handled by tmp_path fixture


# =============================================================================
# Trading System Fixtures — fully isolated, in-memory, no file I/O
# =============================================================================


@pytest.fixture
def trading_ctx():
    """
    Fully-wired TradingContext backed by an in-memory DuckDB.

    Each test gets a completely fresh, isolated context — no shared state,
    no file system side-effects, no interaction with production data.

    Use this as the entry point for all trading-system tests:

        def test_something(trading_ctx):
            trading_ctx.portfolio.adjust_cash(-1000)
            ...
    """
    ctx = create_trading_context(
        db_path=":memory:",
        initial_cash=100_000.0,
        session_id=str(uuid.uuid4()),
    )
    yield ctx
    # DuckDB in-memory connections are GC'd automatically; explicit close for safety
    try:
        ctx.db.close()
    except Exception:
        pass


@pytest.fixture
def signal_cache(trading_ctx):
    """Pre-wired SignalCache from trading_ctx (convenience alias)."""
    return trading_ctx.signal_cache


@pytest.fixture
def risk_state(trading_ctx):
    """Pre-wired RiskState from trading_ctx (convenience alias)."""
    return trading_ctx.risk_state


@pytest.fixture
def portfolio(trading_ctx):
    """Pre-wired PortfolioState from trading_ctx (convenience alias)."""
    return trading_ctx.portfolio


@pytest.fixture
def paper_broker(trading_ctx):
    """Pre-wired PaperBroker from trading_ctx (convenience alias)."""
    return trading_ctx.broker


@pytest.fixture
def kill_switch(trading_ctx):
    """Fresh KillSwitch (not active) from trading_ctx."""
    return trading_ctx.kill_switch


@pytest.fixture
def tick_executor(trading_ctx):
    """
    TickExecutor wired to the in-memory trading context.

    fill_queue is an asyncio.Queue(maxsize=100).  Drain it in tests that
    submit orders to verify fill contents.
    """
    fill_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    executor = TickExecutor(
        signal_cache=trading_ctx.signal_cache,
        risk_state=trading_ctx.risk_state,
        broker=trading_ctx.broker,
        kill_switch=trading_ctx.kill_switch,
        fill_queue=fill_queue,
        session_id=trading_ctx.session_id,
    )
    return executor, fill_queue


@pytest.fixture(autouse=True)
def reset_singletons_and_seeds() -> Generator[None, None, None]:
    """Reset module-level singletons and random seeds before each test.

    Without this, MCP tests that create a TradingContext pollute the global
    _risk_gate and _portfolio_state singletons, causing later tests to see
    stale state (wrong initial_cash, leftover positions, etc.).
    """
    np.random.seed(42)
    try:
        import torch  # noqa: F811 — conditional import; torch may not be installed
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass

    yield

    # Reset ALL module-level singletons so tests are fully isolated.
    # Without this, earlier tests pollute global state for later tests.
    _rg._risk_gate = None
    _ps._portfolio_state = None
    _ps._portfolio_state_ro = None
    _db._managed = None

    try:
        _mcp_state._ctx = None
        _mcp_state._degraded = False
        _mcp_state._degraded_reason = ""
    except Exception:
        pass


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    n_bars = 100
    np.random.seed(42)

    # Generate random walk prices
    returns = np.random.randn(n_bars) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV
    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=pd.date_range(
            "2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"
        ),
    )

    return df


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.market_timezone = "America/New_York"
    settings.database_path = ":memory:"
    settings.alpha_vantage_api_key = "test_key"
    settings.log_level = "DEBUG"
    return settings


# =============================================================================
# Helper Functions for Legacy Tests
# Imported from tests._fixtures.ohlcv_generators for single-source-of-truth.
# Re-exported here under original names for backward compatibility.
# =============================================================================

from tests._fixtures.ohlcv_generators import (  # noqa: F401
    add_atr_column,
    make_flat_market,
    make_impulse_up_legs,
    make_impulse_wave_ohlcv,
    make_monotonic_downtrend,
    make_monotonic_uptrend,
    make_ohlcv_df,
    make_swing_leg,
    make_v_shape_ohlcv,
    make_w_shape_ohlcv,
)
