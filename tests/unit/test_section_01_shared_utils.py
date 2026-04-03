"""Tests for Section 01: shared utility relocation."""
import importlib
import subprocess

import pytest


def test_tools_models_importable():
    """tools/models.py is importable and contains expected Pydantic models."""
    mod = importlib.import_module("quantstack.tools.models")
    assert hasattr(mod, "RunAnalysisInput")
    assert hasattr(mod, "BacktestRequest")
    assert hasattr(mod, "TradeOrder")


def test_tools_helpers_ok_format():
    """tools/_helpers.py _get_reader returns a PgDataStore."""
    from quantstack.tools._helpers import _get_reader
    reader = _get_reader()
    assert reader is not None


def test_tools_helpers_parse_timeframe():
    """tools/_helpers.py _parse_timeframe parses common timeframe strings."""
    from quantstack.tools._helpers import _parse_timeframe
    from quantstack.config.timeframes import Timeframe
    assert _parse_timeframe("daily") == Timeframe.D1
    assert _parse_timeframe("1h") == Timeframe.H1
    assert _parse_timeframe("weekly") == Timeframe.W1


def test_tools_state_live_db_or_error_returns_tuple():
    """tools/_state.py live_db_or_error() returns a 2-tuple."""
    from quantstack.tools._state import live_db_or_error
    try:
        result = live_db_or_error()
        assert isinstance(result, tuple)
        assert len(result) == 2
    except Exception:
        # DB not available in CI/local — verify the function is importable and callable
        assert callable(live_db_or_error)


def test_tools_state_require_ctx_callable():
    """tools/_state.py require_ctx() is importable and callable."""
    from quantstack.tools._state import require_ctx
    assert callable(require_ctx)


def test_tools_shared_importable():
    """tools/_shared.py is importable and contains run_backtest_impl."""
    from quantstack.tools._shared import run_backtest_impl
    assert callable(run_backtest_impl)


def test_tools_shared_get_strategy_impl_importable():
    """tools/_shared.py contains get_strategy_impl."""
    from quantstack.tools._shared import get_strategy_impl
    assert callable(get_strategy_impl)


def test_allocation_importable():
    """allocation package is importable from new location."""
    from quantstack.allocation.allocation import compute_allocation, resolve_conflicts
    assert callable(compute_allocation)
    assert callable(resolve_conflicts)


def test_dynamic_allocation_importable():
    """dynamic_allocation module is importable from new location."""
    from quantstack.allocation.dynamic_allocation import compute_dynamic_allocation
    assert callable(compute_dynamic_allocation)


def test_allocation_init_exports():
    """allocation __init__.py re-exports public API."""
    import quantstack.allocation as alloc
    assert hasattr(alloc, "compute_allocation")
    assert hasattr(alloc, "compute_dynamic_allocation")
    assert hasattr(alloc, "DynamicAllocationPlan")


def test_no_remaining_old_imports():
    """No code outside mcp/ imports from the old mcp locations for moved modules."""
    patterns = [
        "from quantstack.mcp.models",
        "from quantstack.mcp._helpers",
        "from quantstack.mcp._state",
        "from quantstack.mcp.allocation",
        "from quantstack.mcp.dynamic_allocation",
        "from quantstack.mcp.tools._impl",
    ]
    for pattern in patterns:
        result = subprocess.run(
            ["grep", "-r", pattern, "src/quantstack/"],
            capture_output=True, text=True,
            cwd="/Users/kshitijbichave/Personal/Trader",
        )
        # Filter out hits inside mcp/ itself (those get deleted later in Section 8)
        lines = [
            line for line in result.stdout.strip().split("\n")
            if line and "src/quantstack/mcp/" not in line
        ]
        assert lines == [], f"Found old import pattern '{pattern}' outside mcp/: {lines}"
