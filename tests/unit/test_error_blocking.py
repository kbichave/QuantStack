"""Tests for error blocking and node classification (section-05).

Verifies the execution gate halts on blocking node errors and allows
continuation (with safe defaults) when non-blocking nodes fail.
"""

import pytest

from quantstack.graphs.trading.graph import (
    NODE_CLASSIFICATION,
    _execution_gate,
)


# ---------------------------------------------------------------------------
# Node Classification
# ---------------------------------------------------------------------------

EXPECTED_BLOCKING = {"data_refresh", "safety_check", "position_review", "risk_sizing", "execute_exits"}


def test_classification_contains_all_nodes():
    """NODE_CLASSIFICATION has an entry for every trading graph node."""
    expected_nodes = {
        "data_refresh", "safety_check", "market_intel", "plan_day",
        "position_review", "execute_exits", "entry_scan", "earnings_analysis",
        "merge_parallel", "merge_pre_execution", "risk_sizing",
        "portfolio_construction", "portfolio_review", "analyze_options",
        "execute_entries", "reflect", "resolve_symbol_conflicts",
    }
    assert set(NODE_CLASSIFICATION.keys()) == expected_nodes


def test_classification_values_valid():
    """All values are 'blocking' or 'non_blocking'."""
    for node, cls in NODE_CLASSIFICATION.items():
        assert cls in ("blocking", "non_blocking"), f"{node} has invalid class {cls!r}"


def test_blocking_nodes_correct():
    """Exactly the right nodes are classified as blocking."""
    actual_blocking = {n for n, c in NODE_CLASSIFICATION.items() if c == "blocking"}
    assert actual_blocking == EXPECTED_BLOCKING


# ---------------------------------------------------------------------------
# Execution Gate: Blocking Node Errors
# ---------------------------------------------------------------------------


def _state_with_errors(errors: list[str]) -> dict:
    """Minimal state dict with errors list."""
    return {"errors": errors}


def test_blocking_data_refresh_error_halts():
    state = _state_with_errors(["[data_refresh] connection timeout"])
    assert _execution_gate(state) == "halt"


def test_blocking_position_review_error_halts():
    state = _state_with_errors(["[position_review] LLM parse failure"])
    assert _execution_gate(state) == "halt"


def test_blocking_execute_exits_error_halts():
    state = _state_with_errors(["[execute_exits] order submission failed"])
    assert _execution_gate(state) == "halt"


def test_single_blocking_plus_successes_still_halts():
    state = _state_with_errors(["[data_refresh] timeout"])
    assert _execution_gate(state) == "halt"


# ---------------------------------------------------------------------------
# Execution Gate: Non-Blocking Node Errors
# ---------------------------------------------------------------------------


def test_non_blocking_plan_day_continues():
    state = _state_with_errors(["[plan_day] LLM timeout"])
    assert _execution_gate(state) == "continue"


def test_non_blocking_entry_scan_continues():
    state = _state_with_errors(["[entry_scan] no candidates"])
    assert _execution_gate(state) == "continue"


# ---------------------------------------------------------------------------
# Execution Gate: Total Error Threshold
# ---------------------------------------------------------------------------


def test_no_errors_continues():
    state = _state_with_errors([])
    assert _execution_gate(state) == "continue"


def test_two_non_blocking_errors_continues():
    state = _state_with_errors(["[plan_day] err1", "[entry_scan] err2"])
    assert _execution_gate(state) == "continue"


def test_three_non_blocking_errors_halts():
    state = _state_with_errors([
        "[plan_day] err1",
        "[entry_scan] err2",
        "[market_intel] err3",
    ])
    assert _execution_gate(state) == "halt"


def test_more_than_three_errors_halts():
    state = _state_with_errors([
        "[plan_day] err1", "[entry_scan] err2",
        "[market_intel] err3", "[reflect] err4",
    ])
    assert _execution_gate(state) == "halt"


# ---------------------------------------------------------------------------
# Mixed Scenarios
# ---------------------------------------------------------------------------


def test_one_blocking_one_non_blocking_halts():
    state = _state_with_errors(["[data_refresh] timeout", "[plan_day] err"])
    assert _execution_gate(state) == "halt"


def test_zero_blocking_three_non_blocking_halts():
    state = _state_with_errors([
        "[plan_day] err", "[entry_scan] err", "[reflect] err",
    ])
    assert _execution_gate(state) == "halt"
