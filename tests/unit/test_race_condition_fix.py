"""Tests for parallel branch conflict resolution (section-06).

Verifies that resolve_symbol_conflicts removes entry candidates that overlap
with pending exit orders (exits always win, risk-off bias).
"""

import pytest

from quantstack.graphs.trading.nodes import resolve_symbol_conflicts


@pytest.fixture
def state_with_conflict():
    return {
        "exit_orders": [{"symbol": "AAPL", "action": "CLOSE", "reason": "regime_flip_severe"}],
        "entry_candidates": [
            {"symbol": "AAPL", "verdict": "ENTER", "reason": "momentum_breakout"},
            {"symbol": "MSFT", "verdict": "ENTER", "reason": "mean_reversion"},
        ],
    }


@pytest.mark.asyncio
async def test_overlapping_symbol_removed_from_entries(state_with_conflict):
    result = await resolve_symbol_conflicts(state_with_conflict)
    symbols = [c["symbol"] for c in result["entry_candidates"]]
    assert "AAPL" not in symbols
    assert "MSFT" in symbols
    assert "exit_orders" not in result  # exits untouched


@pytest.mark.asyncio
async def test_multiple_overlapping_symbols():
    state = {
        "exit_orders": [
            {"symbol": "AAPL", "action": "CLOSE", "reason": "stop_loss"},
            {"symbol": "TSLA", "action": "TRIM", "reason": "drawdown"},
        ],
        "entry_candidates": [
            {"symbol": "AAPL", "verdict": "ENTER"},
            {"symbol": "TSLA", "verdict": "ENTER"},
            {"symbol": "MSFT", "verdict": "ENTER"},
        ],
    }
    result = await resolve_symbol_conflicts(state)
    symbols = [c["symbol"] for c in result["entry_candidates"]]
    assert symbols == ["MSFT"]


@pytest.mark.asyncio
async def test_no_overlapping_symbols():
    state = {
        "exit_orders": [{"symbol": "AAPL", "action": "CLOSE"}],
        "entry_candidates": [
            {"symbol": "MSFT", "verdict": "ENTER"},
            {"symbol": "GOOG", "verdict": "ENTER"},
        ],
    }
    result = await resolve_symbol_conflicts(state)
    assert len(result["entry_candidates"]) == 2


@pytest.mark.asyncio
async def test_conflict_event_logged(state_with_conflict):
    result = await resolve_symbol_conflicts(state_with_conflict)
    decisions = result["decisions"]
    assert len(decisions) == 1
    d = decisions[0]
    assert d["node"] == "resolve_symbol_conflicts"
    assert len(d["conflicts"]) == 1
    conflict = d["conflicts"][0]
    assert conflict["symbol"] == "AAPL"
    assert conflict["resolution"] == "exit_priority"
    assert d["entries_before"] == 2
    assert d["entries_after"] == 1


@pytest.mark.asyncio
async def test_empty_exit_orders_is_noop():
    state = {
        "exit_orders": [],
        "entry_candidates": [{"symbol": "AAPL", "verdict": "ENTER"}],
    }
    result = await resolve_symbol_conflicts(state)
    assert len(result["entry_candidates"]) == 1


@pytest.mark.asyncio
async def test_empty_entry_candidates_is_noop():
    state = {
        "exit_orders": [{"symbol": "AAPL", "action": "CLOSE"}],
        "entry_candidates": [],
    }
    result = await resolve_symbol_conflicts(state)
    assert result["entry_candidates"] == []
