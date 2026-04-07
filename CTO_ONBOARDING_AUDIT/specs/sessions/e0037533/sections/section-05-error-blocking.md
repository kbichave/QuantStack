# Section 05: Error Blocking & Node Classification

## Background

The trading graph has 16 nodes across parallel branches. Every node catches exceptions and appends to `errors: Annotated[list[str], operator.add]` in `TradingState`. After accumulating 5+ errors, the pipeline still reaches `execute_entries` and can place orders. The only halt mechanism today is `_safety_check_router`, which only inspects the `safety_check` node's output. There is no general-purpose gate that prevents execution when upstream nodes have failed.

The core principle: **anything that feeds risk calculations or manages existing exposure is blocking; anything that generates new opportunities can fail safely.**

## Dependencies

- **section-04-node-output-models**: Each node must have a typed Pydantic output model with a `safe_default()` class method. The execution gate and non-blocking fallback logic depend on these models existing.

## Node Classification

Add a `NODE_CLASSIFICATION` dict to the trading graph module (`src/quantstack/graphs/trading/graph.py`). This dict maps every node name to either `"blocking"` or `"non_blocking"`.

**Blocking nodes** (failure means potential unintended money loss):
- `data_refresh`
- `safety_check`
- `position_review`
- `risk_sizing`
- `execute_exits`

**Non-blocking nodes** (failure means missed opportunity only):
- `plan_day`
- `entry_scan`
- `market_intel`
- `portfolio_construction`
- `execute_entries`
- `resolve_symbol_conflicts` (from section-06)
- `reflect`
- `trade_reflector`
- `earnings_analysis`
- `analyze_options`
- `portfolio_review`

Design note: `execute_exits` is blocking. If closing existing exposure fails, the pipeline must not proceed to open new exposure. A transient `execute_exits` failure blocks entries for that cycle --- this is the correct conservative behavior.

Design note: `execute_entries` is non-blocking. A failure here means we miss an entry, not that we have unmanaged risk.

## Execution Gate

The gate is a **conditional edge function** (like `_safety_check_router`), not a separate graph node. This keeps it deterministic and fast with no LLM involvement.

### Location

Add the function `_execution_gate` in `src/quantstack/graphs/trading/graph.py`, alongside the existing router functions.

### Logic

```python
def _execution_gate(state: TradingState) -> str:
    """Halt pipeline if blocking nodes have errored or total errors exceed threshold."""
    ...
```

The function inspects `state["errors"]` and applies two rules:

1. **Blocking node error**: If any error string in the list contains a blocking node name (the error messages include the originating node), halt the pipeline immediately.
2. **Total error threshold**: If the total error count exceeds 2 (from any source --- blocking or non-blocking), halt as a safety net. The boundary values: count of 2 continues, count of 3 halts.

Return values:
- `"halt"` --- route to `END`, log the reason and which errors triggered it
- `"continue"` --- route to the next node in the pipeline

### Wiring into the Graph

The execution gate must be checked at two points in the trading graph:

1. **Before `risk_sizing`**: After `merge_parallel`, check the gate before entering the risk/execution pipeline. This is the primary gate.

   Current edge: `merge_parallel` -> `risk_sizing`

   New wiring: `merge_parallel` -> (conditional: `_execution_gate`) -> `risk_sizing` | `END`

2. **Before `execute_entries`**: A second check before actual order placement, in case errors accumulated in the `risk_sizing` -> `portfolio_construction` -> `merge_pre_execution` path.

   Current edge: `merge_pre_execution` -> `execute_entries`

   New wiring: `merge_pre_execution` -> (conditional: `_execution_gate`) -> `execute_entries` | `END`

Both conditional edges use the same `_execution_gate` function.

### Error Attribution

For rule 1 (blocking node check) to work, error strings must contain the originating node name. Verify that the existing error-catching pattern in nodes includes the node name in the error message. If not, the node output models from section-04 should standardize error formatting as `f"[{node_name}] {error_description}"`.

### Non-Blocking Node Failures

When a non-blocking node fails:
- The node's `safe_default()` class method (defined on its output model from section-04) provides a typed neutral response
- The pipeline continues with the safe default merged into state
- The error is still appended to the `errors` list and counts toward the total threshold

This means 3 non-blocking failures in a single cycle will still halt the pipeline via rule 2, which is the intended safety net.

## Tests

Place tests in `tests/unit/test_error_blocking.py`.

```python
"""Tests for error blocking and node classification.

Tests verify that the execution gate halts the pipeline when blocking nodes
fail and allows continuation (with safe defaults) when non-blocking nodes fail.
"""
import pytest


# --- Node Classification ---

# Test: NODE_CLASSIFICATION contains every node in the trading graph
# Test: NODE_CLASSIFICATION values are only "blocking" or "non_blocking"
# Test: blocking nodes are exactly: data_refresh, safety_check, position_review, risk_sizing, execute_exits


# --- Execution Gate: Blocking Node Errors ---

# Test: blocking node (data_refresh) error in state -> _execution_gate returns "halt"
# Test: blocking node (position_review) error in state -> _execution_gate returns "halt"
# Test: blocking node (execute_exits) error in state -> _execution_gate returns "halt"
#       (verify that entry pipeline is also blocked when exit fails)
# Test: single blocking error + multiple non-blocking successes -> still halts


# --- Execution Gate: Non-Blocking Node Errors ---

# Test: non-blocking node (plan_day) error -> _execution_gate returns "continue"
#       (safe default used, pipeline proceeds)
# Test: non-blocking node (entry_scan) error -> _execution_gate returns "continue"
#       (empty candidate list from safe_default, pipeline proceeds)


# --- Execution Gate: Total Error Threshold ---

# Test: error count = 0 -> _execution_gate returns "continue"
# Test: error count = 2 (boundary, all non-blocking) -> _execution_gate returns "continue"
# Test: error count = 3 (boundary, all non-blocking) -> _execution_gate returns "halt"
# Test: error count > 3 -> _execution_gate returns "halt"


# --- Mixed Scenarios ---

# Test: 1 blocking error + 1 non-blocking error -> halts (blocking error triggers rule 1,
#       regardless of total count)
# Test: 0 blocking errors + 3 non-blocking errors -> halts (rule 2 safety net)


# --- Graph Wiring ---

# Test: merge_parallel has conditional edge to _execution_gate (not direct edge to risk_sizing)
# Test: merge_pre_execution has conditional edge to _execution_gate (not direct edge to execute_entries)
# Test: _execution_gate "halt" routes to END
# Test: _execution_gate "continue" routes to risk_sizing (first gate) / execute_entries (second gate)
```

## Implementation Checklist

1. Define `NODE_CLASSIFICATION` dict in `src/quantstack/graphs/trading/graph.py` mapping all node names to `"blocking"` or `"non_blocking"`
2. Implement `_execution_gate(state: TradingState) -> str` in the same file
3. Verify error strings contain originating node names (inspect existing error-catching patterns in `src/quantstack/graphs/trading/nodes.py`)
4. Replace `graph.add_edge("merge_parallel", "risk_sizing")` with `graph.add_conditional_edges("merge_parallel", _execution_gate, {"continue": "risk_sizing", "halt": END})`
5. Replace `graph.add_edge("merge_pre_execution", "execute_entries")` with `graph.add_conditional_edges("merge_pre_execution", _execution_gate, {"continue": "execute_entries", "halt": END})`
6. Write tests in `tests/unit/test_error_blocking.py`
7. Run `uv run pytest tests/unit/test_error_blocking.py` to verify

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/graphs/trading/graph.py` | Add `NODE_CLASSIFICATION` dict, `_execution_gate` function, rewire two edges to conditional |
| `tests/unit/test_error_blocking.py` | **NEW** -- unit tests for classification, gate logic, and graph wiring |

## Tradeoffs

- **Why a conditional edge, not a separate node?** A node would add state-merge overhead and appear in traces as a "did nothing" step on every clean cycle. A conditional edge is pure Python, runs in microseconds, and keeps the graph topology readable.
- **Why threshold of 2 (halt at 3)?** Two non-blocking failures in a cycle could be coincidental (e.g., two flaky LLM calls). Three signals a systemic issue. This is configurable if experience shows a different threshold is better.
- **Why string-matching on error messages for node attribution?** It avoids a separate `blocking_errors` state field and works with the existing `errors: list[str]` accumulator. The cost is a coupling between error message format and gate logic --- standardizing error format in section-04's output models mitigates this.
