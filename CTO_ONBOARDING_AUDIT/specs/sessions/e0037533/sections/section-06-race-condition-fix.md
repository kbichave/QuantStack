# Section 06: Race Condition Fix (Parallel Branch Conflict Resolution)

## Background

The trading graph runs two parallel branches from `plan_day`:

1. **`position_review` -> `execute_exits`** — reviews existing positions, generates exit orders
2. **`entry_scan`** (optionally -> `earnings_analysis`) — scans for new entry candidates

Both branches converge at `merge_parallel` (a no-op join node). After merging, the pipeline flows to `risk_sizing`.

The race condition: both branches can reference the same symbol. `exit_orders` may contain a CLOSE order for AAPL while `entry_candidates` contains an ENTER signal for AAPL in the same cycle. The reducer layer (`operator.add` on both lists) does not detect this — it appends both. Downstream, `risk_sizing` and `execute_entries` process the entry candidate, potentially opening a new position in a symbol the system simultaneously decided to exit.

This is a correctness bug, not a performance issue. The system could issue conflicting orders for the same symbol within the same cycle.

## Dependencies

- **section-04-node-output-models**: The `resolve_symbol_conflicts` node returns a typed Pydantic output model. That model must be defined before this node can be implemented. Until section 04 is complete, use a plain dict return as a placeholder.

## Solution

Add a new node `resolve_symbol_conflicts` between `merge_parallel` and `risk_sizing` in the trading graph. This node inspects both lists, finds symbol overlaps, and removes conflicting entries — exits always win (risk-off bias).

### Graph Wiring Change

**File**: `src/quantstack/graphs/trading/graph.py`

Current edge:
```
merge_parallel -> risk_sizing
```

New edge chain:
```
merge_parallel -> resolve_symbol_conflicts -> risk_sizing
```

The change is surgical: replace the single `add_edge("merge_parallel", "risk_sizing")` with two edges, and register the new node.

### Node Logic

**File**: `src/quantstack/graphs/trading/nodes.py`

Add a new node function `resolve_symbol_conflicts` (not a factory — it needs no LLM or tools, so it can be a plain async function like `merge_parallel`).

The logic:

1. Read `exit_orders` from state — extract the set of symbols with pending exits.
2. Read `entry_candidates` from state — extract the set of symbols with pending entries.
3. Compute the intersection (symbols appearing in both lists).
4. If intersection is empty, return state unchanged (no conflict).
5. For each conflicting symbol:
   - Remove from `entry_candidates` (exits take priority).
   - Log the conflict: symbol name, exit reasoning (from the exit order), entry reasoning (from the entry candidate), and the resolution ("exit_priority").
6. Return the filtered `entry_candidates`. Do NOT modify `exit_orders` — exits are untouched.

Key implementation details:

- The node reads `exit_orders` and `entry_candidates` as accumulated lists from the parallel branches. Both use `operator.add` reducers, so by the time `merge_parallel` completes, both lists contain their final values.
- Symbol extraction should handle both `{"symbol": "AAPL"}` dict entries and any nested structure. Defensive `.get("symbol", "")` on each item.
- The node returns only the fields it modifies: `entry_candidates` (the filtered list) and `decisions` (conflict log). It does NOT return `exit_orders` since those are unchanged.
- Since `entry_candidates` is NOT an `operator.add` field in `TradingState` (it's a plain `list[dict]` with last-write-wins semantics), returning the filtered list overwrites the prior value. This is correct behavior.

### Safe Default on Failure

If the node itself throws an exception, the safe default is: drop ALL entry candidates that share a symbol with any exit order. This is more conservative than the normal path (which only drops the intersection) but guarantees no conflicting orders reach execution.

The safe default implementation: wrap the node body in try/except. On failure, compute the intersection conservatively (any symbol in exits blocks all entries for that symbol) and return an empty `entry_candidates` list plus an error entry. This ensures the pipeline can continue without conflicting orders even if the conflict resolution logic itself fails.

```python
async def resolve_symbol_conflicts(state: TradingState) -> dict[str, Any]:
    """Remove entry candidates that conflict with pending exit orders.

    Exits always take priority (risk-off bias). On failure, drops ALL
    entries as a conservative safe default.
    """
    # Implementation: extract symbols, find intersection, filter entries, log conflicts.
    # See logic steps 1-6 above.
    ...
```

### Conflict Event Logging

Each conflict produces a structured log entry appended to the `decisions` list:

```python
{
    "node": "resolve_symbol_conflicts",
    "conflicts": [
        {
            "symbol": "AAPL",
            "exit_reason": "regime_flip_severe",
            "entry_reason": "momentum_breakout",
            "resolution": "exit_priority",
        }
    ],
    "entries_before": 5,
    "entries_after": 3,
}
```

This provides full audit trail for every conflict resolution decision.

### Graph Registration

In `build_trading_graph()`:

1. Import `resolve_symbol_conflicts` from `.nodes`
2. Register the node: `graph.add_node("resolve_symbol_conflicts", resolve_symbol_conflicts)`
3. Replace the edge `graph.add_edge("merge_parallel", "risk_sizing")` with:
   ```python
   graph.add_edge("merge_parallel", "resolve_symbol_conflicts")
   graph.add_edge("resolve_symbol_conflicts", "risk_sizing")
   ```

No retry policy needed — this node is deterministic (no LLM, no I/O, no external calls). It reads state and returns filtered state.

## Tests

**File**: `tests/unit/test_race_condition_fix.py`

All tests construct state dicts directly and call `resolve_symbol_conflicts()` as an async function. No graph compilation needed for unit tests.

### Test: overlapping symbol removed from entries, preserved in exits

Construct state with `exit_orders` containing `{"symbol": "AAPL", "action": "CLOSE"}` and `entry_candidates` containing `{"symbol": "AAPL", "verdict": "ENTER"}` plus `{"symbol": "MSFT", "verdict": "ENTER"}`. Call `resolve_symbol_conflicts`. Assert:
- Returned `entry_candidates` contains only MSFT
- AAPL is not in the returned entries
- `exit_orders` is not in the return dict (unchanged)

### Test: multiple overlapping symbols all resolved

State with exits for AAPL and TSLA, entries for AAPL, TSLA, and MSFT. Assert returned entries contain only MSFT.

### Test: no overlapping symbols, both lists unchanged

State with exits for AAPL, entries for MSFT and GOOG. Assert returned `entry_candidates` equals the original list (no filtering).

### Test: conflict event logged with symbol, exit reasoning, entry reasoning

Assert the returned `decisions` list contains a dict with `"node": "resolve_symbol_conflicts"` and a `"conflicts"` list with one entry per conflicting symbol. Each conflict entry must include `symbol`, `exit_reason`, `entry_reason`, and `resolution`.

### Test: node failure triggers safe default (drops all conflicted entries)

Mock an internal failure (e.g., patch a function to raise). Assert the node returns empty `entry_candidates` and an error in `errors`.

### Test: empty exit_orders or empty entry_candidates is a no-op

Two sub-cases:
- `exit_orders = []`, `entry_candidates = [{"symbol": "AAPL"}]` — entries unchanged.
- `exit_orders = [{"symbol": "AAPL"}]`, `entry_candidates = []` — entries still empty, no crash.

## Verification Checklist

- [ ] `resolve_symbol_conflicts` node added to `nodes.py`
- [ ] Node registered in `graph.py` with correct edges (merge_parallel -> resolve_symbol_conflicts -> risk_sizing)
- [ ] Node imported in `graph.py` imports block
- [ ] Exits always take priority over entries for the same symbol
- [ ] Conflict events logged in `decisions` with full context
- [ ] Safe default on failure drops all conflicted entries
- [ ] Empty lists handled without errors
- [ ] All 6 test cases pass
- [ ] Graph docstring at top of `graph.py` updated to reflect new node in the pipeline
