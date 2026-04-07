# Section 02: State Key Audit

## Purpose

Before migrating the 4 graph states (`TradingState`, `ResearchState`, `SupervisorState`, `SymbolValidationState`) from `TypedDict` to Pydantic `BaseModel` with `extra="forbid"`, we must discover every key that nodes actually write at runtime. Today, `TypedDict` silently accepts any key a node returns — typos, undeclared fields, experimental fields that were never added to the schema. Once `extra="forbid"` is active, every undeclared key becomes a `ValidationError` that crashes the pipeline.

The audit prevents the Pydantic migration (section-03) from being a "discover bugs one by one in production" exercise.

## Dependencies

- **Depends on**: section-01-db-migration-and-policy (the DB migration must be applied so the audit tooling can optionally persist results, though the audit itself only reads source code and inspects runtime returns)
- **Blocks**: section-03-pydantic-state-migration (cannot write Pydantic models until the field inventory is finalized)

## Known Ghost Fields

A preliminary static analysis of the codebase has already identified the following keys returned by nodes that do **not** exist in the corresponding `TypedDict` definitions in `src/quantstack/graphs/state.py`:

### TradingState Ghost Fields

| Ghost Key | Written By | Current TypedDict Has It? | Resolution |
|-----------|-----------|--------------------------|------------|
| `alpha_signals` | `risk_sizing` node (via `make_risk_sizing()`) | No | Add to TradingState as `alpha_signals: list` |
| `alpha_signal_candidates` | `risk_sizing` node | No | Add to TradingState as `alpha_signal_candidates: list[dict]` |

**Evidence**: `risk_sizing` returns `{"alpha_signals": alpha_signals.tolist(), "alpha_signal_candidates": filtered_candidates, ...}` (lines 615-616 of `src/quantstack/graphs/trading/nodes.py`). The `_risk_gate_router` in `graph.py` reads `state.get("alpha_signals", [])` (line 77), confirming the field is load-bearing for control flow — not dead code.

### SupervisorState Ghost Fields

The supervisor graph has standalone functions (`run_attribution`, `run_signal_scoring`, `run_ic_computation`, `run_execution_quality_scoring`, `run_mmc_computation`, `run_regime_detection`) called from within the `scheduled_tasks` node. These functions return dicts with keys like `positions_processed`, `rows_written`, `strategies_scored`, `signals_written`, `strategies_computed`, `ic_decay_events`, `symbols_scored`, `symbols_skipped`, `regime`, `regime_change`, `method`, etc. However, these return values are consumed **within** the `scheduled_tasks` node and rolled into the `scheduled_task_results` list — they do not propagate as top-level state keys. No ghost fields detected for `SupervisorState`.

### ResearchState and SymbolValidationState

Static analysis shows all node returns match declared `TypedDict` fields. No ghost fields detected.

## Audit Approach

The audit has two phases: static analysis (automated, runs in CI) and dynamic validation (runtime instrumentation for paper trading).

### Phase 1: Static Analysis (Automated)

Write a test that programmatically inventories every key returned by every node and asserts membership in the parent state schema. This test runs in CI and catches regressions going forward.

**File**: `tests/unit/test_state_key_audit.py`

```python
# Test: state_key_audit_trading — extract all keys from every return statement in
# trading/nodes.py, assert each key exists in TradingState.__annotations__
# (after ghost fields are added). Use AST parsing or manual enumeration.

# Test: state_key_audit_research — same for ResearchState from research/nodes.py

# Test: state_key_audit_supervisor — same for SupervisorState from supervisor/nodes.py

# Test: state_key_audit_symbol_validation — same for SymbolValidationState
# from research/nodes.py (validate_symbol worker)

# Test: alpha_signals_ghost_resolved — verify that TradingState includes
# alpha_signals and alpha_signal_candidates after the fix is applied
```

**Implementation guidance for the test**:

The most reliable approach is to maintain a manually-curated mapping of `{node_name: set_of_keys_it_returns}` and assert each key exists in the state's `__annotations__`. AST-based extraction of return dict keys is fragile (nodes use variables, conditional returns, etc.), so a curated map verified against actual code is preferable. The map itself serves as living documentation of which node writes which fields.

Example structure:

```python
TRADING_NODE_KEYS: dict[str, set[str]] = {
    "market_intel": {"market_context"},
    "earnings_analysis": {"earnings_analysis"},
    "data_refresh": {"data_refresh_summary", "errors"},
    "safety_check": {"errors", "decisions"},
    "plan_day": {"daily_plan", "earnings_symbols", "decisions", "errors"},
    "position_review": {"position_reviews", "exit_orders", "errors", "decisions"},
    "execute_exits": {"exit_orders", "errors", "decisions"},
    "entry_scan": {"entry_candidates", "errors", "decisions"},
    "merge_parallel": set(),  # returns {}
    "merge_pre_execution": set(),  # returns {}
    "risk_sizing": {
        "alpha_signals", "alpha_signal_candidates",
        "vol_state", "decisions", "errors",
    },
    "portfolio_construction": {
        "portfolio_target_weights", "risk_verdicts",
        "last_covariance", "decisions", "errors",
    },
    "portfolio_review": {"fund_manager_decisions", "decisions", "errors"},
    "analyze_options": {"options_analysis", "decisions", "errors"},
    "execute_entries": {"entry_orders", "decisions", "errors"},
    "reflect": {
        "reflection", "trade_quality_scores",
        "attribution_contexts", "decisions", "errors",
    },
}

def test_all_trading_node_keys_in_state():
    all_node_keys = set()
    for keys in TRADING_NODE_KEYS.values():
        all_node_keys |= keys
    state_fields = set(TradingState.__annotations__.keys())
    undeclared = all_node_keys - state_fields
    assert not undeclared, f"Ghost fields not in TradingState: {undeclared}"
```

Repeat the pattern for `RESEARCH_NODE_KEYS`, `SUPERVISOR_NODE_KEYS`, and `SYMBOL_VALIDATION_NODE_KEYS`.

### Phase 2: Dynamic Validation (Runtime Instrumentation)

Instrument each graph to log every key returned by every node during paper trading. This catches keys that static analysis misses (e.g., keys constructed dynamically, keys returned only under rare conditions).

**Approach**: Add a lightweight wrapper around node return values in each graph's node registration. The wrapper inspects the returned dict keys and logs any key not present in the TypedDict's `__annotations__`. This is temporary instrumentation — remove it after the Pydantic migration is validated.

```python
# Sketch — not a production implementation, just to illustrate the approach.
# Place in a shared utility, e.g., src/quantstack/graphs/_audit.py

def audit_node_return(node_name: str, state_class: type, returned: dict) -> dict:
    """Log any keys in `returned` that aren't declared in `state_class`."""
    declared = set(state_class.__annotations__.keys())
    actual = set(returned.keys())
    ghost = actual - declared
    if ghost:
        logger.warning(
            "GHOST FIELDS: node=%s returned undeclared keys: %s",
            node_name, ghost,
        )
    return returned
```

Run for at least 10 full cycles of paper trading across all 3 graphs. Collect results. Any new ghost fields discovered are added to the Pydantic models in section-03.

## Resolution Plan

For each ghost field discovered by the audit, exactly one of these actions:

1. **Add to state schema** — if the field is read by downstream nodes or router functions (i.e., it's load-bearing). Example: `alpha_signals` is read by `_risk_gate_router` and `portfolio_construction`, so it must be added.

2. **Remove from node return** — if the field is written but never read downstream (dead write). This is the preferred resolution for fields that were experimental or accidental.

3. **Rename to match existing field** — if the field is a typo of an existing declared field (e.g., `daly_plan` instead of `daily_plan`).

## Specific Resolutions (From Static Analysis)

The following changes must be made to `src/quantstack/graphs/state.py` before the Pydantic migration:

### TradingState

Add two fields:

- `alpha_signals: list` — written by `risk_sizing`, read by `_risk_gate_router` and `portfolio_construction`. Contains IC-adjusted Kelly alpha signal values as a flat list of floats.
- `alpha_signal_candidates: list[dict]` — written by `risk_sizing`, read by `portfolio_construction`. Contains the filtered entry candidates that correspond 1:1 with `alpha_signals`.

These fields are critical to the trading pipeline's control flow. Without them declared, the Pydantic migration with `extra="forbid"` would crash `risk_sizing` on every cycle that has entry candidates.

### ResearchState, SupervisorState, SymbolValidationState

No changes needed based on current static analysis. Dynamic validation (Phase 2) may surface additional fields.

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/graphs/state.py` | Add `alpha_signals` and `alpha_signal_candidates` to `TradingState` |
| `tests/unit/test_state_key_audit.py` | **NEW** — Static audit tests for all 4 state classes |
| `src/quantstack/graphs/_audit.py` | **NEW** (temporary) — Runtime instrumentation wrapper for dynamic validation |

## Completion Criteria

1. Static audit test passes: every key returned by every node across all 3 graphs exists in the corresponding state schema
2. `alpha_signals` and `alpha_signal_candidates` are declared in `TradingState`
3. Dynamic instrumentation has been run for 10+ paper trading cycles with zero new ghost fields logged
4. The resulting field inventory is documented and handed off to section-03 (Pydantic migration) as the authoritative list of fields per state class
