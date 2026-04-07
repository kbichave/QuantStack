# Section 03: Pydantic State Schema Migration

## Purpose

The 4 graph states (`TradingState`, `ResearchState`, `SupervisorState`, `SymbolValidationState`) are currently `TypedDict` subclasses in `src/quantstack/graphs/state.py`. Nodes return plain dicts that are merged via `operator.add` (for accumulating lists) or last-write-wins. A typo like `{"daly_plan": "..."}` is silently accepted — the real `daily_plan` field stays stale with no error. This is the root cause of multiple hard-to-diagnose bugs where downstream nodes operate on stale or missing data.

The fix: migrate all 4 states to Pydantic `BaseModel` with `ConfigDict(extra="forbid")`. This catches typos, type mismatches, and undeclared fields at merge time rather than downstream when a stale value causes a bad trade.

## Dependencies

- **section-02-state-key-audit** must be complete before this section begins. That audit discovers every key returned by every node across all 3 graphs and resolves ghost fields (e.g., `_risk_gate_router` references `alpha_signals` which does not appear in `TradingState`). Without the audit, this migration becomes a "discover bugs one by one in production" exercise.

## What This Section Blocks

Sections 04, 05, 06, 07, 08, and 11 all depend on the Pydantic models being in place. Specifically:

- **section-04-node-output-models** — defines per-node output models as subsets of these parent state models
- **section-05-error-blocking** — uses typed state for error gating logic
- **section-06-race-condition-fix** — uses typed output models for the conflict resolution node
- **section-07-circuit-breaker** — safe defaults reference Pydantic model structure
- **section-08-tool-access-control** — references graph config loaded alongside state
- **section-11-message-pruning** — uses typed message metadata from Pydantic models

---

## Tests (Write First)

All tests go in `tests/unit/test_pydantic_state_migration.py`.

### Rejection tests — extra="forbid" catches bad keys

```python
# Test: TradingState rejects unknown key
# Construct TradingState(daly_plan="...") → raises ValidationError
# This is the core safety mechanism — typos caught at construction time

# Test: ResearchState rejects unknown key
# Construct ResearchState(hypotheesis="...") → raises ValidationError

# Test: SupervisorState rejects unknown key
# Test: SymbolValidationState rejects unknown key
```

### Type enforcement tests

```python
# Test: TradingState rejects wrong type
# Construct TradingState(cycle_number="not_an_int") → raises ValidationError

# Test: TradingState rejects wrong nested type
# Construct TradingState(position_reviews="not_a_list") → raises ValidationError
```

### Acceptance tests

```python
# Test: TradingState accepts valid state dict with all required fields
# Construct with every field populated with correct types → no error

# Test: ResearchState, SupervisorState, SymbolValidationState same acceptance pattern
```

### Reducer compatibility tests

```python
# Test: Annotated[list, operator.add] reducer still works with Pydantic BaseModel
# This is critical — LangGraph extracts reducer annotations identically from Pydantic
# Construct two partial state dicts with list fields → verify they accumulate (not overwrite)
```

### Field validator tests

```python
# Test: field_validator catches invalid vol_state value
# Construct TradingState(vol_state="invalid_state") → raises ValidationError
# Valid values: {"low", "normal", "high", "extreme"}

# Test: field_validator catches negative cycle_number
# Construct TradingState(cycle_number=-1) → raises ValidationError
```

### Model validator tests

```python
# Test: model_validator(mode="after") catches cross-field invariant violation
# Example: exit_orders is non-empty but position_reviews is empty → ValidationError
# (you can't generate exits without having reviewed positions)
```

### Input/Output schema tests

```python
# Test: TradingInput accepts valid input schema
# The input schema constrains what the graph accepts at invocation time

# Test: TradingOutput constrains graph output to expected shape
# The output schema constrains what the graph returns to callers
```

---

## Implementation Details

### File: `src/quantstack/graphs/state.py`

This is the only file modified in this section. The current file contains 4 `TypedDict` classes. All 4 are converted to Pydantic `BaseModel` in a single pass — no half-migrated state.

#### Current structure (TypedDict)

```python
from typing import Annotated, TypedDict
import operator

class TradingState(TypedDict):
    cycle_number: int
    regime: str
    # ... ~25 fields total
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]
```

#### Target structure (Pydantic BaseModel)

Each class changes from `TypedDict` to `BaseModel` with these additions:

1. **`model_config = ConfigDict(extra="forbid")`** — the primary safety mechanism. Any key not declared in the model raises `ValidationError` at construction time.

2. **Preserve `Annotated[list[T], operator.add]` reducers** — LangGraph extracts these annotations identically from Pydantic models as from TypedDict. No change needed to the annotation syntax. This is what makes accumulating fields (errors, decisions, validation_results) work across nodes.

3. **Field validators for domain invariants** — use `@field_validator` decorator:
   - `cycle_number`: must be `>= 0`
   - `vol_state` (on TradingState): must be one of `{"low", "normal", "high", "extreme"}`
   - `regime` (on all states that have it): should be one of the known regime values from the regime-strategy matrix (`trending_up`, `trending_down`, `ranging`, `unknown`)

4. **Model validators for cross-field invariants** — use `@model_validator(mode="after")`:
   - On TradingState: if `exit_orders` is non-empty, `position_reviews` must also be non-empty (exits derive from reviews)
   - Additional cross-field rules as discovered during the state key audit

5. **Default values** — all fields need defaults since nodes return partial state dicts (only the fields they update). List fields default to `[]`, string fields to `""`, dict fields to `{}`, int fields to `0`. This matches the current implicit behavior where missing keys in a TypedDict simply don't exist in the returned dict.

6. **Input/Output schemas** — define separate Pydantic models for graph boundaries:
   - `TradingInput` — the subset of TradingState that the graph accepts as invocation input (e.g., `cycle_number`, `regime`, `portfolio_context`)
   - `TradingOutput` — the subset of TradingState that the graph returns to callers
   - Same pattern for Research and Supervisor graphs

#### Import changes

The file's imports change from:

```python
from typing import Annotated, TypedDict
```

to:

```python
from typing import Annotated
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
```

`operator` import stays (still needed for `Annotated[list[T], operator.add]`).

### Key Decisions and Tradeoffs

- **`extra="forbid"` is non-negotiable.** This is the entire point of the migration. Without it, the Pydantic model is no safer than a TypedDict. The cost is that every node return must exactly match declared fields — any undiscovered ghost field from the state key audit will cause a runtime error. This is why section-02 must complete first.

- **All 4 states migrate at once.** A half-migrated codebase where some states are Pydantic and some are TypedDict creates confusion about which validation rules apply where. The blast radius is the same either way (all node returns need updating), so do it atomically.

- **Performance cost is negligible.** Pydantic validation adds microseconds per state construction. Each graph cycle involves seconds of LLM calls, network I/O, and DB queries. The validation overhead is unmeasurable in practice.

- **Every node return dict must be updated.** After this migration, every node across all 3 graphs must return dicts with keys that exactly match the declared fields. Any key mismatch that survived the state key audit will surface immediately as a `ValidationError`. This is the intended behavior — surface bugs at the source rather than downstream.

- **Default values are required for partial returns.** Unlike TypedDict where a node can return `{"daily_plan": "..."}` and only that field updates, Pydantic requires all fields to have values. The defaults (`""`, `[]`, `{}`, `0`) ensure that nodes can still return partial updates. LangGraph's state merge logic handles the rest — it only overwrites fields present in the node's return value.

### Downstream Impact

After this section is complete, every subsequent section that creates node output models (section-04), adds new state fields, or constructs state dicts in tests must use the Pydantic models. Test fixtures that construct raw dicts will need updating to pass validation — this is intentional and surfaces any test that was relying on invalid state.

---

## Implementation Checklist

1. Ensure section-02 (state key audit) results are available — know exactly which fields exist vs. which are ghost fields
2. Write all tests listed above in `tests/unit/test_pydantic_state_migration.py` (they should fail initially)
3. Convert `TradingState` from TypedDict to BaseModel with `extra="forbid"` and defaults
4. Convert `ResearchState` similarly
5. Convert `SupervisorState` similarly
6. Convert `SymbolValidationState` similarly
7. Add `field_validator` decorators for domain invariants
8. Add `model_validator` for cross-field invariants
9. Define `TradingInput`, `TradingOutput` (and equivalents for other graphs) as separate models in the same file
10. Run the full test suite — fix any node return dicts that now fail validation
11. Verify all tests from step 2 pass
