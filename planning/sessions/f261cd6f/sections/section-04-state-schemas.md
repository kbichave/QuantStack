# Section 04: State Schema Design

## Overview

This section defines the TypedDict state schemas for all three LangGraph state graphs: `ResearchState`, `TradingState`, and `SupervisorState`. These schemas are the typed contracts that replace CrewAI's implicit conversational memory. Every piece of data flowing between graph nodes is declared here with explicit types.

The schemas live in a single file: `src/quantstack/graphs/state.py`.

## Dependencies

- **section-01-scaffolding**: The `src/quantstack/graphs/` directory must exist before this file can be created.

## Blocks

- **section-06-supervisor-graph**, **section-07-research-graph**, **section-08-trading-graph**: All graph builders import these state schemas to define their `StateGraph[XState]`.

## Design Principles

### Keep state lean

Store IDs and summaries in graph state. Store heavy data (DataFrames, signal arrays, backtest results) in PostgreSQL application tables and reference them by ID. Graph state is serialized on every checkpoint -- bloating it with large objects degrades performance and makes debugging harder.

### Append-only fields use `Annotated[list[T], operator.add]`

LangGraph's reducer system allows fields to accumulate values across nodes. When a field is declared with `operator.add` as its reducer, each node's return value for that field is *appended* to (via list concatenation), not replaced.

Critical rules for node functions interacting with append-only fields:

- To append items: return a list (e.g., `{"errors": ["something went wrong"]}`)
- To append nothing: return an empty list `[]` for that field, or omit the field entirely from the return dict
- **Never return `None`** for an append-only field -- this will raise a `TypeError` because `None + list` is invalid

This is the most common LangGraph pitfall. Every node function must be aware of which fields are append-only.

### Explicit over implicit

CrewAI's `memory=True` gave agents implicit access to raw conversational output from earlier agents. LangGraph replaces this with explicit typed state. The tradeoff: we lose the ability for agents to reference arbitrary prior conversation text, but we gain predictable, debuggable, testable data flow. Before building any graph, audit existing `tasks.yaml` `context` fields to confirm no task relies on raw conversational output (as opposed to structured results).

## Tests (Write First)

Create `tests/unit/test_state_schemas.py`:

```python
"""Tests for LangGraph state schemas.

Write these tests BEFORE implementing src/quantstack/graphs/state.py.
"""
import operator
from typing import Annotated, get_type_hints


def test_research_state_has_all_required_fields():
    """ResearchState TypedDict has all required fields."""
    from quantstack.graphs.state import ResearchState

    hints = get_type_hints(ResearchState, include_extras=True)
    required = {
        "cycle_number", "regime", "context_summary", "selected_domain",
        "selected_symbols", "hypothesis", "validation_result", "backtest_id",
        "ml_experiment_id", "registered_strategy_id", "errors", "decisions",
    }
    assert required.issubset(hints.keys()), f"Missing fields: {required - hints.keys()}"


def test_trading_state_has_all_required_fields():
    """TradingState TypedDict has all required fields."""
    from quantstack.graphs.state import TradingState

    hints = get_type_hints(TradingState, include_extras=True)
    required = {
        "cycle_number", "regime", "portfolio_context", "daily_plan",
        "position_reviews", "exit_orders", "entry_candidates", "risk_verdicts",
        "fund_manager_decisions", "options_analysis", "entry_orders",
        "reflection", "errors", "decisions",
    }
    assert required.issubset(hints.keys()), f"Missing fields: {required - hints.keys()}"


def test_supervisor_state_has_all_required_fields():
    """SupervisorState TypedDict has all required fields."""
    from quantstack.graphs.state import SupervisorState

    hints = get_type_hints(SupervisorState, include_extras=True)
    required = {
        "cycle_number", "health_status", "diagnosed_issues",
        "recovery_actions", "strategy_lifecycle_actions",
        "scheduled_task_results", "errors",
    }
    assert required.issubset(hints.keys()), f"Missing fields: {required - hints.keys()}"


def test_append_only_fields_use_operator_add_reducer():
    """Append-only fields (errors, decisions) use operator.add as reducer."""
    from quantstack.graphs.state import ResearchState, TradingState, SupervisorState

    for schema in (ResearchState, TradingState, SupervisorState):
        hints = get_type_hints(schema, include_extras=True)
        errors_meta = hints["errors"].__metadata__
        assert operator.add in errors_meta, f"{schema.__name__}.errors missing operator.add reducer"

    # decisions field exists on Research and Trading only
    for schema in (ResearchState, TradingState):
        hints = get_type_hints(schema, include_extras=True)
        decisions_meta = hints["decisions"].__metadata__
        assert operator.add in decisions_meta, f"{schema.__name__}.decisions missing operator.add reducer"


def test_node_returning_empty_list_for_append_field_does_not_error():
    """A node returning [] for an append-only field should not raise."""
    # This is a behavioral test verifying LangGraph's reducer contract.
    # Simulates what happens when operator.add is applied with an empty list.
    existing = ["previous error"]
    update = []
    result = operator.add(existing, update)
    assert result == ["previous error"]


def test_node_omitting_append_field_from_return_does_not_error():
    """A node that omits an append-only field entirely should not error.

    LangGraph treats missing keys in a node's return dict as "no update"
    for that field. This test documents that contract.
    """
    # This is a contract documentation test -- the real verification
    # happens in graph integration tests (section-06, 07, 08).
    # Here we just verify that the reducer (operator.add) is not
    # called when the field is absent from the update dict.
    state = {"errors": ["existing"]}
    update = {}  # node returns no errors key
    # LangGraph would skip the reducer for missing keys.
    # No assertion needed beyond documenting the expectation.
    assert "errors" not in update


def test_node_returning_none_for_append_field_raises():
    """A node returning None for an append-only field should fail.

    operator.add(existing_list, None) raises TypeError. This test
    documents the pitfall so implementers avoid it.
    """
    import pytest

    existing = ["previous error"]
    with pytest.raises(TypeError):
        operator.add(existing, None)
```

## Implementation

Create `src/quantstack/graphs/state.py`:

```python
"""LangGraph state schemas for all graph pipelines.

Each TypedDict defines the complete state contract for a graph. Nodes
read fields they need and return dicts with the fields they update.

Append-only fields use Annotated[list[T], operator.add] so values
accumulate across nodes rather than being overwritten. Nodes must
return [] (not None) for append-only fields they don't want to update,
or omit the field entirely from their return dict.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict
```

### ResearchState

Represents the full state of a single research cycle. The pipeline flows: context loading, domain selection, hypothesis generation, signal validation, backtesting, ML experimentation, strategy registration, and knowledge update.

Fields:

| Field | Type | Purpose |
|-------|------|---------|
| `cycle_number` | `int` | Monotonically increasing cycle counter |
| `regime` | `str` | Current market regime (e.g., "trending_up", "ranging") |
| `context_summary` | `str` | Text summary from context_load node |
| `selected_domain` | `str` | Research domain: "swing", "investment", or "options" |
| `selected_symbols` | `list[str]` | Ticker symbols selected for research |
| `hypothesis` | `str` | Testable hypothesis generated by LLM |
| `validation_result` | `dict` | Signal validation output (includes "passed" bool) |
| `backtest_id` | `str` | Reference to backtest results stored in DB |
| `ml_experiment_id` | `str` | Reference to ML experiment stored in DB |
| `registered_strategy_id` | `str` | ID of newly registered strategy |
| `errors` | `Annotated[list[str], operator.add]` | Append-only error accumulator |
| `decisions` | `Annotated[list[dict], operator.add]` | Append-only audit trail |

### TradingState

Represents the full state of a single trading cycle. The pipeline flows: safety check, daily planning, parallel position review + entry scan, risk sizing, portfolio review, options analysis, entry execution, and reflection.

Fields:

| Field | Type | Purpose |
|-------|------|---------|
| `cycle_number` | `int` | Monotonically increasing cycle counter |
| `regime` | `str` | Current market regime |
| `portfolio_context` | `dict` | Positions, cash, exposure snapshot |
| `daily_plan` | `str` | Daily planner LLM output |
| `position_reviews` | `list[dict]` | Per-position HOLD/TRIM/CLOSE decisions |
| `exit_orders` | `list[dict]` | Executed exit order confirmations |
| `entry_candidates` | `list[dict]` | Scanned entry candidates |
| `risk_verdicts` | `list[dict]` | Per-candidate risk sizing and Kelly results |
| `fund_manager_decisions` | `list[dict]` | APPROVED/REJECTED per candidate |
| `options_analysis` | `list[dict]` | Options structures for eligible candidates |
| `entry_orders` | `list[dict]` | Executed entry order confirmations |
| `reflection` | `str` | Trade reflector analysis and lessons |
| `errors` | `Annotated[list[str], operator.add]` | Append-only error accumulator |
| `decisions` | `Annotated[list[dict], operator.add]` | Append-only audit trail |

### SupervisorState

Represents the full state of a single supervisor cycle. Linear pipeline: health check, diagnosis, recovery, strategy lifecycle, scheduled tasks.

Fields:

| Field | Type | Purpose |
|-------|------|---------|
| `cycle_number` | `int` | Monotonically increasing cycle counter |
| `health_status` | `dict` | System health check results |
| `diagnosed_issues` | `list[dict]` | Issues found by diagnosis |
| `recovery_actions` | `list[dict]` | Actions taken to recover |
| `strategy_lifecycle_actions` | `list[dict]` | Strategy promotions/retirements |
| `scheduled_task_results` | `list[dict]` | Scheduled task outcomes |
| `errors` | `Annotated[list[str], operator.add]` | Append-only error accumulator |

## File Paths

| Action | Path |
|--------|------|
| Create | `src/quantstack/graphs/state.py` |
| Create | `tests/unit/test_state_schemas.py` |

## Implementation Notes

- The `__init__.py` for `src/quantstack/graphs/` should re-export the three state classes for convenient imports: `from quantstack.graphs import ResearchState, TradingState, SupervisorState`.
- All three schemas intentionally use simple types (`str`, `dict`, `list[dict]`, `list[str]`, `int`). Avoid dataclasses or Pydantic models in state -- LangGraph's checkpoint serializer works best with JSON-native types.
- The `validation_result` dict in `ResearchState` must contain a `"passed"` boolean key. The conditional edge after `signal_validation` (section-07) reads `state["validation_result"]["passed"]` to decide routing. This is a cross-section contract.
- The `portfolio_context` dict in `TradingState` is populated by the runner before graph invocation. It contains current positions, available cash, and exposure metrics. Its schema is defined by the existing portfolio query functions.
- No `persist_state` or checkpoint-related fields are needed in these schemas. LangGraph's `AsyncPostgresSaver` (configured in section-11) handles checkpoint persistence transparently after every node.
