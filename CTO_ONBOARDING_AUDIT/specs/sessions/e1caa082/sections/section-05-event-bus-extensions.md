# Section 05: Event Bus Extensions

## Overview

The event bus (`src/quantstack/coordination/event_bus.py`) is a poll-based, append-only PostgreSQL coordination layer that lets the three graph services (research, trading, supervisor) communicate asynchronously. Phase 10 introduces several new subsystems -- tool lifecycle management, overnight autoresearch, feature factory, governance, consensus validation, and metacognitive agents -- that all need to broadcast state changes to other components. This section adds the new `EventType` enum values and an optional payload schema module so that publishers and consumers agree on event structure.

No architectural changes to the event bus itself are required. The existing publish/poll/cursor mechanism handles arbitrary event types. This section is purely additive: new enum members, payload documentation, and tests.

## Dependencies

- **None.** This section has no prerequisites. It is in Batch 1 (parallel with section-01-db-migrations and section-06-prompt-caching).
- Downstream consumers (sections 02, 03, 07, 08, 10, 11, 12, 13) depend on these event types being available.

## Tests First

File: `tests/unit/test_event_bus_extensions.py`

An existing test file `tests/unit/test_event_bus_cursor.py` covers the core publish/poll/cursor mechanics. The new test file validates that the extended event types integrate correctly with those mechanics.

```python
"""Tests for Phase 10 event bus extensions.

Each test verifies that a new EventType value can be published and polled
through the existing EventBus infrastructure without modification to the
bus itself.
"""

import pytest
from quantstack.coordination.event_bus import EventBus, Event, EventType


# --- Enum membership ---

@pytest.mark.parametrize("member_name", [
    "TOOL_ADDED",
    "TOOL_DISABLED",
    "EXPERIMENT_COMPLETED",
    "FEATURE_DECAYED",
    "FEATURE_REPLACED",
    "MANDATE_ISSUED",
    "META_OPTIMIZATION_APPLIED",
    "CONSENSUS_REQUIRED",
    "CONSENSUS_REACHED",
])
def test_new_event_type_is_valid_enum_member(member_name: str) -> None:
    """Each new EventType value must be a valid enum member."""
    assert hasattr(EventType, member_name)
    member = EventType[member_name]
    assert isinstance(member, EventType)


# --- Publish + poll round-trip per event type ---

def test_tool_added_publishes_and_polls(event_bus: EventBus) -> None:
    """TOOL_ADDED event round-trips through publish -> poll."""
    ...

def test_tool_disabled_publishes_and_polls(event_bus: EventBus) -> None:
    """TOOL_DISABLED event round-trips through publish -> poll."""
    ...

def test_experiment_completed_publishes_and_polls(event_bus: EventBus) -> None:
    """EXPERIMENT_COMPLETED event round-trips through publish -> poll."""
    ...

def test_mandate_issued_publishes_and_polls(event_bus: EventBus) -> None:
    """MANDATE_ISSUED event round-trips through publish -> poll."""
    ...

def test_meta_optimization_applied_publishes_and_polls(event_bus: EventBus) -> None:
    """META_OPTIMIZATION_APPLIED event round-trips through publish -> poll."""
    ...

def test_consensus_required_and_reached_publish_and_poll(event_bus: EventBus) -> None:
    """CONSENSUS_REQUIRED and CONSENSUS_REACHED events round-trip correctly."""
    ...

def test_feature_decayed_and_replaced_publish_and_poll(event_bus: EventBus) -> None:
    """FEATURE_DECAYED and FEATURE_REPLACED events round-trip correctly."""
    ...
```

Each round-trip test should follow the same pattern:

1. Create an `Event` with the new `EventType`, a `source_loop`, and a representative payload dict.
2. Call `bus.publish(event)`.
3. Call `bus.poll(consumer_id, event_types=[EventType.THE_TYPE])`.
4. Assert the returned list has exactly one event with matching type, source, and payload.

The `event_bus` fixture should provide an `EventBus` instance backed by an in-memory or test PostgreSQL connection with the `loop_events` and `loop_cursors` tables created. Reuse the pattern from `tests/unit/test_event_bus_cursor.py` if a fixture already exists there, or create one in `tests/unit/conftest.py`.

## Implementation

### 1. Add new EventType members

File: `src/quantstack/coordination/event_bus.py`

Add the following members to the `EventType` enum, grouped under a Phase 10 comment block. Place them after the existing Phase 7 feedback loop events:

```python
# Phase 10: Advanced Research coordination
TOOL_ADDED = "tool_added"
TOOL_DISABLED = "tool_disabled"
EXPERIMENT_COMPLETED = "experiment_completed"
FEATURE_DECAYED = "feature_decayed"
FEATURE_REPLACED = "feature_replaced"
MANDATE_ISSUED = "mandate_issued"
META_OPTIMIZATION_APPLIED = "meta_optimization_applied"
CONSENSUS_REQUIRED = "consensus_required"
CONSENSUS_REACHED = "consensus_reached"
```

That is the entirety of the change to `event_bus.py`. No methods, classes, or logic change.

### 2. Create payload schema documentation

File: `src/quantstack/coordination/event_schemas.py` (new)

This module documents the expected payload structure for each Phase 10 event type. It serves two purposes: (a) a single source of truth for publishers and consumers, and (b) optional runtime validation if a consumer wants to assert payload shape before processing.

Define one `TypedDict` (or `dataclass`) per event type's payload:

| EventType | Payload keys | Types | Notes |
|-----------|-------------|-------|-------|
| `TOOL_ADDED` | `tool_name`, `source` | `str`, `str` | `source` is `"synthesis"` or `"manual"` |
| `TOOL_DISABLED` | `tool_name`, `reason`, `success_rate` | `str`, `str`, `float` | `reason` is human-readable |
| `EXPERIMENT_COMPLETED` | `experiment_id`, `status`, `oos_ic` | `str`, `str`, `float \| None` | `status` is `"winner"` or `"rejected"` |
| `FEATURE_DECAYED` | `feature_id`, `psi`, `ic_current` | `str`, `float`, `float` | PSI > 0.25 triggers this |
| `FEATURE_REPLACED` | `old_feature_id`, `new_feature_id` | `str`, `str` | |
| `MANDATE_ISSUED` | `mandate_id`, `key_directives` | `str`, `str` | `key_directives` is a summary string |
| `META_OPTIMIZATION_APPLIED` | `agent_id`, `change_type`, `change_summary` | `str`, `str`, `str` | `change_type` is `"prompt"`, `"threshold"`, or `"tool"` |
| `CONSENSUS_REQUIRED` | `signal_id`, `symbol`, `notional` | `str`, `str`, `float` | |
| `CONSENSUS_REACHED` | `decision_id`, `consensus_level`, `final_sizing` | `str`, `str`, `float` | `consensus_level` is `"unanimous"`, `"majority"`, or `"minority"` |

The module should expose:

- A `TypedDict` subclass for each payload (e.g., `ToolAddedPayload`, `MandateIssuedPayload`).
- A mapping `EVENT_PAYLOAD_SCHEMAS: dict[EventType, type]` from event type to the corresponding TypedDict for programmatic lookup.
- An optional `validate_payload(event_type: EventType, payload: dict) -> bool` helper that checks required keys are present and types are correct. This is for defensive consumers; the event bus itself does not enforce schemas (it stores payloads as JSONB).

### 3. Consumers that will use these events (reference only)

The following sections will subscribe to these events in their respective `context_load` or cycle-start code. No implementation needed here -- just documenting the contract:

- **section-02 (tool lifecycle):** Publishes `TOOL_ADDED` and `TOOL_DISABLED`. Research/trading graphs poll these to rebuild tool bindings.
- **section-07 (overnight autoresearch):** Publishes `EXPERIMENT_COMPLETED` after each experiment. Supervisor graph polls to track overnight progress.
- **section-08 (feature factory):** Publishes `FEATURE_DECAYED` and `FEATURE_REPLACED` when monitoring detects drift. Research graph polls to update feature sets.
- **section-11 (consensus validation):** Publishes `CONSENSUS_REQUIRED` when routing a trade to consensus, and `CONSENSUS_REACHED` when the decision is made. Trading graph nodes use these for flow control.
- **section-12 (governance):** Publishes `MANDATE_ISSUED` when the CIO agent produces a daily mandate. Trading graph strategy agents poll this at cycle start.
- **section-13 (meta agents):** Publishes `META_OPTIMIZATION_APPLIED` after any prompt, threshold, or tool binding change. Supervisor graph logs these for audit.

## Files Summary

| File | Action | What changes |
|------|--------|-------------|
| `src/quantstack/coordination/event_bus.py` | MODIFY | Add 9 new `EventType` enum members |
| `src/quantstack/coordination/event_schemas.py` | CREATE | Payload TypedDicts, schema mapping, optional validator |
| `tests/unit/test_event_bus_extensions.py` | CREATE | Enum membership tests + publish/poll round-trip tests |

## Verification

After implementation, run:

```bash
uv run pytest tests/unit/test_event_bus_extensions.py -v
```

All tests should pass. Additionally, run the existing event bus tests to confirm no regressions:

```bash
uv run pytest tests/unit/test_event_bus_cursor.py -v
```

The existing `EventBus.publish()` and `EventBus.poll()` methods require no changes -- they already handle any `EventType` member and store payloads as JSON. The only code change to `event_bus.py` is the new enum values.
