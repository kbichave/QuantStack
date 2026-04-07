# Section 06: EventBus Extension — New Event Types

## Overview

The EventBus (`src/quantstack/coordination/event_bus.py`) is a PostgreSQL-backed, poll-based, append-only event log used for inter-loop coordination across the three LangGraph StateGraphs (Research, Trading, Supervisor). It already supports event types like `STRATEGY_PROMOTED`, `REGIME_CHANGE`, `MODEL_DEGRADATION`, `IC_DECAY`, and various risk events.

Three downstream sections require new event types that do not yet exist in the `EventType` enum:

| New Event Type | Published By | Consumed By | Purpose |
|----------------|-------------|-------------|---------|
| `SIGNAL_DEGRADATION` | IC weight adjustment (section-07) | Research graph | A collector's IC has dropped below 0.02 from a previously healthy level; triggers investigation |
| `SIGNAL_CONFLICT` | Conflict resolution (section-09) | Research graph, daily plan | Collectors disagree strongly (spread > 0.5); conviction capped |
| `AGENT_DEGRADATION` | Agent quality tracking (section-11) | Research graph | An agent's rolling win rate dropped below 40%; triggers prompt investigation |

This section is purely additive — three new enum values, no changes to EventBus methods, no schema changes. The `loop_events` table stores `event_type` as TEXT, so new enum values are automatically supported without migration.

## Dependencies

- **None.** This section has no prerequisites and can be implemented in Batch 1.

## Blocks

- **section-07-ic-weight-adjustment**: publishes `SIGNAL_DEGRADATION`
- **section-09-conflict-resolution**: publishes `SIGNAL_CONFLICT`
- **section-11-agent-quality**: publishes `AGENT_DEGRADATION`
- **section-12-sharpe-demotion**: uses EventBus (but uses the already-existing `STRATEGY_DEMOTED` type)

## Tests First

File: `tests/unit/test_eventbus_extension.py`

### Test: new event types are valid EventType members

Verify that `SIGNAL_DEGRADATION`, `SIGNAL_CONFLICT`, and `AGENT_DEGRADATION` exist on the `EventType` enum and serialize to their expected string values (`"signal_degradation"`, `"signal_conflict"`, `"agent_degradation"`).

### Test: new event types are str-compatible

Since `EventType` inherits from `str, Enum`, confirm that `str(EventType.SIGNAL_DEGRADATION)` and direct string comparison work correctly. This matters because the EventBus stores the `.value` in PostgreSQL as TEXT and reconstructs via `EventType(etype)` on poll.

### Test: publish and poll round-trip for each new type

Using a mock `PgConnection` (consistent with existing test patterns), publish an `Event` with each new type and verify that `poll()` returns it with the correct type and payload intact. The payload structures to test:

- `SIGNAL_DEGRADATION`: `{"collector": "technical_rsi", "current_ic": 0.005, "previous_ic": 0.04, "regime": "trending_up"}`
- `SIGNAL_CONFLICT`: `{"symbol": "AAPL", "conflicting_collectors": ["technical_rsi", "ml_direction"], "max_signal": 0.8, "min_signal": 0.2, "spread": 0.6}`
- `AGENT_DEGRADATION`: `{"agent_id": "trade_debater", "win_rate": 0.35, "trade_count": 42, "recent_losses": 8}`

### Test: get_latest works for new types

Verify that `get_latest(EventType.SIGNAL_DEGRADATION)` returns the most recent event of that type, and returns `None` when no events of that type exist.

### Test: count_events filters correctly for new types

Verify that `count_events(event_type=EventType.AGENT_DEGRADATION)` returns only counts for that specific type.

### Test: existing event types unchanged

Regression check — verify that all pre-existing event types (`STRATEGY_PROMOTED`, `REGIME_CHANGE`, `IC_DECAY`, `MODEL_DEGRADATION`, etc.) still exist and have the same string values. This prevents accidental reordering or renaming when editing the enum.

## Implementation

### File to modify

`src/quantstack/coordination/event_bus.py`

### Change

Add three new members to the `EventType` enum class, after the existing entries. Place them in a logical group with a comment indicating their purpose:

```python
class EventType(str, Enum):
    # ... existing members unchanged ...

    # Feedback loop event types (Phase 7)
    SIGNAL_DEGRADATION = "signal_degradation"
    SIGNAL_CONFLICT = "signal_conflict"
    AGENT_DEGRADATION = "agent_degradation"
```

That is the entire implementation. No other files need modification. No new tables, no schema changes, no method changes.

### Why this is safe

1. The `EventType` enum is used in three places within `event_bus.py`: `publish()` reads `.value` to store as TEXT, `poll()` reconstructs via `EventType(etype)` with a `ValueError` fallback for unknown types, and `get_latest()`/`count_events()` filter by `.value`. Adding new members does not change any of these code paths.

2. The `loop_events` table stores `event_type` as an unconstrained TEXT column — no CHECK constraint or PostgreSQL enum type to migrate.

3. No existing code references these new types yet. They become meaningful only when downstream sections (07, 09, 11) publish and poll for them.

### Payload conventions

Each new event type has an expected payload schema. These are not enforced at the EventBus level (payloads are free-form JSON dicts), but downstream publishers and consumers should follow these structures:

**SIGNAL_DEGRADATION** — published when a collector's IC crosses below 0.02 from a previously healthy level:
```python
{
    "collector": str,       # collector name (e.g. "technical_rsi")
    "current_ic": float,    # current rolling 21-day IC
    "previous_ic": float,   # IC before degradation
    "regime": str           # active regime when degradation detected
}
```

**SIGNAL_CONFLICT** — published when collector vote spread exceeds 0.5 during synthesis:
```python
{
    "symbol": str,                    # affected symbol
    "conflicting_collectors": list,   # names of disagreeing collectors
    "max_signal": float,              # highest collector vote
    "min_signal": float,              # lowest collector vote
    "spread": float                   # max - min
}
```

**AGENT_DEGRADATION** — published when an agent's rolling 30-trade win rate drops below 40%:
```python
{
    "agent_id": str,         # agent name (e.g. "trade_debater")
    "win_rate": float,       # current rolling win rate
    "trade_count": int,      # total trades in rolling window
    "recent_losses": int     # losses in last N trades
}
```

### Rollback

Revert the three-line addition to `EventType`. Any events already published with these types will remain in `loop_events` as TEXT rows; on the next `poll()`, they will hit the `except ValueError` fallback and be returned with the raw string type instead of an enum member. No data loss, no crash.
