# Section 7: EventBus ACK Pattern

## Overview

The EventBus (`src/quantstack/coordination/event_bus.py`) is an append-only PostgreSQL event log with per-consumer cursors. Currently, events are fire-and-forget: a publisher writes an event, consumers poll and process it, but there is no confirmation that critical events were actually received and acted upon. For risk events (liquidation orders, regime changes, IC decay), silent drops are unacceptable.

This section adds an acknowledgement (ACK) protocol to the EventBus. Publishers mark certain event types as requiring ACK. Consumers call `bus.ack()` after processing. A supervisor-side monitor detects missed ACKs and escalates through retry, warning, and dead-letter tiers.

### What changes

- **EventBus.publish()** — conditionally sets `requires_ack`, `expected_ack_by` on risk events
- **EventBus.ack()** — new method, sets `acked_at` and `acked_by` on an event row
- **check_missed_acks()** — new supervisor health-check function, detects unacknowledged events and escalates
- **Consumer-side pattern** — each graph runner calls `bus.ack()` after processing ACK-required events
- **Dead letter table** — events that exceed retry thresholds are moved to `dead_letter_events`

### Dependencies

- **Section 01 (DB Schema):** The `loop_events` table must have four new columns (`requires_ack`, `expected_ack_by`, `acked_at`, `acked_by`) and the `dead_letter_events` table must exist. If Section 01 is not yet implemented, you must add these columns and table before starting this section.
- **Section 02 (System Alerts):** `emit_system_alert()` from `src/quantstack/tools/functions/system_alerts.py` is used to create alerts on missed ACKs. If Section 02 is not yet implemented, stub `emit_system_alert()` as a no-op or direct DB insert.

### Files to modify

| File | Change |
|------|--------|
| `src/quantstack/coordination/event_bus.py` | Add `ACK_REQUIRED_EVENTS`, `ACK_TIMEOUT_SECONDS`, modify `publish()`, add `ack()` method, add `check_missed_acks()` |
| `src/quantstack/graphs/supervisor/nodes.py` | Call `check_missed_acks()` from the `health_check` node |
| `src/quantstack/graphs/trading/nodes.py` | Add `bus.ack()` calls after event processing |
| `src/quantstack/graphs/research/graph.py` | Add `bus.ack()` calls after event processing (if research graph polls risk events) |
| `tests/unit/test_eventbus_ack.py` | New file — unit tests |
| `tests/integration/test_eventbus_ack_integration.py` | New file — integration tests |

---

## Tests (Write First)

### Unit Tests — `tests/unit/test_eventbus_ack.py`

```python
"""Unit tests for EventBus ACK pattern."""

import pytest
from datetime import datetime, timedelta, timezone


# Test: publish() sets requires_ack=True for events in ACK_REQUIRED_EVENTS
def test_publish_sets_requires_ack_for_risk_events():
    """Publish a RISK_WARNING event. Query the loop_events row and assert
    requires_ack is True and expected_ack_by is set to ~now + 600s."""


# Test: publish() sets requires_ack=False for events NOT in ACK_REQUIRED_EVENTS
def test_publish_sets_requires_ack_false_for_non_risk_events():
    """Publish a STRATEGY_PROMOTED event. Assert requires_ack is False
    and expected_ack_by is NULL."""


# Test: publish() sets expected_ack_by = now + 600s for ACK-required events
def test_publish_expected_ack_by_is_600s_from_now():
    """Publish a RISK_LIQUIDATION event. Assert expected_ack_by is within
    a few seconds of now + 600s (allow clock skew tolerance)."""


# Test: ack() sets acked_at and acked_by on event row
def test_ack_sets_fields():
    """Publish a RISK_WARNING event, then call bus.ack(event_id, 'trading_graph').
    Query the row and assert acked_at is set and acked_by == 'trading_graph'."""


# Test: ack() is idempotent — re-acking doesn't error or change timestamps
def test_ack_idempotent():
    """Publish and ack an event. Record acked_at. Call ack() again.
    Assert acked_at is unchanged and no exception is raised."""


# Test: check_missed_acks returns empty list when all events ACKed on time
def test_check_missed_acks_all_acked():
    """Publish an ACK-required event, ack it, run check_missed_acks().
    Assert the returned alert list is empty."""


# Test: check_missed_acks detects event with expired expected_ack_by and NULL acked_at
def test_check_missed_acks_detects_expired():
    """Publish a RISK_WARNING event with expected_ack_by in the past (manually
    set or use a short timeout). Don't ack it. Run check_missed_acks().
    Assert at least one alert is returned."""


# Test: check_missed_acks escalation: 1 cycle overdue → retry (re-publish)
def test_check_missed_acks_retry_on_first_overdue():
    """Publish a RISK_WARNING, let it expire by 1 cycle (~300s past deadline).
    Run check_missed_acks(). Assert the event is re-published (new row in
    loop_events with same payload) and no dead-letter row is created."""


# Test: check_missed_acks escalation: 5 cycles overdue → dead letter + CRITICAL alert
def test_check_missed_acks_dead_letter_on_5_cycles():
    """Publish a RISK_WARNING with expected_ack_by far in the past (5+ cycles).
    Run check_missed_acks(). Assert a row exists in dead_letter_events and
    a CRITICAL system alert was created."""


# Test: check_missed_acks ignores events with requires_ack=NULL (migration safety)
def test_check_missed_acks_ignores_null_requires_ack():
    """Insert a loop_events row with requires_ack=NULL (simulating a pre-migration
    event). Run check_missed_acks(). Assert it is not flagged."""


# Test: check_missed_acks grace period after graph restart (no false positives)
def test_check_missed_acks_grace_period():
    """Publish a RISK_WARNING event very recently (within 2 cycle durations).
    Run check_missed_acks(). Even though expected_ack_by hasn't arrived yet,
    the event should NOT appear as missed."""


# Test: dead_letter_events row written with correct original_event_id and retry_count
def test_dead_letter_row_contents():
    """Trigger a dead-letter scenario. Query dead_letter_events and verify
    original_event_id matches the source event, retry_count reflects the
    number of re-publish attempts, and dead_lettered_at is set."""
```

### Integration Tests — `tests/integration/test_eventbus_ack_integration.py`

```python
"""Integration tests for EventBus ACK pattern (requires PostgreSQL)."""

import pytest


# Test: publish risk event → poll → ack → verify acked_at set in DB
def test_publish_poll_ack_roundtrip():
    """Full lifecycle: publish a RISK_ENTRY_HALT event, poll it as 'trading_graph',
    call bus.ack(), then query the DB row directly. Assert acked_at is non-null
    and acked_by == 'trading_graph'."""


# Test: publish risk event → don't ack → run check_missed_acks → verify alert created
def test_missed_ack_creates_alert():
    """Publish a RISK_EMERGENCY event with a short timeout override (e.g., 1s).
    Wait for expiry. Run check_missed_acks(). Query system_alerts table and
    assert a row with category='ack_timeout' exists."""


# Test: publish non-risk event → verify requires_ack=False, no ACK monitoring
def test_non_risk_event_not_monitored():
    """Publish a STRATEGY_PROMOTED event. Run check_missed_acks() after
    a delay. Assert no alert is created for this event."""
```

---

## Implementation Details

### 7.1 ACK Configuration Constants

Add to the top of `src/quantstack/coordination/event_bus.py`:

```python
ACK_REQUIRED_EVENTS: set[EventType] = {
    EventType.RISK_WARNING,
    EventType.RISK_ENTRY_HALT,
    EventType.RISK_LIQUIDATION,
    EventType.RISK_EMERGENCY,
    EventType.IC_DECAY,
    EventType.REGIME_CHANGE,
    EventType.MODEL_DEGRADATION,
}

ACK_TIMEOUT_SECONDS: int = 600
```

The timeout is a fixed 600 seconds for all ACK-required events. This is approximately 2x the supervisor cycle interval (300s) and 2x the trading graph cycle (300s), giving adequate buffer for one full cycle to process and ACK before the monitor flags it. The publisher does not know the consumer's cycle interval (they are different graphs), so a fixed generous timeout avoids coupling.

### 7.2 Modify `publish()` Method

Inside the existing `publish()` method, after the current INSERT, add logic to set the ACK columns:

- If `event.event_type in ACK_REQUIRED_EVENTS`: set `requires_ack=True` and `expected_ack_by = event.created_at + timedelta(seconds=ACK_TIMEOUT_SECONDS)`
- Otherwise: set `requires_ack=False`, leave `expected_ack_by` as NULL

The cleanest approach is to extend the existing INSERT statement to include the new columns rather than issuing a separate UPDATE. The INSERT becomes:

```python
self._conn.execute(
    """
    INSERT INTO loop_events
        (event_id, event_type, source_loop, payload, created_at,
         requires_ack, expected_ack_by)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    [
        event.event_id,
        etype_str,
        event.source_loop,
        payload_json,
        event.created_at,
        needs_ack,
        ack_deadline,  # None if not ACK-required
    ],
)
```

Where `needs_ack` and `ack_deadline` are computed before the INSERT:

```python
needs_ack = event.event_type in ACK_REQUIRED_EVENTS
ack_deadline = (
    event.created_at + timedelta(seconds=ACK_TIMEOUT_SECONDS)
    if needs_ack else None
)
```

### 7.3 New `ack()` Method

Add to the `EventBus` class:

```python
def ack(self, event_id: str, consumer_id: str) -> None:
    """Acknowledge receipt and processing of an event.

    Sets acked_at and acked_by on the event row.
    Idempotent: if already acked, this is a no-op (acked_at is not overwritten).
    """
```

Implementation notes:
- Use `UPDATE loop_events SET acked_at = ?, acked_by = ? WHERE event_id = ? AND acked_at IS NULL`
- The `AND acked_at IS NULL` clause makes re-acking a no-op without a separate SELECT
- Log at DEBUG level: `[EventBus] ACKed {event_id} by {consumer_id}`

### 7.4 `check_missed_acks()` Function

This function runs inside the supervisor graph's `health_check` node. It queries for events that required ACK but were not acknowledged within their deadline.

```python
async def check_missed_acks(conn: PgConnection) -> list:
    """Detect events that missed their ACK deadline and escalate.

    Query: requires_ack=TRUE AND acked_at IS NULL AND expected_ack_by < now()

    Escalation tiers based on how long overdue:
    - 1 cycle overdue (0-600s past deadline): re-publish the event (retry)
    - 3 cycles overdue (600-1500s past deadline): WARNING system alert
    - 5 cycles overdue (1500s+ past deadline): move to dead_letter_events + CRITICAL alert

    Returns list of system alerts created.
    """
```

Key implementation details:

**Escalation logic:** Compute `overdue_seconds = now - expected_ack_by`. Map to escalation tier:
- `overdue_seconds <= 600` (1 cycle): Re-publish the event with the same payload. The new event gets its own fresh `expected_ack_by`. This handles transient consumer downtime.
- `600 < overdue_seconds <= 1500` (3 cycles): Create a WARNING system alert with category `ack_timeout`. Include the event_id, event_type, and how long it's been overdue.
- `overdue_seconds > 1500` (5 cycles): Insert into `dead_letter_events` table with the original event's metadata and a `retry_count` (number of times re-published). Create a CRITICAL system alert. Mark the original event so it is not re-processed (set a flag or update `acked_by` to `'dead_lettered'`).

**Migration safety:** The query explicitly filters on `requires_ack = TRUE`. Pre-migration rows have `requires_ack = NULL`, which does not match `= TRUE` in PostgreSQL. No backfill is needed; the 7-day TTL will age out old rows.

**Grace period:** Do not flag events whose `expected_ack_by` has not yet passed. The query condition `expected_ack_by < now()` naturally handles this. Additionally, after a graph restart, events published in the last 2 cycles (600s) should not be flagged even if overdue — add a minimum age filter: `expected_ack_by < now() - interval '60 seconds'` to provide a small grace buffer.

**Dead letter table write:**
```python
conn.execute(
    """
    INSERT INTO dead_letter_events
        (original_event_id, event_type, source_loop, payload,
         published_at, expected_ack_by, retry_count, dead_lettered_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    [event_id, event_type, source_loop, payload,
     published_at, expected_ack_by, retry_count, now],
)
```

### 7.5 Supervisor Integration

In `src/quantstack/graphs/supervisor/nodes.py`, inside the `health_check` node function, add a call to `check_missed_acks()`. This runs every supervisor cycle (300s). The happy path (all events ACKed on time) returns an empty list and adds negligible overhead — it is a single indexed query.

```python
# In health_check node, after existing checks:
from quantstack.coordination.event_bus import check_missed_acks

missed_ack_alerts = await check_missed_acks(conn)
if missed_ack_alerts:
    logger.warning(f"[Supervisor] {len(missed_ack_alerts)} missed ACK alerts raised")
```

### 7.6 Consumer-Side ACK Pattern

In each graph runner's event processing loop, after processing a polled event, call `ack()` if the event requires it. The change is small and follows this pattern:

```python
events = bus.poll(consumer_id="trading_graph", event_types=risk_events)
for event in events:
    process_event(event)
    if getattr(event, "requires_ack", False):
        bus.ack(event.event_id, consumer_id="trading_graph")
```

**Important:** The `Event` dataclass is currently frozen and does not include `requires_ack`. There are two options:

1. **Extend the `Event` dataclass** to include `requires_ack: bool = False` and populate it during `poll()` from the DB row. This is the cleaner approach.
2. **Query the DB inside `poll()`** to read the `requires_ack` column and set it on the returned Event objects.

Option 1 is preferred. Add `requires_ack: bool = False` to the `Event` dataclass and update `poll()` to read the column from the SELECT and set it on the returned events.

The `poll()` method's SELECT statement needs to be extended:
```sql
SELECT event_id, event_type, source_loop, payload, created_at, requires_ack
FROM loop_events WHERE ...
```

And the Event construction in `poll()` updated:
```python
events.append(
    Event(
        event_id=eid,
        event_type=event_type,
        source_loop=source,
        payload=payload,
        created_at=created,
        requires_ack=bool(requires_ack_val),
    )
)
```

### 7.7 Modification to `Event` Dataclass

Add one field to the existing `Event` frozen dataclass:

```python
@dataclass(frozen=True)
class Event:
    event_type: EventType
    source_loop: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    requires_ack: bool = False  # NEW — populated by publish() and poll()
```

This field is informational on the consumer side. The source of truth is always the DB column.

---

## Design Rationale

**Why a fixed 600s timeout instead of per-event-type timeouts?** The publisher does not know which consumer will process the event or what that consumer's cycle interval is. A per-type timeout would couple the publisher to consumer internals. 600s (~2 cycles) is generous enough that any healthy consumer will process within one cycle, and the monitor won't false-positive during normal operations.

**Why re-publish on first missed ACK instead of immediately alerting?** Transient failures (consumer restart, brief network partition) are common. Re-publishing gives the consumer a second chance without creating alert noise. Only persistent failures (3+ cycles) warrant human-visible alerts.

**Why dead-letter after 5 cycles?** An event that has been re-published and still not ACKed after 5 cycles (2500s / ~42 minutes) indicates a systemic failure. Continuing to re-publish is pointless. Moving to dead letter preserves the event for forensic analysis while stopping the retry loop.

**Why filter on `requires_ack = TRUE` instead of `requires_ack IS NOT FALSE`?** Defensive migration safety. Pre-migration rows have NULL in this column. Using `= TRUE` excludes NULLs, preventing false positives on historical events without requiring a backfill migration.
