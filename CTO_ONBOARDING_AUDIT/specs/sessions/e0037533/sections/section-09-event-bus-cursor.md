# Section 09: Event Bus Cursor Atomicity

## Problem

The event bus (`src/quantstack/coordination/event_bus.py`) updates per-consumer cursors via a DELETE followed by an INSERT. If the process crashes between the DELETE and the INSERT, the cursor is lost. On next poll, the consumer replays events it already processed because its high-water mark disappeared.

The two locations in `EventBus.poll()` that exhibit this pattern (lines 232-240 and 243-251):

```python
# When events were returned:
self._conn.execute("DELETE FROM loop_cursors WHERE consumer_id = ?", [consumer_id])
self._conn.execute("INSERT INTO loop_cursors ... VALUES (?, ?, ?)", [consumer_id, new_cursor, now])

# When no events (heartbeat update):
self._conn.execute("DELETE FROM loop_cursors WHERE consumer_id = ?", [consumer_id])
self._conn.execute("INSERT INTO loop_cursors ... VALUES (?, NULL, ?)", [consumer_id, now])
```

Both paths have the same crash window between DELETE and INSERT.

## Dependencies

- **None.** This section has no dependencies on other sections and can be implemented in Batch 1 (parallel with section-01).

## Background

- The `loop_cursors` table is defined in `src/quantstack/db.py` with `consumer_id TEXT PRIMARY KEY`. The PRIMARY KEY constraint already implies UNIQUE, so no migration is needed for the upsert's `ON CONFLICT` clause.
- The `PgConnection` wrapper fully supports `ON CONFLICT` syntax. The codebase uses `ON CONFLICT` in 25+ locations across `equity_tracker.py`, `benchmark.py`, `risk_state.py`, `order_lifecycle.py`, and many others. All use `?`-style placeholders.
- The codebase convention is `?` placeholders (not `$1` libpq style). The upsert must follow this convention.

## Fix

Replace both DELETE+INSERT pairs in `EventBus.poll()` with a single PostgreSQL upsert statement:

```sql
INSERT INTO loop_cursors (consumer_id, last_event_id, last_polled_at)
VALUES (?, ?, ?)
ON CONFLICT (consumer_id) DO UPDATE SET
  last_event_id = EXCLUDED.last_event_id,
  last_polled_at = EXCLUDED.last_polled_at
```

This is atomic -- either the cursor updates or it does not. There is no intermediate state where the cursor row is absent.

### Changes Required

**File: `src/quantstack/coordination/event_bus.py`**

In the `poll()` method, replace the two cursor-update blocks (the `if events:` and `else:` branches at the end of the method) with a single upsert call. The logic becomes:

1. Determine `new_cursor`: if events were returned, use `events[-1].event_id`; if no events, use `None`.
2. Execute the upsert with `(consumer_id, new_cursor, now)`.

The two branches collapse into one because the only difference between them is whether `last_event_id` is a real event ID or `NULL`. The upsert handles both cases identically.

### What NOT to Change

- The `loop_cursors` table schema -- `consumer_id` is already PRIMARY KEY, which satisfies `ON CONFLICT (consumer_id)`.
- No migration needed for this section.
- The poll query logic, event reconstruction, and pruning logic are unchanged.

## Tests First

Place tests in `tests/unit/test_event_bus_cursor.py`.

```python
# Test: upsert_replaces_delete_insert
# Verify that updating a cursor for an existing consumer_id executes a single
# SQL statement containing ON CONFLICT, not a DELETE followed by INSERT.
# Approach: mock PgConnection.execute, call poll() with events present,
# assert execute was called with SQL containing "ON CONFLICT" for the cursor
# update, and that no "DELETE FROM loop_cursors" call was made.

# Test: new_consumer_creates_cursor
# Call poll() with a consumer_id that has no existing cursor row.
# Verify the upsert INSERT path creates the row (the ON CONFLICT clause
# does nothing because there is no conflict -- it just inserts).

# Test: existing_consumer_updates_cursor
# Seed a cursor row for consumer_id "test_consumer".
# Call poll() which returns events.
# Verify the cursor row is updated to the last event's ID (the ON CONFLICT
# UPDATE path).

# Test: poll_with_no_events_updates_polled_at
# Call poll() when no events exist.
# Verify the cursor row has last_event_id = NULL and last_polled_at updated
# to approximately now (within 1 second tolerance).

# Test: concurrent_cursor_updates_no_lost_cursors
# Two consumers poll concurrently. Verify both cursor rows exist after
# both polls complete. Neither consumer's cursor is lost.
# (This is inherently safe with per-consumer upserts since they operate
# on different PRIMARY KEY values, but the test confirms it.)

# Test: pgconnection_supports_on_conflict
# Execute a raw ON CONFLICT statement through PgConnection against
# loop_cursors. Verify no exception is raised. This is a smoke test
# confirming the DB abstraction layer passes through the syntax.
```

### Test Strategy Notes

- Use the same mocking patterns as existing tests: `MagicMock` for `PgConnection` when testing SQL generation, or a real test database connection when testing actual cursor persistence.
- The "single SQL statement" test is the critical one -- it proves the DELETE+INSERT anti-pattern is gone.
- The concurrency test can use threading with a shared test database to confirm no row-level conflicts between different consumer IDs.

## Implementation Checklist

1. Write the 6 tests in `tests/unit/test_event_bus_cursor.py` (all should fail initially).
2. In `src/quantstack/coordination/event_bus.py`, replace lines 228-251 (the cursor update section of `poll()`) with the single upsert.
3. Run tests -- all 6 should pass.
4. Run the full test suite (`uv run pytest`) to confirm no regressions.

## Failure Modes

- **PgConnection silently rewrites SQL**: Some ORM layers transform queries. The ON CONFLICT smoke test catches this. The codebase already uses ON CONFLICT in 25+ places, so this is low risk.
- **Placeholder mismatch**: If `?` placeholders are not supported for the `EXCLUDED` references, the query will fail at runtime. The test suite catches this before deployment.
- **Transaction boundaries**: If `PgConnection.execute()` auto-commits each statement, the upsert is already atomic as a single statement. If it batches within a transaction, the upsert is still atomic. Either way, the single-statement approach eliminates the crash window.
