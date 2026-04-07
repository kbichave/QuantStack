# Section 07: EventBus Integration

## Background

QuantStack's three LangGraph StateGraphs (trading, research, supervisor) run as independent Docker services. They coordinate via a PostgreSQL-based EventBus (`src/quantstack/coordination/event_bus.py`) that provides an append-only event log with per-consumer cursors. The supervisor publishes events, but the trading graph never polls them. The kill switch (`src/quantstack/execution/kill_switch.py`) writes a filesystem sentinel file on activation but never publishes to the EventBus.

This means a critical safety event (kill switch activation, risk emergency, regime change) can fire without the trading graph learning about it until its next full iteration rediscovers the state via DB scan or sentinel file check. For a system that commits real capital, this latency gap is unacceptable.

This section closes two gaps:
1. **Kill switch publishes `KILL_SWITCH_TRIGGERED` to EventBus** (best-effort, never blocks activation).
2. **Trading graph polls EventBus** at three critical points per cycle. All runners poll at cycle start.

## Dependencies

- No hard dependencies on other sections. The EventBus and kill switch already exist and are functional.
- This section is fully parallelizable with all other Batch 1 work.

## Current State of the Code

**EventType enum** (`src/quantstack/coordination/event_bus.py`, line 51): Already has `RISK_EMERGENCY`, `IC_DECAY`, `REGIME_CHANGE`, `RISK_ALERT`, and others. Does NOT have `KILL_SWITCH_TRIGGERED`.

**EventBus class** (`event_bus.py`, line 93): Accepts a `PgConnection`, exposes `publish(event)`, `poll(consumer_id, event_types)`, `get_latest(event_type)`, and `count_events()`. The `poll()` method advances a per-consumer cursor in `loop_cursors` after reading, so events are consumed idempotently.

**KillSwitch.trigger()** (`kill_switch.py`, line 98): Acquires a lock, writes a `KillSwitchStatus` to a sentinel file, logs CRITICAL, then best-effort calls a registered position closer. No EventBus interaction.

**Trading graph** (`src/quantstack/graphs/trading/graph.py`): 16-node StateGraph. The `safety_check` node runs after `data_refresh` at cycle start. `execute_exits` runs in a parallel branch after `plan_day`. `execute_entries` runs after `merge_pre_execution`, near the end of the pipeline.

**Runners** (`src/quantstack/runners/`): `trading_runner.py`, `research_runner.py`, `supervisor_runner.py`. Each has an async `run_loop()` that rebuilds the graph each cycle, builds initial state, and invokes the graph with a thread ID. None poll the EventBus.

---

## Tests (Write These First)

### Kill switch EventBus publication

```python
# File: tests/coordination/test_eventbus_kill_switch.py

# Test: KILL_SWITCH_TRIGGERED is a valid member of EventType enum
# Test: kill_switch.trigger() publishes KILL_SWITCH_TRIGGERED event to EventBus
# Test: event payload contains "reason" (str) and "triggered_at" (ISO 8601 str)
# Test: EventBus publication failure does NOT prevent kill switch activation
#       (sentinel file still written, CRITICAL still logged, position closer still called)
# Test: EventBus publication failure is logged as a warning
```

### Trading graph EventBus polling

```python
# File: tests/graphs/test_trading_eventbus_polling.py

# Test: safety_check node polls for KILL_SWITCH_TRIGGERED, RISK_EMERGENCY, IC_DECAY, REGIME_CHANGE
# Test: KILL_SWITCH_TRIGGERED received at safety_check -> cycle halted (state["decisions"] contains halted=True)
# Test: RISK_EMERGENCY received at safety_check -> cycle halted
# Test: IC_DECAY received at safety_check -> affected strategy marked as suspended in state
# Test: REGIME_CHANGE received at safety_check -> regime info updated in state
# Test: pre-execute_entries polls for KILL_SWITCH_TRIGGERED, RISK_EMERGENCY only
# Test: KILL_SWITCH_TRIGGERED before execute_entries -> entries skipped, flow proceeds to reflect
# Test: RISK_EMERGENCY before execute_entries -> entries skipped, flow proceeds to reflect
# Test: pre-execute_exits polls for KILL_SWITCH_TRIGGERED only
# Test: KILL_SWITCH_TRIGGERED before execute_exits -> emergency close-all (not normal exit logic)
# Test: consumer cursor advances after polling (subsequent poll returns no duplicate events)
```

### All-runner polling

```python
# File: tests/runners/test_runner_eventbus_polling.py

# Test: trading_runner polls KILL_SWITCH_TRIGGERED at cycle start, halts loop if present
# Test: research_runner polls KILL_SWITCH_TRIGGERED at cycle start, halts loop if present
# Test: supervisor_runner polls KILL_SWITCH_TRIGGERED at cycle start, halts loop if present
# Test: runner poll failure (DB down) does not crash the runner loop (best-effort, logged)
```

### End-to-end propagation

```python
# File: tests/integration/test_kill_switch_propagation.py

# Test: trigger kill switch -> sentinel file written + KILL_SWITCH_TRIGGERED published to EventBus ->
#       all 3 runners detect event within one simulated cycle -> trading graph halts ->
#       execution monitor stops -> position closer fires
```

---

## Implementation

### Step 1: Add `KILL_SWITCH_TRIGGERED` to EventType

**File:** `src/quantstack/coordination/event_bus.py`

Add a new member to the `EventType` enum:

```python
class EventType(str, Enum):
    # ... existing members ...
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
```

Place it logically near the other risk/emergency event types.

### Step 2: Kill switch publishes to EventBus on trigger

**File:** `src/quantstack/execution/kill_switch.py`

Modify `KillSwitch.trigger()` to publish a `KILL_SWITCH_TRIGGERED` event after writing the sentinel file. The publication is best-effort: wrapped in `try/except` so that a DB failure (or any other exception) never delays or prevents kill switch activation.

The EventBus requires a `PgConnection`. Since the kill switch is a singleton and should not hold a persistent DB connection, acquire a connection via `db_conn()` context manager at publish time. If `db_conn()` itself fails (e.g., PostgreSQL is down), the except block catches it and logs a warning.

Pseudocode for the addition inside `trigger()`, after the `_write_sentinel()` call and before the position closer block:

```python
# Best-effort: notify other graphs via EventBus
try:
    from quantstack.db import db_conn
    from quantstack.coordination.event_bus import EventBus, Event, EventType

    with db_conn() as conn:
        bus = EventBus(conn)
        bus.publish(Event(
            event_type=EventType.KILL_SWITCH_TRIGGERED,
            source_loop="kill_switch",
            payload={"reason": reason, "triggered_at": self._status.triggered_at.isoformat()},
        ))
except Exception as exc:
    logger.warning(f"[KILL SWITCH] EventBus publication failed (non-blocking): {exc}")
```

The import is inline here intentionally to avoid a circular dependency (kill_switch is imported early in the boot sequence). This is one of the rare legitimate cases for a deferred import -- the kill switch module must be importable without the full DB stack being initialized.

### Step 3: Trading graph polls EventBus at three points

**File:** `src/quantstack/graphs/trading/nodes.py`

Create a shared helper function for EventBus polling within the trading graph:

```python
def _poll_eventbus(consumer_id: str, event_types: list[EventType]) -> list[Event]:
    """Poll EventBus for specified event types. Best-effort: returns [] on failure."""
```

This helper acquires a `db_conn()`, creates an `EventBus`, calls `poll()`, and returns events. On any exception, it logs a warning and returns an empty list -- polling failure must never crash a graph node.

**Poll point 1: `safety_check` node.** The `make_safety_check()` factory returns a node function. At the beginning of that function (before or after the existing safety logic), poll for `[KILL_SWITCH_TRIGGERED, RISK_EMERGENCY, IC_DECAY, REGIME_CHANGE]` using consumer ID `"trading-graph"`.

Action on events:
- `KILL_SWITCH_TRIGGERED` or `RISK_EMERGENCY`: Set `halted=True` in the node's decision output. The existing `_safety_check_router` in `graph.py` already routes to END when `halted=True`.
- `IC_DECAY`: Extract the affected strategy from the event payload and add it to a `suspended_strategies` list in state. Downstream nodes (entry_scan, risk_sizing) should skip suspended strategies.
- `REGIME_CHANGE`: Update `regime_context` in state so downstream nodes factor in the new regime.

**Poll point 2: Before `execute_entries`.** This requires a new gate node or a modification to the existing `execute_entries` node. The simplest approach: add a pre-check at the top of the `execute_entries` node function. Poll for `[KILL_SWITCH_TRIGGERED, RISK_EMERGENCY]` only. If either is found, skip entry execution entirely (return state unchanged, or set a flag that downstream `reflect` can report on). Do not add a separate graph node for this -- it would change the graph topology. A check inside the existing node is sufficient and keeps the graph shape stable.

**Poll point 3: Before `execute_exits`.** Same pattern as poll point 2, at the top of the `execute_exits` node function. Poll for `[KILL_SWITCH_TRIGGERED]` only. If found, escalate: instead of normal exit evaluation, trigger an emergency close-all by invoking the position closer or marking all positions for immediate exit.

**Consumer ID and cursor management.** All three poll points use the same consumer ID `"trading-graph"`. Since they run sequentially within a single cycle (safety_check first, then execute_exits and execute_entries later), the cursor advances correctly. Events published between safety_check and execute_entries will be caught by the execute_entries poll.

### Step 4: All runners poll at cycle start

**Files:**
- `src/quantstack/runners/trading_runner.py`
- `src/quantstack/runners/research_runner.py`
- `src/quantstack/runners/supervisor_runner.py`

In each runner's `run_loop()` function, at the top of each cycle iteration (after the interval/pause check, before building initial state and invoking the graph), add an EventBus poll for `KILL_SWITCH_TRIGGERED`.

Each runner uses its own consumer ID:
- Trading: `"trading-runner"` (distinct from `"trading-graph"` used inside graph nodes)
- Research: `"research-runner"`
- Supervisor: `"supervisor-runner"`

If a `KILL_SWITCH_TRIGGERED` event is found, log CRITICAL and break out of the loop (or set `shutdown.should_stop = True`). This provides a second layer of kill switch detection beyond the sentinel file -- useful if the sentinel file is on a volume that's slow to propagate across containers.

Wrap the poll in `try/except` so a DB failure does not crash the runner loop. The runner should continue to its next cycle (where it will also check the sentinel file via the existing `is_active()` path).

Example pseudocode for the addition in `run_loop()`:

```python
# Poll EventBus for kill switch (best-effort, non-blocking)
try:
    from quantstack.db import db_conn
    from quantstack.coordination.event_bus import EventBus, EventType

    with db_conn() as conn:
        bus = EventBus(conn)
        events = bus.poll(f"{graph_name}-runner", event_types=[EventType.KILL_SWITCH_TRIGGERED])
        if events:
            logger.critical("[%s] Kill switch event received via EventBus — halting", graph_name)
            break
except Exception:
    logger.warning("[%s] EventBus poll failed at cycle start (non-blocking)", graph_name, exc_info=True)
```

### Step 5: Verify consumer cursor behavior

The EventBus `poll()` method already advances the cursor to the last returned event. This means:
- If safety_check polls and sees a `REGIME_CHANGE`, that event is consumed. The execute_entries poll later in the same cycle will not see it again.
- If a `KILL_SWITCH_TRIGGERED` is published between safety_check and execute_entries, the execute_entries poll will pick it up (the cursor was advanced past safety_check's events, but the new event is after the cursor).
- Each runner's consumer ID is independent from the graph's consumer ID, so the runner poll and the graph node poll are independent consumers that each see all events.

This is the correct behavior. No changes to `EventBus.poll()` are needed.

---

## Latency Analysis

- **Trading cycle:** ~5 minutes. With three poll points (safety_check at cycle start, execute_entries ~3 min in, execute_exits ~2 min in via parallel branch), worst-case latency for a kill switch event to be acted on is ~2-3 minutes.
- **Research cycle:** ~60 seconds. Runner-level poll at cycle start. Max latency: 60 seconds.
- **Supervisor cycle:** Already publishes events; runner-level poll at start. Max latency: one supervisor cycle (~120 seconds).

For a swing-trading system with 5-minute cycles, these latencies are acceptable. The sentinel file check (already in place) provides a faster detection path for same-process kill switch activations.

---

## Key Invariants

1. **EventBus publication never blocks kill switch activation.** The sentinel file write and CRITICAL log happen first; EventBus publication is wrapped in try/except and logged as warning on failure.
2. **Polling failure never crashes a graph node or runner.** All poll calls are wrapped in try/except and return empty results on failure.
3. **Consumer cursors advance idempotently.** Re-polling returns only new events, not duplicates.
4. **The sentinel file remains the primary kill switch mechanism.** EventBus publication is supplementary notification for cross-graph coordination.

---

## Files to Create or Modify

| File | Action | Change |
|------|--------|--------|
| `src/quantstack/coordination/event_bus.py` | Modify | Add `KILL_SWITCH_TRIGGERED` to `EventType` enum |
| `src/quantstack/execution/kill_switch.py` | Modify | Add best-effort EventBus publish in `trigger()` |
| `src/quantstack/graphs/trading/nodes.py` | Modify | Add EventBus polling helper; add polls in safety_check, execute_entries, execute_exits nodes |
| `src/quantstack/runners/trading_runner.py` | Modify | Add EventBus poll at cycle start |
| `src/quantstack/runners/research_runner.py` | Modify | Add EventBus poll at cycle start |
| `src/quantstack/runners/supervisor_runner.py` | Modify | Add EventBus poll at cycle start |
| `tests/coordination/test_eventbus_kill_switch.py` | Create | Tests for kill switch -> EventBus publication |
| `tests/graphs/test_trading_eventbus_polling.py` | Create | Tests for trading graph poll points |
| `tests/runners/test_runner_eventbus_polling.py` | Create | Tests for runner-level polling |
| `tests/integration/test_kill_switch_propagation.py` | Create | End-to-end propagation test |
