# Section 06: EventBus Wiring & CI/CD

## Problem Statement

Four operational gaps prevent the system from running unattended:

1. **Trading Graph never polls EventBus.** The `safety_check` node in `src/quantstack/graphs/trading/nodes.py` (lines 214-245) makes an LLM call to decide if the system is halted, but never checks the EventBus for critical coordination events like `KILL_SWITCH_TRIGGERED`, `RISK_EMERGENCY`, or `IC_DECAY`. This means a kill switch triggered by the Supervisor or an external script takes an entire graph cycle (or longer) to propagate.

2. **Kill switch already publishes events (verified).** The `trigger()` method in `src/quantstack/execution/kill_switch.py` (lines 121-134) already publishes a `KILL_SWITCH_TRIGGERED` event to the EventBus via a best-effort try/except block. This was partially addressed in a prior commit. No changes needed here.

3. **Scheduler containerization is complete (verified).** The `scheduler` service in `docker-compose.yml` (lines 349-385) already has a health check endpoint on port 8422, `restart: unless-stopped`, and `stop_grace_period: 120s`. The health endpoint is implemented in `scripts/scheduler.py` (lines 60-103). No changes needed here.

4. **CI/CD pipelines already re-enabled (verified).** The `.disabled` suffix has been removed from both workflow files. `.github/workflows/ci.yml` and `.github/workflows/release.yml` exist and contain full pipeline definitions (lint, test, security scan, Docker build, release). No changes needed here.

**Remaining work:** The only unresolved item is wiring the EventBus poll into the Trading Graph's `safety_check` node.

## Dependencies

- None. This section has no dependencies on other sections and can be implemented in parallel with sections 01-04.

## Current State of Relevant Code

### EventBus (already complete)

`src/quantstack/coordination/event_bus.py` provides:
- `EventType` enum with `KILL_SWITCH_TRIGGERED`, `RISK_EMERGENCY`, `IC_DECAY` already defined (lines 77, 71, 73)
- `EventBus.poll(consumer_id, event_types)` method (lines 187-281) that reads events since the consumer's last cursor, filters by type, and updates the cursor atomically
- `EventBus.get_latest(event_type)` method (lines 283-307) for one-shot reads
- ACK-required events and missed-ACK escalation (lines 110-121, 343-473)

### Kill Switch (already complete)

`src/quantstack/execution/kill_switch.py` `trigger()` (lines 121-134) already publishes to EventBus:
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

### Trading Graph safety_check (needs modification)

`src/quantstack/graphs/trading/nodes.py` `make_safety_check()` (lines 214-245) currently:
- Makes an LLM call to check if the system is halted
- Returns `{"halted": True/False}` in the decisions list
- Does NOT check the EventBus for coordination events

### Scheduler (already complete)

`scripts/scheduler.py` already has a health endpoint on port 8422 (lines 60-103) returning JSON with status, uptime, job count. The `docker-compose.yml` scheduler service (lines 349-385) already has `restart: unless-stopped`, health check, and memory limits.

### CI/CD (already complete)

`.github/workflows/ci.yml` is live with lint, test, security, Docker build, and integration test jobs. `.github/workflows/release.yml` is live with validate, test, build, PyPI publish, GitHub release, and docs publish jobs.

## Tests (Write First)

### File: `tests/coordination/test_eventbus_wiring.py`

```python
# tests/coordination/test_eventbus_wiring.py

"""Tests for EventBus wiring into Trading Graph safety_check node."""

import pytest

# Test: kill_switch.trigger() publishes KILL_SWITCH_TRIGGERED event to EventBus
#   Setup: Create a KillSwitch instance and an in-memory or test DB EventBus.
#   Act: Call trigger(reason="test halt").
#   Assert: EventBus contains exactly one KILL_SWITCH_TRIGGERED event with
#           payload matching the trigger reason.
#   Note: kill_switch.py already has this code — this test verifies the contract
#         remains intact. Use a mock db_conn or a test database.

# Test: Trading Graph safety_check node polls EventBus for KILL_SWITCH_TRIGGERED
#   Setup: Publish a KILL_SWITCH_TRIGGERED event to the EventBus.
#   Act: Invoke the safety_check node function with a TradingState.
#   Assert: The returned dict has decisions[0]["halted"] == True and
#           the errors list contains a message referencing the kill switch event.

# Test: Trading Graph safety_check node polls EventBus for RISK_EMERGENCY
#   Setup: Publish a RISK_EMERGENCY event to the EventBus.
#   Act: Invoke the safety_check node function.
#   Assert: halted == True with reason referencing the risk emergency.

# Test: Trading Graph safety_check node polls EventBus for IC_DECAY
#   Setup: Publish an IC_DECAY event to the EventBus.
#   Act: Invoke the safety_check node function.
#   Assert: halted == True with reason referencing IC decay.

# Test: safety_check aborts cycle when KILL_SWITCH_TRIGGERED received
#   Setup: Publish KILL_SWITCH_TRIGGERED, then invoke safety_check.
#   Assert: The node returns halted=True. Downstream nodes (daily_plan, etc.)
#           should not execute when safety_check returns halted=True.
#   Note: Downstream gating is handled by the graph's conditional edge logic,
#         not by safety_check itself. This test only verifies safety_check's output.

# Test: safety_check passes through when no halt events exist
#   Setup: Empty EventBus (no events published).
#   Act: Invoke safety_check.
#   Assert: halted == False (or whatever the LLM returns — the EventBus check
#           should not force a halt when there are no halt events).

# Test: safety_check EventBus poll failure is non-blocking
#   Setup: Mock db_conn to raise an exception.
#   Act: Invoke safety_check.
#   Assert: The node still completes (falls through to the LLM-based check).
#           EventBus failure must not crash the Trading Graph.
```

### File: `tests/scripts/test_scheduler_health.py`

```python
# tests/scripts/test_scheduler_health.py

"""Tests for scheduler health endpoint (already implemented — regression tests)."""

import pytest

# Test: scheduler health endpoint returns 200 OK
#   Setup: Start the health server on a test port, set _scheduler_ref to a
#          mock scheduler with .running=True and .get_jobs() returning a list.
#   Act: HTTP GET /health
#   Assert: status code 200, body contains {"status": "running", ...}

# Test: scheduler health endpoint returns 503 when unhealthy
#   Setup: Set _scheduler_ref to a mock scheduler with .running=False.
#   Act: HTTP GET /health
#   Assert: status code 503, body contains {"status": "degraded"}
```

## Implementation

### The only file that needs modification: `src/quantstack/graphs/trading/nodes.py`

Add an EventBus poll at the beginning of the `safety_check` node function, before the LLM call. The poll checks for three halt-worthy event types. If any are found, the node returns `halted=True` immediately without burning an LLM call. If the EventBus poll fails (DB connection error, table missing, etc.), it logs a warning and falls through to the existing LLM-based check.

**Modification to `make_safety_check()`:**

Inside the `safety_check` inner function, add a block before the `llm.ainvoke()` call:

1. Import `db_conn` from `quantstack.db` and `EventBus`, `EventType` from `quantstack.coordination.event_bus`
2. Open a DB connection via `db_conn()` context manager
3. Call `bus.poll("trading_graph_safety", event_types=[EventType.KILL_SWITCH_TRIGGERED, EventType.RISK_EMERGENCY, EventType.IC_DECAY])`
4. If any events are returned, build a halt response immediately:
   - Set `halted = True`
   - Include the event types and payloads in the reason string
   - Return without making the LLM call (saves tokens + latency)
5. Wrap the entire EventBus block in try/except to ensure it is non-blocking — log a warning on failure and proceed to the LLM check

The consumer ID `"trading_graph_safety"` is specific to this node. Each poll advances the cursor, so events are only processed once. The cursor is persisted in the `loop_cursors` table, surviving container restarts.

**Halt event semantics:**

| Event Type | Source | Meaning |
|-----------|--------|---------|
| `KILL_SWITCH_TRIGGERED` | kill_switch.trigger(), Supervisor, external script | Emergency halt — all trading stops immediately |
| `RISK_EMERGENCY` | execution_monitor, risk_gate | Risk limit breached — halt new entries, may require position liquidation |
| `IC_DECAY` | signal IC tracker (Section 16) | Signal quality degraded — halt strategies relying on affected signals |

**Why poll instead of LISTEN/NOTIFY here:** The `safety_check` node runs at the start of every Trading Graph cycle (every ~60-300 seconds). Polling at cycle start is sufficient for the event types being checked — all three are halt conditions where 60-300 second latency is acceptable. Sub-second urgency delivery via PG LISTEN/NOTIFY is a separate concern addressed in Section 16.

### Files that need NO changes (verified current state)

- `src/quantstack/execution/kill_switch.py` — EventBus publish already implemented in `trigger()`
- `docker-compose.yml` — Scheduler service already has health check, restart policy, and resource limits
- `scripts/scheduler.py` — Health endpoint already implemented on port 8422
- `.github/workflows/ci.yml` — Already re-enabled (`.disabled` suffix removed)
- `.github/workflows/release.yml` — Already re-enabled

## Verification Checklist

After implementation:

1. Run `pytest tests/coordination/test_eventbus_wiring.py -v` — all tests pass
2. Run `pytest tests/scripts/test_scheduler_health.py -v` — all tests pass
3. Manually verify: trigger kill switch, then start a Trading Graph cycle. The safety_check node should detect the event and halt without making an LLM call.
4. Verify CI: push a branch and confirm `.github/workflows/ci.yml` triggers (lint, test, security jobs run)
5. Verify scheduler health: `curl http://localhost:8422/health` returns 200 with running status when the scheduler Docker container is up

## Rollback

If the EventBus poll causes issues in safety_check (e.g., DB connection overhead, unexpected events causing false halts):
- Remove the EventBus poll block from `make_safety_check()`. The LLM-based check remains as the fallback. The kill switch's sentinel file mechanism (`~/.quantstack/KILL_SWITCH_ACTIVE`) provides a secondary halt detection path that does not depend on the EventBus.
