# Section 09: Continuous Runner Architecture

## Overview

This section implements the three continuous loop runners that drive the entire CrewAI system. Each crew (Trading, Research, Supervisor) gets a runner module in `src/quantstack/runners/` that runs as a Docker container entry point. The runner creates a fresh crew instance every cycle, calls `crew.kickoff()`, writes a heartbeat, and sleeps until the next cycle.

**Files to create:**

- `src/quantstack/runners/__init__.py`
- `src/quantstack/runners/trading_runner.py`
- `src/quantstack/runners/research_runner.py`
- `src/quantstack/runners/supervisor_runner.py`

**Dependencies (must be implemented first):**

- Section 05 (Crew Workflows) -- `TradingCrew`, `ResearchCrew`, `SupervisorCrew` classes with `crew.kickoff(inputs=...)` interface
- Section 07 (Self-Healing) -- `GracefulShutdown`, `AgentWatchdog`, `write_heartbeat` from `src/quantstack/health/`
- Section 08 (Observability) -- `CrewAIInstrumentor` setup pattern, `langfuse.flush()` on shutdown

---

## Tests First

Create `tests/unit/test_runners.py`. All tests below mock the crew classes, health utilities, and Langfuse to avoid external dependencies.

```python
# tests/unit/test_runners.py

"""
Tests for the continuous runner modules.

All crew classes, health utilities, and Langfuse are mocked.
No external services required.
"""

# --- Market Hours Detection ---

# Test: is_market_hours returns True during NYSE hours (Mon-Fri 9:30-16:00 ET)
#   Fixture: datetime(2026, 4, 1, 14, 0) (Wednesday 10:00 AM ET if UTC-4)
#   Assert: is_market_hours(dt) is True

# Test: is_market_hours returns False on weekends
#   Fixture: datetime(2026, 4, 4, 14, 0) (Saturday)
#   Assert: is_market_hours(dt) is False

# Test: is_market_hours returns False on NYSE holidays
#   Fixture: datetime(2026, 1, 1, 14, 0) (New Year's Day, Thursday)
#   Assert: is_market_hours(dt) is False

# Test: is_market_hours returns False before market open (9:29 AM ET)
#   Assert: is_market_hours(dt) is False

# Test: is_market_hours returns False after market close (4:01 PM ET)
#   Assert: is_market_hours(dt) is False


# --- Cycle Interval Selection ---

# Test: get_cycle_interval returns correct interval based on market hours
#   For trading runner: 300s (market hours), 1800s (after hours), paused (weekend)
#   For research runner: 600s (market hours), 1800s (after hours), 7200s (weekend)
#   For supervisor runner: 300s always


# --- Runner Loop Behavior ---

# Test: runner creates fresh crew instance each cycle (not reusing)
#   Mock: crew factory function, track call count
#   Run: 3 cycles (set should_stop after 3rd)
#   Assert: factory called 3 times, each returning a distinct object

# Test: runner respects should_stop flag (exits loop when True)
#   Set: should_stop = True before first iteration
#   Assert: loop body never executes, function returns cleanly

# Test: runner writes heartbeat after each successful cycle
#   Mock: write_heartbeat
#   Run: 2 cycles
#   Assert: write_heartbeat called 2 times

# Test: runner sleeps for correct interval based on market hours
#   Mock: time.sleep, is_market_hours -> True
#   Run: 1 cycle that completes in 10s
#   Assert: sleep called with (300 - 10) = 290 for trading runner

# Test: runner skips sleep if cycle took longer than interval
#   Mock: cycle takes 400s, interval is 300s
#   Assert: sleep called with 0 (or not called)

# Test: runner catches exceptions per cycle without crashing the loop
#   Mock: crew.kickoff raises RuntimeError on first call, succeeds on second
#   Run: 2 cycles
#   Assert: second cycle completes, heartbeat written once, loop did not exit

# Test: runner logs cycle duration and result to Langfuse
#   Mock: langfuse observe/trace
#   Run: 1 cycle
#   Assert: trace includes cycle_duration_seconds and cycle_result


# --- Shutdown and Cleanup ---

# Test: runner calls langfuse.flush() on shutdown
#   Mock: langfuse client
#   Trigger: set should_stop, let loop exit
#   Assert: flush() called exactly once

# Test: runner persists checkpoint to DB on shutdown
#   Mock: db_conn, checkpoint write function
#   Trigger: set should_stop
#   Assert: checkpoint written with last cycle timestamp and status
```

---

## Implementation Details

### Runner Pattern (shared across all three runners)

Each runner follows an identical structure. The difference is which crew factory it calls and what cycle intervals it uses.

**Entry point:** Each runner is invoked as `python -m quantstack.runners.trading_runner` (the `__main__` block).

**Initialization sequence (happens once at process start):**

1. Apply `nest_asyncio.apply()` -- required because CrewAI internals and the existing async tool implementations both need event loops.
2. Initialize Langfuse instrumentation via `CrewAIInstrumentor().instrument(skip_dep_check=True)`.
3. Register graceful shutdown handlers using `GracefulShutdown` from `src/quantstack/health/shutdown.py`. This sets a module-level `should_stop` flag on SIGTERM/SIGINT.
4. Create a `Langfuse` client instance for explicit trace/flush operations.

**Main loop (repeats until `should_stop` is True):**

1. Record cycle start time.
2. Call the crew factory function to create a fresh crew instance. Never reuse instances -- LLM client objects, tool caches, and context windows accumulate memory across cycles.
3. Build the `inputs` dict from current DB state (portfolio positions, regime, system status). Each runner builds different inputs appropriate to its crew.
4. Start the watchdog timer (`AgentWatchdog` from Section 07). Timeout: 600s for trading, 900s for research, 300s for supervisor. If the watchdog fires, it sets a flag that causes the current cycle result to be discarded.
5. Call `crew.kickoff(inputs=inputs)` inside a try/except. Catch all exceptions -- a single failed cycle must never crash the loop.
6. Cancel the watchdog timer (`watchdog.end_cycle()`).
7. Write heartbeat via `write_heartbeat(crew_name)`.
8. Save checkpoint to PostgreSQL (cycle timestamp, duration, success/failure, error message if any). Use `db_conn()` context manager.
9. Compute sleep duration: `max(0, interval - elapsed)`. If the cycle took longer than the interval, skip sleep entirely and start the next cycle immediately.
10. Sleep (interruptible -- check `should_stop` after waking).

**Shutdown sequence (after loop exits):**

1. Flush Langfuse traces (`langfuse_client.flush()`).
2. Write final checkpoint to DB with status `shutdown`.
3. Log clean exit.

### Market Hours Detection

Implement `is_market_hours(dt: datetime | None = None) -> bool` in `src/quantstack/runners/__init__.py` (or a shared utility within the runners package).

Logic:
- Convert to US/Eastern timezone.
- Check day of week: Monday=0 through Friday=4. Saturday/Sunday return False.
- Check time: 9:30 AM to 4:00 PM ET (inclusive of 9:30, exclusive of 16:00).
- Check against a hardcoded list of NYSE holidays for the current year. The list should cover standard closures (New Year's, MLK Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas). Early closes (e.g., day before Independence Day) are treated as full market hours for simplicity -- the runner just runs fewer useful cycles.

Note: There is an existing `is_market_hours` method in `src/quantstack/core/microstructure/events.py` (line 71). Evaluate whether it can be reused. If it requires instantiating a complex class or has unwanted dependencies, implement a standalone function in the runners package instead.

### Cycle Intervals

```python
# Interval configuration (seconds)
INTERVALS = {
    "trading":    {"market": 300,  "after_hours": 1800, "weekend": None},   # None = paused
    "research":   {"market": 600,  "after_hours": 1800, "weekend": 7200},
    "supervisor": {"market": 300,  "after_hours": 300,  "weekend": 300},
}
```

A `get_cycle_interval(crew_name: str) -> int | None` function returns the appropriate interval. If `None`, the runner sleeps in a polling loop (check every 60s if market hours have resumed) rather than running a cycle.

### trading_runner.py

**Crew factory:** Imports and instantiates `TradingCrew` from `src/quantstack/crews/trading/crew.py`.

**Inputs built each cycle:**
- `portfolio_state` -- current positions, equity, cash (from DB)
- `regime` -- current market regime (from DB or computed)
- `system_status` -- kill switch state, data freshness (from `get_system_status`)
- `daily_plan_exists` -- whether a daily plan has already been generated today (prevents re-planning mid-day)

**Special behavior:**
- Before building inputs, check system status. If kill switch is active, log and skip the cycle (do not create the crew at all).
- The trading runner pauses entirely on weekends (`interval=None`). It enters a polling sleep that checks `is_market_hours()` every 60 seconds.

### research_runner.py

**Crew factory:** Imports and instantiates `ResearchCrew` from `src/quantstack/crews/research/crew.py`.

**Inputs built each cycle:**
- `portfolio_gaps` -- domains/tickers lacking live strategies
- `recent_performance` -- P&L summary by strategy domain
- `pending_hypotheses` -- any queued research tasks from supervisor coordination events
- `symbol_override` -- reads `RESEARCH_SYMBOL_OVERRIDE` env var if set

**Special behavior:**
- Runs on weekends (at 2-hour intervals) since research does not require market data freshness.
- Checks for coordination events from the supervisor (e.g., "run community-intel scan") and adjusts inputs accordingly.

### supervisor_runner.py

**Crew factory:** Imports and instantiates `SupervisorCrew` from `src/quantstack/crews/supervisor/crew.py`.

**Inputs built each cycle:**
- `crew_heartbeats` -- last heartbeat timestamps for trading and research crews
- `service_health` -- reachability of Langfuse, Ollama, ChromaDB
- `strategies_in_forward_testing` -- list of strategies eligible for promotion review
- `scheduled_tasks_due` -- which scheduled tasks are due this cycle (checked against DB timestamps)

**Special behavior:**
- Runs at constant 5-minute intervals regardless of market hours.
- The supervisor is the only runner that never pauses -- it monitors system health 24/7.
- Uses `light` tier LLM (haiku) to minimize cost.

### Checkpoint Persistence

Each runner writes a checkpoint row to PostgreSQL after every cycle. Use the existing `loop_iteration_context` table pattern (or create it if it does not exist).

Schema for checkpoint:
- `crew_name` (text) -- "trading", "research", "supervisor"
- `cycle_timestamp` (timestamptz) -- when the cycle started
- `duration_seconds` (float) -- how long the cycle took
- `status` (text) -- "success", "error", "timeout", "shutdown"
- `error_message` (text, nullable) -- exception message if status is "error"
- `cycle_number` (integer) -- monotonically increasing counter per crew

This is queried by the supervisor for health monitoring and by `status.sh` for display.

### `__init__.py` Exports

The `src/quantstack/runners/__init__.py` file should export the shared utilities:

- `is_market_hours(dt=None) -> bool`
- `get_cycle_interval(crew_name: str) -> int | None`
- `INTERVALS` dict (for testing and configuration)

---

## Error Handling

Each runner wraps the `crew.kickoff()` call in a broad try/except. This is one of the rare cases where catching `Exception` (not bare `except:`) is appropriate -- the runner is a top-level boundary, and a single cycle failure must never kill the process.

On exception:
1. Log the full exception with traceback, cycle number, and crew name.
2. Record the error in the checkpoint table.
3. Do NOT write a heartbeat (a missing heartbeat signals to the supervisor that something went wrong).
4. Continue to the next cycle after sleeping.

If 3 consecutive cycles fail, the runner should log a critical warning. The supervisor's health monitor will detect the stale heartbeat and can take recovery action (e.g., triggering a container restart via Docker API or activating the kill switch).

---

## Key Design Decisions

**Fresh crew instances every cycle:** CrewAI crew objects accumulate state (conversation history, tool caches, LLM client connection pools). Reusing them across cycles leads to memory growth and stale context. The factory pattern ensures each cycle starts clean. The cost is ~1-2 seconds of object construction overhead per cycle, which is negligible compared to the 3-8 minute cycle duration.

**Stateless loop with DB checkpoints:** Mirrors the existing `start.sh` pattern where each Claude CLI invocation was independent. If a runner crashes and Docker restarts it, the new process reads the last checkpoint from DB and continues from a known state. No in-memory state survives across restarts.

**Watchdog as a safety net, not a primary control:** The watchdog timeout (600s for trading) is intentionally generous. It exists to catch truly stuck agents (infinite loops, deadlocked tools), not to enforce cycle timing. Normal cycles complete well within the timeout.
