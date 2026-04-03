# Section 11: Runner Refactor (Sync to Async + Graph Integration)

## Overview

All three runners (`trading_runner.py`, `research_runner.py`, `supervisor_runner.py`) currently follow a synchronous pattern: `crew_factory()` returns a CrewAI crew, `crew.kickoff()` runs it synchronously, `time.sleep()` waits between cycles. This section replaces that with an async pattern: `asyncio.run()` entry point, `graph.ainvoke()` for execution, `asyncio.wait_for()` for timeout enforcement, and `loop.add_signal_handler()` for graceful shutdown in async context.

The shared `run_loop()` function in `trading_runner.py` is the core of this refactor. The research and supervisor runners delegate to it, so fixing `run_loop()` fixes all three.

## Dependencies

- **Section 06 (Supervisor Graph)**: `build_supervisor_graph()` must exist and return a `CompiledStateGraph`
- **Section 07 (Research Graph)**: `build_research_graph()` must exist and return a `CompiledStateGraph`
- **Section 08 (Trading Graph)**: `build_trading_graph()` must exist and return a `CompiledStateGraph`
- **Section 10 (Observability)**: `get_langfuse_handler()` must be available for per-cycle trace callback creation
- **Section 03 (Agent Config)**: `ConfigWatcher` must be available for hot-reload support
- **Section 04 (State Schemas)**: `ResearchState`, `TradingState`, `SupervisorState` TypedDicts must be defined

## Files to Modify

- `src/quantstack/runners/__init__.py` -- unchanged (market hours and interval logic is framework-agnostic)
- `src/quantstack/runners/trading_runner.py` -- full rewrite
- `src/quantstack/runners/research_runner.py` -- full rewrite
- `src/quantstack/runners/supervisor_runner.py` -- full rewrite
- `src/quantstack/health/shutdown.py` -- add async-compatible signal registration method

## Tests First

All tests go in `tests/unit/test_runners.py`. Write these before implementation.

```python
"""Tests for the async runner refactor."""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRunnerEntryPoints:
    """Verify each runner's main() uses asyncio.run()."""

    def test_trading_runner_main_calls_asyncio_run(self):
        """trading_runner.main() must call asyncio.run() to enter the async loop."""
        ...

    def test_research_runner_main_calls_asyncio_run(self):
        """research_runner.main() must call asyncio.run() to enter the async loop."""
        ...

    def test_supervisor_runner_main_calls_asyncio_run(self):
        """supervisor_runner.main() must call asyncio.run() to enter the async loop."""
        ...


class TestRunLoop:
    """Verify the shared async run_loop() behavior."""

    def test_run_loop_is_coroutine(self):
        """run_loop() must be an async function (coroutine)."""
        from quantstack.runners.trading_runner import run_loop
        assert inspect.iscoroutinefunction(run_loop)

    @pytest.mark.asyncio
    async def test_run_loop_uses_wait_for_timeout(self):
        """run_loop() must use asyncio.wait_for() to enforce cycle timeout,
        not a thread-based AgentWatchdog."""
        ...

    @pytest.mark.asyncio
    async def test_run_loop_passes_langfuse_callback_in_config(self):
        """The graph.ainvoke() call must include a LangFuse callback handler
        in its config dict under the 'callbacks' key."""
        ...

    @pytest.mark.asyncio
    async def test_run_loop_passes_thread_id_in_config(self):
        """The graph.ainvoke() call must include a thread_id in config
        for checkpoint persistence (format: '{graph_name}-{date}-cycle-{n}')."""
        ...

    @pytest.mark.asyncio
    async def test_run_loop_rebuilds_graph_each_cycle(self):
        """The graph must be rebuilt from the graph_builder callable each cycle,
        not cached, so hot-reload config changes take effect."""
        ...

    @pytest.mark.asyncio
    async def test_run_loop_stops_on_shutdown_flag(self):
        """When shutdown.should_stop is True, run_loop() exits cleanly."""
        ...

    @pytest.mark.asyncio
    async def test_run_loop_consecutive_failure_tracking(self):
        """After 3 consecutive failures, a critical log is emitted."""
        ...


class TestGracefulShutdownAsync:
    """Verify GracefulShutdown works in async context."""

    @pytest.mark.asyncio
    async def test_signal_handler_sets_stop_flag(self):
        """Signal handler must set should_stop=True in async context."""
        ...

    @pytest.mark.asyncio
    async def test_cleanup_callbacks_run_on_shutdown(self):
        """Registered cleanup callbacks must execute during shutdown."""
        ...


class TestCycleIntervals:
    """Verify interval logic is unchanged after migration."""

    def test_trading_market_hours_300s(self):
        """Trading interval during market hours: 300s (5 min)."""
        from quantstack.runners import get_cycle_interval
        from datetime import datetime
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 4, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        assert get_cycle_interval("trading", dt) == 300

    def test_research_market_hours_600s(self):
        """Research interval during market hours: 600s (10 min)."""
        from quantstack.runners import get_cycle_interval
        from datetime import datetime
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 4, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        assert get_cycle_interval("research", dt) == 600

    def test_supervisor_always_300s(self):
        """Supervisor interval: 300s regardless of market hours."""
        from quantstack.runners import get_cycle_interval
        from datetime import datetime
        from zoneinfo import ZoneInfo
        dt_market = datetime(2026, 4, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        dt_weekend = datetime(2026, 4, 4, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        assert get_cycle_interval("supervisor", dt_market) == 300
        assert get_cycle_interval("supervisor", dt_weekend) == 300
```

## Implementation Details

### 1. The Shared `run_loop()` Function (trading_runner.py)

The current `run_loop()` accepts a `crew_factory: Callable` and calls `crew.kickoff()`. The new version accepts a `graph_builder: Callable` and calls `await graph.ainvoke()`.

**Current signature:**
```python
def run_loop(crew_factory: Callable, shutdown, crew_name: str = "trading") -> None:
```

**New signature:**
```python
async def run_loop(
    graph_builder: Callable[[], CompiledStateGraph],
    initial_state_builder: Callable[[], dict],
    shutdown: GracefulShutdown,
    graph_name: str = "trading",
    watchdog_timeout: int = 600,
) -> None:
```

Key changes inside the loop body:

- **Graph rebuild each cycle**: Call `graph = graph_builder()` at the top of every iteration. This ensures config hot-reload takes effect without caching stale graphs. Graph construction is cheap (topology definition + compile) -- the cost is in `ainvoke()`, not in building.

- **Initial state construction**: Call `initial_state = initial_state_builder()` each cycle. This function queries current market conditions, regime, portfolio state, and returns a typed dict matching the graph's state schema. Each runner provides its own builder.

- **Timeout via `asyncio.wait_for()`**: Replace the thread-based `AgentWatchdog` with:
  ```python
  try:
      final_state = await asyncio.wait_for(
          graph.ainvoke(initial_state, config=invocation_config),
          timeout=watchdog_timeout,
      )
  except asyncio.TimeoutError:
      logger.critical("%s cycle %d timed out after %ds", graph_name, cycle_number, watchdog_timeout)
      status = "timeout"
  ```
  This is simpler and more reliable than a separate watchdog thread in async code. The `AgentWatchdog` class can be retired (or kept as a no-op wrapper for backward compat).

- **LangFuse callback in config**: Each cycle creates a fresh callback handler and passes it in the invocation config:
  ```python
  from quantstack.observability.instrumentation import get_langfuse_handler

  handler = get_langfuse_handler(
      session_id=f"{graph_name}-{date_str}",
      tags=[graph_name, f"cycle-{cycle_number}"],
  )
  invocation_config = {
      "callbacks": [handler],
      "configurable": {"thread_id": f"{graph_name}-{date_str}-cycle-{cycle_number}"},
  }
  ```

- **Thread ID for checkpointing**: The `thread_id` in `configurable` tells `AsyncPostgresSaver` how to namespace the checkpoint. Format: `{graph_name}-{YYYY-MM-DD}-cycle-{N}`. This replaces the manual `save_checkpoint()` SQL call -- LangGraph's checkpointer handles persistence automatically after every node.

- **Legacy checkpoint write**: Keep the `save_checkpoint()` call for backward compatibility during the transition period. It writes to the existing `crew_checkpoints` table so the supervisor's health queries still work. Remove it once the supervisor is also migrated and queries LangGraph checkpoint tables instead.

- **Sleep**: Replace `time.sleep(interval)` with `await asyncio.sleep(interval)`.

- **Heartbeat**: `write_heartbeat()` is synchronous DB I/O. Wrap in `await asyncio.to_thread(write_heartbeat, graph_name)` to avoid blocking the event loop, or convert `write_heartbeat` to async separately.

- **Error inspection**: After `ainvoke()` completes, inspect `final_state.get("errors", [])` to determine if the cycle had issues. Log any errors to LangFuse via the handler.

### 2. trading_runner.py -- Full Structure

```python
"""Trading graph continuous runner -- async cycle every 5 min (market hours)."""

import asyncio
import logging
from datetime import datetime

from quantstack.health.heartbeat import write_heartbeat
from quantstack.health.shutdown import GracefulShutdown
from quantstack.runners import get_cycle_interval

logger = logging.getLogger(__name__)

WATCHDOG_TIMEOUT = 600  # seconds


async def run_loop(
    graph_builder,
    initial_state_builder,
    shutdown,
    graph_name="trading",
    watchdog_timeout=WATCHDOG_TIMEOUT,
):
    """Async runner loop. Rebuilds graph each cycle, invokes with timeout."""
    ...  # Implementation as described above


async def async_main():
    """Async entry point for the trading runner."""
    ...  # Setup instrumentation, ConfigWatcher, checkpointer, graph_builder, initial_state_builder
    ...  # Call await run_loop(...)


def main():
    """Entry point: python -m quantstack.runners.trading_runner"""
    asyncio.run(async_main())
```

The `async_main()` function:

1. Calls `setup_instrumentation()` (remains synchronous -- it's one-time init)
2. Creates a `ConfigWatcher` for `graphs/trading/config/agents.yaml`
3. Creates an `AsyncPostgresSaver` from the DB connection string
4. Defines `graph_builder` as a closure that calls `build_trading_graph(config_watcher, checkpointer)`
5. Defines `initial_state_builder` that returns a `TradingState`-compatible dict with current cycle number, regime, and portfolio context
6. Creates `GracefulShutdown` and installs signal handlers using `loop.add_signal_handler()`
7. Calls `await run_loop(...)`

### 3. research_runner.py -- Full Structure

Same pattern as trading but with:
- `WATCHDOG_TIMEOUT = 900` (research cycles are longer)
- `graph_name = "research"`
- `graph_builder` calls `build_research_graph(config_watcher, checkpointer)`
- `initial_state_builder` returns a `ResearchState`-compatible dict

### 4. supervisor_runner.py -- Full Structure

Same pattern but with:
- `WATCHDOG_TIMEOUT = 300` (supervisor cycles are short)
- `graph_name = "supervisor"`
- `graph_builder` calls `build_supervisor_graph(config_watcher, checkpointer)`
- `initial_state_builder` returns a `SupervisorState`-compatible dict

### 5. GracefulShutdown Async Compatibility

The existing `GracefulShutdown` class uses `signal.signal()` which works but has caveats in async code (signal handlers run in the main thread and can't directly interact with the event loop safely). Two options:

**Option A (recommended)**: Add an `install_async()` method to `GracefulShutdown` that uses `loop.add_signal_handler()`:

```python
def install_async(self, loop: asyncio.AbstractEventLoop) -> None:
    """Register signal handlers using asyncio's event loop (preferred in async context)."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, self._handle_async_signal, sig)

def _handle_async_signal(self, signum: int) -> None:
    """Handle signal in async context (no frame argument)."""
    sig_name = signal.Signals(signum).name
    logger.info("Received %s -- initiating graceful shutdown", sig_name)
    self._should_stop = True
    self._run_cleanup()
```

The runners call `shutdown.install_async(asyncio.get_running_loop())` inside `async_main()` instead of `shutdown.install()`.

**Option B**: Keep `signal.signal()` and accept that cleanup runs in the signal handler context. This works for the simple case (setting a boolean flag) but is fragile if cleanup callbacks do async work.

Go with Option A. It is the correct pattern for asyncio programs.

### 6. Checkpoint and State Persistence

The current `save_checkpoint()` function writes to `crew_checkpoints` table with manual SQL. After migration:

- **Primary persistence**: `AsyncPostgresSaver` (LangGraph's built-in checkpointer) automatically saves state after every node execution. The `thread_id` in config namespaces each cycle's checkpoint.
- **Legacy persistence**: Keep `save_checkpoint()` writing to `crew_checkpoints` during transition. It records cycle-level metadata (duration, status, error message) that the LangGraph checkpointer doesn't track. This is useful for the supervisor's health monitoring queries.
- **Future cleanup**: Once the supervisor graph also reads from LangGraph checkpoint tables, remove `save_checkpoint()` and the `crew_checkpoints` table.

### 7. Initial State Builders

Each runner needs a function that constructs the initial state dict for its graph. These are runner-specific because each graph has a different state schema.

**Trading initial state builder:**
```python
async def build_trading_initial_state(cycle_number: int) -> dict:
    """Build initial TradingState for a cycle."""
    # Queries current regime, portfolio positions, cash, exposure
    # Returns dict matching TradingState TypedDict
    ...
```

**Research initial state builder:**
```python
async def build_research_initial_state(cycle_number: int) -> dict:
    """Build initial ResearchState for a cycle."""
    # Queries current regime, loads context summary
    # Returns dict matching ResearchState TypedDict
    ...
```

**Supervisor initial state builder:**
```python
async def build_supervisor_initial_state(cycle_number: int) -> dict:
    """Build initial SupervisorState for a cycle."""
    # Minimal -- just cycle_number, empty lists
    ...
```

These builders may call async DB functions or tool functions to gather current state. They should be lightweight -- heavy data fetching belongs in the graph's `context_load` node, not in the runner.

### 8. Cycle Intervals -- No Changes

The `get_cycle_interval()` function in `runners/__init__.py` is framework-agnostic. It returns seconds based on market hours. No changes needed. The async `run_loop()` uses the same function and passes the result to `asyncio.sleep()`.

### 9. Error Handling Within run_loop

The loop must handle three failure modes:

1. **Graph invocation error** (exception from `ainvoke()`): Catch, log, increment `consecutive_failures`, write checkpoint with `status="error"`. Continue to next cycle.

2. **Timeout** (`asyncio.TimeoutError` from `wait_for()`): Log critical, write checkpoint with `status="timeout"`. The graph invocation is cancelled automatically by `wait_for()`. Continue to next cycle.

3. **Initial state builder error**: If the state builder fails (e.g., DB down), catch, log, write checkpoint with `status="error"`. Do NOT invoke the graph with incomplete state. Continue to next cycle.

After 3 consecutive failures (any type), emit a critical log. The supervisor graph monitors these via the `crew_checkpoints` table and can trigger recovery actions.

### 10. Removing CrewAI Imports

Each runner currently imports from `quantstack.crews.{domain}.crew`. After migration:

- `trading_runner.py`: Replace `from quantstack.crews.trading.crew import TradingCrew` with `from quantstack.graphs.trading.graph import build_trading_graph`
- `research_runner.py`: Replace `from quantstack.crews.research.crew import ResearchCrew` with `from quantstack.graphs.research.graph import build_research_graph`
- `supervisor_runner.py`: Replace `from quantstack.crews.supervisor.crew import SupervisorCrew` with `from quantstack.graphs.supervisor.graph import build_supervisor_graph`

Also import `ConfigWatcher` from `quantstack.graphs.config_watcher` and `AsyncPostgresSaver` from `langgraph.checkpoint.postgres.aio`.

## Checklist

1. Write tests in `tests/unit/test_runners.py` (stubs above)
2. Add `install_async()` method to `GracefulShutdown` in `src/quantstack/health/shutdown.py`
3. Rewrite `run_loop()` in `trading_runner.py` as async with `graph_builder` + `initial_state_builder` parameters
4. Rewrite `trading_runner.main()` to use `asyncio.run(async_main())`
5. Rewrite `research_runner.py` following same pattern (own `async_main`, own initial state builder)
6. Rewrite `supervisor_runner.py` following same pattern
7. Implement initial state builder functions for each runner
8. Verify `get_cycle_interval()` works unchanged (run existing tests)
9. Run full test suite to confirm no regressions
