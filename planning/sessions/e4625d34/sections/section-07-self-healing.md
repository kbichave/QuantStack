# Section 7: Self-Healing System

## Overview

This section implements the autonomous self-healing infrastructure in `src/quantstack/health/`. The system detects and recovers from failures without human intervention: stuck agents, crashed processes, lost database connections, downed infrastructure services, and LLM provider outages.

All modules live under `src/quantstack/health/`:

| File | Purpose |
|------|---------|
| `heartbeat.py` | File-based heartbeat write + health check |
| `watchdog.py` | Per-cycle agent timeout detection |
| `shutdown.py` | Graceful SIGTERM/SIGINT handler |
| `retry.py` | Exponential backoff retry wrapper, DB reconnect |

**Dependencies:** Depends on section-01-scaffolding (project structure exists). Blocks section-09-runners (runners import health modules).

---

## Tests First

All tests go in `tests/unit/test_health.py`.

```python
import os
import signal
import tempfile
import time
import threading

import pytest


# --- Heartbeat tests ---

class TestHeartbeat:
    """Tests for heartbeat write and health check functions."""

    def test_write_heartbeat_creates_file(self):
        """write_heartbeat creates a file at /tmp/{crew_name}-heartbeat
        containing a unix timestamp."""

    def test_write_heartbeat_updates_existing_file(self):
        """Calling write_heartbeat twice overwrites with newer timestamp."""

    def test_check_health_returns_true_when_fresh(self):
        """check_health returns True when heartbeat was written
        less than max_age_seconds ago."""

    def test_check_health_returns_false_when_stale(self):
        """check_health returns False when heartbeat is older
        than max_age_seconds."""

    def test_check_health_returns_false_when_file_missing(self):
        """check_health returns False when heartbeat file does not exist."""

    def test_check_health_default_max_age_per_crew(self):
        """check_health uses crew-specific defaults:
        trading=120s, research=600s, supervisor=360s."""

    def test_check_cli_exit_code_healthy(self):
        """The check() entry point (called by Docker healthcheck)
        returns exit code 0 when healthy."""

    def test_check_cli_exit_code_unhealthy(self):
        """The check() entry point returns exit code 1 when stale."""


# --- Watchdog tests ---

class TestAgentWatchdog:
    """Tests for stuck-agent watchdog timer."""

    def test_watchdog_triggers_callback_after_timeout(self):
        """AgentWatchdog fires its callback when timeout_seconds elapses
        without end_cycle() being called."""

    def test_watchdog_end_cycle_cancels_timer(self):
        """Calling end_cycle() before timeout prevents callback."""

    def test_watchdog_does_not_trigger_if_cycle_completes(self):
        """If start_cycle() then end_cycle() within timeout,
        callback is never invoked."""

    def test_watchdog_resets_on_new_cycle(self):
        """Calling start_cycle() again after end_cycle() sets a fresh timer."""


# --- Graceful Shutdown tests ---

class TestGracefulShutdown:
    """Tests for SIGTERM/SIGINT signal handler."""

    def test_sigterm_sets_should_stop(self):
        """Receiving SIGTERM sets the should_stop flag to True."""

    def test_sigint_sets_should_stop(self):
        """Receiving SIGINT sets the should_stop flag to True."""

    def test_should_stop_starts_false(self):
        """GracefulShutdown.should_stop is False before any signal."""

    def test_shutdown_calls_cleanup_callbacks(self):
        """Registered cleanup callbacks are invoked on shutdown signal."""


# --- Retry / Resilience tests ---

class TestResilientRetry:
    """Tests for exponential backoff retry wrapper."""

    def test_retries_on_rate_limit_error(self):
        """resilient_call retries when the wrapped function raises
        a rate-limit-style exception, with exponential backoff."""

    def test_fails_after_max_retries(self):
        """resilient_call raises after exhausting max_retries."""

    def test_no_retry_on_non_retryable_error(self):
        """resilient_call does not retry on ValueError or similar
        non-transient errors — re-raises immediately."""

    def test_db_reconnect_retries_on_operational_error(self):
        """db_reconnect_wrapper retries on psycopg OperationalError
        with backoff (2s, 4s, 8s, 16s, max 60s)."""

    def test_db_reconnect_succeeds_after_transient_failure(self):
        """If the DB comes back after 2 retries, the wrapper succeeds."""
```

---

## Implementation Details

### heartbeat.py

**Purpose:** Each crew runner writes a heartbeat file every cycle. Docker health checks read it to determine container health.

**File path:** `src/quantstack/health/heartbeat.py`

**Functions to implement:**

```python
import time
from pathlib import Path

HEARTBEAT_DIR = Path("/tmp")

# Crew-specific max ages (seconds). If heartbeat is older than this, the crew
# is considered unhealthy. These values are wider than the cycle interval to
# allow for slow cycles without false alarms.
DEFAULT_MAX_AGE = {
    "trading": 120,
    "research": 600,
    "supervisor": 360,
}


def write_heartbeat(crew_name: str) -> None:
    """Write current unix timestamp to /tmp/{crew_name}-heartbeat.

    Called by the runner at the end of each successful cycle.
    """


def check_health(crew_name: str, max_age_seconds: int | None = None) -> bool:
    """Return True if the heartbeat file exists and is fresher than max_age_seconds.

    If max_age_seconds is None, uses DEFAULT_MAX_AGE for the crew.
    Returns False if the file is missing or stale.
    """


def check(crew_name: str) -> None:
    """CLI entry point for Docker health check.

    Called as: python -c "from quantstack.health.heartbeat import check; check('trading')"

    Calls sys.exit(0) if healthy, sys.exit(1) if unhealthy.
    """
```

**Docker health check integration:** Each crew service in `docker-compose.yml` uses:

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "from quantstack.health.heartbeat import check; check('trading')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 120s
```

The `start_period` gives the first cycle time to complete before health checks begin failing.

---

### watchdog.py

**Purpose:** Detect stuck agents. If a single crew cycle takes longer than the timeout, the watchdog fires a callback that forces the runner to abandon the current cycle and start fresh.

**File path:** `src/quantstack/health/watchdog.py`

**Class to implement:**

```python
import threading
from typing import Callable


class AgentWatchdog:
    """Timer-based watchdog that detects stuck crew cycles.

    Usage in the runner loop:
        watchdog = AgentWatchdog(timeout_seconds=600, on_timeout=handle_stuck)
        watchdog.start_cycle()
        crew.kickoff(...)
        watchdog.end_cycle()

    If end_cycle() is not called within timeout_seconds of start_cycle(),
    the on_timeout callback fires. The callback should set a flag that
    causes the runner to skip the current result and start a new cycle.
    """

    def __init__(self, timeout_seconds: int, on_timeout: Callable[[], None]) -> None:
        """Initialize with timeout duration and callback.

        Args:
            timeout_seconds: Max allowed seconds per cycle before triggering.
            on_timeout: Zero-arg callable invoked when timeout fires.
        """

    def start_cycle(self) -> None:
        """Start (or reset) the watchdog timer for a new cycle."""

    def end_cycle(self) -> None:
        """Cancel the current timer. Call after cycle completes normally."""
```

**Timeout values:** Trading crew: 600s (10 min). Research crew: 1200s (20 min, ML training can be slow). Supervisor crew: 300s (5 min).

**What happens on timeout:** The `on_timeout` callback sets a threading.Event that the runner checks. The runner does NOT kill the crew thread -- it logs the timeout, marks the cycle as failed, writes a Langfuse event, and proceeds to the next cycle. The stale crew instance is garbage-collected when the runner creates a fresh one.

---

### shutdown.py

**Purpose:** Handle SIGTERM and SIGINT for graceful container shutdown. When Docker Compose sends SIGTERM, the runner finishes its current cycle (up to 60s), flushes Langfuse, persists state, and exits cleanly.

**File path:** `src/quantstack/health/shutdown.py`

**Class to implement:**

```python
import signal
from typing import Callable


class GracefulShutdown:
    """Signal handler for graceful process termination.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register_cleanup(langfuse.flush)
        shutdown.register_cleanup(persist_state)
        shutdown.install()  # registers SIGTERM + SIGINT handlers

        while not shutdown.should_stop:
            run_one_cycle()

        # After loop exits, cleanup callbacks run automatically
    """

    def __init__(self) -> None:
        """Initialize with should_stop=False and empty cleanup list."""

    @property
    def should_stop(self) -> bool:
        """True after SIGTERM or SIGINT received."""

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup function to run on shutdown.

        Callbacks execute in registration order. Each callback gets
        a 10-second timeout; if it hangs, the next one runs anyway.
        """

    def install(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT.

        On signal receipt:
        1. Set should_stop = True (runner loop exits after current cycle)
        2. Run all registered cleanup callbacks
        """
```

**Docker Compose configuration:** The `stop_grace_period` for crew services should be 90 seconds. This gives: up to 60s for the current cycle to finish + up to 30s for cleanup callbacks. If the process hasn't exited by 90s, Docker sends SIGKILL.

```yaml
# In docker-compose.yml for each crew service:
stop_grace_period: 90s
```

---

### retry.py

**Purpose:** Exponential backoff retry logic for transient failures. Two main use cases: (1) wrapping crew kickoff calls for LLM rate limits, and (2) wrapping database connections for transient connection losses.

**File path:** `src/quantstack/health/retry.py`

**Functions to implement:**

```python
import functools
import time
import random
import logging
from typing import TypeVar, Callable, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions considered retryable (transient)
RETRYABLE_EXCEPTIONS: tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    # Add provider-specific rate limit errors as needed
)


def resilient_call(
    fn: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[Type[Exception], ...] | None = None,
    **kwargs,
) -> T:
    """Call fn with exponential backoff + jitter on retryable errors.

    Args:
        fn: The function to call.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).
        max_delay: Cap on delay between retries.
        retryable_exceptions: Exception types to retry on.
            Defaults to RETRYABLE_EXCEPTIONS.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all retries are exhausted,
        or immediately for non-retryable exceptions.
    """


def db_reconnect_wrapper(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator that retries database operations on OperationalError.

    Backoff schedule: 2s, 4s, 8s, 16s, capped at 60s.
    Max retries: 10 (covers ~2 minutes of DB downtime).

    Wraps the existing db_conn() context manager usage pattern.
    """
```

**Backoff formula:** `delay = min(base_delay * 2^attempt + random.uniform(0, 1), max_delay)`. The jitter prevents thundering herd when multiple crew processes retry simultaneously.

**Non-retryable errors:** `ValueError`, `TypeError`, `KeyError`, `AssertionError`, and any other programming error should NOT be retried. The function re-raises these immediately. Only network/transient errors get retried.

---

### __init__.py

**File path:** `src/quantstack/health/__init__.py`

```python
from quantstack.health.heartbeat import write_heartbeat, check_health
from quantstack.health.watchdog import AgentWatchdog
from quantstack.health.shutdown import GracefulShutdown
from quantstack.health.retry import resilient_call, db_reconnect_wrapper
```

---

## Failure Taxonomy Reference

This is the complete map of failure modes and their recovery paths. The health module provides the building blocks; the runners (section-09) wire them together.

| Failure Mode | Detection Mechanism | Recovery Action | Module |
|---|---|---|---|
| LLM provider down | Provider-specific exception during crew kickoff | Fallback chain: Bedrock -> Anthropic -> OpenAI -> Ollama | `retry.py` + section-02 provider fallback |
| Stuck agent (>10 min) | `AgentWatchdog` timeout fires | Abandon cycle, log to Langfuse, start fresh | `watchdog.py` |
| Crashed crew process | Docker health check (heartbeat file stale) | Docker `restart: unless-stopped` restarts container | `heartbeat.py` + Docker |
| Database connection lost | `OperationalError` from psycopg | Exponential backoff reconnect (2s..60s) | `retry.py` |
| Ollama down | Embedding call fails | Skip RAG, continue with DB state + in-cycle memory only | Degraded mode in runners |
| ChromaDB down | HTTP client connection error | Skip RAG, supervisor restarts container | Degraded mode in runners |
| API rate limit (AV, Alpaca) | HTTP 429 response | Exponential backoff with jitter | `retry.py` (existing data adapters also have this) |
| Stale market data | Timestamp check on last OHLCV update | Supervisor triggers data refresh | Coordination tools |

---

## Integration Points

### How runners use these modules (section-09 will implement this)

The runner loop in `src/quantstack/runners/trading_runner.py` (and similarly for research/supervisor) wires together all health components:

```python
def main():
    shutdown = GracefulShutdown()
    shutdown.register_cleanup(langfuse.flush)
    shutdown.install()

    watchdog = AgentWatchdog(timeout_seconds=600, on_timeout=handle_timeout)

    while not shutdown.should_stop:
        watchdog.start_cycle()
        try:
            crew = create_trading_crew()
            result = resilient_call(crew.kickoff, inputs=build_inputs())
        except Exception:
            logger.exception("Cycle failed")
        finally:
            watchdog.end_cycle()

        write_heartbeat("trading")
        sleep_until_next_cycle()
```

This is shown for context only -- section-09 owns the runner implementation.

### How Docker uses heartbeat (section-01 owns docker-compose.yml)

The heartbeat health check command is defined in `docker-compose.yml` per crew service. The `check()` function in `heartbeat.py` is the interface Docker calls.

---

## Design Decisions

**Why file-based heartbeats instead of DB-based:** Docker health checks need to be fast and have no external dependencies. A file read from `/tmp/` is O(1) and works even if the database is down. If we used the DB for heartbeat, a DB outage would cause Docker to restart all crew containers (making the DB outage worse).

**Why threading.Timer for watchdog instead of asyncio:** CrewAI's `crew.kickoff()` is a blocking call. An asyncio-based watchdog would require the runner to be async, which complicates the integration. A background thread timer fires regardless of what the main thread is doing.

**Why cleanup callbacks have individual timeouts:** If Langfuse flush hangs (network issue), we still want DB state persistence to run. Each callback gets 10 seconds independently, so one stuck callback doesn't prevent the others.

**Why not restart the stuck crew mid-cycle:** CrewAI manages its own internal thread/async state. Force-killing a crew mid-execution risks corrupted state, leaked connections, and orphaned LLM calls. It is safer to let the garbage collector clean up the abandoned crew instance and start fresh.
