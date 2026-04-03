# Section 8: Observability (Self-Hosted Langfuse)

## Overview

This section covers integrating self-hosted Langfuse as the observability layer for all CrewAI crews. Langfuse captures LLM calls, tool invocations, agent operations, provider failovers, and custom business events. Every runner must instrument before any crew operations and flush traces on shutdown.

**Files to create/modify:**

- `src/quantstack/observability/__init__.py` — package init
- `src/quantstack/observability/instrumentation.py` — centralized instrumentation setup
- `src/quantstack/observability/tracing.py` — custom trace helpers for business events
- `src/quantstack/observability/flush.py` — shutdown flush utility

**Dependencies on other sections:**

- **Section 1 (Scaffolding):** Langfuse and langfuse-db Docker services must be defined in `docker-compose.yml`. The `langfuse` Python package and `openinference-instrumentation-crewai` must be in `pyproject.toml` under the `[crewai]` optional group.
- **Section 9 (Runners):** Each runner calls the instrumentation and flush utilities from this section. This section defines _what_ to call; section 9 defines _when_ (at runner startup and shutdown).

---

## Tests First

All tests go in `tests/unit/test_observability.py`.

```python
# tests/unit/test_observability.py

"""Tests for Langfuse observability integration.

These tests use mocked Langfuse and CrewAIInstrumentor to verify
wiring without requiring a running Langfuse server.
"""


def test_instrument_crewai_called_before_crew_operations():
    """CrewAIInstrumentor.instrument() must be called exactly once
    at runner startup, before any Crew is instantiated.

    Mock CrewAIInstrumentor and verify .instrument(skip_dep_check=True)
    is called when setup_instrumentation() is invoked."""


def test_langfuse_flush_called_in_shutdown():
    """langfuse.flush() must be called during graceful shutdown
    to avoid losing the last cycle's traces.

    Mock the Langfuse client, call flush_traces(), verify flush()
    was called."""


def test_observe_decorator_applied_to_runner_main():
    """The @observe decorator from langfuse must wrap the runner's
    main loop function so each cycle appears as a top-level trace.

    Verify that the function returned by setup_instrumentation()
    or the runner's main function has the @observe wrapper."""


def test_provider_failover_logged_to_langfuse():
    """When the LLM fallback chain activates, a Langfuse event
    must be emitted with the original provider, the fallback provider,
    and the error that triggered the switch.

    Mock the Langfuse client, call trace_provider_failover(),
    verify a span/event is created with expected metadata."""


def test_strategy_lifecycle_event_includes_reasoning():
    """Strategy promotion/retirement events must include the full
    reasoning text from the LLM agent, not just the outcome.

    Call trace_strategy_lifecycle() with a reasoning string,
    verify the Langfuse span includes that text in its metadata."""


def test_langfuse_env_vars_required():
    """setup_instrumentation() must raise a clear error if
    LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY are not set.

    Unset the env vars, call setup_instrumentation(), expect
    a ValueError with a message naming the missing variable."""
```

---

## Implementation Details

### 8.1 Langfuse Docker Services

Langfuse runs as a Docker Compose service with a dedicated Postgres instance (separate from the quantstack database). These are defined in section 1, but for context:

- **langfuse** service: official `langfuse/langfuse` image, port 3000, health check on `/api/public/health`
- **langfuse-db** service: Postgres instance dedicated to Langfuse

Environment variables required in `.env`:

```
LANGFUSE_SECRET_KEY=sk-lf-...       # generated at first startup
LANGFUSE_PUBLIC_KEY=pk-lf-...       # generated at first startup
LANGFUSE_HOST=http://langfuse:3000  # Docker service name (internal)
NEXTAUTH_SECRET=<random-secret>     # for Langfuse auth
```

The keys are generated via the Langfuse web UI on first access at `http://localhost:3000`. After creation, they go into `.env` and are picked up by crew containers on next restart.

### 8.2 Centralized Instrumentation Setup

Create `src/quantstack/observability/instrumentation.py` with a single entry point that every runner calls once at startup.

```python
# src/quantstack/observability/instrumentation.py

import os
import logging

logger = logging.getLogger(__name__)


def setup_instrumentation() -> None:
    """Initialize Langfuse + CrewAI instrumentation.

    Must be called once per process, before any Crew is instantiated.
    Raises ValueError if required Langfuse env vars are missing.

    Steps:
    1. Validate LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set.
    2. Initialize the Langfuse client (picks up env vars automatically).
    3. Call CrewAIInstrumentor().instrument(skip_dep_check=True) to
       auto-trace all LLM calls, tool invocations, and agent operations.
    4. Log confirmation.

    The skip_dep_check=True flag avoids import-time dependency validation
    that can fail in Docker environments where packages are installed
    but not in the instrumentor's expected layout.
    """
```

Usage in every runner (section 9 handles placement):

```python
from quantstack.observability.instrumentation import setup_instrumentation
setup_instrumentation()
# ... then create and kick off crews
```

### 8.3 Custom Trace Helpers for Business Events

Beyond automatic instrumentation (which captures LLM calls and tool usage), the system needs explicit traces for business-significant events. Create `src/quantstack/observability/tracing.py`.

```python
# src/quantstack/observability/tracing.py

import logging
from langfuse import Langfuse

logger = logging.getLogger(__name__)

# Module-level lazy-init client. Reused across calls within the same process.
_langfuse_client: Langfuse | None = None


def _get_client() -> Langfuse:
    """Return the singleton Langfuse client, initializing on first call."""


def trace_provider_failover(
    original_provider: str,
    fallback_provider: str,
    error: str,
    tier: str,
) -> None:
    """Log a provider failover event to Langfuse.

    Creates a trace with:
    - name: "provider_failover"
    - metadata: original_provider, fallback_provider, error message, model tier
    - level: WARNING

    Called by the LLM provider fallback chain (section 2) whenever a
    provider switch occurs.
    """


def trace_strategy_lifecycle(
    strategy_id: str,
    action: str,
    reasoning: str,
    evidence: dict,
) -> None:
    """Log a strategy promotion/retirement/extension decision.

    Creates a trace with:
    - name: "strategy_lifecycle"
    - metadata: strategy_id, action (promote/retire/extend), full reasoning text
    - input: evidence dict (performance metrics, market conditions, RAG context)

    The reasoning field must be the verbatim LLM output — not a summary.
    This is the audit trail for strategy lifecycle decisions.
    """


def trace_self_healing_event(
    event_type: str,
    details: dict,
) -> None:
    """Log a self-healing event (watchdog trigger, container restart, etc.).

    Creates a trace with:
    - name: "self_healing"
    - metadata: event_type, details dict
    - level: WARNING
    """


def trace_capital_allocation(
    symbol: str,
    recommended_size_pct: float,
    reasoning: str,
    portfolio_context: dict,
) -> None:
    """Log a capital allocation / position sizing decision.

    Creates a trace with:
    - name: "capital_allocation"
    - metadata: symbol, size, reasoning
    - input: portfolio_context dict

    This pairs with the risk agent's output (section 12) to provide
    a complete audit trail of sizing decisions.
    """


def trace_safety_boundary_trigger(
    symbol: str,
    llm_recommendation: dict,
    gate_limit: str,
    gate_value: float,
) -> None:
    """Log when the programmatic safety gate overrides an LLM decision.

    Creates a trace with:
    - name: "safety_boundary_trigger"
    - level: ERROR (this is anomalous — the LLM recommended something unsafe)
    - metadata: symbol, what the LLM recommended, which gate limit was hit

    Critical for monitoring LLM reliability. A high frequency of these
    events indicates the LLM's risk reasoning is miscalibrated.
    """
```

### 8.4 The `@observe` Decorator

Langfuse provides an `@observe()` decorator that wraps a function so its execution appears as a top-level trace (or nested span). Apply this to the runner's per-cycle function so each trading/research/supervisor cycle is a distinct trace in the Langfuse dashboard.

The runners (section 9) apply this as:

```python
from langfuse.decorators import observe

@observe(name="trading_cycle")
def run_one_cycle(inputs: dict) -> dict:
    crew = create_trading_crew()
    result = crew.kickoff(inputs=inputs)
    return result
```

This section does not implement the runners, but the `@observe` decorator is part of the observability contract that runners must honor.

### 8.5 Flush on Shutdown

Langfuse batches trace submissions asynchronously. If the process exits without flushing, the last batch of traces is lost. Create `src/quantstack/observability/flush.py`.

```python
# src/quantstack/observability/flush.py

import logging

logger = logging.getLogger(__name__)


def flush_traces() -> None:
    """Flush all pending Langfuse traces.

    Must be called in the runner's graceful shutdown handler
    (section 7) before process exit.

    Uses the module-level Langfuse client from tracing.py.
    If no client has been initialized (instrumentation was never
    called), this is a no-op.
    """
```

The shutdown handler in section 7 calls this after persisting state to PostgreSQL and before `sys.exit()`.

### 8.6 Cost Tracking

Langfuse automatically tracks token usage and cost per LLM call when the CrewAI instrumentor is active. No additional code is needed. The Langfuse dashboard (http://localhost:3000) provides:

- Cost per agent per day
- Cost per crew per cycle
- Total daily cost across all providers
- Provider usage distribution (useful for verifying fallback chain behavior)

No custom code is required for cost tracking. It is a built-in Langfuse feature once traces include model and token count metadata, which the CrewAI instrumentor provides automatically.

### 8.7 Langfuse Retention

Langfuse accumulates trace data over time. To prevent unbounded growth in the langfuse-db Postgres instance, configure retention:

- **Detailed traces:** 30 days
- **Aggregated metrics:** indefinite

Implementation: the supervisor crew (section 5, task `scheduled_tasks`) runs a monthly cleanup. The cleanup function queries the Langfuse API or directly deletes old rows from the langfuse-db. This is a supervisor responsibility, not an observability module responsibility. The observability module only needs to expose the Langfuse client for the supervisor to use.

---

## Integration Points

### With Section 2 (LLM Providers)

The fallback chain in `src/quantstack/llm/provider.py` calls `trace_provider_failover()` whenever it switches providers. The provider module imports from `quantstack.observability.tracing`.

### With Section 7 (Self-Healing)

The watchdog and recovery logic calls `trace_self_healing_event()` when triggering recovery actions. The graceful shutdown handler calls `flush_traces()` as its final step.

### With Section 9 (Runners)

Each runner calls `setup_instrumentation()` once at startup (before the main loop). Each runner's per-cycle function is decorated with `@observe`. Each runner's shutdown path calls `flush_traces()`.

### With Section 12 (Risk Safety)

The programmatic safety gate calls `trace_safety_boundary_trigger()` whenever it overrides an LLM recommendation. The risk agent's output is logged via `trace_capital_allocation()`.

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/observability/__init__.py` | Create | Package init, re-export `setup_instrumentation`, `flush_traces` |
| `src/quantstack/observability/instrumentation.py` | Create | One-time CrewAI + Langfuse instrumentor setup |
| `src/quantstack/observability/tracing.py` | Create | Custom trace helpers for business events |
| `src/quantstack/observability/flush.py` | Create | Shutdown flush utility |
| `tests/unit/test_observability.py` | Create | Unit tests for all observability functions |
