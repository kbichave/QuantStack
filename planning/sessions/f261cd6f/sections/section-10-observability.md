# Section 10: Observability Changes

## Goal

Remove all CrewAI instrumentation from the observability layer. Replace it with a LangFuse CallbackHandler factory that integrates with LangGraph's callback system. Consolidate `crew_tracing.py` into `tracing.py`. Ensure trace flushing works on shutdown.

This section touches four files in `src/quantstack/observability/` and deletes one.

## Dependencies

- **Section 01 (Scaffolding)**: `langfuse` must be in dependencies (it already is; the new dependency is `langchain-core` which provides the `CallbackHandler` integration point).
- **Blocked by nothing else** -- this section is parallelizable with sections 02-05, 09, and 12.
- **Blocks Section 11 (Runners)**: Runners consume `get_langfuse_handler()` to pass into `graph.ainvoke()` config.

## Tests (Write First)

All tests go in `tests/unit/test_observability.py`. Write these before any implementation.

```python
# tests/unit/test_observability.py

# Test: setup_instrumentation() does not import CrewAIInstrumentor
#   - Patch sys.modules or inspect the function source to confirm
#     no reference to openinference.instrumentation.crewai.
#   - Call setup_instrumentation() with LANGFUSE keys set in env.
#     It must not raise ImportError for crewai-related packages.

# Test: setup_instrumentation() initializes Langfuse callback handler
#   - Mock the langfuse CallbackHandler constructor.
#   - Call setup_instrumentation() with valid env vars.
#   - Assert the handler was constructed (or that the module-level
#     state reflects initialization).

# Test: get_langfuse_handler() returns CallbackHandler with session_id and tags
#   - Call get_langfuse_handler(session_id="test-session", tags=["trading"])
#   - Assert returned object is a langfuse CallbackHandler (or mock equivalent).
#   - Assert session_id and tags are passed through.

# Test: graph invocation with callback handler produces trace spans (mock LangFuse)
#   - Build a trivial 2-node StateGraph.
#   - Invoke it with a mocked langfuse callback handler in config["callbacks"].
#   - Assert the handler received on_chain_start / on_chain_end calls
#     (verifying LangGraph dispatches to the callback).

# Test: trace_provider_failover() still works (pure LangFuse, no CrewAI)
#   - Mock _get_langfuse() to return a fake Langfuse client.
#   - Call trace_provider_failover() with sample args.
#   - Assert lf.trace() was called with name="provider_failover".

# Test: trace_strategy_lifecycle() still works
#   - Same pattern as above, assert name="strategy_lifecycle".

# Test: trace_safety_boundary_trigger() still works
#   - Same pattern, assert name="safety_boundary_trigger".

# Test: crew_tracing.py does not exist (merged into tracing.py)
#   - Assert that the file path
#     src/quantstack/observability/crew_tracing.py does not exist on disk.

# Test: no code imports openinference.instrumentation.crewai
#   - Scan all .py files under src/quantstack/ for the string
#     "openinference.instrumentation.crewai". Assert zero matches.
```

## Current State of the Files

Understanding what exists today is essential before making changes.

### `instrumentation.py` (modify)

Currently:
- Validates `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` env vars.
- Imports `CrewAIInstrumentor` from `openinference.instrumentation.crewai`.
- Calls `CrewAIInstrumentor().instrument(skip_dep_check=True)`.
- Logs success or warns if the package is missing.

### `crew_tracing.py` (delete after merging)

Contains five domain-specific trace functions that call `_get_langfuse()` from `tracing.py`:
- `trace_provider_failover()`
- `trace_strategy_lifecycle()`
- `trace_self_healing_event()`
- `trace_capital_allocation()`
- `trace_safety_boundary_trigger()`

These are pure LangFuse calls with no CrewAI dependency. They must be preserved.

### `tracing.py` (modify -- receives merged content)

Contains:
- `_get_langfuse()` lazy singleton initializer.
- `TracingSpan` wrapper class.
- Several trace helper functions (`trace_optimization`, `trace_textgrad_critique`, `trace_opro_generation`, `trace_judge_verdict`, `trace_research_critique`).
- `flush()` function.

### `flush_util.py` (modify -- ensure shutdown integration)

Contains `flush_traces()` which delegates to `tracing.flush()`. This is called by the runner's graceful shutdown handler.

## Implementation Steps

### Step 1: Rewrite `instrumentation.py`

Remove all CrewAI references. The new `setup_instrumentation()` function should:

1. Validate `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` env vars (keep existing validation).
2. Do NOT import or call `CrewAIInstrumentor`.
3. Initialize the LangFuse client (trigger `_get_langfuse()` from `tracing.py` so it is ready).
4. Log confirmation that LangFuse instrumentation is active.

Add a new factory function in this file:

```python
def get_langfuse_handler(session_id: str, tags: list[str]) -> "CallbackHandler":
    """Create a LangFuse callback handler for a graph invocation.

    This handler is passed to graph.ainvoke() via the config dict:
        config = {"callbacks": [get_langfuse_handler(session_id, tags)]}

    LangGraph automatically dispatches node starts/ends, LLM calls,
    and tool invocations to the handler, producing full trace trees
    in the LangFuse UI.

    Args:
        session_id: Groups related traces (e.g., "trading-2026-04-02-cycle-3").
        tags: Categorization tags (e.g., ["trading", "paper"]).

    Returns:
        A langfuse CallbackHandler instance configured for this session.
    """
```

The implementation should import `CallbackHandler` from `langfuse.callback` and construct it with:
- `session_id=session_id`
- `tags=tags`
- The LangFuse public/secret keys and host from environment (the `CallbackHandler` reads these from env by default, so explicit passing is optional but makes testing easier).

### Step 2: Merge `crew_tracing.py` into `tracing.py`

Move all five functions from `crew_tracing.py` into `tracing.py`:
- `trace_provider_failover()`
- `trace_strategy_lifecycle()`
- `trace_self_healing_event()`
- `trace_capital_allocation()`
- `trace_safety_boundary_trigger()`

These functions are copy-paste moves -- no logic changes. They already import `_get_langfuse` from `tracing.py`, so after the merge they reference it as a module-local function.

Update all docstrings in these functions to remove any mention of "CrewAI". For example, the module docstring currently says "Custom Langfuse trace helpers for CrewAI business events" -- change to "Custom Langfuse trace helpers for QuantStack business events" or similar.

After the merge, delete `crew_tracing.py`.

### Step 3: Update imports across the codebase

Any file that imports from `crew_tracing` must be updated:

```python
# Before
from quantstack.observability.crew_tracing import trace_provider_failover

# After
from quantstack.observability.tracing import trace_provider_failover
```

Search for all imports of `crew_tracing` in `src/quantstack/` and update them.

### Step 4: Update `flush_util.py`

The current implementation already delegates to `tracing.flush()`. Enhance it to also call `langfuse_client.shutdown()` (which flushes and closes the client cleanly) rather than just `flush()`. The `_get_langfuse()` singleton exposes a `shutdown()` method.

```python
def flush_traces() -> None:
    """Flush all pending Langfuse traces and shut down the client.

    Must be called in the runner's graceful shutdown handler
    before process exit. No-op if instrumentation was never initialized.
    """
```

The implementation should:
1. Call `flush()` from `tracing.py` (existing behavior).
2. Call `_get_langfuse().shutdown()` if the client exists (new -- ensures the background worker thread is stopped and all events are sent).

### Step 5: Update `__init__.py` (optional)

If `get_langfuse_handler` should be a top-level export from the `observability` package, add it to `__init__.py`. This is a convenience -- runners can import directly from `instrumentation.py` if preferred.

### Step 6: Remove `openinference-instrumentation-crewai` from dependencies

This is handled in Section 01 (Scaffolding) when `pyproject.toml` is updated. Verify that after this section's changes, no code references this package.

## File Summary

| File | Action |
|------|--------|
| `src/quantstack/observability/instrumentation.py` | Rewrite: remove CrewAI, add `get_langfuse_handler()` factory |
| `src/quantstack/observability/tracing.py` | Modify: add 5 trace functions from `crew_tracing.py`, update docstrings |
| `src/quantstack/observability/crew_tracing.py` | Delete (after merging contents into `tracing.py`) |
| `src/quantstack/observability/flush_util.py` | Modify: add `shutdown()` call alongside `flush()` |
| `src/quantstack/observability/__init__.py` | Optionally add `get_langfuse_handler` export |
| `tests/unit/test_observability.py` | Write all tests listed above |

## Key Design Decisions

**Why a factory function (`get_langfuse_handler`) instead of a singleton handler**: Each graph invocation needs its own `session_id` and `tags` to produce properly scoped traces. A singleton handler would mix traces from different cycles together. The factory creates a fresh handler per cycle, which the runner passes into `graph.ainvoke(state, config={"callbacks": [handler]})`.

**Why merge `crew_tracing.py` rather than just rename it**: The five business-event trace functions share the same `_get_langfuse()` dependency as the existing functions in `tracing.py`. Having them in two files creates an artificial split. After removing "CrewAI" from the names and docstrings, there is no reason for the separation. One file (`tracing.py`) owns all LangFuse trace helpers.

**Why `shutdown()` in addition to `flush()`**: The LangFuse Python client uses a background thread to batch-send events. `flush()` sends pending events but leaves the thread running. `shutdown()` flushes and stops the thread. For process exit, `shutdown()` is the correct call -- it prevents the background thread from keeping the process alive after the main loop exits.
