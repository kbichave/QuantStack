# Section 10: Research Fan-Out Default On

## Overview

The research graph supports two execution paths: a sequential pipeline (hypothesis -> signal validation -> backtest -> ML experiment -> registration) and a fan-out pipeline that distributes hypotheses to parallel validation workers. Fan-out is 3-5x faster but is currently disabled by default (`RESEARCH_FAN_OUT_ENABLED` defaults to `"false"`). This section flips the default to enabled and adds dual-layer throttling to prevent AV quota exhaustion under parallel load.

**Scope:** Three changes in total -- a one-line default flip, a concurrency semaphore, and a quota-aware throttle. No new tables, no new tools, no new graph nodes.

**File paths involved:**

- `src/quantstack/graphs/research/graph.py` -- default flip (line 74 today)
- `src/quantstack/graphs/research/nodes.py` -- semaphore + rate limiter in `fan_out_hypotheses`
- `tests/unit/test_research_fanout.py` -- new test file
- `tests/integration/test_research_fanout_integration.py` -- new test file

**Dependencies:** Section 01 (DB schema) must be complete since the research graph imports from modules that assume tables exist. No dependency on the alert infrastructure -- this section does not emit system alerts.

---

## Tests

Write tests first. All tests go in new files since no research fan-out tests exist today.

### Unit Tests (`tests/unit/test_research_fanout.py`)

```python
# Test: fan_out_enabled defaults to True (env var not set)
# Verify that when RESEARCH_FAN_OUT_ENABLED is absent from the environment,
# the graph builder selects the fan-out path (validate_symbol and filter_results
# nodes present in the compiled graph).

# Test: fan_out_enabled=False when RESEARCH_FAN_OUT_ENABLED=false
# Verify that setting the env var to "false" causes the graph builder to select
# the sequential path (signal_validation, backtest_validation, ml_experiment
# nodes present; validate_symbol absent).

# Test: fan_out semaphore limits concurrent tasks to 10
# Create a mock fan_out_hypotheses invocation with 20 hypotheses. Instrument
# the semaphore to track peak concurrency. Assert peak <= 10.

# Test: AV rate limiter prevents >75 calls/min during fan-out
# Mock the AV rate limiter. Fire 80 requests through fan-out workers.
# Assert the rate limiter was consulted before each AV call and that no
# more than 75 passed within any 60-second window.

# Test: quota monitoring throttles new launches when calls > 60/min
# Set the AV call counter to 61 for the current minute window. Launch a
# fan-out batch. Assert that a >= 1-second delay was injected between
# successive worker launches (throttle mode engaged at 80% of 75 limit).
```

### Integration Tests (`tests/integration/test_research_fanout_integration.py`)

```python
# Test: research graph uses fan-out path by default (fan_out_hypotheses node present)
# Build the research graph with no RESEARCH_FAN_OUT_ENABLED env var set.
# Inspect the compiled graph's node names. Assert "validate_symbol" and
# "filter_results" are present. Assert "signal_validation" is absent.

# Test: research graph uses sequential path when RESEARCH_FAN_OUT_ENABLED=false
# Set RESEARCH_FAN_OUT_ENABLED=false, build the graph. Assert "signal_validation",
# "backtest_validation", and "ml_experiment" are present. Assert "validate_symbol"
# is absent.
```

---

## Implementation

### 10.1 Flip the Default

In `src/quantstack/graphs/research/graph.py`, line 74, change the default from `"false"` to `"true"`:

```python
# Before:
fan_out_enabled = os.environ.get("RESEARCH_FAN_OUT_ENABLED", "false").lower() == "true"

# After:
fan_out_enabled = os.environ.get("RESEARCH_FAN_OUT_ENABLED", "true").lower() == "true"
```

This is a one-line change. Users who want sequential mode can set `RESEARCH_FAN_OUT_ENABLED=false` in their environment or `.env` file.

### 10.2 Add Semaphore + Rate Limiter Dual Throttling

In `src/quantstack/graphs/research/nodes.py`, modify the `fan_out_hypotheses` function to add two layers of throttling:

**Layer 1 -- Concurrency semaphore:** An `asyncio.Semaphore(10)` that limits how many validation workers run concurrently. This bounds memory and connection pressure. Without it, a research cycle with 30 hypotheses would spawn 30 concurrent workers, each holding a DB connection and potentially making AV API calls simultaneously.

**Layer 2 -- AV rate limiter:** The existing AV rate limiter in `src/quantstack/data/fetcher.py` already enforces 75 calls/min globally. Each validation worker passes through this limiter before making AV calls. The semaphore alone is not sufficient because 10 concurrent fast calls can still burst through the rate limit. The rate limiter alone is not sufficient because unlimited concurrency causes memory/connection exhaustion even if individual calls are rate-limited.

The implementation pattern inside `fan_out_hypotheses`:

```python
async def fan_out_hypotheses(state: ResearchState) -> dict:
    """Distribute hypotheses to parallel validation workers.

    Two-layer throttling:
    1. asyncio.Semaphore(10) — bounds concurrent workers (memory/connection pressure)
    2. Existing AV rate limiter (75/min) — bounds API call rate (quota protection)

    The semaphore limits concurrency; the rate limiter limits rate. Both are needed.
    """
    # semaphore = asyncio.Semaphore(10)
    # For each hypothesis, wrap the validation call:
    #   async with semaphore:
    #       result = await validate_single_hypothesis(hypothesis, ...)
    # Gather all results, return updated state with validation_results.
```

The semaphore is created once per `fan_out_hypotheses` invocation (not as a module-level global), because each research cycle should get its own concurrency budget. The AV rate limiter is already module-level and shared across all callers, which is correct -- it protects the global AV quota.

### 10.3 Quota Monitoring with 80% Threshold Throttle

Add a pre-launch check to `fan_out_hypotheses` that reads the current AV call count for the active minute window. If the count exceeds 60 (80% of the 75/min limit), engage throttle mode: insert an `asyncio.sleep(1.0)` between successive worker launches instead of launching them all at once.

The AV call counter already exists in the fetcher module's rate limiter. Expose a read-only accessor if one does not exist:

```python
# In src/quantstack/data/fetcher.py (or wherever the AV rate limiter lives):
def get_av_calls_this_minute() -> int:
    """Return the number of AV API calls made in the current 60-second window."""
```

Then in `fan_out_hypotheses`, before each worker launch:

```python
if get_av_calls_this_minute() > 60:
    await asyncio.sleep(1.0)  # Throttle: let the rate window rotate
```

This is a soft throttle that reduces burst pressure. The hard rate limiter (Layer 2) remains the backstop -- if a call would exceed 75/min, the rate limiter blocks it regardless of the soft throttle.

---

## Key Design Rationale

**Why two layers instead of just the rate limiter?** The rate limiter only gates AV API calls. Fan-out workers also consume DB connections, memory for dataframes, and LLM API calls (for signal validation prompts). Ten concurrent workers is a reasonable ceiling for a single research cycle given typical resource availability. The rate limiter would not prevent 30 workers from all holding DB connections and LLM sessions simultaneously.

**Why 10 for the semaphore?** The research graph runs on a single Docker container. With a DB connection pool of ~20 (typical pg_storage config) and 3 graphs sharing it, 10 concurrent research workers leaves headroom for trading and supervisor graphs. This is a reasonable starting point; it can be tuned via a constant or environment variable if needed.

**Why 80% threshold for the soft throttle?** At 60 calls/min, there are 15 calls of headroom before the hard limit. The 1-second delay between launches spreads remaining capacity across the minute window instead of consuming it in a burst. This avoids the pathological case where fan-out launches 10 workers in the first 5 seconds, they all hit AV, exhaust the quota, and then the remaining workers sit blocked on the rate limiter for 50+ seconds.

**Why not make the semaphore size configurable via env var?** YAGNI. The value 10 is reasonable for the current architecture. If it needs tuning, a one-line code change is cheaper than the indirection of another env var that no one will remember exists. If this becomes a pattern (multiple fan-out paths needing tuning), then extract to config.

---

## Verification Checklist

1. With no `RESEARCH_FAN_OUT_ENABLED` env var set, the research graph compiles with fan-out nodes
2. With `RESEARCH_FAN_OUT_ENABLED=false`, the research graph compiles with sequential nodes
3. Fan-out with 20 hypotheses never exceeds 10 concurrent workers
4. Fan-out with heavy AV usage never exceeds 75 calls/min
5. When AV calls exceed 60/min, worker launches are visibly delayed (check logs)
6. Existing sequential path still works identically when fan-out is disabled
