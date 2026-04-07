# Section 07: Node Circuit Breaker

## Background

If a node (e.g., `daily_planner`) fails 5 consecutive cycles, the graph still calls it on cycle 6. There is no backoff, no circuit breaker, no fallback. The existing `StrategyBreaker` in `src/quantstack/execution/strategy_breaker.py` handles strategy-level P&L breakers -- a completely different concern from node health.

This section introduces a `@circuit_breaker` decorator for graph node functions. The breaker uses a classic three-state model (closed / open / half-open) backed by PostgreSQL for persistence across restarts and concurrent graph cycles. Per-node configurable cooldowns and LLM-specific failure type discrimination ensure the breaker behaves correctly for the different failure modes a trading system encounters.

## Dependencies

- **section-01-db-migration-and-policy**: The `circuit_breaker_state` table must exist before this section's code can run. Schema defined below for reference, but the migration itself belongs to section-01.
- **section-04-node-output-models**: Each node's output model must define a `safe_default()` class method. The circuit breaker returns this safe default when a node is skipped. For blocking nodes, the safe default must include an error flag that the execution gate (section-05) detects.

## Tests First

All tests live in `tests/unit/test_circuit_breaker.py`.

### State Machine Tests

```python
# Test: closed state -- failure increments count, stays closed until threshold
# Test: closed -> open -- 3 consecutive failures trips breaker
# Test: open state -- node skipped, safe default returned
# Test: open -> half_open -- after cooldown_seconds (300s) expires
# Test: half_open success -> closed, counter reset
# Test: half_open failure -> back to open
# Test: success in closed state -> counter reset to 0
```

### DB Persistence Tests

```python
# Test: breaker state persists across test invocations (write, read back, verify)
# Test: concurrent increment -- two overlapping cycles both incrementing -> correct count (atomic)
# Test: initial state for new breaker_key -> default closed/0
```

### LLM Failure Type Tests

```python
# Test: rate limit (429) -> trips immediately regardless of threshold
# Test: token limit exceeded -> does NOT trip breaker, routes to pruning
# Test: parse failure -> counted separately from execution failures
# Test: provider outage (5xx) -> trips immediately
```

### Safe Default Tests

```python
# Test: blocking node circuit-broken -> safe default includes error flag -> execution gate catches it
# Test: non-blocking node circuit-broken -> safe default is neutral, pipeline continues
```

### Alert Tests

```python
# Test: 5 consecutive failures -> Langfuse event emitted + outbound notification triggered
# Test: 5 consecutive on blocking node -> graph halted for cycle
```

## Implementation Details

### File: `src/quantstack/graphs/circuit_breaker.py` (NEW)

This new file contains the decorator, DB operations, and state machine logic.

### DB Table Schema (reference -- created by section-01)

```
circuit_breaker_state
├── breaker_key TEXT PRIMARY KEY  (e.g., "trading/data_refresh")
├── state TEXT DEFAULT 'closed'   (closed / open / half_open)
├── failure_count INT DEFAULT 0
├── last_failure_at TIMESTAMPTZ
├── opened_at TIMESTAMPTZ
├── cooldown_seconds INT DEFAULT 300
└── last_success_at TIMESTAMPTZ
```

### Three-State Model

- **Closed** (normal operation): Failures are counted. When `failure_count` reaches the `threshold` (default 3) consecutive failures, the breaker transitions to Open.
- **Open** (node skipped): The decorated node is not called. Instead, the node's typed safe default is returned immediately. After `cooldown_seconds` elapses (default 300s, configurable per node), the breaker transitions to Half-Open. The cooldown must exceed the graph cycle interval (60-300s) to ensure the breaker actually skips at least one full cycle.
- **Half-Open** (probe): One request is allowed through. If it succeeds, the breaker transitions to Closed and the failure counter resets. If it fails, the breaker transitions back to Open.

### Decorator Signature

```python
def circuit_breaker(threshold: int = 3, alert_threshold: int = 5, cooldown_seconds: int = 300):
    """Decorator for graph node functions. Checks breaker state before invocation,
    updates state after success or failure."""
    ...
```

Applied to node functions like:

```python
@circuit_breaker(threshold=3, alert_threshold=5)
async def data_refresh(state: TradingState) -> DataRefreshOutput:
    ...
```

### Decorator Behavior

1. **Before invocation**: Read breaker state from DB using `breaker_key` (constructed as `"{graph_name}/{node_name}"`).
   - If **Open** and cooldown not expired: return `NodeOutputModel.safe_default()`, skip invocation.
   - If **Open** and cooldown expired: transition to **Half-Open**, allow invocation.
   - If **Closed** or **Half-Open**: proceed with invocation.
2. **After successful invocation**: Transition to **Closed**, reset `failure_count` to 0, update `last_success_at`.
3. **After failed invocation**: Apply LLM failure type discrimination (see below), then increment `failure_count` atomically. If count reaches threshold, transition to **Open** and set `opened_at`.

### Concurrency Safety

Use atomic increment to prevent read-modify-write races when multiple graph cycles overlap:

```sql
UPDATE circuit_breaker_state
SET failure_count = failure_count + 1
WHERE breaker_key = ?
RETURNING failure_count, state
```

This single statement increments and returns the new count atomically. The decorator checks the returned `failure_count` against the threshold to decide whether to trip.

For new breaker keys (first-ever failure), use an upsert:

```sql
INSERT INTO circuit_breaker_state (breaker_key, state, failure_count, last_failure_at)
VALUES (?, 'closed', 1, NOW())
ON CONFLICT (breaker_key) DO UPDATE SET
  failure_count = circuit_breaker_state.failure_count + 1,
  last_failure_at = NOW()
RETURNING failure_count, state
```

### LLM Failure Type Discrimination

Not all failures should trip the breaker the same way:

| Failure Type | Behavior |
|---|---|
| Rate limit (HTTP 429) | Trip **immediately** regardless of threshold. Respect `Retry-After` header as minimum cooldown. |
| Token limit exceeded | Do **not** trip the breaker. Route to message pruning. This is a data problem, not a service problem. |
| Parse failure (JSON decode, validation) | Count **separately** from execution failures. High rate signals prompt degradation, not service failure. |
| Provider outage (5xx, connection error) | Trip **immediately**. |

The decorator must inspect the exception type or error metadata to classify failures. Define an enum or mapping for failure types to keep classification logic centralized.

### Safe Defaults

The safe default is provided by each node's output model (from section-04) via a `safe_default()` class method. For example, `PlanDayOutput.safe_default()` returns neutral bias and no new trades.

For **blocking** nodes (as classified in section-05), the safe default must set an error flag in the returned output. This error flag is what the execution gate inspects to halt the pipeline. The circuit breaker itself does not halt the pipeline -- it delegates that decision to the execution gate by returning an output that the gate recognizes as a failure.

For **non-blocking** nodes, the safe default is neutral and the pipeline continues normally.

### Alerting

When `failure_count` reaches the `alert_threshold` (default 5):
- Emit a Langfuse event with the breaker key, failure count, and last error details.
- Trigger an outbound notification (Slack webhook or email, using existing notification infrastructure).
- If the node is blocking, the pipeline will already be halted by the execution gate (via the safe default's error flag). The alert is informational -- the safety mechanism is the gate, not the alert.

### Integration with Existing Code

The decorator is applied in `src/quantstack/graphs/trading/nodes.py` (and equivalents for research/supervisor). Each node function gains the `@circuit_breaker(...)` decorator. The decorator wraps the async node function transparently -- no changes to graph wiring or node signatures beyond adding the decorator.

All DB operations use `db_conn()` context managers, consistent with the codebase convention.

## TODO Checklist

1. Create `src/quantstack/graphs/circuit_breaker.py` with:
   - Failure type enum/classification
   - DB read/write functions (atomic increment, upsert for new keys, state transitions)
   - Three-state machine logic (closed/open/half-open transitions)
   - `@circuit_breaker` decorator that wraps async node functions
   - Langfuse event emission at alert threshold
2. Write `tests/unit/test_circuit_breaker.py` with all test stubs listed above
3. Apply `@circuit_breaker(...)` decorator to node functions in:
   - `src/quantstack/graphs/trading/nodes.py`
   - `src/quantstack/graphs/research/nodes.py` (if applicable)
   - `src/quantstack/graphs/supervisor/nodes.py` (if applicable)
4. Verify that each node's output model (from section-04) has a working `safe_default()` that the decorator can call
5. Verify that blocking node safe defaults include an error flag compatible with the execution gate (section-05)
