# Section 11: Testing Strategy

## Purpose

This section defines the comprehensive testing strategy for all 10 safety hardening items. It covers test infrastructure, test categories per section, integration and end-to-end test designs, chaos/failure injection scenarios, and concurrent stress tests. This is the final section -- it depends on sections 1-10 being complete (or at least their interfaces being stable) before the integration and end-to-end tests can run.

## Dependencies

- **All sections 1-10**: This section exercises functionality built in every prior section. Unit tests for each section should be written alongside that section's implementation (see TDD stubs below). The integration, end-to-end, and stress tests defined here run after the individual sections are complete.

---

## Test Infrastructure

### Framework and Configuration

The project uses **pytest** as the test runner, invoked via `uv run pytest`. Tests live under `tests/` with this directory structure:

```
tests/
  unit/           # Fast, isolated, no external dependencies
  integration/    # Require PostgreSQL, Docker, or broker stubs
  core/           # Core library tests (indicators, backtesting, ML)
  graphs/         # Graph node and agent tests
  coordination/   # EventBus, kill switch, inter-graph tests
  regression/     # Prevent known bugs from recurring
  benchmarks/     # Performance baselines
  property/       # Property-based / fuzz tests
  fixtures/       # Shared test data
  conftest.py     # Shared fixtures, markers, async support
```

### Markers

Tests are tagged with pytest markers to control execution scope:

- `@pytest.mark.slow` -- tests that take more than 10 seconds (checkpoint crash recovery, stress tests)
- `@pytest.mark.integration` -- require PostgreSQL or Docker
- `@pytest.mark.benchmark` -- performance measurement (not run in CI by default)
- `@pytest.mark.requires_api` -- require live broker or data API keys
- `@pytest.mark.requires_gpu` -- ML tests needing GPU

A typical CI run executes `uv run pytest -m "not slow and not requires_api and not requires_gpu"`. The full suite (including slow and integration) runs nightly or pre-deploy.

### Async Support

Several components (execution monitor, broker WebSocket handlers) are async. Use `pytest-asyncio` with the `asyncio` fixture mode. Async test functions use `@pytest.mark.asyncio`.

### Database Fixtures

After the psycopg3 migration (Section 1), all DB fixtures in `conftest.py` must use `psycopg` (not `psycopg2`). The key fixtures:

```python
@pytest.fixture
def db_conn():
    """Yield a psycopg3 connection with dict_row factory, rolled back after test."""
    # Connect using psycopg3 ConnectionPool
    # Set row_factory=dict_row
    # Wrap in transaction, rollback on teardown
    ...

@pytest.fixture
def db_pool():
    """Yield a psycopg3 ConnectionPool for tests that need pool behavior."""
    ...
```

### Broker Stubs

Tests for stop-loss enforcement, bracket orders, and transaction isolation need broker behavior without hitting live APIs. Use `PaperBroker` as the primary test broker -- it already simulates fills and order tracking. For failure injection (HTTP 500, timeout), create a `FaultyBroker` wrapper:

```python
class FaultyBroker:
    """Wraps any broker adapter and injects configurable failures.
    
    Usage:
        broker = FaultyBroker(PaperBroker(), fail_next_n=3, error=BrokerAPIError("500"))
    """
    def __init__(self, inner, fail_next_n: int = 0, error: Exception = None): ...
    def submit_bracket(self, intent: BracketIntent) -> ...: ...
```

---

## TDD Test Stubs by Section

Each subsection below lists the tests that should be written BEFORE implementing the corresponding section. These are the acceptance criteria -- implementation is correct when all tests pass.

### Section 1: psycopg3 Migration

**File**: `tests/unit/test_pg_connection.py`, `tests/integration/test_psycopg3_migration.py`

```python
# tests/unit/test_pg_connection.py

# Test: PgConnection wraps psycopg3 connection with same interface as psycopg2 version
# Test: execute() handles %s placeholders correctly
# Test: execute() handles ? placeholders via translation (backward compat)
# Test: fetchone() returns dict (not RealDictRow) -- verify key access works
# Test: fetchall() returns list[dict]
# Test: fetchdf() returns pandas DataFrame with correct column names
# Test: connection pool respects max_size (attempt max_size+1 blocks or raises)
# Test: idle connections recycled after max_idle seconds
# Test: OperationalError on execute triggers retry with fresh connection
# Test: JSON columns deserialized correctly (same behavior as psycopg2 custom adapter)
# Test: context manager returns connection to pool on exit
# Test: context manager returns connection to pool on exception
```

```python
# tests/integration/test_psycopg3_migration.py

# Test: ALL existing DB tests pass with psycopg3 driver (full pytest run)
# Test: No integer-indexed row access (grep-based lint check for row[0], row[1] patterns)
```

The lint check for integer-indexed row access should be a standalone script or test that greps the source tree for `row[0]`, `row[1]`, etc. in Python files that interact with the database. This catches the most common psycopg2-to-psycopg3 breakage (psycopg3 with `dict_row` does not support integer indexing).

### Section 2: Stop-Loss Enforcement

**Files**: `tests/unit/test_stop_loss_validation.py`, `tests/unit/test_bracket_orders.py`, `tests/integration/test_stop_loss_reconciliation.py`, `tests/integration/test_stop_loss_chaos.py`

```python
# tests/unit/test_stop_loss_validation.py

# Test: execute_trade() rejects when stop_price is None -- raises ValueError
# Test: execute_trade() accepts when stop_price is provided -- proceeds to risk gate
# Test: OMS submit() rejects entry orders without stop_price
# Test: OMS submit() allows exit/close orders without stop_price (exits don't need stops)
```

```python
# tests/unit/test_bracket_orders.py

# Test: AlpacaBroker.submit_bracket() uses native bracket API when available
# Test: AlpacaBroker.submit_bracket() falls back to entry+contingent SL when bracket fails
# Test: AlpacaBroker.submit_bracket() NEVER falls back to plain order
# Test: PaperBroker.submit_bracket() tracks SL/TP internally
# Test: EtradeBroker.submit_bracket() implements same interface
# Test: BracketIntent model validates required fields (stop_price mandatory)
# Test: client_order_id format includes millisecond precision and random suffix
# Test: client_order_id is deterministic given same inputs (idempotent retry)
```

```python
# tests/unit/test_contingent_sl.py

# Test: fill detected via WebSocket within 2s of broker fill
# Test: SL submitted immediately after fill detection
# Test: SL submission retried 3x with exponential backoff on failure
# Test: all 3 SL retries fail -> kill switch triggered for that symbol
# Test: partial fill + SL rejection -> remaining entry qty cancelled + standalone SL for filled qty
```

```python
# tests/unit/test_post_submission_verification.py

# Test: bracket leg verification runs 5s after submission
# Test: missing leg detected -> SL submitted immediately
# Test: rejected leg detected -> remaining legs cancelled + SL submitted
# Test: verification re-runs every 30s while position open
```

```python
# tests/integration/test_stop_loss_reconciliation.py

# Test: position with active stop order -> no action taken
# Test: position without stop order -> ATR-based SL submitted automatically
# Test: reconciliation logs warning for each auto-fixed position
# Test: trading proceeds after auto-fix (not halted)
# Test: reconciliation runs before first graph cycle
```

```python
# tests/unit/test_bracket_leg_persistence.py

# Test: bracket_legs table created with correct schema
# Test: bracket leg IDs persisted on bracket order submission
# Test: bracket legs queryable after process restart (DB persistence)
```

```python
# tests/integration/test_stop_loss_chaos.py
# (Chaos / failure injection tests)

# Test: broker API degraded -> new entries blocked
# Test: broker API degraded -> SL retries continue with exponential backoff
# Test: broker API recovers -> entries resume
# Test: broker returns HTTP 500 three times during contingent SL fallback ->
#        verify SL eventually placed or kill switch triggered
```

The chaos tests use the `FaultyBroker` wrapper described above to simulate broker API failures at precise points in the bracket order flow.

### Section 3: Prompt Injection Defense

**Files**: `tests/unit/test_prompt_safety.py`, `tests/unit/test_injection_detection.py`, `tests/unit/test_dual_llm_separation.py`, `tests/regression/test_prompt_migration.py`

```python
# tests/unit/test_prompt_safety.py

# Test: safe_prompt() wraps field values in XML tags
# Test: safe_prompt() replaces {field_name} placeholders correctly
# Test: safe_prompt() with missing field raises KeyError (not silent empty)
# Test: template output matches expected format with tags
# Test: MarketDataResponse extracts only typed fields from raw API response
# Test: extra/unexpected fields in raw response are discarded (not passed to prompt)
# Test: malicious content in raw response does not appear in extracted fields
# Test: Pydantic validation rejects response missing required fields
```

```python
# tests/unit/test_injection_detection.py

# Test: detect_injection() flags "ignore previous instructions"
# Test: detect_injection() flags "system:", "assistant:", "human:" prefixes
# Test: detect_injection() flags XML/HTML tags in data fields
# Test: detect_injection() returns detection details for logging (not just bool)
# Test: detect_injection() on clean data returns no findings
# Test: detection events are logged (verify log output)
```

```python
# tests/unit/test_dual_llm_separation.py

# Test: research agent config has NO execution tools in tool categories
# Test: trading agent config has execution tools
# Test: research agent cannot invoke execute_trade tool (tool resolution fails)
# Test: research agent CAN invoke data/analysis tools
# Test: graph node code mediates: research output -> structured data -> trading input
```

```python
# tests/regression/test_prompt_migration.py

# Test: each migrated prompt produces functionally equivalent output to f-string version
# Test: research graph prompts migrated first (highest priority)
```

### Section 4: Output Schema Validation with Retry

**Files**: `tests/unit/test_output_schemas.py`, `tests/unit/test_parse_and_validate.py`, `tests/integration/test_dead_letter_queue.py`

```python
# tests/unit/test_output_schemas.py

# Test: safety_check fallback is {"halted": True} (fail-CLOSED, not fail-OPEN)
# Test: risk assessment fallback rejects (not approves)
# Test: entry_scan fallback is [] (no entries -- safe)
# Test: ALL 21 agent fallbacks documented and classified as fail-safe
# Test: MarketIntelOutput validates known-good agent output sample
# Test: MarketIntelOutput rejects output missing required fields
# Test: EntrySignalOutput validates known-good sample
# Test: (repeat for each of 21 models with representative samples)
# Test: each model's JSON schema is serializable (for retry prompt inclusion)
```

```python
# tests/unit/test_parse_and_validate.py

# Test: valid JSON + valid schema -> parsed and validated correctly
# Test: valid JSON + invalid schema -> retry triggered with schema hint
# Test: invalid JSON -> retry triggered
# Test: retry succeeds -> result returned, flagged as retried in audit trail
# Test: retry fails -> dead letter queue entry created, fail-safe fallback returned
# Test: retried output has "retried" flag in audit trail
```

```python
# tests/integration/test_dead_letter_queue.py

# Test: agent_dead_letters table created with correct schema
# Test: DLQ entry contains agent_name, cycle_id, raw_output, parse_error, retry_attempted
# Test: DLQ queryable by agent_name and time range
# Test: supervisor health check flags agent with >10% DLQ rate
```

### Section 5: Non-Root Containers

**File**: `tests/integration/test_nonroot_containers.py`

```python
# tests/integration/test_nonroot_containers.py

# Test: container runs as non-root user (docker exec whoami -> "quantstack")
# Test: application writes to /app/logs successfully
# Test: kill switch writes sentinel file successfully
# Test: all services pass health checks after rebuild
# Test: init: true prevents zombie process accumulation
```

These tests require Docker and are tagged `@pytest.mark.integration`. They build the image, run the container, and assert on the runtime environment.

### Section 6: Durable Checkpoints (PostgresSaver)

**Files**: `tests/unit/test_checkpointer_factory.py`, `tests/integration/test_checkpoint_recovery.py`, `tests/integration/test_checkpoint_retention.py`

```python
# tests/unit/test_checkpointer_factory.py

# Test: create_checkpointer() returns configured PostgresSaver
# Test: checkpointer uses ConnectionPool (not single connection)
# Test: pool sizing: min_size=2, max_size=6
```

```python
# tests/integration/test_checkpoint_recovery.py

# Test: trading_runner uses PostgresSaver (not MemorySaver)
# Test: research_runner uses PostgresSaver
# Test: supervisor_runner uses PostgresSaver
# Test: thread_id format unchanged: {graph_name}-{date}-cycle-{number}
# Test: kill container mid-cycle -> restart -> graph resumes from last checkpoint
# Test: node that crashed mid-execution re-executes on resume
# Test: completed nodes are NOT re-executed on resume
```

```python
# tests/integration/test_checkpoint_retention.py

# Test: checkpoint pruning removes rows older than 48 hours
# Test: pruning preserves most recent completed cycle per graph
# Test: pruning preserves in-progress cycles
# Test: pruning runs as scheduled job without errors
```

The crash recovery tests are marked `@pytest.mark.slow` -- they involve starting a graph, killing it mid-cycle, restarting, and verifying state continuity.

### Section 7: EventBus Integration

**Files**: `tests/unit/test_kill_switch_eventbus.py`, `tests/coordination/test_eventbus_polling.py`, `tests/coordination/test_kill_switch_e2e.py`

```python
# tests/unit/test_kill_switch_eventbus.py

# Test: kill_switch.trigger() publishes KILL_SWITCH_TRIGGERED event to EventBus
# Test: event payload contains reason and timestamp
# Test: EventBus publication failure does NOT prevent kill switch activation
# Test: EventBus publication failure is logged as warning
# Test: KILL_SWITCH_TRIGGERED added to EventType enum
```

```python
# tests/coordination/test_eventbus_polling.py

# Test: safety_check polls for KILL_SWITCH_TRIGGERED, RISK_EMERGENCY, IC_DECAY, REGIME_CHANGE
# Test: KILL_SWITCH_TRIGGERED at safety_check -> cycle halted
# Test: IC_DECAY at safety_check -> affected strategy suspended
# Test: pre-execute_entries polls for KILL_SWITCH_TRIGGERED, RISK_EMERGENCY
# Test: KILL_SWITCH_TRIGGERED before entries -> entries skipped, go to reflect
# Test: pre-execute_exits polls for KILL_SWITCH_TRIGGERED
# Test: KILL_SWITCH_TRIGGERED before exits -> emergency close-all
# Test: cursor advances after polling regardless of action taken (idempotent)
# Test: research_runner polls KILL_SWITCH_TRIGGERED at cycle start
# Test: supervisor_runner polls KILL_SWITCH_TRIGGERED at cycle start
```

```python
# tests/coordination/test_kill_switch_e2e.py
# (End-to-end kill switch propagation -- the single most critical safety test)

# Test: trigger kill switch ->
#        sentinel file written + EventBus published ->
#        all 3 graphs halt within one cycle ->
#        execution monitor stops ->
#        position closer fires
```

The end-to-end test is the crown jewel of the testing strategy. It validates the full propagation path from kill switch trigger through every downstream consumer. It requires all three graph runners, the execution monitor, and PostgreSQL. Tagged `@pytest.mark.slow` and `@pytest.mark.integration`.

### Section 8: Database Backups

**Files**: `tests/unit/test_backup_script.py`, `tests/integration/test_backup_restore.py`

```python
# tests/unit/test_backup_script.py

# Test: pg_dump creates valid backup file
# Test: pg_restore --list succeeds on backup file (integrity check)
# Test: backup script exits non-zero on pg_dump failure
# Test: backups older than 30 days are deleted
# Test: WAL archive files older than 7 days are pruned
# Test: backup script uses flock to prevent concurrent runs
```

```python
# tests/integration/test_backup_restore.py

# Test: full restore from pg_dump -> all tables intact with correct row counts
# Test: PITR restore from WAL -> data consistent to target timestamp
# Test: supervisor health check detects backup older than 36 hours
# Test: supervisor raises warning event for stale backup
```

### Section 9: Containerize Scheduler

**Files**: `tests/unit/test_scheduler_imports.py`, `tests/integration/test_scheduler_container.py`

```python
# tests/unit/test_scheduler_imports.py

# Test: "from quantstack.runners import scheduler" succeeds without ibkr_mcp
# Test: "python scripts/scheduler.py --dry-run" runs without import errors
```

```python
# tests/integration/test_scheduler_container.py

# Test: scheduler container starts and health check passes
# Test: scheduler container auto-restarts after kill (within 60s)
# Test: APScheduler has all 5 jobs registered
# Test: health endpoint returns job list and next_run times
# Test: SIGTERM triggers clean APScheduler shutdown
```

### Section 10: Transaction Isolation

**Files**: `tests/unit/test_row_locking.py`, `tests/integration/test_concurrent_position_updates.py`, `tests/integration/test_write_path_coverage.py`

```python
# tests/unit/test_row_locking.py

# Test: SELECT FOR UPDATE acquires row lock on position
# Test: second writer blocks until first commits
# Test: lock_timeout fires after 5s -> retry once after 500ms
# Test: second retry timeout -> CRITICAL log, operation continues with stale data
# Test: reader not blocked by writer (MVCC verification)
# Test: single-row constraint: transaction only locks one position row at a time
```

```python
# tests/integration/test_concurrent_position_updates.py

# Test: two concurrent updates on same symbol -> no lost writes
# Test: execution monitor + trading graph race on same position -> both updates applied
# Test: N concurrent writers stress test -> acceptable latency, no lost updates
```

```python
# tests/integration/test_write_path_coverage.py

# Test: alpaca_broker.py fill handler uses locking pattern
# Test: trade_service.py metadata update uses locking pattern
# Test: execution_monitor.py trailing stop update uses locking pattern
# Test: startup reconciliation uses locking pattern
# Test: kill_switch position closer uses locking pattern
```

---

## Cross-Section Integration Tests

Beyond individual section tests, these integration tests validate that sections work together correctly.

### Kill Switch Full Propagation (Sections 2, 7, 10)

This is the highest-priority integration test. It validates the complete kill switch propagation chain:

1. Trigger the kill switch with a reason string
2. Assert sentinel file is written to disk
3. Assert `KILL_SWITCH_TRIGGERED` event published to EventBus
4. Assert trading graph halts at next `safety_check` poll
5. Assert research graph halts at next cycle start poll
6. Assert supervisor graph halts at next cycle start poll
7. Assert execution monitor stops evaluating exits
8. Assert position closer callback fires and attempts to close all open positions
9. Assert position close attempts use the locking pattern from Section 10

This test exercises kill_switch.py, event_bus.py, all three graph runners, execution_monitor.py, and the position update locking. If this test passes, the system's most critical safety mechanism is verified end-to-end.

### Stop-Loss Under Broker Failure (Sections 2, 7)

Chaos test combining bracket order failure with kill switch integration:

1. Submit a bracket order via `submit_bracket()`
2. Inject broker failure (HTTP 500) on the SL leg
3. Assert contingent SL path activates with 3 retries
4. After 3 failures, assert kill switch triggers for that symbol
5. Assert kill switch event propagates via EventBus
6. Assert no further entries attempted for the affected symbol

### Checkpoint Recovery After Stop-Loss Reconciliation (Sections 2, 6)

Validates that crash recovery interacts correctly with startup reconciliation:

1. Start trading runner with a position that has no stop order
2. Kill the runner mid-reconciliation (after detecting the missing stop but before submitting it)
3. Restart the runner
4. Assert reconciliation re-runs (checkpoint did not mark it complete)
5. Assert the missing stop is eventually submitted

### Output Validation with EventBus (Sections 4, 7)

Validates that parse failures in safety-critical agents trigger appropriate events:

1. Force a `safety_check` agent to produce unparseable output
2. Assert `parse_and_validate()` retries once
3. Assert retry also fails (DLQ entry created)
4. Assert fail-CLOSED fallback returns `{"halted": True}`
5. Assert the trading cycle halts (same behavior as an explicit kill switch)

---

## Stress and Chaos Tests

### Concurrent Writer Stress Test (Section 10)

**File**: `tests/benchmarks/test_position_contention.py`

Spawn N threads (start with N=10, scale to N=50) that all attempt to update the same position row simultaneously. Each thread reads the current `quantity`, adds 1, and writes it back -- inside a `SELECT FOR UPDATE` transaction.

Assertions:
- Final quantity equals initial + N (no lost writes)
- No deadlocks (single-row constraint prevents them)
- p99 latency for a single update stays below 500ms at N=10
- All threads complete (no hangs from unbounded lock waits -- 5s timeout catches them)

Mark as `@pytest.mark.slow` and `@pytest.mark.benchmark`.

### Bracket Order Chaos Matrix (Section 2)

Test the bracket order flow against a matrix of failure scenarios:

| Scenario | Entry | SL Leg | TP Leg | Expected Outcome |
|----------|-------|--------|--------|------------------|
| Happy path | Fills | Accepted | Accepted | Position with SL+TP |
| SL rejected | Fills | Rejected | Accepted | Contingent SL path activates |
| All legs rejected | Rejected | N/A | N/A | No position opened |
| Partial fill + SL fail | Partial | Rejected | N/A | Cancel remaining, SL for filled qty |
| Broker timeout on SL | Fills | Timeout | Accepted | 3 retries, then kill switch |
| Entry fills, broker down | Fills | Down | Down | 3 retries, then kill switch |

Each row is a parameterized test case using `FaultyBroker` with different failure configurations.

---

## Test Execution Plan

### During Development (per-section)

When implementing each section, the developer runs only that section's tests:

```bash
# Example: working on Section 2
uv run pytest tests/unit/test_stop_loss_validation.py tests/unit/test_bracket_orders.py -v

# After implementation, run the integration tests
uv run pytest tests/integration/test_stop_loss_reconciliation.py -v -m integration
```

### Pre-Merge (CI)

```bash
# Fast suite: unit + non-slow integration
uv run pytest -m "not slow and not requires_api and not requires_gpu" --tb=short

# Expected runtime: < 5 minutes
```

### Nightly / Pre-Deploy

```bash
# Full suite including slow, stress, and chaos tests
uv run pytest --tb=long -v

# Expected runtime: 15-30 minutes (dominated by checkpoint crash recovery and stress tests)
```

### Post-Deploy Smoke Test

After deploying to the Docker environment, run a targeted smoke test:

```bash
# Verify critical paths in the live environment
uv run pytest tests/coordination/test_kill_switch_e2e.py tests/integration/test_nonroot_containers.py -v -m integration
```

---

## Key Files to Create or Modify

**New test files:**
- `tests/unit/test_pg_connection.py`
- `tests/unit/test_stop_loss_validation.py`
- `tests/unit/test_bracket_orders.py`
- `tests/unit/test_contingent_sl.py`
- `tests/unit/test_post_submission_verification.py`
- `tests/unit/test_bracket_leg_persistence.py`
- `tests/unit/test_prompt_safety.py`
- `tests/unit/test_injection_detection.py`
- `tests/unit/test_dual_llm_separation.py`
- `tests/unit/test_output_schemas.py`
- `tests/unit/test_parse_and_validate.py`
- `tests/unit/test_checkpointer_factory.py`
- `tests/unit/test_kill_switch_eventbus.py`
- `tests/unit/test_backup_script.py`
- `tests/unit/test_row_locking.py`
- `tests/unit/test_scheduler_imports.py`
- `tests/integration/test_psycopg3_migration.py`
- `tests/integration/test_stop_loss_reconciliation.py`
- `tests/integration/test_stop_loss_chaos.py`
- `tests/integration/test_dead_letter_queue.py`
- `tests/integration/test_nonroot_containers.py`
- `tests/integration/test_checkpoint_recovery.py`
- `tests/integration/test_checkpoint_retention.py`
- `tests/integration/test_backup_restore.py`
- `tests/integration/test_scheduler_container.py`
- `tests/integration/test_concurrent_position_updates.py`
- `tests/integration/test_write_path_coverage.py`
- `tests/coordination/test_eventbus_polling.py`
- `tests/coordination/test_kill_switch_e2e.py`
- `tests/regression/test_prompt_migration.py`
- `tests/benchmarks/test_position_contention.py`

**New test helpers:**
- `tests/helpers/faulty_broker.py` -- `FaultyBroker` wrapper for chaos/failure injection

**Modified:**
- `tests/conftest.py` -- update DB fixtures for psycopg3, add `FaultyBroker` fixture
