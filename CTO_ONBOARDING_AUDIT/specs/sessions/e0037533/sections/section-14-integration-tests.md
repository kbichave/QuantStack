# Section 14: Integration Tests

## Background

Sections 05 through 13 each introduce a subsystem -- error blocking, race condition resolution, circuit breakers, tool access control, event bus cursors, dead letter queue, message pruning, pre-trade risk checks, and regime flip handling. Each has its own unit tests that validate behavior in isolation.

Integration tests verify that these subsystems interact correctly when composed inside a running trading graph cycle. The failure modes that matter most are the cross-cutting ones: a circuit breaker tripping on a blocking node must propagate through the execution gate to halt the pipeline; parallel branches producing conflicting symbols must be resolved before risk sizing; breaker state must persist across graph invocations so a tripped node stays open in the next cycle.

These tests run against a wired-up trading graph (or a representative subset) with mocked LLM calls and a test PostgreSQL database. They exercise the real graph edges, real state merging, and real conditional routing -- only the external I/O (LLM providers, brokers, data feeds) is stubbed.

## Dependencies

All of sections 05 through 13 must be complete before these tests can run:

- **section-05-error-blocking**: Node classification, execution gate conditional edge, safe defaults
- **section-06-race-condition-fix**: `resolve_symbol_conflicts` node between merge and risk sizing
- **section-07-circuit-breaker**: `@circuit_breaker` decorator, DB-backed state, safe defaults
- **section-08-tool-access-control**: `blocked_tools` guard in agent executor
- **section-09-event-bus-cursor**: Upsert-based cursor updates
- **section-10-dead-letter-queue**: DLQ writes on parse failure
- **section-11-message-pruning**: Priority-based pruning with compaction
- **section-12-risk-gate-pretrade**: Correlation, heat budget, and sector concentration checks
- **section-13-regime-flip**: `regime_at_entry` storage, regime comparison, stop tightening / auto-exit

The DB migration from **section-01** must also be applied (tables `circuit_breaker_state`, `agent_dlq`, `regime_at_entry` column, `loop_cursors` UNIQUE constraint).

## Tests First

All tests live in `tests/integration/test_trading_graph_phase4.py`.

### Test Infrastructure

The integration tests need a shared fixture pattern. Key fixtures:

- **test database**: A real PostgreSQL instance (or testcontainers-based ephemeral DB) with the section-01 migration applied. Each test gets a clean transaction that rolls back after the test.
- **mock LLM**: A `FakeChatModel` (or `AsyncMock` wrapping `ChatModel`) that returns configurable responses per agent. This already exists in the test suite as a pattern (see `tests/unit/test_config.py` for `ConfigWatcher` fixtures and mock ChatModel factories).
- **graph builder**: A function that constructs the full trading graph with all phase-4 wiring (conflict resolution node, execution gate edges, circuit breaker decorators) but with mocked LLM and data dependencies.
- **state factory**: A helper that produces valid `TradingState` instances with realistic field values, using the Pydantic models from section-03.

```python
# Fixture: pg_test_db -- ephemeral PostgreSQL with phase-4 migration applied, transaction-scoped
# Fixture: mock_llm_factory -- returns FakeChatModel with configurable per-agent responses
# Fixture: trading_graph -- fully wired trading graph with mocked externals
# Fixture: valid_trading_state -- factory producing TradingState with sensible defaults
```

### Test 1: Full Cycle with Conflicting Symbols

Verifies the end-to-end path: parallel branches produce overlapping symbols, `resolve_symbol_conflicts` removes the entry, and execution proceeds with exits only for conflicted symbols.

```python
# Test: full trading graph cycle -- parallel branches produce conflicting symbols
#       -> conflict resolved -> execution proceeds
#
# Setup:
#   - position_review returns exit_orders for ["AAPL", "TSLA"]
#   - entry_scan returns entry_candidates for ["AAPL", "NVDA"]
#   - AAPL appears in both lists (conflict)
#
# Assertions:
#   - After resolve_symbol_conflicts: entry_candidates contains only NVDA
#   - exit_orders still contains both AAPL and TSLA (unchanged)
#   - Conflict event logged with symbol="AAPL", resolution="entry_dropped"
#   - Pipeline reaches risk_sizing (no execution gate halt)
#   - risk_sizing receives the cleaned candidate list
```

### Test 2: Blocking Node Failure Halts Pipeline

Verifies the interaction between error blocking (section-05) and the execution gate: a blocking node error prevents the pipeline from reaching trade execution.

```python
# Test: full cycle -- blocking node failure -> execution gate halts -> clean cycle termination
#
# Setup:
#   - data_refresh node configured to raise an exception (simulate data feed failure)
#   - All other nodes return valid outputs
#
# Assertions:
#   - data_refresh error appended to state.errors with blocking classification
#   - Execution gate conditional edge routes to cycle termination (not risk_sizing)
#   - execute_entries node is never invoked
#   - execute_exits node is never invoked
#   - Cycle terminates cleanly (no unhandled exception, state is valid)
#   - Error reason logged: "blocking node data_refresh failed"
```

### Test 3: Circuit Breaker Trips on Blocking Node, Execution Gate Halts

Verifies the cross-cutting interaction between the circuit breaker (section-07) and the execution gate (section-05): when a breaker trips on a blocking node, the safe default includes an error flag, and the execution gate detects it.

```python
# Test: circuit breaker trips on blocking node -> safe default sets error
#       -> execution gate halts -> graceful end
#
# Setup:
#   - data_refresh breaker pre-loaded in DB as state="open" (already tripped)
#   - All other nodes return valid outputs
#
# Assertions:
#   - data_refresh node function is NOT called (breaker is open, node skipped)
#   - data_refresh safe default returned (DataRefreshOutput.safe_default())
#   - Safe default includes error flag in state.errors
#   - Execution gate detects blocking-node error and halts pipeline
#   - execute_entries is never invoked
#   - Langfuse trace includes breaker-skip event for data_refresh
```

### Test 4: Circuit Breaker State Persists Across Graph Invocations

Verifies that breaker state in PostgreSQL survives across separate graph invocations -- a node that tripped in cycle N remains open in cycle N+1 (assuming cooldown has not expired).

```python
# Test: circuit breaker state persists across graph invocations
#       -> node stays open in next cycle
#
# Setup:
#   - Cycle 1: data_refresh fails 3 times consecutively (threshold met, breaker opens)
#   - Cycle 2: invoked immediately (within cooldown window of 300s)
#
# Assertions:
#   - Cycle 1: breaker transitions closed -> open after 3rd failure
#   - Cycle 1: DB row for "trading/data_refresh" shows state="open"
#   - Cycle 2: data_refresh skipped without calling the node function
#   - Cycle 2: safe default returned, execution gate halts (blocking node)
#   - DB failure_count remains at 3 (not incremented further while open)
#
# Cleanup:
#   - Reset breaker state in DB after test
```

### Test 5: Non-Blocking Failure with Conflict Resolution

Verifies that a non-blocking node failure does not halt the pipeline and that conflict resolution still operates correctly on the remaining valid data.

```python
# Test: non-blocking node failure + symbol conflict in same cycle
#       -> safe default used, conflict still resolved, pipeline continues
#
# Setup:
#   - entry_scan (non-blocking) fails, safe default returns empty candidate list
#   - position_review returns exit_orders for ["AAPL"]
#
# Assertions:
#   - entry_scan error in state.errors but not classified as blocking
#   - resolve_symbol_conflicts receives empty entry_candidates (no conflict possible)
#   - Execution gate allows pipeline to proceed (no blocking errors, error count <= 2)
#   - risk_sizing receives empty candidate list (no entries attempted)
#   - Exit orders processed normally
```

### Test 6: Multiple Error Sources Trigger Safety Net

Verifies the total-error-count safety net: even if all errors come from non-blocking nodes, exceeding the threshold (>2) halts the pipeline.

```python
# Test: 3 non-blocking node errors -> total error count safety net halts pipeline
#
# Setup:
#   - plan_day fails (non-blocking, safe default used)
#   - entry_scan fails (non-blocking, safe default used)
#   - market_intel fails (non-blocking, safe default used)
#   - All blocking nodes succeed
#
# Assertions:
#   - state.errors contains 3 entries (one per failed non-blocking node)
#   - Execution gate detects error count > 2 and halts pipeline
#   - execute_entries never invoked despite no blocking failures
```

## Implementation Details

### File Location

`tests/integration/test_trading_graph_phase4.py`

This file lives alongside the existing `tests/integration/__init__.py`. It contains all 6 integration tests described above plus the shared fixtures.

### Graph Construction for Tests

The test graph should be built using the same builder function that production uses (`src/quantstack/graphs/trading/graph.py`), with dependency injection for:

1. **LLM provider**: Replaced with `FakeChatModel` that returns predetermined JSON responses per agent
2. **Database connection**: Pointed at the test PostgreSQL instance
3. **Data feeds**: Mocked to return fixed OHLCV data (sufficient for correlation checks, ATR calculations)
4. **Broker client**: Mocked Alpaca client that records order submissions without executing

The goal is to exercise the real graph wiring (edges, conditional routing, state merging) while controlling all external dependencies. If the production graph builder does not currently support dependency injection, the test setup should patch at the narrowest possible point (e.g., mock the provider factory, not the entire node function).

### State Setup Patterns

Each test constructs initial state using the Pydantic `TradingState` model (section-03). Common patterns:

- **Conflicting symbols**: Set `exit_orders` and `entry_candidates` to lists with overlapping symbol values
- **Blocking failure**: Patch the target node function to raise a specific exception type
- **Breaker pre-load**: Insert a row into `circuit_breaker_state` with `state='open'` and `opened_at` within cooldown window
- **Error accumulation**: Patch multiple non-blocking nodes to raise, relying on safe defaults to populate `state.errors`

### Assertion Patterns

Integration tests should assert on observable outcomes, not internal implementation:

- **State fields after cycle**: Check `state.errors`, `state.exit_orders`, `state.entry_candidates` after the graph completes
- **Node invocation**: Use `unittest.mock.patch` to wrap node functions and assert `call_count` (was the node called or skipped?)
- **DB state**: Query `circuit_breaker_state` table directly to verify persistence
- **Log/trace events**: If Langfuse is mocked, assert that expected events were emitted (breaker skip, conflict resolution, execution gate halt)
- **Order submissions**: Check the mock broker's recorded orders to verify only expected trades were attempted

### Running the Tests

```bash
# Run only the phase-4 integration tests
uv run pytest tests/integration/test_trading_graph_phase4.py -v

# Run with the rest of the integration suite
uv run pytest tests/integration/ -v
```

The tests require a running PostgreSQL instance. In CI, use `testcontainers` or a Docker-based PostgreSQL service. Locally, the same `quantstack` database can be used with a test schema prefix or transaction rollback isolation.

### What These Tests Do NOT Cover

These integration tests focus on the trading graph's internal coordination. The following are explicitly out of scope and covered by their respective section's unit tests:

- Tool access control enforcement (section-08 unit tests)
- Event bus cursor atomicity (section-09 unit tests)
- DLQ write mechanics (section-10 unit tests)
- Message pruning priority logic (section-11 unit tests)
- Risk gate threshold math for correlation, heat, and sector checks (section-12 unit tests)
- Regime flip stop-tightening arithmetic (section-13 unit tests)

Integration tests verify that these subsystems compose correctly inside the graph, not that their individual logic is correct.
