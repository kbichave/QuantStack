# Section 13: Testing Strategy

## Dependencies

- **Section 09 (Runners):** Runner modules must exist so E2E and shutdown tests can exercise the full loop.
- **Section 11 (Memory Migration):** RAG ingestion must be implemented so RAG degradation tests can verify graceful fallback.
- **Section 12 (Risk Safety):** Programmatic safety boundary must be in place so risk-related tests can validate the defense-in-depth envelope.

All existing tests in `tests/unit/` remain untouched. This section adds new test modules for the CrewAI migration layer.

---

## What This Section Covers

Five concrete test scenarios plus a soak-test specification and a 48-hour verification phase configuration. The tests validate that the CrewAI orchestration layer works end-to-end, degrades gracefully under failure, and shuts down cleanly.

**New test files to create:**

| File | Category | Purpose |
|------|----------|---------|
| `tests/integration/test_e2e_smoke.py` | Integration | One full TradingCrew cycle with mock LLM |
| `tests/integration/test_provider_fallback.py` | Integration | Primary provider failure triggers fallback |
| `tests/integration/test_graceful_shutdown.py` | Integration | SIGTERM persists state and exits cleanly |
| `tests/unit/test_watchdog.py` | Unit | Watchdog triggers after timeout |
| `tests/integration/test_rag_degradation.py` | Integration | Crews survive ChromaDB being unavailable |
| `tests/integration/conftest.py` | Fixtures | Shared fixtures for integration tests |

---

## Test Fixtures

### Mock LLM Provider

All integration tests need a mock LLM that returns deterministic responses without making API calls. Create a shared fixture in `tests/integration/conftest.py`.

The fixture should:
- Patch the LLM provider's `get_model()` to return a fake model identifier.
- Patch CrewAI's LLM call path so that agent reasoning returns canned JSON responses (valid structured output matching what real agents produce).
- Provide a factory function `mock_llm_response(task_name: str) -> str` that returns appropriate canned output per task (e.g., the `safety_check` task returns `{"status": "ok", "kill_switch": false}`; the `risk_sizing` task returns `{"symbol": "AAPL", "recommended_size_pct": 5, "reasoning": "test"}`).

### Test Database

Reuse the existing test PostgreSQL fixture from `tests/conftest.py` (or `tests/unit/conftest.py`). Extend it if needed to seed the tables that runners read on startup (strategies, positions, heartbeats).

### In-Memory ChromaDB

For tests that need RAG, use `chromadb.Client()` (ephemeral in-memory mode) instead of `chromadb.HttpClient()`. Patch the ChromaDB client factory in `src/quantstack/rag/` to return the in-memory client during tests.

---

## Test 1: E2E Smoke Test

**File:** `tests/integration/test_e2e_smoke.py`

**Purpose:** Run one complete TradingCrew cycle with mock LLM and test database. Verify the full 11-task sequential workflow completes without exceptions.

**What it validates:**
- All 11 TradingCrew tasks execute in order (safety_check through persist_state).
- Heartbeat file is written after the cycle.
- At least one audit log entry is created in the database.
- No unhandled exceptions propagate.

**Test outline:**

```python
def test_e2e_smoke_one_trading_cycle(mock_llm, test_db, in_memory_chromadb):
    """One full TradingCrew cycle completes with mock LLM and test DB.

    Arrange: seed test DB with a portfolio state and one open position.
    Act: call create_trading_crew() and crew.kickoff() with mocked inputs.
    Assert:
      - kickoff returns without raising
      - heartbeat file exists at /tmp/trading-heartbeat
      - DB contains audit_log entry for this cycle
      - all 11 task results are present in the crew output
    """
```

**Key implementation notes:**
- The mock LLM must return valid JSON for every task. If any task's expected_output format is wrong, the subsequent task will fail because CrewAI passes outputs as context.
- Seed the test DB with at least one open position (so position_review has something to review) and at least one strategy in `forward_testing` status.
- Set `EXECUTION_ENABLED=false` so execute_entries and execute_exits are no-ops even if the mock produces trade instructions.
- Timeout the test at 120 seconds. If the mock LLM is synchronous, this should complete in under 10 seconds.

---

## Test 2: Provider Fallback Test

**File:** `tests/integration/test_provider_fallback.py`

**Purpose:** Verify that when the primary LLM provider raises an error, the fallback chain activates and the operation succeeds with the secondary provider.

**What it validates:**
- Primary provider failure is caught (not propagated).
- Secondary provider is tried automatically.
- The final result is valid (came from the fallback provider).
- A Langfuse trace (or log entry) records the failover event.

**Test outline:**

```python
def test_provider_fallback_on_primary_failure(monkeypatch):
    """Fallback chain activates when primary provider raises.

    Arrange: configure LLM_PROVIDER=bedrock. Patch bedrock call to raise
    a connection error. Patch anthropic call to return a valid response.
    Act: call get_model("heavy") and invoke an LLM operation.
    Assert:
      - no exception raised
      - response came from anthropic provider
      - failover event was logged
    """
```

```python
def test_all_providers_fail_raises(monkeypatch):
    """When every provider in the chain fails, raise after exhausting all.

    Arrange: patch all 4 providers to raise errors.
    Act: attempt an LLM call.
    Assert: raises a specific exception (e.g., AllProvidersFailedError)
    with details of each provider's failure.
    """
```

**Key implementation notes:**
- The fallback chain order is: bedrock -> anthropic -> openai -> ollama. Test that the order is respected.
- The provider module (`src/quantstack/llm/provider.py`) must expose the fallback logic in a way that is testable without starting a full crew.

---

## Test 3: Graceful Shutdown Test

**File:** `tests/integration/test_graceful_shutdown.py`

**Purpose:** Send SIGTERM to a running runner process and verify that it persists state, flushes Langfuse traces, and exits cleanly within the grace period.

**What it validates:**
- The `should_stop` flag is set on SIGTERM.
- The runner completes its current cycle (does not abort mid-task).
- State is persisted to PostgreSQL before exit.
- Langfuse `flush()` is called.
- The process exits with code 0 within 60 seconds of receiving SIGTERM.

**Test outline:**

```python
def test_graceful_shutdown_on_sigterm(mock_llm, test_db):
    """SIGTERM triggers clean shutdown with state persistence.

    Arrange: start the trading runner in a subprocess (or thread).
    Wait until the first heartbeat file appears (confirms loop is running).
    Act: send SIGTERM to the process.
    Assert:
      - process exits within 60 seconds
      - exit code is 0
      - DB contains persisted checkpoint for this session
      - Langfuse flush was called (mock Langfuse client, check call count)
    """
```

```python
def test_graceful_shutdown_on_sigint(mock_llm, test_db):
    """SIGINT also triggers the same clean shutdown path.

    Same as SIGTERM test but sends SIGINT instead.
    """
```

**Key implementation notes:**
- Running the full runner in a subprocess is the cleanest approach. Use `subprocess.Popen` with a short cycle interval (1 second) and mock LLM.
- Alternative: test the `GracefulShutdown` handler in isolation by importing it and calling the signal handler directly, then asserting `should_stop` is True. This is faster and more reliable for CI.
- Both approaches should be implemented: the unit-level handler test for fast CI, and the subprocess test for thorough validation.

---

## Test 4: Watchdog Test

**File:** `tests/unit/test_watchdog.py`

**Purpose:** Verify that `AgentWatchdog` triggers its callback after the configured timeout, and does not trigger if the cycle completes in time.

**What it validates:**
- Watchdog fires callback after `timeout_seconds`.
- `end_cycle()` cancels the timer so it does not fire.
- Callback receives useful context (which crew, how long it was stuck).

**Test outline:**

```python
def test_watchdog_triggers_after_timeout():
    """Watchdog fires callback when cycle exceeds timeout.

    Arrange: create AgentWatchdog with timeout_seconds=1.
    Register a callback that sets a threading.Event.
    Act: call start_cycle(), then wait 2 seconds.
    Assert: the Event is set (callback fired).
    """
```

```python
def test_watchdog_does_not_trigger_if_cycle_completes():
    """Watchdog is cancelled when end_cycle() is called before timeout.

    Arrange: create AgentWatchdog with timeout_seconds=2.
    Act: call start_cycle(), wait 0.5s, call end_cycle().
    Wait another 2 seconds.
    Assert: callback was NOT called.
    """
```

```python
def test_watchdog_end_cycle_cancels_timer():
    """end_cycle() resets the watchdog for the next cycle.

    Arrange: create AgentWatchdog with timeout_seconds=1.
    Act: start_cycle(), end_cycle() immediately, start_cycle() again.
    Wait 0.5s, end_cycle() again.
    Assert: callback never fired across both cycles.
    """
```

**Key implementation notes:**
- The `AgentWatchdog` lives in `src/quantstack/health/watchdog.py`. It uses `threading.Timer` internally.
- Keep timeouts short in tests (1-2 seconds) to avoid slow test suites.
- Use `threading.Event` for synchronization rather than `time.sleep` with assertions -- it is more reliable and avoids flaky timing-dependent tests.

---

## Test 5: RAG Degradation Test

**File:** `tests/integration/test_rag_degradation.py`

**Purpose:** Verify that crews continue operating when ChromaDB is unavailable. The system should degrade gracefully (skip RAG queries, log warnings) rather than crash.

**What it validates:**
- When `search_knowledge_base_tool` fails due to ChromaDB being down, the tool returns an error message (not an exception).
- The crew cycle completes despite RAG failures.
- A warning/degradation event is logged.
- No data corruption occurs (the crew does not write garbage to the DB because RAG context was missing).

**Test outline:**

```python
def test_crew_continues_when_chromadb_unreachable(mock_llm, test_db):
    """TradingCrew completes a cycle when ChromaDB is down.

    Arrange: patch ChromaDB HttpClient to raise ConnectionError on any call.
    Act: run one TradingCrew cycle.
    Assert:
      - cycle completes without raising
      - heartbeat is written
      - a degradation warning is logged (check log output or DB event)
      - no RAG-dependent data was written to DB with empty/null context
    """
```

```python
def test_rag_tool_returns_error_json_on_failure():
    """search_knowledge_base_tool returns error JSON, not exception.

    Arrange: patch ChromaDB client to raise ConnectionError.
    Act: call search_knowledge_base_tool("test query").
    Assert:
      - returns a string (not raises)
      - returned string is valid JSON
      - JSON contains an "error" key describing the failure
    """
```

**Key implementation notes:**
- The RAG tool wrappers in `src/quantstack/crewai_tools/rag_tools.py` must catch ChromaDB connection errors and return error JSON. This is the pattern established in Section 03 (tool wrappers return error JSON on failure, never raise).
- The mock LLM needs to handle receiving error context from RAG tools gracefully. In practice, the mock just returns its canned response regardless of input, which is fine for this test.

---

## Soak Test Specification

The soak test is not an automated pytest test. It is a manual procedure run before enabling paper trading.

**Procedure:**
1. Start the full Docker Compose stack (`./start.sh`) with a test database and mock market data.
2. Let it run for 24 hours.
3. Monitor the following metrics (via Langfuse dashboard and direct queries):

| Metric | Acceptable Range | How to Check |
|--------|-----------------|--------------|
| Container memory usage | Stable, no upward trend | `docker stats --no-stream` sampled every hour |
| DB connection pool size | Within configured bounds (default: 10) | `SELECT count(*) FROM pg_stat_activity WHERE datname='quantstack'` |
| Heartbeat gaps | Zero gaps > 2x cycle interval | Query heartbeat table for timestamp deltas |
| ChromaDB index size | Grows linearly, not exponentially | `docker exec chromadb du -sh /chroma/chroma` |
| Langfuse trace count | Matches expected cycle count | Langfuse dashboard |
| Python process RSS | Stable (< 1GB per crew) | `docker stats` |

4. If any metric drifts outside acceptable range, investigate and fix before proceeding.

---

## 48-Hour Verification Phase Configuration

Before enabling real paper trading, the system runs in verification mode. This is a runtime configuration, not a separate test suite.

**Configuration:**

Set in `.env`:
```
EXECUTION_ENABLED=false
```

**What this controls:**
- `execute_trade_tool` and `close_position_tool` in `src/quantstack/crewai_tools/execution_tools.py` check this env var. When false, they log the intended action (symbol, side, quantity, reasoning) to both the database audit table and Langfuse, but do not submit orders to Alpaca.
- All other tools work normally: signals, backtests, ML, RAG, data fetching, portfolio queries (read-only).
- Crews run their full reasoning cycles, producing decisions that are logged but not executed.

**Verification checklist (after 48 hours, review in Langfuse):**

| Check | What to Look For | Fail Criteria |
|-------|-----------------|---------------|
| Risk decisions | Position sizes reasonable (1-10% typical) | Any recommendation > 15% (safety gate would catch, but indicates LLM miscalibration) |
| Strategy evaluations | Reasoning cites evidence from RAG and backtest data | Reasoning is generic boilerplate, ignores context |
| Self-healing | Provoke a failure (stop Ollama briefly) and verify recovery | System crashes or does not recover within 2 cycles |
| Cost | Daily LLM cost within $5-10 range | Cost > $20/day indicates runaway loops or excessive retries |
| Cycle consistency | All 3 crews complete cycles on schedule | Any crew has > 3 consecutive failed cycles |
| Memory stability | No crew process exceeds 1GB RSS | Upward memory trend over 48 hours |

**Enabling execution:**

After verification passes, set `EXECUTION_ENABLED=true` in `.env` and restart crew containers:
```bash
docker compose restart trading-crew research-crew supervisor-crew
```

The supervisor crew detects the configuration change on its next cycle and logs the transition event to Langfuse.

---

## Test Infrastructure Notes

**Running tests:**

```bash
# Unit tests only (fast, no external dependencies)
uv run pytest tests/unit/test_watchdog.py -v

# Integration tests (need test DB, slower)
uv run pytest tests/integration/ -v

# All tests
uv run pytest tests/ -v
```

**CI considerations:**
- Unit tests (watchdog) run in any environment.
- Integration tests require PostgreSQL. Use the existing test DB fixture or a disposable Docker Postgres.
- Integration tests do NOT require Ollama, ChromaDB, or Langfuse running -- those are all mocked.
- The soak test and verification phase are manual pre-production procedures, not CI steps.

**Existing tests are preserved.** The CrewAI migration does not modify any existing source files in `src/quantstack/mcp/tools/` or `src/quantstack/execution/`. All existing tests in `tests/unit/` continue to pass unchanged.
