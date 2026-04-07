# Phase 1: Safety Hardening — TDD Plan

Testing framework: **pytest** with markers (slow, integration, benchmark, requires_api, requires_gpu). Tests in `tests/{unit, integration, core, graphs, coordination, regression}`. Fixtures in `tests/conftest.py`. Async support via asyncio fixture.

---

## Section 1: psycopg3 Migration

### Before implementing the migration:

```python
# Test: PgConnection wraps psycopg3 connection with same interface as psycopg2 version
# Test: execute() handles %s placeholders correctly
# Test: execute() handles ? placeholders via translation (backward compat)
# Test: fetchone() returns dict (not RealDictRow) — verify key access works
# Test: fetchall() returns list[dict]
# Test: fetchdf() returns pandas DataFrame with correct column names
# Test: connection pool respects max_size (attempt max_size+1 → blocks or raises)
# Test: idle connections recycled after max_idle seconds
# Test: OperationalError on execute triggers retry with fresh connection
# Test: JSON columns deserialized correctly (same behavior as psycopg2 custom adapter)
# Test: context manager returns connection to pool on exit
# Test: context manager returns connection to pool on exception
```

### Regression (run existing test suite):
```python
# Test: ALL existing DB tests pass with psycopg3 driver (full pytest run)
# Test: No integer-indexed row access (grep-based lint check for row[0], row[1] patterns)
```

---

## Section 2: Mandatory Stop-Loss Enforcement

### Before implementing validation layers:

```python
# Test: execute_trade() rejects when stop_price is None — raises ValueError
# Test: execute_trade() accepts when stop_price is provided — proceeds to risk gate
# Test: OMS submit() rejects entry orders without stop_price
# Test: OMS submit() allows exit/close orders without stop_price (exits don't need stops)
```

### Before implementing bracket-or-contingent pattern:

```python
# Test: AlpacaBroker.submit_bracket() uses native bracket API when available
# Test: AlpacaBroker.submit_bracket() falls back to entry+contingent SL when bracket fails
# Test: AlpacaBroker.submit_bracket() NEVER falls back to plain order (verify old behavior removed)
# Test: PaperBroker.submit_bracket() tracks SL/TP internally
# Test: EtradeBroker.submit_bracket() implements same interface
# Test: BracketIntent model validates required fields (stop_price mandatory)
# Test: client_order_id format includes millisecond precision and random suffix
# Test: client_order_id is deterministic given same inputs (idempotent retry)
```

### Before implementing contingent SL path:

```python
# Test: fill detected via WebSocket within 2s of broker fill
# Test: SL submitted immediately after fill detection
# Test: SL submission retried 3x with exponential backoff on failure
# Test: all 3 SL retries fail → kill switch triggered for that symbol
# Test: partial fill + SL rejection → remaining entry qty cancelled + standalone SL for filled qty
```

### Before implementing post-submission verification:

```python
# Test: bracket leg verification runs 5s after submission
# Test: missing leg detected → SL submitted immediately
# Test: rejected leg detected → remaining legs cancelled + SL submitted
# Test: verification re-runs every 30s while position open
```

### Before implementing startup reconciliation:

```python
# Test: position with active stop order → no action taken
# Test: position without stop order → ATR-based SL submitted automatically
# Test: reconciliation logs warning for each auto-fixed position
# Test: trading proceeds after auto-fix (not halted)
# Test: reconciliation runs before first graph cycle
```

### Before implementing bracket leg persistence:

```python
# Test: bracket_legs table created with correct schema
# Test: bracket leg IDs persisted on bracket order submission
# Test: bracket legs queryable after process restart (DB persistence)
```

### Before implementing circuit breaker integration:

```python
# Test: broker API degraded → new entries blocked
# Test: broker API degraded → SL retries continue with exponential backoff
# Test: broker API recovers → entries resume
```

---

## Section 3: Prompt Injection Defense

### Before implementing field-level extraction (Layer 2 — primary defense):

```python
# Test: MarketDataResponse extracts only typed fields from raw API response
# Test: extra/unexpected fields in raw response are discarded (not passed to prompt)
# Test: malicious content in raw response does not appear in extracted fields
# Test: Pydantic validation rejects response missing required fields
```

### Before implementing XML-tagged templates (Layer 1):

```python
# Test: safe_prompt() wraps field values in XML tags
# Test: safe_prompt() replaces {field_name} placeholders correctly
# Test: safe_prompt() with missing field raises KeyError (not silent empty)
# Test: template output matches expected format with tags
```

### Before implementing injection monitoring (Layer 3):

```python
# Test: detect_injection() flags "ignore previous instructions"
# Test: detect_injection() flags "system:", "assistant:", "human:" prefixes
# Test: detect_injection() flags XML/HTML tags in data fields
# Test: detect_injection() returns detection details for logging (not just bool)
# Test: detect_injection() on clean data returns no findings
# Test: detection events are logged (verify log output)
```

### Before implementing Dual LLM separation (Layer 4):

```python
# Test: research agent config has NO execution tools in tool categories
# Test: trading agent config has execution tools
# Test: research agent cannot invoke execute_trade tool (tool resolution fails)
# Test: research agent CAN invoke data/analysis tools
# Test: graph node code mediates: research output → structured data → trading input
```

### Migration regression:

```python
# Test: each migrated prompt produces functionally equivalent output to f-string version
# Test: research graph prompts migrated first (highest priority)
```

---

## Section 4: Output Schema Validation with Retry

### Before implementing fallback audit:

```python
# Test: safety_check fallback is {"halted": True} (fail-CLOSED, not fail-OPEN)
# Test: risk assessment fallback rejects (not approves)
# Test: entry_scan fallback is [] (no entries — safe)
# Test: ALL 21 agent fallbacks documented and classified as fail-safe
```

### Before implementing Pydantic models:

```python
# Test: MarketIntelOutput validates known-good agent output sample
# Test: MarketIntelOutput rejects output missing required fields
# Test: EntrySignalOutput validates known-good sample
# Test: (repeat for each of 21 models with representative samples)
# Test: each model's JSON schema is serializable (for retry prompt inclusion)
```

### Before implementing parse_and_validate():

```python
# Test: valid JSON + valid schema → parsed and validated correctly
# Test: valid JSON + invalid schema → retry triggered with schema hint
# Test: invalid JSON → retry triggered
# Test: retry succeeds → result returned, flagged as retried in audit trail
# Test: retry fails → dead letter queue entry created, fail-safe fallback returned
# Test: retried output has "retried" flag in audit trail
```

### Before implementing dead letter queue:

```python
# Test: agent_dead_letters table created with correct schema
# Test: DLQ entry contains agent_name, cycle_id, raw_output, parse_error, retry_attempted
# Test: DLQ queryable by agent_name and time range
# Test: supervisor health check flags agent with >10% DLQ rate
```

---

## Section 5: Non-Root Containers

### Before modifying Dockerfile:

```python
# Test: container runs as non-root user (docker exec whoami → "quantstack")
# Test: application writes to /app/logs successfully
# Test: kill switch writes sentinel file successfully
# Test: all services pass health checks after rebuild
# Test: init: true prevents zombie process accumulation
```

---

## Section 6: Durable Checkpoints (PostgresSaver)

### Before implementing checkpointer factory:

```python
# Test: create_checkpointer() returns configured PostgresSaver
# Test: checkpointer uses ConnectionPool (not single connection)
# Test: pool sizing: min_size=2, max_size=6
```

### Before updating runners:

```python
# Test: trading_runner uses PostgresSaver (not MemorySaver)
# Test: research_runner uses PostgresSaver
# Test: supervisor_runner uses PostgresSaver
# Test: thread_id format unchanged: {graph_name}-{date}-cycle-{number}
```

### Before implementing crash recovery:

```python
# Test: kill container mid-cycle → restart → graph resumes from last checkpoint
# Test: node that crashed mid-execution re-executes on resume
# Test: completed nodes are NOT re-executed on resume
```

### Before implementing retention:

```python
# Test: checkpoint pruning removes rows older than 48 hours
# Test: pruning preserves most recent completed cycle per graph
# Test: pruning preserves in-progress cycles
# Test: pruning runs as scheduled job without errors
```

---

## Section 7: EventBus Integration

### Before implementing kill switch publication (1.8):

```python
# Test: kill_switch.trigger() publishes KILL_SWITCH_TRIGGERED event to EventBus
# Test: event payload contains reason and timestamp
# Test: EventBus publication failure does NOT prevent kill switch activation
# Test: EventBus publication failure is logged as warning
# Test: KILL_SWITCH_TRIGGERED added to EventType enum
```

### Before implementing trading graph polling (1.7):

```python
# Test: safety_check polls for KILL_SWITCH_TRIGGERED, RISK_EMERGENCY, IC_DECAY, REGIME_CHANGE
# Test: KILL_SWITCH_TRIGGERED at safety_check → cycle halted
# Test: IC_DECAY at safety_check → affected strategy suspended
# Test: pre-execute_entries polls for KILL_SWITCH_TRIGGERED, RISK_EMERGENCY
# Test: KILL_SWITCH_TRIGGERED before entries → entries skipped, go to reflect
# Test: pre-execute_exits polls for KILL_SWITCH_TRIGGERED
# Test: KILL_SWITCH_TRIGGERED before exits → emergency close-all
# Test: cursor advances after polling regardless of action taken (idempotent)
```

### Before implementing all-graph polling:

```python
# Test: research_runner polls KILL_SWITCH_TRIGGERED at cycle start
# Test: supervisor_runner polls KILL_SWITCH_TRIGGERED at cycle start
```

### End-to-end:

```python
# Test: trigger kill switch → sentinel file written + EventBus published →
#        all 3 graphs halt within one cycle → execution monitor stops →
#        position closer fires
```

---

## Section 8: Automated Database Backups

### Before implementing backup script:

```python
# Test: pg_dump creates valid backup file
# Test: pg_restore --list succeeds on backup file (integrity check)
# Test: backup script exits non-zero on pg_dump failure
# Test: backups older than 30 days are deleted
# Test: WAL archive files older than 7 days are pruned
# Test: backup script uses flock to prevent concurrent runs
```

### Before implementing restore procedure:

```python
# Test: full restore from pg_dump → all tables intact with correct row counts
# Test: PITR restore from WAL → data consistent to target timestamp
```

### Before implementing monitoring:

```python
# Test: supervisor health check detects backup older than 36 hours
# Test: supervisor raises warning event for stale backup
```

---

## Section 9: Containerize Scheduler

### Before fixing import chain:

```python
# Test: "from quantstack.runners import scheduler" succeeds without ibkr_mcp
# Test: "python scripts/scheduler.py --dry-run" runs without import errors
```

### Before implementing Docker service:

```python
# Test: scheduler container starts and health check passes
# Test: scheduler container auto-restarts after kill (within 60s)
# Test: APScheduler has all 5 jobs registered
# Test: health endpoint returns job list and next_run times
# Test: SIGTERM triggers clean APScheduler shutdown
```

---

## Section 10: DB Transaction Isolation

### Before implementing row-level locking:

```python
# Test: SELECT FOR UPDATE acquires row lock on position
# Test: second writer blocks until first commits
# Test: lock_timeout fires after 5s → retry once after 500ms
# Test: second retry timeout → CRITICAL log, operation continues with stale data
# Test: reader not blocked by writer (MVCC verification)
# Test: single-row constraint: transaction only locks one position row at a time
```

### Concurrency:

```python
# Test: two concurrent updates on same symbol → no lost writes
# Test: execution monitor + trading graph race on same position → both updates applied
# Test: N concurrent writers stress test → acceptable latency, no lost updates
```

### Write path coverage:

```python
# Test: alpaca_broker.py fill handler uses locking pattern
# Test: trade_service.py metadata update uses locking pattern
# Test: execution_monitor.py trailing stop update uses locking pattern
# Test: startup reconciliation uses locking pattern
# Test: kill_switch position closer uses locking pattern
```
