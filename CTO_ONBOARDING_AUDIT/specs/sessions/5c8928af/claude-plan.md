# Phase 1: Safety Hardening — Implementation Plan

## Overview

QuantStack is an autonomous trading system built on three LangGraph StateGraphs (trading, research, supervisor) running as Docker services. It uses PostgreSQL for all state, Alpaca for brokerage, and LLM agents for research and trading decisions. The system is currently in paper mode with no open positions.

This plan addresses 10 P0 safety findings from a CTO onboarding audit. These are existential risks: unprotected positions, data loss, prompt injection, silent parse failures, root-running containers, ephemeral checkpoints, broken inter-graph communication, uncontainerized scheduler, and race conditions on position updates. None of these can remain when real capital is deployed.

### Dependencies and Sequencing

The 10 items are not independent. The psycopg3 migration (part of item 1.6) is foundational — it changes the database driver used by everything. Transaction isolation (1.10) depends on the new driver. The EventBus items (1.7, 1.8) are tightly coupled. Everything else is parallelizable.

**Recommended execution order:**
1. **Foundation**: psycopg3 migration (from 1.6) + non-root containers (1.5) — minimal blast radius, unblocks everything
2. **Core safety**: Stop-loss enforcement (1.1) + prompt injection defense (1.3) — largest items, start early
3. **Reliability**: Output validation (1.4) + durable checkpoints (remainder of 1.6) + EventBus (1.7+1.8)
4. **Infrastructure**: DB backups (1.2) + scheduler containerization (1.9) + transaction isolation (1.10)

---

## Section 1: psycopg3 Migration

### Context

The codebase uses psycopg2 (`ThreadedConnectionPool`) in `src/quantstack/db.py`. LangGraph's PostgresSaver requires psycopg3 (`psycopg[binary,pool]`). Rather than maintaining two connection systems, the decision is to migrate db.py entirely to psycopg3.

### Approach

**Replace the connection pool.** psycopg3's `ConnectionPool` from `psycopg_pool` replaces `ThreadedConnectionPool`. Key differences:
- psycopg3 uses `ConnectionPool` (not `ThreadedConnectionPool`) — thread safety is built in
- Connection params: `min_size=4`, `max_size=20` (match current maxconn), `max_lifetime=3600`, `max_idle=600`
- `autocommit=True` default for PostgresSaver compatibility; application queries use explicit transactions

**Update PgConnection wrapper.** The existing `PgConnection` class (lines 110-268) wraps psycopg2 connections with lazy acquisition, retry, and cursor management. Adapt it to psycopg3:
- psycopg3 cursors return native Python types by default (no need for custom type adapters)
- `%s` placeholders still work in psycopg3 (compatibility mode)
- The `?` → `%s` translation in `execute()` can be removed if all callers already use `%s`
- `fetchdf()` (returns pandas DataFrame) needs psycopg3's `cursor.description` for column names
- Error handling: `psycopg.OperationalError` replaces `psycopg2.OperationalError`

**JSON handling.** The current code registers a custom JSON deserializer (lines 61-62) that returns raw strings. psycopg3 handles JSON natively via `psycopg.types.json` — register the same behavior there.

**Idle timeout.** The current `idle_in_transaction_session_timeout = 30s` (line 153) is set per-session via SET command. This works identically in psycopg3.

**Test migration.** Every test fixture that creates a DB connection needs updating. The `conftest.py` fixtures for DB connection management are the primary change surface.

**Migration checklist: all psycopg2 import sites.** The following files import psycopg2 directly and must be migrated:
- `src/quantstack/db.py` — primary connection pool (main change)
- `src/quantstack/storage/pg_storage.py` — direct psycopg2 usage
- `src/quantstack/rag/query.py` — RAG query layer
- `src/quantstack/health/langfuse_retention.py` — health check queries
- `scripts/ewf_analyzer.py` — standalone script
- 8 test files in `tests/` — fixture and assertion code
- `scripts/heartbeat.sh`, `stop.sh` — shell scripts with psycopg2 references

Each file needs: import statement change, connection acquisition change, and audit for integer-indexed row access (`row[0]`, `row[1]`) which breaks with psycopg3's `dict_row`.

**Placeholder audit.** The existing code uses both `?` placeholders (e.g., `trade_service.py:248-254`) and `%s`. The `?` → `%s` translation in PgConnection.execute() works in psycopg3, but all callers should be migrated to `%s` directly to remove the translation layer. This is a bulk search-and-replace task.

### Risks

- **Behavioral differences**: psycopg3 returns plain `dict` with `row_factory=dict_row` vs psycopg2's `RealDictRow` (which supports both key and index access). Any code using `row[0]`, `row[1]` etc. will break. Grep for these patterns across all DB-consuming files.
- **Async implications**: psycopg3 has native async support. The execution monitor is async — this opens the door to `AsyncConnectionPool` later, but for Phase 1, keep the sync pool.
- **Total connection budget**: Main pool (max 20) + checkpointer pool (max 6) + scheduler + backup = ~28 connections. PostgreSQL default max_connections is 100. Verify this is sufficient and document the budget.

---

## Section 2: Mandatory Stop-Loss Enforcement

### Context

The current execution flow: `trade_service.execute_trade()` → `risk_gate.check()` → `alpaca_broker.execute_bracket()`. The bracket order path (trade_service.py:213-223) only fires when `broker.supports_bracket` and both `stop_price` and `target_price` are set. If bracket submission fails, `alpaca_broker.py` line 223 falls back to a plain market order — the position opens with no stop protection.

### Approach

**Layer 1: Reject orders without stop_price.** In `trade_service.py`'s `execute_trade()` function, validate that the `stop_price` parameter is not None before the risk gate check. Note: `stop_price` is a function parameter, not a field on `OrderRequest` — the validation goes on the function's input, not the model. This is the first line of defense — no order can even reach the risk gate without a stop.

**Layer 2: OMS enforcement.** In `order_lifecycle.py`'s `submit()` compliance checks (lines 484-521), add a check: if the order is an entry (not an exit/close), `stop_price` must be set. This catches any code path that bypasses trade_service.

**Layer 3: Bracket-or-contingent pattern across all brokers.** Define a `BracketIntent` model:

```python
class BracketIntent(BaseModel):
    symbol: str
    side: str
    quantity: int
    entry_type: str  # "market" or "limit"
    entry_price: float | None
    stop_price: float
    target_price: float | None
    client_order_id: str  # deterministic: {strategy_id}_{symbol}_{ts_ms}_{leg}_{random4}
```

Each broker adapter implements a `submit_bracket(intent: BracketIntent)` method with three strategies attempted in order:
1. **Native bracket** (if supported) — one API call, broker manages OCO
2. **Entry + separate contingent SL** — submit entry, on fill submit SL as separate order
3. **Reject** — if neither works, reject the order (never fall back to plain)

For Alpaca: use native bracket API (`order_class: "bracket"`). On failure, submit entry, then on fill immediately submit a stop-loss order as a separate `stop` order type. Fill detection uses Alpaca's trade update WebSocket stream (already consumed by execution monitor). Maximum unprotected window: ~2 seconds (WebSocket latency + SL submission). If SL submission fails: retry 3 times with exponential backoff (1s, 2s, 4s). If all retries fail: trigger kill switch for that symbol — the position is unprotectable.

**Partial fill handling.** If the entry leg partially fills and the SL leg is rejected: cancel remaining entry quantity immediately, then submit a standalone SL for the filled quantity only. The position must never have exposure without a corresponding stop.

For PaperBroker: simulate bracket behavior — track SL/TP internally, evaluate on price ticks.

For E*Trade: implement the same interface using E*Trade's conditional order API.

**Layer 4: Post-submission verification.** After bracket submission, verify all legs exist:
- Query broker for the order and its legs after 5 seconds
- If any leg missing or rejected, cancel entry (if unfilled) or submit missing SL immediately
- Re-verify every 30 seconds while position is open (add to execution monitor's tick loop)

**Layer 5: Startup reconciliation.** On trading runner startup, before entering the graph loop:
1. Fetch all open positions from broker
2. Fetch all open orders from broker
3. For each position, check for an active stop order on that symbol
4. If missing: compute ATR-based stop price, submit SL order, log warning
5. Continue trading (auto-fix, don't halt)

**Layer 6: Bracket leg persistence.** Add a `bracket_legs` table:

```python
class BracketLeg(BaseModel):
    parent_order_id: str
    leg_type: str  # "entry", "stop_loss", "take_profit"
    broker_order_id: str
    status: str
    price: float
    created_at: datetime
```

This replaces the current in-memory `Fill` tracking that's lost on crash.

**Circuit breaker integration.** When the broker API is degraded (consecutive failures in AutoTriggerMonitor), the circuit breaker should:
- Stop submitting new entries
- Continue retrying stop-loss submissions with exponential backoff
- A missed entry is a missed opportunity; a missed stop is unbounded loss

### Key Invariants
- No `OrderRequest` can exist with `stop_price=None`
- Bracket failure NEVER degrades to plain order
- Every open position has a broker-side stop order (verified on startup and continuously)
- Bracket leg state survives process restarts (DB, not memory)

---

## Section 3: Prompt Injection Defense

### Context

The agent executor (`agent_executor.py`) builds prompts by concatenating role/goal/backstory from agent config, then each graph node adds context via f-string interpolation. Example from trading nodes.py:80-92: `f"Portfolio: {json.dumps(portfolio_ctx, default=str)}"`. Research nodes.py:219-224 similarly interpolates community ideas, queued ideas, and prefetched context directly.

QuantStack has the "Lethal Trifecta": access to private data (portfolio, strategies), exposure to untrusted content (market APIs, news, knowledge base), and ability to take consequential actions (execute trades). All three in one system.

### Approach

**Layer 1: Structured XML-tagged templates.** Replace all f-string prompt construction with a template system. Create `src/quantstack/graphs/prompt_safety.py` with:

```python
def safe_prompt(template: str, **fields: str) -> str:
    """Build prompt with XML-tagged field boundaries.
    
    Each field value is sanitized and wrapped in XML tags.
    Template uses {field_name} placeholders that are replaced
    with <field_name>sanitized_value</field_name>.
    """
```

Example transformation:
```
# Before (vulnerable)
f"Portfolio: {json.dumps(portfolio_ctx)}"

# After (defended)
safe_prompt("Portfolio: {portfolio}", portfolio=json.dumps(portfolio_ctx))
# Produces: "Portfolio: <portfolio>{...sanitized...}</portfolio>"
```

**Layer 2: Field-level extraction (PRIMARY defense — allowlist approach).** This is the most important defense layer. Instead of trying to strip bad content from raw data (blocklist), extract ONLY known-good fields into typed values before any data enters a prompt:
- Every external data source (market APIs, news feeds, knowledge base) is consumed through a Pydantic model that extracts specific typed fields
- Raw JSON/text from external sources NEVER reaches a prompt template — only extracted, typed values do
- This is an allowlist approach: only explicitly extracted fields pass through, everything else is discarded

```
# BAD (blocklist - vulnerable to novel injection patterns):
sanitize(raw_api_response)  # strip known bad patterns

# GOOD (allowlist - only known-good fields pass through):
data = MarketDataResponse.model_validate(raw_api_response)
safe_prompt("Price: {price}\nVolume: {volume}",
    price=str(data.price), volume=str(data.volume))
```

**Layer 3 (secondary): Pattern-based monitoring.** A `detect_injection()` function scans text for known injection patterns ("ignore previous instructions", "system:", "assistant:", "human:", XML/HTML tags, excessive control characters). This function does NOT block or strip — it LOGS and ALERTS when patterns are detected. This is a monitoring signal, not a security boundary. High detection rates on a data source indicate compromise or adversarial manipulation.

**Layer 4: Dual LLM separation.** Enforce architecturally that research-facing LLMs cannot access execution tools:
- In `agent_executor.py`'s tool category system (lines 55-92), create a strict separation:
  - **Research tool categories**: Signal & Analysis, Data, Features, ML Training, Backtesting, Validation, Knowledge — NO Execution, NO Portfolio mutation
  - **Trading tool categories**: Can access Execution, Portfolio, Risk — but receives only structured data from research, never raw external text
- The graph node code (not LLM) mediates: research agents produce structured recommendations (`symbol`, `direction`, `confidence`, `rationale`), trading agents receive these as validated data objects
- Audit each LLM call in the system for the Lethal Trifecta: document what private data it sees, what untrusted content it processes, what actions it can trigger

**Layer 4: Field-level extraction.** Replace raw `json.dumps()` of full objects with explicit field extraction:
```
# Before: dumps entire portfolio_ctx dict (may contain injected content)
# After: extract only the fields needed
safe_prompt("Positions: {positions}\nCash: {cash}",
    positions=format_positions(portfolio_ctx["positions"]),
    cash=str(portfolio_ctx["cash"]))
```

### Migration Strategy
- Identify all prompt construction sites (grep for f-strings in nodes.py files across all three graphs)
- Migrate one graph at a time: **research first** (highest untrusted data exposure, most critical to defend), then trading, then supervisor (lowest risk, least exposure)
- Run parallel comparison for 2 cycles: old prompts vs new prompts, compare agent outputs
- Roll back if agent behavior degrades significantly

### Key Files to Modify
- New: `src/quantstack/graphs/prompt_safety.py`
- Modify: `src/quantstack/graphs/trading/nodes.py` (all node functions that build prompts)
- Modify: `src/quantstack/graphs/research/nodes.py` (all node functions)
- Modify: `src/quantstack/graphs/supervisor/nodes.py` (all node functions)
- Modify: `src/quantstack/graphs/agent_executor.py` (tool category enforcement)

---

## Section 4: Output Schema Validation with Retry

### Context

`parse_json_response()` in `agent_executor.py:474-521` tries direct JSON parse, then substring extraction, then returns an empty fallback. No retry, no logging beyond debug, no dead letter queue. 21 agents all go through this path.

### Approach

**Define 21 Pydantic models.** One per agent, in `src/quantstack/tools/models.py` (or a new `src/quantstack/graphs/schemas/` directory if models.py gets too large). Each model captures the expected output shape for that specific agent. Examples:

```python
class MarketIntelOutput(BaseModel):
    """Output schema for market_intel agent."""
    headlines: list[str]
    risk_alerts: list[str]
    event_calendar: list[dict]
    sector_news: dict
    sentiment: Literal["bullish", "neutral", "bearish"]

class EntrySignalOutput(BaseModel):
    """Output schema for entry_scan agent."""
    signals: list[SignalCandidate]
    reasoning: str
    regime_assessment: str
```

**Audit ALL existing fallback values.** Before changing `parse_json_response`, audit every call site across all three graphs. Classify each fallback as fail-SAFE (conservative) or fail-OPEN (dangerous). Critical finding from review: the safety_check agent falls back to `{"halted": False}`, which means a parse failure **bypasses the safety check**. This is a P0 safety inversion. All safety-critical fallbacks must fail CLOSED:
- `safety_check` → `{"halted": True, "reason": "parse_failure"}` (HALT on failure)
- `risk_assessment` → reject (not approve) on failure
- `entry_scan` → `[]` (no entries — already safe)
- `position_review` → no exits triggered (conservative — hold positions)

**Enhance parse_json_response.** Rename to `parse_and_validate()`:
1. Attempt JSON parse (existing logic)
2. If JSON parsed, validate against the agent's Pydantic model
3. If validation fails OR JSON parse fails: retry once with the model's JSON schema included in the prompt
4. If retry also fails: log warning, write to dead letter queue, return the fail-safe fallback
5. **Flag retried outputs in the audit trail** so downstream consumers and human reviewers know the output came from a retry path

**Retry mechanism.** The agent executor loop (lines 181-350) already has a round-based conversation. On parse failure:
- Append a user message: "Your response was not valid JSON matching the expected schema. Please respond with valid JSON matching this schema: {model.model_json_schema()}"
- Run one more LLM round
- If this also fails, give up and go to dead letter queue

**Dead letter queue.** New table `agent_dead_letters`:

```python
class AgentDeadLetter(BaseModel):
    agent_name: str
    cycle_id: str
    graph_name: str
    raw_output: str
    parse_error: str
    retry_attempted: bool
    created_at: datetime
```

Add a supervisor health check that queries DLQ frequency per agent over the last 24h. High rate (>10% of cycles for any agent) indicates a prompt quality issue that needs investigation.

**Integration with agent_executor.** The `execute_agent_node()` function needs to accept an optional `output_schema: type[BaseModel]` parameter. Each graph node passes its expected schema. The executor uses it for validation and retry.

---

## Section 5: Non-Root Containers

### Context

The Dockerfile has no `USER` directive. All containers run as root.

### Approach

**Create a non-root user in the Dockerfile.** After installing dependencies but before copying application code:
- `RUN useradd -r -s /bin/false quantstack`
- `RUN chown -R quantstack:quantstack /app /data`
- `USER quantstack`

**Volume mount permissions.** The docker-compose.yml mounts several host directories:
- `./src:/app/src` — read-only in container is fine (live source for dev)
- `./logs:/app/logs` — needs write access
- `~/.quantstack:/root/.quantstack` — this path breaks with non-root user; change to `/home/quantstack/.quantstack` and update the `KILL_SWITCH_SENTINEL` env var

**TA-Lib.** The compiled TA-Lib library in `/usr/local/lib` is readable by all users. No issue.

**Kill switch sentinel.** Currently writes to `~/.quantstack/KILL_SWITCH_ACTIVE`. With non-root user, this becomes `/home/quantstack/.quantstack/KILL_SWITCH_ACTIVE`. Update the default path in `kill_switch.py` or ensure the env var `KILL_SWITCH_SENTINEL` is set in docker-compose.yml.

**Init process.** Add `init: true` to all services in docker-compose.yml. Without an init process, zombie processes can accumulate from subprocess spawning (APScheduler, potential subprocess calls in tools).

### Testing
- Build image, run container, exec `whoami` → verify returns `quantstack`
- Run full trading cycle in paper mode → verify all operations succeed
- Verify kill switch can write sentinel file
- Verify logs are written

---

## Section 6: Durable Checkpoints (PostgresSaver)

### Context

After psycopg3 migration (Section 1), the database layer supports psycopg3. Now switch all three graph runners from MemorySaver to PostgresSaver.

### Approach

**Install dependency.** Add `langgraph-checkpoint-postgres~=3.0` to project dependencies.

**Create a shared checkpointer factory.** In `src/quantstack/db.py` or a new `src/quantstack/checkpointing.py`:

```python
def create_checkpointer() -> PostgresSaver:
    """Create a PostgresSaver backed by the application connection pool.
    
    Uses a dedicated ConnectionPool (psycopg3) sized for checkpoint operations.
    setup() is NOT called here — run it as a deployment step.
    """
```

Pool sizing for checkpointer: `min_size=2, max_size=6` (lower than main pool — checkpoint writes are less frequent than application queries).

**Update all three runners.** In `trading_runner.py`, `research_runner.py`, `supervisor_runner.py`:
- Replace `from langgraph.checkpoint.memory import MemorySaver` with the checkpointer factory
- Thread ID format stays the same: `{graph_name}-{YYYY-MM-DD}-cycle-{cycle_number}`

**Deployment step: setup().** Add a management command or startup script that runs `checkpointer.setup()` once. This creates the 4 checkpoint tables. Do NOT call setup() in the runner — if multiple runners start simultaneously, they'd race on table creation.

Add to `entrypoint.sh`: run setup before launching the graph runner if a flag is set (`--migrate` or env var).

**Crash recovery validation.** The `checkpoint_writes` table is key: LangGraph writes intermediate state after each node. On restart with the same thread_id, it reconstructs from last checkpoint + pending writes. Nodes that crashed mid-execution re-execute.

**Checkpoint data retention.** PostgresSaver stores full state at every node transition. With 16 nodes per trading cycle running every 5 minutes, that is ~4,600 checkpoint rows/day/graph, ~14,000 rows/day across three graphs. Without pruning, this grows to ~420,000 rows/month. Add a scheduled pruning job (run daily by the scheduler): delete checkpoint data older than 48 hours. Keep the most recent completed cycle for each graph plus any in-progress cycles.

### Interaction with Existing Checkpoint Table
The current `graph_checkpoints` table (execution metadata: graph_name, cycle_number, duration, status) is separate from PostgresSaver's tables. Keep both — the existing table is for operational dashboards, PostgresSaver's tables are for state recovery.

---

## Section 7: EventBus Integration (Items 1.7 + 1.8)

### Context

The EventBus (`coordination/event_bus.py`) is PostgreSQL-based with per-consumer cursors. The supervisor publishes events but the trading graph never polls them. The kill switch writes a sentinel file but never publishes to EventBus.

### Approach

**1.8: Kill switch publishes to EventBus.** In `kill_switch.py`'s `trigger()` method (lines 98-128), after writing the sentinel file, publish a `KILL_SWITCH_TRIGGERED` event:

```python
bus.publish(Event(
    event_type=EventType.KILL_SWITCH_TRIGGERED,
    source_loop="kill_switch",
    payload={"reason": reason, "triggered_at": now.isoformat()}
))
```

Add `KILL_SWITCH_TRIGGERED` to the `EventType` enum (not currently present — this is a concrete code change).

**CRITICAL: EventBus publication must be best-effort.** If PostgreSQL is down when the kill switch fires, the EventBus publish will fail. This MUST NOT delay or prevent kill switch activation. Wrap the publish call in try/except, log the failure, and continue. The sentinel file (filesystem-based) is the primary kill switch mechanism; EventBus publication is supplementary notification.

**1.7: Trading graph polls EventBus.** Add polling at three points in the trading graph:

1. **`safety_check` node** (existing, at cycle start): Poll for `KILL_SWITCH_TRIGGERED`, `RISK_EMERGENCY`, `IC_DECAY`, `REGIME_CHANGE`. If kill switch or risk emergency, halt the cycle. If IC_DECAY, mark the affected strategy as suspended in state.

2. **Before `execute_entries`**: Poll for `KILL_SWITCH_TRIGGERED` and `RISK_EMERGENCY` only (narrow scope — we're about to commit capital). If either present, skip entries and go to reflect.

3. **Before `execute_exits`**: Same poll. If kill switch triggered, escalate to emergency close-all rather than normal exit logic.

**All graph runners poll at cycle start.** In each runner's main loop, before invoking the graph, poll for `KILL_SWITCH_TRIGGERED`. This adds ~5 lines per runner.

**EventBus consumer IDs.** Use `trading-graph`, `research-graph`, `supervisor-graph` as consumer IDs. Each maintains its own cursor in `loop_cursors`.

### Latency Analysis
- Trading cycle: ~5 minutes. With polling at start + before entries + before exits, max latency for a critical event is ~2-3 minutes (time between safety_check and execute_entries).
- Research cycle: ~60 seconds. Polling at start is sufficient.
- Supervisor cycle: Already publishes events. Polling at start for kill switch is sufficient.

---

## Section 8: Automated Database Backups

### Context

60+ tables in PostgreSQL, Docker volumes on local disk, no backup mechanism. Decision: local pg_dump + WAL archiving for now, offsite deferred.

### Approach

**Daily pg_dump script.** New `scripts/backup.sh`:
- Run `pg_dump --format=custom` (custom format supports selective restore and compression)
- Output to `/data/quantstack/backups/quantstack_YYYY-MM-DD.dump`
- Retain last 30 days of backups (delete older)
- Verify dump integrity: `pg_restore --list` on the output file
- Exit with non-zero on any failure

**WAL archiving.** Configure PostgreSQL for continuous archiving:
- `wal_level = replica` in postgresql.conf
- `archive_mode = on`
- `archive_command = 'cp %p /data/quantstack/wal_archive/%f'`
- This enables point-in-time recovery (PITR) between daily dumps
- **Retention**: Prune WAL archive files older than 7 days (matching PITR window between daily dumps). Add to backup script or as a separate cron entry. Without pruning, WAL archives grow unbounded and will eventually fill the disk.

**Integration with Docker.** Two options:
1. Add a `backup` service to docker-compose.yml that runs the script on cron (via `ofelia` or similar)
2. Add the backup job to the scheduler (scripts/scheduler.py) since it already runs cron jobs

Option 2 is simpler since scheduler infrastructure exists. Add a daily job at 02:00 UTC.

**Supervisor health check.** Add a check in the supervisor graph: query for the most recent backup file's timestamp. If >36 hours old (allowing for weekend buffer), raise a warning event.

**Restore documentation.** Write a runbook section covering:
- Full restore from pg_dump: `pg_restore -d quantstack backup.dump`
- PITR from WAL: restore base backup, then replay WAL to target timestamp
- Verification: count rows in key tables, compare to pre-backup counts

---

## Section 9: Containerize Scheduler

### Context

The scheduler (`scripts/scheduler.py`) runs 5 APScheduler jobs in a tmux session. It imports from quantstack, which triggers an `ibkr_mcp` import error. The decision is to fix the import chain properly and containerize.

### Approach

**Fix ibkr_mcp import.** Identify where `ibkr_mcp` is imported in the quantstack package. Options:
- If it's a top-level import in an `__init__.py`: make it conditional (`try/except ImportError`) or move it to the module that actually uses it
- If it's a dependency that's not installed: add it to the optional dependencies group, or remove the import if the module is unused
- The goal: `python -c "from quantstack.runners import scheduler"` succeeds without ibkr_mcp installed

**Docker service.** Add to docker-compose.yml:

```yaml
scheduler:
  build: .
  command: python scripts/scheduler.py
  restart: unless-stopped
  volumes: *graph-volumes
  environment: *graph-env
  healthcheck:
    test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8422/health')"]
    interval: 60s
    timeout: 15s
    retries: 3
  mem_limit: 512m
```

**Health endpoint.** Add a lightweight HTTP health endpoint to scheduler.py (e.g., Flask or just a threading HTTP server):
- `GET /health` returns 200 with JSON: `{"status": "running", "jobs": [...], "next_run": "..."}`
- Check that APScheduler is alive and has registered jobs
- Report next scheduled run time for each job

**Graceful shutdown.** Handle SIGTERM in the scheduler to cleanly shut down APScheduler before container stops. The `stop_grace_period` should match the longest job timeout (2 hours for data_refresh — set `stop_grace_period: 7200s` or accept that a running job will be killed).

---

## Section 10: DB Transaction Isolation for Positions

### Context

The execution monitor (async, evaluating exit rules on price ticks) and the trading graph (sizing new entries) can both read and update the same position simultaneously. Default READ COMMITTED allows both to read stale state; one update overwrites the other. Supervisor only reads — no locking needed for reads.

### Approach

**Row-level locking with SELECT FOR UPDATE.** Wrap position update queries in explicit transactions:

```python
def update_position_with_lock(conn, symbol: str, updates: dict):
    """Update a position row with exclusive lock.
    
    Uses SELECT FOR UPDATE to prevent concurrent modifications.
    The lock is held only for the duration of the transaction.
    """
```

Pattern:
1. `BEGIN`
2. `SELECT * FROM positions WHERE symbol = %s FOR UPDATE` — acquires row lock, blocks other writers
3. Apply updates based on current state
4. `COMMIT` — releases lock

**Identify all position write paths.** Based on codebase research, position writes happen in:
- `alpaca_broker.py` after fill: `portfolio.update_position()` — entry/exit
- `trade_service.py:225-259`: position metadata update (strategy, exit levels)
- `execution_monitor.py`: trailing stop updates, exit execution
- Startup reconciliation (new, from Section 2)

- `kill_switch.py` position closer callback (lines 122-128): modifies position state during emergency close

Each of these needs to use the locking pattern. **Single-row constraint**: all position updates operate on one symbol at a time within a single transaction. Never hold locks on multiple position rows in one transaction — this eliminates deadlock risk between the two writers.

**Connection management.** With psycopg3 (from Section 1), explicit transactions are straightforward:
- `with conn.transaction()` context manager handles BEGIN/COMMIT/ROLLBACK
- `cursor.execute("SELECT ... FOR UPDATE")` within the transaction block

**Timeout and retry.** Add `lock_timeout = 5000` (5 seconds) as a session-level setting for position-update connections. If a lock can't be acquired within 5s: retry once after 500ms. If the second attempt also times out, log a CRITICAL alert (this indicates an unexpected contention pattern). For the execution monitor specifically, a failed trailing stop update means the stop price is stale — log the failure and continue with the existing stop price (fail-safe: the position still has protection, just not the latest trailing level).

### What NOT to Lock
- Position reads (monitoring, dashboard, reporting) — no locking needed, MVCC handles this
- Non-position tables (orders, signals, strategies) — no concurrent write contention
- The risk gate's portfolio read — it reads a snapshot, doesn't write

---

## Section 11: Testing Strategy

### Overview

Each section above needs tests. The existing test infrastructure uses pytest with markers (slow, integration, benchmark, requires_api) and directory structure: `tests/{integration, core, graphs, coordination, unit, fixtures, regression, benchmarks, property}`.

### Test Categories by Section

**Section 1 (psycopg3 migration):**
- All existing DB tests must pass with psycopg3 (regression)
- Connection pool behavior: max connections, timeout, retry
- JSON handling compatibility

**Section 2 (stop-loss enforcement):**
- Unit: OrderRequest with `stop_price=None` rejected at trade_service
- Unit: OMS rejects entry orders without stop
- Unit: Bracket failure → contingent SL placed (never plain order)
- Integration: Full bracket order flow through PaperBroker
- Integration: Startup reconciliation detects missing stops, submits them
- Integration: Circuit breaker stops entries but continues SL retries
- **Chaos**: Broker API returns HTTP 500 three times during contingent SL fallback → verify SL eventually placed or kill switch triggered
- Unit: Partial fill + SL rejection → cancel remaining entry, submit standalone SL for filled qty

**Section 3 (prompt injection):**
- Unit: Sanitization strips injection patterns
- Unit: XML-tagged templates produce expected output
- Unit: Research agents cannot access execution tools (tool category enforcement)
- Integration: Adversarial knowledge base entry → verify sanitization
- Regression: Agent outputs remain functionally equivalent after template migration

**Section 4 (output validation):**
- Unit: Valid JSON + valid schema → parsed correctly
- Unit: Valid JSON + invalid schema → retry with schema hint
- Unit: Invalid JSON → retry → dead letter queue
- Unit: Each of 21 Pydantic models validates a known-good output sample
- Integration: DLQ populated on parse failure, queryable

**Section 5 (non-root):**
- Integration: Container runs as non-root (`whoami` test)
- Integration: All services function correctly (health checks pass)
- Integration: Kill switch sentinel file writable

**Section 6 (PostgresSaver):**
- Integration: Kill container mid-cycle → restart → verify resume from last checkpoint
- Unit: Checkpointer factory returns configured PostgresSaver
- Integration: Concurrent graph runs don't corrupt checkpoint state

**Section 7 (EventBus):**
- Unit: Kill switch trigger publishes KILL_SWITCH_TRIGGERED event
- Unit: EventBus publication failure does not prevent kill switch activation (best-effort)
- Integration: Publish IC_DECAY → trading graph receives it within one cycle
- Integration: Kill switch event → all three graphs halt
- **End-to-end**: Kill switch trigger → sentinel file + EventBus → all graphs poll and halt → execution monitor stops → position closer fires. This full propagation path is the most critical safety mechanism.

**Section 8 (backups):**
- Integration: Run pg_dump → corrupt DB → restore → verify tables intact
- Unit: Backup script returns non-zero on failure
- Unit: Old backups pruned correctly

**Section 9 (scheduler):**
- Integration: Scheduler container starts, health check passes
- Integration: Kill container → auto-restart within 60s
- Unit: Import chain works without ibkr_mcp

**Section 10 (transaction isolation):**
- Integration: Two concurrent position updates on same symbol → no lost writes
- Unit: Lock timeout fires correctly
- Integration: Reader not blocked by writer (MVCC)
- **Stress**: N concurrent writers on same position row → verify no lost updates AND acceptable latency

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| psycopg3 migration breaks existing queries | All DB operations fail | Run full test suite after migration; psycopg3's `%s` compatibility mode minimizes syntax changes |
| Dual LLM separation changes agent behavior | Agents produce different/worse outputs | Parallel comparison for 2 cycles before cutover; roll back if degraded |
| PostgresSaver adds latency to graph execution | Slower trading cycles | Benchmark before/after; checkpoint writes are async in LangGraph |
| Non-root containers break volume permissions | Services fail to start | Test with Docker volumes first; fix ownership in Dockerfile |
| Bracket order verification adds latency to entries | Slower order execution | Verification runs async (5s delay, then periodic); doesn't block the entry |
| Row-level locking could deadlock | Position updates hang | 5s lock timeout prevents indefinite blocking; only two writers to reason about |
