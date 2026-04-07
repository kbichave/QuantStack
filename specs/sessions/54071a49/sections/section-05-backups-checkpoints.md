# Section 05: Database Backups & Durable Checkpoints

## Problem

ALL system state lives in PostgreSQL with zero backup procedure. A disk failure, corruption event, or accidental DROP loses everything — trade history, strategy registry, signal data, portfolio state. There is no recovery path.

All three LangGraph StateGraphs (Trading, Research, Supervisor) use in-process `MemorySaver`. When a Docker container crashes or restarts, all graph checkpoint state is lost. The graph starts fresh on next boot, with no knowledge of where it was in its cycle. If a crash happens after `execute_entries` places an order but before the cycle completes, the resumed graph has no memory of the fill and may re-execute, causing duplicate orders.

## Dependencies

**Requires section-01-stop-loss (idempotency guards).** Section 01 adds `client_order_id`-based deduplication to `trade_service.submit_order()` and `execute_bracket()`. This is a hard prerequisite because PostgresSaver crash recovery replays from the last completed super-step, meaning side-effecting nodes (order submission, DB writes) will re-execute after a crash. Without idempotency, crash recovery causes duplicate orders.

**Blocks section-12-multi-mode** which depends on durable graph state for mode transitions.

## Scope

This section has three work items:

1. **Order idempotency guards** (prerequisite, may already exist from section-01)
2. **Daily pg_dump backup sidecar**
3. **PostgresSaver migration** (MemorySaver → AsyncPostgresSaver)

---

## Tests First

### Order Idempotency Tests

```python
# tests/execution/test_order_idempotency.py

# Test: submit_order with duplicate client_order_id is rejected (not re-executed)
# Test: execute_bracket with duplicate client_order_id is rejected
# Test: unique client_order_id passes through normally
```

These tests verify that `trade_service.submit_order()` and `execute_bracket()` track previously submitted `client_order_id` values and reject duplicates with a clear error rather than re-submitting to the broker. The dedup store can be in-memory with DB persistence — it must survive within a graph cycle but can be rebuilt from DB on startup.

### PostgresSaver Tests

```python
# tests/graphs/test_postgres_saver.py (integration marker)

# Test: AsyncPostgresSaver setup() is idempotent (call twice, no error)
#   - Call checkpointer.setup() twice in sequence
#   - Second call must not raise or duplicate tables
#
# Test: graph checkpoint writes to PostgreSQL
#   - Run a minimal graph with PostgresSaver
#   - Query checkpoint tables to verify rows exist
#
# Test: graph resumes from last super-step after restart
#   - Run a graph with a node that records execution count
#   - Interrupt after a known super-step
#   - Resume with same thread_id
#   - Verify execution continues from the interruption point, not from the start
#
# Test: CRITICAL integration test: full Trading Graph cycle → kill mid-cycle
#       after execute_entries → restart → verify no duplicate orders + correct state
#   - This is the key safety test
#   - Run Trading Graph with PostgresSaver and a mock broker
#   - Mock broker records all order submissions
#   - Interrupt after execute_entries node completes
#   - Resume with same thread_id
#   - Verify: broker received exactly 1 order (not 2)
#   - Verify: graph state reflects the completed order
#
# Test: each graph gets its own connection pool (no cross-contamination)
#   - Instantiate pools for trading, research, supervisor graphs
#   - Verify they are distinct pool objects
#   - Verify connections from one pool cannot be checked out by another
```

### Backup Script Tests

```python
# tests/scripts/test_pg_backup.py

# Test: pg_backup.sh produces valid dump file
#   - Run the backup script against a test database
#   - Verify output file exists, is non-empty, and is a valid pg_dump format
#
# Test: pg_restore_test.sh can restore from backup
#   - Create a backup, drop the test database, restore from backup
#   - Verify key tables exist and row counts match
```

---

## Implementation Details

### Work Item 1: Order Idempotency Guards

**Note:** If section-01-stop-loss has already been implemented, the idempotency guards in `trade_service.py` should already exist. Verify before implementing. If they exist, skip to Work Item 2.

**Files to modify:**
- `src/quantstack/execution/trade_service.py` — Add `client_order_id` deduplication

**Approach:** Maintain a set of recently submitted `client_order_id` values. Before submitting any order, check if the `client_order_id` has already been seen. If so, log a warning and return the existing order result rather than re-submitting. The dedup set should be populated from the broker's open/recent orders on startup to handle the case where the process crashed after submission but before recording the ID locally.

Alpaca's API already supports `client_order_id` as an idempotency key — if you submit an order with a duplicate `client_order_id`, Alpaca rejects it. The guard at the application layer prevents even hitting the API with a duplicate, which avoids ambiguous error handling.

### Work Item 2: Daily pg_dump Backup Sidecar

**Files to create/modify:**
- New: `scripts/pg_backup.sh` — Daily pg_dump logic
- New: `scripts/pg_restore_test.sh` — Restore verification script
- `docker-compose.yml` — Add backup sidecar service

**Backup script (`scripts/pg_backup.sh`):**

The script should:
- Run `pg_dump` in custom format (`-Fc`) for efficient compression and selective restore
- Write to a mounted volume (e.g., `/backups/quantstack_YYYY-MM-DD_HH-MM.dump`)
- Retain 7 days of backups, delete older files
- Log success/failure to stdout (Docker captures it)
- Exit with non-zero status on failure so Docker health checks detect it

**Restore test script (`scripts/pg_restore_test.sh`):**

The script should:
- Accept a backup file path as argument
- Create a temporary database
- Restore the backup into it
- Run basic validation queries (table existence, row count sanity)
- Drop the temporary database
- Report pass/fail

**Docker Compose sidecar:**

Add a service to `docker-compose.yml` that:
- Uses the `postgres` image (same version as the main DB)
- Mounts the same network to reach the DB
- Runs `scripts/pg_backup.sh` via cron or a simple sleep loop
- Schedules execution at 01:00 ET daily
- Mounts a `/backups` volume for dump storage
- Has `restart: unless-stopped` policy

Example service definition shape:

```yaml
backup:
  image: postgres:16
  depends_on:
    - postgres
  volumes:
    - ./scripts/pg_backup.sh:/usr/local/bin/pg_backup.sh:ro
    - backup_data:/backups
  environment:
    - PGHOST=postgres
    - PGDATABASE=quantstack
    - PGUSER=${POSTGRES_USER}
    - PGPASSWORD=${POSTGRES_PASSWORD}
  entrypoint: ["/bin/sh", "-c", "# cron or sleep-loop calling pg_backup.sh at 01:00 ET"]
  restart: unless-stopped
```

Add `backup_data` to the `volumes:` section at the bottom of docker-compose.yml.

**WAL archiving** for point-in-time recovery is deferred to Phase 4. Daily full dumps are sufficient for the current paper trading stage.

### Work Item 3: PostgresSaver Migration

**Files to modify:**
- `src/quantstack/graphs/trading/graph.py` — Swap MemorySaver → AsyncPostgresSaver
- `src/quantstack/graphs/research/graph.py` — Same
- `src/quantstack/graphs/supervisor/graph.py` — Same

**Approach: Direct cutover (not gradual migration).** This is a stakeholder decision based on the fact that the system is paper-trading only. If state is corrupted, restart the graph cycle fresh. No MemorySaver state needs to be migrated.

**Package dependency:** `langgraph-checkpoint-postgres` must be in `pyproject.toml` dependencies. Verify it exists; add if missing.

**Connection pool setup:**

Each graph gets its own `AsyncConnectionPool` from `psycopg_pool`. Pool sizing: 2-5 connections per graph. With 3 graphs, that is 6-15 new PG connections. Verify PostgreSQL `max_connections` accommodates this plus existing application connections (data acquisition, signal engine, scheduler). The default PG `max_connections` is 100, which should be sufficient, but verify.

**Graph initialization pattern (conceptual):**

```python
# In each graph.py build function:
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

async def build_graph():
    pool = AsyncConnectionPool(
        conninfo=pg_url,
        min_size=2,
        max_size=5,
    )
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # Idempotent — creates tables if missing

    graph = builder.compile(checkpointer=checkpointer)
    return graph
```

**Thread ID configuration:** The `thread_id` in LangGraph's config identifies a graph run for checkpoint recovery. On restart, passing the same `thread_id` resumes from the last completed super-step. Use a stable identifier per graph type (e.g., `"trading-main"`, `"research-main"`, `"supervisor-main"`) so restarts naturally resume.

**Crash recovery behavior:** When a graph resumes from a checkpoint:
- Nodes that completed before the crash are not re-executed
- The node that was in-progress during the crash re-executes from the start of that super-step
- This is why idempotency guards (Work Item 1) are critical — `execute_entries` may re-run

**Pool lifecycle:** The connection pool must be properly closed on shutdown. Use the pool as an async context manager, or explicitly call `await pool.close()` in the shutdown handler. Leaking pool connections will exhaust PG connection slots over multiple restarts.

---

## Verification Checklist

After implementation, verify:

- [ ] `pg_backup.sh` runs successfully and produces a valid dump file
- [ ] `pg_restore_test.sh` can restore the dump to a fresh database
- [ ] Backup sidecar runs in Docker Compose and produces daily dumps
- [ ] 7-day retention works (older files deleted)
- [ ] All three graphs use AsyncPostgresSaver (no MemorySaver references remain)
- [ ] `checkpointer.setup()` is called on startup and is idempotent
- [ ] Each graph has its own connection pool
- [ ] Graph resumes from last super-step after container restart
- [ ] Duplicate `client_order_id` submissions are rejected (not re-executed)
- [ ] The critical integration test passes: kill mid-cycle → restart → no duplicate orders

## Rollback

- **PostgresSaver:** Revert the three `graph.py` files to use `MemorySaver`. Idempotency guards in `trade_service.py` remain — they are valuable regardless of checkpointer choice.
- **Backup sidecar:** Remove the service from `docker-compose.yml`. No other components depend on it.
