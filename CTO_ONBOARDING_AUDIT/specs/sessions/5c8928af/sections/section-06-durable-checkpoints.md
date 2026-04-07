# Section 06: Durable Checkpoints (PostgresSaver)

## Problem

All three LangGraph graph runners (trading, research, supervisor) use `MemorySaver` for mid-cycle checkpointing. This is in-process memory only. If a container crashes or restarts mid-cycle, all intermediate node state is lost. A crash during `execute_entries` -- after risk approval but before order submission -- leaves the system in an inconsistent state: approved trades never executed, no record of the approval, no ability to resume.

The `graph_checkpoints` PostgreSQL table that already exists only captures final cycle outcomes (graph name, cycle number, duration, status). It does not store intermediate node state and cannot be used for crash recovery.

## Dependency

This section depends on **section-01-psycopg3-migration**. PostgresSaver from `langgraph-checkpoint-postgres` requires psycopg3. The psycopg3 migration must be complete before this work begins.

## Tests First

Write these tests before any implementation. They define the acceptance criteria.

### Checkpointer Factory Tests

```python
# tests/unit/test_checkpointing.py

# Test: create_checkpointer() returns a configured PostgresSaver instance
# Test: the checkpointer uses a psycopg3 ConnectionPool (not a single connection)
# Test: pool sizing is min_size=2, max_size=6
```

### Runner Integration Tests

```python
# tests/integration/test_durable_checkpoints.py

# Test: trading_runner uses PostgresSaver (not MemorySaver)
# Test: research_runner uses PostgresSaver
# Test: supervisor_runner uses PostgresSaver
# Test: thread_id format is unchanged: "{graph_name}-{YYYY-MM-DD}-cycle-{number}"
```

### Crash Recovery Tests

```python
# tests/integration/test_checkpoint_recovery.py

# Test: kill container mid-cycle -> restart -> graph resumes from last checkpoint
# Test: node that crashed mid-execution re-executes on resume
# Test: completed nodes are NOT re-executed on resume
```

### Retention / Pruning Tests

```python
# tests/unit/test_checkpoint_pruning.py

# Test: checkpoint pruning removes rows older than 48 hours
# Test: pruning preserves the most recent completed cycle per graph
# Test: pruning preserves in-progress cycles (do not delete active state)
# Test: pruning runs as a scheduled job without errors
```

## Implementation

### 1. Install the Dependency

The package `langgraph-checkpoint-postgres` is already listed as an optional dependency in `pyproject.toml` under the `[langgraph]` extras group:

```
langgraph-checkpoint-postgres>=3.0.0
psycopg[binary]>=3.1.0
psycopg-pool>=3.1.0
```

After section-01 completes the psycopg3 migration, these will be available in the runtime environment. Verify with `python -c "from langgraph.checkpoint.postgres import PostgresSaver"`.

### 2. Create a Shared Checkpointer Factory

Create a new module at `src/quantstack/checkpointing.py`. This keeps checkpoint concerns separate from the general `db.py` connection pool.

The factory function signature:

```python
# src/quantstack/checkpointing.py

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

def create_checkpointer() -> PostgresSaver:
    """Create a PostgresSaver backed by a dedicated psycopg3 connection pool.

    Pool is sized for checkpoint operations: min_size=2, max_size=6.
    This is intentionally smaller than the main application pool (max_size=20)
    because checkpoint writes are less frequent than application queries.

    setup() is NOT called here. Table creation is a deployment step,
    not a per-startup step. See the setup_checkpoint_tables() function.
    """
    ...
```

Key design decisions for the factory:

- **Dedicated pool, not shared.** The checkpointer gets its own `ConnectionPool` with `min_size=2, max_size=6`. This isolates checkpoint I/O from application queries. The total connection budget becomes: main pool (max 20) + checkpointer pool (max 6) + scheduler + backup = ~28 connections against PostgreSQL's default max_connections of 100. Document this budget in a comment.
- **Connection string from environment.** Read `TRADER_PG_URL` (the same env var used by `db.py`). The checkpointer pool connects to the same database.
- **No `setup()` in the factory.** If three runners start simultaneously and each calls `setup()`, they race on table creation. Instead, provide a separate function for the deployment step.

Also provide a setup function:

```python
def setup_checkpoint_tables() -> None:
    """Create the 4 PostgresSaver tables if they don't exist.

    Run this once as a deployment/migration step, not on every startup.
    Safe to call multiple times (idempotent CREATE IF NOT EXISTS).
    """
    ...
```

### 3. Add a Migration / Setup Step

PostgresSaver requires 4 tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`). These must exist before any runner starts.

Two approaches (pick one during implementation):

**Option A: Management command.** Add a script `scripts/setup_checkpoints.py` that calls `setup_checkpoint_tables()`. Run it once during deployment. Add to deployment documentation.

**Option B: Entrypoint flag.** If an `entrypoint.sh` is created (or already exists), add a `--migrate` flag or `RUN_MIGRATIONS=true` env var that runs `setup_checkpoint_tables()` before launching the runner. Only one container needs this flag set; the others skip it.

Option A is simpler and more explicit. The project does not currently have an `entrypoint.sh`, so creating one just for this is unnecessary overhead. A standalone script is the right call.

### 4. Update All Three Runners

Each runner currently follows the same pattern (shown here for trading_runner.py, lines 161-167):

```python
from langgraph.checkpoint.memory import MemorySaver
# ...
checkpointer = MemorySaver()
```

Replace with:

```python
from quantstack.checkpointing import create_checkpointer
# ...
checkpointer = create_checkpointer()
```

The three files to modify:

- `src/quantstack/runners/trading_runner.py` (line ~161-167)
- `src/quantstack/runners/research_runner.py` (line ~28-34)
- `src/quantstack/runners/supervisor_runner.py` (line ~28-34)

The thread_id format must remain unchanged: `{graph_name}-{YYYY-MM-DD}-cycle-{cycle_number}` (used in `trading_runner.py` line 80). PostgresSaver uses this thread_id to associate checkpoint state with a specific graph run. On restart, passing the same thread_id causes LangGraph to resume from the last checkpoint rather than starting fresh.

The `build_*_graph()` functions already accept a `BaseCheckpointSaver` (via the type hint in each graph.py's `build_*_graph` signature). PostgresSaver is a subclass of `BaseCheckpointSaver`, so no changes needed to the graph builders.

### 5. Crash Recovery: How It Works

PostgresSaver writes to the `checkpoint_writes` table after each node completes. On restart with the same thread_id:

1. LangGraph queries the checkpoint tables for that thread_id
2. It reconstructs state from the last committed checkpoint plus any pending writes
3. The node that was executing when the crash occurred re-executes from the beginning of that node (not mid-node)
4. Nodes that completed before the crash are skipped -- their outputs are already in checkpoint state

This means node functions must be safe to re-execute. Most already are (they query external state and compute fresh). The critical case is `execute_entries`: if a crash occurs after the broker accepts an order but before the checkpoint write commits, the order exists at the broker but the node re-executes on restart. The startup reconciliation from section-02 (which checks for open positions without stops) provides a safety net here.

No code changes are needed to enable crash recovery -- it is a built-in behavior of PostgresSaver. The test suite validates it works correctly in this codebase.

### 6. Checkpoint Data Retention and Pruning

PostgresSaver stores full state at every node transition. Volume estimate:

- Trading graph: ~16 nodes per cycle, cycles every ~5 minutes = ~4,600 checkpoint rows/day
- Research graph: fewer nodes, longer cycles = ~1,000 rows/day
- Supervisor graph: ~500 rows/day
- **Total: ~6,000 rows/day, ~180,000 rows/month**

Without pruning, this grows indefinitely and degrades query performance.

Implement a pruning function:

```python
# src/quantstack/checkpointing.py

def prune_old_checkpoints(retention_hours: int = 48) -> int:
    """Delete checkpoint data older than retention_hours.

    Preserves:
    - The most recent completed cycle per graph (regardless of age)
    - Any in-progress cycles (incomplete checkpoint sequences)

    Returns the number of rows deleted.
    """
    ...
```

Schedule this as a daily job in the scheduler (`scripts/scheduler.py`). Run it at a low-activity time (e.g., 03:00 UTC, after the backup job at 02:00 UTC).

The 48-hour retention window means:
- Any crash within the last 2 days can resume from checkpoint
- Older cycles are only recoverable from the `graph_checkpoints` table (operational metadata, not full state) and database backups (section-08)
- This is sufficient because graph cycles are short (5-60 minutes) and a 2-day-old checkpoint is not useful for recovery

### 7. Interaction with Existing `graph_checkpoints` Table

The existing `graph_checkpoints` table (created in `db.py` line ~1851) stores operational metadata: graph_name, cycle_number, duration_seconds, status, created_at. This is used by the dashboard and TUI for monitoring.

PostgresSaver's tables are separate and serve a different purpose: they store LangGraph's internal state for crash recovery and resumption.

**Keep both.** They serve complementary purposes:
- `graph_checkpoints` = "what happened" (operational dashboards, monitoring)
- PostgresSaver tables = "where to resume" (crash recovery, state reconstruction)

### 8. Test Infrastructure Update

The integration test file `tests/integration/test_graph_trajectory.py` (line 274) currently uses `MemorySaver` for test graph compilation. Update this to use `MemorySaver` intentionally for unit/fast tests (no DB dependency) but add separate integration tests that use `PostgresSaver` via the factory. The test conftest at `tests/integration/conftest.py` (line 64) also imports `MemorySaver` and will need a fixture that provides both options.

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/checkpointing.py` | Checkpointer factory, setup function, pruning function |
| `scripts/setup_checkpoints.py` | One-time migration script to create PostgresSaver tables |
| `tests/unit/test_checkpointing.py` | Factory and pruning unit tests |
| `tests/integration/test_durable_checkpoints.py` | Runner integration tests |
| `tests/integration/test_checkpoint_recovery.py` | Crash recovery integration tests |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/runners/trading_runner.py` | Replace `MemorySaver` import and instantiation with `create_checkpointer()` |
| `src/quantstack/runners/research_runner.py` | Same replacement |
| `src/quantstack/runners/supervisor_runner.py` | Same replacement |
| `scripts/scheduler.py` | Add daily checkpoint pruning job |
| `tests/integration/test_graph_trajectory.py` | Keep `MemorySaver` for fast tests; add PostgresSaver variant |
| `tests/integration/conftest.py` | Add `checkpointer` fixture that provides PostgresSaver for integration tests |

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| PostgresSaver adds latency to graph execution | Slower trading cycles (each node transition writes to DB) | Benchmark before/after. LangGraph checkpoint writes are designed to be lightweight. The dedicated pool prevents contention with application queries. |
| Multiple runners calling `setup()` simultaneously | Race condition on table creation | Never call `setup()` from runners. Use a separate migration script run once during deployment. |
| Checkpoint table bloat without pruning | Degraded DB performance, disk exhaustion | 48-hour retention with daily pruning job. Monitor table size in supervisor health checks. |
| Connection pool exhaustion | Checkpointer and application pool compete for connections | Separate pools with documented budget (28 of 100 max connections). Monitor with `pg_stat_activity`. |
| Node re-execution on crash recovery triggers duplicate orders | Duplicate trades at broker | Startup reconciliation (section-02) detects existing positions. Alpaca's `client_order_id` provides idempotency for order submission. |

## Verification Checklist

After implementation, verify:

1. `python scripts/setup_checkpoints.py` creates the 4 PostgresSaver tables without error
2. All three runners start and complete a full cycle with PostgresSaver (no `MemorySaver` references remain in runner code)
3. Thread IDs in the checkpoint tables match the expected format
4. Killing a runner mid-cycle and restarting it resumes from the last completed node
5. The pruning job runs and deletes old checkpoint rows while preserving recent and in-progress cycles
6. Existing `graph_checkpoints` table continues to receive operational metadata as before
7. Dashboard and TUI continue to work (they read `graph_checkpoints`, not PostgresSaver tables)
