# Section 02: Schema Migration Versioning

## Objective

Add a `schema_migrations` table and a migration runner that tracks which migrations have been applied, executes pending ones in order, and prevents duplicate execution via checksum tracking. Replaces the current "run all CREATE IF NOT EXISTS on every startup" pattern with an ordered, auditable migration system.

## Dependencies

- **section-01-db-decomposition** — migrations.py must exist as the target module.

## Files to Create/Modify

### Files to Modify

| File | Change |
|------|-------------|
| `src/quantstack/db/migrations.py` | Add `schema_migrations` DDL, `run_migrations()` with version tracking, checksum computation, and ordered execution |

### No New Files

The migration runner lives inside `migrations.py` (created in Section 01). No external framework (Alembic, etc.) is used — this is intentional per the plan's anti-goals.

## Implementation Details

### Step 1: schema_migrations Table

Add as the very first migration (bootstrap):

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    checksum TEXT
);
```

### Step 2: Migration Registry

Each migration is a named function with a predictable signature:

```python
def migration_001_create_positions(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS positions (...)""")

def migration_002_create_strategies(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS strategies (...)""")

# ... one function per existing CREATE TABLE block in db.py
```

All existing DDL blocks from the old `run_migrations(conn)` become numbered functions. Order matters — tables with foreign key dependencies must come after their parents.

A registry list defines execution order:

```python
_MIGRATIONS = [
    migration_001_create_positions,
    migration_002_create_strategies,
    # ...
]
```

### Step 3: Checksum Computation

Checksum is SHA-256 of the function's source code:

```python
import hashlib
import inspect

def _checksum(fn) -> str:
    source = inspect.getsource(fn)
    return hashlib.sha256(source.encode()).hexdigest()
```

Purpose: detect if a migration function was modified after being applied. Log a warning but do NOT re-run — manual intervention required for schema changes to already-applied migrations.

### Step 4: Migration Runner

```python
def run_migrations(conn) -> None:
    """Execute pending migrations in order. Track in schema_migrations."""
    # 1. Ensure schema_migrations table exists (bootstrap)
    conn.execute(SCHEMA_MIGRATIONS_DDL)
    
    # 2. Query already-applied migrations
    applied = {row["migration_name"]: row["checksum"]
               for row in conn.execute("SELECT migration_name, checksum FROM schema_migrations").fetchall()}
    
    # 3. For each registered migration:
    for fn in _MIGRATIONS:
        name = fn.__name__
        current_checksum = _checksum(fn)
        
        if name in applied:
            if applied[name] != current_checksum:
                logger.warning(
                    "[Migrations] Checksum mismatch for %s — source changed after apply. "
                    "Manual review required.", name
                )
            continue  # Already applied — skip
        
        # 4. Execute migration
        logger.info("[Migrations] Applying %s ...", name)
        fn(conn)
        
        # 5. Record in schema_migrations
        conn.execute(
            "INSERT INTO schema_migrations (migration_name, checksum) VALUES (%s, %s)",
            [name, current_checksum],
        )
        logger.info("[Migrations] Applied %s ✓", name)
```

### Step 5: Startup Integration

The existing startup code that calls `run_migrations(conn)` continues to work — the function signature is unchanged. Internally it now checks `schema_migrations` before executing each DDL block.

### Important Constraints

- **Anti-goal: No Alembic.** The `schema_migrations` table + named functions pattern is sufficient.
- **Idempotency preserved.** All DDL uses `IF NOT EXISTS`. Even if the tracker is lost, re-running is safe.
- **Checksum is advisory only.** A mismatch logs a warning, never auto-reruns. Schema changes to applied migrations require manual migration scripts.
- **Transaction safety.** Each migration should run in its own transaction (or the entire batch in one transaction, depending on PostgreSQL DDL transaction support — CREATE TABLE is transactional in PG).

## Test Requirements

### TDD Tests

```python
# Test: run_migrations executes pending and skips applied
def test_run_migrations_executes_pending(pg_conn):
    """First run applies all. Second run applies none."""
    run_migrations(pg_conn)
    # Verify schema_migrations has entries
    rows = pg_conn.execute("SELECT migration_name FROM schema_migrations").fetchall()
    assert len(rows) > 0
    count_before = len(rows)
    
    # Second run — no new migrations
    run_migrations(pg_conn)
    rows_after = pg_conn.execute("SELECT migration_name FROM schema_migrations").fetchall()
    assert len(rows_after) == count_before

# Test: migration checksum prevents duplicate execution
def test_checksum_recorded(pg_conn):
    run_migrations(pg_conn)
    row = pg_conn.execute(
        "SELECT checksum FROM schema_migrations WHERE migration_name = %s",
        ["migration_001_create_positions"]
    ).fetchone()
    assert row is not None
    assert len(row["checksum"]) == 64  # SHA-256 hex

# Test: schema_migrations table tracks applied migrations
def test_schema_migrations_table_exists(pg_conn):
    run_migrations(pg_conn)
    row = pg_conn.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'schema_migrations')"
    ).fetchone()
    assert row[0] is True
```

## Acceptance Criteria

1. `schema_migrations` table is created on first `run_migrations()` call
2. Each migration is tracked by name with a SHA-256 checksum and timestamp
3. Previously-applied migrations are skipped on subsequent runs
4. Checksum mismatch logs a warning but does not re-execute
5. All existing tables are still created correctly (no regression)
6. `run_migrations(conn)` function signature is unchanged — callers need no updates
