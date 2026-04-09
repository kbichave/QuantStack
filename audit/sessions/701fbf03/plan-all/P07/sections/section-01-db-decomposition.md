# Section 01: db.py Decomposition

## Objective

Extract connection management and migration logic from the monolithic `db.py` (3,473 LOC) into separate modules under a `db/` package, while maintaining full backward compatibility for all existing imports.

## Dependencies

None — this section can be implemented independently.

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `src/quantstack/db/__init__.py` | Package init that re-exports everything from `connection.py` and `migrations.py` so existing `from quantstack.db import ...` statements continue to work unchanged |
| `src/quantstack/db/connection.py` | Connection pool setup, `pg_conn()` / `db_conn()`, `get_pool()`, `reset_pg_pool()`, the `_DictRow` class, `_compat_row` factory, JSON behaviour config, URL resolution |
| `src/quantstack/db/migrations.py` | All `CREATE TABLE IF NOT EXISTS` blocks extracted as named migration functions. `run_migrations()` function that executes them in order |

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/db.py` | Remove (replaced by `db/` package). If risk-averse, keep as a thin shim that imports from `db/` for one release cycle |

## Implementation Details

### Step 1: Create `src/quantstack/db/` package directory

### Step 2: Extract `connection.py`

Move from `db.py`:
- `_DictRow` class and `_compat_row` factory
- `set_json_loads(...)` call (JSON behaviour override)
- `_resolve_pg_url()` function
- `_pg_pool`, `_pg_pool_lock`, `_get_pg_pool()`, `reset_pg_pool()`
- `pg_conn()` context manager
- `db_conn = pg_conn` alias
- All related imports (`contextlib`, `os`, `threading`, `psycopg`, `psycopg_pool`, `loguru`)

The connection module must be fully self-contained — no imports from other `quantstack.db` submodules.

### Step 3: Extract `migrations.py`

Move from `db.py`:
- All `CREATE TABLE IF NOT EXISTS` DDL blocks
- Convert each into a named function: `def migration_001_create_positions(conn): ...`
- The `run_migrations(conn)` function that calls them in order
- Import `pg_conn` from `quantstack.db.connection`

### Step 4: Create `__init__.py` with re-exports

```python
from quantstack.db.connection import (
    pg_conn,
    db_conn,
    reset_pg_pool,
    _get_pg_pool,
    _resolve_pg_url,
)
from quantstack.db.migrations import run_migrations

# Preserve all existing public API
__all__ = [
    "pg_conn",
    "db_conn",
    "reset_pg_pool",
    "run_migrations",
]
```

### Step 5: Remove old `db.py`

After verifying all imports resolve, delete `src/quantstack/db.py`.

### Important Constraints

- **Zero behavior change.** This is a pure structural extraction.
- Every module that currently does `from quantstack.db import db_conn` or `from quantstack.db import pg_conn` must continue to work without modification.
- The `_DictRow` class and `_compat_row` factory are internal but used by the pool configuration — they must move with the connection code.
- The `set_json_loads(...)` call at module level must execute when the connection module is first imported (same as today).

## Test Requirements

### TDD Tests (write before implementation)

```python
# Test: from quantstack.db import db_conn still works
def test_db_conn_importable_from_package():
    from quantstack.db import db_conn
    assert callable(db_conn)

# Test: from quantstack.db.connection import db_conn works
def test_db_conn_importable_from_connection_module():
    from quantstack.db.connection import db_conn
    assert callable(db_conn)

# Test: from quantstack.db.connection import pg_conn works
def test_pg_conn_importable_from_connection_module():
    from quantstack.db.connection import pg_conn
    assert callable(pg_conn)

# Test: run_migrations importable from both paths
def test_run_migrations_importable():
    from quantstack.db import run_migrations
    from quantstack.db.migrations import run_migrations as rm2
    assert run_migrations is rm2
```

### Verification

- Run `grep -r "from quantstack.db import" src/` and confirm every import still resolves.
- Run `grep -r "from quantstack.db " src/` for any attribute-style imports.
- Run the full test suite — zero regressions expected.

## Acceptance Criteria

1. `src/quantstack/db/` is a proper Python package with `__init__.py`, `connection.py`, `migrations.py`
2. `from quantstack.db import db_conn`, `from quantstack.db import pg_conn`, `from quantstack.db import run_migrations` all work
3. `from quantstack.db.connection import pg_conn` and `from quantstack.db.migrations import run_migrations` also work
4. No existing file outside of `db/` requires import changes
5. All existing tests pass without modification
6. Old `src/quantstack/db.py` file is removed (not left alongside the package)
