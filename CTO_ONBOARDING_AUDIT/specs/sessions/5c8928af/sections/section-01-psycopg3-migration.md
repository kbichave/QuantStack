# Section 01: psycopg3 Migration

## Why This Exists

QuantStack's database layer (`src/quantstack/db.py`) uses psycopg2 via `ThreadedConnectionPool`. LangGraph's `PostgresSaver` (needed for durable checkpoints in Section 06) requires psycopg3 (`psycopg[binary,pool]`). Rather than maintain two connection drivers side-by-side, this migration replaces psycopg2 entirely with psycopg3 across the entire codebase.

This is a **foundational change** -- Sections 06 (durable checkpoints) and 10 (transaction isolation) both depend on this migration being complete before they can begin.

## Scope

The migration touches every file that imports psycopg2. The full list:

**Primary (connection infrastructure):**
- `src/quantstack/db.py` -- connection pool, `PgConnection` wrapper, all DDL migrations

**Application code:**
- `src/quantstack/data/pg_storage.py` -- direct `psycopg2.extras` usage
- `src/quantstack/rag/query.py` -- RAG query layer, uses `psycopg2` and `psycopg2.extras`
- `src/quantstack/health/langfuse_retention.py` -- health check queries (deferred import)

**Scripts:**
- `scripts/ewf_analyzer.py` -- standalone EWF analysis script
- `scripts/heartbeat.sh` -- embedded Python snippet with psycopg2 import
- `stop.sh` -- embedded Python snippet with psycopg2 import

**Tests (8 files):**
- `tests/integration/test_ic_decay_demotion.py`
- `tests/integration/test_blitz_mode.py`
- `tests/integration/test_sizing_pipeline.py`
- `tests/core/execution/test_tca_feedback.py`
- `tests/unit/test_rag_pipeline.py`
- `tests/unit/test_research_wip.py`
- `tests/unit/test_migrations.py`
- `tests/unit/test_ewf_schema.py`

---

## Tests (Write These First)

All tests go in `tests/unit/test_psycopg3_migration.py` unless noted.

### Connection pool behavior

```python
# Test: ConnectionPool (psycopg3) initializes with correct min_size/max_size
# Verify: pool created with min_size=4, max_size=20 (or PG_POOL_MAX env override)

# Test: pool respects max_size -- acquiring max_size+1 connections blocks or raises timeout
# Verify: attempting to exceed the pool raises PoolTimeout (not silently unbounded)

# Test: idle connections are recycled after max_idle seconds
# Verify: a connection idle beyond max_idle is replaced on next acquisition

# Test: context manager (pg_conn) returns connection to pool on clean exit
# Verify: pool.get_stats() shows connection returned after `with pg_conn()` block

# Test: context manager returns connection to pool on exception
# Verify: connection returned even when block raises; no leaked connections
```

### PgConnection wrapper

```python
# Test: execute() handles %s placeholders correctly
# Verify: parameterized INSERT with %s works, data round-trips correctly

# Test: execute() handles ? placeholders via translation (backward compat)
# Verify: ? in query string translated to %s before execution

# Test: fetchone() returns dict (not RealDictRow) -- key access works
# Verify: row["column_name"] returns expected value; row is a plain dict

# Test: fetchall() returns list[dict]
# Verify: all rows accessible by column name

# Test: fetchdf() returns pandas DataFrame with correct column names
# Verify: DataFrame columns match cursor.description, values match inserted data

# Test: OperationalError on execute triggers retry with fresh connection
# Verify: broken connection discarded, new connection acquired, query succeeds on retry
```

### JSON handling

```python
# Test: JSON columns deserialized as raw strings (matching psycopg2 custom adapter behavior)
# Verify: SELECT on a JSONB column returns str, not dict
# Context: psycopg2 had register_default_json(loads=lambda x: x) to return raw strings.
#          psycopg3 equivalent must produce identical behavior.
```

### Regression (lint-style checks)

```python
# Test: no integer-indexed row access remains in codebase
# Verify: grep for row[0], row[1], result[0], etc. in src/ -- zero hits
#         (psycopg3 dict_row returns dicts, integer indexing silently breaks)

# Test: no psycopg2 imports remain in codebase
# Verify: grep for "import psycopg2" in src/, scripts/, tests/ -- zero hits
```

---

## Implementation Details

### Step 1: Update Dependencies

Add to `pyproject.toml` (or equivalent dependency file):

```
psycopg[binary,pool]>=3.1
```

Remove:
```
psycopg2-binary
```

Note: `psycopg[binary]` bundles the C library. `psycopg[pool]` brings in `psycopg_pool.ConnectionPool`.

### Step 2: Replace the Connection Pool in `db.py`

**Current code** (lines 77-92):
```python
_pg_pool: psycopg2.pool.ThreadedConnectionPool | None = None
```

**New implementation:**

Replace with `psycopg_pool.ConnectionPool`. Key parameter mapping:

| psycopg2 `ThreadedConnectionPool` | psycopg3 `ConnectionPool` |
|---|---|
| `minconn=1` | `min_size=4` |
| `maxconn=20` | `max_size=20` |
| (no equivalent) | `max_lifetime=3600` (recycle connections after 1 hour) |
| (no equivalent) | `max_idle=600` (close idle connections after 10 min) |

The pool constructor takes a `conninfo` string (same DSN format as psycopg2) and a `kwargs` dict for connection defaults. Set `autocommit=True` as the pool-level default for PostgresSaver compatibility; application queries use explicit transactions via `with conn.transaction()`.

The pool is thread-safe by default in psycopg3 -- no need for `ThreadedConnectionPool` distinction.

`reset_pg_pool()` calls `pool.close()` (not `pool.closeall()` -- API difference).

### Step 3: Rewrite the PgConnection Wrapper

The `PgConnection` class (lines 110-352) wraps the raw connection. Key changes:

**Constructor:** Accept `psycopg_pool.ConnectionPool` instead of `ThreadedConnectionPool`. Store `self._pool` and lazily acquire via `self._pool.getconn()`.

**`_ensure_raw()`:** Same pattern -- lazy acquisition. The idle-in-transaction timeout is set identically (`SET idle_in_transaction_session_timeout = '30s'`). Use `conn.autocommit = True` temporarily to execute the SET, then switch back.

**`_translate()` static method:** Keep as-is -- the `?` to `%s` translation is backward-compat. psycopg3 still supports `%s` placeholders in its default (non-C) adaptation mode.

**`execute()`:** Replace `psycopg2.OperationalError` with `psycopg.OperationalError`. The retry-on-broken-connection logic is identical in structure. Replace `self._pool.putconn(self._raw, close=True)` with `self._pool.putconn(self._raw)` (psycopg3's pool handles broken connection disposal internally, but explicitly closing is still needed to signal the pool).

**Row factory:** Set `row_factory=dict_row` on cursors so that `fetchone()` returns `dict` and `fetchall()` returns `list[dict]`. Import: `from psycopg.rows import dict_row`. This replaces psycopg2's `RealDictCursor`.

**CRITICAL: `dict_row` vs `RealDictRow` behavioral difference.** psycopg2's `RealDictRow` supports BOTH key access (`row["col"]`) AND integer index access (`row[0]`). psycopg3's `dict_row` returns a plain Python `dict` which does NOT support integer indexing. Any code using `row[0]`, `row[1]`, `result[0]` etc. on DB results will silently break or raise `KeyError`. This is the highest-risk change in the migration.

**`fetchdf()`:** `cursor.description` works identically in psycopg3. No change needed beyond the row factory setting.

**`release()`:** Replace `self._pool.putconn(self._raw)` with the psycopg3 equivalent. `ConnectionPool.putconn()` exists in psycopg3's pool.

### Step 4: JSON Handling

The current code (lines 61-62) registers custom JSON/JSONB deserializers to return raw strings:

```python
psycopg2.extras.register_default_json(loads=lambda x: x)
psycopg2.extras.register_default_jsonb(loads=lambda x: x)
```

In psycopg3, JSON handling is controlled via type adapters. To achieve the same "return raw strings" behavior:

```python
from psycopg.types.json import set_json_loads

set_json_loads(lambda x: x)  # Return JSON/JSONB as raw strings
```

Call this at module level in `db.py`, replacing the psycopg2 registration lines.

### Step 5: Import Statement Migration

For each file in the scope list above, mechanically replace:

| psycopg2 | psycopg3 |
|---|---|
| `import psycopg2` | `import psycopg` |
| `import psycopg2.extras` | `from psycopg.rows import dict_row` (if using RealDictCursor) |
| `import psycopg2.extensions` | (usually not needed -- types are in `psycopg` directly) |
| `import psycopg2.pool` | `from psycopg_pool import ConnectionPool` |
| `psycopg2.OperationalError` | `psycopg.OperationalError` |
| `psycopg2.errors.XxxError` | `psycopg.errors.XxxError` (same submodule name) |
| `psycopg2.extras.RealDictCursor` | `from psycopg.rows import dict_row` (cursor factory) |

### Step 6: Integer-Indexed Row Access Audit

Grep the entire codebase for patterns like `row[0]`, `row[1]`, `result[0]`, `fetchone()[0]`, etc. in files that consume DB results. Each hit must be converted to key-based access (`row["column_name"]`).

Known location from the migration code itself: `run_migrations_pg()` line 489 uses `result[0]` on an advisory lock result. This needs to become either:
- `result["pg_try_advisory_lock"]` (if using dict_row), or
- Use a tuple_row factory for that specific cursor (advisory lock queries are infrastructure, not application)

The migration function's internal cursors (lines 481-548) use raw cursors without the dict_row factory. These can stay as tuple-returning cursors since they're infrastructure code that directly controls cursor creation.

### Step 7: Placeholder Audit

The codebase uses both `?` and `%s` placeholders. The `_translate()` method handles backward compatibility by replacing `?` with `%s`. After migration, both still work identically in psycopg3's `%s`-based parameter mode.

Optional cleanup: search for `?` placeholders in SQL strings and replace with `%s` directly, then remove the `_translate()` method. This is a nice-to-have, not blocking.

### Step 8: Shell Script Updates

`scripts/heartbeat.sh` and `stop.sh` contain embedded Python snippets that `import psycopg2`. Update these to `import psycopg` with the matching connection API.

### Step 9: Test Fixture Updates

`tests/conftest.py` imports `quantstack.db` which triggers the pool initialization. After the migration, this import will use psycopg3 automatically. Test files that directly import psycopg2 (8 files listed above) need the same mechanical import replacement.

---

## Connection Budget

After migration, the total connection budget is:

| Consumer | Connections |
|---|---|
| Main application pool | 4-20 (min_size to max_size) |
| Checkpointer pool (Section 06) | 2-6 |
| Scheduler | 1-2 |
| Backup jobs | 1 |
| **Total maximum** | **~29** |

PostgreSQL default `max_connections` is 100. Budget is well within limits. Document this in the db.py module docstring.

---

## Risks and Failure Modes

1. **Integer-indexed row access breaks silently.** `dict[0]` on a dict raises `KeyError`, not a type error. This will surface as runtime exceptions, not import-time failures. The grep-based lint test catches this before deployment.

2. **JSON behavior difference.** If `set_json_loads(lambda x: x)` is not set, psycopg3 will parse JSON into Python objects by default. This would break all downstream code that calls `json.loads()` on DB-fetched JSON columns (double-parse). The unit test for JSON handling catches this.

3. **`executemany()` behavior.** psycopg3's `executemany()` uses pipeline mode internally for better performance. Behavior should be identical, but test any bulk-insert code paths.

4. **Async implications.** psycopg3 has native async support (`AsyncConnectionPool`). This migration keeps the sync pool only. The async execution monitor could benefit from `AsyncConnectionPool` later, but that is out of scope for this section.

---

## Dependencies

- **Depends on:** Nothing (this is the foundation)
- **Blocks:** Section 06 (Durable Checkpoints -- needs psycopg3 for PostgresSaver), Section 10 (Transaction Isolation -- uses `conn.transaction()` context manager from psycopg3)

---

## Verification Checklist

1. `import psycopg2` grep returns zero results in `src/`, `scripts/`, `tests/`
2. All existing DB tests pass (`uv run pytest tests/unit tests/integration -x`)
3. Connection pool creates and destroys cleanly (no leaked connections in test teardown)
4. JSON columns return raw strings (not parsed dicts)
5. `fetchone()` returns `dict`, not `RealDictRow`
6. No `row[0]` patterns remain in DB-consuming code
7. Shell scripts (`heartbeat.sh`, `stop.sh`) execute without import errors
