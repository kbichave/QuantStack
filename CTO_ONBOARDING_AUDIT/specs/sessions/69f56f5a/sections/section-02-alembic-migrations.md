# Section 02: Alembic Migration Framework

## Background

QuantStack's database schema is managed by `src/quantstack/db.py` (2396 lines), which contains 37+ `_migrate_*_pg()` functions. Each function runs `CREATE TABLE IF NOT EXISTS` DDL. They are all called sequentially from `run_migrations_pg()` at startup, protected by a PostgreSQL advisory lock (key `5145534154`). A module-level `_migrations_done` flag prevents re-running within the same process.

There is no `schema_migrations` table, no `alembic_version` table, and no way to track which migrations have run, failed, or need re-running. Every startup re-executes all 37+ DDL blocks. Adding a column or altering a table requires editing the monolithic `db.py` and hoping the `IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS` pattern is sufficient.

The codebase uses raw SQL everywhere -- there are no SQLAlchemy ORM models, so Alembic's `--autogenerate` will not work. All migrations must be hand-written.

**This section blocks:** section-04-rate-limiter (which needs an Alembic migration for its `rate_limit_buckets` table).

**Dependencies:** None. This can be implemented in Batch 1.

---

## Tests First

All tests go in `tests/integration/test_alembic_migrations.py` and require the `@pytest.mark.integration` marker (they need a running PostgreSQL instance).

**Required fixtures:**
- `test_db` -- a fresh PostgreSQL database created/dropped per test session. The existing `conftest.py` may already provide database fixtures; if not, create one that creates a temporary database from `TRADER_PG_URL`, runs tests against it, and drops it on teardown.

```python
# tests/integration/test_alembic_migrations.py

import pytest

pytestmark = pytest.mark.integration


class TestBaselineMigration:
    """Verify the baseline migration that captures all 37+ existing tables."""

    def test_upgrade_head_on_empty_database_creates_all_tables(self, test_db):
        """Run 'alembic upgrade head' against an empty DB.

        Assert that all expected tables exist afterward. Compare the set of
        table names against a known list extracted from the _migrate_*_pg()
        functions (positions, strategies, signals, fills, audit, etc.).
        """

    def test_upgrade_head_on_existing_database_is_idempotent(self, test_db):
        """Run 'alembic upgrade head' twice in a row.

        The second run must succeed without errors. This validates that
        CREATE TABLE IF NOT EXISTS is used throughout the baseline.
        """

    def test_alembic_current_shows_correct_version_after_upgrade(self, test_db):
        """After upgrade head, 'alembic current' must report the baseline
        revision as the current head."""

    def test_downgrade_base_drops_all_tables(self, test_db):
        """Run upgrade head, then downgrade base.

        All application tables must be gone. Only alembic_version (empty)
        and system tables should remain. This is for development use only.
        """


class TestAdvisoryLock:
    """Verify that concurrent migrations are serialized."""

    def test_concurrent_upgrade_serialized(self, test_db):
        """Launch two alembic upgrade head calls from separate connections
        concurrently. One must proceed while the other waits or is skipped.
        Neither should error.
        """

    def test_advisory_lock_released_on_failure(self, test_db):
        """Simulate a migration failure (e.g., inject bad SQL).

        Verify the advisory lock is released in the finally block so
        subsequent migrations can proceed.
        """


class TestFallbackFlag:
    """Verify the USE_ALEMBIC transition flag."""

    def test_use_alembic_false_uses_old_path(self, test_db, monkeypatch):
        """With USE_ALEMBIC=false (or unset), run_migrations() must call
        the old _migrate_*_pg() functions, not alembic."""

    def test_use_alembic_true_uses_alembic_path(self, test_db, monkeypatch):
        """With USE_ALEMBIC=true, run_migrations() must call
        alembic.command.upgrade(config, 'head')."""

    def test_both_paths_produce_identical_schema(self, test_db):
        """Run old path on one DB, Alembic path on another.

        Compare pg_dump --schema-only output. The schemas must match
        (modulo the alembic_version table).
        """


class TestStartupIntegration:
    """Verify run_migrations() process-level dedup."""

    def test_run_migrations_only_runs_once_per_process(self, test_db):
        """Call run_migrations() twice. The second call must short-circuit
        via the _migrations_done flag without touching the DB."""
```

---

## Implementation Details

### 4.1 -- Alembic Directory Structure

Create the following files at the project root:

```
alembic/
  env.py
  script.py.mako
  versions/
    001_baseline.py
alembic.ini
```

**File: `alembic.ini`**

Standard Alembic config. The `sqlalchemy.url` is intentionally left empty because `env.py` reads `TRADER_PG_URL` from the environment at runtime.

Key settings:
- `script_location = alembic`
- `sqlalchemy.url =` (blank -- overridden in env.py)
- `file_template = %%(rev)s_%%(slug)s` (clean filenames)

**File: `alembic/env.py`**

Must do the following:
- Read `TRADER_PG_URL` from `os.environ` (same env var as `db.py`)
- Override `sqlalchemy.url` in the Alembic config with that value
- Acquire advisory lock `5145534154` (same key as `_MIGRATION_ADVISORY_LOCK` in `db.py`) before running migrations
- Release advisory lock after completion, even on failure (use `try/finally`)
- Support both online (connected) and offline (SQL generation) modes

The advisory lock must use `pg_advisory_lock()` (blocking), not `pg_try_advisory_lock()` (non-blocking). Alembic migrations should wait for the lock, not skip. The current `db.py` uses `pg_try_advisory_lock` and skips if locked -- that's fine for idempotent `IF NOT EXISTS` DDL, but Alembic migrations are not all idempotent, so waiting is the correct behavior.

**File: `alembic/script.py.mako`**

Standard Alembic template. No customization needed beyond the default.

### 4.2 -- Baseline Migration

**File: `alembic/versions/001_baseline.py`**

A single migration file whose `upgrade()` function contains the DDL from all 37+ `_migrate_*_pg()` functions in `db.py`. The approach:

1. Extract the `CREATE TABLE IF NOT EXISTS` statements from each function in `db.py`
2. Also extract any `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, and `CREATE EXTENSION IF NOT EXISTS` statements
3. Place them in `upgrade()` in the same order as `run_migrations_pg()` calls them
4. Use `op.execute()` for raw SQL (since there are no SQLAlchemy models)

The `downgrade()` function drops all tables in reverse dependency order. This is for development/testing only. Include a safety comment: `# WARNING: destroys all data. Never run in production.`

Critical detail: the baseline must capture the **exact** schema that the current `_migrate_*_pg()` functions produce. To verify:
- Run the old migration path on a fresh database
- Run `pg_dump --schema-only` to capture the schema
- Run the Alembic baseline on another fresh database
- Diff the two `pg_dump` outputs -- they must be identical (excluding `alembic_version`)

### 4.3 -- Startup Integration

**File to modify: `src/quantstack/db.py`**

Modify `run_migrations()` (line ~2405) and `run_migrations_pg()` (line ~460) to support the `USE_ALEMBIC` flag:

```python
def run_migrations(conn: PgConnection) -> None:
    """Run all migrations. Called once at startup. Idempotent.

    If USE_ALEMBIC=true, delegates to Alembic's upgrade head.
    Otherwise falls back to the legacy _migrate_*_pg() path.
    """
    if os.getenv("USE_ALEMBIC", "false").lower() == "true":
        _run_alembic_migrations()
    else:
        run_migrations_pg(conn)
```

The new `_run_alembic_migrations()` function:
- Creates an `alembic.config.Config` object programmatically, pointing at the `alembic/` directory
- Sets `sqlalchemy.url` from `TRADER_PG_URL`
- Calls `alembic.command.upgrade(config, "head")`
- Respects the `_migrations_done` module flag (check at top, set after success)
- Does NOT acquire the advisory lock itself -- `env.py` handles that

### 4.4 -- Handling Existing Databases

For production databases that already have all tables:

1. Deploy the code with `USE_ALEMBIC=true`
2. `alembic upgrade head` runs the baseline migration
3. The baseline uses `IF NOT EXISTS` throughout, so every statement is a no-op
4. Alembic creates the `alembic_version` table and stamps the baseline revision

Alternatively, run `alembic stamp head` manually on the production database to mark the baseline as already applied without executing it. This is safer if there's any concern about DDL differences.

### 4.5 -- Fallback Mechanism

The `USE_ALEMBIC` env var defaults to `false` (unset = false). This means:

- **First deployment:** Set `USE_ALEMBIC=true` in `.env`. If Alembic causes issues, set it back to `false` and restart. The old path continues to work because it uses `IF NOT EXISTS`.
- **After 1 week of stable operation:** Remove the flag and make Alembic the only path. Delete the fallback branch in `run_migrations()`.

Add `USE_ALEMBIC` to `.env.example` with documentation:
```
# Migration system: set to 'true' to use Alembic instead of legacy migrations.
# Default: false (legacy path). Switch to true after deploying Alembic baseline.
# Remove this flag entirely after 1 week of stable Alembic operation.
USE_ALEMBIC=false
```

### 4.6 -- Dependency: pyproject.toml

**File to modify: `pyproject.toml`**

Add `alembic` to the project dependencies. Alembic already exists in the virtualenv (confirmed at `.venv/bin/alembic`) but it must be declared as an explicit dependency:

```toml
# In [project.dependencies] or equivalent section
"alembic>=1.13.0",
```

### 4.7 -- Future Migration Workflow

After the baseline is in place, new schema changes follow this workflow:

```bash
# Create a new empty migration
alembic revision -m "add rate_limit_buckets table"

# Edit the generated file: write upgrade() and downgrade() with raw SQL
# (no autogenerate -- this codebase does not use SQLAlchemy ORM models)

# Apply locally
alembic upgrade head

# Verify
alembic current

# Commit the migration file alongside the code that uses the new table
```

Every migration file must have both `upgrade()` and `downgrade()`. The `downgrade()` is required for development rollback, even if it will never be run in production.

---

## Key Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `alembic.ini` | Create | Alembic configuration (sqlalchemy.url overridden at runtime) |
| `alembic/env.py` | Create | Migration environment: reads TRADER_PG_URL, acquires advisory lock 5145534154 |
| `alembic/script.py.mako` | Create | Migration file template (standard Alembic default) |
| `alembic/versions/001_baseline.py` | Create | Baseline migration with all 37+ table DDL from db.py |
| `src/quantstack/db.py` | Modify | Add USE_ALEMBIC flag to run_migrations(), add _run_alembic_migrations() |
| `pyproject.toml` | Modify | Add alembic>=1.13.0 to dependencies |
| `.env.example` | Modify | Add USE_ALEMBIC=false with documentation |
| `tests/integration/test_alembic_migrations.py` | Create | Integration tests for baseline, advisory lock, fallback, idempotency |

---

## Risks and Mitigations

**Risk 1: Baseline migration does not exactly match current schema.**
Some `_migrate_*_pg()` functions do more than `CREATE TABLE IF NOT EXISTS` -- they add columns, create indexes, enable extensions (pgvector). If the baseline misses any of these, the Alembic path produces a different schema than the legacy path.
*Mitigation:* The `test_both_paths_produce_identical_schema` test catches this. Run `pg_dump --schema-only` on both paths and diff before declaring the baseline correct.

**Risk 2: Advisory lock contention during transition.**
The old path uses `pg_try_advisory_lock` (non-blocking skip), while the new Alembic `env.py` uses `pg_advisory_lock` (blocking wait). If both paths run simultaneously (e.g., one container still on old code), they use the same lock key and will serialize correctly. No deadlock risk because only one lock is involved.

**Risk 3: `alembic upgrade head` called before `alembic_version` table exists.**
This is normal -- Alembic creates `alembic_version` automatically on first run. No special handling needed.

**Risk 4: DDL in autocommit vs. transaction.**
The current `db.py` runs DDL in autocommit mode. Alembic by default wraps each migration in a transaction. PostgreSQL supports transactional DDL, so this is actually safer (atomic migration). However, `CREATE INDEX CONCURRENTLY` cannot run inside a transaction. If any future migration needs concurrent index creation, it must use `op.execute()` outside of the migration transaction context. The baseline migration does not need this.
