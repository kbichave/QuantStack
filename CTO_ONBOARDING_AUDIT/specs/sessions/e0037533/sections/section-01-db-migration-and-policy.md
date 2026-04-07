# Section 01: DB Migration and Policy Update

## Overview

This section covers two zero-risk, additive changes that unblock all downstream sections:

1. **CLAUDE.md policy update** -- change the risk gate rule from "Never modify" to "Never weaken or bypass" so that Section 12 (pre-trade risk checks) can legitimately strengthen the gate.
2. **Database migration** -- create two new tables (`circuit_breaker_state`, `agent_dlq`) and verify existing schema elements (`regime_at_entry` column on positions, UNIQUE constraint on `loop_cursors.consumer_id`).

No existing behavior changes. No destructive DDL. Everything is `CREATE TABLE IF NOT EXISTS` and `ADD COLUMN IF NOT EXISTS`, matching the codebase convention.

---

## Dependencies

- **Depends on**: Nothing. This is Batch 1.
- **Blocks**: section-02 (state key audit), section-07 (circuit breaker), section-10 (DLQ), section-12 (risk gate pre-trade), section-13 (regime flip).

---

## Tests (Write First)

File: `tests/unit/test_phase4_db_migration.py`

```python
"""Tests for Phase 4 DB migration and policy update."""

import pytest

# ---- Policy test ----

# Test: CLAUDE.md contains "Never weaken or bypass" (not "Never modify") for risk gate rule
# Implementation: read CLAUDE.md, assert the updated phrasing is present and the old phrasing is absent.

# ---- circuit_breaker_state table ----

# Test: circuit_breaker_state table created with correct schema
#   - breaker_key TEXT PRIMARY KEY
#   - state TEXT DEFAULT 'closed'
#   - failure_count INT DEFAULT 0
#   - last_failure_at TIMESTAMPTZ (nullable)
#   - opened_at TIMESTAMPTZ (nullable)
#   - cooldown_seconds INT DEFAULT 300
#   - last_success_at TIMESTAMPTZ (nullable)

# Test: inserting a row with all defaults works (only breaker_key required)

# Test: atomic increment — UPDATE ... SET failure_count = failure_count + 1 RETURNING failure_count
#   returns correct value

# ---- agent_dlq table ----

# Test: agent_dlq table created with correct schema
#   - id SERIAL PRIMARY KEY
#   - agent_name TEXT NOT NULL
#   - graph_name TEXT NOT NULL
#   - run_id TEXT NOT NULL
#   - input_summary TEXT (nullable)
#   - raw_output TEXT (nullable)
#   - error_type TEXT (nullable)
#   - error_detail TEXT (nullable)
#   - prompt_hash TEXT (nullable)
#   - model_used TEXT (nullable)
#   - created_at TIMESTAMPTZ DEFAULT NOW()
#   - resolved_at TIMESTAMPTZ (nullable)
#   - resolution TEXT (nullable)

# Test: inserting a DLQ row with required fields only (agent_name, graph_name, run_id) succeeds

# ---- Existing schema verification ----

# Test: positions table has regime_at_entry column with DEFAULT 'unknown'
#   (already exists -- verify it's present, don't re-add)

# Test: loop_cursors table has consumer_id as PRIMARY KEY (implies UNIQUE)
#   Query information_schema.table_constraints or pg_indexes to confirm.

# ---- Idempotency ----

# Test: migration is idempotent — running the migration function twice does not error
```

All tests use the existing `conftest.py` DB fixture pattern (test database connection via `db_conn()` context manager). Tests should call the new migration function, then query `information_schema.columns` or attempt DML to verify table structure.

---

## Implementation

### Part 1: CLAUDE.md Policy Update

**File**: `CLAUDE.md` (project root)

**Current text** (line 64):
```
- **Risk gate is LAW.** Every trade passes through `src/quantstack/execution/risk_gate.py`. Never bypass. Never modify. Never auto-patch.
```

**Updated text**:
```
- **Risk gate is LAW.** Every trade passes through `src/quantstack/execution/risk_gate.py`. Never weaken or bypass. Never auto-patch. Strengthening checks (adding new rejection criteria, lowering thresholds) is permitted and encouraged.
```

**Why**: The original "Never modify" wording creates ambiguity -- it could be read as prohibiting the addition of new safety checks (Section 12: correlation, heat budget, sector concentration). The updated wording preserves the intent (prevent weakening) while explicitly permitting strengthening.

### Part 2: Database Migration

**File**: `src/quantstack/db.py`

Add a new migration function `_migrate_phase4_coordination_pg(conn)` and register it in `run_migrations_pg()`.

#### New function: `_migrate_phase4_coordination_pg`

This function creates two tables using the same `CREATE TABLE IF NOT EXISTS` / `_to_pg()` pattern used throughout `db.py`.

**Table 1: `circuit_breaker_state`**

Purpose: Persistent state for the node circuit breaker (Section 07). Stores per-node breaker state so that breaker trips survive process restarts.

Schema:

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `breaker_key` | TEXT | PRIMARY KEY | Format: `"graph_name/node_name"`, e.g. `"trading/data_refresh"` |
| `state` | TEXT | DEFAULT `'closed'` | One of: `closed`, `open`, `half_open` |
| `failure_count` | INTEGER | DEFAULT 0 | Consecutive failure count; reset on success |
| `last_failure_at` | TIMESTAMPTZ | nullable | Timestamp of most recent failure |
| `opened_at` | TIMESTAMPTZ | nullable | When breaker transitioned to `open` |
| `cooldown_seconds` | INTEGER | DEFAULT 300 | Seconds before `open` -> `half_open` probe |
| `last_success_at` | TIMESTAMPTZ | nullable | Timestamp of most recent success |

Concurrency note: Section 07 will use `UPDATE ... SET failure_count = failure_count + 1 ... RETURNING failure_count, state` for atomic increment. The schema supports this -- no application-level locking needed.

**Table 2: `agent_dlq`**

Purpose: Dead letter queue for agent parse/validation failures (Section 10). Captures raw LLM output when `parse_json_response()` fails, enabling debugging and DLQ rate monitoring.

Schema:

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | SERIAL | PRIMARY KEY | Auto-incrementing |
| `agent_name` | TEXT | NOT NULL | Which agent produced the output |
| `graph_name` | TEXT | NOT NULL | Which graph was running |
| `run_id` | TEXT | NOT NULL | Correlates to Langfuse trace |
| `input_summary` | TEXT | nullable | Truncated input state for context |
| `raw_output` | TEXT | nullable | The unparsed LLM output |
| `error_type` | TEXT | nullable | `parse_error`, `validation_error`, `timeout`, `business_rule` |
| `error_detail` | TEXT | nullable | Exception message / details |
| `prompt_hash` | TEXT | nullable | Hash of prompt for clustering failures by prompt variant |
| `model_used` | TEXT | nullable | LLM model identifier |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | When the failure occurred |
| `resolved_at` | TIMESTAMPTZ | nullable | When/if the entry was resolved |
| `resolution` | TEXT | nullable | `manual_override`, `prompt_fixed`, `discarded` |

Add an index on `(agent_name, created_at)` for the 24h rolling window DLQ rate query that Section 10 will use.

#### Existing schema verification (no changes needed)

**`regime_at_entry` on positions table**: Already exists. The `_migrate_portfolio_pg` function creates the positions table with `regime_at_entry TEXT DEFAULT 'unknown'` (line 563 of `db.py`). No migration needed -- just verify in tests.

**`loop_cursors.consumer_id` UNIQUE constraint**: Already satisfied. The `_migrate_coordination_pg` function creates `loop_cursors` with `consumer_id TEXT PRIMARY KEY` (line 933 of `db.py`). A PRIMARY KEY implies UNIQUE. The `ON CONFLICT (consumer_id)` upsert in Section 09 will work without any schema changes.

#### Registration in `run_migrations_pg()`

Add the call to `_migrate_phase4_coordination_pg(conn)` at the end of the migration list in `run_migrations_pg()`, after `_migrate_hnsw_index_pg(conn)`. This follows the existing pattern -- new migration functions are appended, never inserted, so the ordering of existing migrations is preserved.

### Migration function stub

```python
def _migrate_phase4_coordination_pg(conn: PgConnection) -> None:
    """Phase 4: circuit breaker state + agent dead letter queue.

    Additive only — no destructive DDL. Tables used by:
    - circuit_breaker_state: Section 07 (node circuit breaker)
    - agent_dlq: Section 10 (dead letter queue for parse failures)
    """
    # circuit_breaker_state — persistent node-level breaker state
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS circuit_breaker_state (
            breaker_key       TEXT PRIMARY KEY,
            state             TEXT DEFAULT 'closed',
            failure_count     INTEGER DEFAULT 0,
            last_failure_at   TIMESTAMPTZ,
            opened_at         TIMESTAMPTZ,
            cooldown_seconds  INTEGER DEFAULT 300,
            last_success_at   TIMESTAMPTZ
        )
    """))

    # agent_dlq — dead letter queue for agent parse/validation failures
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_dlq_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS agent_dlq (
            id              BIGINT PRIMARY KEY DEFAULT nextval('agent_dlq_seq'),
            agent_name      TEXT NOT NULL,
            graph_name      TEXT NOT NULL,
            run_id          TEXT NOT NULL,
            input_summary   TEXT,
            raw_output      TEXT,
            error_type      TEXT,
            error_detail    TEXT,
            prompt_hash     TEXT,
            model_used      TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            resolved_at     TIMESTAMPTZ,
            resolution      TEXT
        )
    """))
    conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_agent_dlq_agent_created
        ON agent_dlq (agent_name, created_at)
    """)
```

Note: Uses `BIGINT + sequence` instead of `SERIAL` to match the codebase convention (see `closed_trades`, `equity_alerts`, etc. which all use explicit sequences).

---

## Verification Checklist

After implementation, verify:

- [ ] `CLAUDE.md` line about risk gate says "Never weaken or bypass" not "Never modify"
- [ ] `CLAUDE.md` explicitly permits strengthening the risk gate
- [ ] `circuit_breaker_state` table exists with all 7 columns and correct defaults
- [ ] `agent_dlq` table exists with all 13 columns, sequence works, index exists
- [ ] `positions.regime_at_entry` column exists (pre-existing, just confirm)
- [ ] `loop_cursors.consumer_id` has UNIQUE/PRIMARY KEY constraint (pre-existing, just confirm)
- [ ] Running migration twice produces no errors (idempotent)
- [ ] All existing tests still pass (no regressions from additive DDL)
