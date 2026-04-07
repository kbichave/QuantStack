# Section 01: Database Schema — Corporate Actions, System Alerts & Supporting Tables

This section adds all new database tables and schema changes required by Phase 9. Every other section in Phase 9 depends on this one. No other section should define DDL — all table creation lives here.

---

## What Changes and Why

Phase 9 introduces several new subsystems (corporate actions monitoring, system-level alerts, factor exposure tracking, performance attribution, EventBus ACK guarantees). Each needs persistent state. Rather than scatter DDL across feature PRs, all schema work is consolidated here so the database contract is established first and downstream sections can write to tables that already exist.

Additionally, `TradingState` in the graph layer uses `extra="forbid"` (Pydantic strict mode), which means any new state field must be declared before a node can return it. The attribution node (Section 5) needs a `cycle_attribution` field — that declaration belongs here with the schema work, not deferred to the feature section.

---

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/db.py` | New migration functions for 7 tables, ACK column additions to `loop_events`, new `llm_config` table. Registration in the migration runner. |
| `src/quantstack/graphs/state.py` | Add `cycle_attribution: dict = {}` field to `TradingState`. |

---

## Tests

Write these first. All tests go in `tests/unit/test_db_schema_phase9.py` unless noted otherwise.

**Testing framework:** pytest. Use the existing `trading_ctx` fixture for tests that need a live PostgreSQL connection. The `reset_singletons_and_seeds` autouse fixture prevents state pollution between tests.

### Unit Tests

Each test verifies that `run_migrations()` (which calls all `_migrate_*_pg` functions) creates the expected table with correct columns and constraints. Pattern: connect to the test DB, run migrations, then query `information_schema.columns` or attempt constraint-violating inserts.

```python
# tests/unit/test_db_schema_phase9.py

# Test: ensure corporate_actions table exists with columns:
#   symbol (TEXT NOT NULL), event_type (TEXT NOT NULL), source (TEXT NOT NULL),
#   effective_date (DATE NOT NULL), announcement_date (DATE),
#   raw_payload (JSONB), processed (BOOLEAN DEFAULT FALSE),
#   created_at (TIMESTAMPTZ DEFAULT NOW())
# Verify UNIQUE constraint on (symbol, event_type, effective_date, source)

# Test: ensure split_adjustments table exists with columns:
#   symbol (TEXT NOT NULL), effective_date (DATE NOT NULL),
#   split_ratio (DOUBLE PRECISION NOT NULL),
#   old_quantity (DOUBLE PRECISION NOT NULL), new_quantity (DOUBLE PRECISION NOT NULL),
#   old_cost_basis (DOUBLE PRECISION NOT NULL), new_cost_basis (DOUBLE PRECISION NOT NULL),
#   applied_at (TIMESTAMPTZ DEFAULT NOW())
# Verify UNIQUE constraint on (symbol, effective_date, event_type)
#   NOTE: event_type column needed for this constraint even though the dataclass
#   doesn't list it — the constraint distinguishes dividend vs split on the same date.

# Test: ensure system_alerts table exists with:
#   id (BIGSERIAL PRIMARY KEY), category (TEXT NOT NULL), severity (TEXT NOT NULL),
#   status (TEXT NOT NULL DEFAULT 'open'), source (TEXT NOT NULL),
#   title (TEXT NOT NULL), detail (TEXT), metadata (JSONB),
#   acknowledged_by (TEXT), acknowledged_at (TIMESTAMPTZ),
#   escalated_at (TIMESTAMPTZ), resolved_at (TIMESTAMPTZ), resolution (TEXT),
#   created_at (TIMESTAMPTZ DEFAULT NOW())
# Verify BIGSERIAL auto-increments on insert.

# Test: ensure loop_events table gains four new columns:
#   requires_ack (BOOLEAN DEFAULT FALSE),
#   expected_ack_by (TIMESTAMPTZ),
#   acked_at (TIMESTAMPTZ),
#   acked_by (TEXT)
# Use ALTER TABLE ADD COLUMN IF NOT EXISTS pattern for idempotency.

# Test: ensure dead_letter_events table exists with columns:
#   original_event_id (TEXT NOT NULL), event_type (TEXT NOT NULL),
#   source_loop (TEXT NOT NULL), payload (JSONB),
#   published_at (TIMESTAMPTZ NOT NULL), expected_ack_by (TIMESTAMPTZ NOT NULL),
#   retry_count (INTEGER DEFAULT 0),
#   dead_lettered_at (TIMESTAMPTZ DEFAULT NOW())

# Test: ensure factor_config table exists with:
#   config_key (TEXT PRIMARY KEY), value (TEXT NOT NULL),
#   updated_at (TIMESTAMPTZ DEFAULT NOW())
# Verify default rows are populated:
#   beta_drift_threshold=0.3, sector_max_pct=40,
#   momentum_crowding_pct=70, benchmark_symbol=SPY

# Test: ensure factor_exposure_history table exists with:
#   portfolio_beta (DOUBLE PRECISION), sector_weights (JSONB),
#   style_scores (JSONB), momentum_crowding_pct (DOUBLE PRECISION),
#   benchmark_symbol (TEXT), alerts_triggered (INTEGER),
#   computed_at (TIMESTAMPTZ DEFAULT NOW())

# Test: ensure cycle_attribution table exists with:
#   cycle_id (TEXT NOT NULL), graph_cycle_number (INTEGER NOT NULL),
#   total_pnl (DOUBLE PRECISION), factor_contribution (DOUBLE PRECISION),
#   timing_contribution (DOUBLE PRECISION), selection_contribution (DOUBLE PRECISION),
#   cost_contribution (DOUBLE PRECISION), per_position (JSONB),
#   computed_at (TIMESTAMPTZ DEFAULT NOW())

# Test: ensure llm_config table exists with:
#   tier (TEXT PRIMARY KEY), provider (TEXT NOT NULL), model (TEXT NOT NULL),
#   fallback_order (JSONB), updated_at (TIMESTAMPTZ DEFAULT NOW())

# Test: idempotency — calling run_migrations() twice does not fail or duplicate
#   default factor_config rows. Verify row count for factor_config is exactly 4
#   after two migration runs.
```

### Integration Tests

```python
# tests/integration/test_db_constraints_phase9.py

# Test: corporate_actions unique constraint rejects duplicate (symbol, event_type, effective_date, source).
#   Insert a row, then attempt the same insert — expect IntegrityError.

# Test: corporate_actions unique constraint allows same symbol+date with different event_type.
#   Insert (AAPL, dividend, 2024-01-15, alpha_vantage), then (AAPL, split, 2024-01-15, alpha_vantage) — expect success.

# Test: split_adjustments unique constraint rejects duplicate (symbol, effective_date, event_type).

# Test: system_alerts BIGSERIAL auto-increments on insert.
#   Insert two rows, verify second id = first id + 1.

# Test: loop_events existing rows have requires_ack=NULL (not FALSE) after migration.
#   Insert a row before adding columns (simulating pre-migration data),
#   run migration, verify requires_ack IS NULL.

# Test: TradingState accepts cycle_attribution field without ValidationError.
#   Instantiate TradingState(cycle_attribution={"some": "data"}) — must not raise.
#   This validates compatibility with extra="forbid".
```

---

## Implementation Details

### New Migration Functions in `src/quantstack/db.py`

Follow the existing pattern: each migration is a standalone function `_migrate_<name>_pg(conn: PgConnection) -> None` that uses `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for idempotency. Wrap DDL in `_to_pg()` when the SQL contains `DOUBLE` or `JSON` types that need normalization.

#### 1. `_migrate_corporate_actions_pg`

Creates two tables:

**`corporate_actions`** — raw events from Alpha Vantage and EDGAR:

- Columns: `symbol TEXT NOT NULL`, `event_type TEXT NOT NULL`, `source TEXT NOT NULL`, `effective_date DATE NOT NULL`, `announcement_date DATE`, `raw_payload JSONB`, `processed BOOLEAN DEFAULT FALSE`, `created_at TIMESTAMPTZ DEFAULT NOW()`
- Unique constraint: `(symbol, event_type, effective_date, source)` — enables idempotent inserts (INSERT ... ON CONFLICT DO NOTHING)
- Index on `(symbol, effective_date)` for date-range queries

**`split_adjustments`** — audit trail for cost basis adjustments:

- Columns: `symbol TEXT NOT NULL`, `effective_date DATE NOT NULL`, `event_type TEXT NOT NULL DEFAULT 'split'`, `split_ratio DOUBLE PRECISION NOT NULL`, `old_quantity DOUBLE PRECISION NOT NULL`, `new_quantity DOUBLE PRECISION NOT NULL`, `old_cost_basis DOUBLE PRECISION NOT NULL`, `new_cost_basis DOUBLE PRECISION NOT NULL`, `applied_at TIMESTAMPTZ DEFAULT NOW()`
- Unique constraint: `(symbol, effective_date, event_type)` — prevents double-adjustment while allowing multiple event types on the same date
- The `event_type` column exists solely for the unique constraint and defaults to `'split'`

**Invariant** (enforced in application code, not DB): `old_quantity * old_cost_basis == new_quantity * new_cost_basis` (total cost unchanged by split adjustment).

#### 2. `_migrate_system_alerts_pg`

**`system_alerts`** — operational/system events, separate from equity alerts:

- Columns: `id BIGSERIAL PRIMARY KEY`, `category TEXT NOT NULL`, `severity TEXT NOT NULL`, `status TEXT NOT NULL DEFAULT 'open'`, `source TEXT NOT NULL`, `title TEXT NOT NULL`, `detail TEXT`, `metadata JSONB`, `acknowledged_by TEXT`, `acknowledged_at TIMESTAMPTZ`, `escalated_at TIMESTAMPTZ`, `resolved_at TIMESTAMPTZ`, `resolution TEXT`, `created_at TIMESTAMPTZ DEFAULT NOW()`
- Valid categories: `risk_breach`, `service_failure`, `kill_switch`, `data_quality`, `performance_degradation`, `factor_drift`, `ack_timeout`, `thesis_review`
- Valid severities: `info`, `warning`, `critical`, `emergency`
- Valid statuses: `open`, `acknowledged`, `escalated`, `resolved`
- Index on `(status, severity, created_at DESC)` for the dashboard query pattern
- Category/severity/status validation is enforced in application code (Section 2), not CHECK constraints, to keep migration simple and allow future extensibility

#### 3. `_migrate_eventbus_ack_pg`

Extends the existing `loop_events` table and creates a new dead letter table.

**`loop_events` column additions** (via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`):

- `requires_ack BOOLEAN DEFAULT FALSE`
- `expected_ack_by TIMESTAMPTZ`
- `acked_at TIMESTAMPTZ`
- `acked_by TEXT`

Migration note: existing rows will have `requires_ack=NULL` after the column addition (the `DEFAULT FALSE` only applies to new inserts when using `ADD COLUMN IF NOT EXISTS` without a backfill). The ACK monitor query in Section 7 filters on `requires_ack = TRUE`, so NULL rows are safely excluded. No backfill needed — the 7-day TTL will age out old rows naturally.

**`dead_letter_events`** — events that missed their ACK window:

- Columns: `original_event_id TEXT NOT NULL`, `event_type TEXT NOT NULL`, `source_loop TEXT NOT NULL`, `payload JSONB`, `published_at TIMESTAMPTZ NOT NULL`, `expected_ack_by TIMESTAMPTZ NOT NULL`, `retry_count INTEGER DEFAULT 0`, `dead_lettered_at TIMESTAMPTZ DEFAULT NOW()`
- Index on `(dead_lettered_at DESC)` for monitoring queries

#### 4. `_migrate_factor_exposure_pg`

Creates two tables:

**`factor_config`** — configurable thresholds and benchmark:

- Columns: `config_key TEXT PRIMARY KEY`, `value TEXT NOT NULL`, `updated_at TIMESTAMPTZ DEFAULT NOW()`
- Default rows inserted via `INSERT ... ON CONFLICT DO NOTHING` (idempotent):
  - `beta_drift_threshold` = `0.3`
  - `sector_max_pct` = `40`
  - `momentum_crowding_pct` = `70`
  - `benchmark_symbol` = `SPY`

**`factor_exposure_history`** — per-cycle factor snapshots:

- Columns: `portfolio_beta DOUBLE PRECISION`, `sector_weights JSONB`, `style_scores JSONB`, `momentum_crowding_pct DOUBLE PRECISION`, `benchmark_symbol TEXT`, `alerts_triggered INTEGER DEFAULT 0`, `computed_at TIMESTAMPTZ DEFAULT NOW()`
- Index on `(computed_at DESC)` for trend queries

#### 5. `_migrate_cycle_attribution_pg`

**`cycle_attribution`** — per-cycle P&L decomposition:

- Columns: `cycle_id TEXT NOT NULL`, `graph_cycle_number INTEGER NOT NULL`, `total_pnl DOUBLE PRECISION`, `factor_contribution DOUBLE PRECISION`, `timing_contribution DOUBLE PRECISION`, `selection_contribution DOUBLE PRECISION`, `cost_contribution DOUBLE PRECISION`, `per_position JSONB`, `computed_at TIMESTAMPTZ DEFAULT NOW()`
- Unique constraint: `(cycle_id)` — one attribution per cycle
- Index on `(computed_at DESC)` for recent lookups

#### 6. `_migrate_llm_config_pg`

**`llm_config`** — runtime-changeable LLM tier configuration:

- Columns: `tier TEXT PRIMARY KEY`, `provider TEXT NOT NULL`, `model TEXT NOT NULL`, `fallback_order JSONB`, `updated_at TIMESTAMPTZ DEFAULT NOW()`
- No default rows — the code-level defaults in `llm_config.py` serve as the fallback. DB rows are only created when an operator wants to override at runtime.

### Migration Runner Registration

Add the new migration function calls to the runner block in `db.py` (the function containing the `_migrate_*_pg(conn)` call list, around line 649):

```python
            _migrate_loss_aggregation_pg(conn)
            # Phase 9: Missing Roles & Scale
            _migrate_corporate_actions_pg(conn)
            _migrate_system_alerts_pg(conn)
            _migrate_eventbus_ack_pg(conn)
            _migrate_factor_exposure_pg(conn)
            _migrate_cycle_attribution_pg(conn)
            _migrate_llm_config_pg(conn)
```

### TradingState Update in `src/quantstack/graphs/state.py`

Add a new field to `TradingState` (after the existing `pre_execution_brief` field, before the accumulation fields):

```python
    # Cycle P&L attribution (populated by attribution_node after reflect)
    cycle_attribution: dict = {}
```

This field is a plain dict (not append-only) because each cycle produces exactly one attribution snapshot that overwrites the previous. The attribution node (Section 5) will return `{"cycle_attribution": {...}}` and Pydantic will accept it because the field is now declared.

---

## Dependencies

This section has **no dependencies** — it is the foundation for all other Phase 9 sections.

The following sections depend on this one and cannot be implemented until the schema exists:

- Section 02 (System Alerts) — writes to `system_alerts`
- Section 03 (Corporate Actions) — writes to `corporate_actions`, `split_adjustments`
- Section 04 (Factor Exposure) — reads/writes `factor_config`, `factor_exposure_history`
- Section 05 (Performance Attribution) — writes to `cycle_attribution`, reads `cycle_attribution` state field
- Section 06 (Dashboard Alerts) — reads from `system_alerts`
- Section 07 (EventBus ACK) — uses ACK columns on `loop_events`, writes to `dead_letter_events`
- Section 08 (Multi-Mode) — depends on Sections 04, 05 which depend on this
- Section 09 (LLM Unification) — reads/writes `llm_config`

---

## Implementation Checklist

1. Write all tests from the Tests section above (they will fail — no tables exist yet)
2. Implement `_migrate_corporate_actions_pg` in `src/quantstack/db.py`
3. Implement `_migrate_system_alerts_pg` in `src/quantstack/db.py`
4. Implement `_migrate_eventbus_ack_pg` in `src/quantstack/db.py`
5. Implement `_migrate_factor_exposure_pg` in `src/quantstack/db.py`
6. Implement `_migrate_cycle_attribution_pg` in `src/quantstack/db.py`
7. Implement `_migrate_llm_config_pg` in `src/quantstack/db.py`
8. Register all six new migration functions in the migration runner block
9. Add `cycle_attribution: dict = {}` to `TradingState` in `src/quantstack/graphs/state.py`
10. Run tests — all should pass
11. Run `uv run pytest tests/ -k "not slow and not requires_api"` to verify no regressions
