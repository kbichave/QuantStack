# Section 01: Schema Migrations

## Objective

Add 4 new tables to the PostgreSQL migration system for P05 Adaptive Signal Synthesis: `precomputed_ic_weights`, `conviction_calibration` (restructured), `ensemble_ab_results`, and `ensemble_config`.

## Dependencies

None. This section must be completed first -- all other sections depend on these tables existing.

## File to Modify

**`src/quantstack/db.py`**

## Current State

The existing `_migrate_p05_adaptive_synthesis_pg()` function (line 3475) already:
- Adds a `regime` column to `ic_attribution_data`
- Creates an index on `(collector, regime, recorded_at DESC)`
- Creates a `conviction_calibration` table with columns `(id, factor_name, factor_value, forward_return, regime, recorded_at)` -- this is the raw observation table, not the calibrated parameter table we need

The new migration adds 4 tables that do not yet exist.

## Implementation

Append to the existing `_migrate_p05_adaptive_synthesis_pg()` function in `src/quantstack/db.py` (after line 3500, before the `logger.debug` call). Follow the established pattern: `conn.execute(_to_pg("""..."""))` for CREATE TABLE, plain string for CREATE INDEX.

### Table 1: `precomputed_ic_weights`

Stores weekly-precomputed IC-derived synthesis weights per regime per collector. The synthesis hot path reads this instead of instantiating `ICAttributionTracker` on every call.

```python
    # P05 §5.1: Precomputed IC weights — weekly batch output
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS precomputed_ic_weights (
            regime          TEXT NOT NULL,
            collector       TEXT NOT NULL,
            weight          DOUBLE PRECISION NOT NULL,
            ic_value        DOUBLE PRECISION NOT NULL,
            computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (regime, collector)
        )
    """))
```

- **PK**: `(regime, collector)` -- one weight per collector per regime. Upsert on recomputation.
- **No sequence needed** -- composite natural key.

### Table 2: `conviction_calibration_params`

Stores calibrated conviction factor parameters (output of quarterly calibration). Distinct from the existing `conviction_calibration` table which stores raw observation data.

```python
    # P05 §5.3: Calibrated conviction factor parameters
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS conviction_calibration_params (
            factor_name     TEXT NOT NULL,
            param_name      TEXT NOT NULL,
            param_value     DOUBLE PRECISION NOT NULL,
            calibrated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            sample_size     INTEGER NOT NULL DEFAULT 0,
            r_squared       DOUBLE PRECISION DEFAULT 0.0,
            PRIMARY KEY (factor_name, param_name)
        )
    """))
```

- **PK**: `(factor_name, param_name)` -- e.g., `('adx', 'threshold')`, `('adx', 'scale_factor')`.
- Named `conviction_calibration_params` to avoid collision with existing `conviction_calibration` observation table.

### Table 3: `ensemble_ab_results`

Records per-symbol ensemble method outputs for A/B comparison.

```python
    # P05 §5.4: Ensemble A/B test results
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS ensemble_ab_results (
            id              SERIAL PRIMARY KEY,
            symbol          TEXT NOT NULL,
            signal_date     DATE NOT NULL,
            method_name     TEXT NOT NULL,
            signal_value    DOUBLE PRECISION NOT NULL,
            forward_return_5d DOUBLE PRECISION,
            recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """))
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ensemble_ab_date_method "
        "ON ensemble_ab_results (signal_date, method_name)"
    )
```

- **Index on `(signal_date, method_name)`** -- the evaluation query groups by method and filters by date range.
- `forward_return_5d` is nullable; filled in later by a backfill job or the weekly evaluator.

### Table 4: `ensemble_config`

Single-row table storing the currently active ensemble method.

```python
    # P05 §5.4: Active ensemble method config (single-row)
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS ensemble_config (
            id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            active_method   TEXT NOT NULL DEFAULT 'weighted_avg',
            promoted_at     TIMESTAMPTZ,
            evidence_ic     DOUBLE PRECISION,
            evidence_pvalue DOUBLE PRECISION
        )
    """))
    # Seed default row if empty
    conn.execute(
        "INSERT INTO ensemble_config (id, active_method) "
        "VALUES (1, 'weighted_avg') ON CONFLICT (id) DO NOTHING"
    )
```

- **`CHECK (id = 1)`** enforces single-row semantics at the DB level.
- Default row seeded on migration so `SELECT` never returns empty.

### Final logger line

Update the existing logger.debug message to reflect the expanded migration:

```python
    logger.debug("[DB] P05 adaptive synthesis tables migrated (precomputed_ic_weights, conviction_calibration_params, ensemble_ab_results, ensemble_config)")
```

## Edge Cases

1. **Idempotency**: All statements use `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, and `ON CONFLICT DO NOTHING`. Running the migration twice must be a no-op.
2. **Existing `conviction_calibration` table**: We do NOT alter or drop it. The raw observation table remains for data collection. The new `conviction_calibration_params` table stores calibrated outputs.
3. **Empty `ensemble_config`**: The seed INSERT ensures the table always has exactly one row. The `CHECK (id = 1)` constraint prevents accidental multi-row state.
4. **Column type choices**: `DOUBLE PRECISION` for weights/IC values (matches existing pattern throughout db.py). `TIMESTAMPTZ` for all temporal columns (matches existing pattern).

## Tests

File: `tests/unit/test_p05_schema_migrations.py`

```python
"""Verify P05 schema migrations are idempotent."""

def test_p05_migration_idempotent(monkeypatch, tmp_path):
    """Running _migrate_p05_adaptive_synthesis_pg twice succeeds without error."""
    # Use a real test DB or mock PgConnection
    # Call _migrate_p05_adaptive_synthesis_pg(conn) twice
    # Assert no exception raised on second call

def test_precomputed_ic_weights_upsert(db_conn):
    """Upsert into precomputed_ic_weights replaces existing row."""
    # INSERT (regime='trending_up', collector='trend', weight=0.35, ic_value=0.08)
    # INSERT same PK with different weight
    # SELECT and assert latest weight

def test_ensemble_config_single_row(db_conn):
    """ensemble_config enforces single-row via CHECK constraint."""
    # Attempt INSERT with id=2 → expect constraint violation
    # SELECT → exactly one row with active_method='weighted_avg'

def test_conviction_calibration_params_upsert(db_conn):
    """Upsert into conviction_calibration_params."""
    # INSERT (factor_name='adx', param_name='threshold', param_value=15.0, ...)
    # UPDATE same PK → verify overwrite
```

## Verification

After implementation, run:
```bash
python -c "from quantstack.db import db_conn, run_migrations_pg; c = db_conn(); run_migrations_pg(c)"
```

Then verify tables exist:
```sql
SELECT table_name FROM information_schema.tables
WHERE table_name IN ('precomputed_ic_weights', 'conviction_calibration_params', 'ensemble_ab_results', 'ensemble_config');
```

Expected: 4 rows returned.
