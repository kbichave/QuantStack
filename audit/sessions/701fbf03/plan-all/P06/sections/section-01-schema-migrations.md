# Section 01 -- Schema Migrations for Options Desk

## Description

Add three schema changes to support the options desk upgrade: a `portfolio_greeks_history` table for periodic Greek snapshots, an `options_pnl_attribution` table for daily P&L decomposition by Greek, and a new `structure_type` column on the existing `positions` table to track multi-leg structure classification.

All DDL lives in `src/quantstack/db.py` inside the `ensure_tables()` function (or equivalent schema-init block). Tables are created via `CREATE TABLE IF NOT EXISTS` so they are idempotent on restart.

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/db.py` | Add three DDL statements inside `ensure_tables()` |

## What to Implement

### 1. `portfolio_greeks_history` table

```sql
CREATE TABLE IF NOT EXISTS portfolio_greeks_history (
    id              SERIAL PRIMARY KEY,
    snapshot_time   TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol_greeks   JSONB,          -- {symbol: {delta, gamma, theta, vega, rho}}
    strategy_greeks JSONB,          -- {strategy_name: {delta, gamma, theta, vega, rho}}
    portfolio_delta FLOAT NOT NULL DEFAULT 0,
    portfolio_gamma FLOAT NOT NULL DEFAULT 0,
    portfolio_theta FLOAT NOT NULL DEFAULT 0,
    portfolio_vega  FLOAT NOT NULL DEFAULT 0,
    portfolio_rho   FLOAT NOT NULL DEFAULT 0
);
```

Add index: `CREATE INDEX IF NOT EXISTS idx_pgh_snapshot_time ON portfolio_greeks_history (snapshot_time DESC);`

Purpose: one row per snapshot cycle (every 5-15 min during market hours). Enables time-series queries on portfolio-level Greek exposure. JSONB columns allow drill-down without schema changes when symbols/strategies change.

### 2. `options_pnl_attribution` table

```sql
CREATE TABLE IF NOT EXISTS options_pnl_attribution (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    symbol          TEXT NOT NULL,
    delta_pnl       FLOAT NOT NULL DEFAULT 0,
    gamma_pnl       FLOAT NOT NULL DEFAULT 0,
    theta_pnl       FLOAT NOT NULL DEFAULT 0,
    vega_pnl        FLOAT NOT NULL DEFAULT 0,
    unexplained_pnl FLOAT NOT NULL DEFAULT 0,
    total_pnl       FLOAT NOT NULL DEFAULT 0
);
```

Add unique constraint: `CREATE UNIQUE INDEX IF NOT EXISTS idx_opa_date_symbol ON options_pnl_attribution (date, symbol);`

Purpose: one row per symbol per trading day. `unexplained_pnl = total_pnl - (delta + gamma + theta + vega)` captures higher-order effects, discrete hedging slippage, and model error.

### 3. Add `structure_type` column to `positions`

```sql
ALTER TABLE positions ADD COLUMN IF NOT EXISTS structure_type TEXT DEFAULT 'single_leg';
```

Valid values: `single_leg`, `vertical_spread`, `iron_condor`, `butterfly`, `calendar`, `diagonal`, `straddle`, `strangle`, `ratio_spread`. Enforced at the application layer via the `StructureType` enum (section-04), not a DB constraint, to keep migrations simple.

## Tests to Write

File: `tests/unit/test_schema_migrations.py`

1. **test_portfolio_greeks_history_table_created** -- Call `ensure_tables()` against a test DB, then query `information_schema.columns` to verify all columns exist with correct types.
2. **test_options_pnl_attribution_table_created** -- Same pattern; verify unique index on `(date, symbol)` exists.
3. **test_positions_structure_type_column** -- Verify column exists, default is `'single_leg'`, and existing rows are unaffected by the ALTER.
4. **test_idempotent_schema_creation** -- Call `ensure_tables()` twice in succession; second call must not raise.
5. **test_portfolio_greeks_history_insert_roundtrip** -- Insert a row with JSONB payloads, read back, verify JSON structure preserved.
6. **test_pnl_attribution_upsert_conflict** -- Insert two rows for the same `(date, symbol)`, verify the unique index rejects the duplicate (or use ON CONFLICT if upsert semantics are desired).

## Edge Cases

- **Existing `positions` rows**: The `ALTER TABLE ADD COLUMN IF NOT EXISTS` with a DEFAULT ensures existing rows get `'single_leg'` without a full table rewrite on modern PostgreSQL (>= 11).
- **JSONB null handling**: `symbol_greeks` and `strategy_greeks` are nullable. Queries must handle `NULL` vs empty `{}`. Insert code should always write `{}` rather than `NULL` when there are no options positions.
- **Concurrent migrations**: `CREATE TABLE IF NOT EXISTS` and `ADD COLUMN IF NOT EXISTS` are safe under concurrent execution (multiple service instances starting simultaneously).
- **Clock skew on `snapshot_time`**: Uses `DEFAULT now()` (server time). If the inserting service is on a different host, pass explicit timestamps rather than relying on the default.
- **Large JSONB payloads**: With 50+ symbols each having 5 Greeks, the JSONB column is ~5-10 KB per row. At one snapshot per 5 minutes over 6.5 market hours, that is ~78 rows/day, ~390 KB/day -- negligible.
