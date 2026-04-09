# Section 06: Schema Migrations

## Objective

Add database tables and columns to support multi-asset trading. This is independent of the Python code and can run in parallel with section-01.

## Files to Create

### `migrations/P12_001_asset_class_config.sql`
Create the `asset_class_config` table:

```sql
CREATE TABLE IF NOT EXISTS asset_class_config (
    class_name       TEXT PRIMARY KEY,         -- 'equity', 'futures', 'crypto', 'forex', 'fixed_income'
    enabled          BOOLEAN NOT NULL DEFAULT FALSE,
    position_limit_pct NUMERIC(5,4) NOT NULL,  -- max % of equity per position (e.g., 0.0500 = 5%)
    total_exposure_pct NUMERIC(5,4) NOT NULL,  -- max total % of equity for this class
    daily_loss_limit_pct NUMERIC(5,4) NOT NULL, -- max daily loss % before halting this class
    max_positions    INT NOT NULL DEFAULT 10,
    instruments      JSONB NOT NULL DEFAULT '[]'::jsonb,  -- allowed instruments for this class
    margin_type      TEXT NOT NULL DEFAULT 'cash',         -- 'cash', 'span', 'portfolio'
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Seed data:
```sql
INSERT INTO asset_class_config VALUES
    ('equity',   TRUE,  0.0500, 0.6000, 0.0200, 20, '["SPY","QQQ","AAPL","MSFT","NVDA","TSLA","AMD","AMZN","GOOGL","META"]', 'cash'),
    ('options',  TRUE,  0.0300, 0.2000, 0.0200, 15, '[]', 'cash'),
    ('futures',  FALSE, 0.0500, 0.3000, 0.0150,  5, '["ES","NQ","CL","GC","ZN"]', 'span'),
    ('crypto',   FALSE, 0.0200, 0.0600, 0.0100,  3, '["BTC","ETH","SOL"]', 'cash'),
    ('forex',    FALSE, 0.0300, 0.1500, 0.0100,  5, '[]', 'cash'),
    ('fixed_income', FALSE, 0.0500, 0.2000, 0.0100, 5, '[]', 'cash')
ON CONFLICT (class_name) DO NOTHING;
```

### `migrations/P12_002_positions_asset_class.sql`
Add `asset_class` column to the `positions` table:

```sql
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS asset_class TEXT NOT NULL DEFAULT 'equity';

CREATE INDEX IF NOT EXISTS idx_positions_asset_class
    ON positions(asset_class);

-- Add contract_multiplier for futures/options notional calculation
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS contract_multiplier NUMERIC(12,2) DEFAULT 1.0;

-- Add margin_required for futures positions
ALTER TABLE positions
    ADD COLUMN IF NOT EXISTS margin_required NUMERIC(14,2) DEFAULT 0.0;
```

### `migrations/P12_003_cross_asset_signals.sql`
Create table for caching cross-asset signal snapshots:

```sql
CREATE TABLE IF NOT EXISTS cross_asset_signals (
    id              SERIAL PRIMARY KEY,
    signal_name     TEXT NOT NULL,
    signal_value    NUMERIC(12,6),
    signal_metadata JSONB DEFAULT '{}'::jsonb,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cross_asset_signals_name_time
    ON cross_asset_signals(signal_name, computed_at DESC);
```

### `migrations/P12_004_daily_pnl_by_asset_class.sql`
Extend daily P&L tracking:

```sql
CREATE TABLE IF NOT EXISTS daily_pnl_by_asset_class (
    trade_date   DATE NOT NULL,
    asset_class  TEXT NOT NULL,
    realized_pnl NUMERIC(14,2) NOT NULL DEFAULT 0.0,
    unrealized_pnl NUMERIC(14,2) NOT NULL DEFAULT 0.0,
    is_halted    BOOLEAN NOT NULL DEFAULT FALSE,
    halted_at    TIMESTAMPTZ,
    PRIMARY KEY (trade_date, asset_class)
);
```

## Files to Modify

### `src/quantstack/db.py`
Add a migration runner function that executes all `migrations/P12_*.sql` files in order. Use the existing pattern if one exists, or add:

```python
def run_migrations(prefix: str = "P12") -> None:
    """Execute all SQL migration files matching the prefix."""
```

This should be idempotent (IF NOT EXISTS, ON CONFLICT DO NOTHING) so it can be run multiple times safely.

### `src/quantstack/data/pg_storage.py`
Update any queries that read from `positions` to include the new `asset_class` column. Ensure writes default to `'equity'` for backward compatibility.

## Implementation Details

1. All DDL uses `IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS` for idempotency.
2. The `instruments` JSONB column in `asset_class_config` allows dynamic instrument lists without schema changes.
3. The `contract_multiplier` column defaults to 1.0 so existing equity positions are unaffected.
4. Seed data starts with futures and crypto DISABLED (`enabled=FALSE`). They must be explicitly enabled after the adapters are tested.
5. The `daily_pnl_by_asset_class` table is separate from any existing daily P&L tracking — do not modify existing tables that other systems depend on.

## Test Requirements

### `tests/unit/test_schema_migrations.py`
- Run all P12 migrations against a test database
- Verify `asset_class_config` has 6 rows after seeding
- Verify `positions` table has `asset_class`, `contract_multiplier`, `margin_required` columns
- Verify `cross_asset_signals` table exists with correct indexes
- Verify migrations are idempotent (run twice, no errors)
- Verify default values: new position row has `asset_class='equity'`, `contract_multiplier=1.0`

## Acceptance Criteria

- [ ] All 4 migration files created and syntactically valid
- [ ] Migrations are idempotent (safe to run multiple times)
- [ ] Existing `positions` data unaffected (defaults preserve current behavior)
- [ ] `asset_class_config` seeded with correct limits for all 6 asset classes
- [ ] Futures and crypto start DISABLED
- [ ] Migration runner function added to `db.py`
- [ ] `pg_storage.py` updated for new columns
- [ ] All tests pass
