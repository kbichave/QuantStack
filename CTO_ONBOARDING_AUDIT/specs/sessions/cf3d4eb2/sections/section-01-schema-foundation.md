# Section 01: Schema Foundation

## Purpose

This section creates all new database tables and columns required by the Phase 6 execution layer. It is the foundation that every subsequent section depends on. No business logic is implemented here -- only DDL (table creation, indexes, constraints) and the migration function that wires it into the existing startup path.

After completing this section, every downstream section can assume its tables exist and focus purely on application logic.

---

## Background

QuantStack uses an idempotent migration pattern in `src/quantstack/db.py`. Each subsystem has a `_migrate_*_pg(conn)` function that runs `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements. All migration functions are called from `run_migrations_pg()`, which:

1. Acquires a PostgreSQL advisory lock (only one process runs migrations)
2. Sets `autocommit = True` so each DDL commits immediately
3. Calls each `_migrate_*_pg` function in sequence
4. Uses `_to_pg()` helper to normalize `DOUBLE` to `DOUBLE PRECISION`

The `orders` table lives separately in `src/quantstack/execution/order_lifecycle.py` (`_ensure_table()` method), but all other operational tables are in `db.py`.

**Existing tables this section interacts with:**

- **`orders`** (PK: `order_id`) -- order lifecycle, arrival_price, exec_algo, status, fill info
- **`fills`** (PK: `order_id`) -- one row per order, summary fill data (symbol, side, quantities, fill_price, slippage_bps, commission)
- **`positions`** (PK: `symbol`) -- open positions with entry info, stops/targets, option fields
- **`closed_trades`** (PK: auto-increment `id`) -- realized P&L with strategy_id, regime, exit_reason

---

## Tables to Create

### 1. `fill_legs` (supports Section 02: Fill Legs)

Tracks individual fill events for partial fill tracking. The existing `fills` table remains as a summary view.

```sql
CREATE TABLE IF NOT EXISTS fill_legs (
    leg_id          BIGINT PRIMARY KEY DEFAULT nextval('fill_legs_seq'),
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    leg_sequence    INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    filled_at       TIMESTAMPTZ DEFAULT NOW(),
    venue           TEXT,
    UNIQUE (order_id, leg_sequence)
);
CREATE INDEX IF NOT EXISTS fill_legs_order_idx ON fill_legs (order_id);
```

Sequence: `CREATE SEQUENCE IF NOT EXISTS fill_legs_seq START 1`

**Design note:** FK to `orders(order_id)` enforces referential integrity. The `(order_id, leg_sequence)` unique constraint prevents duplicate legs for the same order.

### 2. `tca_parameters` (supports Section 06: TCA EWMA)

Stores EWMA-calibrated cost parameters per symbol and time-of-day bucket.

```sql
CREATE TABLE IF NOT EXISTS tca_parameters (
    symbol          TEXT NOT NULL,
    time_bucket     TEXT NOT NULL,
    ewma_spread_bps DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ewma_impact_bps DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ewma_total_bps  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    last_updated    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, time_bucket)
);
```

**Time buckets:** `morning` (9:30-11:00), `midday` (11:00-14:00), `afternoon` (14:00-15:30), `close` (15:30-16:00).

### 3. `day_trades` (supports Section 04: SEC Compliance -- PDT)

Records intraday round-trips for FINRA 4210 pattern day trader enforcement.

```sql
CREATE SEQUENCE IF NOT EXISTS day_trades_seq START 1;
CREATE TABLE IF NOT EXISTS day_trades (
    id              BIGINT PRIMARY KEY DEFAULT nextval('day_trades_seq'),
    symbol          TEXT NOT NULL,
    open_order_id   TEXT NOT NULL,
    close_order_id  TEXT NOT NULL,
    trade_date      DATE NOT NULL,
    quantity        INTEGER NOT NULL,
    account_equity  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS day_trades_date_idx ON day_trades (trade_date);
```

**Critical context:** Account is below $25K. FINRA 4210 limits accounts under $25K to 3 day trades per rolling 5-business-day window. The 4th is a hard block.

### 4. `pending_wash_losses` (supports Section 04: SEC Compliance -- Wash Sale)

Flags realized losses that may become wash sales if the same symbol is repurchased within 30 calendar days.

```sql
CREATE SEQUENCE IF NOT EXISTS pending_wash_losses_seq START 1;
CREATE TABLE IF NOT EXISTS pending_wash_losses (
    id                  BIGINT PRIMARY KEY DEFAULT nextval('pending_wash_losses_seq'),
    symbol              TEXT NOT NULL,
    loss_amount         DOUBLE PRECISION NOT NULL,
    sell_order_id       TEXT NOT NULL,
    sell_date           DATE NOT NULL,
    window_end          DATE NOT NULL,
    resolved            BOOLEAN DEFAULT FALSE,
    resolved_by_order_id TEXT
);
CREATE INDEX IF NOT EXISTS pending_wash_symbol_idx ON pending_wash_losses (symbol, window_end);
```

### 5. `wash_sale_flags` (supports Section 04: SEC Compliance -- Wash Sale)

Tracks confirmed wash sale events with disallowed loss amounts and adjusted cost basis.

```sql
CREATE SEQUENCE IF NOT EXISTS wash_sale_flags_seq START 1;
CREATE TABLE IF NOT EXISTS wash_sale_flags (
    id                      BIGINT PRIMARY KEY DEFAULT nextval('wash_sale_flags_seq'),
    loss_trade_id           BIGINT NOT NULL,
    replacement_order_id    TEXT NOT NULL,
    disallowed_loss         DOUBLE PRECISION NOT NULL,
    adjusted_cost_basis     DOUBLE PRECISION NOT NULL,
    wash_window_start       DATE NOT NULL,
    wash_window_end         DATE NOT NULL,
    flagged_at              TIMESTAMPTZ DEFAULT NOW()
);
```

### 6. `tax_lots` (supports Section 04: SEC Compliance -- Tax Lots)

FIFO tax lot tracking for Form 8949 reporting. Each buy creates a lot; each sell consumes lots oldest-first.

```sql
CREATE SEQUENCE IF NOT EXISTS tax_lots_seq START 1;
CREATE TABLE IF NOT EXISTS tax_lots (
    lot_id              BIGINT PRIMARY KEY DEFAULT nextval('tax_lots_seq'),
    symbol              TEXT NOT NULL,
    quantity            INTEGER NOT NULL,
    original_quantity   INTEGER NOT NULL,
    cost_basis          DOUBLE PRECISION NOT NULL,
    acquired_date       DATE NOT NULL,
    order_id            TEXT NOT NULL,
    closed_date         DATE,
    exit_price          DOUBLE PRECISION,
    realized_pnl        DOUBLE PRECISION,
    wash_sale_adjustment DOUBLE PRECISION DEFAULT 0.0,
    status              TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS tax_lots_symbol_status_idx ON tax_lots (symbol, status);
```

### 7. `algo_parent_orders` (supports Section 07: Algo Scheduler)

Parent order records for TWAP/VWAP execution. The EMS creates these when an order's `exec_algo` is not IMMEDIATE.

```sql
CREATE TABLE IF NOT EXISTS algo_parent_orders (
    parent_order_id         TEXT PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    side                    TEXT NOT NULL,
    total_quantity          INTEGER NOT NULL,
    algo_type               TEXT NOT NULL,
    start_time              TIMESTAMPTZ NOT NULL,
    end_time                TIMESTAMPTZ NOT NULL,
    arrival_price           DOUBLE PRECISION NOT NULL,
    max_participation_rate  DOUBLE PRECISION DEFAULT 0.02,
    status                  TEXT NOT NULL DEFAULT 'pending',
    filled_quantity         INTEGER DEFAULT 0,
    avg_fill_price          DOUBLE PRECISION DEFAULT 0.0,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);
```

### 8. `algo_child_orders` (supports Section 07: Algo Scheduler)

Child slices generated by TWAP/VWAP scheduling. Each child maps to a broker-submitted order.

```sql
CREATE SEQUENCE IF NOT EXISTS algo_child_orders_seq START 1;
CREATE TABLE IF NOT EXISTS algo_child_orders (
    child_id            TEXT PRIMARY KEY,
    parent_id           TEXT NOT NULL REFERENCES algo_parent_orders(parent_order_id),
    scheduled_time      TIMESTAMPTZ NOT NULL,
    target_quantity     INTEGER NOT NULL,
    filled_quantity     INTEGER DEFAULT 0,
    fill_price          DOUBLE PRECISION DEFAULT 0.0,
    status              TEXT NOT NULL DEFAULT 'pending',
    attempts            INTEGER DEFAULT 0,
    broker_order_id     TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS algo_child_parent_idx ON algo_child_orders (parent_id);
```

### 9. `algo_performance` (supports Section 08: TWAP/VWAP)

Post-completion performance metrics for algo orders. One row per completed parent.

```sql
CREATE TABLE IF NOT EXISTS algo_performance (
    parent_order_id             TEXT PRIMARY KEY,
    symbol                      TEXT NOT NULL,
    side                        TEXT NOT NULL,
    algo_type                   TEXT NOT NULL,
    total_qty                   INTEGER NOT NULL,
    filled_qty                  INTEGER NOT NULL,
    arrival_price               DOUBLE PRECISION NOT NULL,
    avg_fill_price              DOUBLE PRECISION NOT NULL,
    benchmark_vwap              DOUBLE PRECISION,
    implementation_shortfall_bps DOUBLE PRECISION NOT NULL,
    vwap_slippage_bps           DOUBLE PRECISION,
    delay_cost_bps              DOUBLE PRECISION DEFAULT 0.0,
    market_impact_bps           DOUBLE PRECISION DEFAULT 0.0,
    num_children                INTEGER NOT NULL,
    num_children_filled         INTEGER NOT NULL,
    num_children_failed         INTEGER DEFAULT 0,
    max_participation_rate      DOUBLE PRECISION,
    actual_participation_rate   DOUBLE PRECISION,
    decision_time               TIMESTAMPTZ NOT NULL,
    first_fill_time             TIMESTAMPTZ,
    last_fill_time              TIMESTAMPTZ,
    scheduled_end_time          TIMESTAMPTZ NOT NULL
);
```

### 10. `execution_audit` (supports Section 05: Audit Trail)

Best execution audit trail per SEC Rule 606 / FINRA Rule 5310. One row per fill event (IMMEDIATE orders and TWAP/VWAP child fills).

```sql
CREATE SEQUENCE IF NOT EXISTS execution_audit_seq START 1;
CREATE TABLE IF NOT EXISTS execution_audit (
    audit_id                BIGINT PRIMARY KEY DEFAULT nextval('execution_audit_seq'),
    order_id                TEXT NOT NULL,
    fill_leg_id             BIGINT,
    nbbo_bid                DOUBLE PRECISION,
    nbbo_ask                DOUBLE PRECISION,
    nbbo_midpoint           DOUBLE PRECISION,
    fill_price              DOUBLE PRECISION NOT NULL,
    fill_venue              TEXT,
    price_improvement_bps   DOUBLE PRECISION,
    algo_selected           TEXT NOT NULL,
    algo_rationale          TEXT DEFAULT '',
    timestamp_ns            BIGINT NOT NULL,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS execution_audit_order_idx ON execution_audit (order_id);
```

**Note:** `nbbo_bid`, `nbbo_ask`, `nbbo_midpoint`, and `price_improvement_bps` are nullable because NBBO capture may fail (network issue). Section 05 must handle the null case gracefully.

### 11. `slippage_accuracy` (supports Section 11: Slippage Enhancement)

Tracks predicted vs. realized slippage for model accuracy monitoring.

```sql
CREATE SEQUENCE IF NOT EXISTS slippage_accuracy_seq START 1;
CREATE TABLE IF NOT EXISTS slippage_accuracy (
    id                  BIGINT PRIMARY KEY DEFAULT nextval('slippage_accuracy_seq'),
    order_id            TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    predicted_bps       DOUBLE PRECISION NOT NULL,
    realized_bps        DOUBLE PRECISION NOT NULL,
    ratio               DOUBLE PRECISION NOT NULL,
    time_bucket         TEXT,
    recorded_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS slippage_accuracy_symbol_idx ON slippage_accuracy (symbol, recorded_at);
```

### 12. Columns added to `positions` (supports Section 13: Funding Costs)

```sql
ALTER TABLE positions ADD COLUMN IF NOT EXISTS margin_used DOUBLE PRECISION DEFAULT 0.0;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS cumulative_funding_cost DOUBLE PRECISION DEFAULT 0.0;
```

---

## Implementation

### File: `src/quantstack/db.py`

Add a single new migration function `_migrate_execution_layer_pg(conn)` containing all DDL from this section. Then register it in `run_migrations_pg()`.

**Migration function signature:**

```python
def _migrate_execution_layer_pg(conn: PgConnection) -> None:
    """Phase 6 execution layer tables: fill_legs, tca_parameters,
    day_trades, pending_wash_losses, wash_sale_flags, tax_lots,
    algo_parent_orders, algo_child_orders, algo_performance,
    execution_audit, slippage_accuracy. Plus positions columns."""
    ...
```

**Registration in `run_migrations_pg()`:** Add `_migrate_execution_layer_pg(conn)` after the last existing `_migrate_*` call (currently `_migrate_phase4_coordination_pg(conn)`) and before the `logger.info("[DB] PostgreSQL migrations complete")` line.

**Pattern to follow:** Use `_to_pg("""...""")` wrapper for all DDL (handles `DOUBLE` to `DOUBLE PRECISION` normalization). Use `conn.execute()` for each statement. Create sequences before their referencing tables. Add indexes after table creation.

**Foreign key consideration:** The `fill_legs.order_id` FK references `orders(order_id)`, but the `orders` table is created in `order_lifecycle.py`, not in `db.py`. The FK will succeed at runtime because `_ensure_table()` in the OMS runs at OMS construction time, which happens before any fill recording. However, if the migration runs before any OMS instance is created, the FK will fail. Two options:

1. **Safe option:** Drop the FK constraint from DDL, enforce referentially in application code. This matches the existing pattern -- no other table in `db.py` uses cross-module FKs.
2. **Strict option:** Move the `orders` DDL into `db.py` alongside everything else.

**Recommendation:** Use the safe option (no FK in DDL). Add a comment documenting the logical FK relationship. This avoids migration ordering issues and matches existing patterns.

Similarly, `algo_child_orders.parent_id` references `algo_parent_orders` which is created in the same migration function, so the FK is safe as long as the parent table DDL comes first.

---

## Tests

Tests for this section verify that the migration creates all tables with correct schemas and constraints. These run against a test database (existing test infrastructure uses `db_conn()` context managers).

**Test file:** `tests/unit/execution/test_schema_foundation.py`

### Test definitions (stubs):

```python
"""Tests for Phase 6 execution layer schema foundation.

Verifies all new tables are created by _migrate_execution_layer_pg()
with correct columns, constraints, and indexes.
"""

# Test: fill_legs table exists after migration with correct columns
#   - Verify columns: leg_id, order_id, leg_sequence, quantity, price, filled_at, venue
#   - Verify unique constraint on (order_id, leg_sequence)
#   - Verify index on order_id

# Test: tca_parameters table exists with composite PK (symbol, time_bucket)
#   - Insert two rows with same symbol but different time_bucket → succeeds
#   - Insert duplicate (symbol, time_bucket) → raises unique violation

# Test: day_trades table exists with auto-increment PK
#   - Verify index on trade_date

# Test: pending_wash_losses table exists with correct columns
#   - Verify index on (symbol, window_end)
#   - Verify resolved defaults to FALSE

# Test: wash_sale_flags table exists with auto-increment PK

# Test: tax_lots table exists with correct columns
#   - Verify status defaults to 'open'
#   - Verify index on (symbol, status)
#   - Verify wash_sale_adjustment defaults to 0.0

# Test: algo_parent_orders table exists with correct columns
#   - Verify status defaults to 'pending'
#   - Verify max_participation_rate defaults to 0.02

# Test: algo_child_orders table exists with FK to algo_parent_orders
#   - Insert child with nonexistent parent_id → raises FK violation
#   - Verify index on parent_id

# Test: algo_performance table exists with correct columns

# Test: execution_audit table exists with correct columns
#   - Verify NBBO fields are nullable
#   - Verify index on order_id

# Test: slippage_accuracy table exists with correct columns
#   - Verify index on (symbol, recorded_at)

# Test: positions table has new columns after migration
#   - Verify margin_used column exists with default 0.0
#   - Verify cumulative_funding_cost column exists with default 0.0

# Test: migration is idempotent — running twice does not raise errors
#   - Call _migrate_execution_layer_pg(conn) twice
#   - Verify all tables still exist with correct schemas

# Test: unique constraint on fill_legs (order_id, leg_sequence) rejects duplicates
#   - Insert (order_id='O1', leg_sequence=1) → succeeds
#   - Insert (order_id='O1', leg_sequence=1) again → raises IntegrityError
#   - Insert (order_id='O1', leg_sequence=2) → succeeds

# Test: tca_parameters upsert pattern works
#   - INSERT ... ON CONFLICT (symbol, time_bucket) DO UPDATE
#   - Verify updated values overwrite old values
```

### How to run:

```bash
uv run pytest tests/unit/execution/test_schema_foundation.py -v
```

---

## Dependencies

**This section depends on:** Nothing. It is the first section in the implementation order.

**This section blocks:** All other sections (02 through 14). Every section assumes its tables exist.

---

## Checklist

1. Write `_migrate_execution_layer_pg(conn)` in `src/quantstack/db.py` with all 11 table DDL statements and 2 ALTER TABLE statements
2. Create all sequences before their referencing tables
3. Register the function in `run_migrations_pg()` after `_migrate_phase4_coordination_pg(conn)`
4. Use `_to_pg()` wrapper for all DDL containing `DOUBLE` types
5. Drop FK on `fill_legs.order_id` (use safe option -- no cross-module FK, add comment)
6. Keep FK on `algo_child_orders.parent_id` (same migration, safe)
7. Write tests in `tests/unit/execution/test_schema_foundation.py`
8. Verify idempotency: running migration twice produces no errors
9. Verify all indexes are created
10. Verify default values on nullable/defaulted columns
