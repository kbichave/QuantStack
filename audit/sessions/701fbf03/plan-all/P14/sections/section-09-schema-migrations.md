# Section 09: Schema Migrations

## Objective

Add database tables and columns required by the new ML model types: model category tracking, conformal coverage metrics, and GNN graph snapshot storage.

## Files to Create/Modify

### Modified Files

- **`src/quantstack/db.py`** — Add migration statements for new tables and columns in the `_ensure_tables()` or equivalent migration function.

## Implementation Details

### Schema Changes

#### 1. Extend `model_registry` table

Add column:
```sql
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS model_category TEXT DEFAULT 'traditional_ml';
```

Valid values: `traditional_ml`, `transformer`, `gnn`, `rl`, `deep_hedge`.

This allows filtering and querying models by type (e.g., "show all transformer models").

#### 2. Add conformal coverage columns to `model_registry`

```sql
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS coverage_80 FLOAT;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS coverage_90 FLOAT;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS coverage_95 FLOAT;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS avg_interval_width FLOAT;
```

These store the empirical coverage rates and average prediction interval width from conformal calibration (section-01).

#### 3. New table: `conformal_coverage`

```sql
CREATE TABLE IF NOT EXISTS conformal_coverage (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    eval_date DATE NOT NULL,
    target_coverage FLOAT NOT NULL,        -- 0.80, 0.90, or 0.95
    empirical_coverage FLOAT NOT NULL,     -- actual coverage on eval set
    avg_width FLOAT NOT NULL,              -- average interval width
    n_samples INT NOT NULL,                -- size of evaluation set
    calibration_ok BOOLEAN NOT NULL,       -- True if |target - empirical| <= 0.03
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, eval_date, target_coverage)
);
```

This tracks coverage over time — important for detecting calibration drift (model uncertainty estimates degrading).

#### 4. New table: `gnn_graph_snapshots`

```sql
CREATE TABLE IF NOT EXISTS gnn_graph_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,
    n_nodes INT NOT NULL,
    n_edges INT NOT NULL,
    avg_correlation FLOAT,
    graph_json JSONB NOT NULL,              -- serialized graph structure
    build_duration_ms INT,                  -- how long graph construction took
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gnn_snapshots_date ON gnn_graph_snapshots(snapshot_date);
```

Stores daily graph snapshots for analysis and debugging. The `graph_json` field contains the adjacency list, node list, and edge types (not node features — those are recomputed from live data).

### Migration Strategy

All migrations use `IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS` — idempotent and safe to re-run. No destructive operations. This follows the existing pattern in `db.py`.

### Storage Helpers

Add convenience functions in `db.py` or a new `src/quantstack/data/pg_storage.py` section:

```python
def save_conformal_coverage(model_id: str, eval_date: date, target: float, empirical: float, avg_width: float, n_samples: int) -> None:
    """Insert conformal coverage evaluation result."""

def get_conformal_coverage_history(model_id: str, lookback_days: int = 90) -> list[dict]:
    """Fetch recent coverage evaluations for drift monitoring."""

def save_gnn_snapshot(snapshot_date: date, n_nodes: int, n_edges: int, avg_corr: float, graph_json: dict, build_ms: int) -> None:
    """Save daily GNN graph snapshot."""

def get_latest_gnn_snapshot() -> dict | None:
    """Load the most recent graph snapshot."""
```

## Dependencies

- **Internal**: `quantstack.db` (connection pool, migration infrastructure)

## Test Requirements

### `tests/unit/test_schema_migrations.py`

1. **Idempotent migrations**: Run migration twice — second run succeeds without error.
2. **model_category column**: Insert a row with `model_category='transformer'`, read it back.
3. **conformal_coverage table**: Insert and query coverage records.
4. **gnn_graph_snapshots table**: Insert and query graph snapshots.
5. **Unique constraints**: Duplicate (model_id, eval_date, target_coverage) in conformal_coverage raises appropriate error.
6. **Helper functions**: `save_conformal_coverage` and `get_conformal_coverage_history` round-trip correctly.

## Acceptance Criteria

- [ ] `model_registry` table has `model_category` column with default `traditional_ml`
- [ ] `model_registry` table has conformal coverage columns (`coverage_80`, `coverage_90`, `coverage_95`, `avg_interval_width`)
- [ ] `conformal_coverage` table created with unique constraint on (model_id, eval_date, target_coverage)
- [ ] `gnn_graph_snapshots` table created with date index
- [ ] All migrations are idempotent (safe to re-run)
- [ ] Storage helper functions for conformal coverage and GNN snapshots work correctly
- [ ] All unit tests pass
