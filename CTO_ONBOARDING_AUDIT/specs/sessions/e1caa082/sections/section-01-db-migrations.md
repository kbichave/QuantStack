# Section 01: Database Migrations

## Purpose

Phase 10 introduces 10 new tables and a pgvector extension dependency for the knowledge graph. This section defines every table schema, the migration function that creates them, and the Docker image change required to support vector operations. This is the foundation section -- sections 02 through 13 depend on these tables existing.

## Background

All database state lives in PostgreSQL (`quantstack` database). Schema management uses a home-grown idempotent migration system in `src/quantstack/db.py`. There is no Alembic or versioned migration tool -- every table is created via `CREATE TABLE IF NOT EXISTS` inside dedicated `_migrate_*_pg()` functions, called from the top-level `run_migrations_pg()`. The system uses an advisory lock (`pg_try_advisory_lock`) so that when multiple Docker services start simultaneously, only one runs migrations.

Key conventions observed in existing code:
- Each migration function is named `_migrate_<domain>_pg(conn: PgConnection) -> None`
- DDL is wrapped in `_to_pg()` which normalizes `DOUBLE` to `DOUBLE PRECISION` and `JSON` to `JSONB`
- All timestamps use `TIMESTAMPTZ DEFAULT NOW()`
- Primary keys are `TEXT` (UUIDs stored as text, not native `UUID` type)
- JSONB is used for flexible/semi-structured properties
- `ON CONFLICT` upsert patterns are defined at write-time, not in DDL
- Each migration function logs completion via `logger.debug("[DB] ... migrated")`
- The pgvector extension is already enabled (via `_migrate_hnsw_index_pg`) and the Docker image is already `pgvector/pgvector:pg16`

## Docker Image

The `docker-compose.yml` already uses `pgvector/pgvector:pg16` for the `postgres` service. No image change is required. The `CREATE EXTENSION IF NOT EXISTS vector` statement in `_migrate_hnsw_index_pg` already ensures the extension is available.

## Tests First

File: `tests/unit/test_db_migrations.py`

```python
"""Tests for Phase 10 database migrations.

Each test calls the migration function twice to verify idempotency,
then asserts the table exists with the expected columns.
"""

# Test: ensure_schema creates all 10 new tables (idempotent -- run twice, no error)
def test_phase10_tables_created_idempotently(pg_conn):
    """Run _migrate_phase10_pg twice. Second call must not raise.
    Query information_schema.tables for all 10 table names."""

# Test: tool_health table has expected columns and types
def test_tool_health_schema(pg_conn):
    """Verify tool_health has: tool_name TEXT PK, invocation_count INTEGER,
    success_count INTEGER, failure_count INTEGER, avg_latency_ms DOUBLE PRECISION,
    last_invoked TIMESTAMPTZ, last_error TEXT, status TEXT."""

# Test: tool_demand_signals table has expected columns
def test_tool_demand_signals_schema(pg_conn):
    """Verify tool_demand_signals has: id TEXT PK, search_query TEXT,
    requesting_agent TEXT, matched_tool TEXT, created_at TIMESTAMPTZ."""

# Test: autoresearch_experiments table has expected columns
def test_autoresearch_experiments_schema(pg_conn):
    """Verify autoresearch_experiments has: experiment_id TEXT PK, night_date TEXT,
    hypothesis JSONB, hypothesis_source TEXT, oos_ic DOUBLE PRECISION,
    sharpe DOUBLE PRECISION, cost_tokens INTEGER, cost_usd DOUBLE PRECISION,
    duration_seconds INTEGER, status TEXT, rejection_reason TEXT,
    created_at TIMESTAMPTZ."""

# Test: feature_candidates table has expected columns
def test_feature_candidates_schema(pg_conn):
    """Verify feature_candidates has: feature_id TEXT PK, feature_name TEXT,
    definition TEXT, source TEXT, ic DOUBLE PRECISION, ic_stability DOUBLE PRECISION,
    correlation_group TEXT, status TEXT, screening_date TEXT, decay_date TEXT."""

# Test: failure_mode_stats table has expected columns
def test_failure_mode_stats_schema(pg_conn):
    """Verify failure_mode_stats has: id TEXT PK, failure_mode TEXT,
    window_start TEXT, window_end TEXT, frequency INTEGER,
    cumulative_pnl_impact DOUBLE PRECISION, avg_loss_size DOUBLE PRECISION,
    affected_strategies JSONB, updated_at TIMESTAMPTZ."""

# Test: kg_nodes table has vector(1536) column
def test_kg_nodes_has_vector_column(pg_conn):
    """Verify kg_nodes.embedding column is of type vector(1536).
    Query pg_catalog for column type."""

# Test: kg_nodes table has HNSW index on embedding column
def test_kg_nodes_has_hnsw_index(pg_conn):
    """Verify an HNSW index exists on kg_nodes.embedding.
    Query pg_indexes for index using hnsw access method."""

# Test: kg_edges table has expected columns including valid_from/valid_to
def test_kg_edges_schema(pg_conn):
    """Verify kg_edges has: edge_id TEXT PK, source_id TEXT, target_id TEXT,
    edge_type TEXT, weight DOUBLE PRECISION, properties JSONB,
    valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ, created_at TIMESTAMPTZ."""

# Test: consensus_log table has expected columns
def test_consensus_log_schema(pg_conn):
    """Verify consensus_log has: decision_id TEXT PK, signal_id TEXT, symbol TEXT,
    notional DOUBLE PRECISION, bull_vote TEXT, bull_confidence DOUBLE PRECISION,
    bull_reasoning TEXT, bear_vote TEXT, bear_confidence DOUBLE PRECISION,
    bear_reasoning TEXT, arbiter_vote TEXT, arbiter_confidence DOUBLE PRECISION,
    arbiter_reasoning TEXT, consensus_level TEXT, final_sizing_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ."""

# Test: daily_mandates table has expected columns
def test_daily_mandates_schema(pg_conn):
    """Verify daily_mandates has: mandate_id TEXT PK, date TEXT UNIQUE,
    regime_assessment TEXT, allowed_sectors JSONB, blocked_sectors JSONB,
    max_new_positions INTEGER, max_daily_notional DOUBLE PRECISION,
    strategy_directives JSONB, risk_overrides JSONB, focus_areas JSONB,
    reasoning TEXT, created_at TIMESTAMPTZ."""

# Test: meta_optimizations table has expected columns
def test_meta_optimizations_schema(pg_conn):
    """Verify meta_optimizations has: optimization_id TEXT PK, agent_id TEXT,
    change_type TEXT, change_summary TEXT, before_metrics JSONB,
    after_metrics JSONB, status TEXT, reverted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ."""

# Test: all tables have TIMESTAMPTZ timestamps (not plain TIMESTAMP)
def test_all_phase10_tables_use_timestamptz(pg_conn):
    """Query information_schema.columns for all 10 tables.
    Any column containing 'timestamp' in data_type must be 'timestamp with time zone'."""

# Test: pgvector extension is available
def test_pgvector_extension_exists(pg_conn):
    """SELECT * FROM pg_extension WHERE extname = 'vector' must return a row."""
```

## Implementation

### File: `src/quantstack/db.py`

Add a single new migration function `_migrate_phase10_pg(conn)` and register it in `run_migrations_pg()`.

**Registration point.** In `run_migrations_pg()`, add the call after the existing Phase 9 migrations (after `_migrate_llm_config_pg(conn)`):

```python
            _migrate_llm_config_pg(conn)
            # Phase 10: Advanced Research
            _migrate_phase10_pg(conn)

            logger.info("[DB] PostgreSQL migrations complete")
```

**Migration function.** The function creates all 10 tables in a single migration block. This keeps Phase 10 tables grouped and auditable.

```python
def _migrate_phase10_pg(conn: PgConnection) -> None:
    """Phase 10: Advanced Research tables.

    10 new tables for tool lifecycle, overnight autoresearch, feature factory,
    knowledge graph, consensus validation, governance, and meta-agents.
    Additive only -- no destructive DDL.
    """
```

### Table 1: `tool_health`

Per-tool invocation metrics. Used by the tool lifecycle health monitor (section-02) to auto-disable degraded tools.

```sql
CREATE TABLE IF NOT EXISTS tool_health (
    tool_name        TEXT PRIMARY KEY,
    invocation_count INTEGER DEFAULT 0,
    success_count    INTEGER DEFAULT 0,
    failure_count    INTEGER DEFAULT 0,
    avg_latency_ms   DOUBLE PRECISION DEFAULT 0.0,
    last_invoked     TIMESTAMPTZ,
    last_error       TEXT,
    status           TEXT DEFAULT 'active',
    updated_at       TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: `ON CONFLICT (tool_name) DO UPDATE SET invocation_count = tool_health.invocation_count + 1, ...`

### Table 2: `tool_demand_signals`

Tracks when agents search for planned (not-yet-implemented) tools. Weekly aggregation ranks by frequency to prioritize tool synthesis.

```sql
CREATE TABLE IF NOT EXISTS tool_demand_signals (
    id               TEXT PRIMARY KEY,
    search_query     TEXT NOT NULL,
    requesting_agent TEXT NOT NULL,
    matched_tool     TEXT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: simple INSERT (append-only log, no upsert needed).

### Table 3: `autoresearch_experiments`

Overnight experiment log. Each row is one hypothesis tested during the 20:00-04:00 nightly loop.

```sql
CREATE TABLE IF NOT EXISTS autoresearch_experiments (
    experiment_id    TEXT PRIMARY KEY,
    night_date       TEXT NOT NULL,
    hypothesis       JSONB NOT NULL,
    hypothesis_source TEXT NOT NULL,
    oos_ic           DOUBLE PRECISION,
    sharpe           DOUBLE PRECISION,
    cost_tokens      INTEGER DEFAULT 0,
    cost_usd         DOUBLE PRECISION DEFAULT 0.0,
    duration_seconds INTEGER DEFAULT 0,
    status           TEXT DEFAULT 'tested',
    rejection_reason TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: INSERT on creation, then UPDATE status to 'winner'/'validated'/'rejected' as the experiment progresses. `ON CONFLICT (experiment_id) DO UPDATE` for crash recovery (runner restarts and re-processes the same experiment).

Create an index on `(night_date, status)` for the morning validator query that fetches all winners for a given night:

```sql
CREATE INDEX IF NOT EXISTS idx_autoresearch_night_status
    ON autoresearch_experiments (night_date, status)
```

### Table 4: `feature_candidates`

Enumerated and screened features from the autonomous feature factory (section-08).

```sql
CREATE TABLE IF NOT EXISTS feature_candidates (
    feature_id        TEXT PRIMARY KEY,
    feature_name      TEXT NOT NULL,
    definition        TEXT NOT NULL,
    source            TEXT NOT NULL,
    ic                DOUBLE PRECISION,
    ic_stability      DOUBLE PRECISION,
    correlation_group TEXT,
    status            TEXT DEFAULT 'candidate',
    screening_date    TEXT,
    decay_date        TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: INSERT during enumeration, UPDATE during screening (set ic, ic_stability, status), UPDATE during monitoring (set decay_date, status='decayed').

Create an index on `status` for the drift monitor query that fetches active features:

```sql
CREATE INDEX IF NOT EXISTS idx_feature_candidates_status
    ON feature_candidates (status)
```

### Table 5: `failure_mode_stats`

Rolling 30-day failure mode aggregation. Used by the error-driven research pipeline (section-03) to prioritize which failure modes get research attention.

```sql
CREATE TABLE IF NOT EXISTS failure_mode_stats (
    id                    TEXT PRIMARY KEY,
    failure_mode          TEXT NOT NULL,
    window_start          TEXT NOT NULL,
    window_end            TEXT NOT NULL,
    frequency             INTEGER DEFAULT 0,
    cumulative_pnl_impact DOUBLE PRECISION DEFAULT 0.0,
    avg_loss_size         DOUBLE PRECISION DEFAULT 0.0,
    affected_strategies   JSONB DEFAULT '[]'::jsonb,
    updated_at            TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: `ON CONFLICT (id) DO UPDATE` -- recalculated daily as the rolling window shifts.

### Table 6: `kg_nodes`

Knowledge graph nodes. Each node has a type (strategy, factor, hypothesis, result, instrument, regime, evidence), properties stored as JSONB, and a vector embedding for semantic similarity search.

```sql
CREATE TABLE IF NOT EXISTS kg_nodes (
    node_id     TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,
    name        TEXT NOT NULL,
    properties  JSONB DEFAULT '{}'::jsonb,
    embedding   vector(1536),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
)
```

The `vector(1536)` column requires the pgvector extension (already enabled via `_migrate_hnsw_index_pg`). The 1536 dimension matches Amazon Titan Text Embeddings v2 output size.

Create an HNSW index for cosine similarity search:

```sql
CREATE INDEX IF NOT EXISTS idx_kg_nodes_embedding_hnsw
    ON kg_nodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 100)
```

Create an index on `node_type` for type-filtered queries:

```sql
CREATE INDEX IF NOT EXISTS idx_kg_nodes_type
    ON kg_nodes (node_type)
```

### Table 7: `kg_edges`

Knowledge graph edges. Temporal validity (`valid_from`/`valid_to`) prevents stale relationships from influencing decisions -- factor correlations change over time.

```sql
CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id     TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES kg_nodes(node_id),
    target_id   TEXT NOT NULL REFERENCES kg_nodes(node_id),
    edge_type   TEXT NOT NULL,
    weight       DOUBLE PRECISION DEFAULT 1.0,
    properties  JSONB DEFAULT '{}'::jsonb,
    valid_from  TIMESTAMPTZ DEFAULT NOW(),
    valid_to    TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT NOW()
)
```

Create indexes for graph traversal (both directions) and temporal queries:

```sql
CREATE INDEX IF NOT EXISTS idx_kg_edges_source
    ON kg_edges (source_id, edge_type)
```

```sql
CREATE INDEX IF NOT EXISTS idx_kg_edges_target
    ON kg_edges (target_id, edge_type)
```

```sql
CREATE INDEX IF NOT EXISTS idx_kg_edges_valid
    ON kg_edges (valid_from, valid_to)
    WHERE valid_to IS NULL OR valid_to > NOW()
```

Note on foreign keys: the `REFERENCES kg_nodes(node_id)` constraints enforce referential integrity. This is a departure from the existing codebase pattern (most tables do not use foreign keys). The knowledge graph's correctness depends on edge endpoints being valid nodes -- dangling edges would corrupt graph traversal queries. The FK constraint cost is negligible because edge inserts always follow node inserts in the same transaction.

### Table 8: `consensus_log`

Records the 3-agent consensus decision for trades above the notional threshold (section-11).

```sql
CREATE TABLE IF NOT EXISTS consensus_log (
    decision_id        TEXT PRIMARY KEY,
    signal_id          TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    notional           DOUBLE PRECISION NOT NULL,
    bull_vote          TEXT NOT NULL,
    bull_confidence    DOUBLE PRECISION,
    bull_reasoning     TEXT,
    bear_vote          TEXT NOT NULL,
    bear_confidence    DOUBLE PRECISION,
    bear_reasoning     TEXT,
    arbiter_vote       TEXT NOT NULL,
    arbiter_confidence DOUBLE PRECISION,
    arbiter_reasoning  TEXT,
    consensus_level    TEXT NOT NULL,
    final_sizing_pct   DOUBLE PRECISION NOT NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: single INSERT per consensus decision. No updates -- decisions are immutable once recorded.

### Table 9: `daily_mandates`

CIO agent's daily directives (section-12). One row per trading day. The `date` column has a UNIQUE constraint so that at most one mandate exists per day.

```sql
CREATE TABLE IF NOT EXISTS daily_mandates (
    mandate_id          TEXT PRIMARY KEY,
    date                TEXT NOT NULL UNIQUE,
    regime_assessment   TEXT,
    allowed_sectors     JSONB DEFAULT '[]'::jsonb,
    blocked_sectors     JSONB DEFAULT '[]'::jsonb,
    max_new_positions   INTEGER DEFAULT 0,
    max_daily_notional  DOUBLE PRECISION DEFAULT 0.0,
    strategy_directives JSONB DEFAULT '{}'::jsonb,
    risk_overrides      JSONB DEFAULT '{}'::jsonb,
    focus_areas         JSONB DEFAULT '[]'::jsonb,
    reasoning           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: `ON CONFLICT (date) DO UPDATE` -- if the CIO runs twice in a day (e.g., intraday regime change triggers re-evaluation), the latest mandate replaces the earlier one.

### Table 10: `meta_optimizations`

Audit trail for metacognitive agent changes (section-13). Tracks what was changed, before/after metrics, and whether the change was reverted.

```sql
CREATE TABLE IF NOT EXISTS meta_optimizations (
    optimization_id TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    change_type     TEXT NOT NULL,
    change_summary  TEXT,
    before_metrics  JSONB DEFAULT '{}'::jsonb,
    after_metrics   JSONB DEFAULT '{}'::jsonb,
    status          TEXT DEFAULT 'applied',
    reverted_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
)
```

Write pattern: INSERT on application, UPDATE `after_metrics` after evaluation period, UPDATE `status='reverted'` and `reverted_at` if auto-revert triggers.

Create an index on `(agent_id, created_at)` for the meta-agent query that checks recent optimizations:

```sql
CREATE INDEX IF NOT EXISTS idx_meta_optimizations_agent
    ON meta_optimizations (agent_id, created_at DESC)
```

## Dependencies

- **No dependencies.** This is a foundation section. It must be implemented before sections 02-13.
- The pgvector extension and Docker image are already in place (no changes to `docker-compose.yml` needed).

## Deployment Notes

- All DDL is idempotent (`IF NOT EXISTS`). Safe to run against an existing database with partial Phase 10 tables.
- The advisory lock in `run_migrations_pg()` ensures only one service runs migrations. No coordination needed across Docker services.
- Adding new fields to `ResearchState` and `TradingState` (section-04) requires a clean restart with empty checkpoint state. That is section-04's concern, not this section's. The database tables here are independent of graph state.
- The `kg_nodes` HNSW index build is O(n) but the table starts empty. Index creation is instant. If backfilling large volumes of nodes later (section-10), the index will auto-maintain.
