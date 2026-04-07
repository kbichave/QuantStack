"""Tests for Phase 10 database migrations.

Each test calls the migration function twice to verify idempotency,
then asserts the table exists with the expected columns.
"""

import pytest
from quantstack.db import run_migrations, pg_conn


PHASE10_TABLES = [
    "tool_health",
    "tool_demand_signals",
    "autoresearch_experiments",
    "feature_candidates",
    "failure_mode_stats",
    "kg_nodes",
    "kg_edges",
    "consensus_log",
    "daily_mandates",
    "meta_optimizations",
]


@pytest.fixture
def migrated_conn(trading_ctx):
    """Return a connection with all migrations applied."""
    with pg_conn() as conn:
        run_migrations(conn)
        yield conn


def _table_columns(conn, table_name: str) -> dict[str, str]:
    """Return {column_name: data_type} for a table."""
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = %s ORDER BY ordinal_position",
        [table_name],
    ).fetchall()
    return {r["column_name"]: r["data_type"] for r in rows}


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        [table_name],
    ).fetchone()
    return row[0]


class TestPhase10Idempotency:
    def test_phase10_tables_created_idempotently(self, migrated_conn):
        """Run migrations twice. Second call must not raise.
        All 10 tables must exist."""
        # First run already happened in fixture; run again
        run_migrations(migrated_conn)
        for table in PHASE10_TABLES:
            assert _table_exists(migrated_conn, table), f"{table} not created"


class TestToolHealthSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "tool_health")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "tool_health")
        assert cols["tool_name"] == "text"
        assert cols["invocation_count"] == "integer"
        assert cols["success_count"] == "integer"
        assert cols["failure_count"] == "integer"
        assert cols["avg_latency_ms"] == "double precision"
        assert cols["last_invoked"] == "timestamp with time zone"
        assert cols["last_error"] == "text"
        assert cols["status"] == "text"


class TestToolDemandSignalsSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "tool_demand_signals")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "tool_demand_signals")
        assert cols["id"] == "text"
        assert cols["search_query"] == "text"
        assert cols["requesting_agent"] == "text"
        assert cols["matched_tool"] == "text"
        assert cols["created_at"] == "timestamp with time zone"


class TestAutoresearchExperimentsSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "autoresearch_experiments")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "autoresearch_experiments")
        assert cols["experiment_id"] == "text"
        assert cols["night_date"] == "text"
        assert cols["hypothesis"] == "jsonb"
        assert cols["hypothesis_source"] == "text"
        assert cols["oos_ic"] == "double precision"
        assert cols["sharpe"] == "double precision"
        assert cols["cost_tokens"] == "integer"
        assert cols["cost_usd"] == "double precision"
        assert cols["duration_seconds"] == "integer"
        assert cols["status"] == "text"
        assert cols["rejection_reason"] == "text"
        assert cols["created_at"] == "timestamp with time zone"


class TestFeatureCandidatesSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "feature_candidates")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "feature_candidates")
        assert cols["feature_id"] == "text"
        assert cols["feature_name"] == "text"
        assert cols["definition"] == "text"
        assert cols["source"] == "text"
        assert cols["ic"] == "double precision"
        assert cols["ic_stability"] == "double precision"
        assert cols["correlation_group"] == "text"
        assert cols["status"] == "text"
        assert cols["screening_date"] == "text"
        assert cols["decay_date"] == "text"


class TestFailureModeStatsSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "failure_mode_stats")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "failure_mode_stats")
        assert cols["id"] == "text"
        assert cols["failure_mode"] == "text"
        assert cols["window_start"] == "text"
        assert cols["window_end"] == "text"
        assert cols["frequency"] == "integer"
        assert cols["cumulative_pnl_impact"] == "double precision"
        assert cols["avg_loss_size"] == "double precision"
        assert cols["affected_strategies"] == "jsonb"
        assert cols["updated_at"] == "timestamp with time zone"


class TestKgNodesSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "kg_nodes")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "kg_nodes")
        assert cols["node_id"] == "text"
        assert cols["node_type"] == "text"
        assert cols["name"] == "text"
        assert cols["properties"] == "jsonb"
        assert cols["created_at"] == "timestamp with time zone"
        assert cols["updated_at"] == "timestamp with time zone"

    def test_vector_column(self, migrated_conn):
        """Verify kg_nodes.embedding column is of type vector(1536)."""
        row = migrated_conn.execute(
            "SELECT format_type(atttypid, atttypmod) AS col_type "
            "FROM pg_attribute "
            "WHERE attrelid = 'kg_nodes'::regclass "
            "AND attname = 'embedding'",
        ).fetchone()
        assert row is not None, "embedding column not found"
        assert row["col_type"] == "vector(1536)"

    def test_hnsw_index(self, migrated_conn):
        """Verify an HNSW index exists on kg_nodes.embedding."""
        row = migrated_conn.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'kg_nodes' AND indexname = 'idx_kg_nodes_embedding_hnsw'",
        ).fetchone()
        assert row is not None, "HNSW index not found"
        assert "hnsw" in row["indexdef"].lower()


class TestKgEdgesSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "kg_edges")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "kg_edges")
        assert cols["edge_id"] == "text"
        assert cols["source_id"] == "text"
        assert cols["target_id"] == "text"
        assert cols["edge_type"] == "text"
        assert cols["weight"] == "double precision"
        assert cols["properties"] == "jsonb"
        assert cols["valid_from"] == "timestamp with time zone"
        assert cols["valid_to"] == "timestamp with time zone"
        assert cols["created_at"] == "timestamp with time zone"


class TestConsensusLogSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "consensus_log")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "consensus_log")
        assert cols["decision_id"] == "text"
        assert cols["signal_id"] == "text"
        assert cols["symbol"] == "text"
        assert cols["notional"] == "double precision"
        assert cols["bull_vote"] == "text"
        assert cols["bull_confidence"] == "double precision"
        assert cols["bull_reasoning"] == "text"
        assert cols["bear_vote"] == "text"
        assert cols["bear_confidence"] == "double precision"
        assert cols["bear_reasoning"] == "text"
        assert cols["arbiter_vote"] == "text"
        assert cols["arbiter_confidence"] == "double precision"
        assert cols["arbiter_reasoning"] == "text"
        assert cols["consensus_level"] == "text"
        assert cols["final_sizing_pct"] == "double precision"
        assert cols["created_at"] == "timestamp with time zone"


class TestDailyMandatesSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "daily_mandates")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "daily_mandates")
        assert cols["mandate_id"] == "text"
        assert cols["date"] == "text"
        assert cols["regime_assessment"] == "text"
        assert cols["allowed_sectors"] == "jsonb"
        assert cols["blocked_sectors"] == "jsonb"
        assert cols["max_new_positions"] == "integer"
        assert cols["max_daily_notional"] == "double precision"
        assert cols["strategy_directives"] == "jsonb"
        assert cols["risk_overrides"] == "jsonb"
        assert cols["focus_areas"] == "jsonb"
        assert cols["reasoning"] == "text"
        assert cols["created_at"] == "timestamp with time zone"


class TestMetaOptimizationsSchema:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "meta_optimizations")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "meta_optimizations")
        assert cols["optimization_id"] == "text"
        assert cols["agent_id"] == "text"
        assert cols["change_type"] == "text"
        assert cols["change_summary"] == "text"
        assert cols["before_metrics"] == "jsonb"
        assert cols["after_metrics"] == "jsonb"
        assert cols["status"] == "text"
        assert cols["reverted_at"] == "timestamp with time zone"
        assert cols["created_at"] == "timestamp with time zone"


class TestAllPhase10TablesTimestamptz:
    def test_all_timestamps_are_timestamptz(self, migrated_conn):
        """Any timestamp column in Phase 10 tables must be 'timestamp with time zone'."""
        for table in PHASE10_TABLES:
            cols = _table_columns(migrated_conn, table)
            for col_name, dtype in cols.items():
                if "timestamp" in dtype:
                    assert dtype == "timestamp with time zone", (
                        f"{table}.{col_name} is {dtype}, expected timestamp with time zone"
                    )


class TestPgvectorExtension:
    def test_pgvector_extension_exists(self, migrated_conn):
        row = migrated_conn.execute(
            "SELECT * FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row is not None, "pgvector extension not installed"
