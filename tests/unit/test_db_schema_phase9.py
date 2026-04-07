"""Tests for Phase 9 database schema — new tables and column additions."""

import pytest
from quantstack.db import run_migrations, pg_conn


@pytest.fixture
def migrated_conn(trading_ctx):
    """Return a connection with Phase 9 migrations applied."""
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


class TestCorporateActionsTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "corporate_actions")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "corporate_actions")
        assert "symbol" in cols
        assert "event_type" in cols
        assert "source" in cols
        assert "effective_date" in cols
        assert "announcement_date" in cols
        assert "raw_payload" in cols
        assert "processed" in cols
        assert "created_at" in cols

    def test_unique_constraint_rejects_duplicate(self, migrated_conn):
        migrated_conn.execute(
            "INSERT INTO corporate_actions (symbol, event_type, source, effective_date) "
            "VALUES ('AAPL', 'dividend', 'alpha_vantage', '2024-01-15')"
        )
        with pytest.raises(Exception):  # IntegrityError
            migrated_conn.execute(
                "INSERT INTO corporate_actions (symbol, event_type, source, effective_date) "
                "VALUES ('AAPL', 'dividend', 'alpha_vantage', '2024-01-15')"
            )

    def test_unique_constraint_allows_different_event_type(self, migrated_conn):
        migrated_conn.execute(
            "INSERT INTO corporate_actions (symbol, event_type, source, effective_date) "
            "VALUES ('AAPL', 'dividend', 'alpha_vantage', '2024-06-15')"
        )
        # Different event_type should succeed
        migrated_conn.execute(
            "INSERT INTO corporate_actions (symbol, event_type, source, effective_date) "
            "VALUES ('AAPL', 'split', 'alpha_vantage', '2024-06-15')"
        )


class TestSplitAdjustmentsTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "split_adjustments")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "split_adjustments")
        assert "symbol" in cols
        assert "effective_date" in cols
        assert "event_type" in cols
        assert "split_ratio" in cols
        assert "old_quantity" in cols
        assert "new_quantity" in cols
        assert "old_cost_basis" in cols
        assert "new_cost_basis" in cols

    def test_unique_constraint_rejects_duplicate(self, migrated_conn):
        migrated_conn.execute(
            "INSERT INTO split_adjustments (symbol, effective_date, event_type, split_ratio, "
            "old_quantity, new_quantity, old_cost_basis, new_cost_basis) "
            "VALUES ('AAPL', '2024-01-15', 'split', 4.0, 100, 400, 150.0, 37.5)"
        )
        with pytest.raises(Exception):
            migrated_conn.execute(
                "INSERT INTO split_adjustments (symbol, effective_date, event_type, split_ratio, "
                "old_quantity, new_quantity, old_cost_basis, new_cost_basis) "
                "VALUES ('AAPL', '2024-01-15', 'split', 4.0, 100, 400, 150.0, 37.5)"
            )


class TestSystemAlertsTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "system_alerts")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "system_alerts")
        assert "id" in cols
        assert "category" in cols
        assert "severity" in cols
        assert "status" in cols
        assert "source" in cols
        assert "title" in cols
        assert "detail" in cols
        assert "metadata" in cols
        assert "acknowledged_by" in cols
        assert "acknowledged_at" in cols
        assert "escalated_at" in cols
        assert "resolved_at" in cols
        assert "resolution" in cols

    def test_bigserial_auto_increments(self, migrated_conn):
        migrated_conn.execute(
            "INSERT INTO system_alerts (category, severity, source, title) "
            "VALUES ('risk_breach', 'critical', 'test', 'Alert 1')"
        )
        migrated_conn.execute(
            "INSERT INTO system_alerts (category, severity, source, title) "
            "VALUES ('risk_breach', 'warning', 'test', 'Alert 2')"
        )
        rows = migrated_conn.execute(
            "SELECT id FROM system_alerts ORDER BY id"
        ).fetchall()
        assert len(rows) >= 2
        assert rows[-1]["id"] == rows[-2]["id"] + 1


class TestEventBusAckColumns:
    def test_ack_columns_exist(self, migrated_conn):
        cols = _table_columns(migrated_conn, "loop_events")
        assert "requires_ack" in cols
        assert "expected_ack_by" in cols
        assert "acked_at" in cols
        assert "acked_by" in cols

    def test_dead_letter_events_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "dead_letter_events")
        cols = _table_columns(migrated_conn, "dead_letter_events")
        assert "original_event_id" in cols
        assert "event_type" in cols
        assert "retry_count" in cols
        assert "dead_lettered_at" in cols


class TestFactorConfigTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "factor_config")

    def test_default_rows_populated(self, migrated_conn):
        rows = migrated_conn.execute(
            "SELECT config_key, value FROM factor_config ORDER BY config_key"
        ).fetchall()
        config = {r["config_key"]: r["value"] for r in rows}
        assert config["beta_drift_threshold"] == "0.3"
        assert config["sector_max_pct"] == "40"
        assert config["momentum_crowding_pct"] == "70"
        assert config["benchmark_symbol"] == "SPY"

    def test_idempotent_no_duplicate_rows(self, migrated_conn):
        # Run migrations again
        run_migrations(migrated_conn)
        rows = migrated_conn.execute(
            "SELECT COUNT(*) as cnt FROM factor_config"
        ).fetchone()
        assert rows["cnt"] == 4


class TestFactorExposureHistoryTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "factor_exposure_history")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "factor_exposure_history")
        assert "portfolio_beta" in cols
        assert "sector_weights" in cols
        assert "style_scores" in cols
        assert "momentum_crowding_pct" in cols
        assert "benchmark_symbol" in cols
        assert "alerts_triggered" in cols


class TestCycleAttributionTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "cycle_attribution")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "cycle_attribution")
        assert "cycle_id" in cols
        assert "graph_cycle_number" in cols
        assert "total_pnl" in cols
        assert "factor_contribution" in cols
        assert "timing_contribution" in cols
        assert "selection_contribution" in cols
        assert "cost_contribution" in cols
        assert "per_position" in cols


class TestLlmConfigTable:
    def test_table_created(self, migrated_conn):
        assert _table_exists(migrated_conn, "llm_config")

    def test_columns(self, migrated_conn):
        cols = _table_columns(migrated_conn, "llm_config")
        assert "tier" in cols
        assert "provider" in cols
        assert "model" in cols
        assert "fallback_order" in cols

    def test_no_default_rows(self, migrated_conn):
        rows = migrated_conn.execute(
            "SELECT COUNT(*) as cnt FROM llm_config"
        ).fetchone()
        assert rows["cnt"] == 0


class TestTradingStateField:
    def test_cycle_attribution_field_accepted(self):
        from quantstack.graphs.state import TradingState
        state = TradingState(cycle_attribution={"total_pnl": 100.0})
        assert state.cycle_attribution == {"total_pnl": 100.0}

    def test_cycle_attribution_default_empty(self):
        from quantstack.graphs.state import TradingState
        state = TradingState()
        assert state.cycle_attribution == {}
