"""Tests for Phase 6 execution layer schema foundation.

Verifies all new tables are created by _migrate_execution_layer_pg()
with correct columns, constraints, and indexes.
"""

import pytest
from quantstack.db import db_conn, _migrate_execution_layer_pg


@pytest.fixture()
def conn():
    """Yield a live DB connection with execution layer tables migrated.

    Each test runs inside a SAVEPOINT that is rolled back on teardown,
    so tests never leave residual data in shared tables.
    """
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        (table_name,),
    ).fetchone()
    return row[0]


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s)",
        (table_name, column_name),
    ).fetchone()
    return row[0]


def _index_exists(conn, index_name: str) -> bool:
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = %s)",
        (index_name,),
    ).fetchone()
    return row[0]


def _assert_constraint_violation(conn, sql: str):
    """Assert that a SQL statement raises a constraint violation.

    Uses the raw psycopg connection with a nested savepoint so the
    PgConnection wrapper's auto-rollback doesn't destroy the outer
    test savepoint.
    """
    raw = conn._raw
    raw.execute("SAVEPOINT constraint_check")
    cur = raw.cursor()
    with pytest.raises(Exception):
        cur.execute(sql)
    raw.execute("ROLLBACK TO SAVEPOINT constraint_check")
    cur.close()


class TestFillLegs:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "fill_legs")

    def test_columns(self, conn):
        for col in ("leg_id", "order_id", "leg_sequence", "quantity", "price", "filled_at", "venue"):
            assert _column_exists(conn, "fill_legs", col), f"Missing column: {col}"

    def test_unique_constraint_rejects_duplicates(self, conn):
        conn.execute(
            "INSERT INTO fill_legs (order_id, leg_sequence, quantity, price) "
            "VALUES ('O1', 1, 100, 50.0)"
        )
        _assert_constraint_violation(
            conn,
            "INSERT INTO fill_legs (order_id, leg_sequence, quantity, price) "
            "VALUES ('O1', 1, 50, 51.0)",
        )

    def test_unique_allows_different_sequence(self, conn):
        conn.execute(
            "INSERT INTO fill_legs (order_id, leg_sequence, quantity, price) "
            "VALUES ('O2', 1, 100, 50.0)"
        )
        conn.execute(
            "INSERT INTO fill_legs (order_id, leg_sequence, quantity, price) "
            "VALUES ('O2', 2, 50, 51.0)"
        )

    def test_index_exists(self, conn):
        assert _index_exists(conn, "fill_legs_order_idx")


class TestTcaParameters:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "tca_parameters")

    def test_composite_pk_allows_different_buckets(self, conn):
        conn.execute(
            "INSERT INTO tca_parameters (symbol, time_bucket) VALUES ('SPY', 'morning')"
        )
        conn.execute(
            "INSERT INTO tca_parameters (symbol, time_bucket) VALUES ('SPY', 'midday')"
        )

    def test_composite_pk_rejects_duplicate(self, conn):
        conn.execute(
            "INSERT INTO tca_parameters (symbol, time_bucket) VALUES ('QQQ', 'close')"
        )
        _assert_constraint_violation(
            conn,
            "INSERT INTO tca_parameters (symbol, time_bucket) VALUES ('QQQ', 'close')",
        )

    def test_upsert_pattern(self, conn):
        conn.execute(
            "INSERT INTO tca_parameters (symbol, time_bucket, ewma_total_bps, sample_count) "
            "VALUES ('AAPL', 'morning', 5.0, 1) "
            "ON CONFLICT (symbol, time_bucket) DO UPDATE SET "
            "ewma_total_bps = EXCLUDED.ewma_total_bps, sample_count = EXCLUDED.sample_count"
        )
        conn.execute(
            "INSERT INTO tca_parameters (symbol, time_bucket, ewma_total_bps, sample_count) "
            "VALUES ('AAPL', 'morning', 4.5, 2) "
            "ON CONFLICT (symbol, time_bucket) DO UPDATE SET "
            "ewma_total_bps = EXCLUDED.ewma_total_bps, sample_count = EXCLUDED.sample_count"
        )
        row = conn.execute(
            "SELECT ewma_total_bps, sample_count FROM tca_parameters "
            "WHERE symbol = 'AAPL' AND time_bucket = 'morning'"
        ).fetchone()
        assert row[0] == pytest.approx(4.5)
        assert row[1] == 2


class TestDayTrades:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "day_trades")

    def test_date_index(self, conn):
        assert _index_exists(conn, "day_trades_date_idx")


class TestPendingWashLosses:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "pending_wash_losses")

    def test_resolved_defaults_false(self, conn):
        conn.execute(
            "INSERT INTO pending_wash_losses (symbol, loss_amount, sell_order_id, sell_date, window_end) "
            "VALUES ('SPY', 100.0, 'S1', '2026-01-01', '2026-01-31')"
        )
        row = conn.execute(
            "SELECT resolved FROM pending_wash_losses WHERE symbol = 'SPY'"
        ).fetchone()
        assert row[0] is False

    def test_index_exists(self, conn):
        assert _index_exists(conn, "pending_wash_symbol_idx")


class TestWashSaleFlags:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "wash_sale_flags")


class TestTaxLots:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "tax_lots")

    def test_status_defaults_open(self, conn):
        conn.execute(
            "INSERT INTO tax_lots (symbol, quantity, original_quantity, cost_basis, acquired_date, order_id) "
            "VALUES ('AAPL', 100, 100, 150.0, '2026-01-15', 'B1')"
        )
        row = conn.execute(
            "SELECT status FROM tax_lots WHERE order_id = 'B1'"
        ).fetchone()
        assert row[0] == "open"

    def test_wash_sale_adjustment_defaults_zero(self, conn):
        conn.execute(
            "INSERT INTO tax_lots (symbol, quantity, original_quantity, cost_basis, acquired_date, order_id) "
            "VALUES ('AAPL', 100, 100, 150.0, '2026-01-15', 'B2')"
        )
        row = conn.execute(
            "SELECT wash_sale_adjustment FROM tax_lots WHERE order_id = 'B2'"
        ).fetchone()
        assert row[0] == pytest.approx(0.0)

    def test_index_exists(self, conn):
        assert _index_exists(conn, "tax_lots_symbol_status_idx")


class TestAlgoParentOrders:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "algo_parent_orders")

    def test_defaults(self, conn):
        conn.execute(
            "INSERT INTO algo_parent_orders "
            "(parent_order_id, symbol, side, total_quantity, algo_type, "
            "start_time, end_time, arrival_price) "
            "VALUES ('P1', 'SPY', 'buy', 1000, 'twap', NOW(), NOW() + INTERVAL '30 min', 450.0)"
        )
        row = conn.execute(
            "SELECT status, max_participation_rate, filled_quantity, avg_fill_price "
            "FROM algo_parent_orders WHERE parent_order_id = 'P1'"
        ).fetchone()
        assert row[0] == "pending"
        assert row[1] == pytest.approx(0.02)
        assert row[2] == 0
        assert row[3] == pytest.approx(0.0)


class TestAlgoChildOrders:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "algo_child_orders")

    def test_fk_rejects_nonexistent_parent(self, conn):
        _assert_constraint_violation(
            conn,
            "INSERT INTO algo_child_orders "
            "(child_id, parent_id, scheduled_time, target_quantity) "
            "VALUES ('C1', 'NONEXISTENT', NOW(), 100)",
        )

    def test_index_exists(self, conn):
        assert _index_exists(conn, "algo_child_parent_idx")


class TestAlgoPerformance:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "algo_performance")


class TestExecutionAudit:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "execution_audit")

    def test_nbbo_fields_nullable(self, conn):
        conn.execute(
            "INSERT INTO execution_audit "
            "(order_id, fill_price, algo_selected, timestamp_ns) "
            "VALUES ('O1', 450.05, 'immediate', 1234567890)"
        )
        row = conn.execute(
            "SELECT nbbo_bid, nbbo_ask, nbbo_midpoint FROM execution_audit "
            "WHERE order_id = 'O1'"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

    def test_index_exists(self, conn):
        assert _index_exists(conn, "execution_audit_order_idx")


class TestSlippageAccuracy:
    def test_table_exists(self, conn):
        assert _table_exists(conn, "slippage_accuracy")

    def test_index_exists(self, conn):
        assert _index_exists(conn, "slippage_accuracy_symbol_idx")


class TestPositionsNewColumns:
    def test_margin_used_exists(self, conn):
        assert _column_exists(conn, "positions", "margin_used")

    def test_cumulative_funding_cost_exists(self, conn):
        assert _column_exists(conn, "positions", "cumulative_funding_cost")


class TestIdempotency:
    def test_running_migration_twice_succeeds(self, conn):
        _migrate_execution_layer_pg(conn)
        assert _table_exists(conn, "fill_legs")
        assert _table_exists(conn, "execution_audit")
