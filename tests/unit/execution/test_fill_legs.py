"""Tests for fill_legs recording and VWAP computation.

Verifies:
  - record_fill_leg creates rows with auto-incrementing leg_sequence
  - compute_fill_vwap returns correct VWAP for single and multiple legs
  - compute_fill_vwap raises ValueError for nonexistent orders
  - PaperBroker dual-write: single fill creates one fill_leg row
  - PaperBroker dual-write: two partial fills create two legs with correct VWAP
"""

import uuid

import pytest

from quantstack.db import db_conn, _migrate_execution_layer_pg
from quantstack.execution.fill_utils import compute_fill_vwap, record_fill_leg


@pytest.fixture()
def conn():
    """Yield a live DB connection inside a savepoint that rolls back on teardown."""
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


class TestRecordFillLeg:
    def test_creates_row_with_sequence_one(self, conn):
        oid = f"test-{uuid.uuid4()}"
        seq = record_fill_leg(conn, order_id=oid, quantity=100, price=50.0)
        assert seq == 1

        row = conn.execute(
            "SELECT order_id, leg_sequence, quantity, price, venue "
            "FROM fill_legs WHERE order_id = ?",
            [oid],
        ).fetchone()
        assert row[0] == oid
        assert row[1] == 1
        assert float(row[2]) == 100
        assert float(row[3]) == 50.0
        assert row[4] is None  # no venue specified

    def test_auto_increments_sequence(self, conn):
        oid = f"test-{uuid.uuid4()}"
        seq1 = record_fill_leg(conn, order_id=oid, quantity=100, price=50.0)
        seq2 = record_fill_leg(conn, order_id=oid, quantity=50, price=51.0)
        seq3 = record_fill_leg(conn, order_id=oid, quantity=25, price=52.0)

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    def test_different_orders_independent_sequences(self, conn):
        oid_a = f"test-a-{uuid.uuid4()}"
        oid_b = f"test-b-{uuid.uuid4()}"

        seq_a1 = record_fill_leg(conn, order_id=oid_a, quantity=100, price=50.0)
        seq_b1 = record_fill_leg(conn, order_id=oid_b, quantity=200, price=60.0)
        seq_a2 = record_fill_leg(conn, order_id=oid_a, quantity=50, price=51.0)

        assert seq_a1 == 1
        assert seq_b1 == 1
        assert seq_a2 == 2

    def test_venue_is_recorded(self, conn):
        oid = f"test-{uuid.uuid4()}"
        record_fill_leg(conn, order_id=oid, quantity=100, price=50.0, venue="alpaca")

        row = conn.execute(
            "SELECT venue FROM fill_legs WHERE order_id = ?", [oid]
        ).fetchone()
        assert row[0] == "alpaca"


class TestComputeFillVwap:
    def test_single_leg_returns_that_price(self, conn):
        oid = f"test-{uuid.uuid4()}"
        record_fill_leg(conn, order_id=oid, quantity=100, price=50.0)

        vwap = compute_fill_vwap(conn, oid)
        assert vwap == pytest.approx(50.0)

    def test_multiple_legs_correct_vwap(self, conn):
        oid = f"test-{uuid.uuid4()}"
        record_fill_leg(conn, order_id=oid, quantity=100, price=50.0)
        record_fill_leg(conn, order_id=oid, quantity=200, price=55.0)

        # VWAP = (100*50 + 200*55) / (100+200) = (5000 + 11000) / 300 = 53.333...
        vwap = compute_fill_vwap(conn, oid)
        assert vwap == pytest.approx(16000.0 / 300.0)

    def test_raises_for_nonexistent_order(self, conn):
        with pytest.raises(ValueError, match="No fill legs found"):
            compute_fill_vwap(conn, "nonexistent-order-id")


class TestPaperBrokerDualWrite:
    """Test that PaperBroker.execute() creates fill_legs rows."""

    def _make_broker(self, conn):
        from quantstack.execution.paper_broker import PaperBroker
        from quantstack.execution.portfolio_state import PortfolioState

        portfolio = PortfolioState(conn=conn)
        return PaperBroker(conn=conn, portfolio=portfolio)

    def test_single_fill_creates_one_leg(self, conn):
        from quantstack.execution.paper_broker import OrderRequest

        broker = self._make_broker(conn)
        req = OrderRequest(
            symbol="TEST",
            side="buy",
            quantity=10,
            order_type="market",
            current_price=100.0,
            daily_volume=10_000_000,
        )
        fill = broker.execute(req)
        assert not fill.rejected
        assert fill.filled_quantity > 0

        rows = conn.execute(
            "SELECT leg_sequence, quantity, price, venue "
            "FROM fill_legs WHERE order_id = ? ORDER BY leg_sequence",
            [fill.order_id],
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1  # leg_sequence
        assert float(rows[0][1]) == fill.filled_quantity
        assert rows[0][3] == "paper"

    def test_two_partial_fills_create_two_legs_with_correct_vwap(self, conn):
        from quantstack.execution.paper_broker import OrderRequest

        broker = self._make_broker(conn)

        # Use a large quantity relative to daily volume to trigger partial fill.
        # PaperBroker caps at 2% of daily_volume per order.
        # daily_volume=1000 → max_fill = max(1, int(1000 * 0.02)) = 20
        # So requesting 50 shares with daily_volume=1000 gives partial fill of 20.
        order_id = str(uuid.uuid4())

        req1 = OrderRequest(
            order_id=order_id,
            symbol="PARTL",
            side="buy",
            quantity=50,
            order_type="market",
            current_price=100.0,
            daily_volume=1000,
        )
        fill1 = broker.execute(req1)
        assert fill1.partial is True

        # Second fill for the same order (simulating continuation).
        req2 = OrderRequest(
            order_id=order_id,
            symbol="PARTL",
            side="buy",
            quantity=50,
            order_type="market",
            current_price=102.0,
            daily_volume=1000,
        )
        fill2 = broker.execute(req2)

        # Should have two fill_legs rows
        rows = conn.execute(
            "SELECT leg_sequence, quantity, price "
            "FROM fill_legs WHERE order_id = ? ORDER BY leg_sequence",
            [order_id],
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == 1
        assert rows[1][0] == 2

        # VWAP in fills summary should match compute_fill_vwap
        vwap = compute_fill_vwap(conn, order_id)
        summary = conn.execute(
            "SELECT fill_price FROM fills WHERE order_id = ?",
            [order_id],
        ).fetchone()
        assert float(summary[0]) == pytest.approx(vwap)

    def test_rejected_fill_creates_no_legs(self, conn):
        from quantstack.execution.paper_broker import OrderRequest

        broker = self._make_broker(conn)
        req = OrderRequest(
            symbol="TEST",
            side="buy",
            quantity=0,  # Will be rejected: quantity must be > 0
            order_type="market",
            current_price=100.0,
        )
        fill = broker.execute(req)
        assert fill.rejected

        rows = conn.execute(
            "SELECT COUNT(*) FROM fill_legs WHERE order_id = ?",
            [fill.order_id],
        ).fetchone()
        assert int(rows[0]) == 0
