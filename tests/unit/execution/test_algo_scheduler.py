"""Tests for the Algo Scheduler Core / EMS (section 07).

Covers dataclass defaults, parent/child state machines, parent-child invariant,
POV fallback, DB persistence, and crash recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from quantstack.db import db_conn, _migrate_execution_layer_pg
from quantstack.execution.algo_scheduler import (
    AlgoParentOrder,
    ChildOrder,
    transition_parent,
    transition_child,
    normalize_algo_type,
    update_parent_from_children,
    persist_parent,
    persist_child,
    persist_performance,
    startup_recovery,
    AlgoScheduler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 6, 14, 0, 0, tzinfo=timezone.utc)
_LATER = datetime(2026, 4, 6, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


def _make_parent(**overrides) -> AlgoParentOrder:
    defaults = dict(
        parent_order_id="P-TEST-001",
        symbol="SPY",
        side="buy",
        total_quantity=1000,
        algo_type="twap",
        start_time=_NOW,
        end_time=_LATER,
        arrival_price=450.0,
    )
    defaults.update(overrides)
    return AlgoParentOrder(**defaults)


def _make_child(parent_id: str = "P-TEST-001", seq: int = 1, **overrides) -> ChildOrder:
    defaults = dict(
        child_id=f"{parent_id}-C{seq}",
        parent_id=parent_id,
        scheduled_time=_NOW,
        target_quantity=100,
    )
    defaults.update(overrides)
    return ChildOrder(**defaults)


# ---------------------------------------------------------------------------
# Type / default tests
# ---------------------------------------------------------------------------


class TestAlgoParentOrderDefaults:
    def test_status_default(self):
        p = _make_parent()
        assert p.status == "pending"

    def test_filled_quantity_default(self):
        p = _make_parent()
        assert p.filled_quantity == 0

    def test_avg_fill_price_default(self):
        p = _make_parent()
        assert p.avg_fill_price == 0.0

    def test_max_participation_rate_default(self):
        p = _make_parent()
        assert p.max_participation_rate == 0.02

    def test_cancel_reason_default(self):
        p = _make_parent()
        assert p.cancel_reason is None


class TestChildOrderDefaults:
    def test_status_default(self):
        c = _make_child()
        assert c.status == "pending"

    def test_attempts_default(self):
        c = _make_child()
        assert c.attempts == 0

    def test_filled_quantity_default(self):
        c = _make_child()
        assert c.filled_quantity == 0

    def test_broker_order_id_default(self):
        c = _make_child()
        assert c.broker_order_id is None


# ---------------------------------------------------------------------------
# Parent state machine
# ---------------------------------------------------------------------------


class TestParentStateMachine:
    def test_pending_to_active(self):
        p = _make_parent()
        transition_parent(p, "active")
        assert p.status == "active"

    def test_active_to_completing(self):
        p = _make_parent(status="active")
        transition_parent(p, "completing")
        assert p.status == "completing"

    def test_completing_to_completed(self):
        p = _make_parent(status="completing")
        transition_parent(p, "completed")
        assert p.status == "completed"

    def test_active_to_cancelling(self):
        p = _make_parent(status="active")
        transition_parent(p, "cancelling")
        assert p.status == "cancelling"

    def test_cancelling_to_cancelled(self):
        p = _make_parent(status="cancelling")
        transition_parent(p, "cancelled")
        assert p.status == "cancelled"

    def test_invalid_pending_to_completed_raises(self):
        p = _make_parent()
        with pytest.raises(ValueError, match="Invalid parent transition"):
            transition_parent(p, "completed")

    def test_invalid_completed_to_active_raises(self):
        p = _make_parent(status="completed")
        with pytest.raises(ValueError, match="Invalid parent transition"):
            transition_parent(p, "active")

    def test_invalid_cancelled_to_active_raises(self):
        p = _make_parent(status="cancelled")
        with pytest.raises(ValueError, match="Invalid parent transition"):
            transition_parent(p, "active")

    def test_invalid_active_to_pending_raises(self):
        p = _make_parent(status="active")
        with pytest.raises(ValueError, match="Invalid parent transition"):
            transition_parent(p, "pending")


# ---------------------------------------------------------------------------
# Child state machine
# ---------------------------------------------------------------------------


class TestChildStateMachine:
    def test_pending_to_submitted(self):
        c = _make_child()
        transition_child(c, "submitted")
        assert c.status == "submitted"

    def test_submitted_to_filled(self):
        c = _make_child(status="submitted")
        transition_child(c, "filled")
        assert c.status == "filled"

    def test_submitted_to_partially_filled(self):
        c = _make_child(status="submitted")
        transition_child(c, "partially_filled")
        assert c.status == "partially_filled"

    def test_partially_filled_to_filled(self):
        c = _make_child(status="partially_filled")
        transition_child(c, "filled")
        assert c.status == "filled"

    def test_submitted_to_cancelled(self):
        c = _make_child(status="submitted")
        transition_child(c, "cancelled")
        assert c.status == "cancelled"

    def test_submitted_to_expired(self):
        c = _make_child(status="submitted")
        transition_child(c, "expired")
        assert c.status == "expired"

    def test_submitted_to_rejected(self):
        c = _make_child(status="submitted")
        transition_child(c, "rejected")
        assert c.status == "rejected"

    def test_invalid_pending_to_filled_raises(self):
        c = _make_child()
        with pytest.raises(ValueError, match="Invalid child transition"):
            transition_child(c, "filled")

    def test_invalid_filled_to_submitted_raises(self):
        c = _make_child(status="filled")
        with pytest.raises(ValueError, match="Invalid child transition"):
            transition_child(c, "submitted")


# ---------------------------------------------------------------------------
# Parent-child invariant
# ---------------------------------------------------------------------------


class TestParentChildInvariant:
    def test_three_children_fill_aggregation(self):
        parent = _make_parent()
        children = [
            _make_child(seq=1, filled_quantity=100, fill_price=450.0, status="filled"),
            _make_child(seq=2, filled_quantity=150, fill_price=451.0, status="filled"),
            _make_child(seq=3, filled_quantity=50, fill_price=449.0, status="filled"),
        ]
        update_parent_from_children(parent, children)

        assert parent.filled_quantity == 300
        # VWAP = (100*450 + 150*451 + 50*449) / 300
        expected_vwap = (100 * 450.0 + 150 * 451.0 + 50 * 449.0) / 300
        assert parent.avg_fill_price == pytest.approx(expected_vwap)

    def test_no_fills_yields_zero(self):
        parent = _make_parent()
        children = [
            _make_child(seq=1, status="pending"),
            _make_child(seq=2, status="submitted"),
        ]
        update_parent_from_children(parent, children)
        assert parent.filled_quantity == 0
        assert parent.avg_fill_price == 0.0

    def test_partial_fill_only_counts_filled_quantity(self):
        parent = _make_parent()
        children = [
            _make_child(seq=1, filled_quantity=100, fill_price=450.0, status="filled"),
            _make_child(seq=2, filled_quantity=0, fill_price=0.0, status="cancelled"),
        ]
        update_parent_from_children(parent, children)
        assert parent.filled_quantity == 100
        assert parent.avg_fill_price == pytest.approx(450.0)


# ---------------------------------------------------------------------------
# POV fallback
# ---------------------------------------------------------------------------


class TestPovFallback:
    def test_pov_dispatched_as_vwap(self):
        algo, rate = normalize_algo_type("pov", 0.03)
        assert algo == "vwap"

    def test_pov_caps_participation_at_5_percent(self):
        algo, rate = normalize_algo_type("pov", 0.10)
        assert rate == pytest.approx(0.05)

    def test_pov_preserves_lower_rate(self):
        algo, rate = normalize_algo_type("pov", 0.02)
        assert rate == pytest.approx(0.02)

    def test_twap_unchanged(self):
        algo, rate = normalize_algo_type("twap", 0.02)
        assert algo == "twap"
        assert rate == pytest.approx(0.02)

    def test_vwap_unchanged(self):
        algo, rate = normalize_algo_type("VWAP", 0.03)
        assert algo == "vwap"
        assert rate == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


class TestPersistParent:
    def test_insert_and_read(self, conn):
        parent = _make_parent()
        persist_parent(conn, parent)
        row = conn.execute(
            "SELECT parent_order_id, symbol, side, status, filled_quantity, avg_fill_price "
            "FROM algo_parent_orders WHERE parent_order_id = %s",
            (parent.parent_order_id,),
        ).fetchone()
        assert row[0] == "P-TEST-001"
        assert row[1] == "SPY"
        assert row[2] == "buy"
        assert row[3] == "pending"
        assert row[4] == 0
        assert row[5] == pytest.approx(0.0)

    def test_upsert_updates_status(self, conn):
        parent = _make_parent()
        persist_parent(conn, parent)
        parent.status = "active"
        parent.filled_quantity = 500
        parent.avg_fill_price = 450.5
        persist_parent(conn, parent)
        row = conn.execute(
            "SELECT status, filled_quantity, avg_fill_price "
            "FROM algo_parent_orders WHERE parent_order_id = %s",
            (parent.parent_order_id,),
        ).fetchone()
        assert row[0] == "active"
        assert row[1] == 500
        assert row[2] == pytest.approx(450.5)


class TestPersistChild:
    def test_insert_and_read(self, conn):
        # Parent must exist first (FK constraint)
        parent = _make_parent()
        persist_parent(conn, parent)

        child = _make_child()
        persist_child(conn, child)
        row = conn.execute(
            "SELECT child_id, parent_id, status, target_quantity, filled_quantity "
            "FROM algo_child_orders WHERE child_id = %s",
            (child.child_id,),
        ).fetchone()
        assert row[0] == "P-TEST-001-C1"
        assert row[1] == "P-TEST-001"
        assert row[2] == "pending"
        assert row[3] == 100
        assert row[4] == 0

    def test_upsert_updates_fill(self, conn):
        parent = _make_parent()
        persist_parent(conn, parent)
        child = _make_child()
        persist_child(conn, child)

        child.status = "filled"
        child.filled_quantity = 100
        child.fill_price = 450.25
        persist_child(conn, child)

        row = conn.execute(
            "SELECT status, filled_quantity, fill_price "
            "FROM algo_child_orders WHERE child_id = %s",
            (child.child_id,),
        ).fetchone()
        assert row[0] == "filled"
        assert row[1] == 100
        assert row[2] == pytest.approx(450.25)


class TestPersistPerformance:
    def test_writes_after_completion(self, conn):
        parent = _make_parent(
            status="completed",
            filled_quantity=1000,
            avg_fill_price=450.10,
        )
        persist_parent(conn, parent)

        children = [
            _make_child(seq=1, filled_quantity=500, fill_price=450.0, status="filled"),
            _make_child(seq=2, filled_quantity=500, fill_price=450.2, status="filled"),
        ]
        for c in children:
            persist_child(conn, c)

        update_parent_from_children(parent, children)
        persist_performance(conn, parent, children)

        row = conn.execute(
            "SELECT parent_order_id, symbol, filled_qty, num_children, "
            "num_children_filled, implementation_shortfall_bps "
            "FROM algo_performance WHERE parent_order_id = %s",
            (parent.parent_order_id,),
        ).fetchone()
        assert row[0] == "P-TEST-001"
        assert row[1] == "SPY"
        assert row[2] == 1000
        assert row[3] == 2
        assert row[4] == 2
        # IS should be small since fills are near arrival
        assert isinstance(row[5], float)


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


class TestStartupRecovery:
    def test_cancels_active_parents(self, conn):
        parent = _make_parent(status="active")
        persist_parent(conn, parent)
        child = _make_child(status="submitted")
        persist_child(conn, child)

        recovered = startup_recovery(conn)
        assert recovered == 1

        row = conn.execute(
            "SELECT status FROM algo_parent_orders WHERE parent_order_id = %s",
            (parent.parent_order_id,),
        ).fetchone()
        assert row[0] == "cancelled"

    def test_cancels_non_terminal_children(self, conn):
        parent = _make_parent(status="active")
        persist_parent(conn, parent)
        c1 = _make_child(seq=1, status="submitted")
        c2 = _make_child(seq=2, status="filled")
        persist_child(conn, c1)
        persist_child(conn, c2)

        startup_recovery(conn)

        r1 = conn.execute(
            "SELECT status FROM algo_child_orders WHERE child_id = %s",
            (c1.child_id,),
        ).fetchone()
        r2 = conn.execute(
            "SELECT status FROM algo_child_orders WHERE child_id = %s",
            (c2.child_id,),
        ).fetchone()
        assert r1[0] == "cancelled"
        assert r2[0] == "filled"  # already terminal — untouched

    def test_no_active_parents_returns_zero(self, conn):
        # Only pending parents — should not be recovered
        parent = _make_parent(status="pending")
        persist_parent(conn, parent)
        assert startup_recovery(conn) == 0

    def test_recovering_completing_parent(self, conn):
        parent = _make_parent(status="completing")
        persist_parent(conn, parent)
        child = _make_child(status="submitted")
        persist_child(conn, child)

        recovered = startup_recovery(conn)
        assert recovered == 1
        row = conn.execute(
            "SELECT status FROM algo_parent_orders WHERE parent_order_id = %s",
            (parent.parent_order_id,),
        ).fetchone()
        assert row[0] == "cancelled"


# ---------------------------------------------------------------------------
# AlgoScheduler async tests
# ---------------------------------------------------------------------------


class _FakeBroker:
    """Minimal broker stub for scheduler tests."""

    def __init__(self):
        self.cancelled: list[str] = []

    async def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)


class TestAlgoScheduler:
    @pytest.mark.asyncio
    async def test_submit_parent_persists(self, conn):
        broker = _FakeBroker()
        scheduler = AlgoScheduler(broker, conn)
        parent = _make_parent()
        children = [_make_child(seq=1), _make_child(seq=2)]

        await scheduler.submit_parent(parent, children)

        row = conn.execute(
            "SELECT status FROM algo_parent_orders WHERE parent_order_id = %s",
            (parent.parent_order_id,),
        ).fetchone()
        assert row[0] == "pending"
        assert parent.parent_order_id in scheduler._active_parents

    @pytest.mark.asyncio
    async def test_submit_pov_normalizes_to_vwap(self, conn):
        broker = _FakeBroker()
        scheduler = AlgoScheduler(broker, conn)
        parent = _make_parent(algo_type="pov", max_participation_rate=0.10)
        children = [_make_child(seq=1)]

        await scheduler.submit_parent(parent, children)

        assert parent.algo_type == "vwap"
        assert parent.max_participation_rate == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_cancel_parent(self, conn):
        broker = _FakeBroker()
        scheduler = AlgoScheduler(broker, conn)
        parent = _make_parent(status="active")
        c1 = _make_child(seq=1, status="submitted", broker_order_id="BRK-1")
        c2 = _make_child(seq=2, status="filled")

        # Manually register (bypassing submit since parent is already active)
        scheduler._active_parents[parent.parent_order_id] = parent
        scheduler._children[parent.parent_order_id] = [c1, c2]
        persist_parent(conn, parent)
        persist_child(conn, c1)
        persist_child(conn, c2)

        await scheduler.cancel_parent(parent.parent_order_id, "test_cancel")

        assert parent.status == "cancelled"
        assert parent.cancel_reason == "test_cancel"
        assert c1.status == "cancelled"
        assert "BRK-1" in broker.cancelled
        # Parent should be removed from active set
        assert parent.parent_order_id not in scheduler._active_parents

    @pytest.mark.asyncio
    async def test_cancel_all(self, conn):
        broker = _FakeBroker()
        scheduler = AlgoScheduler(broker, conn)

        p1 = _make_parent(parent_order_id="P-1", status="active")
        p2 = _make_parent(parent_order_id="P-2", status="active")
        c1 = _make_child(parent_id="P-1", seq=1, status="submitted")
        c2 = _make_child(parent_id="P-2", seq=1, status="submitted")

        for p in (p1, p2):
            persist_parent(conn, p)
            scheduler._active_parents[p.parent_order_id] = p
        for c, pid in ((c1, "P-1"), (c2, "P-2")):
            persist_child(conn, c)
            scheduler._children[pid] = [c]

        await scheduler.cancel_all("eod_shutdown")

        assert len(scheduler._active_parents) == 0
        assert p1.status == "cancelled"
        assert p2.status == "cancelled"

    @pytest.mark.asyncio
    async def test_run_placeholder_returns(self, conn):
        broker = _FakeBroker()
        scheduler = AlgoScheduler(broker, conn)
        # run() should return without blocking
        await scheduler.run()

    @pytest.mark.asyncio
    async def test_startup_recovery_via_scheduler(self, conn):
        parent = _make_parent(status="active")
        persist_parent(conn, parent)
        child = _make_child(status="submitted")
        persist_child(conn, child)

        broker = _FakeBroker()
        scheduler = AlgoScheduler(broker, conn)
        recovered = await scheduler.startup_recovery()
        assert recovered == 1
