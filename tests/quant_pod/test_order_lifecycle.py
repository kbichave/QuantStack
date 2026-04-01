# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quantstack.execution.order_lifecycle — Sprint 3.

Tests the OMS state machine: submit, acknowledge, partial fill, fill,
reject, cancel, exec algo selection, and compliance checks.
All tests use a PostgreSQL connection via pg_conn().
"""

from __future__ import annotations

import pytest
from quantstack.db import pg_conn
from quantstack.execution.order_lifecycle import (
    ExecAlgoOMS,
    OrderLifecycle,
    OrderStatus,
)


@pytest.fixture
def oms() -> OrderLifecycle:
    with pg_conn() as conn:
        conn.execute("DELETE FROM orders")
        oms_instance = OrderLifecycle(conn)
        yield oms_instance
        conn.execute("ROLLBACK")


def _submit(
    oms: OrderLifecycle, symbol="SPY", side="buy", quantity=100, adv=80_000_000
):
    return oms.submit(
        symbol=symbol,
        side=side,
        quantity=quantity,
        arrival_price=450.0,
        signal_id="test-signal",
        adv=adv,
    )


# ---------------------------------------------------------------------------
# submit → SUBMITTED
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_returns_submitted_order(self, oms):
        order = _submit(oms)
        assert order.status == OrderStatus.SUBMITTED

    def test_order_has_uuid(self, oms):
        order = _submit(oms)
        assert len(order.order_id) > 10

    def test_order_arrival_price_set(self, oms):
        order = _submit(oms)
        assert order.arrival_price == 450.0

    def test_small_order_gets_immediate_algo(self, oms):
        """< 0.2% of ADV → IMMEDIATE."""
        order = oms.submit(
            symbol="SPY",
            side="buy",
            quantity=100,
            arrival_price=450.0,
            adv=80_000_000,  # 100/80M = 0.000125%
        )
        assert order.exec_algo == ExecAlgoOMS.IMMEDIATE

    def test_medium_order_gets_twap_or_vwap(self, oms):
        """~0.5% of ADV → TWAP."""
        order = oms.submit(
            symbol="SPY",
            side="buy",
            quantity=400_000,
            arrival_price=450.0,
            adv=80_000_000,  # 400k/80M = 0.5%
        )
        assert order.exec_algo in (ExecAlgoOMS.TWAP, ExecAlgoOMS.VWAP)

    def test_large_order_gets_vwap_or_pov(self, oms):
        """~3% of ADV → VWAP or POV."""
        order = oms.submit(
            symbol="SPY",
            side="buy",
            quantity=2_400_000,
            arrival_price=450.0,
            adv=80_000_000,
        )
        assert order.exec_algo in (ExecAlgoOMS.VWAP, ExecAlgoOMS.POV)

    def test_duplicate_order_rejected(self, oms):
        """Same symbol+side within 60s → REJECTED by compliance."""
        _submit(oms, symbol="SPY", side="buy")
        order2 = _submit(oms, symbol="SPY", side="buy")
        assert order2.status == OrderStatus.REJECTED

    def test_different_side_not_duplicate(self, oms):
        """BUY then SELL same symbol → not a duplicate."""
        _submit(oms, symbol="SPY", side="buy")
        order2 = _submit(oms, symbol="SPY", side="sell")
        assert order2.status == OrderStatus.SUBMITTED


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_acknowledge_transition(self, oms):
        order = _submit(oms)
        acked = oms.acknowledge(order.order_id)
        assert acked.status == OrderStatus.ACKNOWLEDGED

    def test_record_fill_transitions_to_filled(self, oms):
        order = _submit(oms)
        filled = oms.record_fill(order.order_id, fill_price=450.9)
        assert filled.status == OrderStatus.FILLED
        assert filled.fill_price == 450.9

    def test_partial_fill_then_full_fill(self, oms):
        order = _submit(oms, quantity=200)
        oms.record_partial_fill(order.order_id, filled_quantity=100, fill_price=450.5)
        partial = oms._orders[order.order_id]
        assert partial.status == OrderStatus.PARTIALLY_FILLED
        assert partial.filled_quantity == 100

        filled = oms.record_fill(order.order_id, fill_price=450.9)
        assert filled.status == OrderStatus.FILLED

    def test_reject_order(self, oms):
        order = _submit(oms)
        rejected = oms.reject(order.order_id, reason="Broker refused")
        assert rejected.status == OrderStatus.REJECTED
        assert "Broker refused" in rejected.rejection_reason

    def test_cancel_order(self, oms):
        order = _submit(oms)
        cancelled = oms.cancel(order.order_id)
        assert cancelled.status == OrderStatus.CANCELLED

    def test_cannot_fill_rejected_order(self, oms):
        order = _submit(oms)
        oms.reject(order.order_id, reason="test")
        with pytest.raises((ValueError, KeyError)):
            oms.record_fill(order.order_id, fill_price=450.0)


# ---------------------------------------------------------------------------
# implementation_shortfall_bps
# ---------------------------------------------------------------------------


class TestImplementationShortfall:
    def test_buy_above_arrival_positive_shortfall(self, oms):
        order = _submit(oms, side="buy")  # arrival=450.0
        filled = oms.record_fill(order.order_id, fill_price=451.0)
        sf = filled.implementation_shortfall_bps
        assert sf is not None
        assert sf > 0  # Buy above arrival = cost

    def test_buy_below_arrival_negative_shortfall(self, oms):
        order = _submit(oms, side="buy")  # arrival=450.0
        filled = oms.record_fill(order.order_id, fill_price=449.0)
        assert filled.implementation_shortfall_bps < 0  # Price improvement

    def test_sell_below_arrival_positive_shortfall(self, oms):
        order = _submit(oms, side="sell")  # arrival=450.0
        filled = oms.record_fill(order.order_id, fill_price=449.0)
        assert filled.implementation_shortfall_bps > 0  # Cost


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_session_summary_after_fills(self, oms):
        for i in range(3):
            o = oms.submit(
                symbol=f"SYM{i}",
                side="buy",
                quantity=100,
                arrival_price=100.0,
                signal_id=f"sig{i}",
                adv=10_000_000,
            )
            oms.record_fill(o.order_id, fill_price=100.5)

        summary = oms.session_summary()
        assert summary["order_count"] == 3
        assert summary["fill_rate"] == pytest.approx(1.0, abs=0.01)

    def test_session_summary_empty(self, oms):
        summary = oms.session_summary()
        assert summary["order_count"] == 0
