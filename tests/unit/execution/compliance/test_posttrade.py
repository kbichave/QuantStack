"""Tests for post-trade SEC compliance: Wash Sale tracking and Tax Lot management."""

from datetime import date, datetime, timedelta

import pytest

from quantstack.db import db_conn, _migrate_execution_layer_pg
from quantstack.execution.compliance.posttrade import TaxLotManager, WashSaleTracker


@pytest.fixture()
def conn():
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


def _sell_fill(symbol="AAPL", price=140.0, quantity=100, realized_pnl=-500.0, ts=None):
    return {
        "order_id": "SELL-001",
        "symbol": symbol,
        "side": "sell",
        "quantity": quantity,
        "price": price,
        "timestamp": ts or datetime(2026, 4, 1, 14, 0),
        "realized_pnl": realized_pnl,
    }


def _buy_fill(symbol="AAPL", price=142.0, quantity=100, order_id="BUY-002", ts=None):
    return {
        "order_id": order_id,
        "symbol": symbol,
        "side": "buy",
        "quantity": quantity,
        "price": price,
        "timestamp": ts or datetime(2026, 4, 10, 10, 0),
    }


# =========================================================================
# Wash Sale Tracker
# =========================================================================


class TestWashSaleSellAtLoss:
    """Selling at a loss creates a pending_wash_losses record."""

    def test_creates_pending_record(self, conn):
        tracker = WashSaleTracker()
        result = tracker.on_fill(conn, _sell_fill(realized_pnl=-500.0))

        assert result["action"] == "pending_loss_created"
        assert result["loss_amount"] == 500.0

        row = conn.execute(
            "SELECT symbol, loss_amount, resolved FROM pending_wash_losses "
            "WHERE sell_order_id = 'SELL-001'"
        ).fetchone()
        assert row is not None
        assert row[0] == "AAPL"
        assert row[1] == pytest.approx(500.0)
        assert row[2] is False

    def test_sell_at_profit_creates_no_record(self, conn):
        tracker = WashSaleTracker()
        result = tracker.on_fill(conn, _sell_fill(realized_pnl=200.0))
        assert result["action"] == "none"


class TestWashSaleBuyWithin30Days:
    """Buying the same symbol within 30 days of a loss flags a wash sale."""

    def test_flags_wash_sale(self, conn):
        tracker = WashSaleTracker()

        # Phase 1: sell at loss on April 1.
        tracker.on_fill(conn, _sell_fill(
            realized_pnl=-500.0,
            ts=datetime(2026, 4, 1, 14, 0),
        ))

        # Phase 2: buy on April 10 (within 30 days).
        result = tracker.on_fill(conn, _buy_fill(
            price=142.0,
            quantity=100,
            order_id="BUY-002",
            ts=datetime(2026, 4, 10, 10, 0),
        ))

        assert result["action"] == "wash_sale_flagged"
        assert result["disallowed_loss"] == pytest.approx(500.0)
        # Adjusted basis = buy price + disallowed loss / qty = 142 + 5 = 147
        assert result["adjusted_basis"] == pytest.approx(147.0)

        # Verify the pending record is resolved.
        row = conn.execute(
            "SELECT resolved, resolved_by_order_id FROM pending_wash_losses "
            "WHERE sell_order_id = 'SELL-001'"
        ).fetchone()
        assert row[0] is True
        assert row[1] == "BUY-002"

        # Verify a wash_sale_flags record exists.
        flag = conn.execute(
            "SELECT disallowed_loss, adjusted_cost_basis FROM wash_sale_flags "
            "WHERE replacement_order_id = 'BUY-002'"
        ).fetchone()
        assert flag is not None
        assert flag[0] == pytest.approx(500.0)
        assert flag[1] == pytest.approx(147.0)


class TestWashSaleBuyAfter30Days:
    """Buying after 30 days does NOT flag a wash sale."""

    def test_no_flag_after_window(self, conn):
        tracker = WashSaleTracker()

        # Sell at loss on April 1.
        tracker.on_fill(conn, _sell_fill(
            realized_pnl=-500.0,
            ts=datetime(2026, 4, 1, 14, 0),
        ))

        # Buy on May 5 (>30 calendar days after April 1).
        result = tracker.on_fill(conn, _buy_fill(
            ts=datetime(2026, 5, 5, 10, 0),
            order_id="BUY-LATE",
        ))

        assert result["action"] == "none"


class TestWashSaleCostBasisAdjustment:
    """Wash sale adjusts cost basis on replacement tax lot."""

    def test_tax_lot_basis_adjusted(self, conn):
        tracker = WashSaleTracker()
        lot_mgr = TaxLotManager()

        # Sell at loss.
        tracker.on_fill(conn, _sell_fill(
            realized_pnl=-300.0,
            ts=datetime(2026, 4, 1, 14, 0),
        ))

        # Buy replacement shares — create tax lot first, then process wash sale.
        buy = _buy_fill(price=50.0, quantity=100, order_id="BUY-WASH", ts=datetime(2026, 4, 5, 10, 0))
        lot_mgr.on_fill(conn, buy)

        # Now process the wash sale (which adjusts the lot).
        result = tracker.on_fill(conn, buy)

        assert result["action"] == "wash_sale_flagged"

        # Verify tax lot cost basis was adjusted: 50 + 300/100 = 53.
        row = conn.execute(
            "SELECT cost_basis, wash_sale_adjustment FROM tax_lots "
            "WHERE order_id = 'BUY-WASH' AND status = 'open'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(53.0)
        assert row[1] == pytest.approx(300.0)


class TestWashSaleCheckPending:
    """check_pending returns a warning string when there's an active window."""

    def test_warns_on_active_window(self, conn):
        tracker = WashSaleTracker()
        tracker.on_fill(conn, _sell_fill(
            realized_pnl=-200.0,
            ts=datetime(2026, 4, 1, 14, 0),
        ))
        warning = tracker.check_pending(conn, "AAPL")
        assert warning is not None
        assert "wash sale" in warning.lower()

    def test_no_warning_for_different_symbol(self, conn):
        tracker = WashSaleTracker()
        tracker.on_fill(conn, _sell_fill(
            realized_pnl=-200.0,
            ts=datetime(2026, 4, 1, 14, 0),
        ))
        warning = tracker.check_pending(conn, "MSFT")
        assert warning is None


# =========================================================================
# Tax Lot Manager
# =========================================================================


class TestTaxLotBuyCreatesLot:
    """A buy fill creates a new tax lot."""

    def test_lot_created(self, conn):
        mgr = TaxLotManager()
        buy = _buy_fill(price=150.0, quantity=50, order_id="BUY-LOT1")
        result = mgr.on_fill(conn, buy)

        assert result["action"] == "lot_created"
        assert result["lot_id"] is not None

        row = conn.execute(
            "SELECT symbol, quantity, original_quantity, cost_basis, status "
            "FROM tax_lots WHERE order_id = 'BUY-LOT1'"
        ).fetchone()
        assert row[0] == "AAPL"
        assert row[1] == 50
        assert row[2] == 50
        assert row[3] == pytest.approx(150.0)
        assert row[4] == "open"


class TestTaxLotSellFIFO:
    """Sell matches oldest lots first (FIFO)."""

    def test_fifo_order(self, conn):
        mgr = TaxLotManager()

        # Create two lots: older at $100, newer at $120.
        mgr.on_fill(conn, _buy_fill(
            price=100.0, quantity=50, order_id="LOT-OLD",
            ts=datetime(2026, 3, 1, 10, 0),
        ))
        mgr.on_fill(conn, _buy_fill(
            price=120.0, quantity=50, order_id="LOT-NEW",
            ts=datetime(2026, 3, 15, 10, 0),
        ))

        # Sell 50 shares at $130 — should consume the older lot.
        sell = {
            "order_id": "SELL-FIFO",
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 50,
            "price": 130.0,
            "timestamp": datetime(2026, 4, 1, 14, 0),
        }
        result = mgr.on_fill(conn, sell)

        assert result["action"] == "lots_consumed"
        # P&L = 50 * (130 - 100) = 1500
        assert result["realized_pnl"] == pytest.approx(1500.0)
        assert result["lots_closed"] == 1

        # Older lot should be closed.
        old_lot = conn.execute(
            "SELECT status, realized_pnl FROM tax_lots WHERE order_id = 'LOT-OLD'"
        ).fetchone()
        assert old_lot[0] == "closed"
        assert old_lot[1] == pytest.approx(1500.0)

        # Newer lot should still be open with full quantity.
        new_lot = conn.execute(
            "SELECT status, quantity FROM tax_lots WHERE order_id = 'LOT-NEW'"
        ).fetchone()
        assert new_lot[0] == "open"
        assert new_lot[1] == 50


class TestTaxLotPartialConsumption:
    """Sell consumes part of a lot, leaving the remainder open."""

    def test_partial_lot(self, conn):
        mgr = TaxLotManager()

        mgr.on_fill(conn, _buy_fill(
            price=100.0, quantity=100, order_id="LOT-PART",
            ts=datetime(2026, 3, 1, 10, 0),
        ))

        sell = {
            "order_id": "SELL-PARTIAL",
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 30,
            "price": 110.0,
            "timestamp": datetime(2026, 4, 1, 14, 0),
        }
        result = mgr.on_fill(conn, sell)

        assert result["action"] == "lots_consumed"
        # P&L = 30 * (110 - 100) = 300
        assert result["realized_pnl"] == pytest.approx(300.0)
        assert result["lots_closed"] == 0  # Not fully consumed

        row = conn.execute(
            "SELECT quantity, status FROM tax_lots WHERE order_id = 'LOT-PART'"
        ).fetchone()
        assert row[0] == 70  # 100 - 30
        assert row[1] == "open"


class TestTaxLotRealizedPnL:
    """Realized P&L is computed correctly across multiple lots."""

    def test_multi_lot_pnl(self, conn):
        mgr = TaxLotManager()

        # Two lots: 50 @ $100, 50 @ $120.
        mgr.on_fill(conn, _buy_fill(
            price=100.0, quantity=50, order_id="PNL-1",
            ts=datetime(2026, 3, 1, 10, 0),
        ))
        mgr.on_fill(conn, _buy_fill(
            price=120.0, quantity=50, order_id="PNL-2",
            ts=datetime(2026, 3, 15, 10, 0),
        ))

        # Sell all 100 at $110.
        sell = {
            "order_id": "SELL-ALL",
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 100,
            "price": 110.0,
            "timestamp": datetime(2026, 4, 1, 14, 0),
        }
        result = mgr.on_fill(conn, sell)

        # Lot 1: 50 * (110 - 100) = 500
        # Lot 2: 50 * (110 - 120) = -500
        # Total = 0
        assert result["realized_pnl"] == pytest.approx(0.0)
        assert result["lots_closed"] == 2


class TestTaxLotSellMoreThanOpen:
    """Selling more shares than open lots handles gracefully."""

    def test_graceful_oversell(self, conn):
        mgr = TaxLotManager()

        mgr.on_fill(conn, _buy_fill(
            price=100.0, quantity=30, order_id="SMALL-LOT",
            ts=datetime(2026, 3, 1, 10, 0),
        ))

        sell = {
            "order_id": "SELL-OVER",
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 50,
            "price": 110.0,
            "timestamp": datetime(2026, 4, 1, 14, 0),
        }
        result = mgr.on_fill(conn, sell)

        assert result["action"] == "lots_consumed"
        # Only 30 shares matched against the lot.
        # P&L = 30 * (110 - 100) = 300
        assert result["realized_pnl"] == pytest.approx(300.0)
        assert result["lots_closed"] == 1
        # 20 shares remain unmatched.
        assert result["unfilled_quantity"] == 20
