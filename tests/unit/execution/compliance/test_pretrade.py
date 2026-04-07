"""Tests for pre-trade SEC compliance: PDT rule and Reg-T margin."""

from datetime import datetime, timedelta

import pytest

from quantstack.db import db_conn, _migrate_execution_layer_pg
from quantstack.execution.compliance.pretrade import (
    ComplianceResult,
    MarginCalculator,
    PDTChecker,
)
from quantstack.execution.compliance.calendar import trading_day_for


@pytest.fixture()
def conn():
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


def _make_order(symbol="AAPL", side="sell", quantity=100, price=150.0, ts=None):
    return {
        "order_id": "ORD-001",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "timestamp": ts or datetime(2026, 4, 6, 14, 30),  # Monday afternoon
    }


def _make_position(symbol="AAPL", side="buy", opened_at=None, order_id="OPEN-001"):
    return {
        "symbol": symbol,
        "side": side,
        "opened_at": opened_at or datetime(2026, 4, 6, 10, 0),  # Same day morning
        "order_id": order_id,
    }


# =========================================================================
# PDT Checker
# =========================================================================


class TestPDTZeroDayTrades:
    """0 existing day trades in window -> approved."""

    def test_approved_when_no_day_trades(self, conn):
        checker = PDTChecker()
        order = _make_order()
        positions = [_make_position()]

        result = checker.check(conn, order, account_equity=10_000.0, positions=positions)
        assert result.approved is True


class TestPDTTwoDayTrades:
    """2 existing day trades -> approved (under threshold)."""

    def test_approved_with_two_day_trades(self, conn):
        checker = PDTChecker()
        order = _make_order()
        positions = [_make_position()]
        trade_date = trading_day_for(order["timestamp"])

        # Insert 2 day trades in the current window.
        for i in range(2):
            conn.execute(
                "INSERT INTO day_trades "
                "(symbol, open_order_id, close_order_id, trade_date, quantity, account_equity) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ("AAPL", f"OPEN-{i}", f"CLOSE-{i}", trade_date, 100, 10_000.0),
            )

        result = checker.check(conn, order, account_equity=10_000.0, positions=positions)
        assert result.approved is True


class TestPDTThreeDayTradesBelowEquity:
    """3 day trades + account < $25K -> rejected."""

    def test_rejected_at_three_day_trades_under_25k(self, conn):
        checker = PDTChecker()
        order = _make_order()
        positions = [_make_position()]
        trade_date = trading_day_for(order["timestamp"])

        for i in range(3):
            conn.execute(
                "INSERT INTO day_trades "
                "(symbol, open_order_id, close_order_id, trade_date, quantity, account_equity) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ("AAPL", f"OPEN-{i}", f"CLOSE-{i}", trade_date, 100, 10_000.0),
            )

        result = checker.check(conn, order, account_equity=10_000.0, positions=positions)
        assert result.approved is False
        assert "PDT" in result.reason


class TestPDTThreeDayTradesAboveEquity:
    """3 day trades + account >= $25K -> approved."""

    def test_approved_at_three_day_trades_above_25k(self, conn):
        checker = PDTChecker()
        order = _make_order()
        positions = [_make_position()]
        trade_date = trading_day_for(order["timestamp"])

        for i in range(3):
            conn.execute(
                "INSERT INTO day_trades "
                "(symbol, open_order_id, close_order_id, trade_date, quantity, account_equity) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ("AAPL", f"OPEN-{i}", f"CLOSE-{i}", trade_date, 100, 10_000.0),
            )

        result = checker.check(conn, order, account_equity=25_000.0, positions=positions)
        assert result.approved is True


class TestPDTRecordDayTrade:
    """Same-day close is recorded as a day trade."""

    def test_records_day_trade_on_same_day_close(self, conn):
        checker = PDTChecker()
        fill = {
            "order_id": "CLOSE-001",
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 100,
            "price": 155.0,
            "timestamp": datetime(2026, 4, 6, 15, 0),
        }
        positions = [_make_position(opened_at=datetime(2026, 4, 6, 10, 0))]

        recorded = checker.record_if_day_trade(conn, fill, positions)
        assert recorded is True

        row = conn.execute(
            "SELECT symbol, close_order_id FROM day_trades "
            "WHERE close_order_id = 'CLOSE-001'"
        ).fetchone()
        assert row is not None
        assert row[0] == "AAPL"


class TestPDTYesterdayPositionNotDayTrade:
    """Position opened yesterday, closed today -> NOT a day trade."""

    def test_position_opened_yesterday_is_not_day_trade(self, conn):
        checker = PDTChecker()
        # Opened Friday April 3, closed Monday April 6.
        fill = {
            "order_id": "CLOSE-002",
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 100,
            "price": 155.0,
            "timestamp": datetime(2026, 4, 6, 15, 0),
        }
        positions = [_make_position(opened_at=datetime(2026, 4, 3, 10, 0))]

        recorded = checker.record_if_day_trade(conn, fill, positions)
        assert recorded is False


class TestPDTNotClosing:
    """A buy when we don't hold a short -> not closing, always approved."""

    def test_buy_with_no_position_is_not_day_trade(self, conn):
        checker = PDTChecker()
        order = _make_order(side="buy")
        positions = []  # No open position

        result = checker.check(conn, order, account_equity=5_000.0, positions=positions)
        assert result.approved is True


# =========================================================================
# Margin Calculator
# =========================================================================


class TestMarginLongEquity:
    """Long equity requires 50% Reg-T margin."""

    def test_approved_when_sufficient_margin(self, conn):
        calc = MarginCalculator()
        order = _make_order(side="buy", quantity=100, price=100.0)
        # Notional = $10,000, margin = $5,000. Equity = $6,000.
        result = calc.check(conn, order, positions=[], account_equity=6_000.0)
        assert result.approved is True

    def test_rejected_when_insufficient_margin(self, conn):
        calc = MarginCalculator()
        order = _make_order(side="buy", quantity=100, price=100.0)
        # Notional = $10,000, margin = $5,000. Equity = $4,000.
        result = calc.check(conn, order, positions=[], account_equity=4_000.0)
        assert result.approved is False
        assert "margin" in result.reason.lower()


class TestMarginLongOptions:
    """Long options: premium = margin requirement."""

    def test_options_premium_is_margin(self, conn):
        calc = MarginCalculator()
        order = {
            "symbol": "AAPL240419C150",
            "side": "buy",
            "quantity": 2,
            "price": 5.50,  # premium per share
            "instrument_type": "options",
        }
        # Margin = 2 contracts * $5.50 * 100 = $1,100
        result = calc.check(conn, order, positions=[], account_equity=1_500.0)
        assert result.approved is True

        result = calc.check(conn, order, positions=[], account_equity=1_000.0)
        assert result.approved is False


class TestMarginSellNoRequirement:
    """Sell/close orders require no additional margin."""

    def test_sell_always_approved(self, conn):
        calc = MarginCalculator()
        order = _make_order(side="sell", quantity=1000, price=500.0)
        result = calc.check(conn, order, positions=[], account_equity=1.0)
        assert result.approved is True
