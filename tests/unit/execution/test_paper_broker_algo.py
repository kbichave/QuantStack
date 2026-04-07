"""Tests for PaperBroker algo child-order execution (TWAP/VWAP fills).

Verifies:
  - TWAP child orders fill against historical bar at scheduled_time
  - Fill price uses bar VWAP + directional noise, NOT flat spread model
  - Participation rate caps partial fills when child_qty > bar_volume * rate
  - Fill price always within bar [low, high] range (fuzz test, 50 seeds)
  - Buy fills above bar VWAP (adverse), sell fills below bar VWAP (adverse)
  - IMMEDIATE orders still use existing instant-fill model (unchanged)
  - Missing historical bar falls back to instant-fill with warning
  - Configurable participation_rate
  - Zero bar volume rejects child order
  - Fill records a fill_leg via dual-write
  - Partial fill from participation cap leaves remainder
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from quantstack.db import _migrate_execution_layer_pg, db_conn
from quantstack.execution.paper_broker import (
    BarData,
    Fill,
    OrderRequest,
    PaperBroker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_schema_migrated = False


@pytest.fixture()
def db_connection():
    """Yield a live DB connection inside a savepoint that rolls back on teardown.

    Migrations are run once (module-level flag) on the first connection, then
    that connection is released before being re-acquired for the savepoint
    pattern.  This avoids deadlocks between DDL locks and row-level locks.
    """
    global _schema_migrated
    if not _schema_migrated:
        with db_conn() as c:
            _migrate_execution_layer_pg(c)
        _schema_migrated = True

    with db_conn() as c:
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        try:
            c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")
        except Exception:
            # Savepoint may have been consumed by an error; just rollback.
            c._raw.rollback()


@pytest.fixture()
def broker(db_connection):
    """Broker backed by a real DB connection (for dual-write tests)."""
    return PaperBroker(conn=db_connection)


@pytest.fixture()
def mock_broker():
    """Broker with _record_fill and _update_portfolio stubbed out.

    Most algo-child tests only care about the fill price/quantity logic and
    don't need a real database.
    """
    with db_conn() as c:
        b = PaperBroker(conn=c)
        b._record_fill = MagicMock()  # type: ignore[method-assign]
        b._update_portfolio = MagicMock()  # type: ignore[method-assign]
        yield b


@pytest.fixture()
def sample_bar():
    """A realistic 5-minute bar for SPY."""
    return BarData(
        open=450.10,
        high=450.80,
        low=449.50,
        close=450.60,
        volume=50_000,
        vwap=450.30,
    )


@pytest.fixture()
def base_request():
    return OrderRequest(
        symbol="SPY",
        side="buy",
        quantity=100,
        order_type="market",
        current_price=450.30,
        daily_volume=80_000_000,
    )


# ---------------------------------------------------------------------------
# Test: TWAP child filled against bar at scheduled_time
# ---------------------------------------------------------------------------

class TestAlgoChildFillAgainstBar:
    def test_fill_uses_bar_vwap_as_anchor(self, mock_broker, base_request, sample_bar):
        """Fill price should be based on bar VWAP, not the flat spread model."""
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(base_request, scheduled_time=scheduled)

        assert not fill.rejected
        assert fill.filled_quantity == 100
        assert fill.fill_price >= sample_bar.low
        assert fill.fill_price <= sample_bar.high

    def test_buy_fills_above_vwap(self, mock_broker, sample_bar):
        """Buys get adverse fill: price >= bar VWAP."""
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=50,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(req, scheduled_time=scheduled)

        assert fill.fill_price >= sample_bar.vwap

    def test_sell_fills_below_vwap(self, mock_broker, sample_bar):
        """Sells get adverse fill: price <= bar VWAP."""
        req = OrderRequest(
            symbol="SPY", side="sell", quantity=50,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(req, scheduled_time=scheduled)

        assert fill.fill_price <= sample_bar.vwap


# ---------------------------------------------------------------------------
# Test: Participation rate caps
# ---------------------------------------------------------------------------

class TestParticipationRate:
    def test_partial_fill_when_qty_exceeds_participation_cap(
        self, mock_broker, sample_bar,
    ):
        """child_qty > bar_volume * participation_rate -> partial fill."""
        # bar volume=50000, participation_rate=0.02 -> max_fillable=1000
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=5000,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(
                req, scheduled_time=scheduled, participation_rate=0.02,
            )

        assert fill.partial is True
        assert fill.filled_quantity == 1000  # 50000 * 0.02
        assert fill.requested_quantity == 5000

    def test_remainder_left_after_participation_cap(self, mock_broker, sample_bar):
        """The unfilled remainder is requested - filled."""
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=2000,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(
                req, scheduled_time=scheduled, participation_rate=0.02,
            )

        remainder = fill.requested_quantity - fill.filled_quantity
        assert remainder == 1000  # 2000 - 1000

    def test_configurable_participation_rate(self, mock_broker, sample_bar):
        """Different participation_rate yields different max fill."""
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=10_000,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill_low = mock_broker.execute_algo_child(
                req, scheduled_time=scheduled, participation_rate=0.01,
            )
            fill_high = mock_broker.execute_algo_child(
                req, scheduled_time=scheduled, participation_rate=0.10,
            )

        # 0.01 * 50000 = 500, 0.10 * 50000 = 5000
        assert fill_low.filled_quantity == 500
        assert fill_high.filled_quantity == 5000

    def test_full_fill_when_under_participation_cap(self, mock_broker, sample_bar):
        """Quantity under the cap fills completely."""
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=100,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(
                req, scheduled_time=scheduled, participation_rate=0.02,
            )

        assert fill.filled_quantity == 100
        assert fill.partial is False


# ---------------------------------------------------------------------------
# Test: Fill price within bar range (fuzz test)
# ---------------------------------------------------------------------------

class TestFillPriceRange:
    @pytest.mark.parametrize("seed", range(50))
    def test_fill_price_within_bar_range(self, mock_broker, sample_bar, seed):
        """Fill price must be clamped to [bar.low, bar.high] across 50 random seeds."""
        import random
        random.seed(seed)

        side = "buy" if seed % 2 == 0 else "sell"
        req = OrderRequest(
            symbol="SPY", side=side, quantity=100,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=sample_bar):
            fill = mock_broker.execute_algo_child(req, scheduled_time=scheduled)

        assert fill.fill_price >= sample_bar.low, (
            f"seed={seed}: fill {fill.fill_price} < bar low {sample_bar.low}"
        )
        assert fill.fill_price <= sample_bar.high, (
            f"seed={seed}: fill {fill.fill_price} > bar high {sample_bar.high}"
        )


# ---------------------------------------------------------------------------
# Test: IMMEDIATE orders unchanged
# ---------------------------------------------------------------------------

class TestImmediateOrdersUnchanged:
    def test_execute_still_uses_instant_fill_model(self, broker):
        """The existing execute() path is NOT affected by algo child changes."""
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=100,
            order_type="market", current_price=450.00,
            daily_volume=80_000_000,
        )
        fill = broker.execute(req)

        assert not fill.rejected
        assert fill.filled_quantity == 100
        # Instant-fill uses spread + impact model, price should be close to ref
        assert abs(fill.fill_price - 450.00) < 1.0


# ---------------------------------------------------------------------------
# Test: Missing historical bar fallback
# ---------------------------------------------------------------------------

class TestMissingBarFallback:
    def test_missing_bar_falls_back_to_instant_fill(self, broker):
        """When _get_historical_bar returns None, delegate to execute()."""
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=100,
            current_price=450.00, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(broker, "_get_historical_bar", return_value=None):
            fill = broker.execute_algo_child(req, scheduled_time=scheduled)

        assert not fill.rejected
        assert fill.filled_quantity > 0
        # Should be close to current_price (instant-fill model)
        assert abs(fill.fill_price - 450.00) < 1.0


# ---------------------------------------------------------------------------
# Test: Zero volume bar rejects
# ---------------------------------------------------------------------------

class TestZeroVolumeBar:
    def test_zero_volume_rejects(self, mock_broker):
        """Zero volume bar -> rejection with reason."""
        zero_vol_bar = BarData(
            open=450.10, high=450.80, low=449.50,
            close=450.60, volume=0, vwap=450.30,
        )
        req = OrderRequest(
            symbol="SPY", side="buy", quantity=100,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(mock_broker, "_get_historical_bar", return_value=zero_vol_bar):
            fill = mock_broker.execute_algo_child(req, scheduled_time=scheduled)

        assert fill.rejected is True
        assert "zero volume" in fill.reject_reason.lower()


# ---------------------------------------------------------------------------
# Test: Fill leg dual-write
# ---------------------------------------------------------------------------

class TestFillLegDualWrite:
    def test_algo_child_records_fill_leg(self, db_connection, sample_bar):
        """execute_algo_child should record a fill_leg row via dual-write."""
        b = PaperBroker(conn=db_connection)
        oid = str(uuid.uuid4())
        req = OrderRequest(
            order_id=oid,
            symbol="SPY", side="buy", quantity=100,
            current_price=450.30, daily_volume=80_000_000,
        )
        scheduled = datetime(2026, 4, 6, 10, 30, 0)

        with patch.object(b, "_get_historical_bar", return_value=sample_bar):
            fill = b.execute_algo_child(req, scheduled_time=scheduled)

        assert not fill.rejected

        row = db_connection.execute(
            "SELECT order_id, quantity, price, venue FROM fill_legs WHERE order_id = ?",
            [oid],
        ).fetchone()
        assert row is not None
        assert row[0] == oid
        assert int(row[1]) == fill.filled_quantity
        assert float(row[2]) == fill.fill_price
        assert row[3] == "paper"
