"""
Cross-section integration tests for the execution layer.

Verifies real cross-module wiring between sections 02-12:
  1. TWAP child fill pipeline (02 + 06 + 05 + 08)
  2. PDT counting with algo round-trips (04)
  3. Wash sale into tax lot cost basis adjustment (04)
  4. Algo scheduler crash recovery (07 + 08)
  5. Options pin risk exit (12)
  6. Slippage accuracy drift alert (06 + 11)

All tests use real PostgreSQL with savepoint isolation.
"""

from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from quantstack.db import _migrate_execution_layer_pg, db_conn
from quantstack.execution.algo_scheduler import (
    AlgoParentOrder,
    ChildOrder,
    persist_child,
    persist_parent,
    startup_recovery,
    transition_child,
    transition_parent,
    update_parent_from_children,
)
from quantstack.execution.compliance.pretrade import ComplianceResult, PDTChecker
from quantstack.execution.compliance.posttrade import TaxLotManager, WashSaleTracker
from quantstack.execution.execution_monitor import (
    ExecutionMonitor,
    MonitoredPosition,
    OptionsMonitorRule,
)
from quantstack.execution.fill_utils import compute_fill_vwap, record_fill_leg
from quantstack.execution.paper_broker import BarData, Fill, OrderRequest, PaperBroker
from quantstack.execution.slippage import check_slippage_drift, record_slippage_accuracy
from quantstack.execution.tca_ewma import update_ewma_after_fill
from quantstack.execution.twap_vwap import plan_twap_children
from quantstack.holding_period import HoldingType

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_schema_migrated = False


@pytest.fixture()
def conn():
    """Yield a live PgConnection inside a savepoint that rolls back on teardown.

    Migrations run once per module to avoid DDL lock contention with row-level
    locks inside savepoints.
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
            c._raw.rollback()


# ---------------------------------------------------------------------------
# Test 1: TWAP Child Fill Pipeline (Sections 02, 06, 05, 08)
# ---------------------------------------------------------------------------


class TestTwapChildFillPipeline:
    """Plan TWAP children, execute via paper broker, record fills + TCA."""

    def test_twap_600_shares_3_slices(self, conn):
        """Create a TWAP parent for 600 shares over 15 minutes (3 x 5-min buckets).

        For each child:
          - Execute via PaperBroker.execute_algo_child()
          - Record fill leg via record_fill_leg()
          - Update TCA EWMA via update_ewma_after_fill()

        Verify:
          - 3 fill_legs rows exist for the parent order
          - tca_parameters has sample_count == 3
          - VWAP across legs matches compute_fill_vwap()
        """
        now = datetime.now(timezone.utc)
        parent = AlgoParentOrder(
            parent_order_id=f"INT-TWAP-{uuid.uuid4().hex[:8]}",
            symbol="SPY",
            side="buy",
            total_quantity=600,
            algo_type="twap",
            start_time=now,
            end_time=now + timedelta(minutes=15),
            arrival_price=450.00,
        )

        rng = random.Random(42)
        children = plan_twap_children(parent, bucket_minutes=5, rng=rng)
        assert len(children) == 3
        assert sum(c.target_quantity for c in children) == 600

        # Pre-seed a fill in the fills table so PaperBroker._get_historical_bar
        # can find a reference price for synthetic bar generation.
        conn.execute(
            "INSERT INTO fills "
            "(order_id, symbol, side, requested_quantity, filled_quantity, "
            "fill_price, slippage_bps, commission, partial, rejected, filled_at, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                f"seed-{uuid.uuid4().hex[:8]}", "SPY", "buy", 100, 100,
                450.00, 0.0, 0.0, False, False, now, "",
            ],
        )

        broker = PaperBroker(conn=conn)

        child_order_ids = []
        for child in children:
            order_id = f"{parent.parent_order_id}-fill-{child.child_id}"
            req = OrderRequest(
                order_id=order_id,
                symbol="SPY",
                side="buy",
                quantity=child.target_quantity,
                order_type="market",
                current_price=450.00,
                daily_volume=80_000_000,
            )
            fill = broker.execute_algo_child(
                req, scheduled_time=child.scheduled_time, participation_rate=0.02,
            )
            assert not fill.rejected, f"Child fill rejected: {fill.reject_reason}"
            assert fill.filled_quantity > 0

            # Record fill leg under the parent order ID for aggregation
            record_fill_leg(
                conn,
                order_id=parent.parent_order_id,
                quantity=fill.filled_quantity,
                price=fill.fill_price,
            )

            # Update TCA EWMA
            update_ewma_after_fill(
                conn,
                order_id=order_id,
                symbol="SPY",
                fill_timestamp=child.scheduled_time,
                arrival_price=parent.arrival_price,
                fill_price=fill.fill_price,
                fill_quantity=fill.filled_quantity,
                adv=80_000_000,
            )
            child_order_ids.append(order_id)

        # Verify: 3 fill_legs rows for the parent order
        row = conn.execute(
            "SELECT COUNT(*) FROM fill_legs WHERE order_id = ?",
            [parent.parent_order_id],
        ).fetchone()
        assert row[0] == 3, f"Expected 3 fill legs, got {row[0]}"

        # Verify: tca_parameters has sample_count == 3 for SPY
        tca_row = conn.execute(
            "SELECT sample_count FROM tca_parameters WHERE symbol = ?",
            ["SPY"],
        ).fetchone()
        assert tca_row is not None, "No tca_parameters row for SPY"
        assert tca_row[0] == 3, f"Expected sample_count=3, got {tca_row[0]}"

        # Verify: VWAP from compute_fill_vwap matches manual leg-VWAP
        vwap = compute_fill_vwap(conn, parent.parent_order_id)
        legs = conn.execute(
            "SELECT quantity, price FROM fill_legs WHERE order_id = ?",
            [parent.parent_order_id],
        ).fetchall()
        manual_vwap = sum(q * p for q, p in legs) / sum(q for q, _ in legs)
        assert abs(vwap - manual_vwap) < 1e-6, f"VWAP mismatch: {vwap} vs {manual_vwap}"


# ---------------------------------------------------------------------------
# Test 2: PDT Counting with Algo Round-Trips (Section 04)
# ---------------------------------------------------------------------------


class TestPDTCountingAlgoRoundTrips:
    """Pre-seed day trades, verify PDT rejection at threshold."""

    def test_pdt_blocks_4th_day_trade_under_25k(self, conn):
        """Pre-seed 3 day trades, then verify 4th is blocked when equity < $25K
        and allowed when equity >= $25K.
        """
        from quantstack.execution.compliance.calendar import trading_day_for

        # Use a timestamp during market hours on the most recent trading day
        # so that trading_day_for() returns a consistent date for both the
        # seeded day_trades and the order being checked.
        order_ts = datetime(2026, 4, 3, 14, 30, 0, tzinfo=timezone.utc)  # Friday 10:30 ET
        trade_day = trading_day_for(order_ts)

        checker = PDTChecker()

        # Pre-seed 3 day trades on that same trading day
        for i in range(3):
            conn.execute(
                "INSERT INTO day_trades "
                "(symbol, open_order_id, close_order_id, trade_date, quantity, account_equity) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (f"SEED{i}", f"open-{i}", f"close-{i}", trade_day, 100, 20_000.0),
            )

        # Build an order that would close a same-day position (triggering PDT check)
        order = {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 50,
            "timestamp": order_ts,
        }
        positions = [
            {
                "symbol": "AAPL",
                "side": "long",
                "opened_at": order_ts,
                "order_id": "open-aapl",
            }
        ]

        # With equity < $25K: should be BLOCKED
        result_blocked = checker.check(conn, order, account_equity=20_000.0, positions=positions)
        assert not result_blocked.approved, "Expected PDT block with equity < $25K"
        assert "PDT rule" in result_blocked.reason

        # With equity >= $25K: should be ALLOWED
        result_allowed = checker.check(conn, order, account_equity=30_000.0, positions=positions)
        assert result_allowed.approved, f"Expected PDT pass with equity >= $25K, got: {result_allowed.reason}"


# ---------------------------------------------------------------------------
# Test 3: Wash Sale into Tax Lot Cost Basis Adjustment (Section 04)
# ---------------------------------------------------------------------------


class TestWashSaleTaxLotAdjustment:
    """Buy → sell at loss → buy within 30 days → wash sale flagged + basis adjusted."""

    def test_wash_sale_adjusts_replacement_lot_cost_basis(self, conn):
        """Full wash sale lifecycle:
          1. Buy 100 XYZ at $50 (create tax lot)
          2. Sell at $45 (loss of $500, pending_wash_losses created)
          3. Buy 100 at $48 within 30 days
        Verify: wash_sale_flags row, replacement lot cost_basis = $53.
        """
        lot_mgr = TaxLotManager()
        wash_tracker = WashSaleTracker()
        today = date.today()
        ts = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)

        # Step 1: Buy 100 XYZ at $50
        buy1_event = {
            "order_id": "buy1-xyz",
            "symbol": "XYZ",
            "side": "buy",
            "quantity": 100,
            "price": 50.0,
            "timestamp": ts,
        }
        lot_result = lot_mgr.on_fill(conn, buy1_event)
        assert lot_result["action"] == "lot_created"

        # Step 2: Sell 100 XYZ at $45 (loss = 100 * (45-50) = -$500)
        sell_event = {
            "order_id": "sell1-xyz",
            "symbol": "XYZ",
            "side": "sell",
            "quantity": 100,
            "price": 45.0,
            "timestamp": ts + timedelta(hours=2),
            "realized_pnl": -500.0,
        }
        sell_lot_result = lot_mgr.on_fill(conn, sell_event)
        assert sell_lot_result["action"] == "lots_consumed"
        assert sell_lot_result["realized_pnl"] == -500.0

        wash_sell_result = wash_tracker.on_fill(conn, sell_event)
        assert wash_sell_result["action"] == "pending_loss_created"
        assert wash_sell_result["loss_amount"] == 500.0

        # Step 3: Buy 100 XYZ at $48 (within 30-day window)
        buy2_event = {
            "order_id": "buy2-xyz",
            "symbol": "XYZ",
            "side": "buy",
            "quantity": 100,
            "price": 48.0,
            "timestamp": ts + timedelta(days=5),
        }
        lot_result2 = lot_mgr.on_fill(conn, buy2_event)
        assert lot_result2["action"] == "lot_created"

        wash_buy_result = wash_tracker.on_fill(conn, buy2_event)
        assert wash_buy_result["action"] == "wash_sale_flagged"
        assert wash_buy_result["disallowed_loss"] == 500.0
        # adjusted_basis = $48 + ($500 / 100 shares) = $53
        assert abs(wash_buy_result["adjusted_basis"] - 53.0) < 1e-6

        # Verify: wash_sale_flags row exists
        wsf_row = conn.execute(
            "SELECT disallowed_loss, adjusted_cost_basis FROM wash_sale_flags "
            "WHERE replacement_order_id = %s",
            ("buy2-xyz",),
        ).fetchone()
        assert wsf_row is not None, "No wash_sale_flags row found"
        assert float(wsf_row[0]) == 500.0
        assert abs(float(wsf_row[1]) - 53.0) < 1e-6

        # Verify: replacement tax lot has adjusted cost_basis = $53
        lot_row = conn.execute(
            "SELECT cost_basis, wash_sale_adjustment FROM tax_lots "
            "WHERE order_id = %s AND symbol = %s AND status = 'open'",
            ("buy2-xyz", "XYZ"),
        ).fetchone()
        assert lot_row is not None, "Replacement tax lot not found"
        assert abs(float(lot_row[0]) - 53.0) < 1e-6
        assert float(lot_row[1]) == 500.0


# ---------------------------------------------------------------------------
# Test 4: Algo Scheduler Crash Recovery (Sections 07, 08)
# ---------------------------------------------------------------------------


class TestAlgoSchedulerCrashRecovery:
    """Simulate a crash with in-flight parent + children, then recover."""

    def test_recovery_cancels_active_parent_preserves_filled_children(self, conn):
        """Create parent + 6 children (3 filled, 3 pending).
        Persist to DB, then run startup_recovery().
        Verify: parent cancelled, pending children cancelled, filled children preserved.
        """
        now = datetime.now(timezone.utc)
        parent_id = f"CRASH-{uuid.uuid4().hex[:8]}"

        parent = AlgoParentOrder(
            parent_order_id=parent_id,
            symbol="QQQ",
            side="buy",
            total_quantity=600,
            algo_type="twap",
            start_time=now - timedelta(minutes=15),
            end_time=now + timedelta(minutes=15),
            arrival_price=380.00,
            status="active",
        )

        children: list[ChildOrder] = []
        for i in range(6):
            child = ChildOrder(
                child_id=f"{parent_id}-C{i+1:03d}",
                parent_id=parent_id,
                scheduled_time=now + timedelta(minutes=i * 5),
                target_quantity=100,
            )
            if i < 3:
                # Simulate filled children: transition pending -> submitted -> filled
                transition_child(child, "submitted")
                transition_child(child, "filled")
                child.filled_quantity = 100
                child.fill_price = 380.0 + i * 0.10
            else:
                # Leave as pending (simulating crash before submission)
                pass
            children.append(child)

        # Persist to DB
        persist_parent(conn, parent)
        for child in children:
            persist_child(conn, child)

        # Run crash recovery
        recovered_count = startup_recovery(conn, broker=None)
        assert recovered_count == 1, f"Expected 1 parent recovered, got {recovered_count}"

        # Verify parent is cancelled
        parent_row = conn.execute(
            "SELECT status FROM algo_parent_orders WHERE parent_order_id = %s",
            (parent_id,),
        ).fetchone()
        assert parent_row[0] == "cancelled"

        # Verify: pending children (indices 3-5) are cancelled
        for i in range(3, 6):
            child_row = conn.execute(
                "SELECT status FROM algo_child_orders WHERE child_id = %s",
                (f"{parent_id}-C{i+1:03d}",),
            ).fetchone()
            assert child_row[0] == "cancelled", (
                f"Expected child C{i+1:03d} to be cancelled, got {child_row[0]}"
            )

        # Verify: filled children (indices 0-2) are preserved
        for i in range(3):
            child_row = conn.execute(
                "SELECT status, filled_quantity FROM algo_child_orders WHERE child_id = %s",
                (f"{parent_id}-C{i+1:03d}",),
            ).fetchone()
            assert child_row[0] == "filled", (
                f"Expected child C{i+1:03d} to remain filled, got {child_row[0]}"
            )
            assert child_row[1] == 100


# ---------------------------------------------------------------------------
# Test 5: Options Pin Risk Exit (Section 12)
# ---------------------------------------------------------------------------


class TestOptionsPinRiskExit:
    """Verify pin_risk rule fires when underlying is within 1% of strike near expiry."""

    @pytest.mark.asyncio
    async def test_pin_risk_triggers_near_strike_low_dte(self):
        """DTE=2, strike=150, underlying=$149.50 (0.33% away) → pin_risk fires."""
        pos = MonitoredPosition(
            symbol="AAPL",
            side="long",
            quantity=1,
            holding_type=HoldingType.SWING,
            entry_price=3.00,
            entry_time=datetime.now(ET) - timedelta(hours=2),
            instrument_type="options",
            underlying_symbol="AAPL",
            option_contract="AAPL_240101_C150",
            option_strike=150.0,
            option_expiry=date.today() + timedelta(days=2),
            option_type="call",
            entry_premium=3.00,
            strategy_id="test_pin",
        )

        monitor = ExecutionMonitor(
            broker=MagicMock(),
            price_feed=AsyncMock(),
            portfolio_state=MagicMock(get_positions=MagicMock(return_value=[])),
        )
        # Ensure pin_risk rule is enabled with auto_exit
        monitor._options_rules["pin_risk"] = OptionsMonitorRule("pin_risk", True, "auto_exit")

        mock_greeks = {
            "greeks": {"theta": -0.05, "delta": 0.5, "gamma": 0.03, "vega": 0.10},
            "backend_used": "internal",
            "interpretations": {},
            "risk_metrics": {},
        }

        now = datetime.now(ET)
        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=mock_greeks,
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=149.50, current_time=now,
            )

        assert should_exit is True
        assert "pin_risk" in reason

    @pytest.mark.asyncio
    async def test_pin_risk_no_trigger_price_too_far(self):
        """DTE=2, strike=150, underlying=$145 (3.3% away) → no trigger."""
        pos = MonitoredPosition(
            symbol="AAPL",
            side="long",
            quantity=1,
            holding_type=HoldingType.SWING,
            entry_price=3.00,
            entry_time=datetime.now(ET) - timedelta(hours=2),
            instrument_type="options",
            underlying_symbol="AAPL",
            option_contract="AAPL_240101_C150",
            option_strike=150.0,
            option_expiry=date.today() + timedelta(days=2),
            option_type="call",
            entry_premium=3.00,
            strategy_id="test_pin",
        )

        monitor = ExecutionMonitor(
            broker=MagicMock(),
            price_feed=AsyncMock(),
            portfolio_state=MagicMock(get_positions=MagicMock(return_value=[])),
        )
        monitor._options_rules["pin_risk"] = OptionsMonitorRule("pin_risk", True, "auto_exit")

        mock_greeks = {
            "greeks": {"theta": -0.05, "delta": 0.5, "gamma": 0.03, "vega": 0.10},
            "backend_used": "internal",
            "interpretations": {},
            "risk_metrics": {},
        }

        now = datetime.now(ET)
        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=mock_greeks,
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=145.0, current_time=now,
            )

        assert should_exit is False

    @pytest.mark.asyncio
    async def test_pin_risk_no_trigger_dte_too_high(self):
        """DTE=10, strike=150, underlying=$149.50 → no trigger (DTE >= 3)."""
        pos = MonitoredPosition(
            symbol="AAPL",
            side="long",
            quantity=1,
            holding_type=HoldingType.SWING,
            entry_price=3.00,
            entry_time=datetime.now(ET) - timedelta(hours=2),
            instrument_type="options",
            underlying_symbol="AAPL",
            option_contract="AAPL_240101_C150",
            option_strike=150.0,
            option_expiry=date.today() + timedelta(days=10),
            option_type="call",
            entry_premium=3.00,
            strategy_id="test_pin",
        )

        monitor = ExecutionMonitor(
            broker=MagicMock(),
            price_feed=AsyncMock(),
            portfolio_state=MagicMock(get_positions=MagicMock(return_value=[])),
        )
        monitor._options_rules["pin_risk"] = OptionsMonitorRule("pin_risk", True, "auto_exit")

        mock_greeks = {
            "greeks": {"theta": -0.05, "delta": 0.5, "gamma": 0.03, "vega": 0.10},
            "backend_used": "internal",
            "interpretations": {},
            "risk_metrics": {},
        }

        now = datetime.now(ET)
        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=mock_greeks,
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=149.50, current_time=now,
            )

        assert should_exit is False


# ---------------------------------------------------------------------------
# Test 6: Slippage Accuracy Drift Alert (Sections 06, 11)
# ---------------------------------------------------------------------------


class TestSlippageDriftAlert:
    """Seed bad predicted/realized ratios and verify drift detection."""

    def test_drift_alert_when_model_under_predicts(self, conn):
        """Seed slippage_accuracy rows with realized >> predicted.

        Predicted=5 bps, realized=12 bps → ratio=2.4 (outside [0.5, 2.0]).
        check_slippage_drift() should return an alert string.
        """
        # Seed tca_parameters so the table has context (optional but realistic)
        conn.execute(
            """
            INSERT INTO tca_parameters
                (symbol, time_bucket, ewma_spread_bps, ewma_impact_bps,
                 ewma_total_bps, sample_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, NOW())
            """,
            ["DRIFT_TEST", "morning", 2.0, 3.0, 5.0, 60],
        )

        # Record 25 slippage_accuracy rows with bad ratios (under-prediction)
        for i in range(25):
            record_slippage_accuracy(
                conn,
                order_id=f"drift-{i}",
                symbol="DRIFT_TEST",
                time_bucket="morning",
                predicted_bps=5.0,
                realized_bps=12.0,
            )

        alert = check_slippage_drift(conn, symbol="DRIFT_TEST", lookback_count=20)
        assert alert is not None, "Expected a drift alert but got None"
        assert "under-predicting" in alert
        assert "DRIFT_TEST" in alert

    def test_no_drift_alert_when_model_accurate(self, conn):
        """Predicted=5, realized=5 → ratio=1.0 → no alert."""
        for i in range(25):
            record_slippage_accuracy(
                conn,
                order_id=f"nodrift-{i}",
                symbol="ACCURATE_SYM",
                time_bucket="morning",
                predicted_bps=5.0,
                realized_bps=5.0,
            )

        alert = check_slippage_drift(conn, symbol="ACCURATE_SYM", lookback_count=20)
        assert alert is None, f"Expected no drift alert, got: {alert}"
