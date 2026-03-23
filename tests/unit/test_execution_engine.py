"""
Unit tests for Phase 5 execution engine components.

Covers:
- KillSwitch           (kill_switch.py)
- FillTracker          (fill_tracker.py)
- PreTradeRiskGate     (risk_gate.py)
- SmartOrderRouter     (smart_order_router.py)
- AsyncExecutionLoop   (async_execution_loop.py)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from quantstack.core.execution.async_execution_loop import AsyncExecutionLoop
from quantstack.core.execution.broker import BrokerInterface
from quantstack.core.execution.fill_tracker import FillEvent, FillTracker
from quantstack.core.execution.kill_switch import KillSwitch, KillSwitchError
from quantstack.core.execution.risk_gate import (
    PreTradeRiskGate,
    RiskGateError,
    RiskLimits,
)
from quantstack.core.execution.smart_order_router import (
    SmartOrderRouter,
    SmartOrderRouterError,
)
from quantstack.core.execution.unified_models import (
    UnifiedOrder,
    UnifiedOrderResult,
)
from quantstack.config.timeframes import Timeframe
from quantstack.core.execution.broker import BrokerError
from quantstack.data.streaming.incremental_features import IncrementalFeatures
import time

# ---------------------------------------------------------------------------
# Helpers / shared factories
# ---------------------------------------------------------------------------


def _order(
    symbol: str = "SPY",
    side: str = "buy",
    qty: float = 10.0,
    order_type: str = "market",
    limit_price: float | None = None,
) -> UnifiedOrder:
    return UnifiedOrder(
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type=order_type,
        limit_price=limit_price,
    )


def _fill(
    symbol: str = "SPY",
    side: str = "buy",
    qty: float = 10.0,
    price: float = 100.0,
    order_id: str = "ORD-001",
) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        symbol=symbol,
        side=side,
        filled_qty=qty,
        avg_fill_price=price,
        timestamp=datetime.now(UTC),
    )


def _result(
    order_id: str = "ORD-001",
    status: str = "filled",
    filled_qty: float = 10.0,
    avg_fill_price: float = 100.0,
    symbol: str = "SPY",
    side: str = "buy",
) -> UnifiedOrderResult:
    return UnifiedOrderResult(
        order_id=order_id,
        client_order_id=None,
        symbol=symbol,
        side=side,
        quantity=filled_qty,
        order_type="market",
        limit_price=None,
        stop_price=None,
        status=status,
        filled_qty=filled_qty,
        avg_fill_price=avg_fill_price,
    )


def _mock_broker(
    healthy: bool = True, result: UnifiedOrderResult | None = None
) -> MagicMock:
    """Return a MagicMock that satisfies BrokerInterface."""
    broker = MagicMock(spec=BrokerInterface)
    broker.check_auth.return_value = healthy
    broker.place_order.return_value = result or _result()
    broker.cancel_order.return_value = True
    return broker


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    """Tests for file-sentinel kill switch."""

    @pytest.fixture()
    def ks(self, tmp_path: Path) -> KillSwitch:
        return KillSwitch(sentinel_path=tmp_path / "KILL")

    def test_inactive_by_default(self, ks: KillSwitch) -> None:
        assert not ks.is_active()

    def test_check_passes_when_inactive(self, ks: KillSwitch) -> None:
        ks.check()  # must not raise

    def test_activate_creates_sentinel(self, ks: KillSwitch) -> None:
        ks.activate()
        assert ks.is_active()
        assert ks._path.exists()

    def test_activate_writes_reason(self, ks: KillSwitch) -> None:
        ks.activate(reason="drawdown limit hit")
        assert "drawdown limit hit" in ks._path.read_text()

    def test_check_raises_when_active(self, ks: KillSwitch) -> None:
        ks.activate()
        with pytest.raises(KillSwitchError, match="ACTIVE"):
            ks.check()

    def test_deactivate_removes_sentinel(self, ks: KillSwitch) -> None:
        ks.activate()
        ks.deactivate()
        assert not ks.is_active()
        assert not ks._path.exists()

    def test_deactivate_idempotent(self, ks: KillSwitch) -> None:
        ks.deactivate()  # called when not active — must not raise

    def test_status_inactive(self, ks: KillSwitch) -> None:
        s = ks.status()
        assert s["active"] is False

    def test_status_active_includes_reason(self, ks: KillSwitch) -> None:
        ks.activate(reason="test reason")
        s = ks.status()
        assert s["active"] is True
        assert s["reason"] == "test reason"

    def test_check_passes_after_deactivate(self, ks: KillSwitch) -> None:
        ks.activate()
        ks.deactivate()
        ks.check()  # must not raise


# ---------------------------------------------------------------------------
# FillTracker
# ---------------------------------------------------------------------------


class TestFillTracker:
    """Tests for live position accounting in FillTracker."""

    @pytest.fixture()
    def tracker(self) -> FillTracker:
        return FillTracker(starting_cash=100_000.0)

    # --- Opening positions ---

    def test_open_long_position(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        pos = tracker.get_position("SPY")
        assert pos is not None
        assert pos.quantity == 100.0
        assert pos.avg_cost == 400.0

    def test_open_short_position(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "sell", 50, 400.0))
        pos = tracker.get_position("SPY")
        assert pos is not None
        assert pos.quantity == -50.0

    # --- Adding to position ---

    def test_add_to_long_updates_avg_cost(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("SPY", "buy", 100, 500.0, order_id="ORD-002"))
        pos = tracker.get_position("SPY")
        assert pos.quantity == 200.0
        assert pos.avg_cost == pytest.approx(450.0)

    # --- Reducing / closing ---

    def test_partial_close_reduces_qty(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("SPY", "sell", 40, 420.0, order_id="ORD-002"))
        pos = tracker.get_position("SPY")
        assert pos.quantity == pytest.approx(60.0)

    def test_full_close_zeroes_qty(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("SPY", "sell", 100, 420.0, order_id="ORD-002"))
        pos = tracker.get_position("SPY")
        assert pos.quantity == pytest.approx(0.0)
        assert pos.avg_cost == pytest.approx(0.0)

    def test_realised_pnl_on_close(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("SPY", "sell", 100, 420.0, order_id="ORD-002"))
        pos = tracker.get_position("SPY")
        # 100 shares × $20 gain = $2,000
        assert pos.realised_pnl == pytest.approx(2_000.0)

    def test_realised_pnl_short(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "sell", 100, 400.0))
        tracker.update_fill(_fill("SPY", "buy", 100, 380.0, order_id="ORD-002"))
        pos = tracker.get_position("SPY")
        # 100 × (400 - 380) = $2,000 gain on short
        assert pos.realised_pnl == pytest.approx(2_000.0)

    def test_position_flip(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        # Sell 150: closes 100 long and opens 50 short
        tracker.update_fill(_fill("SPY", "sell", 150, 410.0, order_id="ORD-002"))
        pos = tracker.get_position("SPY")
        assert pos.quantity == pytest.approx(-50.0)
        # After flip, avg_cost should be the flip price
        assert pos.avg_cost == pytest.approx(410.0)

    # --- Price updates ---

    def test_update_price_refreshes_open_pnl(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_price("SPY", 450.0)
        pos = tracker.get_position("SPY")
        assert pos.current_price == 450.0
        assert pos.open_pnl == pytest.approx(5_000.0)  # 100 × $50

    def test_update_prices_bulk(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 10, 400.0))
        tracker.update_fill(_fill("AAPL", "buy", 20, 150.0, order_id="ORD-002"))
        tracker.update_prices({"SPY": 410.0, "AAPL": 160.0})
        assert tracker.get_position("SPY").current_price == 410.0
        assert tracker.get_position("AAPL").current_price == 160.0

    # --- Aggregate queries ---

    def test_daily_realised_pnl(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("SPY", "sell", 100, 420.0, order_id="ORD-002"))
        assert tracker.daily_realised_pnl() == pytest.approx(2_000.0)

    def test_net_exposure(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("AAPL", "buy", 10, 150.0, order_id="ORD-002"))
        # SPY: 100 × 400 = 40,000; AAPL: 10 × 150 = 1,500; total = 41,500
        assert tracker.net_exposure() == pytest.approx(41_500.0)

    def test_position_count(self, tracker: FillTracker) -> None:
        assert tracker.position_count() == 0
        tracker.update_fill(_fill("SPY", "buy", 10, 400.0))
        assert tracker.position_count() == 1
        tracker.update_fill(_fill("SPY", "sell", 10, 410.0, order_id="ORD-002"))
        assert tracker.position_count() == 0

    def test_get_open_positions_excludes_closed(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 10, 400.0))
        tracker.update_fill(_fill("SPY", "sell", 10, 410.0, order_id="ORD-002"))
        assert "SPY" not in tracker.get_open_positions()

    def test_reset_daily_pnl(self, tracker: FillTracker) -> None:
        tracker.update_fill(_fill("SPY", "buy", 100, 400.0))
        tracker.update_fill(_fill("SPY", "sell", 100, 420.0, order_id="ORD-002"))
        tracker.reset_daily_pnl()
        assert tracker.daily_realised_pnl() == pytest.approx(0.0)
        assert tracker.fill_count() == 0


# ---------------------------------------------------------------------------
# PreTradeRiskGate
# ---------------------------------------------------------------------------


class TestPreTradeRiskGate:
    """Tests for synchronous pre-trade risk checks."""

    @pytest.fixture()
    def ks(self, tmp_path: Path) -> KillSwitch:
        return KillSwitch(sentinel_path=tmp_path / "KILL")

    @pytest.fixture()
    def tracker(self) -> FillTracker:
        return FillTracker()

    @pytest.fixture()
    def gate(self, tracker: FillTracker, ks: KillSwitch) -> PreTradeRiskGate:
        limits = RiskLimits(
            max_order_value=10_000.0,
            max_position_value=50_000.0,
            max_positions=5,
            max_orders_per_min=10,
            max_daily_loss=1_000.0,
        )
        return PreTradeRiskGate(limits=limits, fill_tracker=tracker, kill_switch=ks)

    # --- Happy path ---

    def test_passes_clean_order(self, gate: PreTradeRiskGate) -> None:
        gate.check(_order("SPY", "buy", 10), current_price=100.0)  # notional=$1,000

    # --- Kill switch ---

    def test_blocks_on_kill_switch(
        self, gate: PreTradeRiskGate, ks: KillSwitch
    ) -> None:
        ks.activate()
        with pytest.raises(RiskGateError) as exc_info:
            gate.check(_order("SPY", "buy", 10), current_price=100.0)
        assert exc_info.value.rule == "KILL_SWITCH"

    # --- Order size ---

    def test_blocks_oversized_order(self, gate: PreTradeRiskGate) -> None:
        # 200 shares × $100 = $20,000 > limit $10,000
        with pytest.raises(RiskGateError) as exc_info:
            gate.check(_order("SPY", "buy", 200), current_price=100.0)
        assert exc_info.value.rule == "MAX_ORDER_SIZE"

    def test_order_at_exact_limit_passes(self, gate: PreTradeRiskGate) -> None:
        # 100 × $100 = $10,000 == limit — should pass (not strictly greater)
        gate.check(_order("SPY", "buy", 100), current_price=100.0)

    # --- Position size ---

    def test_blocks_when_resulting_position_too_large(
        self, gate: PreTradeRiskGate, tracker: FillTracker
    ) -> None:
        # Pre-fill 495 shares; adding 10 more at $100 → 505 × $100 = $50,500 > $50,000 limit.
        # Order notional: 10 × $100 = $1,000 < $10,000 (order-size check passes).
        tracker.update_fill(_fill("SPY", "buy", 495, 100.0))
        with pytest.raises(RiskGateError) as exc_info:
            gate.check(_order("SPY", "buy", 10), current_price=100.0)
        assert exc_info.value.rule == "MAX_POSITION_SIZE"

    # --- Max positions ---

    def test_blocks_new_position_when_at_limit(
        self, gate: PreTradeRiskGate, tracker: FillTracker
    ) -> None:
        for sym in ["A", "B", "C", "D", "E"]:
            tracker.update_fill(_fill(sym, "buy", 1, 10.0, order_id=sym))
        # Already 5 open positions; opening a 6th should fail
        with pytest.raises(RiskGateError) as exc_info:
            gate.check(_order("NEWSTOCK", "buy", 1), current_price=10.0)
        assert exc_info.value.rule == "MAX_POSITIONS"

    def test_allows_adding_to_existing_position_at_max(
        self, gate: PreTradeRiskGate, tracker: FillTracker
    ) -> None:
        for sym in ["A", "B", "C", "D", "E"]:
            tracker.update_fill(_fill(sym, "buy", 1, 10.0, order_id=sym))
        # Adding to "A" (already open) must not be blocked by max_positions
        gate.check(_order("A", "buy", 1), current_price=10.0)

    # --- Rate limit ---

    def test_blocks_on_rate_limit(self, gate: PreTradeRiskGate) -> None:
        for _ in range(10):
            gate.record_submission()
        with pytest.raises(RiskGateError) as exc_info:
            gate.check(_order("SPY", "buy", 1), current_price=100.0)
        assert exc_info.value.rule == "ORDER_RATE"

    def test_rate_window_resets_after_ttl(self, gate: PreTradeRiskGate) -> None:
        # Manually stuff stale timestamps into the deque
        old_ts = time.monotonic() - 61.0
        for _ in range(10):
            gate._order_times.append(old_ts)
        # All entries are outside the 60s window — check must pass
        gate.check(_order("SPY", "buy", 1), current_price=100.0)

    # --- Daily drawdown ---

    def test_blocks_on_daily_loss(
        self, gate: PreTradeRiskGate, tracker: FillTracker
    ) -> None:
        # Buy at $100, current mark at $88 → open_pnl = 100 × (88 - 100) = -$1,200 < -$1,000
        tracker.update_fill(_fill("SPY", "buy", 100, 100.0))
        tracker.update_price("SPY", 88.0)
        with pytest.raises(RiskGateError) as exc_info:
            gate.check(_order("SPY", "buy", 1), current_price=88.0)
        assert exc_info.value.rule == "DAILY_DRAWDOWN"

    def test_drawdown_check_disabled_when_limit_zero(
        self, tracker: FillTracker, ks: KillSwitch
    ) -> None:
        limits = RiskLimits(max_daily_loss=0.0)
        gate = PreTradeRiskGate(limits=limits, fill_tracker=tracker, kill_switch=ks)
        tracker.update_fill(_fill("SPY", "buy", 100, 100.0))
        tracker.update_price("SPY", 1.0)  # catastrophic loss — still passes
        gate.check(_order("SPY", "buy", 1), current_price=1.0)

    # --- record_submission ---

    def test_record_submission_increments_counter(self, gate: PreTradeRiskGate) -> None:
        assert len(gate._order_times) == 0
        gate.record_submission()
        assert len(gate._order_times) == 1

    # --- status ---

    def test_status_returns_dict(self, gate: PreTradeRiskGate) -> None:
        s = gate.status()
        assert "kill_switch" in s
        assert "orders_last_60s" in s
        assert "daily_pnl" in s


# ---------------------------------------------------------------------------
# SmartOrderRouter
# ---------------------------------------------------------------------------


class TestSmartOrderRouter:
    """Tests for multi-broker order routing."""

    @pytest.fixture()
    def tracker(self) -> FillTracker:
        return FillTracker()

    def _router(
        self,
        tracker: FillTracker,
        alpaca_healthy: bool = True,
        ibkr_healthy: bool = True,
    ) -> SmartOrderRouter:
        alpaca = _mock_broker(
            healthy=alpaca_healthy, result=_result(order_id="ALP-001")
        )
        ibkr = _mock_broker(healthy=ibkr_healthy, result=_result(order_id="IBK-001"))
        return SmartOrderRouter(
            alpaca_broker=alpaca,
            ibkr_broker=ibkr,
            fill_tracker=tracker,
            paper=False,
        )

    # --- Routing logic ---

    def test_equity_routes_to_alpaca_first(self, tracker: FillTracker) -> None:
        router = self._router(tracker)
        result = router.route("ACC1", _order("SPY"), asset_class="equity")
        assert result.order_id == "ALP-001"

    def test_equity_falls_back_to_ibkr_when_alpaca_unhealthy(
        self, tracker: FillTracker
    ) -> None:
        router = self._router(tracker, alpaca_healthy=False)
        result = router.route("ACC1", _order("SPY"), asset_class="equity")
        assert result.order_id == "IBK-001"

    def test_futures_routes_to_ibkr(self, tracker: FillTracker) -> None:
        router = self._router(tracker)
        result = router.route("ACC1", _order("ES"), asset_class="futures")
        assert result.order_id == "IBK-001"

    def test_futures_raises_when_ibkr_not_configured(
        self, tracker: FillTracker
    ) -> None:
        alpaca = _mock_broker(healthy=True)
        router = SmartOrderRouter(
            alpaca_broker=alpaca, fill_tracker=tracker, paper=False
        )
        with pytest.raises(SmartOrderRouterError, match="IBKR"):
            router.route("ACC1", _order("ES"), asset_class="futures")

    def test_raises_when_no_broker_available(self, tracker: FillTracker) -> None:
        router = SmartOrderRouter(fill_tracker=tracker)
        with pytest.raises(SmartOrderRouterError):
            router.route("ACC1", _order("SPY"))

    def test_raises_when_all_brokers_unhealthy(self, tracker: FillTracker) -> None:
        router = self._router(tracker, alpaca_healthy=False, ibkr_healthy=False)
        with pytest.raises(SmartOrderRouterError):
            router.route("ACC1", _order("SPY"))

    # --- Fill recording ---

    def test_fill_recorded_after_placement(self, tracker: FillTracker) -> None:
        router = self._router(tracker)
        router.route("ACC1", _order("SPY", "buy", 10), asset_class="equity")
        pos = tracker.get_position("SPY")
        assert pos is not None
        assert pos.quantity == pytest.approx(10.0)

    def test_no_fill_recorded_on_zero_fill(self, tracker: FillTracker) -> None:
        alpaca = _mock_broker(result=_result(order_id="ALP-001", filled_qty=0.0))
        router = SmartOrderRouter(
            alpaca_broker=alpaca, fill_tracker=tracker, paper=False
        )
        router.route("ACC1", _order("SPY"))
        assert tracker.get_position("SPY") is None

    # --- Paper mode ---

    def test_paper_mode_forces_alpaca(self, tracker: FillTracker) -> None:
        alpaca = _mock_broker(result=_result(order_id="PAPER-001"))
        ibkr = _mock_broker(result=_result(order_id="IBK-001"))
        router = SmartOrderRouter(
            alpaca_broker=alpaca,
            ibkr_broker=ibkr,
            fill_tracker=tracker,
            paper=True,
        )
        result = router.route("ACC1", _order("GC"), asset_class="futures")
        assert result.order_id == "PAPER-001"

    # --- available_brokers ---

    def test_available_brokers_lists_configured(self, tracker: FillTracker) -> None:
        router = self._router(tracker)
        brokers = router.available_brokers()
        assert "alpaca" in brokers
        assert "ibkr" in brokers

    def test_available_brokers_empty_when_none_configured(
        self, tracker: FillTracker
    ) -> None:
        router = SmartOrderRouter(fill_tracker=tracker)
        assert router.available_brokers() == []

    # --- cancel ---

    def test_cancel_delegates_to_alpaca(self, tracker: FillTracker) -> None:
        router = self._router(tracker)
        ok = router.cancel("ACC1", "ORD-123", asset_class="equity")
        assert ok is True

    # --- Health cache ---

    def test_unhealthy_broker_skipped_on_retry(self, tracker: FillTracker) -> None:
        alpaca = _mock_broker(healthy=True)
        alpaca.place_order.side_effect = BrokerError("rejected")
        ibkr = _mock_broker(healthy=True, result=_result(order_id="IBK-001"))
        router = SmartOrderRouter(
            alpaca_broker=alpaca,
            ibkr_broker=ibkr,
            fill_tracker=tracker,
            paper=False,
        )
        result = router.route("ACC1", _order("SPY"))
        assert result.order_id == "IBK-001"


# ---------------------------------------------------------------------------
# AsyncExecutionLoop
# ---------------------------------------------------------------------------


def _warm_features(
    symbol: str = "SPY",
    close: float = 100.0,
    ema_cross: float = 1.0,
    rsi: float = 50.0,
) -> object:
    """Build a minimal IncrementalFeatures-like object."""
    return IncrementalFeatures(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        timeframe=Timeframe.M1,
        close=close,
        ema_fast=close + ema_cross / 2,
        ema_slow=close - ema_cross / 2,
        ema_cross=ema_cross,
        rsi=rsi,
        roc=0.5,
        atr=1.0,
        atr_pct=0.01,
        bb_upper=close + 2,
        bb_lower=close - 2,
        bb_pct_b=0.5,
        volume_ratio=1.0,
        price_to_ema=0.1,
        vwap_deviation=None,
        is_warm=True,
    )


def _cold_features() -> object:
    f = _warm_features()
    object.__setattr__(f, "is_warm", False)
    return f


class TestAsyncExecutionLoop:
    """Tests for the asyncio execution loop."""

    @pytest.fixture()
    def tracker(self) -> FillTracker:
        return FillTracker()

    @pytest.fixture()
    def ks(self, tmp_path: Path) -> KillSwitch:
        return KillSwitch(sentinel_path=tmp_path / "KILL")

    @pytest.fixture()
    def gate(self, tracker: FillTracker, ks: KillSwitch) -> PreTradeRiskGate:
        return PreTradeRiskGate(
            limits=RiskLimits(max_daily_loss=0.0),  # disable drawdown for loop tests
            fill_tracker=tracker,
            kill_switch=ks,
        )

    @pytest.fixture()
    def router(self, tracker: FillTracker) -> SmartOrderRouter:
        broker = _mock_broker(result=_result(order_id="TEST-001"))
        return SmartOrderRouter(alpaca_broker=broker, fill_tracker=tracker, paper=False)

    # --- Cold signal suppression ---

    async def test_cold_features_do_not_produce_orders(
        self,
        gate: PreTradeRiskGate,
        router: SmartOrderRouter,
    ) -> None:
        placed: list = []

        async def evaluator(f):
            return _order("SPY")

        original_route = router.route

        def capturing_route(account_id, order, asset_class="equity"):
            placed.append(order)
            return original_route(account_id, order, asset_class)

        router.route = capturing_route

        loop = AsyncExecutionLoop(evaluator, gate, router, account_id="ACC1")
        await loop.start()
        await loop.on_features(_cold_features())
        await asyncio.sleep(0.05)  # let any tasks drain
        await loop.stop()

        assert placed == []
        assert loop.stats()["signals_warm"] == 0

    # --- Hold signal (evaluator returns None) ---

    async def test_hold_signal_does_not_submit(
        self,
        gate: PreTradeRiskGate,
        router: SmartOrderRouter,
    ) -> None:
        async def hold_evaluator(f):
            return None

        loop = AsyncExecutionLoop(hold_evaluator, gate, router, account_id="ACC1")
        await loop.start()
        await loop.on_features(_warm_features())
        await asyncio.sleep(0.05)
        await loop.stop()

        assert loop.stats()["orders_attempted"] == 0

    # --- Order submission ---

    async def test_warm_signal_submits_order(
        self,
        gate: PreTradeRiskGate,
        router: SmartOrderRouter,
    ) -> None:
        async def buy_evaluator(f):
            return _order(f.symbol, "buy", 5)

        loop = AsyncExecutionLoop(buy_evaluator, gate, router, account_id="ACC1")
        await loop.start()
        await loop.on_features(_warm_features("SPY"))
        await asyncio.sleep(0.1)
        await loop.stop()

        s = loop.stats()
        assert s["orders_attempted"] == 1
        assert s["orders_placed"] == 1

    # --- Risk gate rejection ---

    async def test_risk_gate_rejection_not_counted_as_placed(
        self,
        tracker: FillTracker,
        router: SmartOrderRouter,
        tmp_path: Path,
    ) -> None:
        ks = KillSwitch(sentinel_path=tmp_path / "KILL")
        ks.activate()
        gate = PreTradeRiskGate(
            limits=RiskLimits(max_daily_loss=0.0),
            fill_tracker=tracker,
            kill_switch=ks,
        )

        async def buy_evaluator(f):
            return _order(f.symbol, "buy", 5)

        loop = AsyncExecutionLoop(buy_evaluator, gate, router, account_id="ACC1")
        await loop.start()
        await loop.on_features(_warm_features("SPY"))
        await asyncio.sleep(0.1)
        await loop.stop()

        s = loop.stats()
        assert s["orders_attempted"] == 1
        assert s["orders_placed"] == 0
        assert s["risk_rejections"] == 1

    # --- Evaluator exception ---

    async def test_evaluator_exception_increments_counter(
        self,
        gate: PreTradeRiskGate,
        router: SmartOrderRouter,
    ) -> None:
        async def bad_evaluator(f):
            raise ValueError("strategy error")

        loop = AsyncExecutionLoop(bad_evaluator, gate, router, account_id="ACC1")
        await loop.start()
        await loop.on_features(_warm_features("SPY"))
        await asyncio.sleep(0.1)
        await loop.stop()

        s = loop.stats()
        assert s["exceptions"] == 1
        assert s["orders_placed"] == 0

    # --- Not running ---

    async def test_loop_ignores_features_when_stopped(
        self,
        gate: PreTradeRiskGate,
        router: SmartOrderRouter,
    ) -> None:
        async def buy_evaluator(f):
            return _order(f.symbol, "buy", 5)

        loop = AsyncExecutionLoop(buy_evaluator, gate, router, account_id="ACC1")
        # Never called start() — _running is False
        await loop.on_features(_warm_features("SPY"))
        await asyncio.sleep(0.05)

        assert loop.stats()["signals_received"] == 0

    # --- Stats ---

    async def test_stats_tracks_symbols(
        self,
        gate: PreTradeRiskGate,
        router: SmartOrderRouter,
    ) -> None:
        async def hold(_):
            return None

        loop = AsyncExecutionLoop(hold, gate, router)
        await loop.start()
        await loop.on_features(_warm_features("SPY"))
        await loop.on_features(_warm_features("AAPL"))
        await loop.on_features(_warm_features("SPY"))  # duplicate
        await asyncio.sleep(0.05)
        await loop.stop()

        assert loop.stats()["symbols_tracked"] == 2

    # --- Custom price_fn ---

    async def test_custom_price_fn_used_for_risk_check(
        self,
        tracker: FillTracker,
        tmp_path: Path,
        router: SmartOrderRouter,
    ) -> None:
        ks = KillSwitch(sentinel_path=tmp_path / "KILL")
        # Set a tiny max_order_value so that the custom price (10,000) will trigger it
        limits = RiskLimits(max_order_value=100.0, max_daily_loss=0.0)
        gate = PreTradeRiskGate(limits=limits, fill_tracker=tracker, kill_switch=ks)

        async def buy_evaluator(f):
            return _order(f.symbol, "buy", 5)

        # features.close = 100 → 5×100=$500 > 100 → rejected
        loop = AsyncExecutionLoop(
            buy_evaluator,
            gate,
            router,
            price_fn=lambda sym: 100.0,
        )
        await loop.start()
        await loop.on_features(_warm_features("SPY", close=1.0))  # close=1 would pass
        await asyncio.sleep(0.1)
        await loop.stop()

        s = loop.stats()
        assert s["risk_rejections"] == 1
