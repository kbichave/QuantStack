# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for institutional-grade SimBroker additions.

Covers:
- T1-5: Daily loss limit enforcement
- T1-4: Volume-dependent slippage (Almgren-Chriss fallback path)
- T2-2: Order audit trail completeness
- T2-5: Trade analytics correctness

Design principles:
- No file I/O, no network, no DuckDB
- Deterministic inputs; no LLM
- Test behaviour, not implementation
- Edge cases are more valuable than happy paths
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pytest

from quant_arena.historical.sim_broker import (
    Order,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Position,
    SimBroker,
    Trade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_broker(
    initial_equity: float = 100_000.0,
    slippage_bps: float = 5.0,
    max_position_pct: float = 0.30,
    max_drawdown_halt_pct: float = 0.20,
    max_daily_loss_pct: float = 0.05,
) -> SimBroker:
    """Return a fresh SimBroker with default test settings."""
    return SimBroker(
        initial_equity=initial_equity,
        slippage_bps=slippage_bps,
        commission_per_share=0.0,   # zero commission simplifies P&L assertions
        max_position_pct=max_position_pct,
        max_drawdown_halt_pct=max_drawdown_halt_pct,
        max_daily_loss_pct=max_daily_loss_pct,
    )


def _buy(broker: SimBroker, symbol: str, qty: int, d: date) -> Order:
    broker.update_prices({symbol: 100.0}, d)
    return broker.submit_order(symbol, OrderSide.BUY, qty)


def _sell(broker: SimBroker, symbol: str, qty: int, d: date) -> Order:
    broker.update_prices({symbol: 100.0}, d)
    return broker.submit_order(symbol, OrderSide.SELL, qty)


# ---------------------------------------------------------------------------
# T1-5: Daily loss limit
# ---------------------------------------------------------------------------


class TestDailyLossLimit:
    """Daily loss limit is applied to BUY orders only; SELL orders bypass it."""

    def test_no_halt_within_limit(self):
        """Orders within the daily loss band are accepted."""
        broker = make_broker(max_daily_loss_pct=0.05)
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 100.0}, d0)
        order = broker.submit_order("SPY", OrderSide.BUY, 100)
        assert order.status == OrderStatus.FILLED

    def test_buy_rejected_after_daily_loss(self):
        """A BUY is rejected once the daily P&L loss exceeds the limit."""
        broker = make_broker(
            initial_equity=100_000.0,
            # 1% daily loss limit: 200 shares × ($100→$94) = $1,200 loss = 1.2% > 1%
            max_daily_loss_pct=0.01,
            max_drawdown_halt_pct=0.99,  # keep drawdown halt disabled
        )
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 100.0}, d0)

        # Buy 200 shares (20% of equity — within 30% per-position limit)
        order = broker.submit_order("SPY", OrderSide.BUY, 200)
        assert order.status == OrderStatus.FILLED

        # Price drops 6% intraday on same day → 1.2% equity loss > 1% limit
        broker.update_prices({"SPY": 94.0}, d0)

        # Second BUY should be rejected
        order2 = broker.submit_order("SPY", OrderSide.BUY, 10)
        assert order2.status == OrderStatus.REJECTED
        assert "Daily loss limit" in order2.rejection_reason

    def test_sell_allowed_after_daily_loss(self):
        """SELL orders bypass the daily loss limit so positions can be reduced."""
        broker = make_broker(
            initial_equity=100_000.0,
            # 1% limit: 200 shares × 6% drop = 1.2% equity loss → limit triggered
            max_daily_loss_pct=0.01,
            max_drawdown_halt_pct=0.99,
        )
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 100.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 200)

        # Price drops 6% — daily limit (1%) breached (1.2% loss > 1%)
        broker.update_prices({"SPY": 94.0}, d0)

        # Confirm the limit IS triggered (so we're testing the right thing)
        buy_blocked = broker.submit_order("SPY", OrderSide.BUY, 1)
        assert buy_blocked.status == OrderStatus.REJECTED

        # SELL should still go through despite limit
        sell_order = broker.submit_order("SPY", OrderSide.SELL, 200)
        assert sell_order.status == OrderStatus.FILLED

    def test_daily_limit_resets_on_new_day(self):
        """The daily loss baseline resets at the start of each new trading day."""
        broker = make_broker(
            initial_equity=100_000.0,
            # 1% limit: 200 shares × 6% drop = 1.2% equity loss → triggers
            max_daily_loss_pct=0.01,
            max_drawdown_halt_pct=0.99,
        )
        d0 = date(2024, 1, 2)
        d1 = date(2024, 1, 3)

        # Day 1: buy then see 6% drop → 1.2% daily loss > 1% limit triggered
        broker.update_prices({"SPY": 100.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 200)
        broker.update_prices({"SPY": 94.0}, d0)
        rejected = broker.submit_order("SPY", OrderSide.BUY, 10)
        assert rejected.status == OrderStatus.REJECTED

        # Day 2: new day resets baseline; small buy should pass
        # Close yesterday's position first
        broker.submit_order("SPY", OrderSide.SELL, 200)
        broker.update_prices({"SPY": 94.0}, d1)  # Same price, new date
        ok = broker.submit_order("SPY", OrderSide.BUY, 10)
        assert ok.status == OrderStatus.FILLED

    def test_day_open_equity_snapshot_before_price_update(self):
        """Day-open equity is captured before position prices are updated."""
        # Use slippage_bps=0 so fill price = market price → exact arithmetic
        broker = make_broker(initial_equity=50_000.0, slippage_bps=0.0)
        d0 = date(2024, 1, 2)

        broker.update_prices({"SPY": 100.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 100)

        # Position: 100 shares × $100 = $10k; cash = $40k; equity = $50k
        assert broker._day_open_equity == 50_000.0

        # Same day, price rises — open equity should not change
        broker.update_prices({"SPY": 110.0}, d0)
        assert broker._day_open_equity == 50_000.0

        # New day — open equity should capture current equity at d0 close price ($110)
        d1 = date(2024, 1, 3)
        broker.update_prices({"SPY": 110.0}, d1)
        # equity = cash ($40,000) + 100×$110 = $51,000 (exact because slippage=0)
        assert broker._day_open_equity == pytest.approx(51_000.0)


# ---------------------------------------------------------------------------
# T1-4: Volume-dependent slippage
# ---------------------------------------------------------------------------


class TestVolumeSlippage:
    """Volume / volatility context is stored and used in slippage computation."""

    def test_volumes_stored_after_update(self):
        broker = make_broker()
        d0 = date(2024, 1, 2)
        broker.update_prices(
            {"SPY": 400.0},
            d0,
            volumes={"SPY": 80_000_000.0},
            volatilities={"SPY": 0.15},
        )
        assert broker._volumes["SPY"] == pytest.approx(80_000_000.0)
        assert broker._volatilities["SPY"] == pytest.approx(0.15)

    def test_zero_volume_falls_back_to_flat_bps(self):
        """When volume is 0, slippage defaults to the configured flat basis points."""
        broker = make_broker(slippage_bps=10.0)
        d0 = date(2024, 1, 2)
        # No volumes provided — _volumes is empty
        broker.update_prices({"SPY": 400.0}, d0)
        order = broker.submit_order("SPY", OrderSide.BUY, 10)
        assert order.status == OrderStatus.FILLED
        # Fill price should be close to 400 * (1 + 10bps) = 400.40
        assert order.fill_price == pytest.approx(400.0 * (1 + 10 / 10_000), rel=1e-3)

    def test_high_volume_lowers_effective_bps(self):
        """A tiny order relative to huge volume should incur near-zero market impact."""
        # With participation rate ~ 1e-6 (1 share vs 1M volume), Almgren-Chriss
        # sqrt impact approaches zero; total slippage ≈ spread floor only.
        broker = make_broker(slippage_bps=0.0)  # zero spread floor
        d0 = date(2024, 1, 2)
        broker.update_prices(
            {"SPY": 400.0},
            d0,
            volumes={"SPY": 1_000_000.0},
            volatilities={"SPY": 0.01},
        )
        order = broker.submit_order("SPY", OrderSide.BUY, 1)
        assert order.status == OrderStatus.FILLED
        # Fill should be very close to 400 — impact is negligible
        assert order.fill_price == pytest.approx(400.0, rel=0.01)

    def test_volumes_persist_across_days(self):
        """Volume context from a previous day is still available until overwritten."""
        broker = make_broker()
        d0 = date(2024, 1, 2)
        d1 = date(2024, 1, 3)
        broker.update_prices({"SPY": 100.0}, d0, volumes={"SPY": 50_000.0})
        # New day with no volume update — volume should persist
        broker.update_prices({"SPY": 101.0}, d1)
        assert broker._volumes.get("SPY") == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# T2-2: Order audit trail
# ---------------------------------------------------------------------------


class TestOrderAuditTrail:
    """Every order attempt — filled or rejected — is recorded in the audit trail."""

    def test_filled_order_appears_in_audit(self):
        broker = make_broker()
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 100.0}, d0)
        order = broker.submit_order("SPY", OrderSide.BUY, 50)

        audit = broker.get_order_audit()
        assert len(audit) == 1
        entry = audit[0]
        assert entry["order_id"] == order.order_id
        assert entry["status"] == "filled"
        assert entry["symbol"] == "SPY"
        assert entry["side"] == "buy"
        assert entry["qty_requested"] == 50
        assert entry["qty_filled"] == 50
        assert entry["rejection_reason"] is None

    def test_rejected_order_appears_in_audit(self):
        broker = make_broker()
        d0 = date(2024, 1, 2)
        # No price update — order will be rejected
        order = broker.submit_order("SPY", OrderSide.BUY, 50)

        audit = broker.get_order_audit()
        assert len(audit) == 1
        assert audit[0]["status"] == "rejected"
        assert audit[0]["rejection_reason"] is not None

    def test_audit_contains_slippage_bps(self):
        """Actual slippage bps is computed from fill vs mid price."""
        broker = make_broker(slippage_bps=5.0)
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 200.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 10)

        audit = broker.get_order_audit()
        assert len(audit) == 1
        # Buy: fill > mid → slippage should be positive (~5 bps)
        assert audit[0]["slippage_bps_actual"] == pytest.approx(5.0, rel=0.01)

    def test_audit_grows_monotonically(self):
        """Audit accumulates across multiple orders."""
        broker = make_broker()
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 100.0}, d0)
        for _ in range(5):
            broker.submit_order("SPY", OrderSide.BUY, 10)

        audit = broker.get_order_audit()
        # 5 fills but position piling up — some may be rejected by position limit
        assert len(audit) == 5

    def test_get_order_audit_returns_copy(self):
        """Mutating the returned list must not affect the internal audit trail."""
        broker = make_broker()
        broker.update_prices({"SPY": 100.0}, date(2024, 1, 2))
        broker.submit_order("SPY", OrderSide.BUY, 10)

        audit = broker.get_order_audit()
        audit.clear()
        assert len(broker.get_order_audit()) == 1


# ---------------------------------------------------------------------------
# T2-5: Trade analytics
# ---------------------------------------------------------------------------


class TestTradeAnalytics:
    """get_trade_analytics() returns correct metrics for known trade sequences."""

    def _make_broker_with_trades(self) -> SimBroker:
        """Build a broker with 3 profitable and 2 losing round-trips."""
        broker = make_broker()
        d0 = date(2024, 1, 2)

        # Buy 10 SPY @ 100, sell @ 110 → +$100 (win)
        broker.update_prices({"SPY": 100.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 10)
        broker.update_prices({"SPY": 110.0}, d0 + timedelta(days=1))
        broker.submit_order("SPY", OrderSide.SELL, 10)

        # Buy 10 @ 110, sell @ 100 → -$100 (loss)
        d1 = d0 + timedelta(days=2)
        broker.update_prices({"SPY": 110.0}, d1)
        broker.submit_order("SPY", OrderSide.BUY, 10)
        broker.update_prices({"SPY": 100.0}, d1 + timedelta(days=1))
        broker.submit_order("SPY", OrderSide.SELL, 10)

        # Win: buy @ 100, sell @ 120 → +$200
        d2 = d0 + timedelta(days=5)
        broker.update_prices({"SPY": 100.0}, d2)
        broker.submit_order("SPY", OrderSide.BUY, 10)
        broker.update_prices({"SPY": 120.0}, d2 + timedelta(days=1))
        broker.submit_order("SPY", OrderSide.SELL, 10)

        return broker

    def test_analytics_available(self):
        broker = self._make_broker_with_trades()
        analytics = broker.get_trade_analytics()
        assert "error" not in analytics

    def test_total_closed_trades(self):
        broker = self._make_broker_with_trades()
        analytics = broker.get_trade_analytics()
        # 3 round-trips → 3 closed trades (sells with pnl)
        assert analytics["total_closed_trades"] == 3

    def test_win_rate(self):
        broker = self._make_broker_with_trades()
        analytics = broker.get_trade_analytics()
        assert analytics["win_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_expectancy_positive(self):
        """With 2 wins and 1 loss, and winners > losers, expectancy must be positive."""
        broker = self._make_broker_with_trades()
        analytics = broker.get_trade_analytics()
        assert analytics["expectancy"] > 0

    def test_profit_factor(self):
        """Profit factor = gross_profit / gross_loss."""
        broker = self._make_broker_with_trades()
        analytics = broker.get_trade_analytics()
        gross_profit = analytics["gross_profit"]
        gross_loss = analytics["gross_loss"]
        expected_pf = gross_profit / gross_loss if gross_loss > 0 else None
        if expected_pf is not None:
            assert analytics["profit_factor"] == pytest.approx(expected_pf, rel=1e-3)

    def test_by_symbol_breakdown(self):
        broker = self._make_broker_with_trades()
        analytics = broker.get_trade_analytics()
        assert "SPY" in analytics["by_symbol"]
        assert analytics["by_symbol"]["SPY"]["trades"] == 3

    def test_no_closed_trades_returns_error(self):
        broker = make_broker()
        analytics = broker.get_trade_analytics()
        assert "error" in analytics

    def test_max_consecutive_wins_losses(self):
        """Two back-to-back wins then a loss → max_consecutive_wins=2."""
        broker = make_broker()
        d = date(2024, 1, 2)
        # win
        broker.update_prices({"SPY": 100.0}, d)
        broker.submit_order("SPY", OrderSide.BUY, 10)
        broker.update_prices({"SPY": 110.0}, d + timedelta(1))
        broker.submit_order("SPY", OrderSide.SELL, 10)
        # win
        broker.update_prices({"SPY": 110.0}, d + timedelta(2))
        broker.submit_order("SPY", OrderSide.BUY, 10)
        broker.update_prices({"SPY": 120.0}, d + timedelta(3))
        broker.submit_order("SPY", OrderSide.SELL, 10)
        # loss
        broker.update_prices({"SPY": 120.0}, d + timedelta(4))
        broker.submit_order("SPY", OrderSide.BUY, 10)
        broker.update_prices({"SPY": 110.0}, d + timedelta(5))
        broker.submit_order("SPY", OrderSide.SELL, 10)

        analytics = broker.get_trade_analytics()
        assert analytics["max_consecutive_wins"] == 2
        assert analytics["max_consecutive_losses"] == 1


# ---------------------------------------------------------------------------
# Cash conservation invariant
# ---------------------------------------------------------------------------


class TestCashConservation:
    """cash + positions_market_value == equity must hold after every trade."""

    def _assert_conservation(self, broker: SimBroker) -> None:
        positions_value = sum(p.market_value for p in broker.get_positions().values())
        equity = broker.get_equity()
        assert broker.cash + positions_value == pytest.approx(equity, rel=1e-9)

    def test_conservation_after_buy(self):
        broker = make_broker()
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 200.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 50)
        self._assert_conservation(broker)

    def test_conservation_after_sell(self):
        broker = make_broker()
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 200.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 50)
        broker.update_prices({"SPY": 210.0}, d0 + timedelta(1))
        broker.submit_order("SPY", OrderSide.SELL, 50)
        self._assert_conservation(broker)

    def test_conservation_after_rejection(self):
        """Rejected orders must leave equity unchanged."""
        broker = make_broker()
        equity_before = broker.get_equity()
        # Rejected: no price set
        broker.submit_order("SPY", OrderSide.BUY, 100)
        assert broker.get_equity() == pytest.approx(equity_before)
        self._assert_conservation(broker)

    def test_conservation_across_multiple_symbols(self):
        broker = make_broker(max_position_pct=0.40)
        d0 = date(2024, 1, 2)
        broker.update_prices({"SPY": 100.0, "QQQ": 300.0}, d0)
        broker.submit_order("SPY", OrderSide.BUY, 100)
        broker.submit_order("QQQ", OrderSide.BUY, 20)
        broker.update_prices({"SPY": 105.0, "QQQ": 310.0}, d0 + timedelta(1))
        self._assert_conservation(broker)
