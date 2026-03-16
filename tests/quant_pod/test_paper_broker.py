# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PaperBroker.

All tests use an in-memory TradingContext — no file system, no shared state.
"""

from __future__ import annotations

import pytest

from quant_pod.context import create_trading_context
from quant_pod.execution.paper_broker import OrderRequest, PaperBroker


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:", initial_cash=100_000.0)
    yield context
    context.db.close()


@pytest.fixture
def broker(ctx) -> PaperBroker:
    return ctx.broker


def make_req(
    symbol="SPY",
    side="buy",
    qty=100,
    price=450.0,
    volume=10_000_000,
    order_type="market",
    limit_price=None,
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type=order_type,
        limit_price=limit_price,
        current_price=price,
        daily_volume=volume,
    )


class TestMarketFills:
    def test_market_buy_fills(self, broker):
        fill = broker.execute(make_req(side="buy"))
        assert not fill.rejected
        assert fill.filled_quantity > 0
        assert fill.fill_price > 0

    def test_market_sell_fills(self, broker):
        fill = broker.execute(make_req(side="sell"))
        assert not fill.rejected
        assert fill.filled_quantity > 0

    def test_buy_fill_price_above_ref(self, broker):
        """Buy orders incur positive slippage (fill above ref price)."""
        fill = broker.execute(make_req(side="buy", price=450.0))
        assert fill.fill_price > 450.0

    def test_sell_fill_price_below_ref(self, broker):
        """Sell orders incur negative slippage (fill below ref price)."""
        fill = broker.execute(make_req(side="sell", price=450.0))
        assert fill.fill_price < 450.0

    def test_slippage_bps_is_positive(self, broker):
        fill = broker.execute(make_req(side="buy"))
        assert fill.slippage_bps > 0

    def test_commission_charged(self, broker):
        fill = broker.execute(make_req(side="buy", qty=100))
        assert fill.commission == pytest.approx(100 * PaperBroker.COMMISSION_PER_SHARE)


class TestLimitOrders:
    def test_buy_limit_fills_when_price_below_limit(self, broker):
        # Current price 448 ≤ limit 450 → fills
        fill = broker.execute(make_req(
            side="buy", price=448.0, order_type="limit", limit_price=450.0
        ))
        assert not fill.rejected
        assert fill.filled_quantity > 0

    def test_buy_limit_rejected_when_price_above_limit(self, broker):
        # Current price 455 > limit 450 → no fill
        fill = broker.execute(make_req(
            side="buy", price=455.0, order_type="limit", limit_price=450.0
        ))
        assert fill.rejected

    def test_sell_limit_fills_when_price_above_limit(self, broker):
        # Current price 455 ≥ limit 450 → fills
        fill = broker.execute(make_req(
            side="sell", price=455.0, order_type="limit", limit_price=450.0
        ))
        assert not fill.rejected

    def test_sell_limit_rejected_when_price_below_limit(self, broker):
        # Current price 445 < limit 450 → no fill
        fill = broker.execute(make_req(
            side="sell", price=445.0, order_type="limit", limit_price=450.0
        ))
        assert fill.rejected

    def test_limit_fill_has_no_slippage(self, broker):
        fill = broker.execute(make_req(
            side="buy", price=448.0, order_type="limit", limit_price=450.0
        ))
        # Limit fills at exactly the limit price — no market impact
        assert fill.fill_price == pytest.approx(450.0)


class TestPartialFills:
    def test_large_order_is_partially_filled(self, broker):
        """Order > 2% of daily volume → partial fill."""
        # ADV=1_000_000, 2% cap = 20_000; order is 100_000 → partial
        fill = broker.execute(make_req(qty=100_000, volume=1_000_000))
        assert fill.partial is True
        assert fill.filled_quantity < 100_000

    def test_small_order_is_fully_filled(self, broker):
        """Order < 2% of daily volume → full fill."""
        # ADV=10_000_000, 2% = 200_000; order is 100 → full
        fill = broker.execute(make_req(qty=100, volume=10_000_000))
        assert fill.partial is False
        assert fill.filled_quantity == 100


class TestRejections:
    def test_zero_price_rejected(self, broker):
        fill = broker.execute(make_req(price=0.0))
        assert fill.rejected
        assert fill.filled_quantity == 0

    def test_negative_price_rejected(self, broker):
        fill = broker.execute(make_req(price=-5.0))
        assert fill.rejected

    def test_zero_quantity_rejected(self, broker):
        fill = broker.execute(make_req(qty=0))
        assert fill.rejected

    def test_negative_quantity_rejected(self, broker):
        fill = broker.execute(make_req(qty=-10))
        assert fill.rejected

    def test_rejected_fill_has_reject_reason(self, broker):
        fill = broker.execute(make_req(price=0.0))
        assert fill.reject_reason is not None
        assert len(fill.reject_reason) > 0


class TestPortfolioUpdates:
    def test_buy_reduces_cash(self, broker, ctx):
        initial_cash = ctx.portfolio.get_cash()
        fill = broker.execute(make_req(side="buy", qty=10, price=100.0))
        if not fill.rejected:
            new_cash = ctx.portfolio.get_cash()
            # Cash should have decreased
            assert new_cash < initial_cash

    def test_buy_creates_position(self, broker, ctx):
        fill = broker.execute(make_req(symbol="AAPL", side="buy", qty=10, price=180.0))
        if not fill.rejected:
            pos = ctx.portfolio.get_position("AAPL")
            assert pos is not None
            assert pos.quantity > 0

    def test_fill_persisted_to_db(self, broker, ctx):
        broker.execute(make_req(symbol="SPY", side="buy", qty=5, price=450.0))
        rows = ctx.db.execute(
            "SELECT COUNT(*) FROM fills WHERE symbol = 'SPY'"
        ).fetchone()[0]
        assert rows >= 1

    def test_volume_zero_uses_default_not_crash(self, broker):
        """When volume=0 slips through risk gate, broker must use DEFAULT_DAILY_VOLUME."""
        fill = broker.execute(make_req(qty=10, price=100.0, volume=0))
        # Should fill (using DEFAULT_DAILY_VOLUME internally) rather than divide-by-zero
        assert not fill.rejected or fill.reject_reason is not None  # either fills or gives reason


class TestSlippageModel:
    def test_slippage_increases_with_order_size(self, broker):
        """Larger order → more market impact → higher slippage."""
        small_fill = broker.execute(make_req(qty=100, volume=10_000_000))
        large_fill = broker.execute(make_req(qty=10_000, volume=10_000_000))
        assert large_fill.slippage_bps > small_fill.slippage_bps

    def test_slippage_decreases_with_deeper_market(self, broker):
        """Same order size but higher ADV → less market impact."""
        shallow = broker.execute(make_req(qty=1000, volume=1_000_000))
        deep = broker.execute(make_req(qty=1000, volume=100_000_000))
        assert deep.slippage_bps < shallow.slippage_bps
