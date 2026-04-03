"""Tests for enhanced TCA: spread decomposition and time-of-day effects."""

from __future__ import annotations

import pytest

from quantstack.core.execution.tca_engine import OrderSide, TradeRecord, post_trade_tca


def test_arrival_shortfall_buy_positive():
    """BUY fill at $100.05 with arrival $100.00 -> shortfall = +5.0 bps."""
    record = TradeRecord(
        trade_id="t1", symbol="AAPL", side=OrderSide.BUY,
        shares=100, arrival_price=100.00, fill_price=100.05,
    )
    result = post_trade_tca(record)
    assert abs(result.shortfall_vs_arrival_bps - 5.0) < 0.1


def test_arrival_shortfall_sell_positive():
    """SELL fill at $99.95 with arrival $100.00 -> shortfall = +5.0 bps (cost)."""
    record = TradeRecord(
        trade_id="t2", symbol="AAPL", side=OrderSide.SELL,
        shares=100, arrival_price=100.00, fill_price=99.95,
    )
    result = post_trade_tca(record)
    assert abs(result.shortfall_vs_arrival_bps - 5.0) < 0.1


def test_spread_cost_from_bid_ask():
    """spread_cost_bps computed from bid-ask at arrival."""
    record = TradeRecord(
        trade_id="t3", symbol="AAPL", side=OrderSide.BUY,
        shares=100, arrival_price=100.00, fill_price=100.00,
        bid_at_arrival=99.98, ask_at_arrival=100.02,
    )
    result = post_trade_tca(record)
    # Half spread = (100.02 - 99.98) / 100.00 * 10000 / 2 = 2.0 bps
    assert result.spread_cost_bps is not None
    assert abs(result.spread_cost_bps - 2.0) < 0.1


def test_time_of_day_effect():
    """time_of_day_effect = shortfall - window_avg_shortfall."""
    record = TradeRecord(
        trade_id="t4", symbol="AAPL", side=OrderSide.BUY,
        shares=100, arrival_price=100.00, fill_price=100.08,
    )
    result = post_trade_tca(record, window_avg_shortfall_bps=3.0)
    # Shortfall ~8 bps, window avg 3 bps -> effect ~5 bps
    assert result.time_of_day_effect_bps is not None
    assert abs(result.time_of_day_effect_bps - 5.0) < 0.1


def test_missing_arrival_price_fallback():
    """When arrival_price is 0, use prev_close as fallback."""
    record = TradeRecord(
        trade_id="t5", symbol="AAPL", side=OrderSide.BUY,
        shares=100, arrival_price=0.0, fill_price=100.05,
        prev_close=100.00,
    )
    result = post_trade_tca(record)
    # Should use prev_close as benchmark
    assert abs(result.shortfall_vs_arrival_bps - 5.0) < 0.1
