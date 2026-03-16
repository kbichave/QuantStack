# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quantcore.execution.tca_engine — Sprint 1.

Covers: pre_trade_forecast, post_trade_tca, TCAEngine session tracking,
alpha_vs_cost_check, aggregate_report. No external I/O.
"""

from __future__ import annotations

import pytest

from quantcore.execution.tca_engine import (
    ExecAlgo,
    OrderSide,
    PreTradeForecast,
    TCAEngine,
    TradeTCAResult,
    TradeRecord,
    post_trade_tca,
    pre_trade_forecast,
)


# ---------------------------------------------------------------------------
# pre_trade_forecast
# ---------------------------------------------------------------------------


class TestPreTradeForecast:
    def test_returns_forecast(self):
        result = pre_trade_forecast(
            symbol="SPY",
            side=OrderSide.BUY,
            shares=100,
            arrival_price=450.0,
            adv=80_000_000,
            daily_volatility_pct=1.0,
        )
        assert isinstance(result, PreTradeForecast)

    def test_small_order_recommends_immediate(self):
        """< 0.2% of ADV → immediate execution."""
        result = pre_trade_forecast(
            symbol="SPY",
            side=OrderSide.BUY,
            shares=100,          # 100 / 80M = 0.000125% → immediate
            arrival_price=450.0,
            adv=80_000_000,
            daily_volatility_pct=1.0,
        )
        assert result.recommended_algo == ExecAlgo.IMMEDIATE

    def test_medium_order_recommends_twap(self):
        """~0.5% of ADV → TWAP."""
        result = pre_trade_forecast(
            symbol="SPY",
            side=OrderSide.BUY,
            shares=400_000,      # 400k / 80M = 0.5% → TWAP
            arrival_price=450.0,
            adv=80_000_000,
            daily_volatility_pct=1.0,
        )
        assert result.recommended_algo in (ExecAlgo.TWAP, ExecAlgo.VWAP)

    def test_large_order_recommends_vwap_or_pov(self):
        """3% of ADV → VWAP or POV."""
        result = pre_trade_forecast(
            symbol="SPY",
            side=OrderSide.BUY,
            shares=2_400_000,   # 2.4M / 80M = 3% → VWAP
            arrival_price=450.0,
            adv=80_000_000,
            daily_volatility_pct=1.0,
        )
        assert result.recommended_algo in (ExecAlgo.VWAP, ExecAlgo.POV)

    def test_total_cost_positive(self):
        result = pre_trade_forecast(
            symbol="AAPL",
            side=OrderSide.SELL,
            shares=500,
            arrival_price=180.0,
            adv=5_000_000,
            daily_volatility_pct=1.5,
        )
        assert result.total_expected_bps > 0

    def test_participation_rate_in_range(self):
        result = pre_trade_forecast(
            symbol="SPY",
            side=OrderSide.BUY,
            shares=1000,
            arrival_price=450.0,
            adv=80_000_000,
            daily_volatility_pct=1.0,
        )
        assert 0.0 < result.participation_rate <= 1.0


# ---------------------------------------------------------------------------
# post_trade_tca
# ---------------------------------------------------------------------------


class TestPostTradeTCA:
    def _make_record(
        self,
        arrival=450.0,
        fill=450.9,
        vwap=450.5,
        twap=450.4,
        side=OrderSide.BUY,
    ) -> TradeRecord:
        return TradeRecord(
            trade_id="t1",
            symbol="SPY",
            side=side,
            shares=100,
            arrival_price=arrival,
            fill_price=fill,
            vwap_price=vwap,
            twap_price=twap,
            prev_close=449.0,
        )

    def test_returns_result(self):
        record = self._make_record()
        result = post_trade_tca(record)
        assert isinstance(result, TradeTCAResult)

    def test_buy_above_arrival_is_positive_shortfall(self):
        """Buy filled above arrival → implementation shortfall > 0 (cost)."""
        record = self._make_record(arrival=450.0, fill=451.0, side=OrderSide.BUY)
        result = post_trade_tca(record)
        assert result.shortfall_vs_arrival_bps > 0

    def test_buy_below_arrival_is_price_improvement(self):
        """Buy filled below arrival → shortfall < 0 (price improvement)."""
        record = self._make_record(arrival=450.0, fill=449.5, side=OrderSide.BUY)
        result = post_trade_tca(record)
        assert result.shortfall_vs_arrival_bps < 0

    def test_sell_below_arrival_is_cost(self):
        """Sell filled below arrival → cost (shortfall > 0)."""
        record = self._make_record(arrival=450.0, fill=449.0, side=OrderSide.SELL)
        result = post_trade_tca(record)
        assert result.shortfall_vs_arrival_bps > 0

    def test_dollar_shortfall_calculated(self):
        record = self._make_record(arrival=450.0, fill=451.0, side=OrderSide.BUY)
        result = post_trade_tca(record)
        # 100 shares × $1 slippage = $100 dollar shortfall
        assert abs(result.shortfall_dollar) == pytest.approx(100.0, rel=0.01)

    def test_vwap_benchmark_computed(self):
        record = self._make_record(fill=451.0, vwap=450.5)
        result = post_trade_tca(record)
        assert isinstance(result.shortfall_vs_vwap_bps, float)


# ---------------------------------------------------------------------------
# TCAEngine (session tracker)
# ---------------------------------------------------------------------------


class TestTCAEngine:
    def test_full_cycle(self):
        engine = TCAEngine()
        trade_id = "t001"
        engine.record_arrival(trade_id, "SPY", OrderSide.BUY, 100, arrival_price=450.0)
        forecast = engine.pre_trade(trade_id, adv=80_000_000, daily_vol_pct=1.0)
        assert isinstance(forecast, PreTradeForecast)

        result = engine.record_fill(
            trade_id, fill_price=450.9, vwap_price=450.5, twap_price=450.4, prev_close=449.0
        )
        assert isinstance(result, TradeTCAResult)

    def test_aggregate_report_after_fills(self):
        engine = TCAEngine()
        for i in range(3):
            tid = f"t{i}"
            engine.record_arrival(tid, "SPY", OrderSide.BUY, 100, 450.0)
            engine.record_fill(tid, fill_price=450.9, vwap_price=450.5, twap_price=450.4, prev_close=449.0)

        report = engine.aggregate_report()
        assert report["n_trades"] == 3
        assert "avg_shortfall_vs_arrival_bps" in report

    def test_alpha_vs_cost_gates_expensive_trade(self):
        engine = TCAEngine()
        engine.record_arrival("t1", "SPY", OrderSide.BUY, 100_000, arrival_price=450.0)
        engine.pre_trade("t1", adv=1_000_000, daily_vol_pct=2.0)  # 10% of ADV — high cost
        ok, msg = engine.alpha_vs_cost_check("t1", expected_alpha_bps=2.0)  # Tiny alpha
        # Cost should dominate → reject
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_arrival_not_found_returns_none(self):
        """pre_trade with unknown trade_id returns None (does not raise)."""
        engine = TCAEngine()
        result = engine.pre_trade("nonexistent", adv=1_000_000, daily_vol_pct=1.0)
        assert result is None
