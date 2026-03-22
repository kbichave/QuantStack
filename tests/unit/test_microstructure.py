# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for microstructure module: OrderBook, MatchingEngine, ImpactModels."""

import numpy as np
import pytest
from quantstack.core.microstructure.impact_models import (
    ImpactModel,
    ImpactParams,
    estimate_kyle_lambda,
    square_root_impact,
)
from quantstack.core.microstructure.matching_engine import (
    MatchingEngine,
)
from quantstack.core.microstructure.order_book import Order, OrderBook, OrderType, Side


class TestOrderBook:
    """Tests for OrderBook class."""

    def test_empty_book(self):
        """Test empty order book."""
        book = OrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None
        assert book.mid_price is None

    def test_add_bid_order(self):
        """Test adding a bid order."""
        book = OrderBook()
        order = Order(order_id=1, side=Side.BID, price=100.0, quantity=10)
        book.add_order(order)

        assert book.best_bid == 100.0
        assert book.best_ask is None
        assert 1 in book.orders

    def test_add_ask_order(self):
        """Test adding an ask order."""
        book = OrderBook()
        order = Order(order_id=1, side=Side.ASK, price=101.0, quantity=10)
        book.add_order(order)

        assert book.best_ask == 101.0
        assert book.best_bid is None
        assert 1 in book.orders

    def test_spread_calculation(self):
        """Test spread calculation."""
        book = OrderBook()
        book.add_order(Order(1, Side.BID, 100.0, 10))
        book.add_order(Order(2, Side.ASK, 101.0, 10))

        assert book.spread == 1.0
        assert book.mid_price == 100.5

    def test_multiple_price_levels(self):
        """Test order book with multiple levels."""
        book = OrderBook()

        # Add bids
        book.add_order(Order(1, Side.BID, 100.0, 10))
        book.add_order(Order(2, Side.BID, 99.0, 20))
        book.add_order(Order(3, Side.BID, 98.0, 30))

        # Add asks
        book.add_order(Order(4, Side.ASK, 101.0, 10))
        book.add_order(Order(5, Side.ASK, 102.0, 20))

        assert book.best_bid == 100.0
        assert book.best_ask == 101.0
        assert len(book.orders) == 5

    def test_cancel_order(self):
        """Test order cancellation."""
        book = OrderBook()
        book.add_order(Order(1, Side.BID, 100.0, 10))
        book.add_order(Order(2, Side.BID, 99.0, 20))

        assert book.cancel_order(1)
        assert 1 not in book.orders
        assert book.best_bid == 99.0

    def test_cancel_nonexistent_order(self):
        """Test canceling non-existent order."""
        book = OrderBook()
        assert not book.cancel_order(999)

    def test_get_depth(self):
        """Test order book depth retrieval."""
        book = OrderBook()

        for i in range(5):
            book.add_order(Order(i, Side.BID, 100 - i, 10 + i))
            book.add_order(Order(i + 10, Side.ASK, 101 + i, 10 + i))

        bid_depth, ask_depth = book.get_depth(3)

        assert len(bid_depth) == 3
        assert len(ask_depth) == 3
        assert bid_depth[0][0] == 100  # Best bid price
        assert ask_depth[0][0] == 101  # Best ask price

    def test_imbalance(self):
        """Test order imbalance calculation."""
        book = OrderBook()

        # More bid volume than ask
        book.add_order(Order(1, Side.BID, 100.0, 100))
        book.add_order(Order(2, Side.ASK, 101.0, 50))

        imbalance = book.get_imbalance(1)
        assert imbalance > 0  # Positive = bid heavy

        # Equal volume
        book2 = OrderBook()
        book2.add_order(Order(1, Side.BID, 100.0, 50))
        book2.add_order(Order(2, Side.ASK, 101.0, 50))
        assert abs(book2.get_imbalance(1)) < 0.01


class TestMatchingEngine:
    """Tests for MatchingEngine class."""

    def test_empty_engine(self):
        """Test empty matching engine."""
        engine = MatchingEngine()
        assert engine.book.best_bid is None
        assert engine.book.best_ask is None

    def test_submit_limit_order_no_match(self):
        """Test limit order that rests in book."""
        engine = MatchingEngine()

        order = Order(1, Side.BID, 100.0, 10)
        report = engine.submit_order(order)

        assert report.status == "resting"
        assert report.total_filled == 0
        assert engine.book.best_bid == 100.0

    def test_market_order_fill(self):
        """Test market order execution."""
        engine = MatchingEngine()

        # Add liquidity
        engine.submit_order(Order(1, Side.ASK, 101.0, 100))

        # Submit market buy
        order = Order(0, Side.BID, 0, 50, OrderType.MARKET)
        report = engine.submit_order(order)

        assert report.status == "filled"
        assert report.total_filled == 50
        assert report.avg_price == 101.0
        assert len(report.fills) == 1

    def test_limit_order_crossing(self):
        """Test limit order that crosses the spread."""
        engine = MatchingEngine()

        # Add resting sell
        engine.submit_order(Order(1, Side.ASK, 100.0, 50))

        # Buy at crossing price
        order = Order(2, Side.BID, 100.0, 30)
        report = engine.submit_order(order)

        assert report.status == "filled"
        assert report.total_filled == 30

    def test_partial_fill(self):
        """Test partial order fill."""
        engine = MatchingEngine()

        # Add limited liquidity
        engine.submit_order(Order(1, Side.ASK, 100.0, 30))

        # Try to buy more
        order = Order(0, Side.BID, 0, 50, OrderType.MARKET)
        report = engine.submit_order(order)

        assert report.status == "partial"
        assert report.total_filled == 30
        assert report.remaining == 20

    def test_price_time_priority(self):
        """Test price-time priority matching."""
        engine = MatchingEngine()

        # Add multiple levels
        engine.submit_order(Order(1, Side.ASK, 101.0, 10))
        engine.submit_order(Order(2, Side.ASK, 100.0, 10))  # Better price
        engine.submit_order(Order(3, Side.ASK, 100.0, 10))  # Same price, later time

        # Market buy should hit best price first
        order = Order(0, Side.BID, 0, 15, OrderType.MARKET)
        report = engine.submit_order(order)

        assert report.total_filled == 15
        # First fill at 100, second at 100, partial
        assert report.avg_price == 100.0

    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        engine = MatchingEngine()

        # Add liquidity at different prices
        engine.submit_order(Order(1, Side.ASK, 100.0, 50))
        engine.submit_order(Order(2, Side.ASK, 101.0, 50))

        # Execute orders
        engine.submit_order(Order(0, Side.BID, 0, 50, OrderType.MARKET))
        engine.submit_order(Order(0, Side.BID, 0, 50, OrderType.MARKET))

        vwap = engine.get_vwap()
        assert vwap == pytest.approx(100.5, rel=0.01)


class TestImpactModels:
    """Tests for market impact models."""

    def test_square_root_impact_buy(self):
        """Test square root impact for buy order."""
        impact = square_root_impact(
            order_size=10000,
            daily_volume=1_000_000,
            volatility=0.02,
        )

        assert impact > 0  # Buy should have positive impact
        assert impact < 0.01  # Should be reasonable

    def test_square_root_impact_sell(self):
        """Test square root impact for sell order."""
        impact = square_root_impact(
            order_size=-10000,
            daily_volume=1_000_000,
            volatility=0.02,
        )

        assert impact < 0  # Sell should have negative impact

    def test_square_root_impact_zero_volume(self):
        """Test impact with zero volume."""
        impact = square_root_impact(
            order_size=10000,
            daily_volume=0,
            volatility=0.02,
        )

        assert impact == 0

    def test_impact_model_estimate(self):
        """Test ImpactModel estimate method."""
        model = ImpactModel(volatility=0.02, daily_volume=1_000_000)

        result = model.estimate(order_size=10000, execution_time=1.0)

        assert "permanent" in result
        assert "temporary" in result
        assert "total" in result
        assert result["total"] == result["permanent"] + result["temporary"]

    def test_impact_model_custom_params(self):
        """Test ImpactModel with custom parameters."""
        params = ImpactParams(eta=0.2, gamma=0.1)
        model = ImpactModel(
            volatility=0.02,
            daily_volume=1_000_000,
            params=params,
        )

        result = model.estimate(10000, 1.0)

        assert result["total"] > 0

    def test_execution_cost(self):
        """Test execution cost calculation."""
        model = ImpactModel(volatility=0.02, daily_volume=1_000_000)

        cost = model.execution_cost(
            order_size=10000,
            execution_time=1.0,
            price=100.0,
        )

        assert cost > 0
        assert cost < 10000  # Should be fraction of order value

    def test_optimal_execution_time(self):
        """Test optimal execution time calculation."""
        model = ImpactModel(volatility=0.02, daily_volume=1_000_000)

        opt_time = model.optimal_execution_time(
            order_size=10000,
            risk_aversion=1e-6,
        )

        assert opt_time >= 0.1
        assert opt_time <= 5.0

    def test_kyle_lambda_estimation(self):
        """Test Kyle's lambda estimation."""
        np.random.seed(42)

        # Generate synthetic data with known relationship
        order_flow = np.random.randn(100) * 1000
        true_lambda = 0.001
        price_changes = true_lambda * order_flow + np.random.randn(100) * 0.01

        estimated_lambda = estimate_kyle_lambda(price_changes, order_flow)

        # Should be roughly correct
        assert estimated_lambda > 0


# ---------------------------------------------------------------------------
# OvernightGapPersistence — volume spike and institutional gap extensions
# ---------------------------------------------------------------------------

import pandas as pd
from quantstack.core.features.microstructure import OvernightGapPersistence


class TestOvernightGapPersistenceVolume:
    @pytest.fixture
    def gap_data(self):
        """50 bars with synthetic gaps and matching volume."""
        np.random.seed(19)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5), index=dates)
        # Open with gap: alternate gaps up and down
        open_ = close.shift(1) * (1 + np.where(np.arange(50) % 5 == 0, 0.015, 0.001))
        open_ = pd.Series(open_, index=dates)
        volume = pd.Series(
            1_000_000 + np.random.randint(0, 200_000, 50).astype(float), index=dates
        )
        return open_, close, volume

    def test_volume_spike_column_present_when_volume_given(self, gap_data):
        open_, close, volume = gap_data
        result = OvernightGapPersistence().compute(open_, close, volume)
        assert "volume_spike" in result.columns

    def test_institutional_gap_column_present_when_volume_given(self, gap_data):
        open_, close, volume = gap_data
        result = OvernightGapPersistence().compute(open_, close, volume)
        assert "institutional_gap" in result.columns

    def test_volume_spike_absent_without_volume(self, gap_data):
        open_, close, _ = gap_data
        result = OvernightGapPersistence().compute(open_, close)
        assert "volume_spike" not in result.columns
        assert "institutional_gap" not in result.columns

    def test_volume_spike_binary(self, gap_data):
        open_, close, volume = gap_data
        result = OvernightGapPersistence().compute(open_, close, volume)
        assert set(result["volume_spike"].unique()).issubset({0, 1})

    def test_institutional_gap_binary(self, gap_data):
        open_, close, volume = gap_data
        result = OvernightGapPersistence().compute(open_, close, volume)
        assert set(result["institutional_gap"].unique()).issubset({0, 1})

    def test_institutional_gap_implies_gap_persisted(self, gap_data):
        """institutional_gap = 1 should only occur when gap_persisted = 1."""
        open_, close, volume = gap_data
        result = OvernightGapPersistence().compute(open_, close, volume)
        inst = result[result["institutional_gap"] == 1]
        assert (inst["gap_persisted"] == 1).all()

    def test_volume_spike_fires_on_high_volume(self):
        """Construct explicit high-volume bar and assert spike fires."""
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        close = pd.Series([100.0] * 25, index=dates)
        open_ = pd.Series([100.5] * 25, index=dates)  # gap up every bar
        # Normal volume = 1M; spike on bar 22 = 3M (> 2× 20-bar avg)
        vol = [1_000_000.0] * 25
        vol[22] = 3_000_000.0
        volume = pd.Series(vol, index=dates)
        result = OvernightGapPersistence(volume_spike_mult=2.0).compute(
            open_, close, volume
        )
        assert result["volume_spike"].iloc[22] == 1

    def test_existing_columns_unchanged_with_volume(self, gap_data):
        """Adding volume must not change gap_pct, gap_up, gap_down, etc."""
        open_, close, volume = gap_data
        without = OvernightGapPersistence().compute(open_, close)
        with_vol = OvernightGapPersistence().compute(open_, close, volume)
        for col in ("gap_pct", "gap_up", "gap_down", "gap_filled", "gap_persisted"):
            pd.testing.assert_series_equal(
                without[col], with_vol[col], check_names=False
            )
