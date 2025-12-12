# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for enhanced paper trading module."""

import pytest
from pathlib import Path
from datetime import datetime

import pandas as pd

from quantcore.execution.paper_trading_enhanced import (
    EnhancedPaperTradingEngine,
    EnhancedPaperOrder,
    EnhancedPaperPosition,
    OrderBookState,
    ExecutionQualityMetrics,
)


@pytest.fixture
def engine(tmp_path):
    """Create paper trading engine for testing."""
    return EnhancedPaperTradingEngine(
        initial_capital=100_000.0,
        log_dir=str(tmp_path),
        log_book_states=True,
    )


class TestEnhancedPaperOrder:
    """Tests for EnhancedPaperOrder dataclass."""

    def test_order_creation(self):
        """Test order creation."""
        order = EnhancedPaperOrder(
            order_id="test123",
            timestamp=datetime.now(),
            symbol="WTI",
            side="BUY",
            order_type="MARKET",
            quantity=100,
        )

        assert order.order_id == "test123"
        assert order.status == "PENDING"
        assert order.filled_quantity == 0

    def test_to_dict(self):
        """Test order serialization."""
        order = EnhancedPaperOrder(
            order_id="test123",
            timestamp=datetime.now(),
            symbol="WTI",
            side="BUY",
            order_type="MARKET",
            quantity=100,
            signal_strength=0.8,
        )

        result = order.to_dict()

        assert isinstance(result, dict)
        assert result["symbol"] == "WTI"
        assert result["signal_strength"] == 0.8


class TestOrderBookState:
    """Tests for OrderBookState dataclass."""

    def test_state_creation(self):
        """Test order book state creation."""
        state = OrderBookState(
            timestamp=datetime.now(),
            symbol="WTI",
            mid_price=75.50,
            spread=0.02,
            best_bid=75.49,
            best_ask=75.51,
            bid_depth_5=1000,
            ask_depth_5=1200,
            imbalance=-0.1,
        )

        assert state.mid_price == 75.50
        assert state.spread == 0.02


class TestEnhancedPaperTradingEngine:
    """Tests for EnhancedPaperTradingEngine class."""

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.capital == 100_000.0
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0

    def test_submit_market_order(self, engine):
        """Test market order submission."""
        order = engine.submit_order(
            symbol="WTI",
            side="BUY",
            quantity=100,
            current_price=75.0,
            signal_strength=0.9,
        )

        assert order.status in ["FILLED", "PARTIAL"]
        assert order.filled_quantity > 0
        assert len(engine.orders) == 1

    def test_position_created_on_fill(self, engine):
        """Test position creation on order fill."""
        engine.submit_order(
            symbol="WTI",
            side="BUY",
            quantity=100,
            current_price=75.0,
        )

        assert "WTI" in engine.positions
        assert engine.positions["WTI"].side == "LONG"
        assert engine.positions["WTI"].quantity > 0

    def test_position_close(self, engine):
        """Test closing a position."""
        # Open position
        engine.submit_order(
            symbol="WTI",
            side="BUY",
            quantity=100,
            current_price=75.0,
        )

        initial_qty = engine.positions["WTI"].quantity

        # Close position
        engine.submit_order(
            symbol="WTI",
            side="SELL",
            quantity=initial_qty,
            current_price=76.0,  # Profit
        )

        # Position should be closed
        assert "WTI" not in engine.positions or engine.positions["WTI"].quantity == 0

    def test_update_prices(self, engine):
        """Test price update and mark to market."""
        # Open position
        engine.submit_order(
            symbol="WTI",
            side="BUY",
            quantity=100,
            current_price=75.0,
        )

        # Update prices
        engine.update_prices({"WTI": 76.0})

        assert engine.positions["WTI"].current_price == 76.0
        assert engine.positions["WTI"].unrealized_pnl > 0
        assert len(engine.equity_curve) > 0

    def test_execution_quality_metrics(self, engine):
        """Test execution quality metrics."""
        # Execute some orders
        engine.submit_order("WTI", "BUY", 100, 75.0)
        engine.submit_order("WTI", "SELL", 50, 76.0)

        metrics = engine.get_execution_quality_metrics()

        assert isinstance(metrics, ExecutionQualityMetrics)
        assert metrics.total_orders == 2
        assert metrics.total_volume > 0

    def test_execution_quality_report(self, engine):
        """Test execution quality report generation."""
        engine.submit_order("WTI", "BUY", 100, 75.0)

        report = engine.get_execution_quality_report()

        assert isinstance(report, str)
        assert "EXECUTION QUALITY REPORT" in report
        assert "Slippage" in report

    def test_save_audit_log(self, engine):
        """Test audit log saving."""
        engine.submit_order("WTI", "BUY", 100, 75.0)

        filepath = engine.save_audit_log("test_audit.json")

        assert filepath.exists()

    def test_order_book_logging(self, engine):
        """Test order book state logging."""
        engine.submit_order("WTI", "BUY", 100, 75.0)

        # Should have logged book states
        assert len(engine.book_states) > 0

    def test_get_position_summary(self, engine):
        """Test position summary DataFrame."""
        engine.submit_order("WTI", "BUY", 100, 75.0)
        engine.submit_order("BRENT", "BUY", 50, 80.0)

        summary = engine.get_position_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "symbol" in summary.columns

    def test_get_trade_log(self, engine):
        """Test trade log DataFrame."""
        engine.submit_order("WTI", "BUY", 100, 75.0)
        engine.submit_order("WTI", "SELL", 50, 76.0)

        trade_log = engine.get_trade_log()

        assert isinstance(trade_log, pd.DataFrame)
        assert len(trade_log) == 2
        assert "order_id" in trade_log.columns

    def test_slippage_tracking(self, engine):
        """Test slippage is tracked."""
        order = engine.submit_order("WTI", "BUY", 100, 75.0)

        # Slippage should be calculated
        assert order.slippage >= 0

    def test_market_impact_tracking(self, engine):
        """Test market impact is tracked."""
        order = engine.submit_order("WTI", "BUY", 100, 75.0)

        # Impact should be calculated
        assert order.market_impact >= 0


class TestRealisticBacktestIntegration:
    """Integration tests with RealisticBacktestEngine."""

    def test_paper_trading_matches_backtest_structure(self, engine):
        """Test that paper trading produces similar data to backtest."""
        # Execute trades
        engine.submit_order("WTI", "BUY", 100, 75.0)
        engine.update_prices({"WTI": 76.0})
        engine.submit_order("WTI", "SELL", 100, 76.0)

        metrics = engine.get_execution_quality_metrics()

        # Should have same structure as backtest metrics
        assert hasattr(metrics, "total_orders")
        assert hasattr(metrics, "avg_slippage_bps")
        assert hasattr(metrics, "implementation_shortfall_bps")
