# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for realistic backtesting engine with order book simulation."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantcore.backtesting.realistic_engine import (
    RealisticBacktestConfig,
    RealisticBacktestEngine,
    RealisticBacktestResult,
    OrderBookSnapshot,
    FillRecord,
)


@pytest.fixture
def sample_signals():
    """Create sample signals DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    # Generate signals: 1 = enter long, -1 = exit, 0 = hold
    signals = np.zeros(100)
    signals[10] = 1  # Enter long
    signals[30] = -1  # Exit
    signals[50] = 1  # Enter long again
    signals[80] = -1  # Exit

    directions = ["NONE"] * 100
    directions[10] = "LONG"
    directions[50] = "LONG"

    return pd.DataFrame(
        {
            "signal": signals,
            "signal_direction": directions,
        },
        index=dates,
    )


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    # Generate random walk prices
    returns = np.random.randn(100) * 0.02
    prices = 100 * np.exp(returns.cumsum())

    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(10000, 100000, 100),
        },
        index=dates,
    )


class TestRealisticBacktestConfig:
    """Tests for RealisticBacktestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RealisticBacktestConfig()

        assert config.initial_capital == 100_000.0
        assert config.execution_algo == "market"
        assert config.log_order_book == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RealisticBacktestConfig(
            initial_capital=500_000.0,
            execution_algo="twap",
            volatility=0.03,
        )

        assert config.initial_capital == 500_000.0
        assert config.execution_algo == "twap"
        assert config.volatility == 0.03


class TestRealisticBacktestResult:
    """Tests for RealisticBacktestResult."""

    def test_result_to_dict(self):
        """Test result serialization."""
        result = RealisticBacktestResult(
            initial_capital=100_000,
            final_capital=105_000,
            total_return=5.0,
            sharpe_ratio=1.5,
            max_drawdown=2.0,
            total_trades=10,
            total_volume=1000,
            avg_slippage_bps=2.5,
            avg_impact_bps=1.5,
            total_execution_cost=100,
            implementation_shortfall=4.0,
            win_rate=60,
            profit_factor=1.8,
            avg_trade_pnl=500,
        )

        data = result.to_dict()

        assert "performance" in data
        assert "execution_quality" in data
        assert "trade_stats" in data
        assert data["performance"]["total_return"] == 5.0


class TestRealisticBacktestEngine:
    """Tests for RealisticBacktestEngine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = RealisticBacktestEngine()

        assert engine.capital == 100_000.0
        assert engine.position == 0
        assert len(engine.trades) == 0

    def test_run_backtest(self, sample_signals, sample_prices):
        """Test running a backtest."""
        engine = RealisticBacktestEngine()

        result = engine.run(sample_signals, sample_prices)

        assert isinstance(result, RealisticBacktestResult)
        assert result.initial_capital == 100_000.0

    def test_order_book_logging(self, sample_signals, sample_prices):
        """Test order book states are logged."""
        config = RealisticBacktestConfig(log_order_book=True)
        engine = RealisticBacktestEngine(config)

        result = engine.run(sample_signals, sample_prices)

        assert len(result.order_book_logs) > 0

    def test_fill_logging(self, sample_signals, sample_prices):
        """Test fills are logged."""
        config = RealisticBacktestConfig(log_fills=True)
        engine = RealisticBacktestEngine(config)

        result = engine.run(sample_signals, sample_prices)

        # Should have fills if any trades occurred
        if result.total_trades > 0:
            assert len(result.fill_logs) > 0

    def test_equity_curve(self, sample_signals, sample_prices):
        """Test equity curve is generated."""
        engine = RealisticBacktestEngine()

        result = engine.run(sample_signals, sample_prices)

        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 100_000.0

    def test_slippage_calculation(self, sample_signals, sample_prices):
        """Test slippage is calculated."""
        engine = RealisticBacktestEngine()

        result = engine.run(sample_signals, sample_prices)

        # Slippage should be non-negative
        assert result.avg_slippage_bps >= 0

    def test_impact_calculation(self, sample_signals, sample_prices):
        """Test market impact is calculated."""
        engine = RealisticBacktestEngine()

        result = engine.run(sample_signals, sample_prices)

        # Impact should be non-negative
        assert result.avg_impact_bps >= 0

    def test_execution_algo_market(self, sample_signals, sample_prices):
        """Test market order execution."""
        config = RealisticBacktestConfig(execution_algo="market")
        engine = RealisticBacktestEngine(config)

        result = engine.run(sample_signals, sample_prices)

        assert isinstance(result, RealisticBacktestResult)

    def test_execution_algo_twap(self, sample_signals, sample_prices):
        """Test TWAP execution."""
        config = RealisticBacktestConfig(
            execution_algo="twap",
            execution_horizon=5,
        )
        engine = RealisticBacktestEngine(config)

        result = engine.run(sample_signals, sample_prices)

        assert isinstance(result, RealisticBacktestResult)

    def test_save_logs(self, sample_signals, sample_prices):
        """Test audit log saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RealisticBacktestConfig(log_path=tmpdir)
            engine = RealisticBacktestEngine(config)

            engine.run(sample_signals, sample_prices)

            log_path = Path(tmpdir) / "test_log.json"
            engine.save_logs(str(log_path))

            assert log_path.exists()

    def test_execution_quality_report(self, sample_signals, sample_prices):
        """Test execution quality report generation."""
        engine = RealisticBacktestEngine()
        engine.run(sample_signals, sample_prices)

        report = engine.get_execution_quality_report()

        assert isinstance(report, str)
        assert "EXECUTION QUALITY REPORT" in report

    def test_empty_signals(self, sample_prices):
        """Test with no signals."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        signals = pd.DataFrame(
            {
                "signal": np.zeros(100),
                "signal_direction": ["NONE"] * 100,
            },
            index=dates,
        )

        engine = RealisticBacktestEngine()
        result = engine.run(signals, sample_prices)

        assert result.total_trades == 0
        assert result.final_capital == result.initial_capital

    def test_no_common_index(self):
        """Test with no overlapping indices."""
        dates1 = pd.date_range("2020-01-01", periods=50, freq="D")
        dates2 = pd.date_range("2021-01-01", periods=50, freq="D")

        signals = pd.DataFrame({"signal": np.zeros(50)}, index=dates1)
        prices = pd.DataFrame({"close": np.ones(50) * 100}, index=dates2)

        engine = RealisticBacktestEngine()
        result = engine.run(signals, prices)

        # Should handle gracefully
        assert result.total_trades == 0


class TestOrderBookSnapshot:
    """Tests for OrderBookSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            mid_price=100.0,
            spread=0.02,
            best_bid=99.99,
            best_ask=100.01,
            bid_depth=[(99.99, 100, 5), (99.98, 200, 10)],
            ask_depth=[(100.01, 100, 5), (100.02, 200, 10)],
            imbalance=0.1,
        )

        assert snapshot.mid_price == 100.0
        assert snapshot.spread == 0.02


class TestFillRecord:
    """Tests for FillRecord dataclass."""

    def test_fill_creation(self):
        """Test fill record creation."""
        fill = FillRecord(
            timestamp=datetime.now(),
            order_id=1,
            side="BID",
            price=100.0,
            quantity=10,
            impact=1.5,
            slippage=0.5,
            arrival_price=99.95,
        )

        assert fill.price == 100.0
        assert fill.quantity == 10
