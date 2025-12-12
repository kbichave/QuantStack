# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.data.universe module."""

import pandas as pd
import pytest

from quantcore.data.universe import (
    Sector,
    UniverseSymbol,
    UniverseManager,
    INITIAL_LIQUID_UNIVERSE,
)


class TestSectorEnum:
    """Test Sector enum."""

    def test_sector_values(self):
        """Test sector values."""
        assert Sector.TECHNOLOGY.value == "Technology"
        assert Sector.HEALTHCARE.value == "Healthcare"
        assert Sector.ETF.value == "ETF"

    def test_all_sectors_count(self):
        """Test total number of sectors."""
        assert len(Sector) == 12


class TestUniverseSymbol:
    """Test UniverseSymbol dataclass."""

    def test_basic_creation(self):
        """Test creating a universe symbol."""
        symbol = UniverseSymbol(
            symbol="AAPL",
            name="Apple Inc",
            sector=Sector.TECHNOLOGY,
        )

        assert symbol.symbol == "AAPL"
        assert symbol.name == "Apple Inc"
        assert symbol.sector == Sector.TECHNOLOGY
        assert symbol.is_etf is False

    def test_etf_creation(self):
        """Test creating an ETF symbol."""
        symbol = UniverseSymbol(
            symbol="SPY",
            name="SPDR S&P 500",
            sector=Sector.ETF,
            is_etf=True,
        )

        assert symbol.is_etf is True

    def test_is_liquid_no_metrics(self):
        """Test is_liquid when no metrics set."""
        symbol = UniverseSymbol("TEST", "Test", Sector.TECHNOLOGY)
        # With no metrics, should be liquid by default
        assert symbol.is_liquid is True

    def test_is_liquid_low_volume(self):
        """Test is_liquid fails with low volume."""
        symbol = UniverseSymbol(
            "TEST",
            "Test",
            Sector.TECHNOLOGY,
            avg_option_volume=5000,  # Below threshold of 10000
        )
        assert symbol.is_liquid is False

    def test_is_liquid_high_spread(self):
        """Test is_liquid fails with high spread."""
        symbol = UniverseSymbol(
            "TEST",
            "Test",
            Sector.TECHNOLOGY,
            median_spread_pct=0.10,  # Above threshold of 0.05
        )
        assert symbol.is_liquid is False

    def test_is_liquid_low_oi(self):
        """Test is_liquid fails with low open interest."""
        symbol = UniverseSymbol(
            "TEST",
            "Test",
            Sector.TECHNOLOGY,
            avg_oi=500,  # Below threshold of 1000
        )
        assert symbol.is_liquid is False

    def test_is_liquid_passes_all(self):
        """Test is_liquid passes when all metrics good."""
        symbol = UniverseSymbol(
            "TEST",
            "Test",
            Sector.TECHNOLOGY,
            avg_option_volume=50000,
            median_spread_pct=0.02,
            avg_oi=5000,
        )
        assert symbol.is_liquid is True


class TestInitialLiquidUniverse:
    """Test the initial liquid universe."""

    def test_universe_size(self):
        """Test universe has expected size."""
        # Should have ETFs and stocks
        assert len(INITIAL_LIQUID_UNIVERSE) >= 20  # At least 20 symbols

    def test_etf_count(self):
        """Test ETF count."""
        etfs = [s for s in INITIAL_LIQUID_UNIVERSE.values() if s.is_etf]
        assert len(etfs) >= 10  # At least 10 ETFs

    def test_stock_count(self):
        """Test stock count."""
        stocks = [s for s in INITIAL_LIQUID_UNIVERSE.values() if not s.is_etf]
        assert len(stocks) >= 10  # At least 10 stocks

    def test_key_symbols_present(self):
        """Test key symbols are present."""
        assert "SPY" in INITIAL_LIQUID_UNIVERSE
        assert "QQQ" in INITIAL_LIQUID_UNIVERSE
        assert "AAPL" in INITIAL_LIQUID_UNIVERSE
        assert "MSFT" in INITIAL_LIQUID_UNIVERSE
        assert "NVDA" in INITIAL_LIQUID_UNIVERSE


class TestUniverseManager:
    """Test UniverseManager class."""

    @pytest.fixture
    def manager(self) -> UniverseManager:
        """Create universe manager with default universe."""
        return UniverseManager()

    @pytest.fixture
    def custom_manager(self) -> UniverseManager:
        """Create universe manager with custom universe."""
        custom = {
            "TEST1": UniverseSymbol("TEST1", "Test 1", Sector.TECHNOLOGY),
            "TEST2": UniverseSymbol("TEST2", "Test 2", Sector.HEALTHCARE),
            "TEST_ETF": UniverseSymbol("TEST_ETF", "Test ETF", Sector.ETF, is_etf=True),
        }
        return UniverseManager(custom_symbols=custom)

    def test_default_universe(self, manager):
        """Test manager loads default universe."""
        assert len(manager) == len(INITIAL_LIQUID_UNIVERSE)

    def test_custom_universe(self, custom_manager):
        """Test manager with custom universe."""
        assert len(custom_manager) == 3
        assert "TEST1" in custom_manager

    def test_symbols_property(self, custom_manager):
        """Test symbols property returns sorted list."""
        symbols = custom_manager.symbols
        assert isinstance(symbols, list)
        assert symbols == sorted(symbols)

    def test_etf_symbols_property(self, custom_manager):
        """Test etf_symbols property."""
        etfs = custom_manager.etf_symbols
        assert "TEST_ETF" in etfs
        assert "TEST1" not in etfs

    def test_equity_symbols_property(self, custom_manager):
        """Test equity_symbols property."""
        equities = custom_manager.equity_symbols
        assert "TEST1" in equities
        assert "TEST2" in equities
        assert "TEST_ETF" not in equities

    def test_get_symbols_by_sector(self, custom_manager):
        """Test getting symbols by sector."""
        tech = custom_manager.get_symbols_by_sector(Sector.TECHNOLOGY)
        assert "TEST1" in tech
        assert "TEST2" not in tech

    def test_get_symbol_info(self, custom_manager):
        """Test getting symbol info."""
        info = custom_manager.get_symbol_info("TEST1")
        assert info is not None
        assert info.symbol == "TEST1"
        assert info.sector == Sector.TECHNOLOGY

    def test_get_symbol_info_not_found(self, custom_manager):
        """Test getting info for non-existent symbol."""
        info = custom_manager.get_symbol_info("UNKNOWN")
        assert info is None

    def test_add_symbol(self, custom_manager):
        """Test adding a symbol."""
        custom_manager.add_symbol("NEW", "New Stock", Sector.FINANCIALS)

        assert "NEW" in custom_manager
        info = custom_manager.get_symbol_info("NEW")
        assert info.name == "New Stock"

    def test_remove_symbol(self, custom_manager):
        """Test removing a symbol."""
        assert "TEST1" in custom_manager
        custom_manager.remove_symbol("TEST1")
        assert "TEST1" not in custom_manager

    def test_remove_nonexistent_symbol(self, custom_manager):
        """Test removing non-existent symbol (no error)."""
        custom_manager.remove_symbol("NONEXISTENT")
        # Should not raise error

    def test_update_liquidity_metrics_passes(self, custom_manager):
        """Test updating liquidity metrics that pass gate."""
        result = custom_manager.update_liquidity_metrics(
            "TEST1",
            avg_option_volume=50000,
            median_spread_pct=0.02,
            avg_oi=5000,
        )
        assert result is True
        assert "TEST1" in custom_manager

    def test_update_liquidity_metrics_fails_volume(self, custom_manager):
        """Test updating liquidity metrics that fail on volume."""
        result = custom_manager.update_liquidity_metrics(
            "TEST1",
            avg_option_volume=5000,  # Too low
        )
        assert result is False
        assert "TEST1" not in custom_manager

    def test_update_liquidity_metrics_fails_spread(self, custom_manager):
        """Test updating liquidity metrics that fail on spread."""
        result = custom_manager.update_liquidity_metrics(
            "TEST1",
            median_spread_pct=0.10,  # Too high
        )
        assert result is False

    def test_update_liquidity_metrics_unknown_symbol(self, custom_manager):
        """Test updating metrics for unknown symbol."""
        result = custom_manager.update_liquidity_metrics(
            "UNKNOWN",
            avg_option_volume=50000,
        )
        assert result is False

    def test_len(self, custom_manager):
        """Test __len__ method."""
        assert len(custom_manager) == 3

    def test_contains(self, custom_manager):
        """Test __contains__ method."""
        assert "TEST1" in custom_manager
        assert "UNKNOWN" not in custom_manager

    def test_iter(self, custom_manager):
        """Test __iter__ method."""
        symbols = list(custom_manager)
        assert len(symbols) == 3
        assert symbols == sorted(symbols)


class TestUniverseManagerRefresh:
    """Test universe manager refresh from options data."""

    @pytest.fixture
    def manager(self) -> UniverseManager:
        """Create manager with test symbols."""
        custom = {
            "AAPL": UniverseSymbol("AAPL", "Apple", Sector.TECHNOLOGY),
            "MSFT": UniverseSymbol("MSFT", "Microsoft", Sector.TECHNOLOGY),
        }
        return UniverseManager(custom_symbols=custom)

    def test_refresh_empty_df(self, manager):
        """Test refresh with empty DataFrame."""
        result = manager.refresh_from_options_data(pd.DataFrame())
        assert result == {}

    def test_refresh_with_data(self, manager):
        """Test refresh with options data."""
        options_df = pd.DataFrame(
            {
                "underlying": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "volume": [20000, 25000, 15000, 18000],
                "bid": [5.0, 6.0, 4.0, 5.0],
                "ask": [5.10, 6.10, 4.10, 5.10],
                "open_interest": [5000, 6000, 4000, 5000],
                "expiry": pd.Timestamp.now() + pd.Timedelta(days=30),
            }
        )

        result = manager.refresh_from_options_data(options_df)

        assert "AAPL" in result
        assert "MSFT" in result

    def test_get_universe_summary(self, manager):
        """Test getting universe summary."""
        summary = manager.get_universe_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "symbol" in summary.columns
        assert "sector" in summary.columns
        assert "active" in summary.columns
        assert len(summary) == 2
