# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.data.manager module."""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe
from quantcore.data.base import AssetClass, AssetClassAdapter
from quantcore.data.manager import UnifiedDataManager


class MockEquityAdapter(AssetClassAdapter):
    """Mock equity adapter for testing."""

    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or ["AAPL", "MSFT", "GOOGL"]

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITY

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        # Return sample data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

    def get_available_symbols(self) -> List[str]:
        return self._symbols


class MockCryptoAdapter(AssetClassAdapter):
    """Mock crypto adapter for testing."""

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "open": [50000.0] * 5,
                "high": [51000.0] * 5,
                "low": [49000.0] * 5,
                "close": [50500.0] * 5,
                "volume": [10000] * 5,
            },
            index=dates,
        )

    def get_available_symbols(self) -> List[str]:
        return ["BTC", "ETH", "SOL"]


class TestUnifiedDataManager:
    """Test UnifiedDataManager class."""

    @pytest.fixture
    def manager(self) -> UnifiedDataManager:
        """Create data manager with mocked storage."""
        with patch("quantcore.data.manager.DataStore"):
            return UnifiedDataManager()

    @pytest.fixture
    def equity_adapter(self) -> MockEquityAdapter:
        """Create mock equity adapter."""
        return MockEquityAdapter()

    @pytest.fixture
    def crypto_adapter(self) -> MockCryptoAdapter:
        """Create mock crypto adapter."""
        return MockCryptoAdapter()

    def test_register_adapter(self, manager, equity_adapter):
        """Test registering an adapter."""
        manager.register_adapter(equity_adapter)

        assert AssetClass.EQUITY in manager.adapters
        assert manager.adapters[AssetClass.EQUITY] is equity_adapter

    def test_register_multiple_adapters(self, manager, equity_adapter, crypto_adapter):
        """Test registering multiple adapters."""
        manager.register_adapter(equity_adapter)
        manager.register_adapter(crypto_adapter)

        assert len(manager.adapters) == 2
        assert AssetClass.EQUITY in manager.adapters
        assert AssetClass.CRYPTO in manager.adapters

    def test_register_overwrites_existing(self, manager):
        """Test registering overwrites existing adapter."""
        adapter1 = MockEquityAdapter(["A", "B"])
        adapter2 = MockEquityAdapter(["X", "Y"])

        manager.register_adapter(adapter1)
        manager.register_adapter(adapter2)

        # Should have the second adapter
        assert manager.adapters[AssetClass.EQUITY] is adapter2

    def test_get_adapter(self, manager, equity_adapter):
        """Test getting an adapter."""
        manager.register_adapter(equity_adapter)

        result = manager.get_adapter(AssetClass.EQUITY)
        assert result is equity_adapter

    def test_get_adapter_not_registered(self, manager):
        """Test getting non-registered adapter raises error."""
        with pytest.raises(ValueError, match="No adapter registered"):
            manager.get_adapter(AssetClass.CRYPTO)

    def test_fetch_ohlcv(self, manager, equity_adapter):
        """Test fetching OHLCV data."""
        manager.register_adapter(equity_adapter)

        result = manager.fetch_ohlcv(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            timeframe=Timeframe.D1,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "close" in result.columns

    def test_fetch_ohlcv_with_dates(self, manager, equity_adapter):
        """Test fetching OHLCV with date filters."""
        manager.register_adapter(equity_adapter)

        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)

        result = manager.fetch_ohlcv(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            timeframe=Timeframe.D1,
            start_date=start,
            end_date=end,
        )

        assert isinstance(result, pd.DataFrame)

    def test_fetch_universe(self, manager, equity_adapter, crypto_adapter):
        """Test fetching data for multiple symbols."""
        manager.register_adapter(equity_adapter)
        manager.register_adapter(crypto_adapter)

        symbols = [
            ("AAPL", AssetClass.EQUITY),
            ("BTC", AssetClass.CRYPTO),
        ]

        result = manager.fetch_universe(symbols, Timeframe.D1)

        assert len(result) == 2
        assert ("AAPL", AssetClass.EQUITY) in result
        assert ("BTC", AssetClass.CRYPTO) in result

    def test_fetch_universe_with_error(self, manager, equity_adapter):
        """Test fetching universe handles errors gracefully."""
        manager.register_adapter(equity_adapter)

        # Try to fetch crypto without adapter
        symbols = [
            ("AAPL", AssetClass.EQUITY),
            ("BTC", AssetClass.CRYPTO),  # No adapter registered
        ]

        result = manager.fetch_universe(symbols, Timeframe.D1)

        # Should still have results for AAPL
        assert ("AAPL", AssetClass.EQUITY) in result
        assert len(result[("AAPL", AssetClass.EQUITY)]) > 0

        # BTC should have empty DataFrame
        assert ("BTC", AssetClass.CRYPTO) in result
        assert len(result[("BTC", AssetClass.CRYPTO)]) == 0

    def test_get_all_symbols(self, manager, equity_adapter, crypto_adapter):
        """Test getting all symbols from all adapters."""
        manager.register_adapter(equity_adapter)
        manager.register_adapter(crypto_adapter)

        result = manager.get_all_symbols()

        assert AssetClass.EQUITY in result
        assert AssetClass.CRYPTO in result
        assert "AAPL" in result[AssetClass.EQUITY]
        assert "BTC" in result[AssetClass.CRYPTO]

    def test_validate_universe(self, manager, equity_adapter):
        """Test validating universe symbols."""
        manager.register_adapter(equity_adapter)

        symbols = [
            ("AAPL", AssetClass.EQUITY),
            ("INVALID", AssetClass.EQUITY),
        ]

        result = manager.validate_universe(symbols)

        assert len(result) == 2

        # AAPL should be valid
        aapl_result = [r for r in result if r[0] == "AAPL"][0]
        assert aapl_result[2] is True

        # INVALID should not be valid
        invalid_result = [r for r in result if r[0] == "INVALID"][0]
        assert invalid_result[2] is False

    def test_validate_universe_missing_adapter(self, manager, equity_adapter):
        """Test validation handles missing adapter."""
        manager.register_adapter(equity_adapter)

        symbols = [
            ("AAPL", AssetClass.EQUITY),
            ("BTC", AssetClass.CRYPTO),  # No adapter
        ]

        result = manager.validate_universe(symbols)

        # BTC should be marked invalid due to missing adapter
        btc_result = [r for r in result if r[0] == "BTC"][0]
        assert btc_result[2] is False


class TestDataManagerIntegration:
    """Integration tests for data manager."""

    @pytest.fixture
    def manager_with_adapters(self) -> UnifiedDataManager:
        """Create manager with adapters registered."""
        with patch("quantcore.data.manager.DataStore"):
            manager = UnifiedDataManager()
            manager.register_adapter(MockEquityAdapter())
            manager.register_adapter(MockCryptoAdapter())
            return manager

    def test_full_workflow(self, manager_with_adapters):
        """Test complete fetch workflow."""
        manager = manager_with_adapters

        # Get all symbols
        all_symbols = manager.get_all_symbols()
        assert len(all_symbols) == 2

        # Validate some symbols
        test_symbols = [
            ("AAPL", AssetClass.EQUITY),
            ("BTC", AssetClass.CRYPTO),
        ]
        validation = manager.validate_universe(test_symbols)
        assert all(v[2] for v in validation)

        # Fetch data
        data = manager.fetch_universe(test_symbols, Timeframe.D1)
        assert all(len(df) > 0 for df in data.values())
