# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.data.base module."""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe
from quantcore.data.base import AssetClass, AssetClassAdapter


class TestAssetClassEnum:
    """Test AssetClass enum."""

    def test_values(self):
        """Test enum values."""
        assert AssetClass.EQUITY.value == "EQUITY"
        assert AssetClass.COMMODITY_FUTURES.value == "COMMODITY_FUTURES"
        assert AssetClass.FX.value == "FX"
        assert AssetClass.FIXED_INCOME.value == "FIXED_INCOME"
        assert AssetClass.CRYPTO.value == "CRYPTO"

    def test_count(self):
        """Test number of asset classes."""
        assert len(AssetClass) == 5

    def test_from_value(self):
        """Test creating from value."""
        assert AssetClass("EQUITY") == AssetClass.EQUITY
        assert AssetClass("CRYPTO") == AssetClass.CRYPTO


class MockAdapter(AssetClassAdapter):
    """Mock adapter for testing."""

    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or ["TEST1", "TEST2", "TEST3"]

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
        # Return empty DataFrame for tests
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_available_symbols(self) -> List[str]:
        return self._symbols


class TestAssetClassAdapter:
    """Test AssetClassAdapter abstract class."""

    @pytest.fixture
    def adapter(self) -> MockAdapter:
        """Create mock adapter."""
        return MockAdapter()

    def test_asset_class_property(self, adapter):
        """Test asset_class property."""
        assert adapter.asset_class == AssetClass.EQUITY

    def test_fetch_ohlcv_returns_dataframe(self, adapter):
        """Test fetch_ohlcv returns DataFrame."""
        result = adapter.fetch_ohlcv("TEST1", Timeframe.D1)
        assert isinstance(result, pd.DataFrame)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_get_available_symbols(self, adapter):
        """Test get_available_symbols returns list."""
        symbols = adapter.get_available_symbols()
        assert isinstance(symbols, list)
        assert "TEST1" in symbols

    def test_validate_symbol_valid(self, adapter):
        """Test validate_symbol for valid symbol."""
        assert adapter.validate_symbol("TEST1") is True
        assert adapter.validate_symbol("TEST2") is True

    def test_validate_symbol_invalid(self, adapter):
        """Test validate_symbol for invalid symbol."""
        assert adapter.validate_symbol("INVALID") is False
        assert adapter.validate_symbol("") is False

    def test_get_execution_model_not_implemented(self, adapter):
        """Test get_execution_model raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            adapter.get_execution_model()

    def test_get_regime_detector_not_implemented(self, adapter):
        """Test get_regime_detector raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            adapter.get_regime_detector()

    def test_get_feature_factory_not_implemented(self, adapter):
        """Test get_feature_factory raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            adapter.get_feature_factory()


class TestAdapterSubclassing:
    """Test that adapter can be properly subclassed."""

    def test_must_implement_asset_class(self):
        """Test that subclass must implement asset_class."""

        class IncompleteAdapter(AssetClassAdapter):
            def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
                return pd.DataFrame()

            def get_available_symbols(self):
                return []

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_must_implement_fetch_ohlcv(self):
        """Test that subclass must implement fetch_ohlcv."""

        class IncompleteAdapter(AssetClassAdapter):
            @property
            def asset_class(self):
                return AssetClass.EQUITY

            def get_available_symbols(self):
                return []

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_must_implement_get_available_symbols(self):
        """Test that subclass must implement get_available_symbols."""

        class IncompleteAdapter(AssetClassAdapter):
            @property
            def asset_class(self):
                return AssetClass.EQUITY

            def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
                return pd.DataFrame()

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_can_override_optional_methods(self):
        """Test that optional methods can be overridden."""

        class FullAdapter(AssetClassAdapter):
            @property
            def asset_class(self):
                return AssetClass.EQUITY

            def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
                return pd.DataFrame()

            def get_available_symbols(self):
                return ["SYM"]

            def get_execution_model(self):
                return "custom_execution_model"

            def get_regime_detector(self):
                return "custom_regime_detector"

            def get_feature_factory(self):
                return "custom_feature_factory"

        adapter = FullAdapter()
        assert adapter.get_execution_model() == "custom_execution_model"
        assert adapter.get_regime_detector() == "custom_regime_detector"
        assert adapter.get_feature_factory() == "custom_feature_factory"
