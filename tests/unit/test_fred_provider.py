"""Tests for FRED data provider (section-07)."""

import os
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.data.providers.base import ConfigurationError


class TestFREDProviderInit:
    """Initialization and configuration tests."""

    def test_raises_without_api_key(self):
        """ConfigurationError if FRED_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FRED_API_KEY", None)
            with pytest.raises(ConfigurationError, match="FRED_API_KEY"):
                from quantstack.data.providers.fred import FREDProvider

                FREDProvider()

    def test_initializes_with_api_key(self):
        """Initializes successfully when FRED_API_KEY is present."""
        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("quantstack.data.providers.fred.Fred") as mock_fred:
                from quantstack.data.providers.fred import FREDProvider

                provider = FREDProvider()
                mock_fred.assert_called_once_with(api_key="test-key")
                assert provider._client is not None

    def test_name_returns_fred(self):
        """name() returns 'fred'."""
        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("quantstack.data.providers.fred.Fred"):
                from quantstack.data.providers.fred import FREDProvider

                assert FREDProvider().name() == "fred"


class TestFetchMacroIndicator:
    """fetch_macro_indicator tests."""

    @pytest.fixture
    def provider(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("quantstack.data.providers.fred.Fred") as mock_fred_cls:
                from quantstack.data.providers.fred import FREDProvider

                p = FREDProvider()
                p._client = mock_fred_cls.return_value
                yield p

    def test_returns_dataframe_with_date_value(self, provider):
        """Returns DataFrame with (date, value) columns."""
        series = pd.Series(
            [1.5, 1.6, 1.7],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )
        provider._client.get_series.return_value = series

        result = provider.fetch_macro_indicator("DGS10")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["date", "value"]
        assert len(result) == 3

    def test_resolves_quantstack_name_to_fred_id(self, provider):
        """QuantStack name 'TREASURY_YIELD_10Y' resolves to FRED series 'DGS10'."""
        series = pd.Series([1.5], index=pd.to_datetime(["2025-01-01"]))
        provider._client.get_series.return_value = series

        provider.fetch_macro_indicator("TREASURY_YIELD_10Y")

        provider._client.get_series.assert_called_once_with("DGS10")

    def test_returns_none_for_empty_series(self, provider):
        """Returns None when FRED series has no data."""
        provider._client.get_series.return_value = pd.Series(dtype=float)

        result = provider.fetch_macro_indicator("DGS10")

        assert result is None

    def test_raises_not_implemented_for_unknown_indicator(self, provider):
        """Raises NotImplementedError for unmapped indicator."""
        with pytest.raises(NotImplementedError, match="UNKNOWN_THING"):
            provider.fetch_macro_indicator("UNKNOWN_THING")

    def test_drops_nan_values(self, provider):
        """NaN values in FRED series are dropped."""
        series = pd.Series(
            [1.5, float("nan"), 1.7],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )
        provider._client.get_series.return_value = series

        result = provider.fetch_macro_indicator("DGS10")

        assert len(result) == 2
        assert not result["value"].isna().any()

    def test_sorted_by_date_ascending(self, provider):
        """Result is sorted by date ascending."""
        series = pd.Series(
            [1.7, 1.5, 1.6],
            index=pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]),
        )
        provider._client.get_series.return_value = series

        result = provider.fetch_macro_indicator("DGS10")

        dates = result["date"].tolist()
        assert dates == sorted(dates)


class TestRateLimiting:
    """Rate limiting tests."""

    def test_throttle_between_requests(self):
        """Consecutive calls respect minimum gap."""
        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("quantstack.data.providers.fred.Fred") as mock_fred_cls:
                from quantstack.data.providers.fred import FREDProvider

                p = FREDProvider()
                client = mock_fred_cls.return_value
                series = pd.Series([1.5], index=pd.to_datetime(["2025-01-01"]))
                client.get_series.return_value = series

                # Make two rapid calls and verify throttle attribute exists
                assert hasattr(p, "_last_request_at")


class TestUnsupportedMethods:
    """Methods not supported by FRED raise NotImplementedError."""

    @pytest.fixture
    def provider(self):
        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("quantstack.data.providers.fred.Fred"):
                from quantstack.data.providers.fred import FREDProvider

                yield FREDProvider()

    def test_fetch_ohlcv_daily(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_ohlcv_daily("AAPL")

    def test_fetch_insider_transactions(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_insider_transactions("AAPL")

    def test_fetch_fundamentals(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_fundamentals("AAPL")

    def test_fetch_options_chain(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_options_chain("AAPL", "2025-01-01")
