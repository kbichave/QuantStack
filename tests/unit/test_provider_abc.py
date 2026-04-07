"""Tests for DataProvider ABC, ConfigurationError, and AVProvider adapter."""

import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from quantstack.data.providers.base import ConfigurationError, DataProvider


class MinimalProvider(DataProvider):
    """Provider that only implements name() — all fetch methods should raise."""

    def name(self) -> str:
        return "minimal"


class TestDataProviderABC:
    """Contract tests for the DataProvider abstract base class."""

    def test_subclass_only_implementing_name_raises_not_implemented(self):
        """All fetch methods raise NotImplementedError by default."""
        provider = MinimalProvider()
        assert provider.name() == "minimal"

        for method_name in (
            "fetch_ohlcv_daily",
            "fetch_ohlcv_intraday",
            "fetch_macro_indicator",
            "fetch_fundamentals",
            "fetch_insider_transactions",
            "fetch_institutional_holdings",
            "fetch_earnings_history",
            "fetch_sec_filings",
            "fetch_news_sentiment",
        ):
            with pytest.raises(NotImplementedError):
                getattr(provider, method_name)("AAPL")

        with pytest.raises(NotImplementedError):
            provider.fetch_options_chain("AAPL", "2026-01-01")

    def test_not_implemented_distinguishable_from_none(self):
        """NotImplementedError (unsupported) is different from None (no data)."""

        class MacroOnlyProvider(DataProvider):
            def name(self) -> str:
                return "macro_only"

            def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
                return None  # supported but no data

        provider = MacroOnlyProvider()
        # Supported method returns None (no data)
        assert provider.fetch_macro_indicator("GDP") is None
        # Unsupported method raises
        with pytest.raises(NotImplementedError):
            provider.fetch_ohlcv_daily("AAPL")

    def test_provider_returning_none_is_not_error(self):
        """None return indicates no data, not failure."""

        class EmptyProvider(DataProvider):
            def name(self) -> str:
                return "empty"

            def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
                return None

        provider = EmptyProvider()
        result = provider.fetch_macro_indicator("NONEXISTENT")
        assert result is None


class TestConfigurationError:
    """Tests for the ConfigurationError exception."""

    def test_raised_on_missing_config(self):
        """ConfigurationError is raised for missing required env vars."""
        with pytest.raises(ConfigurationError, match="missing"):
            raise ConfigurationError("API key missing")

    def test_catchable_separately_from_runtime_errors(self):
        """ConfigurationError is distinct from RuntimeError."""
        assert not issubclass(ConfigurationError, RuntimeError)
        try:
            raise ConfigurationError("test")
        except ConfigurationError:
            pass  # caught specifically
        except Exception:
            pytest.fail("ConfigurationError should be caught before generic Exception")


class TestAVProvider:
    """Tests for the Alpha Vantage provider adapter."""

    def _make_provider(self):
        """Create AVProvider with a mock client."""
        mock_client = MagicMock()
        from quantstack.data.providers.alpha_vantage import AVProvider

        return AVProvider(client=mock_client), mock_client

    def test_name_returns_alpha_vantage(self):
        provider, _ = self._make_provider()
        assert provider.name() == "alpha_vantage"

    def test_fetch_macro_indicator_delegates_to_client(self):
        provider, client = self._make_provider()
        client.fetch_economic_indicator.return_value = pd.DataFrame(
            {"value": [2.5]}, index=pd.to_datetime(["2026-01-01"])
        )
        result = provider.fetch_macro_indicator("TREASURY_YIELD_10Y")
        client.fetch_economic_indicator.assert_called_once_with(
            "TREASURY_YIELD", maturity="10year"
        )
        assert result is not None

    def test_fetch_insider_transactions_delegates(self):
        provider, client = self._make_provider()
        client.fetch_insider_transactions.return_value = pd.DataFrame(
            {"owner_name": ["Doe"]}
        )
        result = provider.fetch_insider_transactions("AAPL")
        client.fetch_insider_transactions.assert_called_once_with("AAPL")
        assert result is not None

    def test_fetch_fundamentals_delegates(self):
        provider, client = self._make_provider()
        client.fetch_company_overview.return_value = {"Symbol": "AAPL", "PE": 28.5}
        result = provider.fetch_fundamentals("AAPL")
        client.fetch_company_overview.assert_called_once_with("AAPL")
        assert result == {"Symbol": "AAPL", "PE": 28.5}

    def test_fetch_fundamentals_returns_none_on_empty(self):
        provider, client = self._make_provider()
        client.fetch_company_overview.return_value = {}
        assert provider.fetch_fundamentals("AAPL") is None

    def test_initializes_with_existing_client(self):
        """Dependency injection preserves shared rate limit state."""
        mock_client = MagicMock()
        from quantstack.data.providers.alpha_vantage import AVProvider

        provider = AVProvider(client=mock_client)
        assert provider._client is mock_client

    def test_unknown_macro_indicator_returns_none(self):
        """Indicator not in MACRO_INDICATOR_MAP returns None (not an error)."""
        provider, client = self._make_provider()
        result = provider.fetch_macro_indicator("UNKNOWN_INDICATOR")
        assert result is None
        client.fetch_economic_indicator.assert_not_called()
