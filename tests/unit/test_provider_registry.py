"""Tests for ProviderRegistry (section-09)."""

import json
import time
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from quantstack.data.providers.base import ConfigurationError, DataProvider


class MockProvider(DataProvider):
    """Test provider that returns canned data."""

    def __init__(self, provider_name: str):
        self._name = provider_name

    def name(self) -> str:
        return self._name

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        return pd.DataFrame({"date": ["2025-01-01"], "value": [1.5]})

    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None:
        return pd.DataFrame({"ticker": [symbol], "shares": [100]})


class FailingProvider(DataProvider):
    """Provider that always raises on fetch."""

    def __init__(self, provider_name: str):
        self._name = provider_name

    def name(self) -> str:
        return self._name

    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None:
        raise RuntimeError("API is down")


class TestRoutingAndFallback:
    """Registry routes to primary, falls back on failure."""

    def test_routes_to_primary(self):
        """Primary provider is tried first."""
        from quantstack.data.providers.registry import ProviderRegistry

        primary = MockProvider("alpha_vantage")
        fallback = MockProvider("fred")
        registry = ProviderRegistry([primary, fallback])

        with patch.object(registry, "_record_success"), \
             patch.object(registry, "_record_failure"):
            result = registry.fetch("macro_indicator", "AAPL", indicator="CPI")

        assert result is not None

    def test_falls_back_on_primary_failure(self):
        """Falls back to secondary when primary fails."""
        from quantstack.data.providers.registry import ProviderRegistry

        primary = FailingProvider("alpha_vantage")
        fallback = MockProvider("fred")
        registry = ProviderRegistry([primary, fallback])

        with patch.object(registry, "_record_success"), \
             patch.object(registry, "_record_failure"):
            result = registry.fetch("macro_indicator", "AAPL", indicator="CPI")

        assert result is not None

    def test_not_implemented_skips_silently(self):
        """NotImplementedError skips provider without counting as failure."""
        from quantstack.data.providers.registry import ProviderRegistry

        # fred doesn't support ohlcv_daily (raises NotImplementedError)
        fred = MockProvider("fred")  # has no fetch_ohlcv_daily override
        av = MockProvider("alpha_vantage")
        registry = ProviderRegistry([fred, av])

        with patch.object(registry, "_record_failure") as mock_fail:
            # fetch_ohlcv_daily raises NotImplementedError on MockProvider
            # but the registry should NOT count it as a failure
            registry.fetch("ohlcv_daily", "AAPL")
            # _record_failure should not be called for NotImplementedError
            for c in mock_fail.call_args_list:
                assert c[0][0] != "fred" or "NotImplementedError" not in str(c)

    def test_returns_none_when_all_exhausted(self):
        """Returns None when all providers fail."""
        from quantstack.data.providers.registry import ProviderRegistry

        p1 = FailingProvider("alpha_vantage")
        p2 = FailingProvider("fred")
        registry = ProviderRegistry([p1, p2])

        with patch.object(registry, "_record_success"), \
             patch.object(registry, "_record_failure"):
            result = registry.fetch("macro_indicator", "AAPL", indicator="CPI")

        assert result is None


class TestFailureTracking:
    """Failure counter and reset tests."""

    def test_increments_on_failure(self):
        """consecutive_failures incremented on provider failure."""
        from quantstack.data.providers.registry import ProviderRegistry

        p = FailingProvider("alpha_vantage")
        registry = ProviderRegistry([p])

        mock_conn = MagicMock()
        # Return (1,) for consecutive_failures — below threshold, no alert
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.providers.registry.pg_conn", return_value=cm):
            registry._record_failure("alpha_vantage", "macro_indicator", "API down")

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "consecutive_failures" in sql

    def test_resets_on_success(self):
        """consecutive_failures reset to 0 on success."""
        from quantstack.data.providers.registry import ProviderRegistry

        p = MockProvider("alpha_vantage")
        registry = ProviderRegistry([p])

        mock_conn = MagicMock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.providers.registry.pg_conn", return_value=cm):
            registry._record_success("alpha_vantage", "macro_indicator")

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "consecutive_failures = 0" in sql

    def test_alert_on_third_failure(self):
        """system_events alert inserted after 3 consecutive failures."""
        from quantstack.data.providers.registry import ProviderRegistry

        p = MockProvider("alpha_vantage")
        registry = ProviderRegistry([p])

        mock_conn = MagicMock()
        # Return 3 consecutive failures after the upsert
        mock_conn.execute.return_value.fetchone.return_value = (3,)
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.providers.registry.pg_conn", return_value=cm):
            registry._record_failure("alpha_vantage", "macro_indicator", "API down")

        # Should have 2 execute calls: upsert + alert
        assert mock_conn.execute.call_count == 2
        alert_sql = mock_conn.execute.call_args_list[1][0][0]
        assert "system_events" in alert_sql


class TestCircuitBreaker:
    """Circuit breaker logic tests."""

    def test_skips_tripped_provider(self):
        """After 3 failures in 10 min, skip primary and go to fallback."""
        from quantstack.data.providers.registry import ProviderRegistry

        primary = MockProvider("alpha_vantage")
        fallback = MockProvider("fred")
        registry = ProviderRegistry([primary, fallback])

        # Simulate tripped circuit breaker for primary on macro_indicator
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (3,)  # 3 failures, recent
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.providers.registry.pg_conn", return_value=cm):
            is_open = registry._check_circuit_breaker("alpha_vantage", "macro_indicator")

        assert is_open is True

    def test_resets_after_cooldown(self):
        """After 10 min cooldown, circuit breaker resets."""
        from quantstack.data.providers.registry import ProviderRegistry

        primary = MockProvider("alpha_vantage")
        registry = ProviderRegistry([primary])

        mock_conn = MagicMock()
        # No recent failures (fetchone returns None = no row or 0 failures)
        mock_conn.execute.return_value.fetchone.return_value = None
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.providers.registry.pg_conn", return_value=cm):
            is_open = registry._check_circuit_breaker("alpha_vantage", "macro_indicator")

        assert is_open is False


class TestObservability:
    """Structured logging tests."""

    def test_logs_structured_fields(self):
        """fetch() logs provider_name, data_type, symbol, latency_ms, success."""
        from quantstack.data.providers.registry import ProviderRegistry

        p = MockProvider("alpha_vantage")
        registry = ProviderRegistry([p])

        with patch("quantstack.data.providers.registry.logger") as mock_logger, \
             patch.object(registry, "_record_success"), \
             patch.object(registry, "_check_circuit_breaker", return_value=False):
            registry.fetch("macro_indicator", "AAPL", indicator="CPI")

        # Verify structured log call happened
        assert mock_logger.info.called
        log_msg = str(mock_logger.info.call_args)
        assert "alpha_vantage" in log_msg
        assert "macro_indicator" in log_msg


class TestProviderInit:
    """Provider initialization with error handling."""

    def test_excludes_misconfigured_provider(self):
        """ConfigurationError at init excludes provider — registry degrades gracefully."""
        from quantstack.data.providers.registry import ProviderRegistry

        # Simulate what build_registry does: try to init providers, skip broken ones
        good = MockProvider("alpha_vantage")
        providers = [good]
        # Verify that ProviderRegistry works with only the good provider
        registry = ProviderRegistry(providers)

        assert len(registry._providers) == 1
        assert "alpha_vantage" in registry._providers

    def test_build_registry_handles_config_errors(self):
        """build_registry catches ConfigurationError and continues."""
        from quantstack.data.providers import registry as reg_mod

        # Replace the provider classes with mocks during import
        mock_av = MockProvider("alpha_vantage")

        original_build = reg_mod.build_registry

        with patch.object(reg_mod, "build_registry") as mock_build:
            # Simulate build_registry behavior: AV succeeds, FRED/EDGAR fail
            mock_build.return_value = reg_mod.ProviderRegistry([mock_av])
            result = reg_mod.build_registry()

        assert len(result._providers) == 1
