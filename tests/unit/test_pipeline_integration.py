"""Tests for pipeline integration with ProviderRegistry (section-10)."""

from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import pandas as pd
import pytest

from quantstack.data.providers.base import DataProvider


class MockRegistry:
    """Minimal mock of ProviderRegistry for pipeline tests."""

    def __init__(self):
        self.calls = []

    def fetch(self, data_type, symbol, **kwargs):
        self.calls.append((data_type, symbol, kwargs))
        if data_type == "macro_indicator":
            return pd.DataFrame({"date": ["2025-01-01"], "value": [1.5]})
        if data_type == "insider_transactions":
            return pd.DataFrame({
                "ticker": [symbol],
                "transaction_date": ["2025-01-15"],
                "owner_name": ["Test Owner"],
                "transaction_type": ["buy"],
                "shares": [1000],
                "price_per_share": [150.0],
            })
        if data_type == "institutional_holdings":
            return pd.DataFrame({
                "ticker": [symbol],
                "investor_name": ["Big Fund"],
                "shares_held": [50000],
            })
        if data_type == "fundamentals":
            return {"Symbol": symbol, "MarketCap": "1000000000"}
        return None


class TestPipelineAcceptsRegistry:
    """AcquisitionPipeline constructor accepts a registry."""

    def test_constructor_with_registry(self):
        """Registry parameter is stored on the pipeline instance."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()
        registry = MockRegistry()

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        assert pipeline._registry is registry

    def test_constructor_without_registry(self):
        """Pipeline still works without registry (backward compat)."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()

        pipeline = AcquisitionPipeline(av, store)

        assert pipeline._registry is None
        assert pipeline._av is av


class TestMacroRoutesRegistry:
    """Macro phase uses registry when available."""

    def test_macro_calls_registry(self):
        """When registry is set, macro phase calls registry.fetch('macro_indicator', ...)."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        av.fetch_economic_indicator.return_value = pd.DataFrame()
        store = MagicMock()
        store.save_macro_indicators.return_value = 5
        registry = MockRegistry()

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=mock_conn)
            cm.__exit__ = MagicMock(return_value=False)
            mock_pg.return_value = cm

            pipeline._fetch_and_store_macro("REAL_GDP", "quarterly", None)

        # Registry was called with macro_indicator
        macro_calls = [c for c in registry.calls if c[0] == "macro_indicator"]
        assert len(macro_calls) >= 1


class TestInsiderRoutesRegistry:
    """Insider phase uses registry for EDGAR fallback."""

    def test_insider_calls_registry(self):
        """When registry is set, insider phase calls registry first."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()
        store.save_insider_trades.return_value = 3
        registry = MockRegistry()

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=mock_conn)
            cm.__exit__ = MagicMock(return_value=False)
            mock_pg.return_value = cm

            pipeline._fetch_and_store_insider("AAPL")

        insider_calls = [c for c in registry.calls if c[0] == "insider_transactions"]
        assert len(insider_calls) >= 1
        # AV was NOT called since registry returned data
        av.fetch_insider_transactions.assert_not_called()


class TestFundamentalsRoutesRegistry:
    """Fundamentals phase routes through registry."""

    def test_fundamentals_uses_registry(self):
        """Registry.fetch('fundamentals', symbol) called before AV fallback."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()
        registry = MockRegistry()

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=mock_conn)
            cm.__exit__ = MagicMock(return_value=False)
            mock_pg.return_value = cm

            result = pipeline._fetch_and_store_overview("AAPL")

        fund_calls = [c for c in registry.calls if c[0] == "fundamentals"]
        assert len(fund_calls) >= 1
        assert result is True


class TestBackwardCompatibility:
    """Pipeline works with av_client only (no registry)."""

    def test_macro_without_registry_uses_av(self):
        """Without registry, macro phase falls back to direct AV call."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        av.fetch_economic_indicator.return_value = pd.DataFrame(
            {"date": ["2025-01-01"], "value": [1.5]}
        )
        store = MagicMock()
        store.save_macro_indicators.return_value = 1

        pipeline = AcquisitionPipeline(av, store)  # No registry

        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=mock_conn)
            cm.__exit__ = MagicMock(return_value=False)
            mock_pg.return_value = cm

            pipeline._fetch_and_store_macro("CPI", "monthly", None)

        av.fetch_economic_indicator.assert_called_once()
