"""Tests for SEC filings population (section-11)."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestSecFilingsPhase:
    """Phase 15: SEC filing metadata acquisition."""

    def test_sec_filings_phase_populates(self):
        """SEC filings phase calls registry and stores results."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()
        registry = MagicMock()
        registry.fetch.return_value = pd.DataFrame({
            "accession_number": ["0000320193-25-000001"],
            "symbol": ["AAPL"],
            "form_type": ["10-K"],
            "filing_date": ["2025-01-15"],
            "period_of_report": ["2024-12-31"],
            "primary_doc_url": ["https://sec.gov/doc.htm"],
        })

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.acquisition_pipeline.pg_conn", return_value=cm):
            rows = pipeline._fetch_and_store_sec_filings("AAPL")

        assert rows == 1
        registry.fetch.assert_called_once_with(
            "sec_filings", "AAPL", form_types=["10-K", "10-Q", "8-K"]
        )

    def test_sec_filings_upsert_idempotent(self):
        """Running twice with same data produces same row count."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()
        registry = MagicMock()
        registry.fetch.return_value = pd.DataFrame({
            "accession_number": ["ACC-001", "ACC-002"],
            "form_type": ["10-K", "10-Q"],
            "filing_date": ["2025-01-15", "2025-04-15"],
            "period_of_report": ["2024-12-31", "2025-03-31"],
            "primary_doc_url": ["url1", "url2"],
        })

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.acquisition_pipeline.pg_conn", return_value=cm):
            rows1 = pipeline._fetch_and_store_sec_filings("AAPL")
            rows2 = pipeline._fetch_and_store_sec_filings("AAPL")

        assert rows1 == rows2 == 2

    def test_freshness_skips_recent(self):
        """Skips symbol with filing_date < 90 days ago."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline
        from datetime import date, timedelta

        av = MagicMock()
        store = MagicMock()
        registry = MagicMock()

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        recent_date = date.today() - timedelta(days=30)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (recent_date,)
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.acquisition_pipeline.pg_conn", return_value=cm):
            rows = pipeline._fetch_and_store_sec_filings("AAPL")

        assert rows == 0
        registry.fetch.assert_not_called()

    def test_returns_zero_without_registry(self):
        """Without registry, sec_filings phase returns 0."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()

        pipeline = AcquisitionPipeline(av, store)  # no registry

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.acquisition_pipeline.pg_conn", return_value=cm):
            rows = pipeline._fetch_and_store_sec_filings("AAPL")

        assert rows == 0


class TestEdgarInsiderPhase:
    """Phase 16: EDGAR insider + institutional data."""

    def test_edgar_insider_phase_routes_through_registry(self):
        """EDGAR insider phase calls registry for insider_transactions."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()
        store.save_insider_trades.return_value = 5
        registry = MagicMock()
        registry.fetch.return_value = pd.DataFrame({
            "ticker": ["AAPL"],
            "transaction_date": ["2025-01-15"],
            "owner_name": ["Tim Cook"],
            "transaction_type": ["sell"],
            "shares": [50000],
            "price_per_share": [185.50],
        })

        pipeline = AcquisitionPipeline(av, store, registry=registry)

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.acquisition_pipeline.pg_conn", return_value=cm):
            rows = pipeline._fetch_and_store_edgar_insider("AAPL")

        assert rows == 5
        registry.fetch.assert_called_once_with("insider_transactions", "AAPL")

    def test_edgar_phase_skips_without_registry(self):
        """Without registry, sec_edgar_insider phase skips all symbols."""
        import asyncio
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        av = MagicMock()
        store = MagicMock()

        pipeline = AcquisitionPipeline(av, store)  # No registry

        report = asyncio.get_event_loop().run_until_complete(
            pipeline.run_sec_edgar_insider(["AAPL", "TSLA"])
        )

        assert report.skipped == 2
        assert report.succeeded == 0


class TestSecFilingsTableDDL:
    """sec_filings table DDL exists in _schema.py."""

    def test_ddl_contains_sec_filings(self):
        """_schema.py contains CREATE TABLE for sec_filings."""
        import inspect
        from quantstack.data._schema import SchemaMixin

        source = inspect.getsource(SchemaMixin)
        assert "sec_filings" in source
        assert "accession_number" in source
