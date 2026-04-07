"""Tests for EDGAR data provider (section-08)."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.data.providers.base import ConfigurationError


class TestEDGARProviderInit:
    """Initialization tests."""

    def test_raises_without_user_agent(self):
        """ConfigurationError if EDGAR_USER_AGENT is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EDGAR_USER_AGENT", None)
            with pytest.raises(ConfigurationError, match="EDGAR_USER_AGENT"):
                from quantstack.data.providers.edgar import EDGARProvider

                EDGARProvider()

    def test_initializes_with_user_agent(self):
        """Initializes when EDGAR_USER_AGENT is present."""
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test admin@test.com"}):
            with patch("quantstack.data.providers.edgar.set_identity") as mock_id:
                from quantstack.data.providers.edgar import EDGARProvider

                provider = EDGARProvider()
                mock_id.assert_called_once_with("Test admin@test.com")
                assert provider.name() == "edgar"

    def test_name_returns_edgar(self):
        """name() returns 'edgar'."""
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test admin@test.com"}):
            with patch("quantstack.data.providers.edgar.set_identity"):
                from quantstack.data.providers.edgar import EDGARProvider

                assert EDGARProvider().name() == "edgar"

    def test_cik_cache_starts_empty(self):
        """CIK cache is empty on init."""
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test admin@test.com"}):
            with patch("quantstack.data.providers.edgar.set_identity"):
                from quantstack.data.providers.edgar import EDGARProvider

                p = EDGARProvider()
                assert p._cik_cache == {}


class TestInsiderTransactions:
    """Form 4 / insider transaction tests."""

    @pytest.fixture
    def provider(self):
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test admin@test.com"}):
            with patch("quantstack.data.providers.edgar.set_identity"):
                from quantstack.data.providers.edgar import EDGARProvider

                yield EDGARProvider()

    def test_returns_dataframe_with_correct_schema(self, provider):
        """Returns DataFrame matching insider_trades schema."""
        mock_company = MagicMock()
        mock_filing = MagicMock()
        mock_filing.form = "4"
        mock_filing.filing_date = "2025-01-15"

        # Mock the transaction data
        mock_tx = MagicMock()
        mock_tx.transaction_date = "2025-01-14"
        mock_tx.owner_name = "Tim Cook"
        mock_tx.acquisition_disposition = "D"
        mock_tx.transaction_shares = 50000.0
        mock_tx.transaction_price_per_share = 185.50
        mock_filing.obj.return_value = MagicMock(transactions=[mock_tx])

        mock_filings = MagicMock()
        mock_filings.filter.return_value = [mock_filing]
        mock_company.get_filings.return_value = mock_filings

        with patch("quantstack.data.providers.edgar.Company", return_value=mock_company):
            result = provider.fetch_insider_transactions("AAPL")

        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "ticker", "transaction_date", "owner_name",
            "transaction_type", "shares", "price_per_share",
        }
        assert set(result.columns) == expected_cols

    def test_transaction_type_mapping(self, provider):
        """'A' maps to 'buy', 'D' maps to 'sell'."""
        mock_company = MagicMock()
        mock_filing_a = MagicMock()
        mock_filing_a.form = "4"

        tx_a = MagicMock()
        tx_a.transaction_date = "2025-01-14"
        tx_a.owner_name = "John Doe"
        tx_a.acquisition_disposition = "A"
        tx_a.transaction_shares = 1000.0
        tx_a.transaction_price_per_share = 100.0
        mock_filing_a.obj.return_value = MagicMock(transactions=[tx_a])

        mock_filing_d = MagicMock()
        mock_filing_d.form = "4"

        tx_d = MagicMock()
        tx_d.transaction_date = "2025-01-15"
        tx_d.owner_name = "Jane Doe"
        tx_d.acquisition_disposition = "D"
        tx_d.transaction_shares = -500.0
        tx_d.transaction_price_per_share = 105.0
        mock_filing_d.obj.return_value = MagicMock(transactions=[tx_d])

        mock_filings = MagicMock()
        mock_filings.filter.return_value = [mock_filing_a, mock_filing_d]
        mock_company.get_filings.return_value = mock_filings

        with patch("quantstack.data.providers.edgar.Company", return_value=mock_company):
            result = provider.fetch_insider_transactions("AAPL")

        types = result["transaction_type"].tolist()
        assert types == ["buy", "sell"]
        # Shares should always be absolute
        assert (result["shares"] > 0).all()

    def test_returns_none_for_no_filings(self, provider):
        """Returns None for ticker with no Form 4 filings."""
        mock_company = MagicMock()
        mock_filings = MagicMock()
        mock_filings.filter.return_value = []
        mock_company.get_filings.return_value = mock_filings

        with patch("quantstack.data.providers.edgar.Company", return_value=mock_company):
            result = provider.fetch_insider_transactions("ZZZZZ")

        assert result is None

    def test_returns_none_on_error(self, provider):
        """Returns None on network/parsing error."""
        with patch(
            "quantstack.data.providers.edgar.Company",
            side_effect=Exception("Network error"),
        ):
            result = provider.fetch_insider_transactions("AAPL")

        assert result is None


class TestSecFilings:
    """SEC filing metadata tests."""

    @pytest.fixture
    def provider(self):
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test admin@test.com"}):
            with patch("quantstack.data.providers.edgar.set_identity"):
                from quantstack.data.providers.edgar import EDGARProvider

                yield EDGARProvider()

    def test_returns_filing_metadata(self, provider):
        """Returns DataFrame with filing metadata columns."""
        mock_company = MagicMock()
        mock_filing = MagicMock()
        mock_filing.accession_no = "0000320193-25-000001"
        mock_filing.form = "10-K"
        mock_filing.filing_date = "2025-01-15"
        mock_filing.report_date = "2024-12-31"
        mock_filing.primary_doc_url = "https://sec.gov/doc.htm"

        mock_filings = MagicMock()
        mock_filings.filter.return_value = [mock_filing]
        mock_company.get_filings.return_value = mock_filings

        with patch("quantstack.data.providers.edgar.Company", return_value=mock_company):
            result = provider.fetch_sec_filings("AAPL", ["10-K"])

        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "accession_number", "symbol", "form_type",
            "filing_date", "period_of_report", "primary_doc_url",
        }
        assert set(result.columns) == expected_cols

    def test_returns_none_for_unknown_ticker(self, provider):
        """Returns None for ticker that fails CIK resolution."""
        with patch(
            "quantstack.data.providers.edgar.Company",
            side_effect=Exception("No CIK found"),
        ):
            result = provider.fetch_sec_filings("ZZZZZ")

        assert result is None


class TestUnsupportedMethods:
    """Methods not supported by EDGAR raise NotImplementedError."""

    @pytest.fixture
    def provider(self):
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test admin@test.com"}):
            with patch("quantstack.data.providers.edgar.set_identity"):
                from quantstack.data.providers.edgar import EDGARProvider

                yield EDGARProvider()

    def test_fetch_ohlcv_daily(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_ohlcv_daily("AAPL")

    def test_fetch_options_chain(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_options_chain("AAPL", "2025-01-01")

    def test_fetch_macro_indicator(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_macro_indicator("CPI")

    def test_fetch_news_sentiment(self, provider):
        with pytest.raises(NotImplementedError):
            provider.fetch_news_sentiment("AAPL")
