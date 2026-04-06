"""Tests for AlphaVantageClient new fetcher methods (AV data expansion)."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Patch settings before importing the client so it doesn't read real env vars.
_MOCK_SETTINGS = MagicMock(
    alpha_vantage_api_key="test_key",
    alpha_vantage_base_url="https://www.alphavantage.co/query",
    alpha_vantage_rate_limit=75,
)


def _make_client():
    """Create a client with mocked settings and no DB calls."""
    with patch("quantstack.data.fetcher.get_settings", return_value=_MOCK_SETTINGS):
        from quantstack.data.fetcher import AlphaVantageClient

        client = AlphaVantageClient(api_key="test_key")
    # Disable daily quota DB round-trips.
    client._get_daily_count = lambda: 0
    client._increment_daily_count = lambda: None
    return client


def _mock_response(json_data=None, text=None, status_code=200, ok=True):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if text is not None:
        resp.text = text
    return resp


# ── Precious Metals ──────────────────────────────────────────────────


class TestFetchPreciousMetalsHistory:
    """Tests for fetch_precious_metals_history."""

    GOLD_RESPONSE = {
        "name": "Gold Prices",
        "interval": "monthly",
        "unit": "troy ounce",
        "data": [
            {"date": "2024-01-01", "value": "2050.30"},
            {"date": "2024-02-01", "value": "2100.50"},
            {"date": "2024-03-01", "value": "."},
        ],
    }

    SILVER_RESPONSE = {
        "name": "Silver Prices",
        "interval": "monthly",
        "unit": "troy ounce",
        "data": [
            {"date": "2024-01-01", "value": "23.10"},
            {"date": "2024-02-01", "value": "24.50"},
        ],
    }

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_tuple_of_dataframes(self, mock_get):
        """Successful separate calls produce (gold_df, silver_df) tuple."""
        mock_get.side_effect = [
            _mock_response(json_data=self.GOLD_RESPONSE),
            _mock_response(json_data=self.SILVER_RESPONSE),
        ]
        client = _make_client()
        gold_df, silver_df = client.fetch_precious_metals_history()

        assert isinstance(gold_df, pd.DataFrame)
        assert isinstance(silver_df, pd.DataFrame)
        # "." row should be dropped
        assert len(gold_df) == 2
        assert len(silver_df) == 2
        assert "value" in gold_df.columns
        assert "value" in silver_df.columns

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_empty_on_total_failure(self, mock_get):
        """Both calls fail -> returns (empty, empty)."""
        mock_get.side_effect = Exception("network down")
        client = _make_client()
        gold_df, silver_df = client.fetch_precious_metals_history()

        assert gold_df.empty
        assert silver_df.empty

    @patch("quantstack.data.fetcher.requests.get")
    def test_partial_failure_returns_one_empty(self, mock_get):
        """Gold succeeds, silver fails -> (gold_df, empty)."""
        mock_get.side_effect = [
            _mock_response(json_data=self.GOLD_RESPONSE),
            Exception("silver endpoint down"),
        ]
        client = _make_client()
        gold_df, silver_df = client.fetch_precious_metals_history()

        assert len(gold_df) == 2
        assert silver_df.empty


# ── Commodity History ────────────────────────────────────────────────


class TestFetchCommodityHistory:
    """Tests for fetch_commodity_history."""

    COPPER_RESPONSE = {
        "name": "Copper",
        "interval": "daily",
        "unit": "USD per pound",
        "data": [
            {"date": "2024-01-02", "value": "3.85"},
            {"date": "2024-01-03", "value": "3.90"},
            {"date": "2024-01-04", "value": "."},
        ],
    }

    @patch("quantstack.data.fetcher.requests.get")
    def test_copper_calls_correct_function(self, mock_get):
        """Verify function=COPPER sent in params."""
        mock_get.return_value = _mock_response(json_data=self.COPPER_RESPONSE)
        client = _make_client()
        client.fetch_commodity_history("COPPER")

        call_args = mock_get.call_args
        assert call_args.kwargs.get("params", call_args[1].get("params", {})).get("function") == "COPPER" or \
            call_args[1]["params"]["function"] == "COPPER"

    @patch("quantstack.data.fetcher.requests.get")
    def test_all_commodities_calls_correct_function(self, mock_get):
        """Verify function=ALL_COMMODITIES sent in params."""
        resp_data = {
            "name": "All Commodities",
            "data": [{"date": "2024-01-02", "value": "100.5"}],
        }
        mock_get.return_value = _mock_response(json_data=resp_data)
        client = _make_client()
        client.fetch_commodity_history("ALL_COMMODITIES")

        params = mock_get.call_args[1]["params"]
        assert params["function"] == "ALL_COMMODITIES"

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_dataframe_with_date_and_value(self, mock_get):
        """Verify shape: datetime index, 'value' column."""
        mock_get.return_value = _mock_response(json_data=self.COPPER_RESPONSE)
        client = _make_client()
        df = client.fetch_commodity_history("COPPER")

        assert not df.empty
        assert "value" in df.columns
        assert df.index.name == "timestamp"
        # "." row dropped
        assert len(df) == 2

    @patch("quantstack.data.fetcher.requests.get")
    def test_handles_error_response(self, mock_get):
        """AV error message -> empty DataFrame."""
        mock_get.return_value = _mock_response(
            json_data={"Error Message": "Invalid function"}
        )
        client = _make_client()
        df = client.fetch_commodity_history("COPPER")
        assert df.empty

    @patch("quantstack.data.fetcher.requests.get")
    def test_handles_dot_values(self, mock_get):
        """'.' values are coerced to NaN and dropped."""
        data = {
            "name": "Copper",
            "data": [
                {"date": "2024-01-02", "value": "."},
                {"date": "2024-01-03", "value": "."},
            ],
        }
        mock_get.return_value = _mock_response(json_data=data)
        client = _make_client()
        df = client.fetch_commodity_history("COPPER")
        assert df.empty


# ── Forex Daily ──────────────────────────────────────────────────────


class TestFetchForexDaily:
    """Tests for fetch_forex_daily."""

    FX_RESPONSE = {
        "Meta Data": {
            "1. Information": "Forex Daily Prices",
            "2. From Symbol": "EUR",
            "3. To Symbol": "USD",
        },
        "Time Series FX (Daily)": {
            "2024-01-03": {
                "1. open": "1.1050",
                "2. high": "1.1080",
                "3. low": "1.1020",
                "4. close": "1.1060",
            },
            "2024-01-02": {
                "1. open": "1.1000",
                "2. high": "1.1055",
                "3. low": "1.0990",
                "4. close": "1.1050",
            },
        },
    }

    @patch("quantstack.data.fetcher.requests.get")
    def test_calls_fx_daily_with_correct_params(self, mock_get):
        """Verify function=FX_DAILY, from_symbol, to_symbol."""
        mock_get.return_value = _mock_response(json_data=self.FX_RESPONSE)
        client = _make_client()
        client.fetch_forex_daily("EUR", "USD")

        params = mock_get.call_args[1]["params"]
        assert params["function"] == "FX_DAILY"
        assert params["from_symbol"] == "EUR"
        assert params["to_symbol"] == "USD"

    @patch("quantstack.data.fetcher.requests.get")
    def test_parses_time_series_fx_daily_key(self, mock_get):
        """Verify 'Time Series FX (Daily)' response parsing."""
        mock_get.return_value = _mock_response(json_data=self.FX_RESPONSE)
        client = _make_client()
        df = client.fetch_forex_daily("EUR", "USD")

        assert not df.empty
        assert len(df) == 2

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_close_column(self, mock_get):
        """DataFrame has 'close' column."""
        mock_get.return_value = _mock_response(json_data=self.FX_RESPONSE)
        client = _make_client()
        df = client.fetch_forex_daily("EUR", "USD")

        assert "close" in df.columns
        assert df["close"].dtype == float

    @patch("quantstack.data.fetcher.requests.get")
    def test_handles_empty_response(self, mock_get):
        """Empty/missing key -> empty DataFrame."""
        mock_get.return_value = _mock_response(json_data={"Meta Data": {}})
        client = _make_client()
        df = client.fetch_forex_daily("EUR", "USD")
        assert df.empty


# ── Listing Status ───────────────────────────────────────────────────


class TestFetchListingStatus:
    """Tests for fetch_listing_status."""

    CSV_TEXT = (
        "symbol,name,exchange,assetType,ipoDate,delistingDate,status\n"
        "AAPL,Apple Inc,NYSE,Stock,1980-12-12,,active\n"
        "MSFT,Microsoft Corporation,NASDAQ,Stock,1986-03-13,,active\n"
    )

    @patch("quantstack.data.fetcher.requests.get")
    def test_calls_listing_status_with_state_param(self, mock_get):
        """Verify state=delisted parameter."""
        mock_get.return_value = _mock_response(text=self.CSV_TEXT)
        # Force .json() to raise so it falls through to CSV parsing
        mock_get.return_value.json.side_effect = ValueError("not json")
        client = _make_client()
        client.fetch_listing_status(state="delisted")

        params = mock_get.call_args[1]["params"]
        assert params["function"] == "LISTING_STATUS"
        assert params["state"] == "delisted"

    @patch("quantstack.data.fetcher.requests.get")
    def test_parses_csv_response(self, mock_get):
        """CSV text -> DataFrame with symbol, name columns."""
        mock_get.return_value = _mock_response(text=self.CSV_TEXT)
        mock_get.return_value.json.side_effect = ValueError("not json")
        client = _make_client()
        df = client.fetch_listing_status()

        assert not df.empty
        assert "symbol" in df.columns
        assert "name" in df.columns
        assert len(df) == 2

    @patch("quantstack.data.fetcher.requests.get")
    def test_handles_empty_csv(self, mock_get):
        """Empty CSV -> empty DataFrame."""
        mock_get.return_value = _mock_response(text="")
        mock_get.return_value.json.side_effect = ValueError("not json")
        client = _make_client()
        df = client.fetch_listing_status()
        assert df.empty


# ── PCR (Put/Call Ratio) ─────────────────────────────────────────────


class TestFetchRealtimePCR:
    """Tests for fetch_realtime_pcr."""

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_dict_on_success(self, mock_get):
        """Valid response -> dict with PCR data."""
        pcr_data = {
            "symbol": "AAPL",
            "date": "2024-01-03",
            "put_call_ratio": "0.85",
            "put_volume": "125000",
            "call_volume": "147000",
        }
        mock_get.return_value = _mock_response(json_data=pcr_data)
        client = _make_client()
        result = client.fetch_realtime_pcr("AAPL")

        assert result is not None
        assert isinstance(result, dict)

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_none_on_error(self, mock_get):
        """Error response -> None."""
        mock_get.return_value = _mock_response(
            json_data={"Error Message": "not supported"}
        )
        client = _make_client()
        result = client.fetch_realtime_pcr("AAPL")
        assert result is None

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_none_on_info_note(self, mock_get):
        """AV 'Information' key (demo/blocked) -> None."""
        mock_get.return_value = _mock_response(
            json_data={"Information": "Thank you for using Alpha Vantage! This is a premium endpoint."}
        )
        client = _make_client()
        result = client.fetch_realtime_pcr("AAPL")
        assert result is None


class TestFetchHistoricalPCR:
    """Tests for fetch_historical_pcr."""

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_dataframe_on_success(self, mock_get):
        """Valid -> DataFrame with date, pcr columns."""
        pcr_data = {
            "symbol": "AAPL",
            "data": [
                {"date": "2024-01-02", "put_call_ratio": "0.82"},
                {"date": "2024-01-03", "put_call_ratio": "0.90"},
            ],
        }
        mock_get.return_value = _mock_response(json_data=pcr_data)
        client = _make_client()
        df = client.fetch_historical_pcr("AAPL")

        assert not df.empty
        assert "put_call_ratio" in df.columns
        assert len(df) == 2

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_empty_on_error(self, mock_get):
        """Error/blocked -> empty DataFrame."""
        mock_get.return_value = _mock_response(
            json_data={"Information": "Premium endpoint"}
        )
        client = _make_client()
        df = client.fetch_historical_pcr("AAPL")
        assert df.empty

    @patch("quantstack.data.fetcher.requests.get")
    def test_returns_empty_on_network_failure(self, mock_get):
        """Network failure -> empty DataFrame."""
        mock_get.side_effect = Exception("timeout")
        client = _make_client()
        df = client.fetch_historical_pcr("AAPL")
        assert df.empty
