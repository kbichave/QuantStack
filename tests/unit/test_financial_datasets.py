# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FinancialDatasets.ai integration.

Covers:
  - HTTP client: rate limiter, retry, endpoint methods
  - OHLCV adapter: timeframe mapping, DataFrame contract, empty responses
  - FundamentalsProvider: JSON→DataFrame normalisation
  - Fundamentals schema: save/load round-trip
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.config.timeframes import Timeframe
from quantstack.data.base import AssetClass
from quantstack.data.provider_enum import DataProvider
from quantstack.data.adapters.financial_datasets import FinancialDatasetsAdapter, _SUPPORTED_TIMEFRAMES
from quantstack.data.adapters.financial_datasets_client import FinancialDatasetsClient, _SlidingWindowRateLimiter
from quantstack.data.fundamentals import FundamentalsProvider
from quantstack.data.storage import DataStore
import httpx


# =============================================================================
# Rate limiter tests
# =============================================================================


class TestSlidingWindowRateLimiter:
    def test_no_sleep_under_limit(self):
        limiter = _SlidingWindowRateLimiter(max_requests=100, window_seconds=60.0)
        start = time.monotonic()
        for _ in range(10):
            limiter.acquire()
        elapsed = time.monotonic() - start
        # 10 acquires under a 100 limit should take near-zero time.
        assert elapsed < 1.0

    def test_throttles_when_limit_reached(self):
        # Tiny window: 2 requests per 0.5s.
        limiter = _SlidingWindowRateLimiter(max_requests=2, window_seconds=0.5)
        limiter.acquire()  # 1
        limiter.acquire()  # 2

        start = time.monotonic()
        limiter.acquire()  # 3 — should sleep ~0.5s
        elapsed = time.monotonic() - start
        assert elapsed >= 0.3  # some tolerance


# =============================================================================
# HTTP client tests
# =============================================================================


class TestFinancialDatasetsClient:
    @pytest.fixture
    def mock_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"prices": [{"open": 100, "close": 101}]}
        resp.raise_for_status = MagicMock()
        return resp

    def test_get_historical_prices(self, mock_response):
        client = FinancialDatasetsClient(api_key="test-key", rate_limit_rpm=10000)
        with patch.object(client._client, "get", return_value=mock_response):
            result = client.get_historical_prices(
                "AAPL", "day", 1, "2024-01-01", "2024-12-31"
            )
        assert result is not None
        assert "prices" in result
        client.close()

    def test_get_income_statements(self, mock_response):
        mock_response.json.return_value = {
            "income_statements": [{"revenue": 1000000, "net_income": 200000}]
        }
        client = FinancialDatasetsClient(api_key="test-key", rate_limit_rpm=10000)
        with patch.object(client._client, "get", return_value=mock_response):
            result = client.get_income_statements("AAPL", "annual", 5)
        assert result is not None
        assert "income_statements" in result
        client.close()

    def test_returns_none_on_http_error(self):
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=resp
        )
        client = FinancialDatasetsClient(api_key="bad-key", rate_limit_rpm=10000)
        with patch.object(client._client, "get", return_value=resp):
            result = client.get_company_facts("AAPL")
        assert result is None
        client.close()

    def test_retries_on_429(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.1"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"earnings": []}
        resp_200.raise_for_status = MagicMock()

        client = FinancialDatasetsClient(api_key="test-key", rate_limit_rpm=10000)
        with patch.object(client._client, "get", side_effect=[resp_429, resp_200]):
            result = client.get_earnings("AAPL", 5)
        assert result is not None
        client.close()

    def test_context_manager(self):
        with FinancialDatasetsClient(api_key="test-key") as client:
            assert client is not None


# =============================================================================
# OHLCV adapter tests
# =============================================================================


class TestFinancialDatasetsAdapter:
    def test_provider_enum(self):
        adapter = FinancialDatasetsAdapter(api_key="test-key")
        assert adapter.provider == DataProvider.FINANCIAL_DATASETS
        assert adapter.asset_class == AssetClass.EQUITY
        adapter._client.close()

    def test_supported_timeframes(self):
        expected = {
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
            Timeframe.W1,
        }
        assert _SUPPORTED_TIMEFRAMES == expected

    def test_unsupported_timeframe_raises(self):
        adapter = FinancialDatasetsAdapter(api_key="test-key")
        with pytest.raises(ValueError, match="does not support"):
            adapter.fetch_ohlcv("AAPL", Timeframe.S5)
        adapter._client.close()

    def test_empty_response_returns_empty_df(self):
        adapter = FinancialDatasetsAdapter(api_key="test-key")
        with patch.object(
            adapter._client, "get_all_historical_prices", return_value=[]
        ):
            df = adapter.fetch_ohlcv("AAPL", Timeframe.D1)
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        adapter._client.close()

    def test_ohlcv_contract(self):
        """Verify the DataFrame contract: DatetimeIndex, float64 columns, sorted."""
        mock_prices = [
            {
                "time": "2024-01-02T00:00:00Z",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000,
            },
            {
                "time": "2024-01-03T00:00:00Z",
                "open": 153.0,
                "high": 157.0,
                "low": 152.0,
                "close": 156.0,
                "volume": 1200000,
            },
            {
                "time": "2024-01-01T00:00:00Z",
                "open": 148.0,
                "high": 151.0,
                "low": 147.0,
                "close": 150.0,
                "volume": 900000,
            },
        ]

        adapter = FinancialDatasetsAdapter(api_key="test-key")
        with patch.object(
            adapter._client, "get_all_historical_prices", return_value=mock_prices
        ):
            df = adapter.fetch_ohlcv("AAPL", Timeframe.D1)

        # Contract checks.
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        for col in df.columns:
            assert df[col].dtype == float
        # Sorted ascending.
        assert (df.index == df.index.sort_values()).all()
        adapter._client.close()

    def test_h4_resampling(self):
        """H4 should be derived from H1 data via resampling."""
        # Generate 8 hours of H1 data.
        mock_h1 = [
            {
                "time": f"2024-01-02T{h:02d}:00:00Z",
                "open": 150 + h,
                "high": 155 + h,
                "low": 149 + h,
                "close": 153 + h,
                "volume": 100000,
            }
            for h in range(8)
        ]

        adapter = FinancialDatasetsAdapter(api_key="test-key")
        with patch.object(
            adapter._client, "get_all_historical_prices", return_value=mock_h1
        ):
            df = adapter.fetch_ohlcv("AAPL", Timeframe.H4)

        # H4 resampling from 8 H1 bars should produce 2 H4 bars.
        assert not df.empty
        assert len(df) == 2
        adapter._client.close()


# =============================================================================
# FundamentalsProvider tests
# =============================================================================


class TestFundamentalsProvider:
    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    def test_fetch_income_statements(self, mock_client):
        mock_client.get_income_statements.return_value = {
            "income_statements": [
                {"revenue": 1000, "netIncome": 200, "reportPeriod": "2024-01-01"},
                {"revenue": 900, "netIncome": 180, "reportPeriod": "2023-01-01"},
            ]
        }

        fp = FundamentalsProvider(client=mock_client)
        df = fp.fetch_income_statements("AAPL", "annual", 2)

        assert not df.empty
        assert len(df) == 2
        assert "ticker" in df.columns
        assert df["ticker"].iloc[0] == "AAPL"
        assert "statement_type" in df.columns
        assert df["statement_type"].iloc[0] == "income"
        # Columns should be snake_case.
        assert "net_income" in df.columns or "netincome" in df.columns.str.lower()

    def test_fetch_earnings(self, mock_client):
        mock_client.get_earnings.return_value = {
            "earnings": [
                {
                    "reportDate": "2024-01-25",
                    "estimate": 1.5,
                    "reportedEps": 1.6,
                    "surprise": 0.1,
                },
            ]
        }

        fp = FundamentalsProvider(client=mock_client)
        df = fp.fetch_earnings("NVDA", 5)
        assert not df.empty
        assert "ticker" in df.columns

    def test_fetch_insider_trades(self, mock_client):
        mock_client.get_insider_trades.return_value = {
            "insider_trades": [
                {
                    "transactionDate": "2024-03-01",
                    "ownerName": "Tim Cook",
                    "transactionType": "sell",
                    "shares": 50000,
                    "pricePerShare": 180.0,
                },
            ]
        }

        fp = FundamentalsProvider(client=mock_client)
        df = fp.fetch_insider_trades("AAPL", 10)
        assert not df.empty
        assert df["ticker"].iloc[0] == "AAPL"

    def test_empty_response(self, mock_client):
        mock_client.get_balance_sheets.return_value = None

        fp = FundamentalsProvider(client=mock_client)
        df = fp.fetch_balance_sheets("AAPL")
        assert df.empty

    def test_context_manager(self, mock_client):
        with FundamentalsProvider(client=mock_client) as fp:
            assert fp is not None


# =============================================================================
# Fundamentals schema tests
# =============================================================================


class TestFundamentalsSchema:
    @pytest.fixture
    def store(self):
        """DataStore backed by PostgreSQL."""
        return DataStore()

    def test_schema_creation(self, store):
        """Verify all fundamentals tables exist after init."""
        from quantstack.db import pg_conn
        with pg_conn() as conn:
            tables = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchdf()
        table_names = set(tables["tablename"].tolist())

        expected_tables = {
            "financial_statements",
            "financial_metrics",
            "insider_trades",
            "institutional_ownership",
            "analyst_estimates",
            "sec_filings",
        }
        assert expected_tables.issubset(table_names)
        store.close()

    def test_financial_statements_roundtrip(self, store):
        """Save and load financial statements."""
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "statement_type": "income",
                    "period_type": "annual",
                    "report_period": pd.Timestamp("2024-01-01"),
                    "revenue": 383285000000,
                    "net_income": 96995000000,
                },
            ]
        )
        saved = store.save_financial_statements(df)
        assert saved == 1

        loaded = store.load_financial_statements("AAPL", "income", "annual")
        assert len(loaded) == 1
        assert loaded["revenue"].iloc[0] == 383285000000
        store.close()

    def test_insider_trades_roundtrip(self, store):
        """Save and load insider trades."""
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "transaction_date": pd.Timestamp("2024-03-01"),
                    "owner_name": "Tim Cook",
                    "transaction_type": "sell",
                    "shares": 50000,
                    "price_per_share": 180.0,
                    "total_value": 9000000.0,
                },
            ]
        )
        saved = store.save_insider_trades(df)
        assert saved == 1

        loaded = store.load_insider_trades("AAPL")
        assert len(loaded) == 1
        assert loaded["owner_name"].iloc[0] == "Tim Cook"
        store.close()

    def test_financial_metrics_roundtrip(self, store):
        """Save and load financial metrics."""
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "date": pd.Timestamp("2024-01-01"),
                    "period_type": "annual",
                    "pe_ratio": 28.5,
                    "market_cap": 3e12,
                    "roe": 0.175,
                },
            ]
        )
        saved = store.save_financial_metrics(df)
        assert saved == 1

        loaded = store.load_financial_metrics("AAPL", "annual")
        assert len(loaded) == 1
        assert loaded["pe_ratio"].iloc[0] == pytest.approx(28.5)
        store.close()

    def test_institutional_ownership_roundtrip(self, store):
        """Save and load institutional ownership."""
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "investor_name": "Vanguard Group",
                    "report_date": pd.Timestamp("2024-03-31"),
                    "shares_held": 1300000000,
                    "market_value": 234e9,
                },
            ]
        )
        saved = store.save_institutional_ownership(df)
        assert saved == 1

        loaded = store.load_institutional_ownership("AAPL")
        assert len(loaded) == 1
        store.close()

    def test_sec_filings_roundtrip(self, store):
        """Save and load SEC filings."""
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "accession_number": "0000320193-24-000123",
                    "filing_type": "10-K",
                    "filed_date": pd.Timestamp("2024-10-30"),
                    "url": "https://sec.gov/...",
                },
            ]
        )
        saved = store.save_sec_filings(df)
        assert saved == 1

        loaded = store.load_sec_filings("AAPL", "10-K")
        assert len(loaded) == 1
        assert loaded["accession_number"].iloc[0] == "0000320193-24-000123"
        store.close()


# =============================================================================
# Provider enum test
# =============================================================================


class TestProviderEnum:
    def test_financial_datasets_in_enum(self):
        assert DataProvider.FINANCIAL_DATASETS.value == "financial_datasets"
        assert DataProvider("financial_datasets") == DataProvider.FINANCIAL_DATASETS

    def test_enum_count(self):
        assert len(DataProvider) == 5
