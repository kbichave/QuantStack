# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_fundamentals_av.py — Alpha Vantage fundamental tools:
  - get_earnings_call_transcript
  - get_etf_profile
  - get_top_movers
  - get_market_status
  - get_av_insider_transactions
  - get_av_institutional_holdings

All tests mock AlphaVantageClient to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.quantstack.mcp.conftest import _fn


# ---------------------------------------------------------------------------
# get_earnings_call_transcript
# ---------------------------------------------------------------------------


class TestGetEarningsCallTranscript:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_earnings_call_transcript

        mock_client = MagicMock()
        mock_client.fetch_earnings_call_transcript.return_value = {
            "transcript": "Good morning. Revenue was $50B...",
            "quarter": "Q4",
        }

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_earnings_call_transcript)(
                ticker="AAPL", year=2024, quarter=4,
            )

        assert result["ticker"] == "AAPL"
        assert result["year"] == 2024
        assert result["quarter"] == 4
        assert "data" in result

    @pytest.mark.asyncio
    async def test_api_error(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_earnings_call_transcript

        with patch(
            "quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient",
            side_effect=Exception("Rate limit"),
        ):
            result = await _fn(get_earnings_call_transcript)(
                ticker="AAPL", year=2024, quarter=4,
            )

        assert "error" in result
        assert result["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# get_etf_profile
# ---------------------------------------------------------------------------


class TestGetEtfProfile:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_etf_profile

        mock_client = MagicMock()
        mock_client.fetch_etf_profile.return_value = {
            "holdings": [{"symbol": "AAPL", "weight": 7.0}],
            "sector_weights": {"Technology": 30.0},
        }

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_etf_profile)(ticker="SPY")

        assert result["ticker"] == "SPY"
        assert "data" in result

    @pytest.mark.asyncio
    async def test_error(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_etf_profile

        mock_client = MagicMock()
        mock_client.fetch_etf_profile.side_effect = ValueError("Not an ETF")

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_etf_profile)(ticker="AAPL")

        assert "error" in result


# ---------------------------------------------------------------------------
# get_top_movers
# ---------------------------------------------------------------------------


class TestGetTopMovers:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_top_movers

        mock_client = MagicMock()
        mock_client.fetch_top_gainers_losers.return_value = {
            "top_gainers": [{"ticker": "XYZ", "change_pct": 15.0}],
            "top_losers": [{"ticker": "ABC", "change_pct": -10.0}],
            "most_active": [{"ticker": "SPY", "volume": 100_000_000}],
        }

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_top_movers)()

        assert "data" in result

    @pytest.mark.asyncio
    async def test_error(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_top_movers

        mock_client = MagicMock()
        mock_client.fetch_top_gainers_losers.side_effect = Exception("Network error")

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_top_movers)()

        assert "error" in result


# ---------------------------------------------------------------------------
# get_market_status
# ---------------------------------------------------------------------------


class TestGetMarketStatus:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_market_status

        mock_client = MagicMock()
        mock_client.fetch_market_status.return_value = {
            "markets": [
                {"market_type": "Equity", "region": "United States", "current_status": "open"},
            ],
        }

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_market_status)()

        assert "data" in result

    @pytest.mark.asyncio
    async def test_error(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_market_status

        mock_client = MagicMock()
        mock_client.fetch_market_status.side_effect = Exception("timeout")

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_market_status)()

        assert "error" in result


# ---------------------------------------------------------------------------
# get_av_insider_transactions
# ---------------------------------------------------------------------------


class TestGetAvInsiderTransactions:

    @pytest.mark.asyncio
    async def test_happy_path_dataframe(self):
        """When AV returns a DataFrame, it's converted to records."""
        from quantstack.mcp.tools.qc_fundamentals_av import get_av_insider_transactions

        df = pd.DataFrame({
            "name": ["John Doe"],
            "title": ["CEO"],
            "transaction_type": ["Buy"],
            "shares": [1000],
        })

        mock_client = MagicMock()
        mock_client.fetch_insider_transactions.return_value = df

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_av_insider_transactions)(ticker="NVDA")

        assert result["ticker"] == "NVDA"
        assert result["count"] == 1
        assert len(result["data"]) == 1

    @pytest.mark.asyncio
    async def test_happy_path_dict(self):
        """When AV returns a non-DataFrame (e.g., raw list), uses it directly."""
        from quantstack.mcp.tools.qc_fundamentals_av import get_av_insider_transactions

        mock_client = MagicMock()
        mock_client.fetch_insider_transactions.return_value = [
            {"name": "Jane", "type": "sell"},
        ]

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_av_insider_transactions)(ticker="AAPL")

        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_error(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_av_insider_transactions

        mock_client = MagicMock()
        mock_client.fetch_insider_transactions.side_effect = Exception("Not found")

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_av_insider_transactions)(ticker="ZZZZ")

        assert "error" in result
        assert result["ticker"] == "ZZZZ"


# ---------------------------------------------------------------------------
# get_av_institutional_holdings
# ---------------------------------------------------------------------------


class TestGetAvInstitutionalHoldings:

    @pytest.mark.asyncio
    async def test_happy_path_dataframe(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_av_institutional_holdings

        df = pd.DataFrame({
            "holder": ["Vanguard", "BlackRock"],
            "shares": [1_000_000, 800_000],
        })

        mock_client = MagicMock()
        mock_client.fetch_institutional_holdings.return_value = df

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_av_institutional_holdings)(ticker="AAPL")

        assert result["ticker"] == "AAPL"
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_error(self):
        from quantstack.mcp.tools.qc_fundamentals_av import get_av_institutional_holdings

        mock_client = MagicMock()
        mock_client.fetch_institutional_holdings.side_effect = Exception("Unauthorized")

        with patch("quantstack.mcp.tools.qc_fundamentals_av.AlphaVantageClient", return_value=mock_client):
            result = await _fn(get_av_institutional_holdings)(ticker="MSFT")

        assert "error" in result
