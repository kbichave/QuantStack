# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_fundamentals MCP tools — financial statements, metrics,
earnings, insider trades, institutional ownership, analyst estimates,
company news, stock screener, SEC filings, and company facts.

All tools use _get_reader() for cache lookup and _get_fundamentals_provider()
for API calls. Both are mocked to avoid real I/O.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.quant_pod.mcp.conftest import _fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_reader(*, load_returns=None):
    """Return a mock PgDataStore reader.

    Args:
        load_returns: Dict mapping method names to return values.
            Defaults to empty DataFrame for all load_* methods.
    """
    store = MagicMock()
    load_returns = load_returns or {}
    # Default all load_ methods to return empty DataFrame
    for method_name in (
        "load_financial_statements",
        "load_financial_metrics",
        "load_insider_trades",
        "load_institutional_ownership",
        "load_analyst_estimates",
    ):
        getattr(store, method_name).return_value = load_returns.get(
            method_name, pd.DataFrame()
        )
    store.close.return_value = None
    return store


def _patch_reader(**kwargs):
    """Patch _get_reader() to return a mock store."""
    store = _mock_reader(**kwargs)
    return patch(
        "quantstack.mcp.tools.qc_fundamentals._get_reader",
        return_value=store,
    )


def _mock_writer():
    """Return a mock PgDataStore writer."""
    writer = MagicMock()
    writer.close.return_value = None
    return writer


def _patch_writer():
    """Patch _get_writer() to return a mock."""
    return patch(
        "quantstack.mcp.tools.qc_fundamentals._get_writer",
        return_value=_mock_writer(),
    )


def _mock_fundamentals_provider(*, returns=None):
    """Return a mock FundamentalsProvider.

    Args:
        returns: Dict mapping method names to return values.
    """
    fp = MagicMock()
    returns = returns or {}
    # Default all fetch methods to return empty DataFrame
    for method_name in (
        "fetch_income_statements",
        "fetch_balance_sheets",
        "fetch_cash_flows",
        "fetch_financial_metrics",
        "fetch_insider_trades",
        "fetch_institutional_ownership",
        "fetch_analyst_estimates",
        "fetch_company_news",
        "screen_stocks",
        "fetch_segmented_revenues",
        "fetch_earnings_press_releases",
        "fetch_sec_filings",
        "fetch_sec_filing_items",
        "fetch_interest_rates",
        "fetch_crypto_prices",
    ):
        getattr(fp, method_name).return_value = returns.get(
            method_name, pd.DataFrame()
        )
    # Non-DataFrame returns
    fp.fetch_interest_rates_snapshot.return_value = returns.get(
        "fetch_interest_rates_snapshot", None
    )
    fp.fetch_company_facts.return_value = returns.get(
        "fetch_company_facts", None
    )
    fp.close.return_value = None
    return fp


def _patch_fp(*, returns=None):
    """Patch _get_fundamentals_provider() to return a mock."""
    fp = _mock_fundamentals_provider(returns=returns)
    return patch(
        "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
        return_value=fp,
    )


def _patch_fp_none():
    """Patch _get_fundamentals_provider() to return None (no API key)."""
    return patch(
        "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
        return_value=None,
    )


def _sample_df(n=3, columns=None):
    """Build a small non-empty DataFrame for happy-path tests."""
    columns = columns or ["ticker", "value", "date"]
    data = {col: [f"val_{i}" for i in range(n)] for col in columns}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# get_financial_statements
# ---------------------------------------------------------------------------


class TestGetFinancialStatements:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """When cache has data, return it without calling API."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        cached_df = _sample_df(5)
        with _patch_reader(
            load_returns={"load_financial_statements": cached_df}
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="income"
            )
        assert "error" not in result
        assert result["source"] == "cache"
        assert result["rows"] == 5

    @pytest.mark.asyncio
    async def test_api_fetch_income(self):
        """When cache misses, fetch income statement from API."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        api_df = _sample_df(3)
        with (
            _patch_reader(),
            _patch_fp(returns={"fetch_income_statements": api_df}),
            _patch_writer(),
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="income"
            )
        assert "error" not in result
        assert result["source"] == "api"
        assert result["rows"] == 3

    @pytest.mark.asyncio
    async def test_api_fetch_balance(self):
        """Fetch balance sheet from API."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        api_df = _sample_df(2)
        with (
            _patch_reader(),
            _patch_fp(returns={"fetch_balance_sheets": api_df}),
            _patch_writer(),
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="balance"
            )
        assert "error" not in result
        assert result["statement_type"] == "balance"

    @pytest.mark.asyncio
    async def test_api_fetch_cashflow(self):
        """Fetch cash flow from API."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        api_df = _sample_df(2)
        with (
            _patch_reader(),
            _patch_fp(returns={"fetch_cash_flows": api_df}),
            _patch_writer(),
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="cashflow"
            )
        assert "error" not in result
        assert result["statement_type"] == "cashflow"

    @pytest.mark.asyncio
    async def test_invalid_statement_type(self):
        """Return error for unsupported statement_type."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        with _patch_reader(), _patch_fp():
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="invalid_type"
            )
        assert "error" in result
        assert "Unknown statement_type" in result["error"]

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """Return error when API key is not configured."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        with _patch_reader(), _patch_fp_none():
            result = await _fn(get_financial_statements)(ticker="AAPL")
        assert "error" in result
        assert "FINANCIAL_DATASETS_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_api_response(self):
        """Return error when API returns empty DataFrame."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        with (
            _patch_reader(),
            _patch_fp(returns={"fetch_income_statements": pd.DataFrame()}),
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="income"
            )
        assert "error" in result
        assert "No income" in result["error"]

    @pytest.mark.asyncio
    async def test_api_exception(self):
        """Return error when API call raises."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        mock_fp = _mock_fundamentals_provider()
        mock_fp.fetch_income_statements.side_effect = RuntimeError("API down")

        with (
            _patch_reader(),
            patch(
                "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
                return_value=mock_fp,
            ),
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="income"
            )
        assert "error" in result
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_write_failure_still_returns(self):
        """Even if cache write fails, API data should still be returned."""
        from quantstack.mcp.tools.qc_fundamentals import get_financial_statements

        api_df = _sample_df(3)
        bad_writer = MagicMock()
        bad_writer.save_financial_statements.side_effect = RuntimeError("write fail")
        bad_writer.close.return_value = None

        with (
            _patch_reader(),
            _patch_fp(returns={"fetch_income_statements": api_df}),
            patch(
                "quantstack.mcp.tools.qc_fundamentals._get_writer",
                return_value=bad_writer,
            ),
        ):
            result = await _fn(get_financial_statements)(
                ticker="AAPL", statement_type="income"
            )
        # Should still return data despite cache failure
        assert "error" not in result
        assert result["source"] == "api"


# ---------------------------------------------------------------------------
# get_financial_metrics
# ---------------------------------------------------------------------------


class TestGetFinancialMetrics:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        from quantstack.mcp.tools.qc_fundamentals import get_financial_metrics

        cached_df = _sample_df(4)
        with _patch_reader(
            load_returns={"load_financial_metrics": cached_df}
        ):
            result = await _fn(get_financial_metrics)(ticker="NVDA")
        assert result["source"] == "cache"
        assert result["rows"] == 4

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_financial_metrics

        with _patch_reader(), _patch_fp_none():
            result = await _fn(get_financial_metrics)(ticker="NVDA")
        assert "error" in result
        assert "FINANCIAL_DATASETS_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_api_response(self):
        from quantstack.mcp.tools.qc_fundamentals import get_financial_metrics

        with _patch_reader(), _patch_fp():
            result = await _fn(get_financial_metrics)(ticker="NVDA")
        assert "error" in result
        assert "No financial metrics" in result["error"]


# ---------------------------------------------------------------------------
# get_earnings_data
# ---------------------------------------------------------------------------


class TestGetEarningsData:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_earnings_data

        mock_data = {
            "quarterlyEarnings": [{"quarter": "Q1"}, {"quarter": "Q2"}],
            "annualEarnings": [{"year": "2024"}],
        }
        mock_client = MagicMock()
        mock_client.fetch_earnings_history.return_value = mock_data

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_earnings_data)(ticker="AAPL")
        assert "error" not in result
        assert result["quarterly_count"] == 2
        assert result["annual_count"] == 1

    @pytest.mark.asyncio
    async def test_no_data(self):
        from quantstack.mcp.tools.qc_fundamentals import get_earnings_data

        mock_client = MagicMock()
        mock_client.fetch_earnings_history.return_value = None

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_earnings_data)(ticker="NODATA")
        assert "error" in result
        assert "No earnings data" in result["error"]

    @pytest.mark.asyncio
    async def test_api_exception(self):
        from quantstack.mcp.tools.qc_fundamentals import get_earnings_data

        mock_client = MagicMock()
        mock_client.fetch_earnings_history.side_effect = RuntimeError("timeout")

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_earnings_data)(ticker="AAPL")
        assert "error" in result
        assert "timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_limit_applied(self):
        """The limit parameter should truncate quarterly earnings."""
        from quantstack.mcp.tools.qc_fundamentals import get_earnings_data

        mock_data = {
            "quarterlyEarnings": [{"q": i} for i in range(30)],
            "annualEarnings": [{"y": 2024}],
        }
        mock_client = MagicMock()
        mock_client.fetch_earnings_history.return_value = mock_data

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_earnings_data)(ticker="AAPL", limit=5)
        assert result["quarterly_count"] == 5


# ---------------------------------------------------------------------------
# get_insider_trades
# ---------------------------------------------------------------------------


class TestGetInsiderTrades:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        from quantstack.mcp.tools.qc_fundamentals import get_insider_trades

        cached_df = _sample_df(10)
        with _patch_reader(
            load_returns={"load_insider_trades": cached_df}
        ):
            result = await _fn(get_insider_trades)(ticker="AAPL")
        assert result["source"] == "cache"
        assert result["rows"] == 10

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_insider_trades

        with _patch_reader(), _patch_fp_none():
            result = await _fn(get_insider_trades)(ticker="AAPL")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_api_response(self):
        from quantstack.mcp.tools.qc_fundamentals import get_insider_trades

        with _patch_reader(), _patch_fp():
            result = await _fn(get_insider_trades)(ticker="AAPL")
        assert "error" in result
        assert "No insider trades" in result["error"]


# ---------------------------------------------------------------------------
# get_institutional_ownership
# ---------------------------------------------------------------------------


class TestGetInstitutionalOwnership:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        from quantstack.mcp.tools.qc_fundamentals import get_institutional_ownership

        cached_df = _sample_df(5)
        with _patch_reader(
            load_returns={"load_institutional_ownership": cached_df}
        ):
            result = await _fn(get_institutional_ownership)(ticker="MSFT")
        assert result["source"] == "cache"

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_institutional_ownership

        with _patch_reader(), _patch_fp_none():
            result = await _fn(get_institutional_ownership)(ticker="MSFT")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_analyst_estimates
# ---------------------------------------------------------------------------


class TestGetAnalystEstimates:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        from quantstack.mcp.tools.qc_fundamentals import get_analyst_estimates

        cached_df = _sample_df(3)
        with _patch_reader(
            load_returns={"load_analyst_estimates": cached_df}
        ):
            result = await _fn(get_analyst_estimates)(ticker="GOOGL")
        assert result["source"] == "cache"

    @pytest.mark.asyncio
    async def test_api_fetch(self):
        from quantstack.mcp.tools.qc_fundamentals import get_analyst_estimates

        api_df = _sample_df(2)
        with (
            _patch_reader(),
            _patch_fp(returns={"fetch_analyst_estimates": api_df}),
            _patch_writer(),
        ):
            result = await _fn(get_analyst_estimates)(ticker="GOOGL")
        assert result["source"] == "api"
        assert result["rows"] == 2


# ---------------------------------------------------------------------------
# get_company_news
# ---------------------------------------------------------------------------


class TestGetCompanyNews:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_company_news

        news_df = _sample_df(5, columns=["title", "url", "date"])
        with _patch_fp(returns={"fetch_company_news": news_df}):
            result = await _fn(get_company_news)(ticker="TSLA")
        assert "error" not in result
        assert result["rows"] == 5

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_company_news

        with _patch_fp_none():
            result = await _fn(get_company_news)(ticker="TSLA")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_news(self):
        from quantstack.mcp.tools.qc_fundamentals import get_company_news

        with _patch_fp():
            result = await _fn(get_company_news)(ticker="TSLA")
        assert "error" in result
        assert "No company news" in result["error"]


# ---------------------------------------------------------------------------
# screen_stocks
# ---------------------------------------------------------------------------


class TestScreenStocks:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import screen_stocks

        results_df = _sample_df(10)
        with _patch_fp(returns={"screen_stocks": results_df}):
            result = await _fn(screen_stocks)(
                filters={"market_cap_gt": 1e9}
            )
        assert "error" not in result
        assert result["rows"] == 10

    @pytest.mark.asyncio
    async def test_no_matches(self):
        from quantstack.mcp.tools.qc_fundamentals import screen_stocks

        with _patch_fp():
            result = await _fn(screen_stocks)(
                filters={"market_cap_gt": 1e15}
            )
        assert "error" in result
        assert "No stocks matched" in result["error"]

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import screen_stocks

        with _patch_fp_none():
            result = await _fn(screen_stocks)(filters={})
        assert "error" in result


# ---------------------------------------------------------------------------
# get_segmented_revenues
# ---------------------------------------------------------------------------


class TestGetSegmentedRevenues:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_segmented_revenues

        seg_df = _sample_df(4)
        with _patch_fp(returns={"fetch_segmented_revenues": seg_df}):
            result = await _fn(get_segmented_revenues)(ticker="AAPL")
        assert "error" not in result
        assert result["rows"] == 4

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_segmented_revenues

        with _patch_fp_none():
            result = await _fn(get_segmented_revenues)(ticker="AAPL")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_result(self):
        from quantstack.mcp.tools.qc_fundamentals import get_segmented_revenues

        with _patch_fp():
            result = await _fn(get_segmented_revenues)(ticker="AAPL")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_earnings_press_releases
# ---------------------------------------------------------------------------


class TestGetEarningsPressReleases:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_earnings_press_releases

        pr_df = _sample_df(2)
        with _patch_fp(returns={"fetch_earnings_press_releases": pr_df}):
            result = await _fn(get_earnings_press_releases)(ticker="NVDA")
        assert "error" not in result
        assert result["rows"] == 2

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_earnings_press_releases

        with _patch_fp_none():
            result = await _fn(get_earnings_press_releases)(ticker="NVDA")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_sec_filing_items
# ---------------------------------------------------------------------------


class TestGetSecFilingItems:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Full two-step flow: fetch filing, then fetch items."""
        from quantstack.mcp.tools.qc_fundamentals import get_sec_filing_items

        filings_df = pd.DataFrame(
            [{"accession_number": "0001234567-24-000001", "type": "10-K"}]
        )
        items = [{"section": "1A", "text": "Risk Factors..."}]

        mock_fp = _mock_fundamentals_provider(
            returns={"fetch_sec_filings": filings_df}
        )
        mock_fp.fetch_sec_filing_items.return_value = items

        with patch(
            "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
            return_value=mock_fp,
        ):
            result = await _fn(get_sec_filing_items)(
                ticker="AAPL", filing_type="10-K"
            )

        assert "error" not in result
        assert result["items"] == items
        assert result["accession_number"] == "0001234567-24-000001"

    @pytest.mark.asyncio
    async def test_no_filings_found(self):
        from quantstack.mcp.tools.qc_fundamentals import get_sec_filing_items

        with _patch_fp():
            result = await _fn(get_sec_filing_items)(ticker="NODATA")
        assert "error" in result
        assert "No 10-K filings" in result["error"]

    @pytest.mark.asyncio
    async def test_no_accession_number(self):
        """BUG TEST: When filing exists but has no accession_number field."""
        from quantstack.mcp.tools.qc_fundamentals import get_sec_filing_items

        # DataFrame with a row but accession_number is None/empty
        filings_df = pd.DataFrame([{"accession_number": "", "type": "10-K"}])

        with _patch_fp(returns={"fetch_sec_filings": filings_df}):
            result = await _fn(get_sec_filing_items)(ticker="AAPL")
        assert "error" in result
        assert "accession number" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_sec_filing_items

        with _patch_fp_none():
            result = await _fn(get_sec_filing_items)(ticker="AAPL")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_filing_items_empty(self):
        """When items are empty after fetching filing."""
        from quantstack.mcp.tools.qc_fundamentals import get_sec_filing_items

        filings_df = pd.DataFrame(
            [{"accession_number": "0001234567-24-000001", "type": "10-K"}]
        )
        mock_fp = _mock_fundamentals_provider(
            returns={"fetch_sec_filings": filings_df}
        )
        mock_fp.fetch_sec_filing_items.return_value = []

        with patch(
            "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
            return_value=mock_fp,
        ):
            result = await _fn(get_sec_filing_items)(ticker="AAPL")
        assert "error" in result
        assert "No filing items" in result["error"]

    @pytest.mark.asyncio
    async def test_double_close_on_fetch_items_exception(self):
        """BUG TEST: get_sec_filing_items has both explicit fp.close()
        calls and a `finally: fp.close()` block in the items fetch step.
        When fetch_sec_filing_items raises, fp.close() is called in the
        except block AND in the finally block — double close. This test
        ensures the tool doesn't crash despite the double close."""
        from quantstack.mcp.tools.qc_fundamentals import get_sec_filing_items

        filings_df = pd.DataFrame(
            [{"accession_number": "0001234567-24-000001", "type": "10-K"}]
        )
        mock_fp = _mock_fundamentals_provider(
            returns={"fetch_sec_filings": filings_df}
        )
        mock_fp.fetch_sec_filing_items.side_effect = RuntimeError("parse error")

        with patch(
            "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
            return_value=mock_fp,
        ):
            result = await _fn(get_sec_filing_items)(ticker="AAPL")

        assert "error" in result
        assert "parse error" in result["error"]
        # fp.close() was called at least once (may be called twice — that's the bug)
        assert mock_fp.close.called


# ---------------------------------------------------------------------------
# list_sec_filings
# ---------------------------------------------------------------------------


class TestListSecFilings:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import list_sec_filings

        filings_df = _sample_df(3)
        with _patch_fp(returns={"fetch_sec_filings": filings_df}):
            result = await _fn(list_sec_filings)(ticker="AAPL")
        assert "error" not in result
        assert result["rows"] == 3

    @pytest.mark.asyncio
    async def test_no_filings(self):
        from quantstack.mcp.tools.qc_fundamentals import list_sec_filings

        with _patch_fp():
            result = await _fn(list_sec_filings)(ticker="NODATA")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_company_facts
# ---------------------------------------------------------------------------


class TestGetCompanyFacts:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_company_facts

        facts = {"name": "Apple Inc", "sector": "Technology", "market_cap": 3e12}
        with _patch_fp(returns={"fetch_company_facts": facts}):
            result = await _fn(get_company_facts)(ticker="AAPL")
        assert "error" not in result
        assert result["facts"]["name"] == "Apple Inc"

    @pytest.mark.asyncio
    async def test_no_facts(self):
        from quantstack.mcp.tools.qc_fundamentals import get_company_facts

        with _patch_fp():
            result = await _fn(get_company_facts)(ticker="NODATA")
        assert "error" in result
        assert "No company facts" in result["error"]

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_company_facts

        with _patch_fp_none():
            result = await _fn(get_company_facts)(ticker="AAPL")
        assert "error" in result


# ---------------------------------------------------------------------------
# search_financial_statements
# ---------------------------------------------------------------------------


class TestSearchFinancialStatements:
    @pytest.mark.asyncio
    async def test_happy_path_above(self):
        from quantstack.mcp.tools.qc_fundamentals import search_financial_statements

        results_df = _sample_df(5)
        with _patch_fp(returns={"screen_stocks": results_df}):
            result = await _fn(search_financial_statements)(
                metric="return_on_equity",
                condition="above",
                value=0.15,
            )
        assert "error" not in result
        assert result["rows"] == 5
        assert result["condition"] == "above"

    @pytest.mark.asyncio
    async def test_below_condition(self):
        from quantstack.mcp.tools.qc_fundamentals import search_financial_statements

        results_df = _sample_df(2)
        with _patch_fp(returns={"screen_stocks": results_df}):
            result = await _fn(search_financial_statements)(
                metric="pe_ratio",
                condition="below",
                value=20.0,
            )
        assert "error" not in result
        assert result["condition"] == "below"

    @pytest.mark.asyncio
    async def test_with_sector_filter(self):
        from quantstack.mcp.tools.qc_fundamentals import search_financial_statements

        results_df = _sample_df(3)
        mock_fp = _mock_fundamentals_provider(
            returns={"screen_stocks": results_df}
        )

        with patch(
            "quantstack.mcp.tools.qc_fundamentals._get_fundamentals_provider",
            return_value=mock_fp,
        ):
            result = await _fn(search_financial_statements)(
                metric="return_on_equity",
                condition="above",
                value=0.15,
                sector="Technology",
            )
        assert "error" not in result
        assert result["sector"] == "Technology"

        # Verify that sector was passed in the filters
        call_args = mock_fp.screen_stocks.call_args
        filters_passed = call_args[0][0]
        assert filters_passed.get("sector") == "Technology"
        assert "return_on_equity_gt" in filters_passed

    @pytest.mark.asyncio
    async def test_no_matches(self):
        from quantstack.mcp.tools.qc_fundamentals import search_financial_statements

        with _patch_fp():
            result = await _fn(search_financial_statements)(
                metric="pe_ratio",
                condition="above",
                value=1000.0,
            )
        assert "error" in result
        assert "No companies matched" in result["error"]

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import search_financial_statements

        with _patch_fp_none():
            result = await _fn(search_financial_statements)(
                metric="pe_ratio",
                condition="above",
                value=10.0,
            )
        assert "error" in result


# ---------------------------------------------------------------------------
# get_interest_rates
# ---------------------------------------------------------------------------


class TestGetInterestRates:
    @pytest.mark.asyncio
    async def test_snapshot(self):
        from quantstack.mcp.tools.qc_fundamentals import get_interest_rates

        snapshot = {"fed_funds": 5.25, "treasury_10y": 4.5}
        with _patch_fp(returns={"fetch_interest_rates_snapshot": snapshot}):
            result = await _fn(get_interest_rates)(snapshot=True)
        assert "error" not in result
        assert result["mode"] == "snapshot"

    @pytest.mark.asyncio
    async def test_historical(self):
        from quantstack.mcp.tools.qc_fundamentals import get_interest_rates

        hist_df = _sample_df(20)
        with _patch_fp(returns={"fetch_interest_rates": hist_df}):
            result = await _fn(get_interest_rates)(snapshot=False)
        assert "error" not in result
        assert result["mode"] == "historical"

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_interest_rates

        with _patch_fp_none():
            result = await _fn(get_interest_rates)()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_crypto_prices
# ---------------------------------------------------------------------------


class TestGetCryptoPrices:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_crypto_prices

        crypto_df = _sample_df(30, columns=["open", "high", "low", "close", "volume"])
        with _patch_fp(returns={"fetch_crypto_prices": crypto_df}):
            result = await _fn(get_crypto_prices)(ticker="BTC-USD")
        assert "error" not in result
        assert result["rows"] == 30

    @pytest.mark.asyncio
    async def test_no_data(self):
        from quantstack.mcp.tools.qc_fundamentals import get_crypto_prices

        with _patch_fp():
            result = await _fn(get_crypto_prices)(ticker="FAKE-USD")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from quantstack.mcp.tools.qc_fundamentals import get_crypto_prices

        with _patch_fp_none():
            result = await _fn(get_crypto_prices)(ticker="BTC-USD")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_price_snapshot
# ---------------------------------------------------------------------------


class TestGetPriceSnapshot:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.qc_fundamentals import get_price_snapshot

        quote_df = pd.DataFrame(
            [{"symbol": "AAPL", "price": 195.50, "volume": 50_000_000}]
        )
        mock_client = MagicMock()
        mock_client.fetch_bulk_quotes.return_value = quote_df

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_price_snapshot)(ticker="AAPL")
        assert "error" not in result
        assert result["ticker"] == "AAPL"
        assert "snapshot" in result
        # symbol key should be excluded from snapshot
        assert "symbol" not in result["snapshot"]

    @pytest.mark.asyncio
    async def test_empty_response(self):
        from quantstack.mcp.tools.qc_fundamentals import get_price_snapshot

        mock_client = MagicMock()
        mock_client.fetch_bulk_quotes.return_value = pd.DataFrame()

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_price_snapshot)(ticker="NODATA")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_case_insensitive_symbol_match(self):
        """Ticker matching should be case-insensitive."""
        from quantstack.mcp.tools.qc_fundamentals import get_price_snapshot

        quote_df = pd.DataFrame(
            [{"symbol": "aapl", "price": 195.50, "volume": 50_000_000}]
        )
        mock_client = MagicMock()
        mock_client.fetch_bulk_quotes.return_value = quote_df

        with patch(
            "quantstack.mcp.tools.qc_fundamentals.AlphaVantageClient",
            return_value=mock_client,
        ):
            result = await _fn(get_price_snapshot)(ticker="AAPL")
        assert "error" not in result
        # Should match despite case difference
        assert "snapshot" in result
