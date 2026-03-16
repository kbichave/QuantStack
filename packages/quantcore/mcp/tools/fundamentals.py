# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP fundamentals tools — financial statements, metrics, earnings,
insider trades, institutional ownership, analyst estimates, and company news.

Data sourced from FinancialDatasets.ai.  Results are cached in DuckDB so
subsequent calls for the same ticker are served locally.
"""

from typing import Any

from loguru import logger

from quantcore.mcp._helpers import (
    ServerContext,
    _dataframe_to_dict,
    _get_reader,
    _get_writer,
)
from quantcore.mcp.server import mcp


def _get_fundamentals_provider():
    """Lazily construct a FundamentalsProvider from server settings."""
    ctx: ServerContext = mcp.context
    settings = ctx.settings
    api_key = settings.financial_datasets.api_key
    if not api_key:
        return None
    from quantcore.data.fundamentals import FundamentalsProvider

    return FundamentalsProvider(
        api_key=api_key,
        base_url=settings.financial_datasets.base_url,
        rate_limit_rpm=settings.financial_datasets.rate_limit_rpm,
    )


# =============================================================================
# FINANCIAL STATEMENTS
# =============================================================================


@mcp.tool()
async def get_financial_statements(
    ticker: str,
    statement_type: str = "income",
    period: str = "annual",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Fetch financial statements (income, balance sheet, or cash flow) for a company.

    Fetches from FinancialDatasets.ai and caches in DuckDB. Subsequent calls
    for the same ticker are served from local storage.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")
        statement_type: "income", "balance", or "cashflow"
        period: "annual", "quarterly", or "ttm"
        limit: Number of periods to return (default: 10)

    Returns:
        Dictionary with financial statement data
    """
    # Try local cache first.
    reader = _get_reader()
    cached = reader.load_financial_statements(ticker, statement_type, period, limit)
    if not cached.empty:
        return {
            "ticker": ticker,
            "statement_type": statement_type,
            "period": period,
            "source": "cache",
            "rows": len(cached),
            "data": _dataframe_to_dict(cached),
        }

    # Fetch from API.
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        fetch_map = {
            "income": fp.fetch_income_statements,
            "balance": fp.fetch_balance_sheets,
            "cashflow": fp.fetch_cash_flows,
        }
        fetcher = fetch_map.get(statement_type)
        if fetcher is None:
            return {
                "error": f"Unknown statement_type: {statement_type}. Use income, balance, or cashflow.",
                "ticker": ticker,
            }

        df = fetcher(ticker, period, limit)
        if df.empty:
            return {"error": f"No {statement_type} statements found", "ticker": ticker}

        # Persist to DuckDB.
        try:
            writer = _get_writer()
            writer.save_financial_statements(df)
            writer.close()
        except Exception as exc:
            logger.warning(f"Failed to cache financial statements: {exc}")

        return {
            "ticker": ticker,
            "statement_type": statement_type,
            "period": period,
            "source": "api",
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# FINANCIAL METRICS
# =============================================================================


@mcp.tool()
async def get_financial_metrics(
    ticker: str,
    period: str = "annual",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Fetch financial metrics (valuation, profitability, leverage, growth) for a company.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")
        period: "annual" or "quarterly"
        limit: Number of periods to return

    Returns:
        Dictionary with financial metrics data
    """
    reader = _get_reader()
    cached = reader.load_financial_metrics(ticker, period, limit)
    if not cached.empty:
        return {
            "ticker": ticker,
            "source": "cache",
            "rows": len(cached),
            "data": _dataframe_to_dict(cached),
        }

    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_financial_metrics(ticker, period, limit)
        if df.empty:
            return {"error": "No financial metrics found", "ticker": ticker}

        try:
            writer = _get_writer()
            writer.save_financial_metrics(df)
            writer.close()
        except Exception as exc:
            logger.warning(f"Failed to cache financial metrics: {exc}")

        return {
            "ticker": ticker,
            "source": "api",
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# EARNINGS
# =============================================================================


@mcp.tool()
async def get_earnings_data(
    ticker: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Fetch earnings data with estimates and surprises for a company.

    Args:
        ticker: Stock symbol
        limit: Number of earnings records

    Returns:
        Dictionary with earnings data including estimates, reported EPS, and surprise
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_earnings(ticker, limit)
        if df.empty:
            return {"error": "No earnings data found", "ticker": ticker}

        return {
            "ticker": ticker,
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# INSIDER TRADES
# =============================================================================


@mcp.tool()
async def get_insider_trades(
    ticker: str,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Fetch insider trade transactions for a company.

    Args:
        ticker: Stock symbol
        limit: Number of insider trade records

    Returns:
        Dictionary with insider trade data (owner, transaction type, shares, price)
    """
    reader = _get_reader()
    cached = reader.load_insider_trades(ticker, limit=limit)
    if not cached.empty:
        return {
            "ticker": ticker,
            "source": "cache",
            "rows": len(cached),
            "data": _dataframe_to_dict(cached),
        }

    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_insider_trades(ticker, limit)
        if df.empty:
            return {"error": "No insider trades found", "ticker": ticker}

        try:
            writer = _get_writer()
            writer.save_insider_trades(df)
            writer.close()
        except Exception as exc:
            logger.warning(f"Failed to cache insider trades: {exc}")

        return {
            "ticker": ticker,
            "source": "api",
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# INSTITUTIONAL OWNERSHIP
# =============================================================================


@mcp.tool()
async def get_institutional_ownership(
    ticker: str,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Fetch institutional ownership data for a company.

    Args:
        ticker: Stock symbol
        limit: Number of institutional holder records

    Returns:
        Dictionary with institutional ownership data (investor, shares, value)
    """
    reader = _get_reader()
    cached = reader.load_institutional_ownership(ticker, limit)
    if not cached.empty:
        return {
            "ticker": ticker,
            "source": "cache",
            "rows": len(cached),
            "data": _dataframe_to_dict(cached),
        }

    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_institutional_ownership(ticker, limit)
        if df.empty:
            return {"error": "No institutional ownership data found", "ticker": ticker}

        try:
            writer = _get_writer()
            writer.save_institutional_ownership(df)
            writer.close()
        except Exception as exc:
            logger.warning(f"Failed to cache institutional ownership: {exc}")

        return {
            "ticker": ticker,
            "source": "api",
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# ANALYST ESTIMATES
# =============================================================================


@mcp.tool()
async def get_analyst_estimates(
    ticker: str,
) -> dict[str, Any]:
    """
    Fetch analyst consensus estimates for a company.

    Args:
        ticker: Stock symbol

    Returns:
        Dictionary with analyst estimates (consensus, high, low, number of analysts)
    """
    reader = _get_reader()
    cached = reader.load_analyst_estimates(ticker)
    if not cached.empty:
        return {
            "ticker": ticker,
            "source": "cache",
            "rows": len(cached),
            "data": _dataframe_to_dict(cached),
        }

    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_analyst_estimates(ticker)
        if df.empty:
            return {"error": "No analyst estimates found", "ticker": ticker}

        try:
            writer = _get_writer()
            writer.save_analyst_estimates(df)
            writer.close()
        except Exception as exc:
            logger.warning(f"Failed to cache analyst estimates: {exc}")

        return {
            "ticker": ticker,
            "source": "api",
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# COMPANY NEWS
# =============================================================================


@mcp.tool()
async def get_company_news(
    ticker: str,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Fetch recent company news articles.

    Args:
        ticker: Stock symbol
        limit: Number of news articles to return

    Returns:
        Dictionary with company news articles
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_company_news(ticker, limit)
        if df.empty:
            return {"error": "No company news found", "ticker": ticker}

        return {
            "ticker": ticker,
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# STOCK SCREENER
# =============================================================================


@mcp.tool()
async def screen_stocks(
    filters: dict[str, Any],
) -> dict[str, Any]:
    """
    Screen stocks by financial criteria using FinancialDatasets.ai screener.

    Args:
        filters: Dictionary of screening criteria (e.g., {"market_cap_gt": 1e9, "pe_ratio_lt": 20})

    Returns:
        Dictionary with matching stocks
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured"}

    try:
        df = fp.screen_stocks(filters)
        if df.empty:
            return {"error": "No stocks matched the screening criteria", "filters": filters}

        return {
            "rows": len(df),
            "filters": filters,
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "filters": filters}
    finally:
        fp.close()
