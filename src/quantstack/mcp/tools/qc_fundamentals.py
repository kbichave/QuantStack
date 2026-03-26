# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP fundamentals tools — financial statements, metrics, earnings,
insider trades, institutional ownership, analyst estimates, and company news.

Data sourced from FinancialDatasets.ai.  Results are cached in PostgreSQL so
subsequent calls for the same ticker are served locally.
"""

from typing import Any

from loguru import logger

from quantstack.mcp._helpers import (
    ServerContext,
    _dataframe_to_dict,
    _get_reader,
    _get_writer,
)
from quantstack.data.fetcher import AlphaVantageClient
from quantstack.data.fundamentals import FundamentalsProvider
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



def _get_fundamentals_provider():
    """Lazily construct a FundamentalsProvider from server settings."""
    ctx: ServerContext = mcp.context
    settings = ctx.settings
    api_key = settings.financial_datasets.api_key
    if not api_key:
        return None
    return FundamentalsProvider(
        api_key=api_key,
        base_url=settings.financial_datasets.base_url,
        rate_limit_rpm=settings.financial_datasets.rate_limit_rpm,
    )


# =============================================================================
# FINANCIAL STATEMENTS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_financial_statements(
    ticker: str,
    statement_type: str = "income",
    period: str = "annual",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Fetch financial statements (income, balance sheet, or cash flow) for a company.

    Fetches from FinancialDatasets.ai and caches in PostgreSQL. Subsequent calls
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

        # Persist to PostgreSQL.
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


@domain(Domain.DATA)
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


@domain(Domain.DATA)
@mcp.tool()
async def get_earnings_data(
    ticker: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Fetch earnings data with estimates and surprises for a company.

    Uses Alpha Vantage EARNINGS endpoint (free tier). Returns both annual
    and quarterly EPS history with reported vs estimated and surprise %.

    Args:
        ticker: Stock symbol
        limit: Number of quarterly records to return (most recent first)

    Returns:
        Dictionary with earnings data including estimates, reported EPS, and surprise
    """
    try:
        client = AlphaVantageClient()
        data = client.fetch_earnings_history(ticker)
        if not data:
            return {"error": "No earnings data found", "ticker": ticker}

        quarterly = data.get("quarterlyEarnings", [])[:limit]
        annual = data.get("annualEarnings", [])

        return {
            "ticker": ticker,
            "source": "alpha_vantage",
            "quarterly_count": len(quarterly),
            "annual_count": len(annual),
            "quarterly_earnings": quarterly,
            "annual_earnings": annual,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# =============================================================================
# INSIDER TRADES
# =============================================================================


@domain(Domain.DATA)
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


@domain(Domain.DATA)
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


@domain(Domain.DATA)
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


@domain(Domain.DATA)
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


@domain(Domain.DATA)
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
            return {
                "error": "No stocks matched the screening criteria",
                "filters": filters,
            }

        return {
            "rows": len(df),
            "filters": filters,
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "filters": filters}
    finally:
        fp.close()


# =============================================================================
# SEGMENTED REVENUES
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_segmented_revenues(
    ticker: str,
    period: str = "annual",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Fetch segmented revenue data (business segments, geographic breakdown).

    Args:
        ticker: Stock symbol (e.g., "AAPL", "MSFT")
        period: "annual" or "quarterly"
        limit: Number of periods to return

    Returns:
        Dictionary with segmented revenue data broken down by business line and geography
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_segmented_revenues(ticker, period, limit)
        if df.empty:
            return {"error": "No segmented revenue data found", "ticker": ticker}

        return {
            "ticker": ticker,
            "period": period,
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# EARNINGS PRESS RELEASES
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_earnings_press_releases(
    ticker: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Fetch earnings press releases with management commentary.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")
        limit: Number of press releases to return

    Returns:
        Dictionary with earnings press release data including management guidance
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_earnings_press_releases(ticker, limit)
        if df.empty:
            return {"error": "No earnings press releases found", "ticker": ticker}

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
# SEC FILING ITEMS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_sec_filing_items(
    ticker: str,
    filing_type: str = "10-K",
    section: str | None = None,
) -> dict[str, Any]:
    """
    Fetch SEC filing content at section level.

    First fetches the latest filing of the given type, then retrieves the
    section-level items from that filing.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        filing_type: SEC filing type (e.g., "10-K", "10-Q", "8-K")
        section: Optional specific section to retrieve (e.g., "1A" for Risk Factors)

    Returns:
        Dictionary with SEC filing section content and metadata
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    # Step 1: Fetch the latest filing to get accession_number.
    try:
        filings_df = fp.fetch_sec_filings(ticker, limit=1, filing_type=filing_type)
    except Exception as e:
        fp.close()
        return {
            "error": f"Failed to fetch {filing_type} filings for {ticker}: {e}",
            "ticker": ticker,
        }

    if filings_df.empty:
        fp.close()
        return {
            "error": f"No {filing_type} filings found for {ticker}",
            "ticker": ticker,
        }

    accession_number = filings_df.iloc[0].get("accession_number")
    if not accession_number:
        fp.close()
        return {
            "error": f"No accession number in {filing_type} filing for {ticker}",
            "ticker": ticker,
        }

    # Step 2: Fetch the filing items (section text). Reported separately so step 1
    # metadata is visible in partial-failure diagnostics.
    try:
        items = fp.fetch_sec_filing_items(str(accession_number), section)
    except Exception as e:
        fp.close()
        return {
            "error": f"Failed to fetch items for accession {accession_number}: {e}",
            "ticker": ticker,
            "filing_type": filing_type,
            "accession_number": str(accession_number),
        }
    finally:
        fp.close()

    if not items:
        return {
            "error": f"No filing items found for accession {accession_number}",
            "ticker": ticker,
        }

    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "accession_number": str(accession_number),
        "section": section,
        "items": items,
    }


# =============================================================================
# INTEREST RATES
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_interest_rates(
    snapshot: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Fetch interest rate data (historical or current snapshot).

    Args:
        snapshot: If True, return current rates only. If False, return historical series.
        start_date: Start date for historical data (YYYY-MM-DD). Ignored if snapshot=True.
        end_date: End date for historical data (YYYY-MM-DD). Ignored if snapshot=True.

    Returns:
        Dictionary with interest rate data (Fed Funds, Treasury yields, etc.)
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured"}

    try:
        if snapshot:
            data = fp.fetch_interest_rates_snapshot()
            if not data:
                return {"error": "No interest rate snapshot available"}
            return {
                "mode": "snapshot",
                "data": data,
            }

        df = fp.fetch_interest_rates(start_date, end_date)
        if df.empty:
            return {"error": "No historical interest rate data found"}

        return {
            "mode": "historical",
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        fp.close()


# =============================================================================
# CRYPTO PRICES
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_crypto_prices(
    ticker: str,
    interval: str = "day",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Fetch cryptocurrency OHLCV price data.

    Args:
        ticker: Crypto symbol (e.g., "BTC-USD", "ETH-USD")
        interval: Bar interval — "day", "hour", "minute"
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary with cryptocurrency price bars (open, high, low, close, volume)
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_crypto_prices(ticker, interval, start_date, end_date)
        if df.empty:
            return {"error": "No crypto price data found", "ticker": ticker}

        return {
            "ticker": ticker,
            "interval": interval,
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# PRICE SNAPSHOT
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_price_snapshot(
    ticker: str,
) -> dict[str, Any]:
    """
    Get current price snapshot: last price, open, high, low, change, change%, volume.

    Lightweight alternative to a full OHLCV fetch when you only need the
    latest quote. Uses Alpha Vantage REALTIME_BULK_QUOTES (premium).

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")

    Returns:
        Dictionary with price snapshot data (price, open, high, low, volume,
        previous_close, change, change_percent, latest_day)
    """
    try:
        client = AlphaVantageClient()
        df = client.fetch_bulk_quotes([ticker])
        if df.empty:
            return {"error": "No price snapshot available", "ticker": ticker}

        row = df[df["symbol"].str.upper() == ticker.upper()]
        if row.empty:
            row = df.iloc[[0]]

        snapshot = row.iloc[0].to_dict()
        return {
            "ticker": ticker,
            "source": "alpha_vantage",
            "snapshot": {k: v for k, v in snapshot.items() if k != "symbol"},
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# =============================================================================
# LIST SEC FILINGS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def list_sec_filings(
    ticker: str,
    filing_type: str = "10-K",
    limit: int = 5,
) -> dict[str, Any]:
    """
    List SEC filings for a ticker (metadata only, not content).

    Returns filing dates, accession numbers, and filing types. Use this to
    discover filings before fetching section-level content with get_sec_filing_items.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        filing_type: SEC filing type filter (e.g., "10-K", "10-Q", "8-K")
        limit: Number of filings to return (default: 5)

    Returns:
        Dictionary with filing metadata (accession numbers, dates, types)
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        df = fp.fetch_sec_filings(ticker, limit=limit, filing_type=filing_type)
        if df.empty:
            return {
                "error": f"No {filing_type} filings found",
                "ticker": ticker,
                "filing_type": filing_type,
            }

        return {
            "ticker": ticker,
            "filing_type": filing_type,
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# COMPANY FACTS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def get_company_facts(
    ticker: str,
) -> dict[str, Any]:
    """
    Get company facts: sector, industry, description, market cap, employee count, headquarters.

    Calls FinancialDatasets.ai company facts endpoint. Returns structured
    company overview data useful for fundamental screening and sector analysis.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")

    Returns:
        Dictionary with company facts (name, sector, industry, market_cap, employees, etc.)
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured", "ticker": ticker}

    try:
        facts = fp.fetch_company_facts(ticker)
        if not facts:
            return {"error": "No company facts found", "ticker": ticker}

        return {
            "ticker": ticker,
            "source": "api",
            "facts": facts,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}
    finally:
        fp.close()


# =============================================================================
# SEARCH FINANCIAL STATEMENTS
# =============================================================================


@domain(Domain.DATA)
@mcp.tool()
async def search_financial_statements(
    metric: str,
    condition: str = "above",
    value: float = 0.0,
    sector: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for companies matching a financial metric condition.

    Builds a screening query from the metric, condition, and value parameters
    and runs it against the FinancialDatasets.ai screener. Useful for
    cross-company comparison (e.g., "ROE > 0.15 in Technology sector").

    Args:
        metric: Financial metric name (e.g., "return_on_equity", "pe_ratio",
                "revenue_growth", "debt_to_equity", "free_cash_flow_per_share")
        condition: "above" or "below" — direction of the filter
        value: Threshold value for the condition
        sector: Optional sector filter (e.g., "Technology", "Healthcare")
        limit: Maximum number of results (default: 20)

    Returns:
        Dictionary with matching companies and their metric values
    """
    fp = _get_fundamentals_provider()
    if fp is None:
        return {"error": "FINANCIAL_DATASETS_API_KEY not configured"}

    try:
        # Build the screener filter dict.
        suffix = "gt" if condition.lower() in ("above", "gt", ">") else "lt"
        filters: dict[str, Any] = {
            f"{metric}_{suffix}": value,
            "limit": limit,
        }
        if sector:
            filters["sector"] = sector

        df = fp.screen_stocks(filters)
        if df.empty:
            return {
                "error": "No companies matched the criteria",
                "metric": metric,
                "condition": condition,
                "value": value,
                "sector": sector,
            }

        return {
            "metric": metric,
            "condition": condition,
            "value": value,
            "sector": sector,
            "rows": len(df),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "metric": metric}
    finally:
        fp.close()
