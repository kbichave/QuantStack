# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Alpha Vantage Tools - CrewAI tools for Alpha Vantage API integration.

Provides tools for:
- News sentiment analysis
- Earnings calendar
- IPO calendar
- Economic indicators
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from quant_pod.crewai_compat import BaseTool


# =============================================================================
# ALPHA VANTAGE CLIENT
# =============================================================================


class AlphaVantageClient:
    """Client for Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key (or ALPHAVANTAGE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "demo")
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request."""
        params["apikey"] = self.api_key

        client = self._get_client()
        response = client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        return response.json()

    def _request_csv(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Make API request expecting CSV response."""
        params["apikey"] = self.api_key

        client = self._get_client()
        response = client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        # Parse CSV
        content = response.text
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)

    def get_news_sentiment(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        sort: str = "LATEST",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get news with sentiment analysis.

        Args:
            tickers: Comma-separated tickers (e.g., "AAPL,MSFT")
            topics: Topic filter (e.g., "technology", "earnings")
            time_from: Start time in YYYYMMDDTHHMM format
            time_to: End time in YYYYMMDDTHHMM format
            sort: "LATEST", "EARLIEST", or "RELEVANCE"
            limit: Max articles to return (max 1000)

        Returns:
            Dict with feed, sentiment scores, etc.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "sort": sort,
            "limit": min(limit, 1000),
        }

        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        return self._request(params)

    def get_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        horizon: str = "3month",
    ) -> List[Dict[str, str]]:
        """
        Get upcoming earnings calendar.

        Args:
            symbol: Specific symbol (or None for all)
            horizon: "3month", "6month", or "12month"

        Returns:
            List of earnings events with dates and estimates
        """
        params = {
            "function": "EARNINGS_CALENDAR",
            "horizon": horizon,
        }

        if symbol:
            params["symbol"] = symbol.upper()

        return self._request_csv(params)

    def get_ipo_calendar(self) -> List[Dict[str, str]]:
        """
        Get upcoming IPO calendar.

        Returns:
            List of upcoming IPO events
        """
        params = {"function": "IPO_CALENDAR"}
        return self._request_csv(params)

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get historical earnings for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quarterly and annual earnings
        """
        params = {
            "function": "EARNINGS",
            "symbol": symbol.upper(),
        }
        return self._request(params)

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview and fundamentals.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with company info, ratios, etc.
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol.upper(),
        }
        return self._request(params)


# Global client instance
_av_client: Optional[AlphaVantageClient] = None


def get_alphavantage_client() -> AlphaVantageClient:
    """Get or create the global Alpha Vantage client."""
    global _av_client
    if _av_client is None:
        _av_client = AlphaVantageClient()
    return _av_client


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class FetchNewsSentimentInput(BaseModel):
    """Input for fetch_news_sentiment tool."""

    tickers: Optional[str] = Field(
        None, description="Comma-separated stock tickers (e.g., 'SPY,AAPL,MSFT')"
    )
    topics: Optional[str] = Field(
        None,
        description="Topics: blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets, economy_fiscal, economy_monetary, economy_macro, energy_transportation, finance, life_sciences, manufacturing, real_estate, retail_wholesale, technology",
    )
    limit: int = Field(20, description="Max number of articles to return")


class FetchEarningsCalendarInput(BaseModel):
    """Input for fetch_earnings_calendar tool."""

    symbol: Optional[str] = Field(
        None, description="Specific symbol to check (or None for all upcoming)"
    )
    horizon: str = Field(
        "3month", description="Time horizon: 3month, 6month, or 12month"
    )


class FetchUpcomingEarningsInput(BaseModel):
    """Input for fetch_upcoming_earnings tool."""

    symbols: str = Field(
        ..., description="Comma-separated symbols to check for upcoming earnings"
    )
    days_ahead: int = Field(7, description="Number of days ahead to check")


class FetchCompanyOverviewInput(BaseModel):
    """Input for company overview tool."""

    symbol: str = Field(..., description="Stock symbol")


# =============================================================================
# TOOL CLASSES
# =============================================================================


class FetchNewsSentimentTool(BaseTool):
    """Tool to fetch news with sentiment analysis."""

    name: str = "fetch_news_sentiment"
    description: str = """Fetch recent news articles with AI-generated sentiment scores.
Use this to understand market sentiment around specific stocks or topics.
Returns news headlines, summaries, sentiment scores, and ticker relevance."""
    args_schema: Type[BaseModel] = FetchNewsSentimentInput

    def _run(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Fetch news sentiment."""
        try:
            client = get_alphavantage_client()
            result = client.get_news_sentiment(
                tickers=tickers,
                topics=topics,
                limit=limit,
            )

            # Process and format results
            feed = result.get("feed", [])
            items = result.get("items", "0")

            news_items = []
            for article in feed[:limit]:
                item = {
                    "title": article.get("title", ""),
                    "summary": (
                        article.get("summary", "")[:200] + "..."
                        if len(article.get("summary", "")) > 200
                        else article.get("summary", "")
                    ),
                    "source": article.get("source", ""),
                    "time_published": article.get("time_published", ""),
                    "overall_sentiment": article.get("overall_sentiment_label", ""),
                    "sentiment_score": article.get("overall_sentiment_score", 0),
                }

                # Add ticker-specific sentiment if available
                ticker_sentiment = article.get("ticker_sentiment", [])
                if ticker_sentiment:
                    item["ticker_sentiments"] = [
                        {
                            "ticker": ts.get("ticker"),
                            "sentiment": ts.get("ticker_sentiment_label"),
                            "relevance": ts.get("relevance_score"),
                        }
                        for ts in ticker_sentiment[:5]
                    ]

                news_items.append(item)

            return json.dumps(
                {
                    "success": True,
                    "total_items": items,
                    "returned_items": len(news_items),
                    "sentiment_model": result.get("sentiment_score_definition", ""),
                    "news": news_items,
                }
            )

        except Exception as e:
            logger.error(f"News sentiment fetch failed: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                }
            )


class FetchEarningsCalendarTool(BaseTool):
    """Tool to fetch upcoming earnings calendar."""

    name: str = "fetch_earnings_calendar"
    description: str = """Fetch upcoming earnings announcements.
Use this to find out when companies are reporting earnings.
Can filter by symbol or get all upcoming earnings."""
    args_schema: Type[BaseModel] = FetchEarningsCalendarInput

    def _run(
        self,
        symbol: Optional[str] = None,
        horizon: str = "3month",
    ) -> str:
        """Fetch earnings calendar."""
        try:
            client = get_alphavantage_client()
            earnings = client.get_earnings_calendar(
                symbol=symbol,
                horizon=horizon,
            )

            # Format results
            events = []
            for event in earnings[:50]:  # Limit for response size
                events.append(
                    {
                        "symbol": event.get("symbol", ""),
                        "name": event.get("name", ""),
                        "report_date": event.get("reportDate", ""),
                        "fiscal_date_ending": event.get("fiscalDateEnding", ""),
                        "estimate": event.get("estimate", ""),
                        "currency": event.get("currency", "USD"),
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "horizon": horizon,
                    "count": len(events),
                    "earnings": events,
                }
            )

        except Exception as e:
            logger.error(f"Earnings calendar fetch failed: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                }
            )


class FetchUpcomingEarningsTool(BaseTool):
    """Tool to check for upcoming earnings for specific symbols."""

    name: str = "fetch_upcoming_earnings"
    description: str = """Check if specific symbols have earnings coming up.
Use this to avoid trading around earnings if needed.
Returns dates and estimates for upcoming earnings reports."""
    args_schema: Type[BaseModel] = FetchUpcomingEarningsInput

    def _run(
        self,
        symbols: str,
        days_ahead: int = 7,
    ) -> str:
        """Check upcoming earnings for symbols."""
        try:
            client = get_alphavantage_client()

            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            cutoff_date = datetime.now() + timedelta(days=days_ahead)

            results = {}
            for symbol in symbol_list:
                try:
                    earnings = client.get_earnings_calendar(
                        symbol=symbol,
                        horizon="3month",
                    )

                    upcoming = []
                    for event in earnings:
                        report_date_str = event.get("reportDate", "")
                        if report_date_str:
                            try:
                                report_date = datetime.strptime(
                                    report_date_str, "%Y-%m-%d"
                                )
                                if report_date <= cutoff_date:
                                    upcoming.append(
                                        {
                                            "date": report_date_str,
                                            "estimate": event.get("estimate", ""),
                                            "days_until": (
                                                report_date - datetime.now()
                                            ).days,
                                        }
                                    )
                            except ValueError:
                                pass

                    results[symbol] = {
                        "has_upcoming": len(upcoming) > 0,
                        "events": upcoming,
                    }

                except Exception as e:
                    results[symbol] = {
                        "has_upcoming": False,
                        "error": str(e),
                    }

            # Summary
            symbols_with_earnings = [
                s for s, r in results.items() if r.get("has_upcoming")
            ]

            return json.dumps(
                {
                    "success": True,
                    "days_ahead": days_ahead,
                    "symbols_checked": len(symbol_list),
                    "symbols_with_upcoming_earnings": symbols_with_earnings,
                    "details": results,
                }
            )

        except Exception as e:
            logger.error(f"Upcoming earnings check failed: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                }
            )


class FetchIPOCalendarTool(BaseTool):
    """Tool to fetch upcoming IPO calendar."""

    name: str = "fetch_ipo_calendar"
    description: str = """Fetch upcoming IPO (Initial Public Offering) calendar.
Use this to track new companies coming to market."""
    args_schema: Type[BaseModel] = None

    def _run(self) -> str:
        """Fetch IPO calendar."""
        try:
            client = get_alphavantage_client()
            ipos = client.get_ipo_calendar()

            events = []
            for ipo in ipos[:30]:  # Limit for response size
                events.append(
                    {
                        "symbol": ipo.get("symbol", ""),
                        "name": ipo.get("name", ""),
                        "ipo_date": ipo.get("ipoDate", ""),
                        "price_range_low": ipo.get("priceRangeLow", ""),
                        "price_range_high": ipo.get("priceRangeHigh", ""),
                        "currency": ipo.get("currency", "USD"),
                        "exchange": ipo.get("exchange", ""),
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "count": len(events),
                    "ipos": events,
                }
            )

        except Exception as e:
            logger.error(f"IPO calendar fetch failed: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                }
            )


class FetchCompanyOverviewTool(BaseTool):
    """Tool to fetch company overview and fundamentals."""

    name: str = "fetch_company_overview"
    description: str = """Fetch company overview including financials, ratios, and metrics.
Use this to understand a company's fundamentals before trading."""
    args_schema: Type[BaseModel] = FetchCompanyOverviewInput

    def _run(self, symbol: str) -> str:
        """Fetch company overview."""
        try:
            client = get_alphavantage_client()
            data = client.get_company_overview(symbol)

            if not data or "Symbol" not in data:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"No data found for {symbol}",
                    }
                )

            # Extract key metrics
            overview = {
                "symbol": data.get("Symbol"),
                "name": data.get("Name"),
                "description": (
                    data.get("Description", "")[:500] + "..."
                    if len(data.get("Description", "")) > 500
                    else data.get("Description", "")
                ),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "market_cap": data.get("MarketCapitalization"),
                "pe_ratio": data.get("PERatio"),
                "peg_ratio": data.get("PEGRatio"),
                "book_value": data.get("BookValue"),
                "dividend_yield": data.get("DividendYield"),
                "eps": data.get("EPS"),
                "revenue_ttm": data.get("RevenueTTM"),
                "profit_margin": data.get("ProfitMargin"),
                "52_week_high": data.get("52WeekHigh"),
                "52_week_low": data.get("52WeekLow"),
                "50_day_ma": data.get("50DayMovingAverage"),
                "200_day_ma": data.get("200DayMovingAverage"),
                "beta": data.get("Beta"),
                "analyst_target_price": data.get("AnalystTargetPrice"),
            }

            return json.dumps(
                {
                    "success": True,
                    "overview": overview,
                }
            )

        except Exception as e:
            logger.error(f"Company overview fetch failed: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                }
            )


# =============================================================================
# TOOL FACTORY FUNCTIONS
# =============================================================================


def fetch_news_sentiment_tool() -> FetchNewsSentimentTool:
    """Get the news sentiment tool instance."""
    return FetchNewsSentimentTool()


def fetch_earnings_calendar_tool() -> FetchEarningsCalendarTool:
    """Get the earnings calendar tool instance."""
    return FetchEarningsCalendarTool()


def fetch_upcoming_earnings_tool() -> FetchUpcomingEarningsTool:
    """Get the upcoming earnings tool instance."""
    return FetchUpcomingEarningsTool()


def fetch_ipo_calendar_tool() -> FetchIPOCalendarTool:
    """Get the IPO calendar tool instance."""
    return FetchIPOCalendarTool()


def fetch_company_overview_tool() -> FetchCompanyOverviewTool:
    """Get the company overview tool instance."""
    return FetchCompanyOverviewTool()
