"""Data fetch tools for LangGraph agents."""

import json
from datetime import date, datetime, timedelta
from typing import Annotated, Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import Field

from quantstack.config.settings import get_settings
from quantstack.data.base import AssetClass
from quantstack.data.registry import DataProviderRegistry
from quantstack.signal_engine.collectors.commodity import collect_commodity_signals
from quantstack.signal_engine.collectors.earnings_momentum import collect_earnings_momentum
from quantstack.signal_engine.collectors.put_call_ratio import collect_put_call_ratio
from quantstack.tools._helpers import _dataframe_to_dict, _get_reader, _get_writer, _parse_timeframe


def _get_data_registry() -> DataProviderRegistry:
    """Get a DataProviderRegistry from current settings."""
    return DataProviderRegistry.from_settings(get_settings())


@tool
async def fetch_market_data(
    symbol: Annotated[str, Field(description="Ticker symbol to fetch price data for, e.g. 'SPY', 'AAPL', 'QQQ'")],
    timeframe: Annotated[str, Field(description="Candle interval: 'daily', 'weekly', '1h', or '4h'. Defaults to 'daily'")] = "daily",
    outputsize: Annotated[str, Field(description="Data range: 'compact' for ~6 months of recent bars, 'full' for 5+ years of history")] = "compact",
) -> str:
    """Fetch OHLCV candlestick market data (open, high, low, close, volume) for a stock or ETF from the configured data provider chain. Use when you need historical price bars for charting, backtesting, indicator computation, or trend analysis. Returns JSON with timestamped rows, date range, and row count. Persists data locally for subsequent load_market_data calls. Synonyms: price history, candles, bars, quotes, historical data, time series."""
    try:
        registry = _get_data_registry()
        tf = _parse_timeframe(timeframe)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 6 if outputsize == "full" else 180)

        df = registry.fetch_ohlcv(symbol, AssetClass.EQUITY, tf, start_date, end_date)

        if df.empty:
            return json.dumps({"error": f"No data returned for {symbol}", "symbol": symbol})

        try:
            writer = _get_writer()
            writer.save_ohlcv(df, symbol, tf)
            writer.close()
        except Exception as store_exc:
            logger.warning(f"Data fetched but failed to persist locally: {store_exc}")

        result = {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def fetch_fundamentals(
    symbol: Annotated[str, Field(description="Ticker symbol to retrieve fundamental data for, e.g. 'AAPL', 'MSFT'")],
) -> str:
    """Retrieve fundamental financial data including income statement, balance sheet, and valuation metrics for a stock. Use when evaluating company health, profitability, or intrinsic value. Returns JSON with earnings, revenue, profit margins, P/E ratio, debt levels, and other key financial ratios. Synonyms: financials, balance sheet, income statement, valuation, company data, financial health."""
    try:
        registry = _get_data_registry()
        data = registry.fetch_fundamentals(symbol)
        result = {"symbol": symbol, **data} if isinstance(data, dict) else {"symbol": symbol, "data": data}
    except Exception as e:
        logger.error(f"fetch_fundamentals({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def fetch_earnings_data(
    symbol: Annotated[str, Field(description="Ticker symbol to fetch earnings data for, e.g. 'AAPL', 'NVDA'")],
) -> str:
    """Retrieve earnings event data including EPS estimates, historical surprise history, post-earnings price moves, and implied volatility premium for a stock. Use when planning earnings trades, evaluating expected move vs implied move, or assessing beat/miss probability. Returns JSON with analyst consensus, beat rate, historical reactions, and IV context. Synonyms: earnings report, EPS, quarterly results, earnings surprise, earnings calendar."""
    try:
        registry = _get_data_registry()
        data = registry.fetch_earnings(symbol)
        result = {"symbol": symbol, **data} if isinstance(data, dict) else {"symbol": symbol, "data": data}
    except Exception as e:
        logger.error(f"fetch_earnings_data({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def load_market_data(
    symbol: Annotated[str, Field(description="Ticker symbol to load cached data for, e.g. 'SPY', 'AAPL'")],
    timeframe: Annotated[str, Field(description="Candle interval to load: 'daily', 'weekly', '1h', or '4h'. Defaults to 'daily'")] = "daily",
) -> str:
    """Load previously fetched and persisted OHLCV market data from the local PostgreSQL database. Use when price data has already been fetched via fetch_market_data and you want to avoid redundant API calls. Returns JSON with timestamped price bars, date range, and row count. Provides cached historical candles for backtesting, indicator computation, and charting. Synonyms: read stored data, local cache, saved prices, persisted bars, database lookup."""
    try:
        reader = _get_reader()
        tf = _parse_timeframe(timeframe)
        df = reader.load_ohlcv(symbol, tf)
        reader.close()

        if df.empty:
            return json.dumps({"error": f"No stored data for {symbol}", "symbol": symbol})

        result = {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def list_stored_symbols() -> str:
    """List all ticker symbols that have OHLCV price data stored in the local PostgreSQL database. Use when checking data availability before running backtests or to see which instruments have been previously fetched. Returns JSON with available symbol names and their stored timeframes. Synonyms: available tickers, cached symbols, stored instruments, data inventory, database contents."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_company_facts(
    symbol: Annotated[str, Field(description="Ticker symbol to retrieve SEC filings for, e.g. 'AAPL', 'AMZN'")],
) -> str:
    """Retrieve SEC EDGAR company facts and financial statement line items for a publicly traded stock. Use when you need granular regulatory filing data such as reported revenue, assets, liabilities, shares outstanding, or other XBRL-tagged financial metrics. Returns JSON with structured filing data. Synonyms: SEC filings, EDGAR, 10-K, 10-Q, annual report, quarterly filing, regulatory data."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_analyst_estimates(
    symbol: Annotated[str, Field(description="Ticker symbol to fetch analyst estimates for, e.g. 'NVDA', 'META'")],
) -> str:
    """Retrieve consensus Wall Street analyst estimates including EPS forecasts, revenue projections, price targets, and recommendation ratings for a stock. Use when assessing market expectations, evaluating upside/downside potential, or comparing sell-side sentiment. Returns JSON with forward estimates and consensus metrics. Synonyms: price target, analyst rating, consensus forecast, Wall Street estimates, sell-side coverage, broker recommendations."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def screen_stocks(
    min_market_cap: Annotated[float | None, Field(description="Minimum market capitalization in USD to filter stocks, e.g. 1000000000 for $1B")] = None,
    max_pe: Annotated[float | None, Field(description="Maximum price-to-earnings ratio threshold for value screening")] = None,
    min_revenue_growth: Annotated[float | None, Field(description="Minimum year-over-year revenue growth rate as a decimal, e.g. 0.10 for 10%")] = None,
    sector: Annotated[str | None, Field(description="Sector name to filter by, e.g. 'Technology', 'Healthcare', 'Energy'")] = None,
) -> str:
    """Screen and filter stocks based on fundamental criteria such as market cap, valuation ratios, growth rates, and sector classification. Use when searching for trade candidates that match specific financial characteristics or building a filtered universe for further analysis. Returns JSON with matching tickers and their fundamental metrics. Synonyms: stock screener, filter, scan, universe selection, fundamental screen, equity search."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# AV Data Expansion tools
# ---------------------------------------------------------------------------


@tool
async def get_put_call_ratio(
    symbol: Annotated[str, Field(description="Ticker symbol to compute put-call ratio for, e.g. 'AAPL', 'SPY'")],
    lookback_days: Annotated[int, Field(description="Days of PCR history to analyze. Defaults to 30")] = 30,
) -> str:
    """Retrieve put-call ratio (PCR) signal for a stock derived from options chain volume data. High PCR (>80th percentile) signals excess fear and is contrarian bullish; low PCR (<20th percentile) signals excess greed and is contrarian bearish. Use when assessing options sentiment, market positioning, or contrarian entry timing. Returns JSON with current PCR, percentile rank, signal direction, and volume breakdown. Synonyms: put call ratio, options sentiment, PCR, fear greed, options volume, contrarian indicator."""
    try:
        store = _get_reader()
        result = await collect_put_call_ratio(symbol, store)
        if not result:
            result = {"symbol": symbol, "warning": "No options volume data available"}
        else:
            result["symbol"] = symbol
    except Exception as e:
        logger.error(f"get_put_call_ratio({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def get_earnings_momentum(
    symbol: Annotated[str, Field(description="Ticker symbol to check earnings momentum for, e.g. 'NVDA', 'META'")],
    quarters: Annotated[int, Field(description="Quarters of earnings history to analyze. Defaults to 8")] = 8,
) -> str:
    """Retrieve earnings momentum signal including consecutive beat/miss streaks, average surprise percentage, post-earnings announcement drift (PEAD) detection, days to next earnings, and composite momentum score. Use when planning earnings trades, assessing fundamental momentum, or timing entries around earnings catalysts. Returns JSON with streak, surprise stats, drift flag, next earnings estimate, and momentum score [-1, 1]. Synonyms: earnings surprise, EPS momentum, beat rate, PEAD, earnings drift, quarterly results momentum."""
    try:
        store = _get_reader()
        result = await collect_earnings_momentum(symbol, store)
        if not result:
            result = {"symbol": symbol, "warning": "No earnings data available"}
        else:
            result["symbol"] = symbol
    except Exception as e:
        logger.error(f"get_earnings_momentum({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)


@tool
async def get_commodity_signals(
    lookback_days: Annotated[int, Field(description="Days of commodity history to analyze. Defaults to 60")] = 60,
) -> str:
    """Retrieve cross-commodity signals including gold/silver/copper ratios, sector rotation indicators, USD strength, and commodity regime classification. Use when assessing macro conditions, sector rotation, inflation expectations, or risk-on/risk-off regime shifts. Returns JSON with commodity ratios, momentum, regime label, and USD strength index. Synonyms: commodity regime, gold silver ratio, copper signal, inflation indicator, macro commodities, risk on risk off."""
    try:
        store = _get_reader()
        # Commodity collector is global — symbol is ignored but required by API
        result = await collect_commodity_signals("_GLOBAL", store)
        if not result:
            result = {"warning": "No commodity macro data available"}
        result["lookback_days"] = lookback_days
    except Exception as e:
        logger.error(f"get_commodity_signals() failed: {e}")
        result = {"error": str(e)}
    return json.dumps(result, default=str)


@tool
async def get_forex_rates(
    lookback_days: Annotated[int, Field(description="Days of forex history to analyze. Defaults to 30")] = 30,
) -> str:
    """Retrieve forex rate data for major currency pairs (EURUSD, USDJPY) with momentum and USD strength index. Use when assessing dollar strength, currency-driven sector impacts, or macro risk positioning. Returns JSON with latest rates, momentum over lookback period, and a composite USD strength score. Synonyms: forex, currency, dollar strength, EURUSD, USDJPY, FX rates, currency momentum."""
    try:
        store = _get_reader()
        start = (date.today() - timedelta(days=lookback_days)).isoformat()

        pairs: dict[str, Any] = {}
        for pair in ("EURUSD", "USDJPY"):
            df = store.load_macro_indicator(pair, start_date=start)
            if df.empty:
                pairs[pair] = {"available": False}
                continue
            latest = float(df["value"].iloc[-1])
            first = float(df["value"].iloc[0])
            momentum = (latest - first) / first if first != 0 else 0.0
            pairs[pair] = {
                "latest": latest,
                "period_start": first,
                "momentum_pct": round(momentum * 100, 4),
                "data_points": len(df),
            }

        # USD strength: USD up when EURUSD falls and USDJPY rises
        eurusd_mom = pairs.get("EURUSD", {}).get("momentum_pct", 0.0)
        usdjpy_mom = pairs.get("USDJPY", {}).get("momentum_pct", 0.0)
        usd_strength = round((-eurusd_mom + usdjpy_mom) / 2, 4)

        result = {
            "pairs": pairs,
            "usd_strength_score": usd_strength,
            "lookback_days": lookback_days,
        }
    except Exception as e:
        logger.error(f"get_forex_rates() failed: {e}")
        result = {"error": str(e)}
    return json.dumps(result, default=str)


@tool
async def check_listing_status(
    symbol: Annotated[str, Field(description="Ticker symbol to check listing status for, e.g. 'AAPL', 'GE'")],
) -> str:
    """Check whether a stock is actively listed or has been delisted. Queries the company_overview table for delisting status. Use when validating trade candidates, screening watchlists, or detecting stale positions in delisted securities. Returns JSON with symbol, listed status, and delisting date if applicable. Synonyms: delisted, listing status, active ticker, removed from exchange, ticker validity."""
    try:
        store = _get_reader()
        delisted = store.get_delisted_symbols()
        is_delisted = symbol.upper() in [s.upper() for s in delisted]
        result = {
            "symbol": symbol.upper(),
            "is_delisted": is_delisted,
            "status": "delisted" if is_delisted else "active",
        }
    except Exception as e:
        logger.error(f"check_listing_status({symbol}) failed: {e}")
        result = {"error": str(e), "symbol": symbol}
    return json.dumps(result, default=str)
