"""Options analysis tools for LangGraph agents."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


@tool
async def fetch_options_chain(
    symbol: Annotated[str, Field(description="Ticker symbol to retrieve the options chain for, e.g. 'AAPL' or 'SPY'")],
    expiry_min_days: Annotated[int, Field(description="Minimum days to expiration (DTE) filter; contracts expiring sooner are excluded")] = 7,
    expiry_max_days: Annotated[int, Field(description="Maximum days to expiration (DTE) filter; contracts expiring later are excluded")] = 45,
    option_type: Annotated[str | None, Field(description="Filter by contract type: 'call' or 'put'. Use None to return both calls and puts")] = None,
) -> str:
    """Retrieves the full options chain for a given equity symbol, including strike prices, bid/ask quotes, implied volatility (IV), and Greeks (delta, gamma, theta, vega). Use when you need to scan available put/call contracts, compare expirations, or evaluate premium levels across the chain. Returns JSON with calls and puts organized by expiration date.
    """
    try:
        from quantstack.data.providers import DataProviderRegistry

        registry = DataProviderRegistry()
        chain = await registry.get_options_chain(
            symbol=symbol,
            expiry_min_days=expiry_min_days,
            expiry_max_days=expiry_max_days,
        )
        if option_type:
            chain = {k: v for k, v in chain.items() if k == option_type or k == f"{option_type}s"}
        return json.dumps(chain, default=str)
    except Exception as e:
        logger.error(f"fetch_options_chain({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})


@tool
async def compute_greeks(
    spot: Annotated[float, Field(description="Current underlying stock price (spot price) used for Greeks calculation")],
    strike: Annotated[float, Field(description="Option strike price (exercise price) of the contract")],
    time_to_expiry: Annotated[float, Field(description="Time to expiration in years, e.g. 0.25 for 3 months or 30/365 for 30 days")],
    volatility: Annotated[float, Field(description="Annualized implied volatility (IV) as a decimal, e.g. 0.30 for 30% IV")],
    risk_free_rate: Annotated[float, Field(description="Annualized risk-free interest rate as a decimal, e.g. 0.05 for 5%")] = 0.05,
    option_type: Annotated[str, Field(description="Contract type: 'call' for call option or 'put' for put option")] = "call",
    dividend_yield: Annotated[float, Field(description="Continuous dividend yield as a decimal for dividend-paying underlyings, e.g. 0.02 for 2%")] = 0,
) -> str:
    """Calculates option Greeks (delta, gamma, theta, vega, rho) using the Black-Scholes-Merton model. Use when you need sensitivity analysis for an options position, hedging ratios, or risk decomposition. Provides delta exposure, gamma convexity, theta time decay, vega volatility sensitivity, and rho rate sensitivity. Returns JSON with all five Greeks for the specified put/call contract parameters.
    """
    try:
        from quantstack.core.options.engine import compute_greeks_dispatch

        result = compute_greeks_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"compute_greeks failed: {e}")
        return json.dumps({"error": str(e)})


@tool
async def price_option(
    spot: Annotated[float, Field(description="Current underlying stock price (spot price) for option valuation")],
    strike: Annotated[float, Field(description="Option strike price (exercise price) of the contract to price")],
    time_to_expiry: Annotated[float, Field(description="Time to expiration in years, e.g. 0.25 for 3 months")],
    volatility: Annotated[float, Field(description="Annualized implied volatility (IV) as a decimal, e.g. 0.30 for 30%")],
    option_type: Annotated[str, Field(description="Contract type: 'call' for call option or 'put' for put option")] = "call",
) -> str:
    """Computes the theoretical fair value of a European option using the Black-Scholes pricing model. Use when you need to estimate option premium, detect mispricing between market price and theoretical value, or evaluate put/call parity. Returns the computed option price given spot, strike, implied volatility, and time to expiration.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_implied_vol(
    market_price: Annotated[float, Field(description="Observed market price (premium) of the option contract")],
    spot: Annotated[float, Field(description="Current underlying stock price (spot price)")],
    strike: Annotated[float, Field(description="Option strike price (exercise price) of the contract")],
    time_to_expiry: Annotated[float, Field(description="Time to expiration in years, e.g. 0.25 for 3 months")],
    option_type: Annotated[str, Field(description="Contract type: 'call' for call option or 'put' for put option")] = "call",
) -> str:
    """Calculates implied volatility (IV) by inverting the Black-Scholes model from the observed market price of an option. Use when you need to extract the market's volatility expectation, compare IV across strikes for skew analysis, or detect volatility mispricing. Returns the annualized implied volatility as a decimal for the specified put/call contract.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def analyze_option_structure(
    symbol: Annotated[str, Field(description="Ticker symbol to analyze, e.g. 'AAPL' or 'SPY'")],
) -> str:
    """Analyzes the implied volatility surface and volatility skew for a given equity symbol. Use when you need to assess put/call skew, term structure of IV across expirations, or detect volatility smile anomalies. Provides skew slope, term structure shape, and surface-level insights for options strategy selection."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_iv_surface(
    symbol: Annotated[str, Field(description="Ticker symbol to retrieve the IV surface for, e.g. 'AAPL' or 'SPY'")],
) -> str:
    """Retrieves the full implied volatility (IV) surface grid across strikes and expirations for a given symbol. Use when you need to visualize or analyze the volatility smile, term structure, or detect relative value opportunities across the options chain. Returns IV values organized by strike price and days to expiration (DTE)."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def score_trade_structure(
    symbol: Annotated[str, Field(description="Underlying ticker symbol for the options trade, e.g. 'AAPL'")],
    legs: Annotated[list, Field(description="List of option legs, each a dict with keys like 'strike', 'expiry', 'option_type' (call/put), 'action' (buy/sell), and 'quantity'")],
) -> str:
    """Scores a proposed multi-leg options trade structure (spread, straddle, strangle, condor, butterfly) based on risk/reward profile, Greeks exposure, and probability of profit. Use when evaluating whether to enter a complex options position. Returns a composite score with breakdown by max loss, max gain, breakeven levels, and Greek sensitivities.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def simulate_trade_outcome(
    symbol: Annotated[str, Field(description="Underlying ticker symbol for the options trade simulation, e.g. 'AAPL'")],
    legs: Annotated[list, Field(description="List of option legs, each a dict with keys like 'strike', 'expiry', 'option_type' (call/put), 'action' (buy/sell), and 'quantity'")],
    scenarios: Annotated[list | None, Field(description="List of scenario dicts specifying price and/or volatility changes, e.g. [{'price_move_pct': -5, 'iv_change': 0.10}]. Use None for default stress scenarios")] = None,
) -> str:
    """Simulates profit and loss (P&L) outcomes for a multi-leg options trade under user-defined or default stress scenarios including price shocks and volatility changes. Use when you need to stress-test a spread, straddle, strangle, or condor before execution. Returns per-scenario P&L, max profit, max loss, and breakeven analysis for the proposed options structure.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
