"""Options analysis tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def fetch_options_chain(
    symbol: str,
    expiry_min_days: int = 7,
    expiry_max_days: int = 45,
    option_type: str | None = None,
) -> str:
    """Fetch option chain with prices and Greeks for a symbol.

    Args:
        symbol: Ticker symbol.
        expiry_min_days: Minimum DTE to include (default 7).
        expiry_max_days: Maximum DTE to include (default 45).
        option_type: Filter to "call" or "put" only. None returns both.

    Returns JSON with calls and puts including strike, price, IV, and Greeks.
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
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = "call",
    dividend_yield: float = 0,
) -> str:
    """Compute option Greeks (Delta, Gamma, Theta, Vega, Rho).

    Args:
        spot: Current underlying price.
        strike: Option strike price.
        time_to_expiry: Time to expiry in years.
        volatility: Implied volatility (annualized).
        risk_free_rate: Risk-free rate.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield.
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
    spot: float, strike: float, time_to_expiry: float,
    volatility: float, option_type: str = "call",
) -> str:
    """Price an option using Black-Scholes.

    Args:
        spot: Underlying price.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        volatility: Implied volatility.
        option_type: "call" or "put".
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_implied_vol(
    market_price: float, spot: float, strike: float,
    time_to_expiry: float, option_type: str = "call",
) -> str:
    """Compute implied volatility from market price.

    Args:
        market_price: Observed option price.
        spot: Underlying price.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        option_type: "call" or "put".
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def analyze_option_structure(symbol: str) -> str:
    """Analyze the IV surface and skew for a symbol."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_iv_surface(symbol: str) -> str:
    """Get the implied volatility surface for a symbol."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def score_trade_structure(symbol: str, legs: list) -> str:
    """Score a proposed options trade structure.

    Args:
        symbol: Underlying symbol.
        legs: List of option legs.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def simulate_trade_outcome(symbol: str, legs: list, scenarios: list | None = None) -> str:
    """Simulate P&L for an options trade under various scenarios.

    Args:
        symbol: Underlying symbol.
        legs: List of option legs.
        scenarios: Price/vol scenarios to simulate.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
