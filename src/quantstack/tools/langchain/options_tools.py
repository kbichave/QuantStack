"""Options analysis tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def fetch_options_chain(symbol: str, days_to_expiry: int = 30) -> str:
    """Fetch option chain with prices and Greeks for a symbol.

    Args:
        symbol: Ticker symbol.
        days_to_expiry: Target DTE for the options chain.

    Returns JSON with calls and puts including strike, price, IV, and Greeks.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "get_options_chain", symbol=symbol, days_to_expiry=days_to_expiry
    )
    return json.dumps(result, default=str)


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
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "compute_greeks",
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    return json.dumps(result, default=str)
