# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Vollib adapter for options pricing, implied volatility, and Greeks.

Wraps py_vollib and py_vollib_vectorized for:
- Black-Scholes / Black-Scholes-Merton pricing
- Implied volatility calculation
- Greeks computation (delta, gamma, theta, vega, rho)

This adapter isolates the external library and provides a clean interface
for QuantCore's options engine and MCP tools.
"""

from typing import Literal

import numpy as np
from loguru import logger

from quantstack.core.options.models import OptionType
from py_vollib.black_scholes_merton import black_scholes_merton as scalar_bsm
from py_vollib.black_scholes_merton.greeks.analytical import (
    delta as scalar_delta,
    gamma as scalar_gamma,
    rho as scalar_rho,
    theta as scalar_theta,
    vega as scalar_vega,
)
from py_vollib.black_scholes_merton.implied_volatility import (
    implied_volatility as scalar_iv,
)
from py_vollib_vectorized import (
    vectorized_black_scholes_merton as vec_bsm,
    vectorized_implied_volatility as vec_iv,
)
from py_vollib_vectorized.greeks import (
    delta as vec_delta,
    gamma as vec_gamma,
    rho as vec_rho,
    theta as vec_theta,
    vega as vec_vega,
)

from quantstack.core.options.pricing import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility as internal_implied_volatility,
)

# Type alias for option type
OptionTypeStr = Literal["call", "put", "c", "p"]


def _normalize_option_type(option_type: OptionTypeStr) -> str:
    """Normalize option type to single character format for vollib."""
    opt = option_type.lower()
    if opt in ("call", "c"):
        return "c"
    elif opt in ("put", "p"):
        return "p"
    else:
        raise ValueError(
            f"Invalid option type: {option_type}. Must be 'call', 'put', 'c', or 'p'."
        )


def _validate_inputs(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float | None = None,
    rate: float = 0.0,
) -> None:
    """Validate common inputs for pricing functions."""
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"Strike price must be positive, got {strike}")
    if time_to_expiry < 0:
        raise ValueError(f"Time to expiry must be non-negative, got {time_to_expiry}")
    if vol is not None and vol <= 0:
        raise ValueError(f"Volatility must be positive, got {vol}")


def _opt_enum(opt_type: str) -> OptionType:
    """Convert single-char option type to OptionType enum."""
    return OptionType.CALL if opt_type == "c" else OptionType.PUT


def bs_price_vollib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionTypeStr = "call",
) -> float:
    """
    Calculate Black-Scholes-Merton option price using vollib.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years (e.g., 0.25 for 3 months)
        vol: Annualized volatility (e.g., 0.20 for 20%)
        rate: Risk-free interest rate (annualized)
        dividend_yield: Continuous dividend yield (annualized)
        option_type: "call", "put", "c", or "p"

    Returns:
        Option price

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_inputs(spot, strike, time_to_expiry, vol, rate)
    opt_type = _normalize_option_type(option_type)

    # Handle expiry edge case
    if time_to_expiry <= 0:
        # Return intrinsic value at expiry
        if opt_type == "c":
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)

    price = vec_bsm(
        flag=opt_type,
        S=spot,
        K=strike,
        t=time_to_expiry,
        r=rate,
        sigma=vol,
        q=dividend_yield,
        return_as="numpy",
    )
    # vec_bsm returns a 1-element array even for scalar inputs
    return float(np.asarray(price).ravel()[0])


def implied_vol_vollib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_price: float,
    option_type: OptionTypeStr = "call",
    tolerance: float = 1e-6,
) -> float | None:
    """
    Calculate implied volatility from market price using vollib.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_price: Market price of the option
        option_type: "call", "put", "c", or "p"
        tolerance: Convergence tolerance for solver

    Returns:
        Implied volatility, or None if solution not found

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_inputs(spot, strike, time_to_expiry, rate=rate)
    opt_type = _normalize_option_type(option_type)

    if option_price <= 0:
        logger.debug(f"Option price {option_price} is non-positive, cannot compute IV")
        return None

    if time_to_expiry <= 0:
        logger.debug("Time to expiry is zero, cannot compute IV")
        return None

    # Check for arbitrage: price should be above intrinsic
    if opt_type == "c":
        intrinsic = max(
            0,
            spot * np.exp(-dividend_yield * time_to_expiry)
            - strike * np.exp(-rate * time_to_expiry),
        )
    else:
        intrinsic = max(
            0,
            strike * np.exp(-rate * time_to_expiry)
            - spot * np.exp(-dividend_yield * time_to_expiry),
        )

    if option_price < intrinsic * 0.99:
        logger.debug(f"Option price {option_price} below intrinsic {intrinsic}")
        return None

    try:
        iv = vec_iv(
            price=option_price,
            S=spot,
            K=strike,
            t=time_to_expiry,
            r=rate,
            flag=opt_type,
            q=dividend_yield,
            return_as="numpy",
        )
        result = float(iv)

        # Sanity check
        if np.isnan(result) or result <= 0 or result > 5.0:
            return None
        return result
    except Exception as e:
        logger.debug(f"Vectorized IV failed ({e}), trying scalar")

    try:
        iv = scalar_iv(
            price=option_price,
            S=spot,
            K=strike,
            t=time_to_expiry,
            r=rate,
            flag=opt_type,
            q=dividend_yield,
        )
        result = float(iv)

        if np.isnan(result) or result <= 0 or result > 5.0:
            return None
        return result
    except Exception as e:
        logger.warning(
            f"vollib IV calculation failed: {e}, falling back to internal"
        )
        return internal_implied_volatility(
            market_price=option_price,
            S=spot,
            K=strike,
            T=time_to_expiry,
            r=rate,
            option_type=_opt_enum(opt_type),
            q=dividend_yield,
        )


def greeks_vollib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionTypeStr = "call",
) -> dict[str, float]:
    """
    Calculate option Greeks using vollib.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        vol: Annualized volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call", "put", "c", or "p"

    Returns:
        Dictionary with delta, gamma, theta, vega, rho

    Note:
        - Theta is expressed as daily decay (per calendar day)
        - Vega is per 1% change in volatility
        - Rho is per 1% change in interest rate
    """
    _validate_inputs(spot, strike, time_to_expiry, vol, rate)
    opt_type = _normalize_option_type(option_type)

    # Handle expiry edge case
    if time_to_expiry <= 0 or vol <= 0:
        if opt_type == "c":
            delta_val = 1.0 if spot > strike else 0.0
        else:
            delta_val = -1.0 if spot < strike else 0.0

        return {
            "delta": delta_val,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    common_args = {
        "flag": opt_type,
        "S": spot,
        "K": strike,
        "t": time_to_expiry,
        "r": rate,
        "sigma": vol,
        "q": dividend_yield,
        "return_as": "numpy",
    }

    def _scalar(arr) -> float:
        return float(np.asarray(arr).ravel()[0])

    return {
        "delta": _scalar(vec_delta(**common_args)),
        "gamma": _scalar(vec_gamma(**common_args)),
        "theta": _scalar(vec_theta(**common_args)) / 365.0,  # Convert to daily
        "vega": _scalar(vec_vega(**common_args)) / 100.0,  # Per 1% vol change
        "rho": _scalar(vec_rho(**common_args)) / 100.0,  # Per 1% rate change
    }


def bs_price_vectorized(
    spots: np.ndarray | list[float],
    strikes: np.ndarray | list[float],
    times_to_expiry: np.ndarray | list[float],
    vols: np.ndarray | list[float],
    rates: np.ndarray | list[float] | float = 0.05,
    dividend_yields: np.ndarray | list[float] | float = 0.0,
    option_types: np.ndarray | list[str] | str = "call",
) -> np.ndarray:
    """
    Vectorized Black-Scholes-Merton pricing for multiple options.

    Args:
        spots: Array of underlying prices
        strikes: Array of strike prices
        times_to_expiry: Array of times to expiration
        vols: Array of volatilities
        rates: Risk-free rate(s)
        dividend_yields: Dividend yield(s)
        option_types: Option type(s) - "call", "put", "c", "p"

    Returns:
        NumPy array of option prices
    """
    spots = np.atleast_1d(spots)
    strikes = np.atleast_1d(strikes)
    times_to_expiry = np.atleast_1d(times_to_expiry)
    vols = np.atleast_1d(vols)

    # Broadcast scalars
    if isinstance(rates, (int, float)):
        rates = np.full_like(spots, rates)
    if isinstance(dividend_yields, (int, float)):
        dividend_yields = np.full_like(spots, dividend_yields)

    # Normalize option types
    if isinstance(option_types, str):
        flags = np.array([_normalize_option_type(option_types)] * len(spots))
    else:
        flags = np.array([_normalize_option_type(ot) for ot in option_types])

    prices = vec_bsm(
        flag=flags,
        S=spots,
        K=strikes,
        t=times_to_expiry,
        r=rates,
        sigma=vols,
        q=dividend_yields,
        return_as="numpy",
    )
    return np.array(prices)


def implied_vol_vectorized(
    spots: np.ndarray | list[float],
    strikes: np.ndarray | list[float],
    times_to_expiry: np.ndarray | list[float],
    rates: np.ndarray | list[float],
    dividend_yields: np.ndarray | list[float],
    option_prices: np.ndarray | list[float],
    option_types: np.ndarray | list[str] | str = "call",
) -> np.ndarray:
    """
    Vectorized implied volatility calculation.

    Args:
        spots: Array of underlying prices
        strikes: Array of strike prices
        times_to_expiry: Array of times to expiration
        rates: Array of risk-free rates
        dividend_yields: Array of dividend yields
        option_prices: Array of market prices
        option_types: Option type(s)

    Returns:
        NumPy array of implied volatilities (NaN where not solvable)
    """
    spots = np.atleast_1d(spots)
    strikes = np.atleast_1d(strikes)
    times_to_expiry = np.atleast_1d(times_to_expiry)
    rates = np.atleast_1d(rates)
    dividend_yields = np.atleast_1d(dividend_yields)
    option_prices = np.atleast_1d(option_prices)

    if isinstance(option_types, str):
        flags = np.array([_normalize_option_type(option_types)] * len(spots))
    else:
        flags = np.array([_normalize_option_type(ot) for ot in option_types])

    ivs = vec_iv(
        price=option_prices,
        S=spots,
        K=strikes,
        t=times_to_expiry,
        r=rates,
        flag=flags,
        q=dividend_yields,
        return_as="numpy",
    )
    return np.array(ivs)
