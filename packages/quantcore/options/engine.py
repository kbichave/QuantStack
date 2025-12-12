# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Options pricing engine with backend dispatch.

Provides unified interface for options pricing that can route to different backends:
- vollib: Fast Black-Scholes pricing for European options
- financepy: American options and exotics
- internal: Fallback to QuantCore's internal implementation

This engine is the main entry point for all options calculations in QuantCore MCP.
"""

from typing import Any, Dict, Literal, Optional, Union
from loguru import logger

# Type aliases
OptionType = Literal["call", "put", "c", "p"]
ExerciseStyle = Literal["european", "american"]
PricingBackend = Literal["vollib", "financepy", "internal", "auto"]


def price_option_dispatch(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    exercise_style: ExerciseStyle = "european",
    backend: PricingBackend = "auto",
) -> Dict[str, Any]:
    """
    Price an option using the specified or auto-selected backend.

    This is the main entry point for options pricing in QuantCore.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        vol: Annualized volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        exercise_style: "european" or "american"
        backend: Pricing backend ("vollib", "financepy", "internal", "auto")

    Returns:
        Dictionary with:
            - price: Option price
            - backend_used: Which backend was used
            - greeks: Delta, gamma, theta, vega, rho
            - additional info based on backend
    """
    # Auto-select backend based on exercise style
    if backend == "auto":
        if exercise_style == "american":
            backend = "financepy"
        else:
            backend = "vollib"

    # Dispatch to appropriate backend
    if backend == "vollib" and exercise_style == "european":
        return _price_with_vollib(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
        )
    elif backend == "financepy" or exercise_style == "american":
        return _price_with_financepy(
            spot,
            strike,
            time_to_expiry,
            vol,
            rate,
            dividend_yield,
            option_type,
            exercise_style,
        )
    else:
        return _price_with_internal(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
        )


def _price_with_vollib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> Dict[str, Any]:
    """Price using vollib backend."""
    try:
        from quantcore.options.adapters.vollib_adapter import (
            bs_price_vollib,
            greeks_vollib,
        )

        price = bs_price_vollib(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
        )

        greeks = greeks_vollib(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
        )

        return {
            "price": round(price, 4),
            "backend_used": "vollib",
            "exercise_style": "european",
            "greeks": {k: round(v, 6) for k, v in greeks.items()},
            "inputs": {
                "spot": spot,
                "strike": strike,
                "time_to_expiry": time_to_expiry,
                "volatility": vol,
                "rate": rate,
                "dividend_yield": dividend_yield,
                "option_type": option_type,
            },
        }

    except Exception as e:
        logger.warning(f"vollib pricing failed: {e}, falling back to internal")
        return _price_with_internal(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
        )


def _price_with_financepy(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
) -> Dict[str, Any]:
    """Price using financepy backend."""
    try:
        from quantcore.options.adapters.financepy_adapter import (
            price_vanilla_financepy,
            price_american_option,
        )

        if exercise_style == "american":
            result = price_american_option(
                spot,
                strike,
                time_to_expiry,
                vol,
                rate,
                dividend_yield,
                option_type,
                num_steps=100,
            )

            return {
                "price": round(result["price"], 4),
                "backend_used": "financepy",
                "exercise_style": "american",
                "european_price": round(result["european_price"], 4),
                "early_exercise_premium": round(result["early_exercise_premium"], 4),
                "greeks": {
                    "delta": round(result["delta"], 6),
                    "gamma": round(result["gamma"], 6),
                },
                "inputs": {
                    "spot": spot,
                    "strike": strike,
                    "time_to_expiry": time_to_expiry,
                    "volatility": vol,
                    "rate": rate,
                    "dividend_yield": dividend_yield,
                    "option_type": option_type,
                },
            }
        else:
            price = price_vanilla_financepy(
                spot,
                strike,
                time_to_expiry,
                vol,
                rate,
                dividend_yield,
                option_type,
                exercise_style,
            )

            # Get Greeks from vollib or internal
            greeks = compute_greeks_dispatch(
                spot,
                strike,
                time_to_expiry,
                vol,
                rate,
                dividend_yield,
                option_type,
                backend="vollib",
            )

            return {
                "price": round(price, 4),
                "backend_used": "financepy",
                "exercise_style": exercise_style,
                "greeks": greeks.get("greeks", {}),
                "inputs": {
                    "spot": spot,
                    "strike": strike,
                    "time_to_expiry": time_to_expiry,
                    "volatility": vol,
                    "rate": rate,
                    "dividend_yield": dividend_yield,
                    "option_type": option_type,
                },
            }

    except Exception as e:
        logger.warning(f"financepy pricing failed: {e}, falling back to internal")
        return _price_with_internal(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
        )


def _price_with_internal(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> Dict[str, Any]:
    """Price using internal QuantCore implementation."""
    from quantcore.options.pricing import black_scholes_price, black_scholes_greeks
    from quantcore.options.models import OptionType as OptType

    opt_enum = OptType.CALL if option_type.lower() in ("call", "c") else OptType.PUT

    price = black_scholes_price(
        S=spot,
        K=strike,
        T=time_to_expiry,
        r=rate,
        sigma=vol,
        option_type=opt_enum,
        q=dividend_yield,
    )

    greeks = black_scholes_greeks(
        S=spot,
        K=strike,
        T=time_to_expiry,
        r=rate,
        sigma=vol,
        option_type=opt_enum,
        q=dividend_yield,
    )

    return {
        "price": round(price, 4),
        "backend_used": "internal",
        "exercise_style": "european",
        "greeks": {
            "delta": round(greeks.delta, 6),
            "gamma": round(greeks.gamma, 6),
            "theta": round(greeks.theta, 6),
            "vega": round(greeks.vega, 6),
            "rho": round(greeks.rho, 6),
        },
        "inputs": {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "volatility": vol,
            "rate": rate,
            "dividend_yield": dividend_yield,
            "option_type": option_type,
        },
    }


def compute_greeks_dispatch(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    backend: PricingBackend = "auto",
) -> Dict[str, Any]:
    """
    Compute option Greeks using the specified backend.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        vol: Annualized volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        backend: Pricing backend

    Returns:
        Dictionary with Greeks and interpretations
    """
    if backend == "auto":
        backend = "vollib"

    if backend == "vollib":
        try:
            from quantcore.options.adapters.vollib_adapter import greeks_vollib

            greeks = greeks_vollib(
                spot, strike, time_to_expiry, vol, rate, dividend_yield, option_type
            )

            return {
                "greeks": {k: round(v, 6) for k, v in greeks.items()},
                "backend_used": "vollib",
                "interpretations": _interpret_greeks(greeks, option_type),
                "risk_metrics": _compute_risk_metrics(greeks, spot),
            }

        except Exception as e:
            logger.warning(f"vollib Greeks failed: {e}, using internal")

    # Fallback to internal
    from quantcore.options.pricing import black_scholes_greeks
    from quantcore.options.models import OptionType as OptType

    opt_enum = OptType.CALL if option_type.lower() in ("call", "c") else OptType.PUT

    greeks = black_scholes_greeks(
        S=spot,
        K=strike,
        T=time_to_expiry,
        r=rate,
        sigma=vol,
        option_type=opt_enum,
        q=dividend_yield,
    )

    greeks_dict = {
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "theta": greeks.theta,
        "vega": greeks.vega,
        "rho": greeks.rho,
    }

    return {
        "greeks": {k: round(v, 6) for k, v in greeks_dict.items()},
        "backend_used": "internal",
        "interpretations": _interpret_greeks(greeks_dict, option_type),
        "risk_metrics": _compute_risk_metrics(greeks_dict, spot),
    }


def compute_iv_dispatch(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_price: float,
    option_type: OptionType = "call",
    backend: PricingBackend = "auto",
) -> Dict[str, Any]:
    """
    Compute implied volatility from market price.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_price: Market price of the option
        option_type: "call" or "put"
        backend: Pricing backend

    Returns:
        Dictionary with IV and related metrics
    """
    if backend == "auto":
        backend = "vollib"

    if backend == "vollib":
        try:
            from quantcore.options.adapters.vollib_adapter import implied_vol_vollib

            iv = implied_vol_vollib(
                spot,
                strike,
                time_to_expiry,
                rate,
                dividend_yield,
                option_price,
                option_type,
            )

            if iv is not None:
                return {
                    "implied_volatility": round(iv, 4),
                    "iv_pct": round(iv * 100, 2),
                    "backend_used": "vollib",
                    "inputs": {
                        "spot": spot,
                        "strike": strike,
                        "time_to_expiry": time_to_expiry,
                        "option_price": option_price,
                        "option_type": option_type,
                    },
                    "analysis": _analyze_iv(iv, spot, strike, time_to_expiry),
                }

        except Exception as e:
            logger.warning(f"vollib IV failed: {e}")

    # Fallback to internal
    from quantcore.options.pricing import implied_volatility
    from quantcore.options.models import OptionType as OptType

    opt_enum = OptType.CALL if option_type.lower() in ("call", "c") else OptType.PUT

    iv = implied_volatility(
        market_price=option_price,
        S=spot,
        K=strike,
        T=time_to_expiry,
        r=rate,
        option_type=opt_enum,
        q=dividend_yield,
    )

    if iv is not None:
        return {
            "implied_volatility": round(iv, 4),
            "iv_pct": round(iv * 100, 2),
            "backend_used": "internal",
            "inputs": {
                "spot": spot,
                "strike": strike,
                "time_to_expiry": time_to_expiry,
                "option_price": option_price,
                "option_type": option_type,
            },
            "analysis": _analyze_iv(iv, spot, strike, time_to_expiry),
        }

    return {
        "error": "Could not solve for implied volatility",
        "reason": "Price may be below intrinsic value or inputs are invalid",
    }


def _interpret_greeks(greeks: Dict[str, float], option_type: str) -> Dict[str, str]:
    """Generate human-readable interpretations of Greeks."""
    delta = greeks.get("delta", 0)
    gamma = greeks.get("gamma", 0)
    theta = greeks.get("theta", 0)
    vega = greeks.get("vega", 0)
    rho = greeks.get("rho", 0)

    return {
        "delta": f"Option moves ${abs(delta):.2f} for $1 underlying move",
        "gamma": f"Delta changes by {gamma:.4f} per $1 underlying move",
        "theta": f"Option loses ${abs(theta):.2f} per day from time decay",
        "vega": f"Option moves ${vega:.2f} for 1% IV change",
        "rho": f"Option moves ${rho:.2f} for 1% rate change",
        "probability_itm": f"~{abs(delta)*100:.0f}% implied probability of expiring ITM",
    }


def _compute_risk_metrics(greeks: Dict[str, float], spot: float) -> Dict[str, float]:
    """Compute position risk metrics from Greeks."""
    delta = greeks.get("delta", 0)
    gamma = greeks.get("gamma", 0)
    theta = greeks.get("theta", 0)
    vega = greeks.get("vega", 0)

    return {
        "shares_to_hedge": round(-delta * 100, 0),  # Per contract
        "daily_theta_cost": round(theta * 100, 2),  # Per contract
        "iv_sensitivity": round(vega * 100, 2),  # Per contract per 1% IV
        "dollar_delta": round(delta * spot * 100, 2),  # Dollar exposure per contract
        "dollar_gamma": round(gamma * spot * spot * 100 / 100, 2),  # Gamma P&L estimate
    }


def _analyze_iv(iv: float, spot: float, strike: float, tte: float) -> Dict[str, Any]:
    """Analyze implied volatility context."""
    import numpy as np

    moneyness = np.log(spot / strike)

    # IV interpretation
    if iv < 0.15:
        iv_level = "very_low"
    elif iv < 0.25:
        iv_level = "low"
    elif iv < 0.40:
        iv_level = "moderate"
    elif iv < 0.60:
        iv_level = "high"
    else:
        iv_level = "very_high"

    # Expected move (1 standard deviation)
    expected_move_pct = iv * np.sqrt(tte) * 100
    expected_move_dollars = spot * iv * np.sqrt(tte)

    return {
        "iv_level": iv_level,
        "moneyness": round(moneyness, 4),
        "expected_move_1sd_pct": round(expected_move_pct, 2),
        "expected_move_1sd_dollars": round(expected_move_dollars, 2),
        "annualized_move_1sd": round(iv * spot, 2),
    }


# Convenience functions for common operations


def quick_price(
    spot: float,
    strike: float,
    days_to_expiry: int,
    vol: float,
    option_type: OptionType = "call",
    rate: float = 0.05,
) -> float:
    """
    Quick option pricing with minimal parameters.

    Args:
        spot: Current price
        strike: Strike price
        days_to_expiry: Days until expiration
        vol: Volatility (e.g., 0.25 for 25%)
        option_type: "call" or "put"
        rate: Risk-free rate

    Returns:
        Option price
    """
    tte = days_to_expiry / 365.0
    result = price_option_dispatch(spot, strike, tte, vol, rate, 0.0, option_type)
    return result["price"]


def quick_greeks(
    spot: float,
    strike: float,
    days_to_expiry: int,
    vol: float,
    option_type: OptionType = "call",
) -> Dict[str, float]:
    """
    Quick Greeks calculation with minimal parameters.

    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    tte = days_to_expiry / 365.0
    result = compute_greeks_dispatch(spot, strike, tte, vol, 0.05, 0.0, option_type)
    return result["greeks"]


def quick_iv(
    spot: float,
    strike: float,
    days_to_expiry: int,
    option_price: float,
    option_type: OptionType = "call",
) -> Optional[float]:
    """
    Quick IV calculation with minimal parameters.

    Returns:
        Implied volatility or None if not solvable
    """
    tte = days_to_expiry / 365.0
    result = compute_iv_dispatch(
        spot, strike, tte, 0.05, 0.0, option_price, option_type
    )
    return result.get("implied_volatility")
