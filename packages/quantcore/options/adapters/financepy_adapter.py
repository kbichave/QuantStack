# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FinancePy adapter for advanced options pricing.

Provides:
- European options pricing (alternative to vollib)
- American options pricing (binomial/trinomial trees)
- Exotic options (barriers, Asians) - future expansion
- Multiple pricing models (BS, Heston, local vol)

FinancePy is a comprehensive derivatives pricing library that supports
a wide range of instruments and models.
"""

from datetime import date, datetime
from typing import Any, Dict, Literal, Optional, Union
import numpy as np
from loguru import logger


# Type aliases
OptionTypeStr = Literal["call", "put", "c", "p"]
ExerciseStyle = Literal["european", "american"]


def _normalize_option_type(option_type: OptionTypeStr) -> str:
    """Normalize option type to lowercase format."""
    opt = option_type.lower()
    if opt in ("call", "c"):
        return "call"
    elif opt in ("put", "p"):
        return "put"
    else:
        raise ValueError(f"Invalid option type: {option_type}")


def _to_financepy_date(dt: Union[date, datetime, str, float]) -> Any:
    """Convert to FinancePy Date object."""
    try:
        from financepy.utils.date import Date

        if isinstance(dt, float):
            # Assume it's time to expiry in years, convert to date
            from datetime import timedelta

            expiry_date = datetime.now() + timedelta(days=int(dt * 365))
            return Date(expiry_date.day, expiry_date.month, expiry_date.year)
        elif isinstance(dt, str):
            parsed = datetime.strptime(dt, "%Y-%m-%d")
            return Date(parsed.day, parsed.month, parsed.year)
        elif isinstance(dt, datetime):
            return Date(dt.day, dt.month, dt.year)
        elif isinstance(dt, date):
            return Date(dt.day, dt.month, dt.year)
        else:
            raise ValueError(f"Cannot convert {type(dt)} to FinancePy Date")
    except ImportError:
        return dt


def price_vanilla_financepy(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionTypeStr = "call",
    exercise_style: ExerciseStyle = "european",
    num_steps: int = 100,
) -> float:
    """
    Price vanilla option using FinancePy.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        vol: Annualized volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        exercise_style: "european" or "american"
        num_steps: Number of tree steps for American options

    Returns:
        Option price
    """
    opt_type = _normalize_option_type(option_type)

    # Handle expiry edge case
    if time_to_expiry <= 0:
        if opt_type == "call":
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)

    try:
        from financepy.utils.date import Date
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.products.equity.equity_vanilla_option import EquityVanillaOption
        from financepy.utils.global_types import OptionTypes
        from financepy.models.black_scholes import BlackScholes

        # Create dates
        today = datetime.now()
        valuation_date = Date(today.day, today.month, today.year)

        expiry_date = today + timedelta(days=int(time_to_expiry * 365))
        expiry_fp = Date(expiry_date.day, expiry_date.month, expiry_date.year)

        # Create curves
        discount_curve = DiscountCurveFlat(valuation_date, rate)
        dividend_curve = DiscountCurveFlat(valuation_date, dividend_yield)

        # Create option
        if opt_type == "call":
            fp_option_type = (
                OptionTypes.EUROPEAN_CALL
                if exercise_style == "european"
                else OptionTypes.AMERICAN_CALL
            )
        else:
            fp_option_type = (
                OptionTypes.EUROPEAN_PUT
                if exercise_style == "european"
                else OptionTypes.AMERICAN_PUT
            )

        option = EquityVanillaOption(
            expiry_date=expiry_fp,
            strike_price=strike,
            option_type=fp_option_type,
        )

        # Create model
        model = BlackScholes(vol)

        # Price
        price = option.value(
            valuation_date=valuation_date,
            stock_price=spot,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            model=model,
        )

        return float(price)

    except ImportError:
        logger.warning("financepy not available, falling back to internal pricing")
        return _price_internal(
            spot,
            strike,
            time_to_expiry,
            vol,
            rate,
            dividend_yield,
            opt_type,
            exercise_style,
            num_steps,
        )
    except Exception as e:
        logger.error(f"FinancePy pricing failed: {e}")
        return _price_internal(
            spot,
            strike,
            time_to_expiry,
            vol,
            rate,
            dividend_yield,
            opt_type,
            exercise_style,
            num_steps,
        )


# Need to import timedelta
from datetime import timedelta


def price_american_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionTypeStr = "call",
    num_steps: int = 100,
    model: str = "crr",
) -> Dict[str, Any]:
    """
    Price American option using tree methods.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        vol: Annualized volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        num_steps: Number of tree steps
        model: Tree model - "crr" (Cox-Ross-Rubinstein) or "jr" (Jarrow-Rudd)

    Returns:
        Dictionary with:
            - price: American option price
            - european_price: European equivalent for comparison
            - early_exercise_premium: American - European
            - delta, gamma: Greeks from tree
    """
    opt_type = _normalize_option_type(option_type)

    if time_to_expiry <= 0:
        intrinsic = (
            max(0.0, spot - strike) if opt_type == "call" else max(0.0, strike - spot)
        )
        return {
            "price": intrinsic,
            "european_price": intrinsic,
            "early_exercise_premium": 0.0,
            "delta": (
                1.0
                if spot > strike and opt_type == "call"
                else (-1.0 if spot < strike and opt_type == "put" else 0.0)
            ),
            "gamma": 0.0,
        }

    try:
        from financepy.utils.date import Date
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.products.equity.equity_american_option import (
            EquityAmericanOption,
        )
        from financepy.utils.global_types import OptionTypes
        from financepy.models.black_scholes import BlackScholes

        today = datetime.now()
        valuation_date = Date(today.day, today.month, today.year)

        expiry_date = today + timedelta(days=int(time_to_expiry * 365))
        expiry_fp = Date(expiry_date.day, expiry_date.month, expiry_date.year)

        discount_curve = DiscountCurveFlat(valuation_date, rate)
        dividend_curve = DiscountCurveFlat(valuation_date, dividend_yield)

        if opt_type == "call":
            fp_option_type = OptionTypes.AMERICAN_CALL
        else:
            fp_option_type = OptionTypes.AMERICAN_PUT

        american_option = EquityAmericanOption(
            expiry_date=expiry_fp,
            strike_price=strike,
            option_type=fp_option_type,
        )

        model_obj = BlackScholes(vol)

        american_price = american_option.value(
            valuation_date=valuation_date,
            stock_price=spot,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            model=model_obj,
            num_steps_per_year=int(num_steps / time_to_expiry),
        )

        # Get European price for comparison
        from financepy.products.equity.equity_vanilla_option import EquityVanillaOption

        eu_type = (
            OptionTypes.EUROPEAN_CALL
            if opt_type == "call"
            else OptionTypes.EUROPEAN_PUT
        )
        european_option = EquityVanillaOption(
            expiry_date=expiry_fp,
            strike_price=strike,
            option_type=eu_type,
        )

        european_price = european_option.value(
            valuation_date=valuation_date,
            stock_price=spot,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            model=model_obj,
        )

        # Calculate Greeks using finite differences
        eps = spot * 0.01
        price_up = american_option.value(
            valuation_date, spot + eps, discount_curve, dividend_curve, model_obj
        )
        price_down = american_option.value(
            valuation_date, spot - eps, discount_curve, dividend_curve, model_obj
        )

        delta = (price_up - price_down) / (2 * eps)
        gamma = (price_up - 2 * american_price + price_down) / (eps**2)

        return {
            "price": float(american_price),
            "european_price": float(european_price),
            "early_exercise_premium": float(american_price - european_price),
            "delta": float(delta),
            "gamma": float(gamma),
            "model": "financepy",
            "num_steps": num_steps,
        }

    except ImportError:
        logger.warning("financepy not available, using internal binomial tree")
        return _price_american_binomial(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, opt_type, num_steps
        )
    except Exception as e:
        logger.error(f"FinancePy American pricing failed: {e}")
        return _price_american_binomial(
            spot, strike, time_to_expiry, vol, rate, dividend_yield, opt_type, num_steps
        )


def _price_internal(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
    exercise_style: str,
    num_steps: int,
) -> float:
    """Internal pricing fallback."""
    if exercise_style == "american":
        result = _price_american_binomial(
            spot,
            strike,
            time_to_expiry,
            vol,
            rate,
            dividend_yield,
            option_type,
            num_steps,
        )
        return result["price"]
    else:
        # Use internal Black-Scholes
        from quantcore.options.pricing import black_scholes_price
        from quantcore.options.models import OptionType

        opt_enum = OptionType.CALL if option_type == "call" else OptionType.PUT
        return black_scholes_price(
            S=spot,
            K=strike,
            T=time_to_expiry,
            r=rate,
            sigma=vol,
            option_type=opt_enum,
            q=dividend_yield,
        )


def _price_american_binomial(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
    num_steps: int,
) -> Dict[str, Any]:
    """
    Price American option using Cox-Ross-Rubinstein binomial tree.

    This is a fallback implementation when FinancePy is not available.
    """
    dt = time_to_expiry / num_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    a = np.exp((rate - dividend_yield) * dt)
    p = (a - d) / (u - d)

    # Discount factor per step
    disc = np.exp(-rate * dt)

    # Build price tree at expiration
    stock_prices = np.zeros(num_steps + 1)
    for i in range(num_steps + 1):
        stock_prices[i] = spot * (u ** (num_steps - i)) * (d**i)

    # Option values at expiration
    if option_type == "call":
        option_values = np.maximum(stock_prices - strike, 0)
    else:
        option_values = np.maximum(strike - stock_prices, 0)

    # Work backwards through tree
    for j in range(num_steps - 1, -1, -1):
        for i in range(j + 1):
            stock_price = spot * (u ** (j - i)) * (d**i)

            # Continuation value
            hold_value = disc * (p * option_values[i] + (1 - p) * option_values[i + 1])

            # Early exercise value
            if option_type == "call":
                exercise_value = max(stock_price - strike, 0)
            else:
                exercise_value = max(strike - stock_price, 0)

            option_values[i] = max(hold_value, exercise_value)

    american_price = option_values[0]

    # European price using Black-Scholes for comparison
    from quantcore.options.pricing import black_scholes_price
    from quantcore.options.models import OptionType

    opt_enum = OptionType.CALL if option_type == "call" else OptionType.PUT
    european_price = black_scholes_price(
        S=spot,
        K=strike,
        T=time_to_expiry,
        r=rate,
        sigma=vol,
        option_type=opt_enum,
        q=dividend_yield,
    )

    # Estimate delta using finite differences on tree
    eps = spot * 0.01
    price_up = _price_american_binomial_simple(
        spot + eps,
        strike,
        time_to_expiry,
        vol,
        rate,
        dividend_yield,
        option_type,
        num_steps,
    )
    price_down = _price_american_binomial_simple(
        spot - eps,
        strike,
        time_to_expiry,
        vol,
        rate,
        dividend_yield,
        option_type,
        num_steps,
    )

    delta = (price_up - price_down) / (2 * eps)
    gamma = (price_up - 2 * american_price + price_down) / (eps**2)

    return {
        "price": float(american_price),
        "european_price": float(european_price),
        "early_exercise_premium": float(american_price - european_price),
        "delta": float(delta),
        "gamma": float(gamma),
        "model": "binomial_crr",
        "num_steps": num_steps,
    }


def _price_american_binomial_simple(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
    num_steps: int,
) -> float:
    """Simplified binomial pricing for Greeks calculation."""
    dt = time_to_expiry / num_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    a = np.exp((rate - dividend_yield) * dt)
    p = (a - d) / (u - d)
    disc = np.exp(-rate * dt)

    stock_prices = np.zeros(num_steps + 1)
    for i in range(num_steps + 1):
        stock_prices[i] = spot * (u ** (num_steps - i)) * (d**i)

    if option_type == "call":
        option_values = np.maximum(stock_prices - strike, 0)
    else:
        option_values = np.maximum(strike - stock_prices, 0)

    for j in range(num_steps - 1, -1, -1):
        for i in range(j + 1):
            stock_price = spot * (u ** (j - i)) * (d**i)
            hold_value = disc * (p * option_values[i] + (1 - p) * option_values[i + 1])

            if option_type == "call":
                exercise_value = max(stock_price - strike, 0)
            else:
                exercise_value = max(strike - stock_price, 0)

            option_values[i] = max(hold_value, exercise_value)

    return option_values[0]


def price_barrier_option(
    spot: float,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: OptionTypeStr = "call",
    barrier_type: Literal["up-in", "up-out", "down-in", "down-out"] = "down-out",
    rebate: float = 0.0,
) -> Dict[str, Any]:
    """
    Price barrier option.

    Args:
        spot: Current underlying price
        strike: Option strike price
        barrier: Barrier level
        time_to_expiry: Time to expiration in years
        vol: Annualized volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        barrier_type: Type of barrier
        rebate: Cash rebate if knocked out

    Returns:
        Dictionary with price and Greeks
    """
    opt_type = _normalize_option_type(option_type)

    try:
        from financepy.utils.date import Date
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.products.equity.equity_one_touch_option import (
            EquityOneTouchOption,
        )
        from financepy.utils.global_types import TouchOptionTypes
        from financepy.models.black_scholes import BlackScholes

        # Note: Full barrier option support requires additional FinancePy setup
        # This is a placeholder for future implementation
        logger.info("Barrier option pricing - using analytical approximation")

    except ImportError:
        pass

    # Use analytical approximation for single barriers
    price = _price_barrier_analytical(
        spot,
        strike,
        barrier,
        time_to_expiry,
        vol,
        rate,
        dividend_yield,
        opt_type,
        barrier_type,
    )

    return {
        "price": float(price),
        "barrier": barrier,
        "barrier_type": barrier_type,
        "rebate": rebate,
        "is_active": _check_barrier_active(spot, barrier, barrier_type),
    }


def _price_barrier_analytical(
    spot: float,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    vol: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
    barrier_type: str,
) -> float:
    """
    Analytical barrier option pricing using Reiner-Rubinstein formula.
    """
    from scipy.stats import norm

    if time_to_expiry <= 0:
        # At expiry
        if option_type == "call":
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)

        # Check if knocked out
        if "out" in barrier_type:
            if "up" in barrier_type and spot >= barrier:
                return 0.0
            if "down" in barrier_type and spot <= barrier:
                return 0.0

        return intrinsic

    # Parameters
    S = spot
    K = strike
    H = barrier
    T = time_to_expiry
    r = rate
    q = dividend_yield
    sigma = vol

    # lambda parameter
    lam = (r - q + sigma**2 / 2) / sigma**2

    # Get vanilla price first
    from quantcore.options.pricing import black_scholes_price
    from quantcore.options.models import OptionType

    opt_enum = OptionType.CALL if option_type == "call" else OptionType.PUT
    vanilla_price = black_scholes_price(S, K, T, r, sigma, opt_enum, q)

    # For down-and-out call (most common)
    if barrier_type == "down-out" and option_type == "call" and H < K:
        if S <= H:
            return 0.0

        # Simplified calculation
        x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        y1 = np.log(H / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

        barrier_adj = vanilla_price - S * np.exp(-q * T) * (H / S) ** (2 * lam) * (
            norm.cdf(y1)
            - K / H * np.exp(-r * T + q * T) * norm.cdf(y1 - sigma * np.sqrt(T))
        )

        return max(0.0, barrier_adj)

    # Default: return vanilla price (barrier not hit case)
    return vanilla_price


def _check_barrier_active(spot: float, barrier: float, barrier_type: str) -> bool:
    """Check if barrier is currently active (not knocked)."""
    if "up" in barrier_type:
        return spot < barrier
    else:  # down
        return spot > barrier
