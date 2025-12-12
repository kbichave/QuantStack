# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Quantsbin adapter for option structure analysis and payoff profiles.

Provides:
- Multi-leg option structure analysis
- Payoff profile calculation
- Break-even analysis
- Greeks aggregation for spreads
- Structure visualization data

Quantsbin is designed for structured options analysis and payoff diagrams.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class OptionLegSpec:
    """Specification for a single option leg."""

    option_type: Literal["call", "put"]
    strike: float
    expiry_days: int  # Days to expiry
    quantity: int  # Positive for long, negative for short
    premium: Optional[float] = None  # Premium per contract (if known)
    iv: Optional[float] = None


@dataclass
class StructureSpec:
    """Specification for a multi-leg option structure."""

    underlying_symbol: str
    underlying_price: float
    legs: List[OptionLegSpec]
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0


def analyze_structure_quantsbin(
    structure_spec: Union[StructureSpec, Dict[str, Any]],
    price_range_pct: float = 0.30,
    num_points: int = 100,
) -> Dict[str, Any]:
    """
    Analyze multi-leg option structure for payoff, Greeks, and key metrics.

    Args:
        structure_spec: Structure specification (StructureSpec or dict)
        price_range_pct: Range around spot for payoff profile (e.g., 0.30 = ±30%)
        num_points: Number of points for payoff profile

    Returns:
        Dictionary with:
            - payoff_profile: Price grid vs payoff at expiry
            - current_value: Current theoretical value
            - greeks: Aggregated Greeks (delta, gamma, theta, vega)
            - break_evens: Break-even price points
            - max_profit: Maximum profit
            - max_loss: Maximum loss
            - risk_reward_ratio: Max profit / Max loss
            - pop: Probability of profit (if IV provided)
    """
    # Convert dict to StructureSpec if needed
    if isinstance(structure_spec, dict):
        spec = _dict_to_structure_spec(structure_spec)
    else:
        spec = structure_spec

    if len(spec.legs) == 0:
        return {"error": "No legs in structure"}

    try:
        from quantsbin.derivativepricing import EqOption
        from quantsbin.derivativepricing.namesnmapper import EngineType

        return _analyze_with_quantsbin(spec, price_range_pct, num_points)

    except ImportError:
        logger.info("quantsbin not available, using internal analysis")
        return _analyze_structure_internal(spec, price_range_pct, num_points)


def _dict_to_structure_spec(d: Dict[str, Any]) -> StructureSpec:
    """Convert dictionary to StructureSpec."""
    legs = []
    for leg_dict in d.get("legs", []):
        legs.append(
            OptionLegSpec(
                option_type=leg_dict["option_type"].lower(),
                strike=leg_dict["strike"],
                expiry_days=leg_dict.get("expiry_days", leg_dict.get("dte", 30)),
                quantity=leg_dict["quantity"],
                premium=leg_dict.get("premium"),
                iv=leg_dict.get("iv"),
            )
        )

    return StructureSpec(
        underlying_symbol=d.get("underlying_symbol", d.get("symbol", "UNKNOWN")),
        underlying_price=d.get("underlying_price", d.get("spot", 100.0)),
        legs=legs,
        risk_free_rate=d.get("risk_free_rate", d.get("rate", 0.05)),
        dividend_yield=d.get("dividend_yield", 0.0),
    )


def _analyze_with_quantsbin(
    spec: StructureSpec,
    price_range_pct: float,
    num_points: int,
) -> Dict[str, Any]:
    """Analyze structure using quantsbin library."""
    from quantsbin.derivativepricing import EqOption
    from quantsbin.derivativepricing.namesnmapper import EngineType

    spot = spec.underlying_price
    rate = spec.risk_free_rate
    div_yield = spec.dividend_yield

    # Price range for payoff profile
    min_price = spot * (1 - price_range_pct)
    max_price = spot * (1 + price_range_pct)
    price_grid = np.linspace(min_price, max_price, num_points)

    # Calculate payoff at each price point
    payoff_at_expiry = np.zeros(num_points)
    current_value = 0.0
    total_premium = 0.0

    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0

    for leg in spec.legs:
        tte = leg.expiry_days / 365.0
        vol = leg.iv or 0.25  # Default vol if not provided

        # Create quantsbin option
        opt_type = "Call" if leg.option_type == "call" else "Put"

        try:
            option = EqOption(
                option_type=opt_type,
                strike=leg.strike,
                expiry_date=tte,
                spot0=spot,
                rate=rate,
                div_yield=div_yield,
                volatility=vol,
            )

            # Calculate current price and Greeks
            price = option.price(engine_type=EngineType.BSMEngine)
            delta = option.delta(engine_type=EngineType.BSMEngine)
            gamma = option.gamma(engine_type=EngineType.BSMEngine)
            theta = option.theta(engine_type=EngineType.BSMEngine)
            vega = option.vega(engine_type=EngineType.BSMEngine)

            # Aggregate with position sizing
            current_value += leg.quantity * price * 100
            total_delta += leg.quantity * delta * 100
            total_gamma += leg.quantity * gamma * 100
            total_theta += leg.quantity * theta * 100
            total_vega += leg.quantity * vega * 100

            if leg.premium:
                total_premium += leg.quantity * leg.premium * 100
            else:
                total_premium += leg.quantity * price * 100

        except Exception as e:
            logger.warning(f"quantsbin calculation failed for leg: {e}")
            continue

        # Calculate payoff at expiry for each price
        for i, price_point in enumerate(price_grid):
            if leg.option_type == "call":
                intrinsic = max(0, price_point - leg.strike)
            else:
                intrinsic = max(0, leg.strike - price_point)

            # P&L = intrinsic - premium paid (or + premium received)
            leg_premium = leg.premium if leg.premium else price
            payoff_at_expiry[i] += leg.quantity * (intrinsic - leg_premium) * 100

    # Find break-even points
    break_evens = _find_break_evens(price_grid, payoff_at_expiry)

    # Max profit and loss
    max_profit = float(np.max(payoff_at_expiry))
    max_loss = float(np.min(payoff_at_expiry))

    # Risk/reward ratio
    if abs(max_loss) > 0:
        risk_reward = abs(max_profit / max_loss)
    else:
        risk_reward = float("inf")

    # Calculate probability of profit (rough estimate)
    pop = _estimate_pop(price_grid, payoff_at_expiry, spot, vol, tte) if vol else None

    return {
        "structure_type": _identify_structure_type(spec.legs),
        "underlying_symbol": spec.underlying_symbol,
        "underlying_price": spot,
        "num_legs": len(spec.legs),
        "payoff_profile": {
            "prices": price_grid.tolist(),
            "payoffs": payoff_at_expiry.tolist(),
        },
        "current_value": float(current_value),
        "total_premium": float(total_premium),
        "net_debit_credit": "debit" if total_premium > 0 else "credit",
        "greeks": {
            "delta": float(total_delta),
            "gamma": float(total_gamma),
            "theta": float(total_theta),
            "vega": float(total_vega),
        },
        "break_evens": break_evens,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "risk_reward_ratio": risk_reward,
        "probability_of_profit": pop,
        "is_defined_risk": abs(max_loss) < float("inf"),
    }


def _analyze_structure_internal(
    spec: StructureSpec,
    price_range_pct: float,
    num_points: int,
) -> Dict[str, Any]:
    """Internal structure analysis without quantsbin."""
    from quantcore.options.pricing import black_scholes_price, black_scholes_greeks
    from quantcore.options.models import OptionType

    spot = spec.underlying_price
    rate = spec.risk_free_rate
    div_yield = spec.dividend_yield

    # Price range
    min_price = spot * (1 - price_range_pct)
    max_price = spot * (1 + price_range_pct)
    price_grid = np.linspace(min_price, max_price, num_points)

    payoff_at_expiry = np.zeros(num_points)
    current_value = 0.0
    total_premium = 0.0

    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0

    avg_vol = 0.25  # Default
    avg_tte = 30 / 365.0

    for leg in spec.legs:
        tte = leg.expiry_days / 365.0
        vol = leg.iv or 0.25
        avg_vol = vol
        avg_tte = tte

        opt_type = OptionType.CALL if leg.option_type == "call" else OptionType.PUT

        # Current price and Greeks
        try:
            price = black_scholes_price(
                spot, leg.strike, tte, rate, vol, opt_type, div_yield
            )
            greeks = black_scholes_greeks(
                spot, leg.strike, tte, rate, vol, opt_type, div_yield
            )

            current_value += leg.quantity * price * 100
            total_delta += leg.quantity * greeks.delta * 100
            total_gamma += leg.quantity * greeks.gamma * 100
            total_theta += leg.quantity * greeks.theta * 100
            total_vega += leg.quantity * greeks.vega * 100

            leg_premium = leg.premium if leg.premium else price
            total_premium += leg.quantity * leg_premium * 100

        except Exception as e:
            logger.warning(f"Internal pricing failed: {e}")
            continue

        # Payoff at expiry
        for i, price_point in enumerate(price_grid):
            if leg.option_type == "call":
                intrinsic = max(0, price_point - leg.strike)
            else:
                intrinsic = max(0, leg.strike - price_point)

            leg_premium = leg.premium if leg.premium else price
            payoff_at_expiry[i] += leg.quantity * (intrinsic - leg_premium) * 100

    # Break-evens
    break_evens = _find_break_evens(price_grid, payoff_at_expiry)

    # Max profit/loss
    max_profit = float(np.max(payoff_at_expiry))
    max_loss = float(np.min(payoff_at_expiry))

    if abs(max_loss) > 0:
        risk_reward = abs(max_profit / max_loss)
    else:
        risk_reward = float("inf")

    pop = _estimate_pop(price_grid, payoff_at_expiry, spot, avg_vol, avg_tte)

    return {
        "structure_type": _identify_structure_type(spec.legs),
        "underlying_symbol": spec.underlying_symbol,
        "underlying_price": spot,
        "num_legs": len(spec.legs),
        "payoff_profile": {
            "prices": price_grid.tolist(),
            "payoffs": payoff_at_expiry.tolist(),
        },
        "current_value": float(current_value),
        "total_premium": float(total_premium),
        "net_debit_credit": "debit" if total_premium > 0 else "credit",
        "greeks": {
            "delta": float(total_delta),
            "gamma": float(total_gamma),
            "theta": float(total_theta),
            "vega": float(total_vega),
        },
        "break_evens": break_evens,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "risk_reward_ratio": risk_reward,
        "probability_of_profit": pop,
        "is_defined_risk": abs(max_loss) < float("inf"),
    }


def _find_break_evens(price_grid: np.ndarray, payoffs: np.ndarray) -> List[float]:
    """Find break-even price points where payoff crosses zero."""
    break_evens = []

    for i in range(len(payoffs) - 1):
        if payoffs[i] * payoffs[i + 1] < 0:
            # Linear interpolation to find exact crossing
            slope = (payoffs[i + 1] - payoffs[i]) / (price_grid[i + 1] - price_grid[i])
            if slope != 0:
                be = price_grid[i] - payoffs[i] / slope
                break_evens.append(round(float(be), 2))

    return break_evens


def _estimate_pop(
    price_grid: np.ndarray,
    payoffs: np.ndarray,
    spot: float,
    vol: float,
    tte: float,
) -> Optional[float]:
    """Estimate probability of profit using lognormal distribution."""
    if tte <= 0 or vol <= 0:
        return None

    from scipy.stats import norm

    # Find profitable region
    profitable_mask = payoffs > 0
    if not np.any(profitable_mask):
        return 0.0
    if np.all(profitable_mask):
        return 100.0

    # Simple estimation: probability price ends in profitable region
    # Using Black-Scholes lognormal assumption
    log_spot = np.log(spot)
    log_std = vol * np.sqrt(tte)

    # Find price ranges that are profitable
    profitable_prices = price_grid[profitable_mask]

    if len(profitable_prices) == 0:
        return 0.0

    # Calculate probability for each profitable range
    total_prob = 0.0

    # Simple approach: use min and max profitable prices
    min_prof = profitable_prices.min()
    max_prof = profitable_prices.max()

    # Probability between min and max profitable prices
    z_min = (np.log(min_prof) - log_spot) / log_std
    z_max = (np.log(max_prof) - log_spot) / log_std

    prob = norm.cdf(z_max) - norm.cdf(z_min)

    return round(float(prob * 100), 1)


def _identify_structure_type(legs: List[OptionLegSpec]) -> str:
    """Identify the type of option structure based on legs."""
    if len(legs) == 1:
        leg = legs[0]
        direction = "Long" if leg.quantity > 0 else "Short"
        opt_type = "Call" if leg.option_type == "call" else "Put"
        return f"{direction} {opt_type}"

    if len(legs) == 2:
        leg1, leg2 = legs

        # Both same expiry?
        same_expiry = leg1.expiry_days == leg2.expiry_days

        # Vertical spread
        if same_expiry and leg1.option_type == leg2.option_type:
            if leg1.quantity * leg2.quantity < 0:
                opt_type = "Call" if leg1.option_type == "call" else "Put"
                long_leg = leg1 if leg1.quantity > 0 else leg2
                short_leg = leg2 if leg1.quantity > 0 else leg1

                if opt_type == "Call":
                    if long_leg.strike < short_leg.strike:
                        return "Bull Call Spread"
                    else:
                        return "Bear Call Spread"
                else:
                    if long_leg.strike > short_leg.strike:
                        return "Bear Put Spread"
                    else:
                        return "Bull Put Spread"

        # Straddle/Strangle
        if same_expiry and leg1.option_type != leg2.option_type:
            if leg1.quantity * leg2.quantity > 0:
                if leg1.strike == leg2.strike:
                    direction = "Long" if leg1.quantity > 0 else "Short"
                    return f"{direction} Straddle"
                else:
                    direction = "Long" if leg1.quantity > 0 else "Short"
                    return f"{direction} Strangle"

        # Calendar spread
        if not same_expiry and leg1.strike == leg2.strike:
            return "Calendar Spread"

    if len(legs) == 4:
        # Could be iron condor, butterfly, etc.
        strikes = sorted([l.strike for l in legs])

        # Iron condor: 4 different strikes
        if len(set(strikes)) == 4:
            return "Iron Condor"

        # Butterfly: 3 strikes, middle one has 2x quantity
        if len(set(strikes)) == 3:
            return "Butterfly"

    return f"Custom {len(legs)}-Leg Structure"


def get_standard_structures() -> Dict[str, Dict[str, Any]]:
    """
    Get specifications for standard option structures.

    Returns:
        Dictionary of structure templates
    """
    return {
        "bull_call_spread": {
            "description": "Buy lower strike call, sell higher strike call",
            "legs_template": [
                {"option_type": "call", "strike_offset": -1, "quantity": 1},
                {"option_type": "call", "strike_offset": 1, "quantity": -1},
            ],
            "max_profit": "Width - Net Debit",
            "max_loss": "Net Debit",
            "breakeven": "Lower Strike + Net Debit",
        },
        "bear_put_spread": {
            "description": "Buy higher strike put, sell lower strike put",
            "legs_template": [
                {"option_type": "put", "strike_offset": 1, "quantity": 1},
                {"option_type": "put", "strike_offset": -1, "quantity": -1},
            ],
            "max_profit": "Width - Net Debit",
            "max_loss": "Net Debit",
            "breakeven": "Higher Strike - Net Debit",
        },
        "iron_condor": {
            "description": "Sell OTM strangle, buy further OTM strangle for protection",
            "legs_template": [
                {"option_type": "put", "strike_offset": -2, "quantity": 1},
                {"option_type": "put", "strike_offset": -1, "quantity": -1},
                {"option_type": "call", "strike_offset": 1, "quantity": -1},
                {"option_type": "call", "strike_offset": 2, "quantity": 1},
            ],
            "max_profit": "Net Credit",
            "max_loss": "Width - Net Credit",
            "breakeven": "Multiple (between short strikes)",
        },
        "long_straddle": {
            "description": "Buy ATM call and ATM put",
            "legs_template": [
                {"option_type": "call", "strike_offset": 0, "quantity": 1},
                {"option_type": "put", "strike_offset": 0, "quantity": 1},
            ],
            "max_profit": "Unlimited",
            "max_loss": "Total Premium",
            "breakeven": "Strike ± Total Premium",
        },
        "short_strangle": {
            "description": "Sell OTM call and OTM put",
            "legs_template": [
                {"option_type": "call", "strike_offset": 1, "quantity": -1},
                {"option_type": "put", "strike_offset": -1, "quantity": -1},
            ],
            "max_profit": "Net Credit",
            "max_loss": "Unlimited",
            "breakeven": "Multiple (outside short strikes)",
        },
    }


def build_structure_from_template(
    template_name: str,
    underlying_symbol: str,
    underlying_price: float,
    atm_strike: float,
    strike_width: float,
    expiry_days: int,
    quantity_multiplier: int = 1,
    iv: float = 0.25,
) -> StructureSpec:
    """
    Build a structure specification from a template.

    Args:
        template_name: Name of template (e.g., "bull_call_spread")
        underlying_symbol: Symbol
        underlying_price: Current price
        atm_strike: ATM strike price
        strike_width: Width between strikes
        expiry_days: Days to expiry
        quantity_multiplier: Scale the quantities
        iv: Implied volatility for pricing

    Returns:
        StructureSpec ready for analysis
    """
    templates = get_standard_structures()

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")

    template = templates[template_name]
    legs = []

    for leg_template in template["legs_template"]:
        strike = atm_strike + leg_template["strike_offset"] * strike_width

        legs.append(
            OptionLegSpec(
                option_type=leg_template["option_type"],
                strike=strike,
                expiry_days=expiry_days,
                quantity=leg_template["quantity"] * quantity_multiplier,
                iv=iv,
            )
        )

    return StructureSpec(
        underlying_symbol=underlying_symbol,
        underlying_price=underlying_price,
        legs=legs,
    )
