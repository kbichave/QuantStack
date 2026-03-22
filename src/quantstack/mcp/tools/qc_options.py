# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP options tools — pricing, Greeks, IV, and structure analysis.

Extracted from ``quantcore.mcp.server`` to keep tool modules focused.
All helpers come from ``quantcore.mcp._helpers``; the ``mcp`` singleton
is imported from ``quantcore.mcp.server``.
"""

import os
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantstack.core.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn
from quantstack.core.options.adapters.financepy_adapter import (
    price_american_option as price_american,
)
from quantstack.core.options.adapters.quantsbin_adapter import (
    analyze_structure_quantsbin,
)
from quantstack.core.options.engine import (
    compute_greeks_dispatch,
    compute_iv_dispatch,
    price_option_dispatch,
)
from quantstack.data.adapters.alpaca import AlpacaAdapter
from quantstack.data.adapters.polygon_adapter import PolygonAdapter
from quantstack.data.storage import DataStore
from quantstack.mcp._helpers import (
    _get_reader,
    _parse_timeframe,
)
from quantstack.mcp.server import mcp


# =============================================================================
# OPTIONS TOOLS
# =============================================================================


@mcp.tool()
async def price_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    exercise_style: str = "european",
) -> dict[str, Any]:
    """
    Calculate option price using production-grade pricing engine.

    Supports both European and American options with automatic backend selection:
    - European options: Uses vollib (Black-Scholes-Merton)
    - American options: Uses financepy (binomial tree)

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years (e.g., 0.25 for 3 months)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free interest rate (e.g., 0.05 for 5%)
        option_type: "call" or "put"
        dividend_yield: Continuous dividend yield (e.g., 0.02 for 2%)
        exercise_style: "european" or "american"

    Returns:
        Dictionary with option price, Greeks, and analysis
    """
    try:
        result = price_option_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            exercise_style=exercise_style,
            backend="auto",
        )

        # Add analysis section for compatibility
        moneyness = spot / strike
        is_call = option_type.lower() in ("call", "c")
        itm = (spot > strike) if is_call else (spot < strike)
        intrinsic = max(0, spot - strike) if is_call else max(0, strike - spot)

        result["analysis"] = {
            "moneyness": round(moneyness, 4),
            "is_itm": itm,
            "days_to_expiry": round(time_to_expiry * 365),
            "intrinsic_value": round(intrinsic, 4),
            "time_value": round(result["price"] - intrinsic, 4),
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """
    Compute option Greeks (sensitivities) using production-grade engine.

    Uses vollib for fast, accurate Greeks calculation with automatic
    fallback to internal implementation if needed.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free interest rate
        option_type: "call" or "put"
        dividend_yield: Continuous dividend yield

    Returns:
        Dictionary with detailed Greeks, interpretations, and risk metrics
    """
    try:
        result = compute_greeks_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            backend="auto",
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_implied_vol(
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_price: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> dict[str, Any]:
    """
    Calculate implied volatility from market option price.

    Uses Newton-Raphson method via vollib for fast, accurate IV calculation.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        option_price: Market price of the option
        risk_free_rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"

    Returns:
        Dictionary with implied volatility and analysis
    """
    try:
        result = compute_iv_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_price=option_price,
            option_type=option_type,
            backend="auto",
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def analyze_option_structure(
    structure_spec: dict[str, Any],
    price_range_pct: float = 0.30,
) -> dict[str, Any]:
    """
    Analyze multi-leg option structure for payoff, Greeks, and key metrics.

    Supports all standard structures: verticals, straddles, strangles,
    iron condors, butterflies, and custom multi-leg positions.

    Args:
        structure_spec: Structure specification dictionary with:
            - underlying_symbol: Symbol (e.g., "SPY")
            - underlying_price: Current price (e.g., 450.0)
            - legs: List of leg dictionaries, each with:
                - option_type: "call" or "put"
                - strike: Strike price
                - expiry_days: Days to expiration
                - quantity: Positive for long, negative for short
                - premium: (optional) Entry premium
                - iv: (optional) Implied volatility
            - risk_free_rate: (optional) Rate, default 0.05
        price_range_pct: Range around spot for payoff profile (default 30%)

    Returns:
        Dictionary with:
            - structure_type: Identified structure name
            - payoff_profile: Price grid vs payoff at expiry
            - greeks: Aggregated position Greeks
            - break_evens: Break-even price points
            - max_profit, max_loss: Profit/loss boundaries
            - risk_reward_ratio: Max profit / Max loss
            - probability_of_profit: Estimated POP

    Example:
        analyze_option_structure({
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {"option_type": "call", "strike": 445, "expiry_days": 30, "quantity": 1, "iv": 0.20},
                {"option_type": "call", "strike": 455, "expiry_days": 30, "quantity": -1, "iv": 0.18}
            ]
        })
    """
    try:
        result = analyze_structure_quantsbin(
            structure_spec=structure_spec,
            price_range_pct=price_range_pct,
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_portfolio_stats(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Compute comprehensive portfolio performance statistics.

    Uses ffn library for production-grade analytics including Sharpe,
    Sortino, Calmar ratios, drawdown analysis, and distribution metrics.

    Args:
        equity_curve: List of portfolio equity values over time
        risk_free_rate: Annual risk-free rate for ratio calculations
        periods_per_year: Trading periods per year (252 for daily, 52 for weekly)

    Returns:
        Dictionary with:
            - Return metrics: total_return, cagr, annualized_return
            - Risk metrics: volatility, max_drawdown, VaR, CVaR
            - Ratios: sharpe_ratio, sortino_ratio, calmar_ratio
            - Distribution: skewness, kurtosis, best/worst day
            - Drawdown details: duration, recovery time
    """
    try:
        # Convert to pandas Series
        equity = pd.Series(equity_curve)

        result = compute_portfolio_stats_ffn(
            equity_curve=equity,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def price_american_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    num_steps: int = 100,
) -> dict[str, Any]:
    """
    Price American option using binomial tree method.

    American options can be exercised at any time before expiration,
    which may be valuable for dividend-paying stocks (calls) or
    deep ITM puts.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        num_steps: Number of tree steps (more = accurate but slower)

    Returns:
        Dictionary with:
            - price: American option price
            - european_price: European equivalent price
            - early_exercise_premium: Value of early exercise right
            - delta, gamma: Greeks from tree
    """
    try:
        result = price_american(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            num_steps=num_steps,
        )

        result["inputs"] = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "option_type": option_type,
        }

        return result

    except Exception as e:
        return {"error": str(e)}


async def _compute_option_chain_impl(
    symbol: str,
    expiry_date: str | None = None,
    min_delta: float = 0.05,
    max_delta: float = 0.95,
) -> dict[str, Any]:
    """Core logic for compute_option_chain — callable from other tools."""
    store = _get_reader()

    try:
        df = store.load_ohlcv(symbol, _parse_timeframe("daily"))

        if df.empty:
            return {"error": f"No price data for {symbol}"}

        underlying_price = float(df["close"].iloc[-1])

        if underlying_price > 100:
            strike_increment = 5.0
        elif underlying_price > 50:
            strike_increment = 2.5
        else:
            strike_increment = 1.0

        vol_estimate = 0.25
        atm_strike = round(underlying_price / strike_increment) * strike_increment

        strikes = [
            atm_strike + i * strike_increment
            for i in range(-6, 7)
            if (atm_strike + i * strike_increment) > 0
        ]

        dte = 30
        tte = dte / 365.0
        rate = 0.05
        div_yield = 0.01

        calls = []
        puts = []

        for strike in strikes:
            call_result = price_option_dispatch(
                underlying_price, strike, tte, vol_estimate, rate, div_yield, "call"
            )
            call_greeks = call_result.get("greeks", {})
            call_delta = abs(call_greeks.get("delta", 0))

            if min_delta <= call_delta <= max_delta:
                calls.append(
                    {
                        "strike": strike,
                        "bid": round(call_result["price"] * 0.98, 2),
                        "ask": round(call_result["price"] * 1.02, 2),
                        "mid": round(call_result["price"], 2),
                        "iv": vol_estimate,
                        "delta": round(call_greeks.get("delta", 0), 4),
                        "gamma": round(call_greeks.get("gamma", 0), 6),
                        "theta": round(call_greeks.get("theta", 0), 4),
                        "vega": round(call_greeks.get("vega", 0), 4),
                        "moneyness": round(np.log(underlying_price / strike), 4),
                        "dte": dte,
                    }
                )

            put_result = price_option_dispatch(
                underlying_price, strike, tte, vol_estimate, rate, div_yield, "put"
            )
            put_greeks = put_result.get("greeks", {})
            put_delta = abs(put_greeks.get("delta", 0))

            if min_delta <= put_delta <= max_delta:
                puts.append(
                    {
                        "strike": strike,
                        "bid": round(put_result["price"] * 0.98, 2),
                        "ask": round(put_result["price"] * 1.02, 2),
                        "mid": round(put_result["price"], 2),
                        "iv": vol_estimate,
                        "delta": round(put_greeks.get("delta", 0), 4),
                        "gamma": round(put_greeks.get("gamma", 0), 6),
                        "theta": round(put_greeks.get("theta", 0), 4),
                        "vega": round(put_greeks.get("vega", 0), 4),
                        "moneyness": round(np.log(underlying_price / strike), 4),
                        "dte": dte,
                    }
                )

        return {
            "symbol": symbol,
            "underlying_price": underlying_price,
            "as_of": str(df.index[-1]),
            "calls": sorted(calls, key=lambda x: x["strike"]),
            "puts": sorted(puts, key=lambda x: x["strike"]),
            "chain_metrics": {
                "atm_strike": atm_strike,
                "atm_iv": vol_estimate,
                "num_calls": len(calls),
                "num_puts": len(puts),
                "dte": dte,
            },
            "note": "Synthetic chain - use broker MCP for live data",
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def compute_option_chain(
    symbol: str,
    expiry_date: str | None = None,
    min_delta: float = 0.05,
    max_delta: float = 0.95,
) -> dict[str, Any]:
    """
    Compute normalized option chain data for a symbol.

    Returns clean, normalized chain data suitable for OptionStructuringAgent.
    Calculates moneyness, IVs, and Greeks for all strikes.

    Args:
        symbol: Underlying symbol
        expiry_date: Target expiry (YYYY-MM-DD), uses nearest if not provided
        min_delta: Minimum delta to include (filters far OTM)
        max_delta: Maximum delta to include (filters deep ITM)

    Returns:
        Dictionary with:
            - calls: List of call options with Greeks
            - puts: List of put options with Greeks
            - underlying_price: Current price
            - expiries: Available expiration dates
            - chain_metrics: ATM IV, put/call skew, etc.
    """
    return await _compute_option_chain_impl(symbol, expiry_date, min_delta, max_delta)


@mcp.tool()
async def compute_multi_leg_price(
    legs: list[dict[str, Any]],
    underlying_price: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """
    Price a multi-leg options structure in one call.

    Aggregates pricing, Greeks, and risk metrics across all legs.
    Used by OptionStructuringAgent for rapid structure evaluation.

    Args:
        legs: List of leg specifications, each with:
            - option_type: "call" or "put"
            - strike: Strike price
            - expiry_days: Days to expiration
            - quantity: Number of contracts (negative for short)
            - iv: Implied volatility (optional, defaults to 0.25)
        underlying_price: Current underlying price
        rate: Risk-free rate
        dividend_yield: Continuous dividend yield

    Returns:
        Dictionary with:
            - total_price: Net premium (debit or credit)
            - leg_prices: Individual leg prices
            - net_greeks: Aggregated position Greeks
            - max_profit: Maximum profit potential
            - max_loss: Maximum loss potential
            - break_evens: Break-even prices
    """
    try:
        if not legs:
            return {"error": "No legs provided"}

        leg_results = []
        total_premium = 0.0
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        net_rho = 0.0

        for i, leg in enumerate(legs):
            opt_type = leg.get("option_type", "call")
            strike = leg["strike"]
            expiry_days = leg.get("expiry_days", 30)
            quantity = leg.get("quantity", 1)
            iv = leg.get("iv", 0.25)

            tte = expiry_days / 365.0

            result = price_option_dispatch(
                underlying_price, strike, tte, iv, rate, dividend_yield, opt_type
            )

            price = result["price"]
            greeks = result.get("greeks", {})

            # Aggregate (negative quantity = short position = credit)
            leg_premium = price * quantity * 100  # Per contract value
            total_premium += leg_premium

            net_delta += greeks.get("delta", 0) * quantity * 100
            net_gamma += greeks.get("gamma", 0) * quantity * 100
            net_theta += greeks.get("theta", 0) * quantity * 100
            net_vega += greeks.get("vega", 0) * quantity * 100
            net_rho += greeks.get("rho", 0) * quantity * 100

            leg_results.append(
                {
                    "leg_index": i,
                    "option_type": opt_type,
                    "strike": strike,
                    "expiry_days": expiry_days,
                    "quantity": quantity,
                    "price_per_contract": round(price, 2),
                    "total_value": round(leg_premium, 2),
                    "delta": round(greeks.get("delta", 0), 4),
                }
            )

        # Determine structure type and estimate max profit/loss
        is_debit = total_premium > 0

        # Simplified max profit/loss calculation
        strikes = sorted([leg["strike"] for leg in legs])

        if is_debit:
            max_loss = -abs(total_premium)
            max_profit = None  # Potentially unlimited for naked calls
        else:
            max_profit = abs(total_premium)
            max_loss = None  # Need more analysis for spreads

        # For defined risk spreads, calculate actual max loss
        if len(legs) >= 2:
            strike_width = max(strikes) - min(strikes)
            if strike_width > 0:
                if is_debit:
                    max_profit = strike_width * 100 - abs(total_premium)
                else:
                    max_loss = -(strike_width * 100 - abs(total_premium))

        return {
            "underlying_price": underlying_price,
            "total_premium": round(total_premium, 2),
            "is_debit": is_debit,
            "leg_prices": leg_results,
            "net_greeks": {
                "delta": round(net_delta, 2),
                "gamma": round(net_gamma, 4),
                "theta": round(net_theta, 2),
                "vega": round(net_vega, 2),
                "rho": round(net_rho, 2),
            },
            "risk_profile": {
                "max_profit": round(max_profit, 2) if max_profit else "unlimited",
                "max_loss": round(max_loss, 2) if max_loss else "undefined",
                "risk_reward_ratio": (
                    round(abs(max_profit / max_loss), 2)
                    if max_profit and max_loss and max_loss != 0
                    else None
                ),
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def score_trade_structure(
    structure_spec: dict[str, Any],
    vol_surface: dict[str, Any] | None = None,
    market_regime: str | None = None,
) -> dict[str, Any]:
    """
    Score an options structure for trade quality.

    Unified scoring model combining:
    - Expected value based on structure analysis
    - Convexity score (gamma/theta ratio)
    - Vol surface alignment
    - Regime suitability

    Args:
        structure_spec: Structure specification with legs and underlying
        vol_surface: Optional SABR surface for vol edge detection
        market_regime: Optional regime ("bull", "bear", "sideways")

    Returns:
        Dictionary with:
            - total_score: Overall score 0-100
            - component_scores: Individual scoring factors
            - recommendation: "strong_buy", "buy", "neutral", "avoid"
            - risk_flags: Any concerns identified
    """
    try:
        # Analyze structure
        analysis = analyze_structure_quantsbin(structure_spec)

        if "error" in analysis:
            return {"error": analysis["error"]}

        greeks = analysis.get("greeks", {})
        max_profit = analysis.get("max_profit", 0)
        max_loss = analysis.get("max_loss", 0)
        is_defined_risk = analysis.get("is_defined_risk", False)

        scores = {}
        risk_flags = []

        # 1. Risk/Reward Score (0-25)
        if max_loss and max_loss != 0:
            rr_ratio = abs(max_profit / max_loss) if max_profit else 0
            if rr_ratio >= 3:
                scores["risk_reward"] = 25
            elif rr_ratio >= 2:
                scores["risk_reward"] = 20
            elif rr_ratio >= 1:
                scores["risk_reward"] = 15
            elif rr_ratio >= 0.5:
                scores["risk_reward"] = 10
            else:
                scores["risk_reward"] = 5
                risk_flags.append("Poor risk/reward ratio")
        else:
            scores["risk_reward"] = 10
            if not is_defined_risk:
                risk_flags.append("Undefined risk structure")

        # 2. Convexity Score (0-25) - gamma/theta tradeoff
        gamma = abs(greeks.get("gamma", 0))
        theta = abs(greeks.get("theta", 0))

        if theta > 0:
            convexity = gamma / theta
            if convexity >= 0.5:
                scores["convexity"] = 25
            elif convexity >= 0.2:
                scores["convexity"] = 20
            elif convexity >= 0.1:
                scores["convexity"] = 15
            else:
                scores["convexity"] = 10
        else:
            scores["convexity"] = 15  # No theta decay

        # 3. Delta Alignment Score (0-25)
        delta = greeks.get("delta", 0)

        if market_regime:
            if market_regime == "bull" and delta > 0:
                scores["regime_alignment"] = min(25, 15 + abs(delta) * 10)
            elif market_regime == "bear" and delta < 0:
                scores["regime_alignment"] = min(25, 15 + abs(delta) * 10)
            elif market_regime == "sideways" and abs(delta) < 20:
                scores["regime_alignment"] = 25
            else:
                scores["regime_alignment"] = 10
                risk_flags.append("Delta misaligned with regime")
        else:
            scores["regime_alignment"] = 15  # Neutral without regime info

        # 4. Probability Score (0-25)
        pop = analysis.get("probability_of_profit")
        if pop:
            if pop >= 70:
                scores["probability"] = 25
            elif pop >= 50:
                scores["probability"] = 20
            elif pop >= 35:
                scores["probability"] = 15
            else:
                scores["probability"] = 10
                risk_flags.append("Low probability of profit")
        else:
            scores["probability"] = 15

        # Calculate total score
        total_score = sum(scores.values())

        # Determine recommendation
        if total_score >= 85:
            recommendation = "strong_buy"
        elif total_score >= 70:
            recommendation = "buy"
        elif total_score >= 50:
            recommendation = "neutral"
        else:
            recommendation = "avoid"

        return {
            "total_score": total_score,
            "max_score": 100,
            "component_scores": scores,
            "recommendation": recommendation,
            "risk_flags": risk_flags,
            "structure_type": analysis.get("structure_type", "unknown"),
            "analysis_summary": {
                "max_profit": max_profit,
                "max_loss": max_loss,
                "delta": delta,
                "is_defined_risk": is_defined_risk,
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def simulate_trade_outcome(
    trade_template: dict[str, Any],
    num_scenarios: int = 1000,
    holding_days: int = 30,
    vol_shock_range: float = 0.10,
) -> dict[str, Any]:
    """
    Simulate trade P&L distribution using Monte Carlo.

    Runs scenarios with:
    - Price paths based on historical vol
    - Vol surface shifts
    - Time decay effects

    Args:
        trade_template: Trade template from generate_trade_template
        num_scenarios: Number of Monte Carlo paths
        holding_days: Days to simulate holding
        vol_shock_range: Max vol change (+/- as fraction)

    Returns:
        Dictionary with:
            - pnl_distribution: Percentile P&Ls
            - expected_pnl: Mean P&L
            - probability_profit: % of winning scenarios
            - var_95: 95% Value at Risk
            - scenario_stats: Summary statistics
    """

    try:
        legs = trade_template.get("legs", [])
        underlying_price = trade_template.get("underlying_price")

        if not legs or not underlying_price:
            return {
                "error": "Invalid trade template - missing legs or underlying_price"
            }

        # Entry premium (current value)
        entry_result = await compute_multi_leg_price(
            legs=legs,
            underlying_price=underlying_price,
        )
        entry_premium = entry_result.get("total_premium", 0)

        # Simulate scenarios
        np.random.seed(42)

        # Base parameters
        base_vol = legs[0].get("iv", 0.25)
        annual_vol = base_vol
        daily_vol = annual_vol / np.sqrt(252)

        pnl_results = []

        for _ in range(num_scenarios):
            # Simulate price move
            price_return = np.random.normal(0, daily_vol * np.sqrt(holding_days))
            final_price = underlying_price * np.exp(price_return)

            # Simulate vol change
            vol_change = np.random.uniform(-vol_shock_range, vol_shock_range)
            new_vol = max(0.05, base_vol * (1 + vol_change))

            # Update legs with new time to expiry
            new_legs = []
            for leg in legs:
                new_leg = leg.copy()
                new_leg["expiry_days"] = max(
                    1, leg.get("expiry_days", 30) - holding_days
                )
                new_leg["iv"] = new_vol
                new_legs.append(new_leg)

            # Calculate exit value
            exit_result = await compute_multi_leg_price(
                legs=new_legs,
                underlying_price=final_price,
            )
            exit_premium = exit_result.get("total_premium", 0)

            # P&L = exit value - entry value
            pnl = exit_premium - entry_premium
            pnl_results.append(pnl)

        pnl_array = np.array(pnl_results)

        # Calculate statistics
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pnl_percentiles = {
            f"p{p}": round(float(np.percentile(pnl_array, p)), 2) for p in percentiles
        }

        return {
            "num_scenarios": num_scenarios,
            "holding_days": holding_days,
            "entry_premium": round(entry_premium, 2),
            "expected_pnl": round(float(np.mean(pnl_array)), 2),
            "median_pnl": round(float(np.median(pnl_array)), 2),
            "std_pnl": round(float(np.std(pnl_array)), 2),
            "probability_profit": round(
                float(np.sum(pnl_array > 0) / len(pnl_array) * 100), 1
            ),
            "var_95": round(float(np.percentile(pnl_array, 5)), 2),  # 5th percentile
            "cvar_95": round(
                float(np.mean(pnl_array[pnl_array <= np.percentile(pnl_array, 5)])), 2
            ),
            "max_profit_scenario": round(float(np.max(pnl_array)), 2),
            "max_loss_scenario": round(float(np.min(pnl_array)), 2),
            "pnl_percentiles": pnl_percentiles,
        }

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# LIVE OPTIONS CHAIN TOOLS (Phase 1 — v0.5.0)
# =============================================================================


async def _get_options_chain_impl(
    symbol: str,
    expiry_min_days: int = 7,
    expiry_max_days: int = 45,
    option_type: str | None = None,
) -> dict[str, Any]:
    """Core logic for get_options_chain — callable from other tools."""
    # Try Alpaca first, then Polygon, then synthetic fallback
    contracts: list[dict] | None = None
    data_source = "synthetic"

    alpaca_key = os.getenv("ALPACA_API_KEY", "")
    if alpaca_key:
        try:
            adapter = AlpacaAdapter(
                api_key=alpaca_key, secret_key=os.getenv("ALPACA_SECRET_KEY", "")
            )
            contracts = adapter.fetch_options_chain(
                symbol, expiry_min_days, expiry_max_days
            )
            if contracts:
                data_source = "alpaca"
        except Exception:
            contracts = None

    if not contracts:
        polygon_key = os.getenv("POLYGON_API_KEY", "")
        if polygon_key:
            try:
                adapter = PolygonAdapter(api_key=polygon_key)
                contracts = adapter.fetch_options_chain(
                    symbol, expiry_min_days, expiry_max_days
                )
                if contracts:
                    data_source = "polygon"
            except Exception:
                contracts = None

    # Fallback: synthetic chain from compute_option_chain
    if not contracts:
        synthetic = await _compute_option_chain_impl(symbol=symbol)
        if "error" in synthetic:
            return synthetic
        underlying_price = synthetic["underlying_price"]
        all_contracts = []
        for c in synthetic.get("calls", []):
            all_contracts.append(
                {
                    **c,
                    "option_type": "call",
                    "underlying": symbol,
                    "expiry": None,
                    "contract_id": f"{symbol}_call_{c['strike']}",
                    "open_interest": None,
                    "volume": None,
                    "last": None,
                    "mid": c.get("mid", c.get("ask")),
                }
            )
        for p in synthetic.get("puts", []):
            all_contracts.append(
                {
                    **p,
                    "option_type": "put",
                    "underlying": symbol,
                    "expiry": None,
                    "contract_id": f"{symbol}_put_{p['strike']}",
                    "open_interest": None,
                    "volume": None,
                    "last": None,
                    "mid": p.get("mid", p.get("ask")),
                }
            )
        contracts = all_contracts
        data_source = "synthetic"
        as_of = synthetic.get("as_of", "synthetic")
    else:
        store = _get_reader()
        try:
            df = store.load_ohlcv(symbol, _parse_timeframe("daily"))
            underlying_price = float(df["close"].iloc[-1]) if not df.empty else None
            as_of = str(df.index[-1]) if not df.empty else "unknown"
        except Exception:
            underlying_price = None
            as_of = "unknown"
        finally:
            store.close()

        # Write to options_chains table for get_iv_surface to read later
        try:
            ds = DataStore()
            today_str = str(date.today())
            for c in contracts:
                ds.conn.execute(
                    """
                    INSERT OR REPLACE INTO options_chains
                        (contract_id, underlying, data_date, expiry, strike, option_type,
                         bid, ask, mid, last, volume, open_interest, iv,
                         delta, gamma, theta, vega, rho)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        c.get("contract_id"),
                        c.get("underlying"),
                        today_str,
                        c.get("expiry"),
                        c.get("strike"),
                        c.get("option_type"),
                        c.get("bid"),
                        c.get("ask"),
                        c.get("mid"),
                        c.get("last"),
                        c.get("volume"),
                        c.get("open_interest"),
                        c.get("iv"),
                        c.get("delta"),
                        c.get("gamma"),
                        c.get("theta"),
                        c.get("vega"),
                        None,
                    ],
                )
            ds.close()
        except Exception:
            pass  # Storage write is best-effort; chain data is still returned

    # Filter by option_type if requested
    if option_type:
        otype = option_type.lower()
        contracts = [c for c in contracts if c.get("option_type", "").lower() == otype]

    calls = [c for c in contracts if c.get("option_type", "").lower() == "call"]
    puts = [c for c in contracts if c.get("option_type", "").lower() == "put"]

    # Compute ATM metrics
    atm_iv = None
    atm_strike = None
    if underlying_price and contracts:
        atm_contracts = sorted(
            contracts, key=lambda c: abs((c.get("strike") or 0) - underlying_price)
        )
        if atm_contracts:
            atm_strike = atm_contracts[0].get("strike")
            atm_ivs = [c["iv"] for c in atm_contracts[:4] if c.get("iv")]
            atm_iv = round(sum(atm_ivs) / len(atm_ivs), 4) if atm_ivs else None

    return {
        "symbol": symbol,
        "underlying_price": underlying_price,
        "as_of": as_of,
        "source": data_source,
        "contracts": contracts,
        "calls": sorted(
            calls, key=lambda c: (c.get("expiry") or "", c.get("strike") or 0)
        ),
        "puts": sorted(
            puts, key=lambda c: (c.get("expiry") or "", c.get("strike") or 0)
        ),
        "chain_metrics": {
            "atm_strike": atm_strike,
            "atm_iv": atm_iv,
            "num_calls": len(calls),
            "num_puts": len(puts),
            "expiry_range_days": f"{expiry_min_days}-{expiry_max_days}",
        },
        "note": f"Data source: {data_source}"
        + (
            " — use live broker data for execution"
            if data_source == "synthetic"
            else ""
        ),
    }


@mcp.tool()
async def get_options_chain(
    symbol: str,
    expiry_min_days: int = 7,
    expiry_max_days: int = 45,
    option_type: str | None = None,
) -> dict[str, Any]:
    """
    Fetch live options chain from broker data (Alpaca → Polygon fallback).

    Returns real bid/ask, IV, and Greeks for each contract. Use this tool
    for /options execution decisions. For strategy design and backtesting,
    use compute_option_chain (synthetic).

    Falls back to synthetic compute_option_chain if broker data is unavailable
    (no API key, market closed, no options subscription). The 'source' field
    in the response indicates where the data came from.

    Args:
        symbol: Underlying equity symbol (e.g., "SPY").
        expiry_min_days: Minimum DTE to include (default 7 — avoids gamma pins).
        expiry_max_days: Maximum DTE to include (default 45 — standard range).
        option_type: Filter to "call" or "put" only. None returns both.

    Returns:
        Dict with:
            - symbol, underlying_price, as_of
            - contracts: list of contract dicts (strike, expiry, dte, bid, ask,
                         mid, iv, delta, gamma, theta, vega, open_interest, volume)
            - calls: contracts filtered to calls
            - puts: contracts filtered to puts
            - source: "alpaca" | "polygon" | "synthetic"
            - chain_metrics: atm_strike, atm_iv, num_calls, num_puts
    """
    return await _get_options_chain_impl(
        symbol, expiry_min_days, expiry_max_days, option_type
    )


@mcp.tool()
async def get_iv_surface(
    symbol: str,
) -> dict[str, Any]:
    """
    Compute IV surface metrics from today's stored options chain data.

    Reads the options_chains table populated by get_options_chain. If today's
    data is not stored, calls get_options_chain first.

    Returns IV rank, IV percentile, skew (25-delta), and term structure —
    the inputs needed for the /options structure selection decision matrix.

    Args:
        symbol: Underlying equity symbol (e.g., "SPY").

    Returns:
        Dict with:
            - iv_rank: Current IV vs 52-week range (0–100), None until history builds
            - iv_percentile: % of days in past year with lower IV
            - atm_iv_30d: ATM IV for nearest 30-DTE expiry
            - atm_iv_60d: ATM IV for nearest 60-DTE expiry
            - skew_25d: 25-delta put IV minus 25-delta call IV (positive = put skew)
            - term_structure: list of {expiry, dte, atm_iv} sorted by DTE
            - data_source: where chain data came from
    """
    today = str(date.today())
    today_date = date.today()

    def _load_rows():
        ds = DataStore()
        rows = ds.conn.execute(
            """
            SELECT expiry, strike, option_type, iv, delta
            FROM options_chains
            WHERE underlying = ? AND data_date = ? AND iv IS NOT NULL
            ORDER BY expiry, strike
        """,
            [symbol, today],
        ).fetchall()
        ds.close()
        return rows

    try:
        rows = _load_rows()
    except Exception:
        rows = []

    if not rows:
        chain_result = await _get_options_chain_impl(symbol=symbol)
        if "error" in chain_result:
            return chain_result
        try:
            rows = _load_rows()
        except Exception:
            rows = []

    if not rows:
        return {
            "symbol": symbol,
            "iv_rank": None,
            "iv_percentile": None,
            "atm_iv_30d": None,
            "atm_iv_60d": None,
            "skew_25d": None,
            "term_structure": [],
            "data_source": "none",
            "note": "No options chain data available for today",
        }

    expiry_contracts: dict[str, list] = defaultdict(list)
    for expiry, strike, option_type, iv, delta in rows:
        expiry_contracts[str(expiry)].append(
            {
                "strike": strike,
                "option_type": option_type,
                "iv": float(iv),
                "delta": float(delta) if delta else None,
            }
        )

    term_structure = []
    for expiry_str, contracts in sorted(expiry_contracts.items()):
        try:
            exp_date = date.fromisoformat(expiry_str)
        except (ValueError, TypeError):
            continue
        dte = (exp_date - today_date).days
        if dte < 0:
            continue

        calls = [
            c
            for c in contracts
            if c["option_type"] == "call" and c["iv"] and c["delta"]
        ]
        if calls:
            atm_call = min(calls, key=lambda c: abs((c["delta"] or 0) - 0.50))
            term_structure.append(
                {"expiry": expiry_str, "dte": dte, "atm_iv": round(atm_call["iv"], 4)}
            )

    def _nearest_iv(dte_target: int) -> float | None:
        if not term_structure:
            return None
        closest = min(term_structure, key=lambda x: abs(x["dte"] - dte_target))
        return closest["atm_iv"] if abs(closest["dte"] - dte_target) <= 15 else None

    atm_iv_30d = _nearest_iv(30)
    atm_iv_60d = _nearest_iv(60)

    # 25-delta skew for front month
    skew_25d = None
    if term_structure:
        front_expiry = term_structure[0]["expiry"]
        front_contracts = expiry_contracts.get(front_expiry, [])
        put_25d = next(
            (
                c["iv"]
                for c in front_contracts
                if c["option_type"] == "put"
                and c["delta"]
                and abs(abs(c["delta"]) - 0.25) < 0.05
            ),
            None,
        )
        call_25d = next(
            (
                c["iv"]
                for c in front_contracts
                if c["option_type"] == "call"
                and c["delta"]
                and abs(c["delta"] - 0.25) < 0.05
            ),
            None,
        )
        if put_25d and call_25d:
            skew_25d = round(put_25d - call_25d, 4)

    return {
        "symbol": symbol,
        "iv_rank": None,  # Requires 52-week daily chain history to compute
        "iv_percentile": None,
        "atm_iv_30d": atm_iv_30d,
        "atm_iv_60d": atm_iv_60d,
        "skew_25d": skew_25d,
        "term_structure": term_structure,
        "data_source": "options_chains_table",
        "note": "iv_rank/iv_percentile require 52-week of daily chain data — will populate over time",
    }
