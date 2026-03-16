# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP options tools — pricing, Greeks, IV, and structure analysis.

Extracted from ``quantcore.mcp.server`` to keep tool modules focused.
All helpers come from ``quantcore.mcp._helpers``; the ``mcp`` singleton
is imported from ``quantcore.mcp.server``.
"""

from typing import Any

import numpy as np

from quantcore.mcp._helpers import (
    _get_reader,
    _parse_timeframe,
)
from quantcore.mcp.server import mcp


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
    from quantcore.options.engine import price_option_dispatch

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
    from quantcore.options.engine import compute_greeks_dispatch

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
    from quantcore.options.engine import compute_iv_dispatch

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
    from quantcore.options.adapters.quantsbin_adapter import analyze_structure_quantsbin

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
    import pandas as pd

    from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

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
    from quantcore.options.adapters.financepy_adapter import (
        price_american_option as price_american,
    )

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
    from quantcore.options.engine import (
        price_option_dispatch,
    )

    store = _get_reader()

    try:
        # Get underlying price
        df = store.load_ohlcv(symbol, _parse_timeframe("daily"))

        if df.empty:
            return {"error": f"No price data for {symbol}"}

        underlying_price = float(df["close"].iloc[-1])

        # Generate synthetic chain strikes (in production, this would come from broker MCP)
        if underlying_price > 100:
            strike_increment = 5.0
        elif underlying_price > 50:
            strike_increment = 2.5
        else:
            strike_increment = 1.0

        # Calculate strike range based on delta bounds
        # Use rough approximation: 10-90 delta covers ~1 std dev
        vol_estimate = 0.25  # Default IV estimate
        atm_strike = round(underlying_price / strike_increment) * strike_increment

        # Generate strikes from -30% to +30%
        strikes = [
            atm_strike + i * strike_increment
            for i in range(-6, 7)
            if (atm_strike + i * strike_increment) > 0
        ]

        # Default expiry: 30 DTE
        dte = 30
        tte = dte / 365.0
        rate = 0.05
        div_yield = 0.01

        calls = []
        puts = []

        for strike in strikes:
            # Calculate call metrics
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

            # Calculate put metrics
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
    from quantcore.options.engine import price_option_dispatch

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
    from quantcore.options.adapters.quantsbin_adapter import analyze_structure_quantsbin

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
            return {"error": "Invalid trade template - missing legs or underlying_price"}

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
                new_leg["expiry_days"] = max(1, leg.get("expiry_days", 30) - holding_days)
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
            "probability_profit": round(float(np.sum(pnl_array > 0) / len(pnl_array) * 100), 1),
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
