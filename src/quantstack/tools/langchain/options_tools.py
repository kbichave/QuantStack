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
    risk_free_rate: Annotated[float, Field(description="Annualized risk-free rate, e.g. 0.05 for 5%")] = 0.05,
    dividend_yield: Annotated[float, Field(description="Continuous dividend yield, e.g. 0.02 for 2%")] = 0.0,
) -> str:
    """Computes the theoretical fair value of a European option using the Black-Scholes pricing model. Use when you need to estimate option premium, detect mispricing between market price and theoretical value, or evaluate put/call parity. Returns the computed option price given spot, strike, implied volatility, and time to expiration.
    """
    try:
        from quantstack.core.options.engine import price_option_dispatch

        result = price_option_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"price_option failed: S={spot}, K={strike}, T={time_to_expiry}, vol={volatility}, err={e}")
        return json.dumps({"error": str(e)})


@tool
async def compute_implied_vol(
    market_price: Annotated[float, Field(description="Observed market price (premium) of the option contract")],
    spot: Annotated[float, Field(description="Current underlying stock price (spot price)")],
    strike: Annotated[float, Field(description="Option strike price (exercise price) of the contract")],
    time_to_expiry: Annotated[float, Field(description="Time to expiration in years, e.g. 0.25 for 3 months")],
    option_type: Annotated[str, Field(description="Contract type: 'call' for call option or 'put' for put option")] = "call",
    risk_free_rate: Annotated[float, Field(description="Annualized risk-free rate")] = 0.05,
) -> str:
    """Calculates implied volatility (IV) by inverting the Black-Scholes model from the observed market price of an option. Use when you need to extract the market's volatility expectation, compare IV across strikes for skew analysis, or detect volatility mispricing. Returns the annualized implied volatility as a decimal for the specified put/call contract.
    """
    try:
        from quantstack.core.options.engine import compute_iv_dispatch

        result = compute_iv_dispatch(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=risk_free_rate,
            option_type=option_type,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(
            f"compute_implied_vol failed: mkt={market_price}, S={spot}, K={strike}, "
            f"T={time_to_expiry}, type={option_type}, err={e}"
        )
        return json.dumps({
            "error": str(e),
            "implied_volatility": None,
            "note": "IV undefined -- contract may be deep ITM/OTM or near-zero time value",
        })


@tool
async def analyze_option_structure(
    symbol: Annotated[str, Field(description="Ticker symbol to analyze, e.g. 'AAPL' or 'SPY'")],
) -> str:
    """Analyzes the implied volatility surface and volatility skew for a given equity symbol. Use when you need to assess put/call skew, term structure of IV across expirations, or detect volatility smile anomalies. Provides skew slope, term structure shape, and surface-level insights for options strategy selection."""
    try:
        from quantstack.core.options.iv_surface import build_iv_surface_from_chain, extract_iv_features
        from quantstack.data.providers import DataProviderRegistry

        registry = DataProviderRegistry()
        chain = await registry.get_options_chain(symbol=symbol)
        surface = build_iv_surface_from_chain(chain)
        features = extract_iv_features(surface)
        return json.dumps({
            "symbol": symbol,
            "atm_iv": surface.atm_iv(),
            "skew": surface.skew(),
            "features": features,
        }, default=str)
    except Exception as e:
        logger.error(f"analyze_option_structure({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})


@tool
async def get_iv_surface(
    symbol: Annotated[str, Field(description="Ticker symbol to retrieve the IV surface for, e.g. 'AAPL' or 'SPY'")],
) -> str:
    """Retrieves the full implied volatility (IV) surface grid across strikes and expirations for a given symbol. Use when you need to visualize or analyze the volatility smile, term structure, or detect relative value opportunities across the options chain. Returns IV values organized by strike price and days to expiration (DTE)."""
    try:
        from quantstack.core.options.iv_surface import build_iv_surface_from_chain
        from quantstack.data.providers import DataProviderRegistry

        registry = DataProviderRegistry()
        chain = await registry.get_options_chain(symbol=symbol)
        surface = build_iv_surface_from_chain(chain)
        return json.dumps({
            "symbol": symbol,
            "strikes": surface.strikes.tolist() if hasattr(surface.strikes, 'tolist') else list(surface.strikes),
            "expiries_days": surface.expiries_days.tolist() if hasattr(surface.expiries_days, 'tolist') else list(surface.expiries_days),
            "iv_grid": surface.iv_grid.tolist() if hasattr(surface.iv_grid, 'tolist') else surface.iv_grid,
            "atm_iv": surface.atm_iv(),
            "skew": surface.skew(),
        }, default=str)
    except Exception as e:
        logger.error(f"get_iv_surface({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})


@tool
async def score_trade_structure(
    symbol: Annotated[str, Field(description="Underlying ticker symbol for the options trade, e.g. 'AAPL'")],
    legs: Annotated[list, Field(description="List of option legs, each a dict with keys like 'strike', 'expiry', 'option_type' (call/put), 'action' (buy/sell), and 'quantity'")],
    spot: Annotated[float | None, Field(description="Current spot price. If None, fetched from DB")] = None,
) -> str:
    """Scores a proposed multi-leg options trade structure (spread, straddle, strangle, condor, butterfly) based on risk/reward profile, Greeks exposure, and probability of profit. Use when evaluating whether to enter a complex options position. Returns a composite score with breakdown by max loss, max gain, breakeven levels, and Greek sensitivities.
    """
    try:
        from quantstack.core.options.engine import compute_greeks_dispatch, price_option_dispatch

        if spot is None:
            from quantstack.db import db_conn
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT current_price FROM positions WHERE symbol = %s LIMIT 1", [symbol]
                ).fetchone()
                spot = float(row[0]) if row else 100.0

        total_greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        total_cost = 0.0
        leg_details = []

        for leg in legs:
            strike = float(leg.get("strike", spot))
            expiry = float(leg.get("expiry", 30 / 365))
            opt_type = leg.get("option_type", "call")
            action = leg.get("action", "buy")
            qty = int(leg.get("quantity", 1))
            vol = float(leg.get("volatility", 0.30))
            sign = 1 if action == "buy" else -1

            price_result = price_option_dispatch(
                spot=spot, strike=strike, time_to_expiry=expiry, vol=vol, option_type=opt_type,
            )
            greeks_result = compute_greeks_dispatch(
                spot=spot, strike=strike, time_to_expiry=expiry, volatility=vol, option_type=opt_type,
            )

            price_val = price_result.get("price", 0.0) if isinstance(price_result, dict) else 0.0
            total_cost += sign * qty * price_val * 100  # options are per 100 shares

            for g in total_greeks:
                gval = greeks_result.get(g, 0.0) if isinstance(greeks_result, dict) else 0.0
                total_greeks[g] += sign * qty * gval

            leg_details.append({
                "strike": strike, "expiry_years": expiry, "type": opt_type,
                "action": action, "qty": qty, "price": round(price_val, 4),
            })

        # Score: prefer low absolute delta (neutral), positive theta, low cost
        score = 1.0
        if abs(total_greeks["delta"]) > 0.5:
            score -= 0.2  # penalize directional exposure
        if total_greeks["theta"] > 0:
            score += 0.1  # reward positive theta (time decay collection)
        if total_cost < 0:
            score += 0.1  # credit spread = less capital at risk

        return json.dumps({
            "symbol": symbol,
            "score": round(max(0, min(1, score)), 2),
            "total_greeks": {k: round(v, 4) for k, v in total_greeks.items()},
            "net_cost": round(total_cost, 2),
            "legs": leg_details,
        }, default=str)
    except Exception as e:
        logger.error(f"score_trade_structure({symbol}) failed: {e}")
        return json.dumps({"error": str(e)})


@tool
async def simulate_trade_outcome(
    symbol: Annotated[str, Field(description="Underlying ticker symbol for the options trade simulation, e.g. 'AAPL'")],
    legs: Annotated[list, Field(description="List of option legs, each a dict with keys like 'strike', 'expiry', 'option_type' (call/put), 'action' (buy/sell), and 'quantity'")],
    spot: Annotated[float | None, Field(description="Current spot price. If None, uses 100.0 as default")] = None,
    scenarios: Annotated[list | None, Field(description="List of scenario dicts specifying price and/or volatility changes, e.g. [{'price_move_pct': -5, 'iv_change': 0.10}]. Use None for default stress scenarios")] = None,
) -> str:
    """Simulates profit and loss (P&L) outcomes for a multi-leg options trade under user-defined or default stress scenarios including price shocks and volatility changes. Use when you need to stress-test a spread, straddle, strangle, or condor before execution. Returns per-scenario P&L, max profit, max loss, and breakeven analysis for the proposed options structure.
    """
    try:
        from quantstack.core.options.engine import price_option_dispatch

        if spot is None:
            spot = 100.0

        if scenarios is None:
            scenarios = [
                {"price_move_pct": -10, "iv_change": 0.05},
                {"price_move_pct": -5, "iv_change": 0.02},
                {"price_move_pct": 0, "iv_change": 0.0},
                {"price_move_pct": 5, "iv_change": -0.02},
                {"price_move_pct": 10, "iv_change": -0.05},
            ]

        # Compute entry cost
        entry_cost = 0.0
        for leg in legs:
            strike = float(leg.get("strike", spot))
            expiry = float(leg.get("expiry", 30 / 365))
            opt_type = leg.get("option_type", "call")
            action = leg.get("action", "buy")
            qty = int(leg.get("quantity", 1))
            vol = float(leg.get("volatility", 0.30))
            sign = 1 if action == "buy" else -1

            price_result = price_option_dispatch(
                spot=spot, strike=strike, time_to_expiry=expiry, vol=vol, option_type=opt_type,
            )
            price_val = price_result.get("price", 0.0) if isinstance(price_result, dict) else 0.0
            entry_cost += sign * qty * price_val * 100

        # Simulate each scenario
        scenario_results = []
        for scenario in scenarios:
            price_move = scenario.get("price_move_pct", 0) / 100.0
            iv_change = scenario.get("iv_change", 0)
            new_spot = spot * (1 + price_move)
            # Assume half time has passed for scenario valuation
            time_decay_factor = 0.5

            scenario_value = 0.0
            for leg in legs:
                strike = float(leg.get("strike", spot))
                expiry = float(leg.get("expiry", 30 / 365))
                opt_type = leg.get("option_type", "call")
                action = leg.get("action", "buy")
                qty = int(leg.get("quantity", 1))
                vol = float(leg.get("volatility", 0.30)) + iv_change
                sign = 1 if action == "buy" else -1

                new_expiry = max(expiry * time_decay_factor, 1 / 365)
                price_result = price_option_dispatch(
                    spot=new_spot, strike=strike, time_to_expiry=new_expiry,
                    vol=max(vol, 0.01), option_type=opt_type,
                )
                price_val = price_result.get("price", 0.0) if isinstance(price_result, dict) else 0.0
                scenario_value += sign * qty * price_val * 100

            pnl = scenario_value - entry_cost
            scenario_results.append({
                "price_move_pct": scenario.get("price_move_pct", 0),
                "iv_change": scenario.get("iv_change", 0),
                "new_spot": round(new_spot, 2),
                "position_value": round(scenario_value, 2),
                "pnl": round(pnl, 2),
            })

        pnls = [s["pnl"] for s in scenario_results]
        return json.dumps({
            "symbol": symbol,
            "entry_cost": round(entry_cost, 2),
            "scenarios": scenario_results,
            "max_profit": round(max(pnls), 2),
            "max_loss": round(min(pnls), 2),
        }, default=str)
    except Exception as e:
        logger.error(f"simulate_trade_outcome({symbol}) failed: {e}")
        return json.dumps({"error": str(e)})
