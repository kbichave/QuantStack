# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Options flow signal collector for SignalEngine.

Computes dealer-positioning signals from an Alpaca options chain snapshot:
  - GEX (Gamma Exposure): net gamma × OI × spot²; negative → amplifying regime
  - Gamma flip level: strike where net GEX crosses zero (key S/R level)
  - DEX (Delta Exposure): net directional bias of open interest
  - Max Pain: expiry strike minimising total dollar OI loss
  - IV Skew: OTM put IV minus OTM call IV at ≈25-delta moneyness
  - VRP (Vol Risk Premium): ATM IV minus 30-day realised vol
  - O/S ratio: options volume / stock volume (high = informed options trading)

Failure modes:
  - No Alpaca options subscription → all fields None, collector_failures records it
  - Alpaca unavailable → same graceful degradation
  - Chain has no Greeks/IV → GEX/DEX/skew skip; max_pain still computed from OI
"""

from __future__ import annotations

import math
import os
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Black-Scholes helpers (for IV inversion fallback when API Greeks missing)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(spot: float, strike: float, dte_years: float, iv: float, r: float, is_call: bool) -> float:
    """Black-Scholes call/put price."""
    if dte_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * dte_years) / (iv * math.sqrt(dte_years))
    d2 = d1 - iv * math.sqrt(dte_years)
    if is_call:
        return spot * _norm_cdf(d1) - strike * math.exp(-r * dte_years) * _norm_cdf(d2)
    return strike * math.exp(-r * dte_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _bs_gamma(spot: float, strike: float, dte_years: float, iv: float, r: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if dte_years <= 0 or iv <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * dte_years) / (iv * math.sqrt(dte_years))
    phi_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return phi_d1 / (spot * iv * math.sqrt(dte_years))


def _bs_delta(spot: float, strike: float, dte_years: float, iv: float, r: float, is_call: bool) -> float:
    """Black-Scholes delta."""
    if dte_years <= 0 or iv <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * dte_years) / (iv * math.sqrt(dte_years))
    if is_call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


# ---------------------------------------------------------------------------
# Core signal computation
# ---------------------------------------------------------------------------


def compute_options_flow_signals(
    contracts: list[dict],
    spot: float,
    realized_vol_30d: float | None = None,
    r: float = 0.05,
) -> dict[str, Any]:
    """
    Compute dealer-positioning signals from a list of options contracts.

    Parameters
    ----------
    contracts : list[dict]
        Each dict must contain at minimum: option_type, strike, open_interest.
        Optional (improves accuracy): gamma, delta, implied_volatility, expiry.
    spot : float
        Current underlying price.
    realized_vol_30d : float | None
        30-day realised vol annualised (e.g. 0.18 = 18%). Used for VRP.
    r : float
        Risk-free rate. Default 5%.

    Returns
    -------
    dict with keys: gex, gamma_flip, above_gamma_flip, dex, max_pain,
                    iv_skew, vrp, n_contracts, call_oi, put_oi
    """
    result: dict[str, Any] = {
        "gex": None,
        "gamma_flip": None,
        "above_gamma_flip": None,
        "dex": None,
        "max_pain": None,
        "iv_skew": None,
        "vrp": None,
        "n_contracts": len(contracts),
        "call_oi": None,
        "put_oi": None,
    }

    if not contracts or spot <= 0:
        return result

    # Filter to contracts with valid strike and OI
    valid = [c for c in contracts if c.get("strike") and c.get("open_interest") is not None]
    if not valid:
        return result

    calls = [c for c in valid if c.get("option_type") == "call"]
    puts  = [c for c in valid if c.get("option_type") == "put"]

    result["call_oi"] = sum(c["open_interest"] for c in calls)
    result["put_oi"]  = sum(c["open_interest"] for c in puts)

    # -----------------------------------------------------------------------
    # GEX and DEX — requires gamma/delta per contract
    # Fall back to BS-derived values when API Greeks are absent
    # -----------------------------------------------------------------------
    net_gex_by_strike: dict[float, float] = {}
    net_dex_total = 0.0
    gex_total = 0.0

    for c in valid:
        strike = c["strike"]
        oi = c["open_interest"]
        is_call = c.get("option_type") == "call"
        iv = c.get("implied_volatility")
        gamma = c.get("gamma")
        delta = c.get("delta")
        expiry_str = c.get("expiry")

        # DTE in years
        dte_years = 30.0 / 252.0  # fallback ~30 days
        if expiry_str:
            try:
                from datetime import date as _date
                parts = expiry_str.split("-")
                exp_date = _date(int(parts[0]), int(parts[1]), int(parts[2]))
                dte_days = max((exp_date - _date.today()).days, 0)
                dte_years = dte_days / 365.0
            except Exception:
                pass

        # If API didn't provide gamma/delta, use BS with IV if available
        if gamma is None and iv is not None and dte_years > 0:
            gamma = _bs_gamma(spot, strike, dte_years, iv, r)
        if delta is None and iv is not None and dte_years > 0:
            delta = _bs_delta(spot, strike, dte_years, iv, r, is_call)

        if gamma is not None and oi is not None:
            # GEX convention: dealer is short calls → negative gamma → call GEX positive
            # dealer is short puts → positive gamma → put GEX negative
            # Net GEX per strike for dealer (gamma scalping regime signal)
            contract_multiplier = 100
            gex_contribution = gamma * oi * contract_multiplier * spot**2 * 0.01
            if is_call:
                net_gex_by_strike[strike] = net_gex_by_strike.get(strike, 0.0) + gex_contribution
            else:
                net_gex_by_strike[strike] = net_gex_by_strike.get(strike, 0.0) - gex_contribution
            gex_total += gex_contribution if is_call else -gex_contribution

        if delta is not None and oi is not None:
            sign = 1 if is_call else -1
            net_dex_total += sign * delta * oi * 100

    if net_gex_by_strike:
        result["gex"] = round(gex_total, 0)
        result["dex"] = round(net_dex_total, 0)

        # Gamma flip: strike where cumulative GEX transitions sign
        # Sort strikes and find zero crossing
        sorted_strikes = sorted(net_gex_by_strike.keys())
        cum_gex = 0.0
        flip_strike = None
        prev_cum = None
        for st in sorted_strikes:
            cum_gex += net_gex_by_strike[st]
            if prev_cum is not None and prev_cum * cum_gex < 0:
                # Linear interpolation for zero crossing
                prev_strike = sorted_strikes[sorted_strikes.index(st) - 1]
                ratio = abs(prev_cum) / (abs(prev_cum) + abs(cum_gex))
                flip_strike = round(prev_strike + ratio * (st - prev_strike), 2)
                break
            prev_cum = cum_gex

        result["gamma_flip"] = flip_strike
        result["above_gamma_flip"] = (
            int(spot > flip_strike) if flip_strike is not None else None
        )

    # -----------------------------------------------------------------------
    # Max Pain: strike minimising total dollar OI loss at expiry
    # -----------------------------------------------------------------------
    all_strikes = sorted(set(c["strike"] for c in valid))
    if all_strikes:
        min_pain = None
        min_pain_strike = None
        for test_strike in all_strikes:
            pain = 0.0
            for c in valid:
                k = c["strike"]
                oi = c["open_interest"] or 0
                if c.get("option_type") == "call":
                    pain += max(test_strike - k, 0) * oi * 100
                else:
                    pain += max(k - test_strike, 0) * oi * 100
            if min_pain is None or pain < min_pain:
                min_pain = pain
                min_pain_strike = test_strike
        result["max_pain"] = min_pain_strike

    # -----------------------------------------------------------------------
    # IV Skew: OTM put IV minus OTM call IV at ~25-delta moneyness
    # Use contracts with |delta| closest to 0.25 on each side
    # -----------------------------------------------------------------------
    put_ivs_otm = [
        c["implied_volatility"]
        for c in puts
        if c.get("implied_volatility")
        and c.get("delta") is not None
        and abs(c["delta"]) >= 0.15
        and abs(c["delta"]) <= 0.35
        and c["strike"] < spot
    ]
    call_ivs_otm = [
        c["implied_volatility"]
        for c in calls
        if c.get("implied_volatility")
        and c.get("delta") is not None
        and abs(c["delta"]) >= 0.15
        and abs(c["delta"]) <= 0.35
        and c["strike"] > spot
    ]
    if put_ivs_otm and call_ivs_otm:
        result["iv_skew"] = round(
            sum(put_ivs_otm) / len(put_ivs_otm) - sum(call_ivs_otm) / len(call_ivs_otm), 4
        )

    # -----------------------------------------------------------------------
    # VRP: ATM IV minus realised vol
    # ATM = call strike closest to spot
    # -----------------------------------------------------------------------
    if realized_vol_30d is not None:
        atm_calls = sorted(
            [c for c in calls if c.get("implied_volatility")],
            key=lambda c: abs(c["strike"] - spot),
        )
        if atm_calls:
            atm_iv = atm_calls[0]["implied_volatility"]
            result["vrp"] = round(atm_iv - realized_vol_30d, 4)

    return result


# ---------------------------------------------------------------------------
# SignalEngine integration
# ---------------------------------------------------------------------------


def collect_options_flow(symbol: str, spot: float, realized_vol_30d: float | None = None) -> dict[str, Any]:
    """
    Fetch Alpaca options chain and compute flow signals.

    Returns empty dict on any failure — SignalEngine records the failure
    in collector_failures and continues.
    """
    try:
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            logger.debug(f"[options_flow] {symbol}: ALPACA credentials not set, skipping")
            return {}

        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest

        cli = OptionHistoricalDataClient(api_key, secret_key)
        req = OptionChainRequest(underlying_symbol=symbol)
        chain = cli.get_option_chain(req)

        contracts = []
        for sym, snap in chain.items():
            try:
                option_type = "call" if sym[-9] == "C" else "put"
                strike = int(sym[-8:]) / 1000.0
            except Exception:
                continue

            greeks = snap.greeks if hasattr(snap, "greeks") and snap.greeks else None
            contracts.append(
                {
                    "symbol": sym,
                    "option_type": option_type,
                    "strike": strike,
                    "implied_volatility": float(snap.implied_volatility) if hasattr(snap, "implied_volatility") and snap.implied_volatility else None,
                    "open_interest": int(snap.open_interest) if hasattr(snap, "open_interest") and snap.open_interest else 0,
                    "delta": float(greeks.delta) if greeks and greeks.delta is not None else None,
                    "gamma": float(greeks.gamma) if greeks and greeks.gamma is not None else None,
                }
            )

        signals = compute_options_flow_signals(contracts, spot=spot, realized_vol_30d=realized_vol_30d)
        logger.debug(f"[options_flow] {symbol}: GEX={signals.get('gex')}, gamma_flip={signals.get('gamma_flip')}")
        return {f"opt_{k}": v for k, v in signals.items()}

    except Exception as exc:
        logger.debug(f"[options_flow] {symbol}: failed — {exc}")
        return {}
