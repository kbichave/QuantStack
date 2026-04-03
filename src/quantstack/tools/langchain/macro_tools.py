"""Macro and market breadth tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def get_credit_market_signals() -> str:
    """Assess credit market stress as a macro context gate for bottom entries.

    Uses ETF price proxies to measure credit spread dynamics without needing
    direct bond data. All data loaded from local OHLCV cache.

    Signals computed:
    - HY/IG ratio (HYG/LQD): rising = credit spreads widening = risk-off
    - Yield curve slope proxy (TLT/SHY ratio): flattening/inverting = stress
    - Dollar direction (UUP): rising = risk-off, flight to safety
    - Gold vs TLT divergence: gold rising + bonds falling = inflation fear
    - Credit regime: "widening" | "stable" | "contracting"

    BOTTOM ENTRY RULE:
      If credit_regime == "widening": DO NOT enter bottom strategies.
      If credit_regime == "stable" or "contracting": proceed to tier_3 check.

    Returns JSON with credit_regime, hy_spread_zscore, yield_curve_slope,
    dollar_direction, risk_on_score, bottom_signal, and ETF details.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_market_breadth() -> str:
    """Compute market breadth using sector ETF proxies.

    Measures the width of market participation -- how many sectors/indexes
    are above their key moving averages. Breadth cascade (broad deterioration)
    precedes most major selloffs. Breadth stabilization (divergence from price)
    often precedes bottoms.

    Uses 15 ETFs as proxy: 11 sector ETFs + SPY + QQQ + IWM + MDY.

    Signals:
    - breadth_score: % of ETFs above 50d SMA (0-1)
    - breadth_trend: 5-day change in score (rising/falling/stable)
    - breadth_divergence: SPY making new low but breadth score stabilizing
    - sectors_above_all_mas: count above 20d+50d+200d simultaneously
    - weakest_sectors / strongest_sectors by relative performance

    Returns JSON with breadth_score, breadth_trend, breadth_divergence,
    sector breakdown, and bottom_signal assessment.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
