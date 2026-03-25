# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Capitulation detection MCP tool.

Computes a composite institutional-grade capitulation score using signals
that are NOT retail indicators (not RSI, not MACD, not Bollinger Bands).

Signals used (all tier_2_smart_money or above):
  - Volume exhaustion  : down-day volume trending below 20d average (sellers exhausted)
  - Support integrity  : number of times price tested 52-week low zone without breaking
  - Williams VIX Fix   : synthetic VIX extreme (WVF > upper BB = fear washout)
  - PercentR Exhaustion: both short and long lookbacks simultaneously at bottom
  - Consecutive down   : normalized run of down closes vs historical distribution

Composite: vol_exhaustion(0.25) + support_integrity(0.25) + wvf(0.20) + pct_r(0.20) + consec_down(0.10)
"""

from typing import Any

import numpy as np
import pandas as pd

from quantstack.config.timeframes import Timeframe
from quantstack.core.features.momentum import PercentRExhaustion
from quantstack.core.features.volatility import WilliamsVIXFix
from quantstack.mcp._helpers import _get_reader
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


@domain(Domain.INTEL)
@mcp.tool()
async def get_capitulation_score(
    symbol: str,
    lookback_days: int = 20,
) -> dict[str, Any]:
    """
    Compute institutional-grade capitulation score for a symbol.

    Uses ONLY tier_2_smart_money and tier_3_institutional signals — no retail
    indicators (RSI/MACD/BB). Designed for buy-the-bottom strategies.

    Components:
    - Volume exhaustion (0.25): down-day volume declining = sellers running out
    - Support integrity (0.25): 52-week low zone tested 3+ times without breaking
    - Williams VIX Fix extreme (0.20): synthetic fear gauge at Bollinger extreme
    - PercentR dual exhaustion (0.20): both short+long lookback simultaneously at bottom
    - Consecutive down bars (0.10): normalized run-length vs historical distribution

    Score > 0.65: high-conviction capitulation — combine with get_institutional_accumulation
    Score 0.40-0.65: partial washout — watch, not yet actionable
    Score < 0.40: no capitulation signal

    IMPORTANT: Score alone is not an entry signal. Must also check:
      - get_institutional_accumulation(symbol) > 0.55 (smart money accumulating)
      - get_credit_market_signals() credit_regime != "widening" (macro not deteriorating)
      - piotroski_f_score >= 6 (not a fundamentally broken business)

    Args:
        symbol: Stock ticker (e.g., "RDDT", "SPY")
        lookback_days: Window for volume exhaustion and consecutive-down normalization

    Returns:
        Dict with capitulation_score (0-1), component scores, support level,
        support_test_count, and recommendation ("high_conviction"|"watch"|"not_ready")

    SIGNAL TIER: tier_3_institutional
    WORKFLOW: get_signal_brief → [if bearish + high drawdown] → get_capitulation_score → get_institutional_accumulation → debate
    RELATED: get_institutional_accumulation, get_credit_market_signals, get_market_breadth
    """
    store = _get_reader()
    try:
        df = store.load_ohlcv(symbol, Timeframe.D1)
    except Exception as exc:
        return {"error": f"Could not load data for {symbol}: {exc}", "symbol": symbol}
    finally:
        store.close()

    if df is None or len(df) < 60:
        return {
            "error": f"Insufficient data for {symbol} (need 60+ bars, got {len(df) if df is not None else 0})",
            "symbol": symbol,
            "capitulation_score": 0.0,
        }

    hi = df["high"]
    lo = df["low"]
    cl = df["close"]
    vol = df["volume"]

    result: dict[str, Any] = {"symbol": symbol}

    # ------------------------------------------------------------------
    # 1. Volume exhaustion — down-day volume trending below 20d average
    # ------------------------------------------------------------------
    vol_exhaustion_score = 0.0
    try:
        vol_ma20 = vol.rolling(20).mean()
        # Identify down days (close < open or close < prior close)
        down_day = cl < cl.shift(1)
        down_vol = vol.where(down_day, other=np.nan)
        down_vol_ma5 = down_vol.rolling(5, min_periods=1).mean()
        # Exhaustion if recent down-day volume is below 20d avg
        recent_down_vol = down_vol_ma5.iloc[-1]
        avg_vol = vol_ma20.iloc[-1]
        if pd.notna(recent_down_vol) and pd.notna(avg_vol) and avg_vol > 0:
            ratio = recent_down_vol / avg_vol
            # ratio < 0.7 = strong exhaustion, < 0.9 = moderate
            if ratio < 0.7:
                vol_exhaustion_score = 1.0
            elif ratio < 0.9:
                vol_exhaustion_score = 0.6
            elif ratio < 1.1:
                vol_exhaustion_score = 0.3
            result["volume_exhaustion_ratio"] = round(float(ratio), 3)
            result["volume_exhaustion_trend"] = "declining" if ratio < 0.9 else "elevated"
    except Exception:
        pass
    result["volume_exhaustion_score"] = round(vol_exhaustion_score, 3)

    # ------------------------------------------------------------------
    # 2. Support integrity — count tests of 52-week low zone (±3%)
    # ------------------------------------------------------------------
    support_integrity_score = 0.0
    support_level = None
    support_test_count = 0
    try:
        window = min(252, len(df))
        year_low = lo.iloc[-window:].min()
        support_zone_upper = year_low * 1.03  # ±3% zone
        support_level = float(year_low)

        # Count bars where low touched the support zone without closing below it
        in_zone = (lo.iloc[-window:] <= support_zone_upper) & (cl.iloc[-window:] >= year_low * 0.97)
        # Group consecutive touches into "tests" (separate by > 5 bars gap)
        tests = 0
        last_test_idx = -10
        for i, touched in enumerate(in_zone):
            if touched and (i - last_test_idx) > 5:
                tests += 1
                last_test_idx = i
        support_test_count = tests

        # Score based on number of tests (3+ = zone well-defended)
        if tests >= 5:
            support_integrity_score = 1.0
        elif tests >= 3:
            support_integrity_score = 0.75
        elif tests >= 2:
            support_integrity_score = 0.5
        elif tests >= 1:
            support_integrity_score = 0.25
    except Exception:
        pass
    result["support_integrity_score"] = round(support_integrity_score, 3)
    result["support_level"] = round(support_level, 2) if support_level else None
    result["support_test_count"] = support_test_count
    result["current_price"] = round(float(cl.iloc[-1]), 2)
    if support_level:
        result["pct_above_support"] = round(
            (float(cl.iloc[-1]) - support_level) / support_level * 100, 1
        )

    # ------------------------------------------------------------------
    # 3. Williams VIX Fix extreme
    # ------------------------------------------------------------------
    wvf_score = 0.0
    try:
        wvf_df = WilliamsVIXFix(lookback=22, bb_period=20, bb_dev=2.0).compute(hi, lo, cl)
        wvf_val = float(wvf_df["wvf"].iloc[-1])
        wvf_extreme = bool(wvf_df["wvf_extreme"].iloc[-1])
        result["wvf"] = round(wvf_val, 4)
        result["wvf_extreme"] = wvf_extreme
        # Percentile rank of WVF over last 252 bars
        wvf_series = wvf_df["wvf"].dropna()
        if len(wvf_series) > 20:
            pct_rank = float((wvf_series < wvf_val).sum()) / len(wvf_series)
            result["wvf_percentile_rank"] = round(pct_rank, 3)
            if wvf_extreme:
                wvf_score = 1.0
            elif pct_rank > 0.90:
                wvf_score = 0.75
            elif pct_rank > 0.75:
                wvf_score = 0.5
            elif pct_rank > 0.60:
                wvf_score = 0.25
    except Exception:
        pass
    result["wvf_score"] = round(wvf_score, 3)

    # ------------------------------------------------------------------
    # 4. PercentR dual exhaustion
    # ------------------------------------------------------------------
    pct_r_score = 0.0
    try:
        pct_r_df = PercentRExhaustion(short=14, long=112).compute(hi, lo, cl)
        exhaustion_bottom = bool(pct_r_df["exhaustion_bottom"].iloc[-1])
        pct_r_short = float(pct_r_df["pct_r_short"].iloc[-1])
        pct_r_long = float(pct_r_df["pct_r_long"].iloc[-1])
        result["exhaustion_bottom"] = exhaustion_bottom
        result["pct_r_short"] = round(pct_r_short, 2)
        result["pct_r_long"] = round(pct_r_long, 2)
        if exhaustion_bottom:
            pct_r_score = 1.0
        # Both at bottom half even if not extreme
        elif pct_r_short < -70 and pct_r_long < -70:
            pct_r_score = 0.6
        elif pct_r_short < -80:
            pct_r_score = 0.3
    except Exception:
        pass
    result["exhaustion_score"] = round(pct_r_score, 3)

    # ------------------------------------------------------------------
    # 5. Consecutive down bars (normalized)
    # ------------------------------------------------------------------
    consec_down_score = 0.0
    try:
        # Count current consecutive down bars
        consec = 0
        for i in range(len(cl) - 1, max(len(cl) - 30, 0), -1):
            if cl.iloc[i] < cl.iloc[i - 1]:
                consec += 1
            else:
                break

        # Compute historical distribution of consecutive-down runs
        runs = []
        current_run = 0
        for i in range(1, len(cl)):
            if cl.iloc[i] < cl.iloc[i - 1]:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if runs:
            p75 = float(np.percentile(runs, 75))
            p90 = float(np.percentile(runs, 90))
            if consec >= p90:
                consec_down_score = 1.0
            elif consec >= p75:
                consec_down_score = 0.5
            elif consec >= 3:
                consec_down_score = 0.25
        result["consecutive_down_bars"] = consec
    except Exception:
        pass
    result["consecutive_down_score"] = round(consec_down_score, 3)

    # ------------------------------------------------------------------
    # Composite capitulation score
    # ------------------------------------------------------------------
    composite = (
        vol_exhaustion_score * 0.25
        + support_integrity_score * 0.25
        + wvf_score * 0.20
        + pct_r_score * 0.20
        + consec_down_score * 0.10
    )
    result["capitulation_score"] = round(composite, 3)

    if composite >= 0.65:
        result["recommendation"] = "high_conviction"
        result["interpretation"] = (
            "Strong capitulation signal. Combine with get_institutional_accumulation > 0.55 "
            "and credit_regime != 'widening' before entry."
        )
    elif composite >= 0.40:
        result["recommendation"] = "watch"
        result["interpretation"] = (
            "Partial washout. Monitor for further development. "
            f"Weakest component: {_weakest_component(result)}"
        )
    else:
        result["recommendation"] = "not_ready"
        result["interpretation"] = "No capitulation signal. Selling pressure may continue."

    result["signal_tier"] = "tier_3_institutional"
    result["component_weights"] = {
        "volume_exhaustion": 0.25,
        "support_integrity": 0.25,
        "wvf_extreme": 0.20,
        "pct_r_exhaustion": 0.20,
        "consecutive_down": 0.10,
    }

    return result


def _weakest_component(result: dict) -> str:
    components = {
        "volume_exhaustion": result.get("volume_exhaustion_score", 0),
        "support_integrity": result.get("support_integrity_score", 0),
        "wvf": result.get("wvf_score", 0),
        "pct_r": result.get("exhaustion_score", 0),
        "consec_down": result.get("consecutive_down_score", 0),
    }
    return min(components, key=components.get)
