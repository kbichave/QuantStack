# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Macro and breadth signal MCP tools.

Two tier_4_regime_macro tools:
  - get_credit_market_signals(): HY/IG spread proxy via ETFs, yield curve, dollar/gold
  - get_market_breadth(): sector ETF breadth cascade — % above 20/50/200d SMA

Both serve as CONTEXT GATES before bottom entries. If credit spreads are widening
or breadth is in free-fall (not stabilizing), no institutional bottom strategy
should enter regardless of tier_3 signals.

Data: uses locally cached OHLCV from DataStore (no external API calls).
ETF proxies avoid the need for S&P 500 constituent-level data.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.universe import CREDIT_ETFS, INITIAL_LIQUID_UNIVERSE, SECTOR_ETFS
from quantstack.mcp._helpers import _get_reader
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain

# ---------------------------------------------------------------------------
# Credit market ETF proxies — derived from universe so names stay in sync.
# ---------------------------------------------------------------------------
_CREDIT_ETFS = {sym: INITIAL_LIQUID_UNIVERSE[sym].name for sym in CREDIT_ETFS}

# ---------------------------------------------------------------------------
# Sector ETF breadth proxies
# ---------------------------------------------------------------------------
_SECTOR_ETFS = list(SECTOR_ETFS)


@domain(Domain.INTEL)
@tool_def()
async def get_credit_market_signals() -> dict[str, Any]:
    """
    Assess credit market stress as a macro context gate for bottom entries.

    Uses ETF price proxies to measure credit spread dynamics without needing
    direct bond data. All data loaded from local OHLCV cache (no external calls).

    Signals computed:
    - HY/IG ratio (HYG/LQD): rising = credit spreads widening = risk-off
    - Yield curve slope proxy (TLT/SHY ratio): flattening/inverting = stress
    - Dollar direction (UUP): rising = risk-off, flight to safety
    - Gold vs TLT divergence: gold rising + bonds falling = inflation fear (not safe haven)
    - Credit regime: "widening" | "stable" | "contracting"

    BOTTOM ENTRY RULE:
      If credit_regime == "widening": DO NOT enter bottom strategies.
      If credit_regime == "stable" or "contracting": proceed to tier_3 check.
      Historical bottoms (2009, 2020, 2022) all formed when spreads began contracting
      BEFORE equity prices recovered.

    Returns:
        Dict with credit_regime, hy_spread_zscore, yield_curve_slope,
        dollar_direction, risk_on_score, bottom_signal (bool), and ETF details
    """
    store = _get_reader()
    etf_data: dict[str, pd.Series] = {}

    try:
        for ticker in _CREDIT_ETFS:
            try:
                df = store.load_ohlcv(ticker, Timeframe.D1)
                if df is not None and len(df) >= 30:
                    etf_data[ticker] = df["close"]
            except Exception as exc:
                logger.debug(f"[macro_signals] load_ohlcv({ticker}) failed: {exc}")
    finally:
        store.close()

    result: dict[str, Any] = {
        "etfs_loaded": list(etf_data.keys()),
        "signal_tier": "tier_4_regime_macro",
    }

    if len(etf_data) < 3:
        result["credit_regime"] = "unknown"
        result["bottom_signal"] = False
        result["error"] = (
            f"Insufficient ETF data (loaded {len(etf_data)}/{len(_CREDIT_ETFS)}). "
            "Run acquire_historical_data for HYG, LQD, TLT, IEF, SHY, GLD, UUP first."
        )
        return result

    # ------------------------------------------------------------------
    # 1. HY/IG spread proxy — HYG/LQD ratio z-score
    # ------------------------------------------------------------------
    hy_spread_zscore = None
    hy_spread_direction = "unknown"
    try:
        if "HYG" in etf_data and "LQD" in etf_data:
            hyg = etf_data["HYG"]
            lqd = etf_data["LQD"]
            # Align on common dates
            ratio = hyg / lqd
            ratio = ratio.dropna()
            if len(ratio) >= 30:
                mean = ratio.rolling(60, min_periods=20).mean()
                std = ratio.rolling(60, min_periods=20).std()
                zscore = ((ratio - mean) / std).iloc[-1]
                hy_spread_zscore = round(float(zscore), 2)
                # Positive z-score = ratio rising = HY underperforming IG = spreads widening
                chg_5d = ratio.iloc[-1] / ratio.iloc[-6] - 1 if len(ratio) >= 6 else 0
                chg_20d = ratio.iloc[-1] / ratio.iloc[-21] - 1 if len(ratio) >= 21 else 0
                if chg_5d > 0.005 or chg_20d > 0.02:
                    hy_spread_direction = "widening"
                elif chg_5d < -0.005 or chg_20d < -0.02:
                    hy_spread_direction = "contracting"
                else:
                    hy_spread_direction = "stable"
                result["hy_spread_zscore"] = hy_spread_zscore
                result["hy_spread_direction"] = hy_spread_direction
    except Exception as exc:
        logger.debug(f"[macro_signals] HY/IG spread computation failed: {exc}")

    # ------------------------------------------------------------------
    # 2. Yield curve slope proxy — TLT/SHY ratio
    # ------------------------------------------------------------------
    yield_curve_slope = "unknown"
    try:
        if "TLT" in etf_data and "SHY" in etf_data:
            tlt = etf_data["TLT"]
            shy = etf_data["SHY"]
            ratio = (tlt / shy).dropna()
            if len(ratio) >= 20:
                chg_20d = ratio.iloc[-1] / ratio.iloc[-21] - 1
                if chg_20d > 0.02:
                    yield_curve_slope = "steepening"  # long-end rallying = risk-off
                elif chg_20d < -0.02:
                    yield_curve_slope = "flattening"  # long-end selling = inflation or risk-on
                else:
                    yield_curve_slope = "neutral"
                result["yield_curve_slope"] = yield_curve_slope
                result["tlt_shy_ratio_chg_20d"] = round(float(chg_20d * 100), 2)
    except Exception as exc:
        logger.debug(f"[macro_signals] yield curve slope computation failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Dollar direction (UUP)
    # ------------------------------------------------------------------
    dollar_direction = "unknown"
    try:
        if "UUP" in etf_data:
            uup = etf_data["UUP"]
            if len(uup) >= 21:
                chg_20d = float(uup.iloc[-1] / uup.iloc[-21] - 1)
                if chg_20d > 0.01:
                    dollar_direction = "strengthening"  # risk-off
                elif chg_20d < -0.01:
                    dollar_direction = "weakening"  # risk-on
                else:
                    dollar_direction = "neutral"
                result["dollar_direction"] = dollar_direction
                result["dollar_chg_20d_pct"] = round(chg_20d * 100, 2)
    except Exception as exc:
        logger.debug(f"[macro_signals] dollar direction computation failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Gold vs TLT — flight to safety vs inflation fear
    # ------------------------------------------------------------------
    flight_to_quality = False
    try:
        if "GLD" in etf_data and "TLT" in etf_data:
            gld = etf_data["GLD"]
            tlt = etf_data["TLT"]
            if len(gld) >= 21 and len(tlt) >= 21:
                gld_chg = float(gld.iloc[-1] / gld.iloc[-21] - 1)
                tlt_chg = float(tlt.iloc[-1] / tlt.iloc[-21] - 1)
                # Both rising = true flight to quality (bottom precursor when combined with equity selling)
                flight_to_quality = gld_chg > 0.01 and tlt_chg > 0.01
                result["gold_20d_chg_pct"] = round(gld_chg * 100, 2)
                result["tlt_20d_chg_pct"] = round(tlt_chg * 100, 2)
                result["flight_to_quality_active"] = flight_to_quality
    except Exception as exc:
        logger.debug(f"[macro_signals] gold/TLT flight-to-quality computation failed: {exc}")

    # ------------------------------------------------------------------
    # Composite credit regime classification
    # ------------------------------------------------------------------
    # Primary driver: HY spread direction
    # Secondary: dollar strength
    regime_signals = []
    if hy_spread_direction == "widening":
        regime_signals.append("widening")
    elif hy_spread_direction == "contracting":
        regime_signals.append("contracting")

    if dollar_direction == "strengthening":
        regime_signals.append("widening")
    elif dollar_direction == "weakening":
        regime_signals.append("contracting")

    widening_count = regime_signals.count("widening")
    contracting_count = regime_signals.count("contracting")

    if widening_count > contracting_count:
        credit_regime = "widening"
    elif contracting_count > widening_count:
        credit_regime = "contracting"
    elif len(regime_signals) > 0:
        credit_regime = "stable"
    else:
        credit_regime = "unknown"

    result["credit_regime"] = credit_regime

    # Risk-on score (0-1, higher = more risk-on conditions)
    risk_on_components = []
    if hy_spread_direction == "contracting":
        risk_on_components.append(0.8)
    elif hy_spread_direction == "stable":
        risk_on_components.append(0.5)
    elif hy_spread_direction == "widening":
        risk_on_components.append(0.2)
    if dollar_direction == "weakening":
        risk_on_components.append(0.7)
    elif dollar_direction == "neutral":
        risk_on_components.append(0.5)
    elif dollar_direction == "strengthening":
        risk_on_components.append(0.3)

    risk_on_score = float(np.mean(risk_on_components)) if risk_on_components else 0.5
    result["risk_on_score"] = round(risk_on_score, 3)

    # Bottom signal requires: credit NOT widening AND at least one contracting signal
    bottom_signal = (credit_regime in ("stable", "contracting")) and risk_on_score >= 0.45
    result["bottom_signal"] = bottom_signal

    if credit_regime == "widening":
        result["interpretation"] = (
            "Credit spreads widening — macro is deteriorating. "
            "DO NOT enter bottom strategies regardless of technical signals."
        )
    elif credit_regime == "contracting":
        result["interpretation"] = (
            "Credit spreads contracting — macro support improving. "
            "Proceed to tier_3 signal check (capitulation + accumulation)."
        )
    elif credit_regime == "stable":
        result["interpretation"] = (
            "Credit spreads stable — macro neutral. "
            "Bottom entries viable if tier_3 signals confirm."
        )
    else:
        result["interpretation"] = (
            "Insufficient ETF data to classify credit regime. "
            "Treat as unknown — use caution."
        )

    return result


@domain(Domain.INTEL)
@tool_def()
async def get_market_breadth() -> dict[str, Any]:
    """
    Compute market breadth using sector ETF proxies.

    Measures the width of market participation — how many sectors/indexes
    are above their key moving averages. Breadth cascade (broad deterioration)
    precedes most major selloffs. Breadth stabilization (divergence from price)
    often precedes bottoms.

    Uses 15 ETFs as proxy: 11 sector ETFs + SPY + QQQ + IWM + MDY.
    No S&P 500 constituent data needed.

    Signals:
    - breadth_score: % of ETFs above 50d SMA (0-1)
    - breadth_trend: 5-day change in score (rising/falling/stable)
    - breadth_divergence: SPY making new low but breadth score stabilizing
    - sectors_above_all_mas: count above 20d+50d+200d simultaneously
    - weakest_sectors / strongest_sectors by relative performance

    BOTTOM ENTRY RULE:
      breadth_score < 0.15: Market-wide washout — extreme but not yet reversing
      breadth_divergence = True + breadth_trend = "rising": Hidden accumulation
      breadth_score > 0.70 + falling: Late-cycle warning, not a bottom setup

    Returns:
        Dict with breadth_score, breadth_trend, breadth_divergence,
        sector breakdown, and bottom_signal assessment
    """
    store = _get_reader()
    etf_data: dict[str, pd.DataFrame] = {}

    try:
        for ticker in _SECTOR_ETFS:
            try:
                df = store.load_ohlcv(ticker, Timeframe.D1)
                if df is not None and len(df) >= 50:
                    etf_data[ticker] = df[["close"]]
            except Exception as exc:
                logger.debug(f"[macro_signals] load_ohlcv({ticker}) failed: {exc}")
    finally:
        store.close()

    result: dict[str, Any] = {
        "etfs_loaded": list(etf_data.keys()),
        "signal_tier": "tier_4_regime_macro",
    }

    if len(etf_data) < 5:
        result["breadth_score"] = None
        result["bottom_signal"] = False
        result["error"] = (
            f"Only {len(etf_data)} of {len(_SECTOR_ETFS)} sector ETFs available. "
            "Run acquire_historical_data for sector ETFs first."
        )
        return result

    # ------------------------------------------------------------------
    # Compute moving averages and above/below for each ETF
    # ------------------------------------------------------------------
    sector_stats = {}
    above_20d = []
    above_50d = []
    above_200d = []
    above_all = []
    perf_20d = {}

    for ticker, df in etf_data.items():
        cl = df["close"]
        last = float(cl.iloc[-1])
        try:
            sma20 = float(cl.rolling(20).mean().iloc[-1])
            sma50 = float(cl.rolling(50).mean().iloc[-1])
            sma200 = float(cl.rolling(200, min_periods=100).mean().iloc[-1]) if len(cl) >= 100 else None

            a20 = last > sma20
            a50 = last > sma50
            a200 = (last > sma200) if sma200 else None

            above_20d.append(a20)
            above_50d.append(a50)
            if a200 is not None:
                above_200d.append(a200)
            if a20 and a50 and (a200 is True):
                above_all.append(ticker)

            perf = float(cl.iloc[-1] / cl.iloc[-21] - 1) if len(cl) >= 21 else 0.0
            perf_20d[ticker] = round(perf * 100, 1)

            sector_stats[ticker] = {
                "above_20d": a20,
                "above_50d": a50,
                "above_200d": a200,
                "perf_20d_pct": round(perf * 100, 1),
            }
        except Exception as exc:
            logger.debug(f"[macro_signals] sector SMA computation for {ticker} failed: {exc}")

    n = len(above_50d)
    breadth_score = round(sum(above_50d) / n, 3) if n > 0 else None
    breadth_20d = round(sum(above_20d) / len(above_20d), 3) if above_20d else None

    result["breadth_score"] = breadth_score
    result["breadth_score_20d"] = breadth_20d
    result["sectors_above_50d_sma"] = sum(above_50d)
    result["sectors_above_200d_sma"] = sum(above_200d) if above_200d else None
    result["sectors_above_all_mas"] = len(above_all)
    result["total_etfs_measured"] = n
    result["sector_details"] = sector_stats

    # Strongest and weakest by 20-day performance
    sorted_perf = sorted(perf_20d.items(), key=lambda x: x[1])
    result["weakest_sectors"] = [t for t, _ in sorted_perf[:3]]
    result["strongest_sectors"] = [t for t, _ in sorted_perf[-3:]]

    # ------------------------------------------------------------------
    # Breadth trend (5-day change via rolling)
    # ------------------------------------------------------------------
    breadth_trend = "unknown"
    try:
        # Use SPY as reference — if SPY is in data, compute breadth change
        if "SPY" in etf_data:
            spy_cl = etf_data["SPY"]["close"]
            # Compute breadth score 5 bars ago using the same set
            above_50d_5ago = []
            for ticker, df in etf_data.items():
                cl = df["close"]
                if len(cl) >= 55:
                    sma50_5ago = float(cl.rolling(50).mean().iloc[-6])
                    above_50d_5ago.append(cl.iloc[-6] > sma50_5ago)

            if above_50d_5ago:
                score_5ago = sum(above_50d_5ago) / len(above_50d_5ago)
                delta = (breadth_score or 0) - score_5ago
                if delta > 0.05:
                    breadth_trend = "rising"
                elif delta < -0.05:
                    breadth_trend = "falling"
                else:
                    breadth_trend = "stable"
                result["breadth_delta_5d"] = round(delta, 3)
    except Exception as exc:
        logger.debug(f"[macro_signals] breadth trend computation failed: {exc}")
    result["breadth_trend"] = breadth_trend

    # ------------------------------------------------------------------
    # Breadth divergence — SPY making new lows but breadth stabilizing
    # ------------------------------------------------------------------
    breadth_divergence = False
    try:
        if "SPY" in etf_data:
            spy_cl = etf_data["SPY"]["close"]
            spy_new_low_20d = float(spy_cl.iloc[-1]) <= float(spy_cl.iloc[-21:].min()) * 1.01
            # Divergence: SPY at/near 20d low but breadth not falling
            if spy_new_low_20d and breadth_trend in ("rising", "stable"):
                breadth_divergence = True
            result["spy_at_20d_low"] = spy_new_low_20d
    except Exception as exc:
        logger.debug(f"[macro_signals] breadth divergence computation failed: {exc}")
    result["breadth_divergence"] = breadth_divergence

    # ------------------------------------------------------------------
    # Bottom signal assessment
    # ------------------------------------------------------------------
    bottom_signal = False
    if breadth_score is not None:
        # Broad washout with stabilization = best bottom setup
        if breadth_score < 0.30 and breadth_divergence:
            bottom_signal = True
            result["interpretation"] = (
                f"Breadth washout ({breadth_score:.0%} above 50d) with divergence — "
                "hidden accumulation pattern. Strong bottom setup when combined with "
                "tier_3 institutional signals."
            )
        elif breadth_score < 0.15:
            result["interpretation"] = (
                f"Extreme breadth washout ({breadth_score:.0%} above 50d). "
                "Not yet stabilizing — wait for breadth_trend to turn 'rising'."
            )
        elif breadth_score < 0.30:
            result["interpretation"] = (
                f"Broad market weakness ({breadth_score:.0%} above 50d). "
                "Watch for breadth divergence as bottom confirmation."
            )
        elif breadth_score > 0.70 and breadth_trend == "falling":
            result["interpretation"] = (
                f"Breadth deteriorating from high levels ({breadth_score:.0%} above 50d). "
                "Late-cycle distribution — not a bottom setup."
            )
        else:
            result["interpretation"] = (
                f"Breadth score {breadth_score:.0%} — moderate conditions. "
                "Not a clear bottom or top signal."
            )
    else:
        result["interpretation"] = "Insufficient data for breadth assessment."

    result["bottom_signal"] = bottom_signal

    return result


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
