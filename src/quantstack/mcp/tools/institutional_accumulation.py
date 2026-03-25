# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Institutional accumulation MCP tool.

Combines tier_3_institutional signals to answer: "Is smart money accumulating
this stock at current levels?" Uses:

  1. Insider cluster score  — CEO/CFO-weighted net buy ratio (30-day lookback)
  2. GEX signal             — positive GEX = dealers long gamma = mean-reversion support
  3. IV skew extreme        — put skew z-score > 2.0 = maximum fear = contrarian buy
  4. Institutional direction — ownership trend (accumulating/stable/distributing)

All components are optional; score degrades gracefully when data is unavailable.

Composite weights:
  insider_cluster(0.30) + gex_support(0.25) + iv_skew_extreme(0.25) + inst_direction(0.20)
"""

import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from quantstack.config.timeframes import Timeframe
from quantstack.data.fetcher import AlphaVantageClient
from quantstack.mcp._helpers import _get_reader
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain

# Insider title importance weights — C-suite signals matter more
_TITLE_WEIGHTS: dict[str, float] = {
    "ceo": 3.0,
    "chief executive": 3.0,
    "cfo": 3.0,
    "chief financial": 3.0,
    "coo": 2.5,
    "chief operating": 2.5,
    "president": 2.0,
    "svp": 1.5,
    "evp": 1.5,
    "senior vice": 1.5,
    "vice president": 1.2,
    "vp ": 1.2,
    "director": 1.0,
    "general counsel": 1.0,
    "chief": 1.5,
}

_LOOKBACK_DAYS = 90
_CLUSTER_MIN_INSIDERS = 2  # minimum distinct insiders for "cluster" designation


@domain(Domain.INTEL)
@mcp.tool()
async def get_institutional_accumulation(
    symbol: str,
) -> dict[str, Any]:
    """
    Assess whether institutional/smart money is accumulating a symbol.

    Combines tier_3_institutional signals — NOT retail indicators. Designed
    for buy-the-bottom strategies that need confirmation beyond price action.

    Components:
    - Insider cluster score (0.30): CEO/CFO-weighted net buy ratio over 90 days.
      Score >0.7 with 3+ distinct insiders = very high conviction.
    - GEX support (0.25): Positive gamma exposure = dealers long gamma = natural
      mean-reversion support. Negative GEX = amplifying regime = bad for bottoms.
    - IV skew extreme (0.25): Put skew z-score >2.0 = maximum fear = potential
      sentiment extreme. Crowds are maximally short = squeeze fuel.
    - Institutional direction (0.20): 13F ownership trend from flow collector.
      "accumulating" = multiple institutions increasing positions.

    Score > 0.55: Institutional accumulation underway — combine with capitulation_score > 0.65
    Score 0.35-0.55: Neutral/mixed signals — wait for clarity
    Score < 0.35: Distribution or no signal — avoid bottom entries

    IMPORTANT: A high score is not a standalone entry signal. Must also check:
      - get_capitulation_score(symbol) > 0.65 (technical washout confirmed)
      - get_credit_market_signals() credit_regime != "widening" (macro gate)

    Args:
        symbol: Stock ticker (e.g., "RDDT", "NVDA")

    Returns:
        Dict with accumulation_score (0-1), component scores, insider details,
        GEX signal, IV skew context, and recommendation

    SIGNAL TIER: tier_3_institutional
    WORKFLOW: get_capitulation_score → get_institutional_accumulation → get_credit_market_signals → debate
    RELATED: get_capitulation_score, get_credit_market_signals, get_market_breadth
    """
    result: dict[str, Any] = {"symbol": symbol}

    # ------------------------------------------------------------------
    # 1. Insider cluster score (AV API — works without DB)
    # ------------------------------------------------------------------
    insider_score = 0.5  # neutral if no data
    insider_details = []
    distinct_buyers = 0
    try:
        av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if av_key:
            client = AlphaVantageClient()
            df = client.fetch_insider_transactions(symbol)
            if df is not None and not df.empty:
                # Filter to lookback window
                df.columns = df.columns.str.lower().str.replace(" ", "_")
                date_col = next(
                    (c for c in df.columns if "date" in c), None
                )
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    cutoff = datetime.now() - timedelta(days=_LOOKBACK_DAYS)
                    df = df[df[date_col] >= cutoff]

                if not df.empty:
                    type_col = next(
                        (c for c in df.columns if "type" in c or "acquisition" in c), None
                    )
                    title_col = next(
                        (c for c in df.columns if "title" in c), None
                    )
                    name_col = next(
                        (c for c in df.columns if "name" in c or "owner" in c), None
                    )
                    shares_col = next(
                        (c for c in df.columns if "shares" in c), None
                    )

                    weighted_buys = 0.0
                    weighted_sells = 0.0
                    buyers = set()
                    sellers = set()

                    for _, row in df.iterrows():
                        is_buy = False
                        if type_col:
                            tx_type = str(row.get(type_col, "")).upper()
                            is_buy = "BUY" in tx_type or "ACQUISITION" in tx_type or "A" == tx_type.strip()

                        title = str(row.get(title_col, "")).lower() if title_col else ""
                        weight = 1.0
                        for keyword, w in _TITLE_WEIGHTS.items():
                            if keyword in title:
                                weight = max(weight, w)

                        shares = 0.0
                        if shares_col:
                            try:
                                shares = float(str(row.get(shares_col, 0)).replace(",", ""))
                            except (ValueError, TypeError):
                                shares = 1.0

                        name = str(row.get(name_col, "unknown")) if name_col else "unknown"

                        if is_buy:
                            weighted_buys += weight * max(shares, 1.0)
                            buyers.add(name)
                            insider_details.append({
                                "name": name,
                                "title": title,
                                "action": "buy",
                                "weight": round(weight, 1),
                            })
                        else:
                            weighted_sells += weight * max(shares, 1.0)
                            sellers.add(name)

                    distinct_buyers = len(buyers)
                    total = weighted_buys + weighted_sells
                    if total > 0:
                        raw_score = weighted_buys / total
                        # Boost for cluster (3+ distinct insiders buying)
                        if distinct_buyers >= 3:
                            insider_score = min(1.0, raw_score * 1.3)
                        elif distinct_buyers >= _CLUSTER_MIN_INSIDERS:
                            insider_score = min(1.0, raw_score * 1.1)
                        else:
                            insider_score = raw_score
                    elif distinct_buyers == 0:
                        insider_score = 0.3  # no transactions = slightly negative signal
        else:
            insider_score = 0.5  # neutral if no API key
    except Exception:
        insider_score = 0.5

    result["insider_cluster_score"] = round(insider_score, 3)
    result["insider_distinct_buyers"] = distinct_buyers
    result["insider_details"] = insider_details[:10]  # top 10
    result["insider_cluster_designation"] = (
        "cluster" if distinct_buyers >= 3 else "isolated" if distinct_buyers >= 1 else "none"
    )

    # ------------------------------------------------------------------
    # 2. GEX signal (from OHLCV + options data — graceful degradation)
    # ------------------------------------------------------------------
    gex_score = 0.5  # neutral default
    gex_signal = "unknown"
    gamma_flip_distance_pct = None
    try:
        # Try to get GEX from options_flow collector
        from quantstack.signal_engine.collectors.options_flow import collect_options_flow
        store = _get_reader()
        options_data = await collect_options_flow(symbol, store)

        if options_data:
            gex = options_data.get("opt_gex")
            gamma_flip = options_data.get("opt_gamma_flip")
            iv_skew_zscore = options_data.get("opt_iv_skew_zscore")
            current_price = options_data.get("spot_price")

            if gex is not None:
                if gex > 0:
                    gex_score = 0.8  # positive GEX = dealers long gamma = support
                    gex_signal = "positive_gex_support"
                elif gex < 0:
                    gex_score = 0.2  # negative GEX = amplifying = bad for bottoms
                    gex_signal = "negative_gex_amplifying"
                else:
                    gex_score = 0.5
                    gex_signal = "neutral"
                result["opt_gex"] = float(gex)

            if gamma_flip and current_price:
                dist = (float(current_price) - float(gamma_flip)) / float(gamma_flip) * 100
                gamma_flip_distance_pct = round(dist, 1)
                result["gamma_flip"] = float(gamma_flip)
                result["gamma_flip_distance_pct"] = gamma_flip_distance_pct
    except Exception:
        pass

    result["gex_score"] = round(gex_score, 3)
    result["gex_signal"] = gex_signal

    # ------------------------------------------------------------------
    # 3. IV skew extreme (options flow data)
    # ------------------------------------------------------------------
    iv_skew_score = 0.5  # neutral default
    iv_skew_extreme = False
    try:
        from quantstack.signal_engine.collectors.options_flow import collect_options_flow
        store = _get_reader()
        try:
            options_data = await collect_options_flow(symbol, store)
        finally:
            store.close()

        if options_data:
            iv_skew_zscore = options_data.get("opt_iv_skew_zscore")
            if iv_skew_zscore is not None:
                zscore = float(iv_skew_zscore)
                result["iv_skew_zscore"] = round(zscore, 2)
                if zscore >= 2.5:
                    iv_skew_score = 1.0
                    iv_skew_extreme = True
                elif zscore >= 2.0:
                    iv_skew_score = 0.85
                    iv_skew_extreme = True
                elif zscore >= 1.5:
                    iv_skew_score = 0.65
                elif zscore >= 1.0:
                    iv_skew_score = 0.5
                elif zscore < 0:
                    iv_skew_score = 0.3  # call skew = complacency
    except Exception:
        pass

    result["iv_skew_score"] = round(iv_skew_score, 3)
    result["iv_skew_extreme"] = iv_skew_extreme

    # ------------------------------------------------------------------
    # 4. Institutional direction (from flow collector if API key available)
    # ------------------------------------------------------------------
    inst_score = 0.5  # neutral default
    inst_direction = "unknown"
    try:
        fd_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
        if fd_key:
            from quantstack.signal_engine.collectors.flow import _collect_flow_sync
            from quantstack.data.storage import DataStore
            store = DataStore(read_only=True)
            try:
                flow_data = _collect_flow_sync(symbol, store)
            finally:
                store.close()

            if flow_data:
                inst_direction = flow_data.get("institutional_direction", "stable")
                if inst_direction == "accumulating":
                    inst_score = 0.85
                elif inst_direction == "stable":
                    inst_score = 0.5
                elif inst_direction == "distributing":
                    inst_score = 0.15
    except Exception:
        pass

    result["institutional_direction"] = inst_direction
    result["institutional_score"] = round(inst_score, 3)

    # ------------------------------------------------------------------
    # Composite accumulation score
    # ------------------------------------------------------------------
    composite = (
        insider_score * 0.30
        + gex_score * 0.25
        + iv_skew_score * 0.25
        + inst_score * 0.20
    )
    result["accumulation_score"] = round(composite, 3)

    if composite >= 0.55:
        result["recommendation"] = "accumulating"
        result["interpretation"] = (
            "Institutional accumulation signals present. "
            "Combine with capitulation_score > 0.65 for entry."
        )
    elif composite >= 0.35:
        result["recommendation"] = "neutral"
        result["interpretation"] = "Mixed signals. No clear accumulation pattern."
    else:
        result["recommendation"] = "distributing"
        result["interpretation"] = (
            "Distribution signals — institutions appear to be selling. "
            "Avoid bottom entries."
        )

    result["signal_tier"] = "tier_3_institutional"
    result["component_weights"] = {
        "insider_cluster": 0.30,
        "gex_support": 0.25,
        "iv_skew_extreme": 0.25,
        "institutional_direction": 0.20,
    }

    return result
