# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Indicator and feature-computation MCP tools.

Provides tools for computing technical indicators, full feature matrices,
and QuantAgent pattern/trend features. Extracted from ``server.py`` to keep
the tool surface modular.
"""

from typing import Any

import pandas as pd

from quantstack.config.timeframes import Timeframe
from quantstack.core.features.factory import MultiTimeframeFeatureFactory
from quantstack.core.features.quantagents_pattern import QuantAgentsPatternFeatures
from quantstack.core.features.quantagents_trend import QuantAgentsTrendFeatures
from quantstack.core.features.technical_indicators import TechnicalIndicators
from quantstack.mcp._helpers import (
    _dataframe_to_dict,
    _get_reader,
    _parse_timeframe,
)
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



@domain(Domain.DATA)
@tool_def()
async def compute_technical_indicators(
    symbol: str,
    timeframe: str = "daily",
    indicators: list[str] = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Compute technical indicators for a symbol.

    SIGNAL TIER WARNING — not all indicators are equal:
      tier_1_retail (EXIT TIMING ONLY — never use as entry gate):
        RSI, MACD, STOCH, CCI, WILLIAMS_R, BBANDS, ADX, SMA, EMA, OBV
      tier_2_smart_money (valid as secondary confirmation):
        ATR (for risk sizing), VWAP, WILLIAMS_R (as PercentRExhaustion proxy)
      For tier_3_institutional signals (PRIMARY entry gates), use instead:
        get_capitulation_score(), get_institutional_accumulation(),
        get_signal_brief() (for GEX, LSV, insider_cluster, IV skew z-score)

    Args:
        symbol: Stock symbol
        timeframe: "1h", "4h", "daily", "weekly"
        indicators: List of indicators to compute. If None, computes core set.
                   Options: ["RSI", "MACD", "ATR", "SMA", "EMA", "BBANDS",
                            "STOCH", "ADX", "OBV", "VWAP", "WILLIAMS_R"]
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, only data up to this date is used.

    Returns:
        Dictionary with computed indicator values
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Initialize indicator computer
        tech = TechnicalIndicators(
            tf,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility=True,
            enable_volume=True,
        )

        # Compute all indicators
        result_df = tech.compute(df)

        # Filter to requested indicators if specified
        if indicators:
            indicator_cols = []
            for ind in indicators:
                ind_lower = ind.lower()
                matching = [c for c in result_df.columns if ind_lower in c.lower()]
                indicator_cols.extend(matching)

            # Always include OHLCV
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            keep_cols = ohlcv_cols + list(set(indicator_cols))
            keep_cols = [c for c in keep_cols if c in result_df.columns]
            result_df = result_df[keep_cols]

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "indicators_computed": [
                c
                for c in result_df.columns
                if c not in ["open", "high", "low", "close", "volume"]
            ],
            "rows": len(result_df),
            "data": _dataframe_to_dict(result_df),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.DATA)
@tool_def()
async def compute_all_features(
    symbol: str,
    timeframe: str = "daily",
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Compute all available features for a symbol (200+ indicators).

    This includes:
    - Trend features (EMA, SMA, ADX, etc.)
    - Momentum features (RSI, MACD, Stochastic, etc.)
    - Volatility features (ATR, Bollinger Bands, etc.)
    - Volume features (OBV, VWAP, etc.)
    - Market structure (swing points, support/resistance)
    - Candlestick patterns
    - Wave analysis

    Args:
        symbol: Stock symbol
        timeframe: "1h", "4h", "daily", "weekly"
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, only data up to this date is used.

    Returns:
        Dictionary with all computed features
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data found for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Initialize factory with all features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=(tf in [Timeframe.H4, Timeframe.D1]),
            include_technical_indicators=True,
            include_trendlines=True,
            include_candlestick_patterns=True,
            include_gann_features=True,
        )

        # Compute features
        result_df = factory.compute_all_timeframes({tf: df})[tf]

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "total_features": len(result_df.columns),
            "feature_names": list(result_df.columns),
            "rows": len(result_df),
            "data": _dataframe_to_dict(result_df, max_rows=50),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.DATA)
@tool_def()
async def list_available_indicators() -> dict[str, Any]:
    """
    List all available technical indicators with signal tier classification.

    CRITICAL — signal hierarchy for strategy design:
      tier_1_retail: Use ONLY for exit timing or rough trend context. NEVER as entry gate.
        → RSI, MACD, Stochastic, CCI, Williams %R, Bollinger Bands, SMA crossover, ADX
        → These are retail noise: >90% of traders use them → front-run by market makers
        → Workshop lesson (iteration 3): Stoch/RSI rules fire ~90% of time = always-on trap
      tier_2_smart_money: Valid as secondary entry confirmation.
        → FVG (Fair Value Gap), Order Blocks, CVD (Cumulative Vol Delta), VPIN,
          Williams VIX Fix, PercentR Exhaustion, BOS/CHoCH, Supertrend, Kyle's Lambda
      tier_3_institutional: PRIMARY entry gate. Need ≥2 non-neutral before any entry.
        → GEX (Gamma Exposure), gamma flip level, IV skew z-score, LSV herding,
          insider cluster score, institutional direction, capitulation_score, accumulation_score
        → Use: get_capitulation_score(), get_institutional_accumulation(), get_signal_brief()
      tier_4_regime_macro: Context gate. If deteriorating, no bottom strategy enters.
        → HMM regime state, credit_regime, breadth_score, piotroski_f_score, yield curve

    Returns:
        Dictionary with indicator categories, tier classifications, and usage guidance
    """
    return {
        "total_indicators": 200,
        "signal_hierarchy": {
            "tier_1_retail": {
                "description": "Retail noise — exit timing ONLY, never as primary entry gate",
                "indicators": ["RSI", "MACD", "STOCH", "CCI", "WILLIAMS_R", "BBANDS",
                               "ADX", "SMA_CROSSOVER", "OBV", "MFI", "EMA", "SAR",
                               "all candlestick patterns"],
                "permitted_uses": ["exit_timing", "rough_trend_context"],
                "forbidden_uses": ["primary_entry_gate", "standalone_entry"],
            },
            "tier_2_smart_money": {
                "description": "Order-flow derived — valid as secondary entry confirmation",
                "indicators": ["FVG (Fair Value Gap)", "ORDER_BLOCKS", "CVD (Cumulative Volume Delta)",
                               "VPIN", "WILLIAMS_VIX_FIX", "PERCENT_R_EXHAUSTION",
                               "BOS_CHoCH (Break of Structure)", "SUPERTREND",
                               "KYLE_LAMBDA", "HAWKES_INTENSITY", "BB_WIDTH_COMPRESSION"],
                "permitted_uses": ["secondary_entry_confirmation", "setup_filter"],
            },
            "tier_3_institutional": {
                "description": "Dealer/institutional positioning — PRIMARY entry gate (need ≥2)",
                "indicators": ["GEX (Gamma Exposure)", "GAMMA_FLIP_LEVEL", "DEX (Delta Exposure)",
                               "IV_SKEW_ZSCORE", "LSV_HERDING", "INSIDER_CLUSTER_SCORE",
                               "INSTITUTIONAL_DIRECTION", "PUT_CALL_OI_RATIO",
                               "CAPITULATION_SCORE (get_capitulation_score)",
                               "ACCUMULATION_SCORE (get_institutional_accumulation)"],
                "mcp_tools": ["get_capitulation_score", "get_institutional_accumulation",
                              "get_signal_brief (opt_gex, opt_iv_skew_zscore, flow_signal)"],
                "permitted_uses": ["primary_entry_gate"],
            },
            "tier_4_regime_macro": {
                "description": "Context gate — if deteriorating, DO NOT enter any bottom strategy",
                "indicators": ["HMM_REGIME", "HMM_STABILITY", "CREDIT_REGIME",
                               "BREADTH_SCORE", "YIELD_CURVE_SLOPE", "MACRO_RATE_REGIME",
                               "PIOTROSKI_F_SCORE", "EGARCH_REGIME"],
                "mcp_tools": ["get_credit_market_signals", "get_market_breadth",
                              "get_regime", "get_signal_brief (macro_rate_regime, breadth fields)"],
                "permitted_uses": ["context_gate", "regime_filter"],
            },
        },
        "categories": {
            "moving_averages": {
                "tier": "tier_1_retail",
                "count": 10,
                "indicators": [
                    "SMA (Simple Moving Average) [tier_1: context only]",
                    "EMA (Exponential Moving Average) [tier_1: context only]",
                    "WMA (Weighted Moving Average) [tier_1: context only]",
                    "DEMA (Double EMA) [tier_1: context only]",
                    "TEMA (Triple EMA) [tier_1: context only]",
                    "TRIMA (Triangular MA) [tier_1: context only]",
                    "KAMA (Kaufman Adaptive MA) [tier_1: context only]",
                    "MAMA (MESA Adaptive MA) [tier_1: context only]",
                    "VWAP (Volume-Weighted Average Price) [tier_2: intraday reference]",
                    "T3 (Triple Smooth EMA) [tier_1: context only]",
                ],
            },
            "oscillators": {
                "tier": "tier_1_retail",
                "warning": "RETAIL NOISE — exit timing only, never primary entry gate",
                "count": 14,
                "indicators": [
                    "RSI [tier_1: exit timing only — RSI>70 take profit, RSI<30 means nothing alone]",
                    "MACD [tier_1: lagging exit timing only]",
                    "STOCH [tier_1: exit timing only — fires ~90% of time in OR-logic]",
                    "ADX [tier_1: trend strength context only, not entry gate]",
                    "WILLIAMS_R [tier_1: use PercentRExhaustion (tier_2) instead for exhaustion]",
                    "CCI [tier_1: exit timing only]",
                    "MFI [tier_1: use CVD (tier_2) instead for volume-price analysis]",
                    "AROON [tier_1: lagging, not edge-generating]",
                    "ROC [tier_1: momentum context only]",
                    "MOM [tier_1: context only]",
                    "PPO [tier_1: exit timing only]",
                    "CMO [tier_1: exit timing only]",
                    "ULTOSC [tier_1: exit timing only]",
                    "TRIX [tier_1: exit timing only]",
                ],
            },
            "volatility": {
                "count": 8,
                "indicators": [
                    "ATR [tier_2 for RISK SIZING only — stop placement, position sizing]",
                    "NATR [tier_2: normalized ATR for sizing]",
                    "BBANDS [tier_1 as entry gate, tier_2 as compression SETUP FILTER only]",
                    "KELTNER [tier_1: context only]",
                    "DONCHIAN [tier_1: context only]",
                    "TRANGE [tier_1: raw true range]",
                    "SAR [tier_1: exit trailing stop only]",
                    "REALIZED_VOL [tier_2: VRP computation input]",
                ],
            },
            "volume": {
                "count": 6,
                "indicators": [
                    "OBV [tier_1: lagging, use CVD (tier_2) instead]",
                    "AD [tier_1: lagging]",
                    "ADOSC [tier_1: lagging]",
                    "CMF [tier_1: lagging]",
                    "VWAP [tier_2: intraday reference price]",
                    "VOLUME_PROFILE [tier_2: identifies high-volume nodes for S/R]",
                ],
            },
            "smart_money_tier2": {
                "tier": "tier_2_smart_money",
                "description": "Valid as secondary entry confirmation",
                "indicators": [
                    "FAIR_VALUE_GAP (FVG) — bullish/bearish imbalance zones",
                    "ORDER_BLOCKS — last opposing candle before institutional impulse",
                    "BREAKER_BLOCKS — violated order blocks (shift in market structure)",
                    "BOS_CHoCH — Break of Structure / Change of Character (trend confirmation)",
                    "CVD (Cumulative Volume Delta) — buy/sell pressure from candle structure",
                    "VPIN — informed trading probability (tick-level when available)",
                    "HAWKES_INTENSITY — institutional order flow burst detection",
                    "WILLIAMS_VIX_FIX — synthetic fear gauge. Extreme = washout signal",
                    "PERCENT_R_EXHAUSTION — dual-timeframe bottom exhaustion",
                    "SUPERTREND — better trend-following than SMA crossover",
                    "KYLE_LAMBDA — price impact proxy (OHLCV approximation)",
                    "LAGUERRE_RSI — reduced-lag momentum",
                ],
            },
            "institutional_tier3": {
                "tier": "tier_3_institutional",
                "description": "PRIMARY entry gate — need ≥2 non-neutral for any strategy entry",
                "indicators": [
                    "GEX (Gamma Exposure) — positive=dealer long gamma=mean-reversion support",
                    "GAMMA_FLIP_LEVEL — key strike where GEX crosses zero",
                    "DEX (Delta Exposure) — net OI directional bias",
                    "MAX_PAIN — option expiry gravitational target",
                    "IV_SKEW_ZSCORE — put skew extreme (>2.0 = max fear = contrarian buy)",
                    "VRP (Vol Risk Premium) — IV minus realized vol, elevated = edge for sellers",
                    "LSV_HERDING — institutional crowding measure (13F-based)",
                    "INSIDER_CLUSTER_SCORE — CEO/CFO-weighted insider buy ratio",
                    "INSTITUTIONAL_DIRECTION — 13F ownership trend",
                    "PUT_CALL_OI_RATIO — crowded short = squeeze fuel",
                    "CAPITULATION_SCORE — composite washout score (get_capitulation_score)",
                    "ACCUMULATION_SCORE — composite smart money accumulation (get_institutional_accumulation)",
                ],
                "mcp_tools": ["get_capitulation_score(symbol)", "get_institutional_accumulation(symbol)",
                              "get_signal_brief(symbol) → opt_gex, opt_iv_skew_zscore, flow_signal"],
            },
            "regime_macro_tier4": {
                "tier": "tier_4_regime_macro",
                "description": "Context gate — deteriorating = block all bottom entries",
                "indicators": [
                    "HMM_REGIME — 4-state probabilistic regime (not binary ADX label)",
                    "HMM_STABILITY — regime confidence score (>0.7 = confirmed)",
                    "CREDIT_REGIME — HYG/LQD spread direction (get_credit_market_signals)",
                    "BREADTH_SCORE — % sector ETFs above 50d SMA (get_market_breadth)",
                    "BREADTH_DIVERGENCE — SPY at low but breadth stabilizing = hidden accumulation",
                    "YIELD_CURVE_SLOPE — TLT/SHY proxy (steepening into selloff = bottom precursor)",
                    "MACRO_RATE_REGIME — rate cycle position",
                    "PIOTROSKI_F_SCORE — fundamental quality gate (≥7 for investment entries)",
                    "EGARCH_REGIME — explosive vol (persistence>1.0 = widen stops, reduce size)",
                ],
                "mcp_tools": ["get_credit_market_signals()", "get_market_breadth()",
                              "get_regime(symbol)", "get_signal_brief(symbol) → macro fields"],
            },
            "market_structure": {
                "tier": "tier_2_smart_money",
                "count": 10,
                "indicators": [
                    "SWING_HIGH [tier_2: institutional structure]",
                    "SWING_LOW [tier_2: institutional structure]",
                    "SUPPORT_LEVELS [tier_2: key price memory]",
                    "RESISTANCE_LEVELS [tier_2: key price memory]",
                    "HH/HL/LH/LL [tier_2: trend structure]",
                    "TREND_DIRECTION [tier_2: structural trend]",
                    "BREAKOUT_SIGNALS [tier_2: structure break]",
                ],
            },
            "candlestick_patterns": {
                "tier": "tier_1_retail",
                "warning": "RETAIL NOISE — widely known, low edge standalone",
                "count": 40,
                "indicators": [
                    "DOJI, HAMMER, ENGULFING, MORNING_STAR, EVENING_STAR [tier_1: context only]",
                    "Use FVG and Order Blocks (tier_2) instead for institutional-grade structure",
                    "... and 35+ more patterns (all tier_1)",
                ],
            },
        },
    }


@domain(Domain.DATA)
@tool_def()
async def compute_feature_matrix(
    symbol: str,
    timeframe: str = "daily",
    include_all: bool = False,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Compute full feature matrix for ML/multi-factor agents.

    Returns all 200+ indicators and features computed by QuantCore
    for a symbol, ready for ML model input or factor analysis.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        include_all: If True, includes all features; if False, core set only
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with feature matrix and metadata
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=include_all,
            include_technical_indicators=True,
            include_trendlines=include_all,
            include_candlestick_patterns=include_all,
            include_gann_features=include_all,
        )

        features = factory.compute_all_timeframes({tf: df})[tf]

        # Get latest row as dict
        latest = features.iloc[-1].to_dict()

        # Clean up NaN values and coerce numerics to float.
        # Some feature columns (e.g. wave_role) produce string values —
        # keep those as-is rather than attempting float conversion.
        def _clean_value(v):
            if isinstance(v, str):
                return v
            try:
                return float(v) if pd.notna(v) else None
            except (TypeError, ValueError):
                return None

        latest = {k: _clean_value(v) for k, v in latest.items()}

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "timestamp": str(features.index[-1]),
            "total_features": len(features.columns),
            "feature_names": list(features.columns),
            "latest_values": latest,
            "data_points": len(features),
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@domain(Domain.DATA)
@tool_def()
async def compute_quantagent_features(
    symbol: str,
    timeframe: str = "daily",
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Compute QuantAgent pattern and trend features.

    Returns comprehensive pattern recognition and trend analysis features
    inspired by the Y-Research QuantAgent framework.

    Features include:
    - Pattern: pullback, breakout, consolidation, bar streaks
    - Trend: multi-horizon slopes, regime, quality (R²), alignment

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe ("daily", "1h", "4h")
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with pattern and trend features
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        if len(df) < 50:
            return {
                "error": f"Insufficient data for {symbol} (need 50+ bars)",
                "symbol": symbol,
            }

        # Compute pattern features
        pattern_calc = QuantAgentsPatternFeatures(tf)
        pattern_features = pattern_calc.compute(df)

        # Compute trend features
        trend_calc = QuantAgentsTrendFeatures(tf)
        trend_features = trend_calc.compute(df)

        # Get latest values
        latest_pattern = pattern_features.iloc[-1]
        latest_trend = trend_features.iloc[-1]

        # Build response
        pattern_dict = {}
        for name in pattern_calc.get_feature_names():
            val = latest_pattern.get(name)
            pattern_dict[name.replace("qa_pattern_", "")] = (
                float(val) if pd.notna(val) else None
            )

        trend_dict = {}
        for name in trend_calc.get_feature_names():
            val = latest_trend.get(name)
            trend_dict[name.replace("qa_trend_", "")] = (
                float(val) if pd.notna(val) else None
            )

        # Interpret key signals
        signals = []

        # Pattern signals
        if pattern_dict.get("is_pullback") == 1:
            signals.append("Pullback detected in uptrend")
        elif pattern_dict.get("is_pullback") == -1:
            signals.append("Bounce detected in downtrend")

        if pattern_dict.get("is_breakout") == 1:
            signals.append("Bullish breakout attempt")
        elif pattern_dict.get("is_breakout") == -1:
            signals.append("Bearish breakdown attempt")

        if pattern_dict.get("consolidation") == 1:
            signals.append("Price consolidating in range")

        if pattern_dict.get("mr_opportunity") == 1:
            signals.append("Mean reversion long opportunity")
        elif pattern_dict.get("mr_opportunity") == -1:
            signals.append("Mean reversion short opportunity")

        # Trend signals
        trend_regime = trend_dict.get("regime")
        if trend_regime == 1:
            signals.append("Uptrend regime")
        elif trend_regime == -1:
            signals.append("Downtrend regime")
        else:
            signals.append("Sideways/choppy regime")

        trend_quality = trend_dict.get("quality_med")
        if trend_quality and trend_quality > 0.8:
            signals.append("High trend quality (strong directional move)")

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "timestamp": str(df.index[-1]),
            "pattern_features": pattern_dict,
            "trend_features": trend_dict,
            "signals": signals,
            "summary": {
                "trend_regime": (
                    "up"
                    if trend_regime == 1
                    else "down" if trend_regime == -1 else "sideways"
                ),
                "trend_strength": round(trend_dict.get("strength_med", 0) or 0, 2),
                "trend_quality": round(trend_dict.get("quality_med", 0) or 0, 2),
                "is_consolidating": pattern_dict.get("consolidation") == 1,
                "has_pullback_signal": pattern_dict.get("is_pullback") != 0,
                "has_breakout_signal": pattern_dict.get("is_breakout") != 0,
            },
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
