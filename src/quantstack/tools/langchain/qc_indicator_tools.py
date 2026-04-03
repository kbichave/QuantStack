"""Indicator and feature-computation tools for LangGraph agents.

Note: compute_all_features is already wrapped in ml_tools.py, so it is
intentionally excluded here.
"""

import json
from typing import Optional

from langchain_core.tools import tool


@tool
async def compute_technical_indicators(
    symbol: str,
    timeframe: str = "daily",
    indicators: Optional[list[str]] = None,
    end_date: Optional[str] = None,
) -> str:
    """Compute technical indicators for a symbol.

    SIGNAL TIER WARNING -- not all indicators are equal:
      tier_1_retail (EXIT TIMING ONLY -- never use as entry gate):
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

    Returns JSON with computed indicator values.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def list_available_indicators() -> str:
    """List all available technical indicators with signal tier classification.

    CRITICAL -- signal hierarchy for strategy design:
      tier_1_retail: Use ONLY for exit timing or rough trend context.
      tier_2_smart_money: Valid as secondary entry confirmation.
      tier_3_institutional: PRIMARY entry gate (need >=2 non-neutral).
      tier_4_regime_macro: Context gate.

    Returns JSON with indicator categories, tier classifications, and usage guidance.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_feature_matrix(
    symbol: str,
    timeframe: str = "daily",
    include_all: bool = False,
    end_date: Optional[str] = None,
) -> str:
    """Compute full feature matrix for ML/multi-factor agents.

    Returns all 200+ indicators and features computed by QuantCore
    for a symbol, ready for ML model input or factor analysis.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        include_all: If True, includes all features; if False, core set only
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns JSON with feature matrix and metadata.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_quantagent_features(
    symbol: str,
    timeframe: str = "daily",
    end_date: Optional[str] = None,
) -> str:
    """Compute QuantAgent pattern and trend features.

    Returns comprehensive pattern recognition and trend analysis features
    inspired by the Y-Research QuantAgent framework.

    Features include:
    - Pattern: pullback, breakout, consolidation, bar streaks
    - Trend: multi-horizon slopes, regime, quality (R-squared), alignment

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe ("daily", "1h", "4h")
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns JSON with pattern and trend features.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
