"""Indicator and feature-computation tools for LangGraph agents.

Note: compute_all_features is already wrapped in ml_tools.py, so it is
intentionally excluded here.
"""

import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field


@tool
async def compute_technical_indicators(
    symbol: Annotated[str, Field(description="Ticker symbol to compute indicators for, e.g. 'AAPL', 'SPY', 'QQQ'")],
    timeframe: Annotated[str, Field(description="Candle interval for indicator computation: '1h', '4h', 'daily', or 'weekly'")] = "daily",
    indicators: Annotated[Optional[list[str]], Field(description="List of indicators to compute: RSI, MACD, ATR, SMA, EMA, BBANDS, STOCH, ADX, OBV, VWAP, WILLIAMS_R. Use None for core set")] = None,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Computes technical indicators (RSI, MACD, ATR, SMA, EMA, BBANDS, STOCH, ADX, OBV, VWAP) for a given symbol and timeframe. Use when you need oscillator, momentum, or volatility readings for entry/exit timing, trend confirmation, or feature engineering. Returns JSON with computed indicator values and signal tier classification.

    SIGNAL TIER WARNING -- not all indicators are equal:
      tier_1_retail (EXIT TIMING ONLY -- never use as entry gate):
        RSI, MACD, STOCH, CCI, WILLIAMS_R, BBANDS, ADX, SMA, EMA, OBV
      tier_2_smart_money (valid as secondary confirmation):
        ATR (for risk sizing), VWAP, WILLIAMS_R (as PercentRExhaustion proxy)
      For tier_3_institutional signals (PRIMARY entry gates), use instead:
        get_capitulation_score(), get_institutional_accumulation(),
        get_signal_brief() (for GEX, LSV, insider_cluster, IV skew z-score)
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def list_available_indicators() -> str:
    """Retrieves the full catalog of available technical indicators with signal tier classification and usage guidance. Use when you need to discover which indicators (RSI, MACD, ATR, BBANDS, STOCH, ADX, OBV, VWAP, SMA, EMA) are available, understand their tier hierarchy (retail vs smart-money vs institutional), or determine which indicators to pass to compute_technical_indicators. Returns JSON with indicator categories, tier classifications, and strategy design guidance.

    CRITICAL -- signal hierarchy for strategy design:
      tier_1_retail: Use ONLY for exit timing or rough trend context.
      tier_2_smart_money: Valid as secondary entry confirmation.
      tier_3_institutional: PRIMARY entry gate (need >=2 non-neutral).
      tier_4_regime_macro: Context gate.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_feature_matrix(
    symbol: Annotated[str, Field(description="Ticker symbol to compute features for, e.g. 'AAPL', 'SPY', 'NVDA'")],
    timeframe: Annotated[str, Field(description="Candle interval for feature computation: 'daily', '1h', '4h', or 'weekly'")] = "daily",
    include_all: Annotated[bool, Field(description="If True returns all 200+ features; if False returns core feature subset only")] = False,
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Computes the full feature matrix (200+ indicators and engineered features) for ML model training, multi-factor analysis, or feature importance evaluation. Use when you need a comprehensive feature set for LightGBM, XGBoost, random forest input, or factor-based alpha research. Provides momentum, volatility, volume, trend, and pattern features computed by QuantCore. Returns JSON with feature matrix and metadata.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def compute_quantagent_features(
    symbol: Annotated[str, Field(description="Ticker symbol to compute QuantAgent features for, e.g. 'AAPL', 'SPY', 'TSLA'")],
    timeframe: Annotated[str, Field(description="Candle interval: 'daily', '1h', or '4h' for pattern and trend analysis")] = "daily",
    end_date: Annotated[Optional[str], Field(description="End date filter in YYYY-MM-DD format for historical simulation or backtesting")] = None,
) -> str:
    """Computes QuantAgent pattern recognition and trend analysis features including pullback detection, breakout signals, consolidation patterns, bar streaks, multi-horizon slopes, regime classification, trend quality (R-squared), and alignment scores. Use when you need structured pattern features for swing trading, trend-following strategies, or as ML input features. Inspired by the Y-Research QuantAgent framework. Returns JSON with pattern and trend features.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
