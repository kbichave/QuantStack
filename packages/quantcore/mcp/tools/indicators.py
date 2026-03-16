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

from quantcore.config.timeframes import Timeframe
from quantcore.mcp._helpers import (
    _dataframe_to_dict,
    _get_reader,
    _parse_timeframe,
)
from quantcore.mcp.server import mcp


@mcp.tool()
async def compute_technical_indicators(
    symbol: str,
    timeframe: str = "daily",
    indicators: list[str] = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Compute technical indicators for a symbol.

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
    from quantcore.features.technical_indicators import TechnicalIndicators

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
                c for c in result_df.columns if c not in ["open", "high", "low", "close", "volume"]
            ],
            "rows": len(result_df),
            "data": _dataframe_to_dict(result_df),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
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
    from quantcore.features.factory import MultiTimeframeFeatureFactory

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


@mcp.tool()
async def list_available_indicators() -> dict[str, Any]:
    """
    List all available technical indicators and their descriptions.

    Returns:
        Dictionary with indicator categories and their indicators
    """
    return {
        "total_indicators": 200,
        "categories": {
            "moving_averages": {
                "count": 10,
                "indicators": [
                    "SMA (Simple Moving Average)",
                    "EMA (Exponential Moving Average)",
                    "WMA (Weighted Moving Average)",
                    "DEMA (Double EMA)",
                    "TEMA (Triple EMA)",
                    "TRIMA (Triangular MA)",
                    "KAMA (Kaufman Adaptive MA)",
                    "MAMA (MESA Adaptive MA)",
                    "VWAP (Volume-Weighted Average Price)",
                    "T3 (Triple Smooth EMA)",
                ],
            },
            "oscillators": {
                "count": 23,
                "indicators": [
                    "RSI (Relative Strength Index)",
                    "MACD (Moving Average Convergence/Divergence)",
                    "STOCH (Stochastic Oscillator)",
                    "ADX (Average Directional Index)",
                    "WILLIAMS_R (Williams %R)",
                    "CCI (Commodity Channel Index)",
                    "MFI (Money Flow Index)",
                    "AROON (Aroon Indicator)",
                    "ROC (Rate of Change)",
                    "MOM (Momentum)",
                    "PPO (Percentage Price Oscillator)",
                    "CMO (Chande Momentum Oscillator)",
                    "ULTOSC (Ultimate Oscillator)",
                    "TRIX (Triple Smooth EMA Rate of Change)",
                ],
            },
            "volatility": {
                "count": 8,
                "indicators": [
                    "ATR (Average True Range)",
                    "NATR (Normalized ATR)",
                    "BBANDS (Bollinger Bands)",
                    "KELTNER (Keltner Channels)",
                    "DONCHIAN (Donchian Channels)",
                    "TRANGE (True Range)",
                    "SAR (Parabolic SAR)",
                    "REALIZED_VOL (Realized Volatility)",
                ],
            },
            "volume": {
                "count": 6,
                "indicators": [
                    "OBV (On-Balance Volume)",
                    "AD (Accumulation/Distribution)",
                    "ADOSC (AD Oscillator)",
                    "CMF (Chaikin Money Flow)",
                    "VWAP (Volume-Weighted Price)",
                    "VOLUME_PROFILE (Volume Profile)",
                ],
            },
            "market_structure": {
                "count": 10,
                "indicators": [
                    "SWING_HIGH",
                    "SWING_LOW",
                    "SUPPORT_LEVELS",
                    "RESISTANCE_LEVELS",
                    "HH (Higher High)",
                    "HL (Higher Low)",
                    "LH (Lower High)",
                    "LL (Lower Low)",
                    "TREND_DIRECTION",
                    "BREAKOUT_SIGNALS",
                ],
            },
            "candlestick_patterns": {
                "count": 40,
                "indicators": [
                    "DOJI",
                    "HAMMER",
                    "ENGULFING",
                    "MORNING_STAR",
                    "EVENING_STAR",
                    "THREE_WHITE_SOLDIERS",
                    "THREE_BLACK_CROWS",
                    "SPINNING_TOP",
                    "MARUBOZU",
                    "HARAMI",
                    "... and 30+ more patterns",
                ],
            },
            "advanced": {
                "count": 15,
                "indicators": [
                    "ELLIOTT_WAVE",
                    "GANN_ANGLES",
                    "TRENDLINES",
                    "FIBONACCI_LEVELS",
                    "ZSCORE",
                    "MEAN_REVERSION_SIGNAL",
                ],
            },
        },
    }


@mcp.tool()
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
    from quantcore.features.factory import MultiTimeframeFeatureFactory

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

        # Clean up NaN values
        latest = {k: (float(v) if pd.notna(v) else None) for k, v in latest.items()}

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


@mcp.tool()
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
    from quantcore.features.quantagents_pattern import QuantAgentsPatternFeatures
    from quantcore.features.quantagents_trend import QuantAgentsTrendFeatures

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
            pattern_dict[name.replace("qa_pattern_", "")] = float(val) if pd.notna(val) else None

        trend_dict = {}
        for name in trend_calc.get_feature_names():
            val = latest_trend.get(name)
            trend_dict[name.replace("qa_trend_", "")] = float(val) if pd.notna(val) else None

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
                    "up" if trend_regime == 1 else "down" if trend_regime == -1 else "sideways"
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
