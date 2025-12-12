# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP Server - FastMCP Implementation.

Exposes QuantCore's quantitative trading functionality as MCP tools:
- Data fetching and storage
- Technical indicators (200+)
- Backtesting
- Options pricing
- Research/statistical analysis
- Risk management

Usage:
    python -m quantcore.mcp.server
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantcore.config.settings import Settings, get_settings
from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS


# =============================================================================
# MCP Server Initialization
# =============================================================================


@dataclass
class ServerContext:
    """Shared context for MCP server."""

    settings: Settings
    data_store: Any = None
    feature_factory: Any = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize and cleanup server resources."""
    logger.info("QuantCore MCP Server starting...")

    # Initialize shared context
    settings = get_settings()
    ctx = ServerContext(settings=settings)

    # Lazy import to avoid circular imports
    from quantcore.data.storage import DataStore
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    ctx.data_store = DataStore()
    ctx.feature_factory = MultiTimeframeFeatureFactory(
        include_rrg=False,  # RRG requires benchmark data
        include_waves=True,
        include_technical_indicators=True,
    )

    server.context = ctx
    logger.info("QuantCore MCP Server initialized")

    yield

    # Cleanup
    if ctx.data_store:
        ctx.data_store.close()
    logger.info("QuantCore MCP Server stopped")


# Create the FastMCP server
mcp = FastMCP(
    name="QuantCore Trading Platform",
    instructions="Quantitative trading research platform with 200+ technical indicators, "
    "backtesting, options pricing, and ML integration.",
    lifespan=lifespan,
)


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_timeframe(tf_str: str) -> Timeframe:
    """Parse timeframe string to Timeframe enum."""
    tf_map = {
        "1h": Timeframe.H1,
        "h1": Timeframe.H1,
        "hourly": Timeframe.H1,
        "4h": Timeframe.H4,
        "h4": Timeframe.H4,
        "1d": Timeframe.D1,
        "d1": Timeframe.D1,
        "daily": Timeframe.D1,
        "d": Timeframe.D1,
        "1w": Timeframe.W1,
        "w1": Timeframe.W1,
        "weekly": Timeframe.W1,
        "w": Timeframe.W1,
    }
    return tf_map.get(tf_str.lower(), Timeframe.D1)


def _dataframe_to_dict(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
    """Convert DataFrame to serializable dict with truncation."""
    if df.empty:
        return {"data": [], "columns": [], "rows": 0}

    # Truncate if needed
    truncated = len(df) > max_rows
    if truncated:
        df = df.tail(max_rows)

    # Convert datetime index to string
    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "data": data.reset_index().to_dict(orient="records"),
        "columns": list(df.columns),
        "rows": len(df),
        "truncated": truncated,
    }


def _serialize_result(obj: Any) -> Any:
    """Serialize various result types to JSON-compatible format."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return _dataframe_to_dict(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _serialize_result(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_result(v) for v in obj]
    return obj


# =============================================================================
# DATA TOOLS
# =============================================================================


@mcp.tool()
async def fetch_market_data(
    symbol: str,
    timeframe: str = "daily",
    outputsize: str = "compact",
) -> Dict[str, Any]:
    """
    Fetch OHLCV market data from Alpha Vantage API.

    Args:
        symbol: Stock/ETF symbol (e.g., "SPY", "AAPL", "QQQ")
        timeframe: Data frequency - "daily", "1h", "4h", "weekly"
        outputsize: "compact" (100 bars) or "full" (20+ years)

    Returns:
        Dictionary with OHLCV data and metadata
    """
    from quantcore.data.fetcher import AlphaVantageClient

    client = AlphaVantageClient()
    tf = _parse_timeframe(timeframe)

    try:
        if tf == Timeframe.D1:
            df = client.fetch_daily(symbol, outputsize=outputsize)
        elif tf == Timeframe.W1:
            df = client.fetch_weekly(symbol)
        else:
            # Intraday
            interval = "60min" if tf == Timeframe.H1 else "60min"
            df = client.fetch_intraday(symbol, interval=interval, outputsize=outputsize)

        if df.empty:
            return {"error": f"No data returned for {symbol}", "symbol": symbol}

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@mcp.tool()
async def load_market_data(
    symbol: str,
    timeframe: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load OHLCV data from local DuckDB storage.

    Args:
        symbol: Stock symbol
        timeframe: "1h", "4h", "daily", "weekly"
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        Dictionary with OHLCV data
    """
    from quantcore.data.storage import DataStore

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        df = store.load_ohlcv(symbol, tf, start_dt, end_dt)

        if df.empty:
            return {
                "error": f"No data found for {symbol} at {tf.value}",
                "symbol": symbol,
                "hint": "Try fetch_market_data first to download data",
            }

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "data": _dataframe_to_dict(df),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
async def list_stored_symbols() -> Dict[str, Any]:
    """
    List all symbols stored in the local database with their metadata.

    Returns:
        Dictionary with symbols and their available timeframes
    """
    from quantcore.data.storage import DataStore

    store = DataStore()

    try:
        # Query metadata table
        result = store.conn.execute(
            """
            SELECT symbol, timeframe, 
                   first_timestamp, last_timestamp, row_count
            FROM data_metadata
            ORDER BY symbol, timeframe
        """
        ).fetchall()

        symbols = {}
        for row in result:
            sym, tf, first_ts, last_ts, count = row
            if sym not in symbols:
                symbols[sym] = {"timeframes": {}}
            symbols[sym]["timeframes"][tf] = {
                "first_date": str(first_ts) if first_ts else None,
                "last_date": str(last_ts) if last_ts else None,
                "row_count": count,
            }

        return {
            "symbols": symbols,
            "total_symbols": len(symbols),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# FEATURE/INDICATOR TOOLS
# =============================================================================


@mcp.tool()
async def compute_technical_indicators(
    symbol: str,
    timeframe: str = "daily",
    indicators: List[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
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
    from quantcore.data.storage import DataStore
    from quantcore.features.technical_indicators import TechnicalIndicators

    store = DataStore()
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


@mcp.tool()
async def compute_all_features(
    symbol: str,
    timeframe: str = "daily",
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
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
    from quantcore.data.storage import DataStore
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
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
        result_df = factory.compute_single_timeframe(df, tf)

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
async def list_available_indicators() -> Dict[str, Any]:
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


# =============================================================================
# BACKTESTING TOOLS
# =============================================================================


@mcp.tool()
async def run_backtest(
    symbol: str,
    strategy_type: str = "mean_reversion",
    timeframe: str = "daily",
    initial_capital: float = 100000.0,
    position_size_pct: float = 10.0,
    stop_loss_atr: float = 2.0,
    take_profit_atr: float = 3.0,
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a backtest on historical data.

    Args:
        symbol: Stock symbol to backtest
        strategy_type: "mean_reversion", "trend_following", or "momentum"
        timeframe: "1h", "4h", "daily"
        initial_capital: Starting capital
        position_size_pct: Position size as % of equity
        stop_loss_atr: Stop loss in ATR multiples
        take_profit_atr: Take profit in ATR multiples
        zscore_entry: Z-score threshold to enter (for mean reversion)
        zscore_exit: Z-score threshold to exit (for mean reversion)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, only data up to this date is used for backtest.

    Returns:
        Dictionary with backtest results and metrics
    """
    from quantcore.data.storage import DataStore
    from quantcore.backtesting.engine import BacktestEngine, BacktestConfig
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
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

        # Compute features for signal generation
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_single_timeframe(df, tf)

        # Generate signals based on strategy
        signals_df = _generate_strategy_signals(
            features_df, strategy_type, zscore_entry, zscore_exit
        )

        # Run backtest
        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            stop_loss_atr_multiple=stop_loss_atr,
            take_profit_atr_multiple=take_profit_atr,
        )

        engine = BacktestEngine(config)
        result = engine.run(signals_df, df)

        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "timeframe": tf.value,
            "metrics": {
                "total_return": round(result.total_return, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 2),
                "max_drawdown": round(result.max_drawdown, 2),
                "win_rate": round(result.win_rate, 2),
                "total_trades": result.total_trades,
                "profit_factor": round(result.profit_factor, 2),
            },
            "trades": result.trades[:20] if result.trades else [],
            "equity_curve_sample": (
                result.equity_curve[-50:] if result.equity_curve else []
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


def _generate_strategy_signals(
    df: pd.DataFrame,
    strategy_type: str,
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
) -> pd.DataFrame:
    """Generate trading signals based on strategy type."""
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["signal_direction"] = "NONE"

    if strategy_type == "mean_reversion":
        # Use z-score for mean reversion
        if "close_zscore_20" in df.columns:
            zscore = df["close_zscore_20"]
        else:
            close = df["close"]
            mean = close.rolling(20).mean()
            std = close.rolling(20).std()
            zscore = (close - mean) / std

        signals.loc[zscore < -zscore_entry, "signal"] = 1
        signals.loc[zscore < -zscore_entry, "signal_direction"] = "LONG"
        signals.loc[zscore > zscore_entry, "signal"] = -1
        signals.loc[zscore > zscore_entry, "signal_direction"] = "SHORT"

    elif strategy_type == "trend_following":
        # Use EMA crossover
        if "ema_20" in df.columns and "ema_50" in df.columns:
            ema_fast = df["ema_20"]
            ema_slow = df["ema_50"]
        else:
            ema_fast = df["close"].ewm(span=20).mean()
            ema_slow = df["close"].ewm(span=50).mean()

        signals.loc[ema_fast > ema_slow, "signal"] = 1
        signals.loc[ema_fast > ema_slow, "signal_direction"] = "LONG"
        signals.loc[ema_fast < ema_slow, "signal"] = -1
        signals.loc[ema_fast < ema_slow, "signal_direction"] = "SHORT"

    elif strategy_type == "momentum":
        # Use RSI
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"]
        else:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        signals.loc[rsi < 30, "signal"] = 1
        signals.loc[rsi < 30, "signal_direction"] = "LONG"
        signals.loc[rsi > 70, "signal"] = -1
        signals.loc[rsi > 70, "signal_direction"] = "SHORT"

    return signals


@mcp.tool()
async def get_backtest_metrics(
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    total_trades: int,
) -> Dict[str, Any]:
    """
    Analyze and interpret backtest metrics.

    Args:
        total_return: Total return percentage
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum peak-to-trough decline
        win_rate: Percentage of winning trades
        total_trades: Total number of trades

    Returns:
        Dictionary with metric analysis and interpretation
    """
    analysis = {
        "metrics": {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
        },
        "interpretation": {},
        "overall_rating": "",
    }

    # Sharpe ratio interpretation
    if sharpe_ratio >= 2.0:
        analysis["interpretation"]["sharpe"] = "Excellent risk-adjusted returns"
    elif sharpe_ratio >= 1.0:
        analysis["interpretation"]["sharpe"] = "Good risk-adjusted returns"
    elif sharpe_ratio >= 0.5:
        analysis["interpretation"]["sharpe"] = "Moderate risk-adjusted returns"
    else:
        analysis["interpretation"]["sharpe"] = "Poor risk-adjusted returns"

    # Max drawdown interpretation
    if max_drawdown >= -10:
        analysis["interpretation"]["drawdown"] = "Excellent drawdown control"
    elif max_drawdown >= -20:
        analysis["interpretation"]["drawdown"] = "Acceptable drawdown"
    elif max_drawdown >= -30:
        analysis["interpretation"]["drawdown"] = "High drawdown risk"
    else:
        analysis["interpretation"]["drawdown"] = "Severe drawdown risk"

    # Win rate interpretation
    if win_rate >= 60:
        analysis["interpretation"]["win_rate"] = "High win rate"
    elif win_rate >= 45:
        analysis["interpretation"]["win_rate"] = "Moderate win rate"
    else:
        analysis["interpretation"]["win_rate"] = "Low win rate - needs good R:R"

    # Trade count
    if total_trades < 30:
        analysis["interpretation"]["trades"] = "Insufficient sample size"
    elif total_trades < 100:
        analysis["interpretation"]["trades"] = "Moderate sample size"
    else:
        analysis["interpretation"]["trades"] = "Good statistical significance"

    # Overall rating
    score = 0
    if sharpe_ratio >= 1.0:
        score += 2
    if max_drawdown >= -20:
        score += 2
    if win_rate >= 50:
        score += 1
    if total_trades >= 50:
        score += 1

    if score >= 5:
        analysis["overall_rating"] = "Strong strategy"
    elif score >= 3:
        analysis["overall_rating"] = "Moderate strategy"
    else:
        analysis["overall_rating"] = "Needs improvement"

    return analysis


# =============================================================================
# RESEARCH/ANALYSIS TOOLS
# =============================================================================


@mcp.tool()
async def run_adf_test(
    symbol: str,
    timeframe: str = "daily",
    column: str = "close",
    max_lags: Optional[int] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Augmented Dickey-Fuller test for stationarity.

    Tests whether a time series is stationary (mean-reverting) or has a unit root.
    A p-value < 0.05 indicates the series is stationary.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        column: Column to test ("close", "returns", "spread")
        max_lags: Maximum lags to include (auto if None)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    from quantcore.data.storage import DataStore
    from quantcore.research.stat_tests import adf_test

    store = DataStore()
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

        # Get series to test
        if column == "returns":
            series = df["close"].pct_change().dropna()
        elif column in df.columns:
            series = df[column]
        else:
            series = df["close"]

        # Run ADF test
        result = adf_test(series, max_lags=max_lags)

        return {
            "symbol": symbol,
            "column": column,
            "test_name": result.test_name,
            "statistic": (
                round(result.statistic, 4) if not np.isnan(result.statistic) else None
            ),
            "p_value": round(result.p_value, 4),
            "is_stationary": result.is_significant,
            "critical_values": result.critical_values,
            "interpretation": result.additional_info.get("interpretation", ""),
            "recommendation": (
                "Series is stationary - suitable for mean reversion strategies"
                if result.is_significant
                else "Series is non-stationary - consider differencing or trend-following"
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
async def compute_alpha_decay(
    symbol: str,
    timeframe: str = "daily",
    signal_column: str = "rsi_14",
    max_lag: int = 20,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze how a trading signal's predictive power decays over time.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        signal_column: Feature to analyze as signal
        max_lag: Maximum forward lag to analyze
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with IC decay curve, half-life, and optimal holding period
    """
    from quantcore.data.storage import DataStore
    from quantcore.research.alpha_decay import AlphaDecayAnalyzer
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
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

        # Compute features
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_single_timeframe(df, tf)

        # Get signal and returns
        if signal_column not in features_df.columns:
            # Try to find a matching column
            matches = [
                c for c in features_df.columns if signal_column.lower() in c.lower()
            ]
            if matches:
                signal_column = matches[0]
            else:
                return {
                    "error": f"Signal column {signal_column} not found",
                    "available": list(features_df.columns)[:20],
                }

        signal = features_df[signal_column].dropna()
        returns = df["close"].pct_change().dropna()

        # Align
        common_idx = signal.index.intersection(returns.index)
        signal = signal.loc[common_idx]
        returns = returns.loc[common_idx]

        # Run analysis
        analyzer = AlphaDecayAnalyzer(max_lag=max_lag)
        result = analyzer.analyze(signal, returns)

        return {
            "symbol": symbol,
            "signal_column": signal_column,
            "half_life_bars": round(result.half_life, 1),
            "decay_rate": round(result.decay_rate, 4),
            "optimal_holding_period": result.optimal_holding_period,
            "ic_by_lag": {k: round(v, 4) for k, v in result.ic_by_lag.items()},
            "turnover": round(result.turnover, 4),
            "interpretation": (
                f"Signal loses half its predictive power in {result.half_life:.1f} bars. "
                f"Optimal holding period is {result.optimal_holding_period} bars."
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
async def compute_information_coefficient(
    symbol: str,
    timeframe: str = "daily",
    signal_column: str = "rsi_14",
    forward_return_periods: int = 5,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute Information Coefficient (IC) between a signal and forward returns.

    IC measures the correlation between a predictive signal and subsequent returns.
    IC > 0.05 is generally considered meaningful.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        signal_column: Feature to analyze
        forward_return_periods: Forward return horizon in bars
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with IC value, t-statistic, and interpretation
    """
    from quantcore.data.storage import DataStore
    from quantcore.features.factory import MultiTimeframeFeatureFactory
    from scipy import stats

    store = DataStore()
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

        # Compute features
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_single_timeframe(df, tf)

        # Get signal
        if signal_column not in features_df.columns:
            matches = [
                c for c in features_df.columns if signal_column.lower() in c.lower()
            ]
            if matches:
                signal_column = matches[0]
            else:
                return {"error": f"Signal column {signal_column} not found"}

        signal = features_df[signal_column]
        forward_returns = (
            df["close"]
            .pct_change(forward_return_periods)
            .shift(-forward_return_periods)
        )

        # Align and clean
        common_idx = signal.dropna().index.intersection(forward_returns.dropna().index)
        signal_clean = signal.loc[common_idx]
        returns_clean = forward_returns.loc[common_idx]

        # Compute IC (Spearman rank correlation)
        ic, p_value = stats.spearmanr(signal_clean, returns_clean)

        # Compute t-statistic
        n = len(signal_clean)
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else 0

        return {
            "symbol": symbol,
            "signal_column": signal_column,
            "forward_period": forward_return_periods,
            "ic": round(ic, 4),
            "p_value": round(p_value, 4),
            "t_statistic": round(t_stat, 2),
            "sample_size": n,
            "is_significant": p_value < 0.05,
            "interpretation": (
                "Strong predictive signal"
                if abs(ic) > 0.1
                else (
                    "Moderate predictive signal"
                    if abs(ic) > 0.05
                    else "Weak or no predictive signal"
                )
            ),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
async def run_monte_carlo(
    symbol: str,
    timeframe: str = "daily",
    n_simulations: int = 1000,
    strategy_params: Optional[Dict[str, float]] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation to test strategy robustness.

    Randomly perturbs entry/exit timing and slippage to assess
    strategy stability under realistic conditions.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        n_simulations: Number of simulations to run
        strategy_params: Strategy parameters (entry_zscore, exit_zscore, etc.)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with simulation statistics
    """
    from quantcore.data.storage import DataStore
    from quantcore.analysis.monte_carlo import run_monte_carlo_simulation
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
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

        # Compute features for spread/zscore
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_single_timeframe(df, tf)

        # Add spread_zscore column (use close z-score as proxy)
        if "close_zscore_20" in features_df.columns:
            features_df["spread_zscore"] = features_df["close_zscore_20"]
        else:
            close = df["close"]
            mean = close.rolling(20).mean()
            std = close.rolling(20).std()
            features_df["spread_zscore"] = (close - mean) / std

        features_df["spread"] = df["close"]

        # Default params
        params = strategy_params or {
            "entry_zscore": 2.0,
            "exit_zscore": 0.0,
            "stop_loss_zscore": 5.0,
            "position_size": 2000,
        }

        # Run simulation
        result = run_monte_carlo_simulation(
            features_df,
            params,
            n_simulations=min(n_simulations, 500),  # Cap for performance
        )

        if "error" in result:
            return {"error": result["error"], "symbol": symbol}

        return {
            "symbol": symbol,
            "n_simulations": n_simulations,
            "statistics": _serialize_result(result),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


# =============================================================================
# OPTIONS TOOLS
# =============================================================================


@mcp.tool()
async def price_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    exercise_style: str = "european",
) -> Dict[str, Any]:
    """
    Calculate option price using production-grade pricing engine.

    Supports both European and American options with automatic backend selection:
    - European options: Uses vollib (Black-Scholes-Merton)
    - American options: Uses financepy (binomial tree)

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years (e.g., 0.25 for 3 months)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free interest rate (e.g., 0.05 for 5%)
        option_type: "call" or "put"
        dividend_yield: Continuous dividend yield (e.g., 0.02 for 2%)
        exercise_style: "european" or "american"

    Returns:
        Dictionary with option price, Greeks, and analysis
    """
    from quantcore.options.engine import price_option_dispatch

    try:
        result = price_option_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            exercise_style=exercise_style,
            backend="auto",
        )

        # Add analysis section for compatibility
        moneyness = spot / strike
        is_call = option_type.lower() in ("call", "c")
        itm = (spot > strike) if is_call else (spot < strike)
        intrinsic = max(0, spot - strike) if is_call else max(0, strike - spot)

        result["analysis"] = {
            "moneyness": round(moneyness, 4),
            "is_itm": itm,
            "days_to_expiry": round(time_to_expiry * 365),
            "intrinsic_value": round(intrinsic, 4),
            "time_value": round(result["price"] - intrinsic, 4),
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute option Greeks (sensitivities) using production-grade engine.

    Uses vollib for fast, accurate Greeks calculation with automatic
    fallback to internal implementation if needed.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free interest rate
        option_type: "call" or "put"
        dividend_yield: Continuous dividend yield

    Returns:
        Dictionary with detailed Greeks, interpretations, and risk metrics
    """
    from quantcore.options.engine import compute_greeks_dispatch

    try:
        result = compute_greeks_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            backend="auto",
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_implied_vol(
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_price: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> Dict[str, Any]:
    """
    Calculate implied volatility from market option price.

    Uses Newton-Raphson method via vollib for fast, accurate IV calculation.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        option_price: Market price of the option
        risk_free_rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"

    Returns:
        Dictionary with implied volatility and analysis
    """
    from quantcore.options.engine import compute_iv_dispatch

    try:
        result = compute_iv_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_price=option_price,
            option_type=option_type,
            backend="auto",
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def fit_vol_surface(
    symbol: str,
    spot_price: float,
    quotes: Dict[str, Any],
    risk_free_rate: float = 0.05,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """
    Fit SABR volatility surface to market IV quotes.

    SABR model captures volatility smile/skew patterns for more realistic
    option pricing and risk management.

    Args:
        symbol: Underlying symbol
        spot_price: Current underlying price
        quotes: Dictionary with 'strikes' and 'ivs' lists, plus 'dte'
            Example: {"strikes": [95, 100, 105], "ivs": [0.25, 0.22, 0.24], "dte": 30}
        risk_free_rate: Risk-free rate for forward calculation
        beta: SABR beta parameter (0=normal, 1=lognormal, 0.5=CIR)

    Returns:
        Dictionary with SABR parameters, fit quality, and interpolated smile
    """
    from quantcore.options.adapters.pysabr_adapter import (
        fit_sabr_surface,
        get_sabr_skew_metrics,
    )
    import pandas as pd

    try:
        # Validate inputs
        strikes = quotes.get("strikes", [])
        ivs = quotes.get("ivs", [])
        dte = quotes.get("dte", 30)

        if len(strikes) < 3 or len(strikes) != len(ivs):
            return {"error": "Need at least 3 strike/IV pairs with matching lengths"}

        # Create DataFrame
        quotes_df = pd.DataFrame({"strike": strikes, "iv": ivs})

        # Calculate forward price
        time_to_expiry = dte / 365.0
        forward = spot_price * np.exp(risk_free_rate * time_to_expiry)

        # Fit SABR
        result = fit_sabr_surface(
            quotes=quotes_df,
            forward=forward,
            time_to_expiry=time_to_expiry,
            beta=beta,
        )

        # Add skew metrics
        skew_metrics = get_sabr_skew_metrics(
            result["params"],
            forward=forward,
            time_to_expiry=time_to_expiry,
        )

        result["symbol"] = symbol
        result["spot_price"] = spot_price
        result["forward_price"] = forward
        result["dte"] = dte
        result["skew_metrics"] = skew_metrics

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def analyze_option_structure(
    structure_spec: Dict[str, Any],
    price_range_pct: float = 0.30,
) -> Dict[str, Any]:
    """
    Analyze multi-leg option structure for payoff, Greeks, and key metrics.

    Supports all standard structures: verticals, straddles, strangles,
    iron condors, butterflies, and custom multi-leg positions.

    Args:
        structure_spec: Structure specification dictionary with:
            - underlying_symbol: Symbol (e.g., "SPY")
            - underlying_price: Current price (e.g., 450.0)
            - legs: List of leg dictionaries, each with:
                - option_type: "call" or "put"
                - strike: Strike price
                - expiry_days: Days to expiration
                - quantity: Positive for long, negative for short
                - premium: (optional) Entry premium
                - iv: (optional) Implied volatility
            - risk_free_rate: (optional) Rate, default 0.05
        price_range_pct: Range around spot for payoff profile (default 30%)

    Returns:
        Dictionary with:
            - structure_type: Identified structure name
            - payoff_profile: Price grid vs payoff at expiry
            - greeks: Aggregated position Greeks
            - break_evens: Break-even price points
            - max_profit, max_loss: Profit/loss boundaries
            - risk_reward_ratio: Max profit / Max loss
            - probability_of_profit: Estimated POP

    Example:
        analyze_option_structure({
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {"option_type": "call", "strike": 445, "expiry_days": 30, "quantity": 1, "iv": 0.20},
                {"option_type": "call", "strike": 455, "expiry_days": 30, "quantity": -1, "iv": 0.18}
            ]
        })
    """
    from quantcore.options.adapters.quantsbin_adapter import analyze_structure_quantsbin

    try:
        result = analyze_structure_quantsbin(
            structure_spec=structure_spec,
            price_range_pct=price_range_pct,
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_portfolio_stats(
    equity_curve: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Compute comprehensive portfolio performance statistics.

    Uses ffn library for production-grade analytics including Sharpe,
    Sortino, Calmar ratios, drawdown analysis, and distribution metrics.

    Args:
        equity_curve: List of portfolio equity values over time
        risk_free_rate: Annual risk-free rate for ratio calculations
        periods_per_year: Trading periods per year (252 for daily, 52 for weekly)

    Returns:
        Dictionary with:
            - Return metrics: total_return, cagr, annualized_return
            - Risk metrics: volatility, max_drawdown, VaR, CVaR
            - Ratios: sharpe_ratio, sortino_ratio, calmar_ratio
            - Distribution: skewness, kurtosis, best/worst day
            - Drawdown details: duration, recovery time
    """
    from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn
    import pandas as pd

    try:
        # Convert to pandas Series
        equity = pd.Series(equity_curve)

        result = compute_portfolio_stats_ffn(
            equity_curve=equity,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def price_american_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Price American option using binomial tree method.

    American options can be exercised at any time before expiration,
    which may be valuable for dividend-paying stocks (calls) or
    deep ITM puts.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        option_type: "call" or "put"
        num_steps: Number of tree steps (more = accurate but slower)

    Returns:
        Dictionary with:
            - price: American option price
            - european_price: European equivalent price
            - early_exercise_premium: Value of early exercise right
            - delta, gamma: Greeks from tree
    """
    from quantcore.options.adapters.financepy_adapter import (
        price_american_option as price_american,
    )

    try:
        result = price_american(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            num_steps=num_steps,
        )

        result["inputs"] = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "option_type": option_type,
        }

        return result

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# RISK TOOLS
# =============================================================================


@mcp.tool()
async def compute_position_size(
    equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 20.0,
    alignment_score: float = 1.0,
) -> Dict[str, Any]:
    """
    Calculate position size using ATR-based risk management.

    Args:
        equity: Total account equity
        entry_price: Planned entry price
        stop_loss_price: Stop loss price level
        risk_per_trade_pct: Percentage of equity to risk per trade
        max_position_pct: Maximum position as % of equity
        alignment_score: Cross-timeframe alignment score (0-1)

    Returns:
        Dictionary with position size and risk details
    """
    from quantcore.risk.position_sizing import ATRPositionSizer

    try:
        sizer = ATRPositionSizer(
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
        )

        result = sizer.calculate(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            alignment_score=alignment_score,
        )

        return {
            "position": {
                "shares": round(result.shares, 2),
                "notional_value": round(result.notional_value, 2),
                "position_pct_of_equity": round(
                    result.notional_value / equity * 100, 2
                ),
            },
            "risk": {
                "risk_amount": round(result.risk_amount, 2),
                "risk_pct": round(result.risk_pct, 2),
                "risk_per_share": round(abs(entry_price - stop_loss_price), 2),
            },
            "adjustments": {
                "alignment_multiplier": round(result.alignment_multiplier, 2),
                "was_capped": result.notional_value >= equity * max_position_pct / 100,
            },
            "trade_details": {
                "entry_price": entry_price,
                "stop_loss": stop_loss_price,
                "stop_distance_pct": round(
                    abs(entry_price - stop_loss_price) / entry_price * 100, 2
                ),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_max_drawdown(
    equity_curve: List[float],
) -> Dict[str, Any]:
    """
    Compute maximum drawdown and drawdown statistics.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Dictionary with drawdown metrics
    """
    try:
        equity = np.array(equity_curve)

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown at each point
        drawdown = (equity - running_max) / running_max * 100

        # Find max drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()

        # Find peak before max drawdown
        peak_idx = running_max[: max_dd_idx + 1].argmax()

        # Find recovery point (if any)
        recovery_idx = None
        for i in range(max_dd_idx, len(equity)):
            if equity[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break

        # Calculate current drawdown
        current_dd = drawdown[-1]

        return {
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_idx": int(max_dd_idx),
            "peak_idx": int(peak_idx),
            "peak_value": round(equity[peak_idx], 2),
            "trough_value": round(equity[max_dd_idx], 2),
            "recovery_idx": int(recovery_idx) if recovery_idx else None,
            "drawdown_duration": int(max_dd_idx - peak_idx),
            "recovery_duration": (
                int(recovery_idx - max_dd_idx) if recovery_idx else None
            ),
            "current_drawdown_pct": round(current_dd, 2),
            "is_in_drawdown": current_dd < 0,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# MAS-ORIENTED TOOLS (Multi-Agent System Optimization)
# =============================================================================


@mcp.tool()
async def get_symbol_snapshot(
    symbol: str,
    timeframe: str = "daily",
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a unified snapshot of a symbol for fast MAS reasoning.

    Combines price, technical, volatility, and wave data in a single call
    to reduce agent reasoning overhead and tool call latency.

    Args:
        symbol: Stock/ETF symbol
        timeframe: Data timeframe ("daily", "1h", "4h")
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, returns snapshot as of this date.

    Returns:
        Dictionary with:
            - price: Latest OHLCV
            - technicals: RSI, MACD, ATR, key MAs
            - volatility: Realized vol, IV rank (if available)
            - trend: Direction, strength, regime
            - levels: Support/resistance, key pivots
    """
    from quantcore.data.storage import DataStore
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
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

        # Get latest bar
        latest = df.iloc[-1]

        # Compute features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=False,
            include_technical_indicators=True,
        )
        features = factory.compute_single_timeframe(df, tf)
        latest_features = features.iloc[-1]

        # Extract key metrics
        snapshot = {
            "symbol": symbol,
            "timeframe": tf.value,
            "timestamp": str(df.index[-1]),
            "price": {
                "open": round(latest["open"], 2),
                "high": round(latest["high"], 2),
                "low": round(latest["low"], 2),
                "close": round(latest["close"], 2),
                "volume": int(latest["volume"]),
                "change_pct": (
                    round((latest["close"] / df["close"].iloc[-2] - 1) * 100, 2)
                    if len(df) > 1
                    else 0
                ),
            },
            "technicals": {},
            "volatility": {},
            "trend": {},
            "levels": {},
        }

        # Technical indicators
        for col in ["rsi_14", "macd", "macd_signal", "adx_14"]:
            if col in latest_features.index:
                snapshot["technicals"][col] = round(float(latest_features[col]), 2)

        # ATR
        if "atr_14" in latest_features.index:
            atr = float(latest_features["atr_14"])
            snapshot["technicals"]["atr"] = round(atr, 2)
            snapshot["technicals"]["atr_pct"] = round(atr / latest["close"] * 100, 2)

        # Volatility
        returns = df["close"].pct_change().dropna()
        if len(returns) > 20:
            realized_vol = float(returns.tail(20).std() * np.sqrt(252))
            snapshot["volatility"]["realized_vol_20d"] = round(realized_vol, 4)

        # Trend
        if "ema_20" in latest_features.index and "ema_50" in latest_features.index:
            ema_20 = float(latest_features["ema_20"])
            ema_50 = float(latest_features["ema_50"])
            snapshot["trend"]["ema_alignment"] = (
                "bullish" if ema_20 > ema_50 else "bearish"
            )
            snapshot["trend"]["above_ema_20"] = latest["close"] > ema_20

        # Support/Resistance (simple high/low levels)
        if len(df) > 20:
            snapshot["levels"]["high_20d"] = round(float(df["high"].tail(20).max()), 2)
            snapshot["levels"]["low_20d"] = round(float(df["low"].tail(20).min()), 2)

        return snapshot

    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
async def get_market_regime_snapshot(
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get high-level market regime for global filters in MAS.

    Returns market-wide regime information useful for gating trades
    and adjusting strategy parameters across all agents.

    Args:
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, returns regime as of this date.

    Returns:
        Dictionary with:
            - trend_regime: bull, bear, or sideways
            - volatility_regime: low, normal, or high (based on VIX proxy)
            - breadth: Market breadth indicator
            - risk_appetite: Risk-on or risk-off signal
    """
    from quantcore.data.storage import DataStore
    from quantcore.hierarchy.regime_classifier import WeeklyRegimeClassifier
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()

    try:
        # Use SPY as market proxy
        df = store.load_ohlcv("SPY", _parse_timeframe("daily"))

        if df.empty:
            return {"error": "No SPY data available for regime detection"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No SPY data before {end_date}"}

        # Compute features
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features = factory.compute_single_timeframe(df, _parse_timeframe("daily"))

        # Classify regime
        classifier = WeeklyRegimeClassifier()
        regime_ctx = classifier.classify(features)

        # Calculate volatility regime
        returns = df["close"].pct_change().dropna()
        current_vol = (
            float(returns.tail(20).std() * np.sqrt(252)) if len(returns) > 20 else 0.15
        )

        if current_vol < 0.12:
            vol_regime = "low"
        elif current_vol < 0.20:
            vol_regime = "normal"
        else:
            vol_regime = "high"

        # Trend detection
        latest = features.iloc[-1]
        close = df["close"].iloc[-1]

        ema_20 = float(latest.get("ema_20", close))
        ema_50 = float(latest.get("ema_50", close))

        if close > ema_20 > ema_50:
            trend = "bull"
        elif close < ema_20 < ema_50:
            trend = "bear"
        else:
            trend = "sideways"

        return {
            "timestamp": str(df.index[-1]),
            "market_proxy": "SPY",
            "regime": {
                "trend": trend,
                "volatility": vol_regime,
                "confidence": round(regime_ctx.confidence, 2),
            },
            "metrics": {
                "spy_price": round(close, 2),
                "realized_vol_20d": round(current_vol, 4),
                "ema_alignment": regime_ctx.ema_alignment,
                "momentum_score": round(regime_ctx.momentum_score, 2),
            },
            "signals": {
                "allows_long": regime_ctx.allows_long(),
                "allows_short": regime_ctx.allows_short(),
                "risk_appetite": (
                    "risk_on"
                    if trend == "bull" and vol_regime != "high"
                    else "risk_off"
                ),
            },
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def generate_trade_template(
    symbol: str,
    direction: str,
    structure_type: str = "vertical",
    expiry_days: int = 30,
    risk_amount: float = 500.0,
    underlying_price: Optional[float] = None,
    iv_estimate: float = 0.25,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a fully structured trade template for MAS approval workflow.

    Creates a complete trade specification that can be passed to
    HumanApprovalAgent or AutoExecutionAgent for final validation.

    Args:
        symbol: Underlying symbol
        direction: "bullish", "bearish", or "neutral"
        structure_type: "vertical", "single", "straddle", "iron_condor"
        expiry_days: Days to target expiration
        risk_amount: Maximum dollar risk for the trade
        underlying_price: Current price (fetched if not provided)
        iv_estimate: Estimated IV for pricing
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, uses price as of this date.

    Returns:
        Dictionary with complete trade template:
            - legs: Fully specified option legs
            - risk: Max loss, max profit, break-evens
            - greeks: Position Greeks
            - validation_status: Pre-validation results
    """
    from quantcore.data.storage import DataStore
    from quantcore.options.adapters.quantsbin_adapter import analyze_structure_quantsbin
    from quantcore.options.engine import price_option_dispatch

    try:
        # Get current price if not provided
        if underlying_price is None:
            store = DataStore()
            df = store.load_ohlcv(symbol, _parse_timeframe("daily"))
            store.close()

            if df.empty:
                return {"error": f"No price data for {symbol}"}

            # Filter to end_date if provided (for historical simulation)
            if end_date and not df.empty:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
                if df.empty:
                    return {"error": f"No price data for {symbol} before {end_date}"}

            underlying_price = float(df["close"].iloc[-1])

        # Round to nearest strike increment
        if underlying_price > 100:
            strike_increment = 5.0
        elif underlying_price > 50:
            strike_increment = 2.5
        else:
            strike_increment = 1.0

        atm_strike = round(underlying_price / strike_increment) * strike_increment

        # Build legs based on structure type
        legs = []

        if structure_type == "single":
            opt_type = "call" if direction == "bullish" else "put"
            strike = atm_strike if direction != "neutral" else atm_strike

            legs.append(
                {
                    "option_type": opt_type,
                    "strike": strike,
                    "expiry_days": expiry_days,
                    "quantity": 1,
                    "iv": iv_estimate,
                }
            )

        elif structure_type == "vertical":
            if direction == "bullish":
                legs = [
                    {
                        "option_type": "call",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": 1,
                        "iv": iv_estimate,
                    },
                    {
                        "option_type": "call",
                        "strike": atm_strike + strike_increment * 2,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate * 0.95,
                    },
                ]
            elif direction == "bearish":
                legs = [
                    {
                        "option_type": "put",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": 1,
                        "iv": iv_estimate,
                    },
                    {
                        "option_type": "put",
                        "strike": atm_strike - strike_increment * 2,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate * 0.95,
                    },
                ]
            else:  # neutral - iron butterfly
                legs = [
                    {
                        "option_type": "put",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate,
                    },
                    {
                        "option_type": "call",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate,
                    },
                ]

        elif structure_type == "straddle":
            legs = [
                {
                    "option_type": "call",
                    "strike": atm_strike,
                    "expiry_days": expiry_days,
                    "quantity": 1 if direction != "neutral" else -1,
                    "iv": iv_estimate,
                },
                {
                    "option_type": "put",
                    "strike": atm_strike,
                    "expiry_days": expiry_days,
                    "quantity": 1 if direction != "neutral" else -1,
                    "iv": iv_estimate,
                },
            ]

        elif structure_type == "iron_condor":
            width = strike_increment * 2
            legs = [
                {
                    "option_type": "put",
                    "strike": atm_strike - width * 2,
                    "expiry_days": expiry_days,
                    "quantity": 1,
                    "iv": iv_estimate * 1.1,
                },
                {
                    "option_type": "put",
                    "strike": atm_strike - width,
                    "expiry_days": expiry_days,
                    "quantity": -1,
                    "iv": iv_estimate * 1.05,
                },
                {
                    "option_type": "call",
                    "strike": atm_strike + width,
                    "expiry_days": expiry_days,
                    "quantity": -1,
                    "iv": iv_estimate * 0.95,
                },
                {
                    "option_type": "call",
                    "strike": atm_strike + width * 2,
                    "expiry_days": expiry_days,
                    "quantity": 1,
                    "iv": iv_estimate * 0.9,
                },
            ]

        # Analyze structure
        structure_spec = {
            "underlying_symbol": symbol,
            "underlying_price": underlying_price,
            "legs": legs,
        }

        analysis = analyze_structure_quantsbin(structure_spec)

        # Calculate quantity based on risk
        if analysis.get("max_loss") and analysis["max_loss"] < 0:
            max_loss_per_contract = abs(analysis["max_loss"])
            quantity = max(1, int(risk_amount / max_loss_per_contract))
        else:
            quantity = 1

        # Scale legs by quantity
        for leg in legs:
            leg["quantity"] = leg["quantity"] * quantity

        # Recalculate with scaled quantity
        structure_spec["legs"] = legs
        analysis = analyze_structure_quantsbin(structure_spec)

        return {
            "template_id": f"{symbol}_{structure_type}_{direction}_{expiry_days}d",
            "symbol": symbol,
            "direction": direction,
            "structure_type": analysis.get("structure_type", structure_type),
            "underlying_price": underlying_price,
            "legs": legs,
            "risk_profile": {
                "max_profit": analysis.get("max_profit", 0),
                "max_loss": analysis.get("max_loss", 0),
                "break_evens": analysis.get("break_evens", []),
                "risk_reward_ratio": analysis.get("risk_reward_ratio", 0),
                "probability_of_profit": analysis.get("probability_of_profit"),
            },
            "greeks": analysis.get("greeks", {}),
            "validation": {
                "is_defined_risk": analysis.get("is_defined_risk", False),
                "within_risk_limit": abs(analysis.get("max_loss", 0))
                <= risk_amount * 1.1,
                "has_positive_expectancy": analysis.get("max_profit", 0)
                > abs(analysis.get("max_loss", 0)) * 0.3,
            },
            "execution_notes": {
                "order_type": "limit",
                "price_target": (
                    round(analysis.get("total_premium", 0) / 100 / quantity, 2)
                    if quantity > 0
                    else 0
                ),
                "time_in_force": "day",
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def validate_trade(
    trade_template: Dict[str, Any],
    account_equity: float = 100000.0,
    max_position_pct: float = 5.0,
    max_daily_loss_pct: float = 2.0,
    current_daily_pnl: float = 0.0,
) -> Dict[str, Any]:
    """
    Validate a trade template against risk rules before execution.

    Final gate before sending orders to broker. Checks position limits,
    risk constraints, and structure validity.

    Args:
        trade_template: Trade template from generate_trade_template
        account_equity: Total account equity
        max_position_pct: Maximum position size as % of equity
        max_daily_loss_pct: Maximum daily loss allowed as % of equity
        current_daily_pnl: Current day's realized P&L

    Returns:
        Dictionary with validation results:
            - is_valid: Boolean pass/fail
            - checks: Individual check results
            - warnings: Non-blocking concerns
            - rejection_reasons: If invalid, why
    """
    try:
        checks = {}
        warnings = []
        rejection_reasons = []

        # Extract key metrics
        max_loss = abs(
            trade_template.get("risk_profile", {}).get("max_loss", float("inf"))
        )
        is_defined_risk = trade_template.get("validation", {}).get(
            "is_defined_risk", False
        )
        symbol = trade_template.get("symbol", "UNKNOWN")

        # Check 1: Defined risk
        checks["defined_risk"] = is_defined_risk
        if not is_defined_risk:
            rejection_reasons.append("Trade has undefined/unlimited risk")

        # Check 2: Position size limit
        max_position_value = account_equity * max_position_pct / 100
        checks["within_position_limit"] = max_loss <= max_position_value
        if not checks["within_position_limit"]:
            rejection_reasons.append(
                f"Max loss ${max_loss:.0f} exceeds position limit ${max_position_value:.0f}"
            )

        # Check 3: Daily loss limit
        remaining_daily_risk = (
            account_equity * max_daily_loss_pct / 100 + current_daily_pnl
        )
        checks["within_daily_limit"] = max_loss <= remaining_daily_risk
        if not checks["within_daily_limit"]:
            rejection_reasons.append(
                f"Max loss would exceed daily limit. Remaining: ${remaining_daily_risk:.0f}"
            )

        # Check 4: Greeks sanity
        greeks = trade_template.get("greeks", {})
        delta = abs(greeks.get("delta", 0))

        checks["delta_reasonable"] = delta < 500  # Max 500 delta per position
        if not checks["delta_reasonable"]:
            rejection_reasons.append(f"Delta exposure {delta:.0f} too high")

        # Check 5: Legs consistency
        legs = trade_template.get("legs", [])
        checks["has_legs"] = len(legs) > 0
        if not checks["has_legs"]:
            rejection_reasons.append("No legs in trade template")

        # Check 6: Break-evens exist for spreads
        break_evens = trade_template.get("risk_profile", {}).get("break_evens", [])
        if len(legs) > 1 and len(break_evens) == 0:
            warnings.append("Multi-leg structure has no calculated break-evens")

        # Check 7: Risk/reward sanity
        rr_ratio = trade_template.get("risk_profile", {}).get("risk_reward_ratio", 0)
        if rr_ratio < 0.2:
            warnings.append(f"Risk/reward ratio {rr_ratio:.2f} is unfavorable")

        # Check 8: Expiration not too close
        legs_expiry = [leg.get("expiry_days", 30) for leg in legs]
        min_expiry = min(legs_expiry) if legs_expiry else 30
        checks["sufficient_time"] = min_expiry >= 3
        if not checks["sufficient_time"]:
            warnings.append(f"Expiration in {min_expiry} days - consider rolling")

        # Overall validation
        is_valid = len(rejection_reasons) == 0

        return {
            "is_valid": is_valid,
            "symbol": symbol,
            "checks": checks,
            "warnings": warnings,
            "rejection_reasons": rejection_reasons,
            "risk_summary": {
                "max_loss": max_loss,
                "max_position_allowed": max_position_value,
                "remaining_daily_risk": remaining_daily_risk,
                "delta_exposure": delta,
            },
            "approval_status": "APPROVED" if is_valid else "REJECTED",
        }

    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "approval_status": "ERROR",
        }


@mcp.tool()
async def compute_feature_matrix(
    symbol: str,
    timeframe: str = "daily",
    include_all: bool = False,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
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
    from quantcore.data.storage import DataStore
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
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

        features = factory.compute_single_timeframe(df, tf)

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
async def run_screener(
    symbols: Optional[List[str]] = None,
    min_price: float = 10.0,
    max_price: float = 500.0,
    min_volume: int = 100000,
    trend_filter: Optional[str] = None,
    rsi_oversold: Optional[float] = None,
    rsi_overbought: Optional[float] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run multi-factor screener across symbols.

    Filters symbols by price, volume, trend, and technical conditions.
    Used by OpportunityDiscoveryAgent to find trade candidates.

    Args:
        symbols: List of symbols to screen (uses config defaults if None)
        min_price: Minimum price filter
        max_price: Maximum price filter
        min_volume: Minimum average volume
        trend_filter: "bullish", "bearish", or None
        rsi_oversold: RSI below this value = oversold
        rsi_overbought: RSI above this value = overbought
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with matching symbols and their key metrics
    """
    from quantcore.data.storage import DataStore
    from quantcore.config.settings import get_settings
    from quantcore.features.technical_indicators import TechnicalIndicators

    store = DataStore()
    tf = _parse_timeframe("daily")
    end_dt = pd.to_datetime(end_date) if end_date else None

    try:
        # Get symbols to screen
        if symbols is None:
            settings = get_settings()
            symbols = settings.symbols[:20]  # Limit for performance

        matches = []

        for symbol in symbols:
            try:
                df = store.load_ohlcv(symbol, tf)

                # Filter to end_date if provided (for historical simulation)
                if end_dt is not None and not df.empty:
                    df = df[df.index <= end_dt]

                if df.empty or len(df) < 50:
                    continue

                latest = df.iloc[-1]
                price = float(latest["close"])
                volume = float(df["volume"].tail(20).mean())

                # Price filter
                if price < min_price or price > max_price:
                    continue

                # Volume filter
                if volume < min_volume:
                    continue

                # Compute indicators
                tech = TechnicalIndicators(tf)
                features = tech.compute(df)
                latest_features = features.iloc[-1]

                # Trend filter
                if trend_filter:
                    ema_20 = float(latest_features.get("ema_20", price))
                    ema_50 = float(latest_features.get("ema_50", price))

                    is_bullish = price > ema_20 > ema_50
                    is_bearish = price < ema_20 < ema_50

                    if trend_filter == "bullish" and not is_bullish:
                        continue
                    if trend_filter == "bearish" and not is_bearish:
                        continue

                # RSI filter
                rsi = float(latest_features.get("rsi_14", 50))

                if rsi_oversold and rsi > rsi_oversold:
                    continue
                if rsi_overbought and rsi < rsi_overbought:
                    continue

                # Add to matches
                matches.append(
                    {
                        "symbol": symbol,
                        "price": round(price, 2),
                        "volume": int(volume),
                        "rsi": round(rsi, 1),
                        "trend": "bullish" if price > ema_20 else "bearish",
                        "change_1d": (
                            round((price / df["close"].iloc[-2] - 1) * 100, 2)
                            if len(df) > 1
                            else 0
                        ),
                    }
                )

            except Exception as e:
                continue

        # Sort by volume
        matches.sort(key=lambda x: x["volume"], reverse=True)

        return {
            "total_screened": len(symbols),
            "matches": len(matches),
            "filters_applied": {
                "price_range": [min_price, max_price],
                "min_volume": min_volume,
                "trend": trend_filter,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
            },
            "results": matches[:20],  # Top 20
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def compute_option_chain(
    symbol: str,
    expiry_date: Optional[str] = None,
    min_delta: float = 0.05,
    max_delta: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute normalized option chain data for a symbol.

    Returns clean, normalized chain data suitable for OptionStructuringAgent.
    Calculates moneyness, IVs, and Greeks for all strikes.

    Args:
        symbol: Underlying symbol
        expiry_date: Target expiry (YYYY-MM-DD), uses nearest if not provided
        min_delta: Minimum delta to include (filters far OTM)
        max_delta: Maximum delta to include (filters deep ITM)

    Returns:
        Dictionary with:
            - calls: List of call options with Greeks
            - puts: List of put options with Greeks
            - underlying_price: Current price
            - expiries: Available expiration dates
            - chain_metrics: ATM IV, put/call skew, etc.
    """
    from quantcore.data.storage import DataStore
    from quantcore.options.engine import (
        price_option_dispatch,
        compute_greeks_dispatch,
        compute_iv_dispatch,
    )

    store = DataStore()

    try:
        # Get underlying price
        df = store.load_ohlcv(symbol, _parse_timeframe("daily"))

        if df.empty:
            return {"error": f"No price data for {symbol}"}

        underlying_price = float(df["close"].iloc[-1])

        # Generate synthetic chain strikes (in production, this would come from broker MCP)
        if underlying_price > 100:
            strike_increment = 5.0
        elif underlying_price > 50:
            strike_increment = 2.5
        else:
            strike_increment = 1.0

        # Calculate strike range based on delta bounds
        # Use rough approximation: 10-90 delta covers ~1 std dev
        vol_estimate = 0.25  # Default IV estimate
        atm_strike = round(underlying_price / strike_increment) * strike_increment

        # Generate strikes from -30% to +30%
        strikes = [
            atm_strike + i * strike_increment
            for i in range(-6, 7)
            if (atm_strike + i * strike_increment) > 0
        ]

        # Default expiry: 30 DTE
        dte = 30
        tte = dte / 365.0
        rate = 0.05
        div_yield = 0.01

        calls = []
        puts = []

        for strike in strikes:
            # Calculate call metrics
            call_result = price_option_dispatch(
                underlying_price, strike, tte, vol_estimate, rate, div_yield, "call"
            )

            call_greeks = call_result.get("greeks", {})
            call_delta = abs(call_greeks.get("delta", 0))

            if min_delta <= call_delta <= max_delta:
                calls.append(
                    {
                        "strike": strike,
                        "bid": round(call_result["price"] * 0.98, 2),
                        "ask": round(call_result["price"] * 1.02, 2),
                        "mid": round(call_result["price"], 2),
                        "iv": vol_estimate,
                        "delta": round(call_greeks.get("delta", 0), 4),
                        "gamma": round(call_greeks.get("gamma", 0), 6),
                        "theta": round(call_greeks.get("theta", 0), 4),
                        "vega": round(call_greeks.get("vega", 0), 4),
                        "moneyness": round(np.log(underlying_price / strike), 4),
                        "dte": dte,
                    }
                )

            # Calculate put metrics
            put_result = price_option_dispatch(
                underlying_price, strike, tte, vol_estimate, rate, div_yield, "put"
            )

            put_greeks = put_result.get("greeks", {})
            put_delta = abs(put_greeks.get("delta", 0))

            if min_delta <= put_delta <= max_delta:
                puts.append(
                    {
                        "strike": strike,
                        "bid": round(put_result["price"] * 0.98, 2),
                        "ask": round(put_result["price"] * 1.02, 2),
                        "mid": round(put_result["price"], 2),
                        "iv": vol_estimate,
                        "delta": round(put_greeks.get("delta", 0), 4),
                        "gamma": round(put_greeks.get("gamma", 0), 6),
                        "theta": round(put_greeks.get("theta", 0), 4),
                        "vega": round(put_greeks.get("vega", 0), 4),
                        "moneyness": round(np.log(underlying_price / strike), 4),
                        "dte": dte,
                    }
                )

        return {
            "symbol": symbol,
            "underlying_price": underlying_price,
            "as_of": str(df.index[-1]),
            "calls": sorted(calls, key=lambda x: x["strike"]),
            "puts": sorted(puts, key=lambda x: x["strike"]),
            "chain_metrics": {
                "atm_strike": atm_strike,
                "atm_iv": vol_estimate,
                "num_calls": len(calls),
                "num_puts": len(puts),
                "dte": dte,
            },
            "note": "Synthetic chain - use broker MCP for live data",
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def compute_multi_leg_price(
    legs: List[Dict[str, Any]],
    underlying_price: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """
    Price a multi-leg options structure in one call.

    Aggregates pricing, Greeks, and risk metrics across all legs.
    Used by OptionStructuringAgent for rapid structure evaluation.

    Args:
        legs: List of leg specifications, each with:
            - option_type: "call" or "put"
            - strike: Strike price
            - expiry_days: Days to expiration
            - quantity: Number of contracts (negative for short)
            - iv: Implied volatility (optional, defaults to 0.25)
        underlying_price: Current underlying price
        rate: Risk-free rate
        dividend_yield: Continuous dividend yield

    Returns:
        Dictionary with:
            - total_price: Net premium (debit or credit)
            - leg_prices: Individual leg prices
            - net_greeks: Aggregated position Greeks
            - max_profit: Maximum profit potential
            - max_loss: Maximum loss potential
            - break_evens: Break-even prices
    """
    from quantcore.options.engine import price_option_dispatch

    try:
        if not legs:
            return {"error": "No legs provided"}

        leg_results = []
        total_premium = 0.0
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        net_rho = 0.0

        for i, leg in enumerate(legs):
            opt_type = leg.get("option_type", "call")
            strike = leg["strike"]
            expiry_days = leg.get("expiry_days", 30)
            quantity = leg.get("quantity", 1)
            iv = leg.get("iv", 0.25)

            tte = expiry_days / 365.0

            result = price_option_dispatch(
                underlying_price, strike, tte, iv, rate, dividend_yield, opt_type
            )

            price = result["price"]
            greeks = result.get("greeks", {})

            # Aggregate (negative quantity = short position = credit)
            leg_premium = price * quantity * 100  # Per contract value
            total_premium += leg_premium

            net_delta += greeks.get("delta", 0) * quantity * 100
            net_gamma += greeks.get("gamma", 0) * quantity * 100
            net_theta += greeks.get("theta", 0) * quantity * 100
            net_vega += greeks.get("vega", 0) * quantity * 100
            net_rho += greeks.get("rho", 0) * quantity * 100

            leg_results.append(
                {
                    "leg_index": i,
                    "option_type": opt_type,
                    "strike": strike,
                    "expiry_days": expiry_days,
                    "quantity": quantity,
                    "price_per_contract": round(price, 2),
                    "total_value": round(leg_premium, 2),
                    "delta": round(greeks.get("delta", 0), 4),
                }
            )

        # Determine structure type and estimate max profit/loss
        is_debit = total_premium > 0

        # Simplified max profit/loss calculation
        strikes = sorted([leg["strike"] for leg in legs])

        if is_debit:
            max_loss = -abs(total_premium)
            max_profit = None  # Potentially unlimited for naked calls
        else:
            max_profit = abs(total_premium)
            max_loss = None  # Need more analysis for spreads

        # For defined risk spreads, calculate actual max loss
        if len(legs) >= 2:
            strike_width = max(strikes) - min(strikes)
            if strike_width > 0:
                if is_debit:
                    max_profit = strike_width * 100 - abs(total_premium)
                else:
                    max_loss = -(strike_width * 100 - abs(total_premium))

        return {
            "underlying_price": underlying_price,
            "total_premium": round(total_premium, 2),
            "is_debit": is_debit,
            "leg_prices": leg_results,
            "net_greeks": {
                "delta": round(net_delta, 2),
                "gamma": round(net_gamma, 4),
                "theta": round(net_theta, 2),
                "vega": round(net_vega, 2),
                "rho": round(net_rho, 2),
            },
            "risk_profile": {
                "max_profit": round(max_profit, 2) if max_profit else "unlimited",
                "max_loss": round(max_loss, 2) if max_loss else "undefined",
                "risk_reward_ratio": (
                    round(abs(max_profit / max_loss), 2)
                    if max_profit and max_loss and max_loss != 0
                    else None
                ),
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def score_trade_structure(
    structure_spec: Dict[str, Any],
    vol_surface: Optional[Dict[str, Any]] = None,
    market_regime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score an options structure for trade quality.

    Unified scoring model combining:
    - Expected value based on structure analysis
    - Convexity score (gamma/theta ratio)
    - Vol surface alignment
    - Regime suitability

    Args:
        structure_spec: Structure specification with legs and underlying
        vol_surface: Optional SABR surface for vol edge detection
        market_regime: Optional regime ("bull", "bear", "sideways")

    Returns:
        Dictionary with:
            - total_score: Overall score 0-100
            - component_scores: Individual scoring factors
            - recommendation: "strong_buy", "buy", "neutral", "avoid"
            - risk_flags: Any concerns identified
    """
    from quantcore.options.adapters.quantsbin_adapter import analyze_structure_quantsbin

    try:
        # Analyze structure
        analysis = analyze_structure_quantsbin(structure_spec)

        if "error" in analysis:
            return {"error": analysis["error"]}

        greeks = analysis.get("greeks", {})
        max_profit = analysis.get("max_profit", 0)
        max_loss = analysis.get("max_loss", 0)
        is_defined_risk = analysis.get("is_defined_risk", False)

        scores = {}
        risk_flags = []

        # 1. Risk/Reward Score (0-25)
        if max_loss and max_loss != 0:
            rr_ratio = abs(max_profit / max_loss) if max_profit else 0
            if rr_ratio >= 3:
                scores["risk_reward"] = 25
            elif rr_ratio >= 2:
                scores["risk_reward"] = 20
            elif rr_ratio >= 1:
                scores["risk_reward"] = 15
            elif rr_ratio >= 0.5:
                scores["risk_reward"] = 10
            else:
                scores["risk_reward"] = 5
                risk_flags.append("Poor risk/reward ratio")
        else:
            scores["risk_reward"] = 10
            if not is_defined_risk:
                risk_flags.append("Undefined risk structure")

        # 2. Convexity Score (0-25) - gamma/theta tradeoff
        gamma = abs(greeks.get("gamma", 0))
        theta = abs(greeks.get("theta", 0))

        if theta > 0:
            convexity = gamma / theta
            if convexity >= 0.5:
                scores["convexity"] = 25
            elif convexity >= 0.2:
                scores["convexity"] = 20
            elif convexity >= 0.1:
                scores["convexity"] = 15
            else:
                scores["convexity"] = 10
        else:
            scores["convexity"] = 15  # No theta decay

        # 3. Delta Alignment Score (0-25)
        delta = greeks.get("delta", 0)

        if market_regime:
            if market_regime == "bull" and delta > 0:
                scores["regime_alignment"] = min(25, 15 + abs(delta) * 10)
            elif market_regime == "bear" and delta < 0:
                scores["regime_alignment"] = min(25, 15 + abs(delta) * 10)
            elif market_regime == "sideways" and abs(delta) < 20:
                scores["regime_alignment"] = 25
            else:
                scores["regime_alignment"] = 10
                risk_flags.append("Delta misaligned with regime")
        else:
            scores["regime_alignment"] = 15  # Neutral without regime info

        # 4. Probability Score (0-25)
        pop = analysis.get("probability_of_profit")
        if pop:
            if pop >= 70:
                scores["probability"] = 25
            elif pop >= 50:
                scores["probability"] = 20
            elif pop >= 35:
                scores["probability"] = 15
            else:
                scores["probability"] = 10
                risk_flags.append("Low probability of profit")
        else:
            scores["probability"] = 15

        # Calculate total score
        total_score = sum(scores.values())

        # Determine recommendation
        if total_score >= 85:
            recommendation = "strong_buy"
        elif total_score >= 70:
            recommendation = "buy"
        elif total_score >= 50:
            recommendation = "neutral"
        else:
            recommendation = "avoid"

        return {
            "total_score": total_score,
            "max_score": 100,
            "component_scores": scores,
            "recommendation": recommendation,
            "risk_flags": risk_flags,
            "structure_type": analysis.get("structure_type", "unknown"),
            "analysis_summary": {
                "max_profit": max_profit,
                "max_loss": max_loss,
                "delta": delta,
                "is_defined_risk": is_defined_risk,
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def simulate_trade_outcome(
    trade_template: Dict[str, Any],
    num_scenarios: int = 1000,
    holding_days: int = 30,
    vol_shock_range: float = 0.10,
) -> Dict[str, Any]:
    """
    Simulate trade P&L distribution using Monte Carlo.

    Runs scenarios with:
    - Price paths based on historical vol
    - Vol surface shifts
    - Time decay effects

    Args:
        trade_template: Trade template from generate_trade_template
        num_scenarios: Number of Monte Carlo paths
        holding_days: Days to simulate holding
        vol_shock_range: Max vol change (+/- as fraction)

    Returns:
        Dictionary with:
            - pnl_distribution: Percentile P&Ls
            - expected_pnl: Mean P&L
            - probability_profit: % of winning scenarios
            - var_95: 95% Value at Risk
            - scenario_stats: Summary statistics
    """
    from quantcore.options.engine import price_option_dispatch

    try:
        legs = trade_template.get("legs", [])
        underlying_price = trade_template.get("underlying_price")

        if not legs or not underlying_price:
            return {
                "error": "Invalid trade template - missing legs or underlying_price"
            }

        # Entry premium (current value)
        entry_result = await compute_multi_leg_price(
            legs=legs,
            underlying_price=underlying_price,
        )
        entry_premium = entry_result.get("total_premium", 0)

        # Simulate scenarios
        np.random.seed(42)

        # Base parameters
        base_vol = legs[0].get("iv", 0.25)
        annual_vol = base_vol
        daily_vol = annual_vol / np.sqrt(252)

        pnl_results = []

        for _ in range(num_scenarios):
            # Simulate price move
            price_return = np.random.normal(0, daily_vol * np.sqrt(holding_days))
            final_price = underlying_price * np.exp(price_return)

            # Simulate vol change
            vol_change = np.random.uniform(-vol_shock_range, vol_shock_range)
            new_vol = max(0.05, base_vol * (1 + vol_change))

            # Update legs with new time to expiry
            new_legs = []
            for leg in legs:
                new_leg = leg.copy()
                new_leg["expiry_days"] = max(
                    1, leg.get("expiry_days", 30) - holding_days
                )
                new_leg["iv"] = new_vol
                new_legs.append(new_leg)

            # Calculate exit value
            exit_result = await compute_multi_leg_price(
                legs=new_legs,
                underlying_price=final_price,
            )
            exit_premium = exit_result.get("total_premium", 0)

            # P&L = exit value - entry value
            pnl = exit_premium - entry_premium
            pnl_results.append(pnl)

        pnl_array = np.array(pnl_results)

        # Calculate statistics
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pnl_percentiles = {
            f"p{p}": round(float(np.percentile(pnl_array, p)), 2) for p in percentiles
        }

        return {
            "num_scenarios": num_scenarios,
            "holding_days": holding_days,
            "entry_premium": round(entry_premium, 2),
            "expected_pnl": round(float(np.mean(pnl_array)), 2),
            "median_pnl": round(float(np.median(pnl_array)), 2),
            "std_pnl": round(float(np.std(pnl_array)), 2),
            "probability_profit": round(
                float(np.sum(pnl_array > 0) / len(pnl_array) * 100), 1
            ),
            "var_95": round(float(np.percentile(pnl_array, 5)), 2),  # 5th percentile
            "cvar_95": round(
                float(np.mean(pnl_array[pnl_array <= np.percentile(pnl_array, 5)])), 2
            ),
            "max_profit_scenario": round(float(np.max(pnl_array)), 2),
            "max_loss_scenario": round(float(np.min(pnl_array)), 2),
            "pnl_percentiles": pnl_percentiles,
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def compute_quantagent_features(
    symbol: str,
    timeframe: str = "daily",
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
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
    from quantcore.data.storage import DataStore
    from quantcore.features.quantagents_pattern import QuantAgentsPatternFeatures
    from quantcore.features.quantagents_trend import QuantAgentsTrendFeatures

    store = DataStore()
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


# =============================================================================
# RESEARCH TOOLS (NEW)
# =============================================================================


@mcp.tool()
async def run_walkforward(
    symbol: str,
    timeframe: str = "daily",
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    expanding: bool = True,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run walk-forward validation for a trading signal.

    Walk-forward validation is the gold standard for evaluating trading strategies.
    It respects temporal ordering and prevents lookahead bias.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe ("daily", "1h", "4h")
        n_splits: Number of walk-forward folds
        test_size: Size of each test period (in bars)
        min_train_size: Minimum training set size
        expanding: If True, training window expands; if False, rolls
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with fold results, OOS performance, and stability metrics
    """
    from quantcore.data.storage import DataStore
    from quantcore.research.walkforward import WalkForwardValidator, WalkForwardFold

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        required_size = min_train_size + n_splits * test_size
        if len(df) < required_size:
            return {
                "error": f"Insufficient data: need {required_size} bars, have {len(df)}"
            }

        # Initialize validator
        validator = WalkForwardValidator(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
            gap=1,  # 1 bar embargo
            expanding=expanding,
        )

        # Collect fold info
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(validator.split(df)):
            folds.append(
                {
                    "fold_id": fold_idx + 1,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_start": str(df.index[train_idx[0]].date()),
                    "train_end": str(df.index[train_idx[-1]].date()),
                    "test_start": str(df.index[test_idx[0]].date()),
                    "test_end": str(df.index[test_idx[-1]].date()),
                }
            )

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "n_splits": n_splits,
            "test_size": test_size,
            "min_train_size": min_train_size,
            "expanding": expanding,
            "folds": folds,
            "total_bars": len(df),
            "data_start": str(df.index[0].date()),
            "data_end": str(df.index[-1].date()),
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def validate_signal(
    signal: List[float],
    returns: List[float],
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Run comprehensive signal validation suite.

    Performs statistical tests to validate a trading signal:
    - ADF stationarity test
    - Information Coefficient (IC) analysis
    - Lagged cross-correlations
    - Harvey-Liu multiple testing correction

    Args:
        signal: Signal values (same length as returns)
        returns: Forward returns
        significance_level: Significance level for hypothesis tests

    Returns:
        Dictionary with test results and recommendations
    """
    from quantcore.research.stat_tests import (
        adf_test,
        lagged_cross_correlation,
        harvey_liu_correction,
        information_coefficient_test,
    )

    try:
        signal_series = pd.Series(signal)
        returns_series = pd.Series(returns)

        if len(signal_series) != len(returns_series):
            return {"error": "Signal and returns must have same length"}

        if len(signal_series) < 30:
            return {"error": "Need at least 30 observations"}

        # ADF test on signal
        adf_result = adf_test(signal_series, significance_level=significance_level)

        # IC analysis
        ic_result = information_coefficient_test(signal_series, returns_series)

        # Lagged correlations
        lag_corrs = lagged_cross_correlation(signal_series, returns_series, max_lag=10)

        # Prepare results
        results = {
            "sample_size": len(signal_series),
            "stationarity": {
                "adf_statistic": (
                    float(adf_result.statistic)
                    if not np.isnan(adf_result.statistic)
                    else None
                ),
                "p_value": float(adf_result.p_value),
                "is_stationary": adf_result.is_significant,
                "interpretation": adf_result.additional_info.get(
                    "interpretation", "unknown"
                ),
            },
            "information_coefficient": {
                "ic": float(ic_result.statistic),
                "t_statistic": (
                    float(ic_result.additional_info.get("t_stat", 0))
                    if ic_result.additional_info
                    else 0
                ),
                "is_significant": ic_result.is_significant,
            },
            "lagged_correlations": {
                str(k): float(v) if not np.isnan(v) else None
                for k, v in lag_corrs.items()
            },
            "recommendations": [],
        }

        # Add recommendations
        if not adf_result.is_significant:
            results["recommendations"].append(
                "Signal is non-stationary - consider differencing or detrending"
            )

        if not ic_result.is_significant:
            results["recommendations"].append(
                "IC not significant - signal may have weak predictive power"
            )
        else:
            results["recommendations"].append(
                f"IC is significant at {significance_level} level"
            )

        # Check for decay pattern
        if lag_corrs:
            lag_1 = lag_corrs.get(1, 0)
            lag_5 = lag_corrs.get(5, 0)
            if lag_1 > 0 and lag_5 < lag_1 * 0.5:
                results["recommendations"].append(
                    "Signal shows alpha decay - consider shorter holding periods"
                )

        return results

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def diagnose_signal(
    signal: List[float],
    returns: List[float],
    cost_bps: float = 5.0,
) -> Dict[str, Any]:
    """
    Run comprehensive signal diagnostics.

    Provides detailed analysis of signal quality including:
    - IC and IC Information Ratio
    - Alpha decay analysis
    - Turnover and holding period
    - Cost-adjusted performance

    Args:
        signal: Position signal values
        returns: Return series
        cost_bps: Transaction cost in basis points

    Returns:
        Dictionary with comprehensive signal diagnostics
    """
    from quantcore.research.quant_metrics import run_signal_diagnostics

    try:
        signal_series = pd.Series(signal)
        returns_series = pd.Series(returns)

        if len(signal_series) != len(returns_series):
            return {"error": "Signal and returns must have same length"}

        if len(signal_series) < 50:
            return {"error": "Need at least 50 observations for diagnostics"}

        # Run diagnostics
        report = run_signal_diagnostics(
            signal=signal_series,
            returns=returns_series,
            cost_bps=cost_bps,
        )

        return report.to_dict()

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def detect_leakage(
    symbol: str,
    timeframe: str = "daily",
    feature_columns: Optional[List[str]] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect data leakage and lookahead bias in features.

    Checks for:
    - Feature lookahead: Features computed using future data
    - Label leakage: Labels containing future information
    - Suspicious correlations indicating leakage
    - Temporal alignment issues

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        feature_columns: Specific feature columns to check (None = all)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        LeakageReport with findings, severity, and recommendations
    """
    from quantcore.data.storage import DataStore
    from quantcore.research.leak_diagnostics import LeakageDiagnostics
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < 100:
            return {"error": "Need at least 100 bars for leakage detection"}

        # Compute features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=False,
            include_technical_indicators=True,
        )
        features = factory.compute_features(df, tf)

        if feature_columns:
            features = features[[c for c in feature_columns if c in features.columns]]

        # Compute returns and labels
        returns = df["close"].pct_change()
        labels = (returns.shift(-1) > 0).astype(int)  # Simple forward return label

        # Run diagnostics
        diagnostics = LeakageDiagnostics()
        report = diagnostics.run_full_diagnostics(
            features=features,
            labels=labels,
            prices=df["close"],
            returns=returns,
        )

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "features_checked": len(features.columns),
            "has_leakage": report.has_leakage,
            "severity": report.severity,
            "issues": report.issues[:10],  # Limit to top 10 issues
            "issue_count": len(report.issues),
            "recommendations": report.recommendations,
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# RISK TOOLS (NEW)
# =============================================================================


@mcp.tool()
async def compute_var(
    returns: List[float],
    confidence_levels: List[float] = [0.95, 0.99],
    method: str = "historical",
    horizon_days: int = 1,
) -> Dict[str, Any]:
    """
    Compute Value at Risk (VaR) and Expected Shortfall (CVaR).

    Supports multiple calculation methods:
    - Historical: Uses empirical distribution of returns
    - Parametric: Assumes normal distribution
    - Monte Carlo: Simulates future returns

    Args:
        returns: Historical returns series
        confidence_levels: VaR confidence levels (e.g., [0.95, 0.99])
        method: Calculation method ("historical", "parametric", "monte_carlo")
        horizon_days: VaR horizon in days

    Returns:
        Dictionary with VaR, CVaR, and distribution statistics
    """
    from scipy.stats import norm

    try:
        returns_arr = np.array(returns)

        if len(returns_arr) < 30:
            return {"error": "Need at least 30 returns for VaR calculation"}

        # Scale returns to horizon
        if horizon_days > 1:
            returns_arr = returns_arr * np.sqrt(horizon_days)

        result = {
            "method": method,
            "horizon_days": horizon_days,
            "sample_size": len(returns_arr),
            "var": {},
            "cvar": {},
        }

        mean = np.mean(returns_arr)
        std = np.std(returns_arr)

        for conf in confidence_levels:
            alpha = 1 - conf
            conf_str = f"{int(conf * 100)}"

            if method == "historical":
                var = -np.percentile(returns_arr, alpha * 100)
                # CVaR = average of returns below VaR
                cvar = -np.mean(returns_arr[returns_arr <= -var])

            elif method == "parametric":
                z_score = norm.ppf(alpha)
                var = -(mean + z_score * std)
                # Parametric CVaR
                cvar = -(mean - std * norm.pdf(z_score) / alpha)

            elif method == "monte_carlo":
                # Simulate 10000 returns
                simulated = np.random.normal(mean, std, 10000)
                var = -np.percentile(simulated, alpha * 100)
                cvar = -np.mean(simulated[simulated <= -var])

            else:
                return {"error": f"Unknown method: {method}"}

            result["var"][conf_str] = round(
                float(var) * 100, 4
            )  # Convert to percentage
            result["cvar"][conf_str] = round(float(cvar) * 100, 4)

        result["statistics"] = {
            "mean_return": round(float(mean) * 100, 4),
            "volatility": round(float(std) * 100, 4),
            "skewness": round(float(pd.Series(returns_arr).skew()), 4),
            "kurtosis": round(float(pd.Series(returns_arr).kurtosis()), 4),
            "max_loss": round(float(-np.min(returns_arr)) * 100, 4),
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def stress_test_portfolio(
    positions: List[Dict[str, Any]],
    scenarios: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run stress tests on an options portfolio.

    Tests portfolio against predefined historical scenarios:
    - 2008 Lehman: -40% price, +80% vol
    - 2020 COVID: -35% price, +100% vol
    - 2018 Volmageddon: -5% price, +150% vol
    - And more...

    Args:
        positions: List of positions, each with:
            - symbol: Underlying symbol
            - option_type: "call" or "put"
            - strike: Strike price
            - expiry_days: Days to expiration
            - quantity: Number of contracts
            - current_price: Current option price
        scenarios: Specific scenarios to test (None = all)

    Returns:
        Dictionary with P&L under each scenario
    """
    from quantcore.risk.stress_testing import STRESS_SCENARIOS, PortfolioStressTester
    from quantcore.options.models import OptionsPosition, OptionType

    try:
        if not positions:
            return {"error": "No positions provided"}

        # Convert to OptionsPosition objects
        opt_positions = []
        for pos in positions:
            opt_type = (
                OptionType.CALL
                if pos.get("option_type", "call").lower() == "call"
                else OptionType.PUT
            )
            opt_positions.append(
                OptionsPosition(
                    symbol=pos.get("symbol", "SPY"),
                    option_type=opt_type,
                    strike=float(pos.get("strike", 100)),
                    expiry_days=int(pos.get("expiry_days", 30)),
                    quantity=int(pos.get("quantity", 1)),
                    entry_price=float(pos.get("current_price", 5)),
                    current_price=float(pos.get("current_price", 5)),
                    underlying_price=float(pos.get("underlying_price", 100)),
                    iv=float(pos.get("iv", 0.25)),
                )
            )

        # Select scenarios
        if scenarios:
            test_scenarios = {
                k: v for k, v in STRESS_SCENARIOS.items() if k in scenarios
            }
        else:
            test_scenarios = STRESS_SCENARIOS

        # Run stress tests
        results = []
        for scenario_name, (price_shock, vol_shock) in test_scenarios.items():
            total_pnl = 0
            position_pnls = []

            for pos in opt_positions:
                # Apply shocks
                new_underlying = pos.underlying_price * (1 + price_shock)
                new_vol = pos.iv * (1 + vol_shock)

                # Recalculate option price (simplified)
                from quantcore.options.pricing import black_scholes_price

                new_price = black_scholes_price(
                    S=new_underlying,
                    K=pos.strike,
                    T=max(pos.expiry_days / 365, 0.001),
                    sigma=new_vol,
                    r=0.05,
                    q=0,
                    option_type="call" if pos.option_type == OptionType.CALL else "put",
                )

                pnl = (new_price - pos.current_price) * pos.quantity * 100
                total_pnl += pnl
                position_pnls.append(
                    {
                        "symbol": pos.symbol,
                        "type": pos.option_type.value,
                        "strike": pos.strike,
                        "pnl": round(pnl, 2),
                    }
                )

            results.append(
                {
                    "scenario": scenario_name,
                    "price_shock_pct": round(price_shock * 100, 1),
                    "vol_shock_pct": round(vol_shock * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "position_pnls": position_pnls,
                }
            )

        # Sort by worst case
        results.sort(key=lambda x: x["total_pnl"])

        return {
            "portfolio_size": len(positions),
            "scenarios_tested": len(results),
            "worst_case": results[0] if results else None,
            "best_case": results[-1] if results else None,
            "all_scenarios": results,
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def check_risk_limits(
    equity: float,
    daily_pnl: float,
    open_trades: int,
    open_exposure_pct: float,
    drawdown_pct: float,
    max_daily_loss_pct: float = 2.0,
    max_drawdown_pct: float = 10.0,
    max_concurrent_trades: int = 5,
    max_exposure_pct: float = 80.0,
) -> Dict[str, Any]:
    """
    Check current risk state against limits.

    Evaluates:
    - Daily P&L limits
    - Drawdown limits
    - Position count limits
    - Exposure limits

    Args:
        equity: Current equity
        daily_pnl: Today's P&L
        open_trades: Number of open positions
        open_exposure_pct: Current exposure as % of equity
        drawdown_pct: Current drawdown from peak
        max_daily_loss_pct: Maximum daily loss allowed (%)
        max_drawdown_pct: Maximum drawdown allowed (%)
        max_concurrent_trades: Maximum open positions
        max_exposure_pct: Maximum exposure allowed (%)

    Returns:
        RiskState with status, breaches, and recommendations
    """
    from quantcore.risk.controls import RiskStatus

    try:
        messages = []
        breaches = []
        status = "NORMAL"

        daily_loss_pct = (daily_pnl / equity) * 100 if equity > 0 else 0

        # Check daily loss
        if abs(daily_loss_pct) >= max_daily_loss_pct:
            breaches.append("daily_loss")
            messages.append(
                f"Daily loss limit breached: {daily_loss_pct:.2f}% >= {max_daily_loss_pct}%"
            )
            status = "HALTED"
        elif abs(daily_loss_pct) >= max_daily_loss_pct * 0.8:
            messages.append(f"Approaching daily loss limit: {daily_loss_pct:.2f}%")
            if status != "HALTED":
                status = "CAUTION"

        # Check drawdown
        if abs(drawdown_pct) >= max_drawdown_pct:
            breaches.append("drawdown")
            messages.append(
                f"Drawdown limit breached: {drawdown_pct:.2f}% >= {max_drawdown_pct}%"
            )
            status = "HALTED"
        elif abs(drawdown_pct) >= max_drawdown_pct * 0.8:
            messages.append(f"Approaching drawdown limit: {drawdown_pct:.2f}%")
            if status not in ["HALTED"]:
                status = "CAUTION"

        # Check position count
        if open_trades >= max_concurrent_trades:
            breaches.append("position_count")
            messages.append(
                f"Position limit reached: {open_trades} >= {max_concurrent_trades}"
            )
            if status not in ["HALTED"]:
                status = "RESTRICTED"

        # Check exposure
        if open_exposure_pct >= max_exposure_pct:
            breaches.append("exposure")
            messages.append(
                f"Exposure limit breached: {open_exposure_pct:.2f}% >= {max_exposure_pct}%"
            )
            if status not in ["HALTED"]:
                status = "RESTRICTED"

        can_trade = status in ["NORMAL", "CAUTION"]
        size_multiplier = (
            1.0 if status == "NORMAL" else 0.5 if status == "CAUTION" else 0.0
        )

        return {
            "status": status,
            "can_trade": can_trade,
            "size_multiplier": size_multiplier,
            "breaches": breaches,
            "messages": messages,
            "current_state": {
                "equity": equity,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": round(daily_loss_pct, 2),
                "drawdown_pct": drawdown_pct,
                "open_trades": open_trades,
                "open_exposure_pct": open_exposure_pct,
            },
            "limits": {
                "max_daily_loss_pct": max_daily_loss_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "max_concurrent_trades": max_concurrent_trades,
                "max_exposure_pct": max_exposure_pct,
            },
        }

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# MICROSTRUCTURE TOOLS (NEW)
# =============================================================================


@mcp.tool()
async def analyze_liquidity(
    symbol: str,
    timeframe: str = "daily",
    window: int = 20,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze liquidity characteristics of a symbol.

    Computes:
    - Bid-ask spread estimates (Corwin-Schultz, Roll)
    - Volume analysis vs historical average
    - Liquidity score (0-1)

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        window: Lookback window for calculations
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        LiquidityFeatures with spread estimates and scores
    """
    from quantcore.data.storage import DataStore
    from quantcore.microstructure.liquidity import LiquidityAnalyzer

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < window + 10:
            return {"error": f"Need at least {window + 10} bars"}

        # Analyze liquidity
        analyzer = LiquidityAnalyzer(spread_threshold_bps=30, min_volume_ratio=0.5)
        features = analyzer.analyze(df, window=window)

        # Get latest features
        latest = features.iloc[-1] if not features.empty else None

        if latest is None:
            return {"error": "Could not compute liquidity features"}

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "window": window,
            "timestamp": str(df.index[-1]),
            "spread_estimates": {
                "corwin_schultz_bps": round(float(latest.get("cs_spread_bps", 0)), 2),
                "roll_spread_bps": round(float(latest.get("roll_spread_bps", 0)), 2),
                "combined_spread_bps": round(
                    float(latest.get("estimated_spread_bps", 0)), 2
                ),
            },
            "volume": {
                "current": int(df["volume"].iloc[-1]),
                "avg_20d": int(df["volume"].rolling(20).mean().iloc[-1]),
                "vs_avg_ratio": round(float(latest.get("volume_vs_avg", 1)), 2),
            },
            "liquidity_score": round(float(latest.get("liquidity_score", 0.5)), 2),
            "is_liquid": bool(latest.get("is_liquid", True)),
            "recommendations": [
                (
                    "Liquid"
                    if latest.get("is_liquid", True)
                    else "Low liquidity - widen stops"
                ),
                f"Estimated round-trip cost: {latest.get('estimated_spread_bps', 0) * 2:.1f} bps",
            ],
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def analyze_volume_profile(
    symbol: str,
    timeframe: str = "daily",
    lookback_days: int = 20,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze volume profile and VWAP levels.

    Identifies:
    - Volume-weighted average price (VWAP)
    - High volume nodes (support/resistance)
    - Intraday volume patterns

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        lookback_days: Days to analyze
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Volume profile with VWAP levels and key nodes
    """
    from quantcore.data.storage import DataStore
    from quantcore.microstructure.volume_profile import VolumeProfileAnalyzer

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        # Use last N days
        df = df.tail(lookback_days * (1 if tf == Timeframe.D1 else 7))

        if len(df) < 10:
            return {"error": "Insufficient data for volume profile"}

        # Calculate VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        current_price = float(df["close"].iloc[-1])
        current_vwap = float(vwap.iloc[-1])

        # Volume by price analysis (simplified)
        price_range = df["high"].max() - df["low"].min()
        n_bins = 10
        bin_size = price_range / n_bins

        volume_nodes = []
        for i in range(n_bins):
            bin_low = df["low"].min() + i * bin_size
            bin_high = bin_low + bin_size
            mask = (df["close"] >= bin_low) & (df["close"] < bin_high)
            vol = df.loc[mask, "volume"].sum()
            volume_nodes.append(
                {
                    "price_low": round(bin_low, 2),
                    "price_high": round(bin_high, 2),
                    "volume": int(vol),
                }
            )

        # Find high volume nodes (potential S/R)
        volume_nodes.sort(key=lambda x: x["volume"], reverse=True)
        high_volume_levels = volume_nodes[:3]

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "lookback_days": lookback_days,
            "vwap": {
                "current": round(current_vwap, 2),
                "price_vs_vwap_pct": round((current_price / current_vwap - 1) * 100, 2),
                "above_vwap": current_price > current_vwap,
            },
            "high_volume_nodes": high_volume_levels,
            "volume_stats": {
                "total_volume": int(df["volume"].sum()),
                "avg_daily_volume": int(df["volume"].mean()),
                "highest_volume_day": (
                    str(df["volume"].idxmax().date())
                    if hasattr(df["volume"].idxmax(), "date")
                    else str(df["volume"].idxmax())
                ),
            },
            "current_price": current_price,
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def get_trading_calendar(
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get market trading calendar information.

    Returns:
    - Market hours
    - Upcoming holidays
    - Early close days
    - Current market status

    Args:
        year: Year to query (default: current year)
        month: Month to query (default: current month)

    Returns:
        Calendar with holidays and trading hours
    """
    from datetime import datetime, time as dt_time
    from quantcore.microstructure.events import TradingCalendar

    try:
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        # Standard US market hours
        market_hours = {
            "regular_open": "09:30",
            "regular_close": "16:00",
            "pre_market_open": "04:00",
            "pre_market_close": "09:30",
            "after_hours_open": "16:00",
            "after_hours_close": "20:00",
            "timezone": "America/New_York",
        }

        # US market holidays 2024-2025
        holidays = {
            2024: [
                ("2024-01-01", "New Year's Day"),
                ("2024-01-15", "MLK Day"),
                ("2024-02-19", "Presidents Day"),
                ("2024-03-29", "Good Friday"),
                ("2024-05-27", "Memorial Day"),
                ("2024-06-19", "Juneteenth"),
                ("2024-07-04", "Independence Day"),
                ("2024-09-02", "Labor Day"),
                ("2024-11-28", "Thanksgiving"),
                ("2024-12-25", "Christmas"),
            ],
            2025: [
                ("2025-01-01", "New Year's Day"),
                ("2025-01-20", "MLK Day"),
                ("2025-02-17", "Presidents Day"),
                ("2025-04-18", "Good Friday"),
                ("2025-05-26", "Memorial Day"),
                ("2025-06-19", "Juneteenth"),
                ("2025-07-04", "Independence Day"),
                ("2025-09-01", "Labor Day"),
                ("2025-11-27", "Thanksgiving"),
                ("2025-12-25", "Christmas"),
            ],
        }

        # Early close days (1pm ET)
        early_closes = {
            2024: [
                ("2024-07-03", "Day before Independence Day"),
                ("2024-11-29", "Day after Thanksgiving"),
                ("2024-12-24", "Christmas Eve"),
            ],
            2025: [
                ("2025-07-03", "Day before Independence Day"),
                ("2025-11-28", "Day after Thanksgiving"),
                ("2025-12-24", "Christmas Eve"),
            ],
        }

        # Check if market is open now (simplified)
        is_weekday = now.weekday() < 5
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()

        is_market_hours = is_weekday and market_open <= current_time <= market_close

        return {
            "query": {"year": year, "month": month},
            "market_hours": market_hours,
            "holidays": holidays.get(year, []),
            "early_closes": early_closes.get(year, []),
            "current_status": {
                "is_market_hours": is_market_hours,
                "current_time": str(now),
                "next_open": "09:30 ET" if not is_market_hours else "Market is open",
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_event_calendar(
    symbol: Optional[str] = None,
    days_ahead: int = 14,
) -> Dict[str, Any]:
    """
    Get upcoming economic and earnings events.

    Returns events that may affect trading:
    - FOMC meetings
    - CPI/PPI releases
    - NFP (Non-Farm Payrolls)
    - Earnings dates (if symbol provided)
    - Options expiration

    Args:
        symbol: Stock symbol for earnings (optional)
        days_ahead: Days to look ahead

    Returns:
        Calendar of upcoming events with impact ratings
    """
    from datetime import datetime, timedelta
    from quantcore.microstructure.events import EventType

    try:
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)

        # Static economic calendar (would be fetched from API in production)
        # These are example dates for illustration
        economic_events = [
            {
                "date": "2024-12-18",
                "event": "FOMC Decision",
                "type": "FOMC",
                "impact": "HIGH",
            },
            {
                "date": "2024-12-20",
                "event": "PCE Inflation",
                "type": "INFLATION",
                "impact": "HIGH",
            },
            {
                "date": "2025-01-10",
                "event": "NFP Report",
                "type": "NFP",
                "impact": "HIGH",
            },
            {
                "date": "2025-01-15",
                "event": "CPI Release",
                "type": "CPI",
                "impact": "HIGH",
            },
            {
                "date": "2025-01-29",
                "event": "FOMC Decision",
                "type": "FOMC",
                "impact": "HIGH",
            },
        ]

        # Options expiration dates
        opex_events = []
        for month_offset in range(2):
            # Third Friday of each month
            first_day = (
                datetime(now.year, now.month + month_offset, 1)
                if now.month + month_offset <= 12
                else datetime(now.year + 1, (now.month + month_offset) % 12, 1)
            )
            # Find third Friday
            day = 1
            fridays = 0
            while fridays < 3:
                test_date = first_day.replace(day=day)
                if test_date.weekday() == 4:  # Friday
                    fridays += 1
                    if fridays == 3:
                        opex_events.append(
                            {
                                "date": test_date.strftime("%Y-%m-%d"),
                                "event": "Monthly Options Expiration",
                                "type": "OPEX",
                                "impact": "MEDIUM",
                            }
                        )
                day += 1

        # Combine and filter by date range
        all_events = economic_events + opex_events
        filtered_events = [
            e
            for e in all_events
            if now.strftime("%Y-%m-%d") <= e["date"] <= end_date.strftime("%Y-%m-%d")
        ]

        # Sort by date
        filtered_events.sort(key=lambda x: x["date"])

        result = {
            "query": {
                "symbol": symbol,
                "days_ahead": days_ahead,
                "start_date": now.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            },
            "economic_events": filtered_events,
            "event_count": len(filtered_events),
            "high_impact_count": len(
                [e for e in filtered_events if e["impact"] == "HIGH"]
            ),
        }

        # Add blackout recommendations
        blackout_dates = [e["date"] for e in filtered_events if e["impact"] == "HIGH"]
        result["blackout_dates"] = blackout_dates
        result["recommendation"] = (
            "Avoid new positions on blackout dates"
            if blackout_dates
            else "No high-impact events in range"
        )

        return result

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# EXECUTION TOOLS (NEW)
# =============================================================================


@mcp.tool()
async def execute_paper_trade(
    symbol: str,
    direction: str,
    quantity: int,
    structure_type: str = "single",
    confidence: float = 0.5,
    reason: str = "",
) -> Dict[str, Any]:
    """
    Execute a paper trade for testing strategies.

    Simulates realistic order execution with:
    - Slippage estimation
    - Fill simulation
    - Position tracking

    Note: This is internal simulation. For broker paper trading,
    use Alpaca MCP or eTrade sandbox.

    Args:
        symbol: Stock symbol
        direction: "long" or "short"
        quantity: Number of shares/contracts
        structure_type: Trade type ("single", "vertical", "iron_condor")
        confidence: Signal confidence (0-1)
        reason: Trade rationale

    Returns:
        Order confirmation with simulated fill
    """
    from datetime import datetime
    import uuid
    from quantcore.data.storage import DataStore

    store = DataStore()

    try:
        # Get current price
        df = store.load_ohlcv(symbol, Timeframe.D1)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        current_price = float(df["close"].iloc[-1])

        # Simulate slippage (0.05% for liquid stocks)
        slippage_pct = 0.0005
        if direction.lower() == "long":
            fill_price = current_price * (1 + slippage_pct)
        else:
            fill_price = current_price * (1 - slippage_pct)

        # Generate order
        order_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now()

        order = {
            "order_id": order_id,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "structure_type": structure_type,
            "confidence": confidence,
            "reason": reason,
            "execution": {
                "status": "FILLED",
                "fill_price": round(fill_price, 2),
                "fill_timestamp": timestamp.isoformat(),
                "slippage_bps": round(slippage_pct * 10000, 1),
                "notional_value": round(fill_price * quantity, 2),
            },
            "market_context": {
                "current_price": current_price,
                "bid_estimate": round(current_price * 0.9999, 2),
                "ask_estimate": round(current_price * 1.0001, 2),
            },
        }

        return {
            "success": True,
            "order": order,
            "message": f"Paper trade executed: {direction} {quantity} {symbol} @ {fill_price:.2f}",
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def get_order_book_snapshot(
    symbol: str,
    levels: int = 5,
) -> Dict[str, Any]:
    """
    Get simulated order book snapshot.

    Returns bid/ask levels with depth for market microstructure analysis.

    Note: This is a simulated order book based on historical data.
    For live order book, use broker MCP.

    Args:
        symbol: Stock symbol
        levels: Number of price levels to show

    Returns:
        Order book with bid/ask levels, spread, and depth
    """
    from quantcore.data.storage import DataStore
    from quantcore.microstructure.order_book import OrderBook, Side, Order

    store = DataStore()

    try:
        df = store.load_ohlcv(symbol, Timeframe.D1)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Get current price and volatility
        current_price = float(df["close"].iloc[-1])
        daily_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
        avg_volume = float(df["volume"].tail(20).mean())

        # Estimate spread from high-low
        spread_estimate = daily_range / current_price * 0.1  # Simplified
        half_spread = max(spread_estimate / 2, 0.0001)

        # Generate synthetic order book
        bids = []
        asks = []

        for i in range(levels):
            # Bid side (below mid)
            bid_price = current_price * (1 - half_spread - i * 0.0001)
            bid_size = int(avg_volume / 100 * (levels - i) / levels)
            bids.append(
                {
                    "level": i + 1,
                    "price": round(bid_price, 2),
                    "size": bid_size,
                    "orders": max(1, bid_size // 100),
                }
            )

            # Ask side (above mid)
            ask_price = current_price * (1 + half_spread + i * 0.0001)
            ask_size = int(avg_volume / 100 * (levels - i) / levels)
            asks.append(
                {
                    "level": i + 1,
                    "price": round(ask_price, 2),
                    "size": ask_size,
                    "orders": max(1, ask_size // 100),
                }
            )

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        spread = best_ask - best_bid
        spread_bps = (spread / current_price) * 10000

        return {
            "symbol": symbol,
            "timestamp": str(df.index[-1]),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": round((best_bid + best_ask) / 2, 2),
            "spread": round(spread, 4),
            "spread_bps": round(spread_bps, 2),
            "bids": bids,
            "asks": asks,
            "depth": {
                "total_bid_size": sum(b["size"] for b in bids),
                "total_ask_size": sum(a["size"] for a in asks),
                "imbalance": round(
                    (sum(b["size"] for b in bids) - sum(a["size"] for a in asks))
                    / (sum(b["size"] for b in bids) + sum(a["size"] for a in asks)),
                    3,
                ),
            },
            "note": "Simulated order book based on historical data. Use broker MCP for live data.",
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# VALIDATION TOOLS (NEW)
# =============================================================================


@mcp.tool()
async def run_purged_cv(
    symbol: str,
    timeframe: str = "daily",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run purged K-Fold cross-validation.

    Implements Lopez de Prado's purged CV to prevent data leakage:
    - Purging: Removes training samples overlapping with test period
    - Embargo: Adds gap between train and test

    Essential for validating trading strategies without lookahead bias.

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        n_splits: Number of CV folds
        embargo_pct: Percentage of data to embargo after train
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        CV splits with train/test indices and temporal boundaries
    """
    from quantcore.data.storage import DataStore
    from quantcore.validation.purged_cv import PurgedKFoldCV

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < n_splits * 50:
            return {
                "error": f"Need at least {n_splits * 50} bars for {n_splits}-fold CV"
            }

        # Initialize CV
        cv = PurgedKFoldCV(
            n_splits=n_splits,
            embargo_pct=embargo_pct,
        )

        # Collect splits
        splits = []
        for fold_idx, split in enumerate(cv.split(df)):
            splits.append(
                {
                    "fold": fold_idx + 1,
                    "train_size": len(split.train_indices),
                    "test_size": len(split.test_indices),
                    "train_start": (
                        str(split.train_start.date())
                        if hasattr(split.train_start, "date")
                        else str(split.train_start)
                    ),
                    "train_end": (
                        str(split.train_end.date())
                        if hasattr(split.train_end, "date")
                        else str(split.train_end)
                    ),
                    "test_start": (
                        str(split.test_start.date())
                        if hasattr(split.test_start, "date")
                        else str(split.test_start)
                    ),
                    "test_end": (
                        str(split.test_end.date())
                        if hasattr(split.test_end, "date")
                        else str(split.test_end)
                    ),
                    "embargo_size": int(len(df) * embargo_pct),
                }
            )

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "total_bars": len(df),
            "n_splits": n_splits,
            "embargo_pct": embargo_pct,
            "splits": splits,
            "data_range": {
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
            },
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def check_lookahead_bias(
    symbol: str,
    timeframe: str = "daily",
    feature_columns: Optional[List[str]] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check for lookahead bias in features.

    Detects features that may contain future information:
    - High correlation with future returns (lag 0 or negative)
    - Perfect prediction of future events
    - Temporal misalignment

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        feature_columns: Specific columns to check (None = all)
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Report with suspect features and recommendations
    """
    from quantcore.data.storage import DataStore
    from quantcore.validation.integrity import validate_no_lookahead
    from quantcore.features.factory import MultiTimeframeFeatureFactory
    from scipy import stats

    store = DataStore()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < 100:
            return {"error": "Need at least 100 bars for lookahead detection"}

        # Compute features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=False,
            include_technical_indicators=True,
        )
        features = factory.compute_features(df, tf)

        if feature_columns:
            features = features[[c for c in feature_columns if c in features.columns]]

        # Calculate forward returns
        returns = df["close"].pct_change().shift(-1)  # Future return

        # Check each feature for lookahead
        suspect_features = []
        clean_features = []

        for col in features.columns[:50]:  # Limit to first 50 features
            feature = features[col].dropna()
            common_idx = feature.index.intersection(returns.dropna().index)

            if len(common_idx) < 30:
                continue

            # Correlation with future return at lag 0 (contemporaneous)
            corr, _ = stats.spearmanr(feature.loc[common_idx], returns.loc[common_idx])

            if abs(corr) > 0.3:  # Suspiciously high correlation
                suspect_features.append(
                    {
                        "feature": col,
                        "correlation_with_future": round(corr, 3),
                        "severity": "HIGH" if abs(corr) > 0.5 else "MEDIUM",
                        "reason": "High correlation with next-period return",
                    }
                )
            else:
                clean_features.append(col)

        has_lookahead = len(suspect_features) > 0

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "features_checked": len(features.columns),
            "has_lookahead_bias": has_lookahead,
            "suspect_features": suspect_features,
            "suspect_count": len(suspect_features),
            "clean_features_sample": clean_features[:10],
            "recommendations": [
                (
                    f"Remove or fix {len(suspect_features)} suspect features"
                    if has_lookahead
                    else "No obvious lookahead bias detected"
                ),
                "Use proper train/test splits with embargo period",
                "Verify features use only past data (shift appropriately)",
            ],
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# MCP RESOURCES
# =============================================================================


@mcp.resource("quantcore://symbols")
async def get_symbols_resource() -> str:
    """Get list of tracked symbols."""
    from quantcore.config.settings import get_settings

    settings = get_settings()
    return json.dumps(
        {
            "symbols": settings.symbols,
            "benchmark": settings.benchmark_symbol,
        }
    )


@mcp.resource("quantcore://config")
async def get_config_resource() -> str:
    """Get current platform configuration."""
    from quantcore.config.settings import get_settings

    settings = get_settings()
    return json.dumps(
        {
            "database_path": settings.database_path,
            "data_start_date": settings.data_start_date,
            "data_end_date": settings.data_end_date,
            "train_end_date": settings.train_end_date,
            "risk_per_trade_bps": settings.max_risk_per_trade_bps,
            "max_concurrent_trades": settings.max_concurrent_trades,
            "transaction_cost_bps": settings.total_transaction_cost_bps,
        }
    )


@mcp.resource("quantcore://indicators")
async def get_indicators_resource() -> str:
    """Get available indicators catalog."""
    result = await list_available_indicators()
    return json.dumps(result)


@mcp.resource("quantcore://timeframes")
async def get_timeframes_resource() -> str:
    """Get available timeframes and their parameters."""
    timeframes = {}
    for tf in Timeframe:
        params = TIMEFRAME_PARAMS[tf]
        timeframes[tf.value] = {
            "ema_fast": params.ema_fast,
            "ema_slow": params.ema_slow,
            "rsi_period": params.rsi_period,
            "atr_period": params.atr_period,
            "resample_rule": params.resample_rule,
        }
    return json.dumps({"timeframes": timeframes})


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Run the MCP server."""
    import asyncio

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    # Run server
    mcp.run()


if __name__ == "__main__":
    main()
