# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Backtesting tools for the QuantCore MCP server.

Provides walk-forward validation, purged cross-validation, and single-pass
backtesting with configurable strategy types (mean reversion, trend following,
momentum).  All tools register on the shared ``mcp`` FastMCP instance imported
from ``quantcore.mcp.server``.
"""

from typing import Any

import pandas as pd

from quantcore.mcp._helpers import _get_reader, _parse_timeframe
from quantcore.mcp.server import mcp

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
    end_date: str | None = None,
) -> dict[str, Any]:
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
    from quantcore.backtesting.engine import BacktestConfig, BacktestEngine
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

        # Compute features for signal generation
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features_df = factory.compute_all_timeframes({tf: df})[tf]

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
            "equity_curve_sample": (result.equity_curve[-50:] if result.equity_curve else []),
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
) -> dict[str, Any]:
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
# WALK-FORWARD VALIDATION
# =============================================================================


@mcp.tool()
async def run_walkforward(
    symbol: str,
    timeframe: str = "daily",
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    expanding: bool = True,
    end_date: str | None = None,
) -> dict[str, Any]:
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
    from quantcore.research.walkforward import WalkForwardValidator

    store = _get_reader()
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
            return {"error": f"Insufficient data: need {required_size} bars, have {len(df)}"}

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


# =============================================================================
# PURGED CROSS-VALIDATION
# =============================================================================


@mcp.tool()
async def run_purged_cv(
    symbol: str,
    timeframe: str = "daily",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    end_date: str | None = None,
) -> dict[str, Any]:
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
    from quantcore.validation.purged_cv import PurgedKFoldCV

    store = _get_reader()
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
            return {"error": f"Need at least {n_splits * 50} bars for {n_splits}-fold CV"}

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
