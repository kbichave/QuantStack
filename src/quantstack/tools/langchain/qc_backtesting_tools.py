"""Backtesting tools for LangGraph agents."""

import json
from typing import Optional

from langchain_core.tools import tool


@tool
async def run_backtest_template(
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
) -> str:
    """Run a backtest on historical data.

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

    Returns JSON with backtest results and metrics.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_backtest_metrics(
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    total_trades: int,
) -> str:
    """Analyze and interpret backtest metrics.

    Args:
        total_return: Total return percentage
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum peak-to-trough decline
        win_rate: Percentage of winning trades
        total_trades: Total number of trades

    Returns JSON with metric analysis and interpretation.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def run_walkforward_template(
    symbol: str,
    timeframe: str = "daily",
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    expanding: bool = True,
    end_date: Optional[str] = None,
) -> str:
    """Run walk-forward validation for a trading signal.

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

    Returns JSON with fold results, OOS performance, and stability metrics.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def run_purged_cv(
    symbol: str,
    timeframe: str = "daily",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    end_date: Optional[str] = None,
) -> str:
    """Run purged K-Fold cross-validation.

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

    Returns JSON with CV splits, train/test indices, and temporal boundaries.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
