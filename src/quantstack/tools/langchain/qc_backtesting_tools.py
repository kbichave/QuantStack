"""Backtesting tools for LangGraph agents."""

import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import Field


@tool
async def run_backtest_template(
    symbol: str = Field(
        description="Stock ticker symbol to backtest (e.g. 'AAPL', 'QQQ').",
    ),
    strategy_type: str = Field(
        default="mean_reversion",
        description="Strategy type to simulate: 'mean_reversion', 'trend_following', or 'momentum'.",
    ),
    timeframe: str = Field(
        default="daily",
        description="Bar timeframe for the historical simulation: '1h', '4h', or 'daily'.",
    ),
    initial_capital: float = Field(
        default=100000.0,
        description="Starting portfolio capital in USD for the backtest simulation.",
    ),
    position_size_pct: float = Field(
        default=10.0,
        description="Position size as a percentage of total equity per trade.",
    ),
    stop_loss_atr: float = Field(
        default=2.0,
        description="Stop-loss distance expressed in ATR multiples (e.g. 2.0 = 2x ATR).",
    ),
    take_profit_atr: float = Field(
        default=3.0,
        description="Take-profit target expressed in ATR multiples (e.g. 3.0 = 3x ATR).",
    ),
    zscore_entry: float = Field(
        default=2.0,
        description="Z-score threshold to trigger entry for mean-reversion strategies.",
    ),
    zscore_exit: float = Field(
        default=0.5,
        description="Z-score threshold to trigger exit for mean-reversion strategies.",
    ),
    end_date: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format to cap the historical simulation window.",
    ),
) -> str:
    """Computes a full backtest of a trading strategy on historical OHLCV data, returning key performance metrics. Use when you need to evaluate a mean-reversion, trend-following, or momentum strategy on a single ticker with configurable risk parameters. Calculates Sharpe ratio, max drawdown, total return, win rate, profit factor, and per-trade P&L. Supports daily, hourly, and 4-hour timeframes. Returns JSON with equity curve summary, trade log, and aggregated out-of-sample (OOS) performance statistics for strategy validation."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_backtest_metrics(
    total_return: float = Field(
        description="Total return percentage from the backtest (e.g. 12.5 for 12.5%).",
    ),
    sharpe_ratio: float = Field(
        description="Annualized Sharpe ratio — risk-adjusted return metric from the backtest.",
    ),
    max_drawdown: float = Field(
        description="Maximum peak-to-trough drawdown percentage (e.g. -15.0 for a 15% decline).",
    ),
    win_rate: float = Field(
        description="Percentage of trades that were profitable (e.g. 55.0 for 55% win rate).",
    ),
    total_trades: int = Field(
        description="Total number of completed round-trip trades in the backtest.",
    ),
) -> str:
    """Analyzes and interprets backtest performance metrics, providing qualitative assessment and actionable recommendations. Use when you have raw backtest output and need to evaluate whether a strategy meets production deployment thresholds. Computes risk-adjusted scoring across Sharpe ratio, max drawdown, win rate, profit factor, and trade frequency. Returns JSON with metric-by-metric interpretation, overall strategy grade, and go/no-go recommendation for paper or live trading."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def run_walkforward_template(
    symbol: str = Field(
        description="Stock ticker symbol to run walk-forward analysis on (e.g. 'AAPL', 'SPY').",
    ),
    timeframe: str = Field(
        default="daily",
        description="Bar timeframe for the simulation: 'daily', '1h', or '4h'.",
    ),
    n_splits: int = Field(
        default=5,
        description="Number of walk-forward folds (splits) for the time-series cross-validation.",
    ),
    test_size: int = Field(
        default=252,
        description="Number of bars in each out-of-sample (OOS) test period per fold.",
    ),
    min_train_size: int = Field(
        default=504,
        description="Minimum number of bars in the training window before the first test fold.",
    ),
    expanding: bool = Field(
        default=True,
        description="If True, uses an expanding training window; if False, uses a rolling (fixed-size) window.",
    ),
    end_date: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format to cap the historical data window.",
    ),
) -> str:
    """Runs walk-forward validation (WFV) on a trading signal, the gold standard for out-of-sample strategy evaluation. Use when you need to assess strategy robustness across multiple temporal folds without lookahead bias. Computes per-fold train/test Sharpe ratio, drawdown, return, and win rate, then aggregates OOS stability metrics. Supports expanding and rolling training windows. Returns JSON with fold-by-fold results, aggregated OOS performance, and stability scores for strategy approval. Essential for validating backtest results before paper or live deployment."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def run_purged_cv(
    symbol: str = Field(
        description="Stock ticker symbol to validate (e.g. 'AAPL', 'TSLA').",
    ),
    timeframe: str = Field(
        default="daily",
        description="Bar timeframe for the data: 'daily', '1h', or '4h'.",
    ),
    n_splits: int = Field(
        default=5,
        description="Number of K-Fold cross-validation splits.",
    ),
    embargo_pct: float = Field(
        default=0.01,
        description="Fraction of total data to embargo (gap) between train and test sets to prevent leakage (e.g. 0.01 = 1%).",
    ),
    end_date: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format to cap the historical data window.",
    ),
) -> str:
    """Runs Lopez de Prado's purged K-Fold cross-validation to evaluate a trading strategy without data leakage or lookahead bias. Use when standard walk-forward validation is insufficient and you need rigorous purged CV with embargo periods between train and test sets. Purging removes training samples that overlap with the test period; embargo adds a temporal gap to prevent information bleed. Computes per-fold Sharpe ratio, drawdown, and return across all splits. Returns JSON with CV split boundaries, train/test indices, temporal boundaries, and aggregated out-of-sample (OOS) performance metrics."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
