"""Backtesting tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool
from pydantic import Field

logger = logging.getLogger(__name__)


@tool
async def run_backtest(
    strategy_id: str = Field(description="Unique strategy identifier from register_strategy to load entry/exit rules for backtesting"),
    symbol: str = Field(description="Ticker symbol to run the historical backtest on, e.g. 'AAPL' or 'SPY'"),
    initial_capital: float = Field(default=100000, description="Starting portfolio capital in USD for the backtest simulation"),
    position_size_pct: float = Field(default=0.10, description="Position size as a fraction of total capital per trade, e.g. 0.10 for 10%"),
    start_date: str | None = Field(default=None, description="Backtest start date in YYYY-MM-DD format; omit to use all available historical data"),
    end_date: str | None = Field(default=None, description="Backtest end date in YYYY-MM-DD format; omit to use data through the most recent date"),
) -> str:
    """Run a full historical backtest simulation for a registered strategy using its entry and exit rules. Use when evaluating strategy performance, validating trading signals, or comparing strategy variants. Returns JSON with total return, Sharpe ratio, max drawdown, win rate, trade count, and profit factor. Computes equity curve metrics and updates strategy status to backtested on success."""
    try:
        from quantstack.tools._shared import run_backtest_impl

        result = await run_backtest_impl(
            strategy_id=strategy_id,
            symbol=symbol,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            start_date=start_date,
            end_date=end_date,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"run_backtest({symbol}) failed: {e}")
        return json.dumps({"error": str(e)})


@tool
async def run_walkforward(
    symbol: str = Field(description="Ticker symbol to run walk-forward analysis on, e.g. 'AAPL' or 'QQQ'"),
    strategy_type: str = Field(default="mean_reversion", description="Strategy type to evaluate: 'mean_reversion', 'momentum', 'trend_following', or other registered strategy types"),
    n_splits: int = Field(default=5, description="Number of train/test walk-forward splits for out-of-sample validation"),
) -> str:
    """Run walk-forward optimization and out-of-sample validation analysis on a trading strategy. Use when testing strategy robustness, detecting overfitting, or performing rolling window cross-validation. Returns JSON with per-split performance metrics including Sharpe ratio, drawdown, and stability scores across time periods."""
    try:
        from quantstack.core.backtesting.walkforward_service import (
            run_walkforward as _run_wfv,
        )

        result = await _run_wfv(
            strategy_id=strategy_type,
            symbol=symbol,
            n_splits=n_splits,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"run_walkforward({symbol}) failed: {e}")
        return json.dumps({"error": str(e)})


@tool
async def run_backtest_options(
    symbol: str = Field(description="Underlying ticker symbol for the options backtest, e.g. 'AAPL' or 'SPY'"),
    strategy_type: str = Field(default="covered_call", description="Options strategy type to backtest: 'covered_call', 'iron_condor', 'straddle', 'put_spread', or 'call_spread'"),
) -> str:
    """Run a historical backtest simulation specifically for options strategies on an underlying symbol. Use when evaluating covered calls, iron condors, spreads, straddles, or other derivatives strategies. Returns JSON with options-specific performance metrics including premium collected, assignment rate, and risk-adjusted returns."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
