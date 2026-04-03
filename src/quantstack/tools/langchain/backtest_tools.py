"""Backtesting tools for LangGraph agents."""

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def run_backtest(
    strategy_id: str,
    symbol: str,
    initial_capital: float = 100000,
    position_size_pct: float = 0.10,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Run a backtest for a registered strategy using its entry/exit rules.

    The strategy must already be registered (via register_strategy).
    Loads the strategy's rules from DB, generates signals, and runs
    a full backtest. Updates the strategy status to 'backtested' on success.

    Returns JSON with total return, Sharpe ratio, max drawdown,
    win rate, trade count, profit factor.

    Args:
        strategy_id: The strategy ID to backtest (from register_strategy).
        symbol: Ticker symbol to test on.
        initial_capital: Starting capital (default 100000).
        position_size_pct: Position size as fraction of capital (default 0.10 = 10%).
        start_date: Optional start date (YYYY-MM-DD). Defaults to all available data.
        end_date: Optional end date (YYYY-MM-DD).
    """
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
    symbol: str,
    strategy_type: str = "mean_reversion",
    n_splits: int = 5,
) -> str:
    """Run walk-forward analysis on a strategy.

    Args:
        symbol: Ticker symbol.
        strategy_type: Strategy type.
        n_splits: Number of walk-forward splits.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def run_backtest_options(
    symbol: str,
    strategy_type: str = "covered_call",
) -> str:
    """Run options-specific backtest.

    Args:
        symbol: Underlying symbol.
        strategy_type: Options strategy type.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
