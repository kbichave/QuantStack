"""Backtesting tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def run_backtest(
    symbol: str,
    strategy_type: str = "mean_reversion",
    timeframe: str = "daily",
    initial_capital: float = 100000,
    position_size_pct: float = 10,
    stop_loss_atr: float = 2,
    take_profit_atr: float = 3,
) -> str:
    """Run a backtest on historical data to validate a trading strategy.

    Returns JSON with total return, Sharpe ratio, max drawdown,
    win rate, trade count, and equity curve.

    Args:
        symbol: Ticker symbol.
        strategy_type: Strategy type (e.g., "mean_reversion", "momentum").
        timeframe: "daily", "weekly", or "monthly".
        initial_capital: Starting capital.
        position_size_pct: Position size as % of capital.
        stop_loss_atr: Stop loss in ATR multiples.
        take_profit_atr: Take profit in ATR multiples.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "run_backtest",
        symbol=symbol,
        strategy_type=strategy_type,
        timeframe=timeframe,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        stop_loss_atr=stop_loss_atr,
        take_profit_atr=take_profit_atr,
    )
    return json.dumps(result, default=str)
