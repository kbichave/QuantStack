"""
Simple equity backtester.

Backtests equity signals with:
- 100 shares per trade
- $0 commission
- Position changes on signal change
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Record of a single trade."""

    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl: float
    shares: int


@dataclass
class BacktestResult:
    """Results from backtesting."""

    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_pnl: float
    trades: List[TradeRecord] = field(default_factory=list)


def backtest_signals(
    signals: pd.Series,
    prices: pd.DataFrame,
    shares_per_trade: int = 100,
    initial_equity: float = 100000,
) -> BacktestResult:
    """
    Backtest signals with $0 commission and 100 shares per trade.

    Args:
        signals: Series of signals (1=LONG, -1=SHORT, 0=FLAT)
        prices: DataFrame with 'open', 'close' columns
        shares_per_trade: Number of shares per trade (default 100)
        initial_equity: Starting equity

    Returns:
        BacktestResult with all metrics
    """
    if signals.empty or prices.empty:
        return BacktestResult(
            total_pnl=0,
            total_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            num_trades=0,
            avg_trade_pnl=0,
        )

    # Align signals with prices
    signals = signals.reindex(prices.index).fillna(0).astype(int)

    equity = initial_equity
    position = 0
    entry_price = 0.0

    equity_curve = [equity]
    trades = []
    daily_returns = []
    peak_equity = equity
    max_drawdown = 0.0

    for i in range(1, len(prices)):
        current_signal = signals.iloc[i - 1]
        current_open = prices.iloc[i]["open"]
        current_close = prices.iloc[i]["close"]
        prev_close = prices.iloc[i - 1]["close"]

        # Calculate return if in position
        if position != 0:
            price_return = (current_close - prev_close) / prev_close
            pnl = position * price_return * shares_per_trade * prev_close
            equity += pnl
            daily_returns.append(pnl / equity_curve[-1] if equity_curve[-1] > 0 else 0)
        else:
            daily_returns.append(0)

        # Check for position change
        if current_signal != position:
            # Close existing position
            if position != 0:
                exit_pnl = position * (current_open - entry_price) * shares_per_trade
                trades.append(
                    TradeRecord(
                        direction="LONG" if position > 0 else "SHORT",
                        entry_price=entry_price,
                        exit_price=current_open,
                        pnl=exit_pnl,
                        shares=shares_per_trade,
                    )
                )

            # Open new position
            if current_signal != 0:
                entry_price = current_open
                position = current_signal
            else:
                position = 0
                entry_price = 0.0

        # Track equity and drawdown
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    # Close final position
    if position != 0:
        final_close = prices.iloc[-1]["close"]
        exit_pnl = position * (final_close - entry_price) * shares_per_trade
        trades.append(
            TradeRecord(
                direction="LONG" if position > 0 else "SHORT",
                entry_price=entry_price,
                exit_price=final_close,
                pnl=exit_pnl,
                shares=shares_per_trade,
            )
        )

    # Calculate metrics
    total_pnl = equity - initial_equity
    total_return = total_pnl / initial_equity

    winning_trades = [t for t in trades if t.pnl > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0

    # Sharpe ratio (annualized for hourly data)
    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252 * 7)
    else:
        sharpe = 0

    return BacktestResult(
        total_pnl=total_pnl,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        num_trades=len(trades),
        avg_trade_pnl=total_pnl / len(trades) if trades else 0,
        trades=trades,
    )
