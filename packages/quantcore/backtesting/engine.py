"""
Core backtesting engine for WTI trading strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.utils.formatting import (
    print_info,
    print_success,
    print_error,
    print_money,
    print_section,
)
from quantcore.validation.input_validation import DataFrameValidator


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100000.0
    max_concurrent_trades: int = 5
    commission_per_trade: float = 1.0
    slippage_pct: float = 0.001
    position_size_pct: float = 0.1
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0


@dataclass
class BacktestResult:
    """Result from a backtest run."""

    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    profit_factor: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Runs backtests on signal DataFrames against price data.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize with configuration."""
        self.config = config or BacktestConfig()
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            signals: DataFrame with signal columns (signal, signal_direction, etc.)
            price_data: OHLCV price data

        Returns:
            BacktestResult with metrics

        Raises:
            ValueError: If price_data fails OHLCV validation
        """
        # Validate price data
        validation_result = DataFrameValidator.validate_ohlcv(
            price_data, name="price_data", raise_on_error=True
        )
        validation_result.log_warnings()

        # Basic signal validation
        if signals is None or signals.empty:
            logger.warning("BacktestEngine.run: signals DataFrame is empty")
            return BacktestResult()

        capital = self.config.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity_curve = [capital]

        # Align data
        common_idx = signals.index.intersection(price_data.index)
        if len(common_idx) == 0:
            return BacktestResult()

        signals = signals.loc[common_idx]
        prices = price_data.loc[common_idx]

        for i in range(1, len(common_idx)):
            idx = common_idx[i]
            prev_idx = common_idx[i - 1]

            current_price = prices.loc[idx, "close"]
            signal = signals.loc[idx].get("signal", 0)
            direction = signals.loc[idx].get("signal_direction", "NONE")

            # Entry
            if position == 0 and signal == 1:
                if direction == "LONG":
                    position = 1
                    entry_price = current_price * (1 + self.config.slippage_pct)
                    capital -= self.config.commission_per_trade
                elif direction == "SHORT":
                    position = -1
                    entry_price = current_price * (1 - self.config.slippage_pct)
                    capital -= self.config.commission_per_trade

            # Exit on opposite signal or flat
            elif position != 0:
                should_exit = False
                if direction == "NONE" or signal == 0:
                    should_exit = True
                elif position == 1 and direction == "SHORT":
                    should_exit = True
                elif position == -1 and direction == "LONG":
                    should_exit = True

                if should_exit:
                    exit_price = current_price * (
                        1 - self.config.slippage_pct * position
                    )
                    if position == 1:
                        pnl = (exit_price - entry_price) * (
                            capital * self.config.position_size_pct / entry_price
                        )
                    else:
                        pnl = (entry_price - exit_price) * (
                            capital * self.config.position_size_pct / entry_price
                        )

                    capital += pnl - self.config.commission_per_trade
                    trades.append(
                        {
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "direction": "LONG" if position == 1 else "SHORT",
                        }
                    )
                    position = 0

            # Mark to market
            if position == 1:
                mtm = capital + (current_price - entry_price) * (
                    capital * self.config.position_size_pct / entry_price
                )
            elif position == -1:
                mtm = capital + (entry_price - current_price) * (
                    capital * self.config.position_size_pct / entry_price
                )
            else:
                mtm = capital
            equity_curve.append(mtm)

        # Calculate metrics
        self.trades = trades
        self.equity_curve = equity_curve

        return self._calculate_result(trades, equity_curve)

    def _calculate_result(
        self, trades: List[Dict], equity_curve: List[float]
    ) -> BacktestResult:
        """Calculate result metrics."""
        if not trades:
            return BacktestResult(equity_curve=equity_curve)

        equity = np.array(equity_curve)
        total_trades = len(trades)
        winners = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = winners / total_trades if total_trades > 0 else 0

        # Returns and Sharpe
        if len(equity) > 1:
            returns = np.diff(equity) / (equity[:-1] + 1e-8)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-8)
        max_dd = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

        # Total return
        total_return = (
            (equity[-1] - equity[0]) / equity[0] * 100 if len(equity) > 0 else 0
        )

        # Profit factor
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_trades=total_trades,
            win_rate=win_rate * 100,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve,
        )


def calculate_metrics(
    final_capital: float,
    initial_capital: float,
    trades: List[Dict],
    equity_curve: List[float],
) -> Dict[str, float]:
    """Calculate standard metrics from backtest results."""
    equity_curve = np.array(equity_curve)

    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-8)
        max_dd = np.max(drawdown) * 100
    else:
        sharpe = 0
        max_dd = 0

    total_trades = len(trades)
    profitable = sum(1 for t in trades if t.get("pnl", 0) > 0)
    total_pnl = final_capital - initial_capital

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / initial_capital * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": profitable / max(1, total_trades) * 100,
        "total_trades": total_trades,
        "profitable_trades": profitable,
    }


def run_backtest_with_params(
    data: pd.DataFrame,
    initial_capital: float,
    entry_zscore: float,
    exit_zscore: float,
    position_size: int,
    spread_cost: float,
    stop_loss_zscore: Optional[float],
    return_trades_list: bool = False,
) -> Dict[str, Any]:
    """Run backtest with specific parameters."""
    results = {
        "initial_capital": initial_capital,
        "final_capital": initial_capital,
        "total_return": 0,
        "total_pnl": 0,
        "total_return_pct": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "win_rate": 0,
        "total_trades": 0,
        "profitable_trades": 0,
    }

    if len(data) < 10:
        return results

    capital = initial_capital
    position = 0
    entry_price = 0
    entry_idx = 0
    trades = []
    equity_curve = [capital]

    for i in range(1, len(data)):
        row = data.iloc[i]
        zscore = row["spread_zscore"]
        spread = row["spread"]

        # Entry signals
        if position == 0:
            if zscore < -entry_zscore:
                position = 1
                entry_price = spread
                entry_idx = i
                capital -= spread_cost * position_size
            elif zscore > entry_zscore:
                position = -1
                entry_price = spread
                entry_idx = i
                capital -= spread_cost * position_size

        # Exit signals
        elif position == 1:
            # Stop loss
            if stop_loss_zscore and zscore < -stop_loss_zscore:
                pnl = (
                    spread - entry_price
                ) * position_size - spread_cost * position_size
                capital += pnl
                trades.append(
                    {
                        "pnl": pnl,
                        "type": "LONG_STOP",
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "LONG",
                    }
                )
                position = 0
            # Take profit
            elif zscore > exit_zscore:
                pnl = (
                    spread - entry_price
                ) * position_size - spread_cost * position_size
                capital += pnl
                trades.append(
                    {
                        "pnl": pnl,
                        "type": "LONG",
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "LONG",
                    }
                )
                position = 0

        elif position == -1:
            # Stop loss
            if stop_loss_zscore and zscore > stop_loss_zscore:
                pnl = (
                    entry_price - spread
                ) * position_size - spread_cost * position_size
                capital += pnl
                trades.append(
                    {
                        "pnl": pnl,
                        "type": "SHORT_STOP",
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "SHORT",
                    }
                )
                position = 0
            # Take profit
            elif zscore < -exit_zscore:
                pnl = (
                    entry_price - spread
                ) * position_size - spread_cost * position_size
                capital += pnl
                trades.append(
                    {
                        "pnl": pnl,
                        "type": "SHORT",
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "SHORT",
                    }
                )
                position = 0

        # Mark to market
        if position == 1:
            mtm = capital + (spread - entry_price) * position_size
        elif position == -1:
            mtm = capital + (entry_price - spread) * position_size
        else:
            mtm = capital
        equity_curve.append(mtm)

    # Close any open position
    if position != 0:
        spread = data.iloc[-1]["spread"]
        if position == 1:
            pnl = (spread - entry_price) * position_size - spread_cost * position_size
        else:
            pnl = (entry_price - spread) * position_size - spread_cost * position_size
        capital += pnl
        trades.append({"pnl": pnl, "type": "CLOSE"})

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)
        results["sharpe_ratio"] = (
            np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        )

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-8)
        results["max_drawdown"] = np.max(drawdown) * 100

    results["final_capital"] = capital
    results["total_return"] = capital - initial_capital
    results["total_pnl"] = capital - initial_capital  # Alias for consistent naming
    results["total_return_pct"] = (capital - initial_capital) / initial_capital * 100
    results["total_trades"] = len(trades)
    results["profitable_trades"] = sum(1 for t in trades if t["pnl"] > 0)
    results["win_rate"] = (
        results["profitable_trades"] / max(1, results["total_trades"]) * 100
    )

    # Return trades list and equity curve if requested (for plotting)
    if return_trades_list:
        results["trades_list"] = trades
        results["equity_curve"] = equity_curve

    return results


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    spread_df: pd.DataFrame,
    initial_capital: float = 100000,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Run backtest with optimized parameters and calculate profit."""
    print_section("Final Backtest with Optimized Parameters")

    results = {
        "initial_capital": initial_capital,
        "final_capital": initial_capital,
        "total_return": 0,
        "total_pnl": 0,
        "total_return_pct": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "win_rate": 0,
        "total_trades": 0,
        "profitable_trades": 0,
    }

    if spread_df.empty or len(spread_df) < 100:
        print_error("Insufficient spread data for backtest")
        return results

    # Filter to valid data with z-scores
    backtest_data = spread_df.dropna(subset=["spread_zscore"]).copy()

    if len(backtest_data) < 100:
        print_error("Not enough valid data points")
        return results

    # Use optimized parameters or defaults
    if params:
        ENTRY_ZSCORE = params.get("entry_zscore", 2.0)
        EXIT_ZSCORE = params.get("exit_zscore", 0.0)
        POSITION_SIZE = params.get("position_size", 1000)
        SPREAD_COST = params.get("spread_cost", 0.05)
        STOP_LOSS_ZSCORE = params.get("stop_loss_zscore", None)
    else:
        ENTRY_ZSCORE = 2.0
        EXIT_ZSCORE = 0.0
        POSITION_SIZE = 1000
        SPREAD_COST = 0.05
        STOP_LOSS_ZSCORE = None

    print_info(f"Running backtest on {len(backtest_data)} bars...")
    print_info(
        f"  Period: {backtest_data.index[0].date()} to {backtest_data.index[-1].date()}"
    )
    print_info(f"  Parameters:")
    print_info(f"    Entry Z-score: ±{ENTRY_ZSCORE}")
    print_info(f"    Exit Z-score: ±{EXIT_ZSCORE}")
    print_info(f"    Position size: {POSITION_SIZE} barrels")
    print_info(f"    Spread cost: ${SPREAD_COST}/barrel")
    print_info(
        f"    Stop loss Z-score: {STOP_LOSS_ZSCORE if STOP_LOSS_ZSCORE else 'None'}"
    )

    # Simple spread mean reversion strategy
    capital = initial_capital
    position = 0  # +1 = long spread (long WTI, short Brent), -1 = short spread
    entry_price = 0
    trades = []
    equity_curve = [capital]

    for i in range(1, len(backtest_data)):
        row = backtest_data.iloc[i]

        zscore = row["spread_zscore"]
        spread = row["spread"]

        # Entry signals
        if position == 0:
            if zscore < -ENTRY_ZSCORE:
                position = 1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
            elif zscore > ENTRY_ZSCORE:
                position = -1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE

        # Exit signals
        elif position == 1:
            if STOP_LOSS_ZSCORE and zscore < -STOP_LOSS_ZSCORE:
                pnl = (
                    spread - entry_price
                ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                capital += pnl
                trades.append(
                    {
                        "pnl": pnl,
                        "type": "LONG_STOP",
                        "entry": entry_price,
                        "exit": spread,
                    }
                )
                position = 0
            elif zscore > EXIT_ZSCORE:
                pnl = (
                    spread - entry_price
                ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                capital += pnl
                trades.append(
                    {"pnl": pnl, "type": "LONG", "entry": entry_price, "exit": spread}
                )
                position = 0

        elif position == -1:
            if STOP_LOSS_ZSCORE and zscore > STOP_LOSS_ZSCORE:
                pnl = (
                    entry_price - spread
                ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                capital += pnl
                trades.append(
                    {
                        "pnl": pnl,
                        "type": "SHORT_STOP",
                        "entry": entry_price,
                        "exit": spread,
                    }
                )
                position = 0
            elif zscore < -EXIT_ZSCORE:
                pnl = (
                    entry_price - spread
                ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                capital += pnl
                trades.append(
                    {"pnl": pnl, "type": "SHORT", "entry": entry_price, "exit": spread}
                )
                position = 0

        # Mark to market
        if position == 1:
            mtm = capital + (spread - entry_price) * POSITION_SIZE
        elif position == -1:
            mtm = capital + (entry_price - spread) * POSITION_SIZE
        else:
            mtm = capital

        equity_curve.append(mtm)

    # Close any open position
    if position != 0:
        spread = backtest_data.iloc[-1]["spread"]
        if position == 1:
            pnl = (spread - entry_price) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
        else:
            pnl = (entry_price - spread) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
        capital += pnl
        trades.append(
            {
                "pnl": pnl,
                "type": "LONG" if position == 1 else "SHORT",
                "entry": entry_price,
                "exit": spread,
            }
        )

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    results["final_capital"] = capital
    results["total_return"] = capital - initial_capital
    results["total_pnl"] = capital - initial_capital  # Alias for consistent naming
    results["total_return_pct"] = (capital - initial_capital) / initial_capital * 100
    results["sharpe_ratio"] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    results["max_drawdown"] = np.max(drawdown) * 100

    # Trade stats
    results["total_trades"] = len(trades)
    results["profitable_trades"] = sum(1 for t in trades if t["pnl"] > 0)
    results["win_rate"] = (
        results["profitable_trades"] / max(1, results["total_trades"]) * 100
    )

    # Print results
    print_success(f"Backtest complete!")
    print()

    is_profit = results["total_return"] > 0
    print_money("Total P&L", results["total_return"], is_profit)
    print_money("Final Capital", results["final_capital"], is_profit)
    print()

    print(f"📊 Performance Metrics:")
    print(f"    Return: {results['total_return_pct']:.2f}%")
    print(f"    Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"    Win Rate: {results['win_rate']:.1f}%")
    print(f"    Total Trades: {results['total_trades']}")
    print(f"    Profitable: {results['profitable_trades']}")

    if trades:
        avg_win = (
            np.mean([t["pnl"] for t in trades if t["pnl"] > 0])
            if any(t["pnl"] > 0 for t in trades)
            else 0
        )
        avg_loss = (
            np.mean([t["pnl"] for t in trades if t["pnl"] < 0])
            if any(t["pnl"] < 0 for t in trades)
            else 0
        )
        print(f"    Avg Win: ${avg_win:,.2f}")
        print(f"    Avg Loss: ${avg_loss:,.2f}")

    return results
