# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Performance reporting for backtests."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from quantcore.backtesting.engine import BacktestResult


class PerformanceReport:
    """
    Generate performance reports from backtest results.

    Provides formatted output and summary statistics.
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize with backtest result.

        Args:
            result: BacktestResult from BacktestEngine
        """
        self.result = result

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary."""
        return {
            "total_trades": self.result.total_trades,
            "win_rate": self.result.win_rate,
            "sharpe_ratio": self.result.sharpe_ratio,
            "max_drawdown": self.result.max_drawdown,
            "total_return": self.result.total_return,
            "profit_factor": self.result.profit_factor,
        }

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Trades:    {self.result.total_trades}")
        print(f"Win Rate:        {self.result.win_rate:.1f}%")
        print(f"Sharpe Ratio:    {self.result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:    {self.result.max_drawdown:.1f}%")
        print(f"Total Return:    {self.result.total_return:.1f}%")
        print(f"Profit Factor:   {self.result.profit_factor:.2f}")
        print("=" * 60 + "\n")

    def get_trades(self) -> List[Dict]:
        """Get list of trades."""
        return self.result.trades

    def get_equity_curve(self) -> List[float]:
        """Get equity curve."""
        return self.result.equity_curve

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Backtest Performance Report",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {self.result.total_trades} |",
            f"| Win Rate | {self.result.win_rate:.1f}% |",
            f"| Sharpe Ratio | {self.result.sharpe_ratio:.2f} |",
            f"| Max Drawdown | {self.result.max_drawdown:.1f}% |",
            f"| Total Return | {self.result.total_return:.1f}% |",
            f"| Profit Factor | {self.result.profit_factor:.2f} |",
            "",
        ]

        if self.result.trades:
            lines.extend(
                [
                    "## Trade Summary",
                    "",
                    f"- **Winners**: {sum(1 for t in self.result.trades if t['pnl'] > 0)}",
                    f"- **Losers**: {sum(1 for t in self.result.trades if t['pnl'] <= 0)}",
                    f"- **Avg Win**: ${np.mean([t['pnl'] for t in self.result.trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in self.result.trades) else 0:.2f}",
                    f"- **Avg Loss**: ${np.mean([t['pnl'] for t in self.result.trades if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in self.result.trades) else 0:.2f}",
                    "",
                ]
            )

        return "\n".join(lines)
