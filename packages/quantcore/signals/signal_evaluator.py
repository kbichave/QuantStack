"""
Signal Evaluation and Performance Metrics.

Comprehensive signal and strategy analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Standard performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float


def compute_ic(
    signal: pd.Series,
    returns: pd.Series,
    lag: int = 1,
) -> float:
    """
    Compute Information Coefficient (IC).

    Correlation between signal and future returns.

    Args:
        signal: Signal series
        returns: Return series
        lag: Forward return lag

    Returns:
        IC (Pearson correlation)
    """
    forward_returns = returns.shift(-lag)
    return signal.corr(forward_returns)


def compute_ic_series(
    signal: pd.Series,
    returns: pd.Series,
    window: int = 21,
    lag: int = 1,
) -> pd.Series:
    """
    Compute rolling IC.

    Args:
        signal: Signal series
        returns: Return series
        window: Rolling window
        lag: Forward return lag

    Returns:
        Rolling IC series
    """
    forward_returns = returns.shift(-lag)
    return signal.rolling(window).corr(forward_returns)


def compute_sharpe(
    returns: pd.Series,
    rf_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sharpe ratio.

    Args:
        returns: Return series
        rf_rate: Risk-free rate (annualized)
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - rf_rate / periods_per_year

    if returns.std() == 0:
        return 0.0

    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def compute_sortino(
    returns: pd.Series,
    rf_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sortino ratio.

    Uses downside deviation instead of total volatility.

    Args:
        returns: Return series
        rf_rate: Risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - rf_rate / periods_per_year
    downside = returns[returns < 0]

    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    downside_std = downside.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def compute_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Series]:
    """
    Compute maximum drawdown.

    Args:
        returns: Return series

    Returns:
        Tuple of (max_drawdown, drawdown_series)
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max

    return drawdown.min(), drawdown


def compute_turnover(signal: pd.Series) -> float:
    """
    Compute annualized turnover.

    Args:
        signal: Position signal

    Returns:
        Annualized turnover (two-way)
    """
    daily_turnover = signal.diff().abs().mean()
    return daily_turnover * 252


class SignalEvaluator:
    """
    Comprehensive signal evaluation.

    Features:
    - IC analysis
    - Return attribution
    - Risk metrics
    - Decay analysis

    Example:
        evaluator = SignalEvaluator()
        metrics = evaluator.evaluate(signal, returns)

        print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"IC: {evaluator.ic:.3f}")
    """

    def __init__(
        self,
        periods_per_year: int = 252,
        rf_rate: float = 0.0,
    ):
        """
        Initialize evaluator.

        Args:
            periods_per_year: Trading periods per year
            rf_rate: Risk-free rate (annualized)
        """
        self.periods_per_year = periods_per_year
        self.rf_rate = rf_rate

        self.ic: float = 0.0
        self.ic_series: pd.Series = pd.Series(dtype=float)
        self.returns: pd.Series = pd.Series(dtype=float)

    def evaluate(
        self,
        signal: pd.Series,
        returns: pd.Series,
        cost_bps: float = 0.0,
    ) -> PerformanceMetrics:
        """
        Full signal evaluation.

        Args:
            signal: Position signal
            returns: Asset returns
            cost_bps: Transaction cost in basis points

        Returns:
            PerformanceMetrics with all stats
        """
        # Compute strategy returns
        gross_returns = signal.shift(1) * returns

        # Subtract costs
        if cost_bps > 0:
            turnover = signal.diff().abs()
            costs = turnover * cost_bps / 10000
            self.returns = gross_returns - costs
        else:
            self.returns = gross_returns

        self.returns = self.returns.dropna()

        # IC
        self.ic = compute_ic(signal, returns)
        self.ic_series = compute_ic_series(signal, returns)

        # Return metrics
        total_return = (1 + self.returns).prod() - 1
        ann_return = (1 + total_return) ** (
            self.periods_per_year / len(self.returns)
        ) - 1
        volatility = self.returns.std() * np.sqrt(self.periods_per_year)

        # Risk-adjusted
        sharpe = compute_sharpe(self.returns, self.rf_rate, self.periods_per_year)
        sortino = compute_sortino(self.returns, self.rf_rate, self.periods_per_year)
        max_dd, _ = compute_max_drawdown(self.returns)
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # Win/loss analysis
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]

        win_rate = len(wins) / len(self.returns) if len(self.returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

    def alpha_decay_analysis(
        self,
        signal: pd.Series,
        returns: pd.Series,
        max_lag: int = 20,
    ) -> pd.Series:
        """
        Analyze alpha decay over holding periods.

        Args:
            signal: Position signal
            returns: Asset returns
            max_lag: Maximum holding period to test

        Returns:
            Series of IC at each lag
        """
        ics = []
        for lag in range(1, max_lag + 1):
            # Cumulative forward returns
            forward_ret = returns.rolling(lag).sum().shift(-lag)
            ic = signal.corr(forward_ret)
            ics.append(ic)

        return pd.Series(ics, index=range(1, max_lag + 1))

    def ic_analysis(self) -> Dict:
        """
        Detailed IC analysis.

        Returns:
            Dictionary with IC statistics
        """
        ic_mean = self.ic_series.mean()
        ic_std = self.ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0  # IC Information Ratio

        # Percentage of positive IC months
        monthly_ic = self.ic_series.resample("M").mean()
        pct_positive = (monthly_ic > 0).mean()

        return {
            "ic": self.ic,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "pct_positive_months": pct_positive,
        }

    def generate_report(
        self,
        signal: pd.Series,
        returns: pd.Series,
        cost_bps: float = 5.0,
    ) -> Dict:
        """
        Generate comprehensive evaluation report.

        Args:
            signal: Position signal
            returns: Asset returns
            cost_bps: Transaction costs

        Returns:
            Dictionary with full report
        """
        # Main metrics
        metrics = self.evaluate(signal, returns, cost_bps)

        # IC analysis
        ic_stats = self.ic_analysis()

        # Alpha decay
        decay = self.alpha_decay_analysis(signal, returns)

        # Turnover
        turnover = compute_turnover(signal)

        # Monthly returns
        monthly_returns = self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        return {
            "performance": {
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "calmar_ratio": metrics.calmar_ratio,
            },
            "trading": {
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "turnover": turnover,
            },
            "signal_quality": ic_stats,
            "alpha_decay": decay.to_dict(),
            "monthly_returns": monthly_returns.to_dict(),
        }


def compare_signals(
    signals: Dict[str, pd.Series],
    returns: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compare multiple signals.

    Args:
        signals: Dictionary of signal name -> series
        returns: Asset returns
        cost_bps: Transaction costs

    Returns:
        DataFrame with comparison metrics
    """
    evaluator = SignalEvaluator()
    results = []

    for name, signal in signals.items():
        metrics = evaluator.evaluate(signal, returns, cost_bps)
        results.append(
            {
                "signal": name,
                "sharpe": metrics.sharpe_ratio,
                "sortino": metrics.sortino_ratio,
                "max_dd": metrics.max_drawdown,
                "ic": evaluator.ic,
                "turnover": compute_turnover(signal),
                "win_rate": metrics.win_rate,
            }
        )

    return pd.DataFrame(results).set_index("signal")
