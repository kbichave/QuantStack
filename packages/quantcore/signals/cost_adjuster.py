"""
Cost-Adjusted Signal Generation.

Adjusts signals for transaction costs and filters low-edge trades.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CostParams:
    """Transaction cost parameters."""

    commission_bps: float = 1.0  # Commission in basis points
    spread_bps: float = 2.0  # Half-spread in basis points
    impact_bps: float = 1.0  # Market impact in basis points
    borrow_cost_bps: float = 0.0  # Short borrow cost (annualized bps)


def estimate_transaction_cost(
    trade_size: float,
    price: float,
    daily_volume: float,
    volatility: float,
    params: Optional[CostParams] = None,
) -> float:
    """
    Estimate total transaction cost.

    Args:
        trade_size: Absolute trade size
        price: Current price
        daily_volume: Average daily volume
        volatility: Daily volatility
        params: Cost parameters

    Returns:
        Cost in basis points
    """
    params = params or CostParams()

    # Fixed costs
    fixed_cost = params.commission_bps + params.spread_bps

    # Impact cost (square-root model)
    if daily_volume > 0:
        participation = abs(trade_size) / daily_volume
        impact_cost = params.impact_bps * np.sqrt(participation) * volatility / 0.02
    else:
        impact_cost = params.impact_bps

    return fixed_cost + impact_cost


def net_of_cost_signal(
    signal: pd.Series,
    expected_return_bps: pd.Series,
    cost_bps: float = 5.0,
) -> pd.Series:
    """
    Adjust signal for transaction costs.

    Only trade when expected return exceeds cost.

    Args:
        signal: Raw position signal
        expected_return_bps: Expected return in basis points
        cost_bps: Transaction cost in basis points

    Returns:
        Cost-adjusted signal
    """
    # Required edge to trade
    position_change = signal.diff().abs()
    required_return = position_change * cost_bps

    # Zero out signal where expected return doesn't cover costs
    adjusted = signal.copy()
    adjusted[expected_return_bps.abs() < required_return] = 0

    return adjusted


def minimum_edge_filter(
    signal: pd.Series,
    min_edge_bps: float = 5.0,
    edge_estimate: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Filter signals below minimum edge threshold.

    Args:
        signal: Position signal
        min_edge_bps: Minimum required edge in basis points
        edge_estimate: Optional edge estimate (uses signal magnitude if None)

    Returns:
        Filtered signal
    """
    if edge_estimate is None:
        edge_estimate = signal.abs() * 100  # Rough conversion to bps

    filtered = signal.copy()
    filtered[edge_estimate.abs() < min_edge_bps] = 0

    return filtered


class CostAdjuster:
    """
    Cost-aware signal adjustment.

    Features:
    - Transaction cost estimation
    - Turnover penalty
    - Position smoothing to reduce trading

    Example:
        adjuster = CostAdjuster(
            commission_bps=1.0,
            spread_bps=2.0,
            turnover_penalty=0.5,
        )

        adjusted = adjuster.adjust(raw_signal, returns=expected_returns)
    """

    def __init__(
        self,
        commission_bps: float = 1.0,
        spread_bps: float = 2.0,
        impact_bps: float = 1.0,
        turnover_penalty: float = 0.5,
        min_trade_size: float = 0.05,
    ):
        """
        Initialize cost adjuster.

        Args:
            commission_bps: Commission per trade
            spread_bps: Half-spread cost
            impact_bps: Base market impact
            turnover_penalty: Penalty for turnover (0-1)
            min_trade_size: Minimum position change to trade
        """
        self.params = CostParams(
            commission_bps=commission_bps,
            spread_bps=spread_bps,
            impact_bps=impact_bps,
        )
        self.turnover_penalty = turnover_penalty
        self.min_trade_size = min_trade_size

        self.cost_history: pd.Series = pd.Series(dtype=float)

    def estimate_cost(
        self,
        position_change: pd.Series,
        daily_volume: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Estimate trading costs.

        Args:
            position_change: Absolute position changes
            daily_volume: Daily volume series
            volatility: Volatility series

        Returns:
            Estimated cost in basis points
        """
        # Fixed costs
        cost = self.params.commission_bps + self.params.spread_bps

        # Variable impact
        if volatility is not None:
            # Scale impact by volatility
            vol_factor = volatility / volatility.mean()
            cost = cost + self.params.impact_bps * vol_factor
        else:
            cost = cost + self.params.impact_bps

        # Scale by trade size
        return cost * position_change.abs()

    def adjust(
        self,
        signal: pd.Series,
        returns: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Adjust signal for costs.

        Args:
            signal: Raw position signal
            returns: Expected returns (optional)
            volatility: Volatility (optional)

        Returns:
            Cost-adjusted signal
        """
        adjusted = signal.copy()

        # 1. Apply smoothing to reduce turnover
        if self.turnover_penalty > 0:
            halflife = int(1 + self.turnover_penalty * 5)
            adjusted = adjusted.ewm(halflife=halflife).mean()

        # 2. Apply minimum trade size filter
        position_change = adjusted.diff().abs()
        small_trades = position_change < self.min_trade_size

        # Keep previous position for small trades
        for i in range(1, len(adjusted)):
            if small_trades.iloc[i]:
                adjusted.iloc[i] = adjusted.iloc[i - 1]

        # 3. Zero out positions with insufficient edge
        if returns is not None:
            expected_pnl = returns * adjusted
            cost = self.estimate_cost(adjusted.diff().abs(), volatility=volatility)
            net_pnl = expected_pnl - cost / 10000  # Convert bps to decimal

            # Zero positions with negative expected net PnL
            adjusted[net_pnl < 0] = 0

        return adjusted

    def compute_net_returns(
        self,
        signal: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Compute returns net of transaction costs.

        Args:
            signal: Position signal
            returns: Raw returns

        Returns:
            Net returns
        """
        # Gross returns
        gross = signal.shift(1) * returns

        # Trading costs
        turnover = signal.diff().abs()
        cost_bps = self.estimate_cost(turnover)
        cost = cost_bps / 10000  # Convert to decimal

        # Store cost history
        self.cost_history = cost

        return gross - cost

    def turnover_analysis(
        self,
        signal: pd.Series,
    ) -> Dict:
        """
        Analyze signal turnover.

        Args:
            signal: Position signal

        Returns:
            Dictionary with turnover statistics
        """
        turnover = signal.diff().abs()

        # Annualized turnover (assuming daily data)
        annual_turnover = turnover.mean() * 252

        # Two-way turnover
        long_short_turnover = (signal.diff().abs()).sum()

        # Holding period (average)
        position_changes = (signal.diff().abs() > 0.01).sum()
        avg_holding = len(signal) / (position_changes + 1)

        return {
            "daily_turnover": turnover.mean(),
            "annual_turnover": annual_turnover,
            "total_turnover": long_short_turnover,
            "avg_holding_days": avg_holding,
            "n_trades": position_changes,
        }


def optimize_cost_threshold(
    signal: pd.Series,
    returns: pd.Series,
    cost_bps: float,
    thresholds: np.ndarray = np.linspace(0, 0.5, 20),
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal signal threshold considering costs.

    Args:
        signal: Raw signal
        returns: Returns series
        cost_bps: Cost in basis points
        thresholds: Thresholds to test

    Returns:
        Tuple of (optimal_threshold, results_df)
    """
    results = []

    for thresh in thresholds:
        # Apply threshold
        pos = signal.copy()
        pos[signal.abs() < thresh] = 0

        # Compute net returns
        gross_ret = (pos.shift(1) * returns).sum()
        turnover = pos.diff().abs().sum()
        cost = turnover * cost_bps / 10000
        net_ret = gross_ret - cost

        results.append(
            {
                "threshold": thresh,
                "gross_return": gross_ret,
                "cost": cost,
                "net_return": net_ret,
                "turnover": turnover,
            }
        )

    results_df = pd.DataFrame(results)
    optimal_idx = results_df["net_return"].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, "threshold"]

    return optimal_threshold, results_df
