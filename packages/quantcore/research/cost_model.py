"""
Transaction Cost Model.

Comprehensive model for trading costs:
- Bid-ask spread costs
- Market impact (Kyle lambda)
- Slippage estimation
- Brokerage commissions
- Total cost analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from loguru import logger


@dataclass
class TradeCost:
    """Cost breakdown for a single trade."""

    spread_cost: float  # Half spread paid
    impact_cost: float  # Market impact
    slippage: float  # Execution slippage
    commission: float  # Broker fees
    total_cost: float  # Total in basis points


@dataclass
class CostModelParams:
    """Parameters for cost model."""

    half_spread_bps: float = 5.0  # Half spread in basis points
    impact_coefficient: float = 0.1  # Kyle lambda equivalent
    slippage_vol_mult: float = 0.5  # Slippage = mult * volatility
    commission_per_share: float = 0.005  # Per-share commission


class TransactionCostModel:
    """
    Transaction cost model for realistic backtesting.

    Components:
    1. Spread cost: Half the bid-ask spread
    2. Market impact: Price movement due to order (Kyle lambda model)
    3. Slippage: Random execution variance
    4. Commission: Broker fees

    Based on literature:
    - Kyle (1985): Linear price impact
    - Almgren & Chriss (2001): Optimal execution with impact
    """

    def __init__(self, params: Optional[CostModelParams] = None):
        """
        Initialize cost model.

        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or CostModelParams()
        self._fitted = False
        self._impact_history: pd.DataFrame = None

    def estimate_cost(
        self,
        order_size_shares: float,
        price: float,
        adv: float,  # Average daily volume in shares
        volatility: float,  # Daily volatility (decimal)
        is_buy: bool = True,
    ) -> TradeCost:
        """
        Estimate total cost for a trade.

        Args:
            order_size_shares: Number of shares to trade
            price: Current price
            adv: Average daily volume in shares
            volatility: Daily volatility
            is_buy: True for buy, False for sell

        Returns:
            TradeCost with breakdown
        """
        trade_value = order_size_shares * price
        participation_rate = order_size_shares / adv if adv > 0 else 0

        # 1. Spread cost (half spread)
        spread_cost_bps = self.params.half_spread_bps
        spread_cost = trade_value * spread_cost_bps / 10000

        # 2. Market impact (Kyle lambda style)
        # Impact ~ lambda * sqrt(Q/V) * sigma * P
        # Where Q = order size, V = daily volume, sigma = volatility
        impact_bps = (
            self.params.impact_coefficient
            * np.sqrt(participation_rate)
            * volatility
            * 10000
        )
        impact_cost = trade_value * impact_bps / 10000

        # 3. Slippage (volatility-based random component)
        slippage_bps = (
            self.params.slippage_vol_mult * volatility * 10000 * np.random.randn()
        )
        slippage_bps = max(0, slippage_bps)  # Slippage is always a cost
        slippage = trade_value * slippage_bps / 10000

        # 4. Commission
        commission = order_size_shares * self.params.commission_per_share

        # Total
        total_cost = spread_cost + impact_cost + slippage + commission
        total_bps = total_cost / trade_value * 10000 if trade_value > 0 else 0

        return TradeCost(
            spread_cost=spread_cost_bps,
            impact_cost=impact_bps,
            slippage=slippage_bps,
            commission=commission / trade_value * 10000 if trade_value > 0 else 0,
            total_cost=total_bps,
        )

    def fit_impact_model(
        self,
        trades: pd.DataFrame,
        price_col: str = "price",
        volume_col: str = "volume",
        trade_size_col: str = "trade_size",
        price_change_col: str = "price_change",
    ) -> Dict:
        """
        Fit market impact model from historical trade data.

        Uses Kyle (1985) style regression:
        ΔP = λ * sign(Q) * sqrt(|Q|/V) + ε

        Args:
            trades: DataFrame with trade data
            price_col: Column for price
            volume_col: Column for daily volume
            trade_size_col: Column for trade size (signed)
            price_change_col: Column for price change after trade

        Returns:
            Dictionary with fitted parameters
        """
        trades = trades.dropna(subset=[trade_size_col, volume_col, price_change_col])

        if len(trades) < 30:
            logger.warning("Insufficient trades for impact model fitting")
            return {"lambda": self.params.impact_coefficient, "r_squared": 0}

        # Compute normalized order flow
        order_flow = trades[trade_size_col].values
        volume = trades[volume_col].values
        price_change = trades[price_change_col].values

        # Signed sqrt participation
        signed_sqrt_participation = np.sign(order_flow) * np.sqrt(
            np.abs(order_flow) / (volume + 1e-10)
        )

        # Simple OLS regression
        X = signed_sqrt_participation.reshape(-1, 1)
        y = price_change

        # Add constant
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS: beta = (X'X)^(-1) X'y
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

            # Predictions and R-squared
            y_pred = X_with_const @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Update model parameter
            fitted_lambda = beta[1]
            self.params.impact_coefficient = abs(fitted_lambda)
            self._fitted = True

            return {
                "lambda": fitted_lambda,
                "intercept": beta[0],
                "r_squared": r_squared,
                "n_trades": len(trades),
            }

        except Exception as e:
            logger.error(f"Impact model fitting failed: {e}")
            return {"lambda": self.params.impact_coefficient, "r_squared": 0}

    def compute_execution_shortfall(
        self,
        orders: pd.DataFrame,
        arrival_price: float,
        execution_prices: pd.Series,
        execution_sizes: pd.Series,
    ) -> Dict:
        """
        Compute Implementation Shortfall (IS).

        IS = (Avg Execution Price - Arrival Price) / Arrival Price

        Decomposes into:
        - Delay cost
        - Market impact
        - Timing cost

        Args:
            orders: Order records
            arrival_price: Price at decision time
            execution_prices: Prices of each fill
            execution_sizes: Sizes of each fill

        Returns:
            Dictionary with IS breakdown
        """
        total_size = execution_sizes.sum()

        if total_size == 0:
            return {"total_is_bps": 0, "error": "no_fills"}

        # Volume-weighted average execution price
        vwap_exec = (execution_prices * execution_sizes).sum() / total_size

        # Implementation shortfall
        is_decimal = (vwap_exec - arrival_price) / arrival_price
        is_bps = is_decimal * 10000

        return {
            "arrival_price": arrival_price,
            "vwap_execution": vwap_exec,
            "total_shares": total_size,
            "total_is_bps": is_bps,
            "interpretation": f"{'Outperformed' if is_bps < 0 else 'Underperformed'} arrival by {abs(is_bps):.1f} bps",
        }

    def simulate_costs_for_strategy(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volumes: pd.Series,
        volatility: pd.Series,
        position_value: float = 100000,
    ) -> pd.DataFrame:
        """
        Simulate transaction costs for a strategy's signal series.

        Args:
            signals: Trading signals (-1, 0, 1 or continuous)
            prices: Price series
            volumes: Volume series
            volatility: Volatility series
            position_value: Dollar value per position

        Returns:
            DataFrame with cost estimates per trade
        """
        # Align all series
        common_idx = signals.index.intersection(prices.index).intersection(
            volumes.index
        )
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]
        volumes = volumes.loc[common_idx]
        volatility = volatility.reindex(common_idx).ffill()

        # Detect trades (signal changes)
        signal_changes = signals.diff().fillna(0)
        trade_mask = signal_changes != 0

        results = []

        for ts in signals.index[trade_mask]:
            signal_change = signal_changes.loc[ts]
            price = prices.loc[ts]
            vol = volumes.loc[ts]
            daily_vol = volatility.loc[ts] if ts in volatility.index else 0.02

            # Trade size
            shares = position_value * abs(signal_change) / price

            # Estimate cost
            cost = self.estimate_cost(
                order_size_shares=shares,
                price=price,
                adv=vol,
                volatility=daily_vol,
                is_buy=signal_change > 0,
            )

            results.append(
                {
                    "timestamp": ts,
                    "signal_change": signal_change,
                    "shares": shares,
                    "price": price,
                    "spread_bps": cost.spread_cost,
                    "impact_bps": cost.impact_cost,
                    "slippage_bps": cost.slippage,
                    "commission_bps": cost.commission,
                    "total_cost_bps": cost.total_cost,
                }
            )

        return pd.DataFrame(results)

    def generate_cost_report(
        self,
        cost_df: pd.DataFrame,
    ) -> str:
        """Generate summary report of trading costs."""
        if cost_df.empty:
            return "No trades to analyze"

        n_trades = len(cost_df)
        total_cost = cost_df["total_cost_bps"].sum()
        avg_cost = cost_df["total_cost_bps"].mean()

        report = f"""
Transaction Cost Analysis
=========================

Trade Summary:
  Number of trades: {n_trades}
  Total cost: {total_cost:.1f} bps
  Average cost per trade: {avg_cost:.1f} bps

Cost Breakdown (average per trade):
  Spread:     {cost_df['spread_bps'].mean():.1f} bps ({cost_df['spread_bps'].mean() / avg_cost * 100:.0f}%)
  Impact:     {cost_df['impact_bps'].mean():.1f} bps ({cost_df['impact_bps'].mean() / avg_cost * 100:.0f}%)
  Slippage:   {cost_df['slippage_bps'].mean():.1f} bps ({cost_df['slippage_bps'].mean() / avg_cost * 100:.0f}%)
  Commission: {cost_df['commission_bps'].mean():.1f} bps ({cost_df['commission_bps'].mean() / avg_cost * 100:.0f}%)

Annualized Cost Drag:
  Assuming 252 trading days: {total_cost / len(cost_df) * 252:.0f} bps/year
"""
        return report


def estimate_break_even_alpha(
    avg_cost_bps: float,
    holding_period_days: float,
    annual_vol: float = 0.20,
) -> float:
    """
    Estimate minimum alpha needed to break even after costs.

    Args:
        avg_cost_bps: Average round-trip cost in bps
        holding_period_days: Average holding period
        annual_vol: Annual volatility of strategy

    Returns:
        Minimum annual alpha (decimal) needed
    """
    trades_per_year = 252 / holding_period_days
    annual_cost = avg_cost_bps * trades_per_year / 10000

    # Break-even alpha = annual cost
    # For Sharpe > 1, need alpha > annual_vol + cost

    return annual_cost


def optimal_trade_size(
    alpha_signal: float,
    price: float,
    adv: float,
    volatility: float,
    risk_aversion: float = 1.0,
    impact_coeff: float = 0.1,
) -> float:
    """
    Compute optimal trade size balancing alpha capture vs impact.

    From Almgren-Chriss (2001): Optimal trade = alpha / (2 * lambda * sqrt(V))

    Args:
        alpha_signal: Expected return signal (decimal)
        price: Current price
        adv: Average daily volume
        volatility: Daily volatility
        risk_aversion: Risk aversion parameter
        impact_coeff: Market impact coefficient

    Returns:
        Optimal number of shares to trade
    """
    if alpha_signal == 0:
        return 0

    # Optimal participation rate
    # Q* = alpha / (2 * lambda) when impact = lambda * sqrt(Q/V)
    opt_participation = (alpha_signal / (2 * impact_coeff)) ** 2

    # Cap at reasonable participation
    opt_participation = min(opt_participation, 0.05)  # Max 5% ADV

    opt_shares = opt_participation * adv

    return opt_shares
