"""
Transaction cost modeling and cost-aware labeling.

For realistic P&L estimation in backtests.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.execution.slippage import SlippageModel, CompositeSlippageModel


@dataclass
class TransactionCosts:
    """Breakdown of transaction costs."""

    spread_bps: float
    slippage_bps: float
    commission_bps: float
    total_one_way_bps: float
    total_round_trip_bps: float


class TransactionCostModel:
    """
    Complete transaction cost model.

    Combines:
    - Spread (bid-ask)
    - Slippage (market impact)
    - Commission (broker fees)
    """

    def __init__(
        self,
        base_spread_bps: float = 2.0,
        base_commission_bps: float = 1.0,
        slippage_model: Optional[SlippageModel] = None,
    ):
        """
        Initialize transaction cost model.

        Args:
            base_spread_bps: Base spread in bps
            base_commission_bps: Commission per trade in bps
            slippage_model: Slippage model instance
        """
        self.base_spread_bps = base_spread_bps
        self.base_commission_bps = base_commission_bps
        self.slippage_model = slippage_model or CompositeSlippageModel()

    def estimate_costs(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float,
        spread_bps: Optional[float] = None,
    ) -> TransactionCosts:
        """
        Estimate total transaction costs.

        Args:
            trade_size: Trade size in shares
            price: Current price
            volume: Recent volume
            volatility: Current volatility
            spread_bps: Override spread estimate

        Returns:
            TransactionCosts breakdown
        """
        spread = spread_bps or self.base_spread_bps

        # Get slippage estimate
        slippage_est = self.slippage_model.estimate(
            trade_size, price, volume, volatility, spread
        )

        one_way = spread + slippage_est.market_impact_bps + self.base_commission_bps
        round_trip = one_way * 2

        return TransactionCosts(
            spread_bps=spread,
            slippage_bps=slippage_est.market_impact_bps,
            commission_bps=self.base_commission_bps,
            total_one_way_bps=one_way,
            total_round_trip_bps=round_trip,
        )

    def apply_entry_cost(
        self,
        price: float,
        direction: str,
        costs: TransactionCosts,
    ) -> float:
        """
        Apply entry transaction cost to price.

        Args:
            price: Raw entry price
            direction: LONG or SHORT
            costs: Transaction costs

        Returns:
            Adjusted entry price
        """
        cost_factor = costs.total_one_way_bps / 10000

        if direction == "LONG":
            # Buy at higher price (pay spread + slippage)
            return price * (1 + cost_factor)
        else:
            # Sell at lower price
            return price * (1 - cost_factor)

    def apply_exit_cost(
        self,
        price: float,
        direction: str,
        costs: TransactionCosts,
    ) -> float:
        """
        Apply exit transaction cost to price.

        Args:
            price: Raw exit price
            direction: LONG or SHORT
            costs: Transaction costs

        Returns:
            Adjusted exit price
        """
        cost_factor = costs.total_one_way_bps / 10000

        if direction == "LONG":
            # Sell at lower price
            return price * (1 - cost_factor)
        else:
            # Buy to cover at higher price
            return price * (1 + cost_factor)


class ImplementationShortfall:
    """
    Implementation shortfall analysis.

    Measures the difference between decision price and execution price.
    """

    def __init__(self):
        """Initialize implementation shortfall analyzer."""
        self._trades: list = []

    def record_trade(
        self,
        decision_price: float,
        execution_price: float,
        direction: str,
        size: float,
        timestamp: pd.Timestamp,
    ) -> float:
        """
        Record a trade and calculate shortfall.

        Args:
            decision_price: Price when decision was made
            execution_price: Actual execution price
            direction: LONG or SHORT
            size: Trade size
            timestamp: Execution time

        Returns:
            Shortfall in bps
        """
        if direction == "LONG":
            shortfall_pct = (execution_price - decision_price) / decision_price
        else:
            shortfall_pct = (decision_price - execution_price) / decision_price

        shortfall_bps = shortfall_pct * 10000

        self._trades.append(
            {
                "timestamp": timestamp,
                "direction": direction,
                "decision_price": decision_price,
                "execution_price": execution_price,
                "size": size,
                "shortfall_bps": shortfall_bps,
            }
        )

        return shortfall_bps

    def get_summary(self) -> dict:
        """Get implementation shortfall summary."""
        if not self._trades:
            return {"n_trades": 0}

        df = pd.DataFrame(self._trades)

        return {
            "n_trades": len(df),
            "avg_shortfall_bps": df["shortfall_bps"].mean(),
            "median_shortfall_bps": df["shortfall_bps"].median(),
            "std_shortfall_bps": df["shortfall_bps"].std(),
            "total_shortfall_bps": df["shortfall_bps"].sum(),
            "pct_positive": (df["shortfall_bps"] > 0).mean() * 100,
        }


class CostAwareLabeler:
    """
    Labels trades with cost-adjusted TP/SL levels.

    Accounts for transaction costs in label generation
    to avoid labeling unrealistic winners.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        tp_atr_multiple: float = 1.5,
        sl_atr_multiple: float = 1.0,
        max_hold_bars: int = 6,
    ):
        """
        Initialize cost-aware labeler.

        Args:
            cost_model: Transaction cost model
            tp_atr_multiple: Take profit in ATR multiples
            sl_atr_multiple: Stop loss in ATR multiples
            max_hold_bars: Maximum bars to hold
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.tp_atr_multiple = tp_atr_multiple
        self.sl_atr_multiple = sl_atr_multiple
        self.max_hold_bars = max_hold_bars

    def compute_adjusted_levels(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        volume: float,
        volatility: float,
        trade_size: float = 100,
    ) -> Tuple[float, float]:
        """
        Compute cost-adjusted TP and SL levels.

        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: LONG or SHORT
            volume: Recent volume
            volatility: Current volatility
            trade_size: Expected trade size

        Returns:
            Tuple of (adjusted_tp, adjusted_sl)
        """
        # Get cost estimate
        costs = self.cost_model.estimate_costs(
            trade_size, entry_price, volume, volatility
        )

        # Round trip cost adjustment
        cost_adjustment = entry_price * (costs.total_round_trip_bps / 10000)

        if direction == "LONG":
            # TP must overcome costs
            raw_tp = entry_price + (self.tp_atr_multiple * atr)
            adjusted_tp = raw_tp - cost_adjustment

            # SL is tighter after costs
            raw_sl = entry_price - (self.sl_atr_multiple * atr)
            adjusted_sl = raw_sl + cost_adjustment
        else:
            # Short direction
            raw_tp = entry_price - (self.tp_atr_multiple * atr)
            adjusted_tp = raw_tp + cost_adjustment

            raw_sl = entry_price + (self.sl_atr_multiple * atr)
            adjusted_sl = raw_sl - cost_adjustment

        return adjusted_tp, adjusted_sl

    def label_trades(
        self,
        df: pd.DataFrame,
        direction: str = "LONG",
        atr_column: str = "atr",
        volume_column: str = "volume",
    ) -> pd.DataFrame:
        """
        Label trades with cost-adjusted outcomes.

        Args:
            df: DataFrame with OHLCV and features
            direction: Trade direction
            atr_column: ATR column name
            volume_column: Volume column name

        Returns:
            DataFrame with cost-adjusted labels
        """
        result = df.copy()

        label_col = f"label_{direction.lower()}_cost_adj"
        result[label_col] = np.nan
        result[f"{label_col}_tp"] = np.nan
        result[f"{label_col}_sl"] = np.nan

        for i in range(len(result) - self.max_hold_bars):
            entry_idx = i
            entry_price = result["close"].iloc[entry_idx]
            atr = result[atr_column].iloc[entry_idx]
            volume = result[volume_column].iloc[entry_idx]
            volatility = result.get("atr_pct", pd.Series([1.0])).iloc[entry_idx]

            if pd.isna(atr) or atr <= 0:
                continue

            # Get cost-adjusted levels
            adj_tp, adj_sl = self.compute_adjusted_levels(
                entry_price, atr, direction, volume, volatility
            )

            result.iloc[entry_idx, result.columns.get_loc(f"{label_col}_tp")] = adj_tp
            result.iloc[entry_idx, result.columns.get_loc(f"{label_col}_sl")] = adj_sl

            # Evaluate outcome
            outcome = self._evaluate_trade(
                result.iloc[entry_idx + 1 : entry_idx + 1 + self.max_hold_bars],
                entry_price,
                adj_tp,
                adj_sl,
                direction,
            )

            result.iloc[entry_idx, result.columns.get_loc(label_col)] = outcome

        return result

    def _evaluate_trade(
        self,
        future_bars: pd.DataFrame,
        entry: float,
        tp: float,
        sl: float,
        direction: str,
    ) -> int:
        """Evaluate trade outcome."""
        for _, bar in future_bars.iterrows():
            if direction == "LONG":
                if bar["low"] <= sl:
                    return 0  # Loss
                if bar["high"] >= tp:
                    return 1  # Win
            else:
                if bar["high"] >= sl:
                    return 0  # Loss
                if bar["low"] <= tp:
                    return 1  # Win

        return 0  # Timeout = loss
