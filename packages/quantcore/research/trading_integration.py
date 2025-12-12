"""
Trading Integration Module.

Bridges quant research modules (math_models, microstructure, paper_replications)
into actual trading decisions for WTI and equity pipelines.

Usage:
    from quantcore.research.trading_integration import TradingEnhancer
    enhancer = TradingEnhancer(volatility=0.02, daily_volume=1_000_000)

    # Kalman-filtered spread estimate
    spread_estimate = enhancer.filter_spread(spread_series)

    # Position size with impact
    position = enhancer.compute_position_with_impact(signal, max_position=100)

    # Optimal execution schedule
    schedule = enhancer.optimal_execution_schedule(shares=1000, horizon=20)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Import quant modules
from quantcore.math.kalman_filter import KalmanFilter, LocalLevelModel
from quantcore.math.stochastic_vol import HestonModel
from microstructure.impact_models import ImpactModel, ImpactParams
from quantcore.research.paper_replications.almgren_chriss import AlmgrenChrissExecutor
from quantcore.research.paper_replications.avellaneda_stoikov import reservation_price


@dataclass
class EnhancedSignal:
    """Enhanced trading signal with impact-adjusted sizing."""

    raw_signal: float
    adjusted_signal: float
    position_size: float
    execution_schedule: Optional[List[float]]
    estimated_cost_bps: float
    spread_estimate: Optional[float]


class TradingEnhancer:
    """
    Enhances trading decisions using quant research modules.

    Integrates:
    - Kalman filter for spread/level estimation
    - Impact models for position sizing
    - Almgren-Chriss for execution scheduling
    """

    def __init__(
        self,
        volatility: float = 0.02,
        daily_volume: float = 1_000_000,
        impact_eta: float = 2.5e-7,
        impact_gamma: float = 2.5e-7,
        risk_aversion: float = 1e-6,
    ):
        """
        Initialize trading enhancer.

        Args:
            volatility: Daily volatility
            daily_volume: Average daily volume
            impact_eta: Temporary impact coefficient
            impact_gamma: Permanent impact coefficient
            risk_aversion: Risk aversion for execution
        """
        self.volatility = volatility
        self.daily_volume = daily_volume
        self.risk_aversion = risk_aversion

        # Initialize models
        self.impact_model = ImpactModel(
            volatility=volatility,
            daily_volume=daily_volume,
            params=ImpactParams(eta=impact_eta, gamma=impact_gamma),
        )

        self.executor = AlmgrenChrissExecutor(
            volatility=volatility,
            daily_volume=daily_volume,
            eta=impact_eta,
            gamma=impact_gamma,
        )

        # Kalman filter for spread estimation
        self.spread_filter = LocalLevelModel(
            sigma_eta=volatility * 0.1,  # State noise
            sigma_epsilon=volatility,  # Observation noise
        )

        self._spread_states = None

    def filter_spread(self, spread_series: pd.Series) -> pd.Series:
        """
        Apply Kalman filter to spread for smoother estimates.

        Args:
            spread_series: Raw spread time series

        Returns:
            Kalman-filtered spread estimates
        """
        values = spread_series.dropna().values
        states = self.spread_filter.filter(values)
        self._spread_states = states

        filtered = pd.Series(
            [s.x[0] for s in states],
            index=spread_series.dropna().index,
            name="filtered_spread",
        )

        return filtered.reindex(spread_series.index).ffill()

    def get_spread_confidence(self) -> Optional[np.ndarray]:
        """Get confidence intervals for spread estimates."""
        if self._spread_states is None:
            return None

        stds = np.array([np.sqrt(s.P[0, 0]) for s in self._spread_states])
        return stds

    def estimate_execution_cost(
        self,
        order_size: float,
        execution_time: float = 1.0,
    ) -> Dict[str, float]:
        """
        Estimate execution cost for a given order.

        Args:
            order_size: Number of shares/contracts
            execution_time: Days to execute

        Returns:
            Dict with cost components
        """
        impact = self.impact_model.estimate(order_size, execution_time)

        return {
            "permanent_impact_bps": impact["permanent"] * 10000,
            "temporary_impact_bps": impact["temporary"] * 10000,
            "total_impact_bps": impact["total"] * 10000,
            "participation_rate": impact["participation"],
        }

    def compute_position_with_impact(
        self,
        signal_strength: float,
        max_position: float,
        current_position: float = 0,
    ) -> float:
        """
        Compute position size accounting for market impact.

        Reduces position size for larger trades to minimize impact costs.

        Args:
            signal_strength: Signal in [-1, 1]
            max_position: Maximum allowed position
            current_position: Current position

        Returns:
            Target position accounting for impact
        """
        # Raw target position
        raw_target = signal_strength * max_position
        trade_size = raw_target - current_position

        if abs(trade_size) < 1:
            return raw_target

        # Estimate impact
        impact = self.impact_model.estimate(trade_size, 1.0)
        impact_cost = abs(impact["total"])

        # Reduce position proportionally to impact
        # If impact > 50bps, reduce by up to 50%
        reduction_factor = max(0.5, 1.0 - impact_cost * 100)

        adjusted_trade = trade_size * reduction_factor
        return current_position + adjusted_trade

    def optimal_execution_schedule(
        self,
        shares: float,
        horizon: int = 20,
        risk_aversion: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute optimal execution schedule using Almgren-Chriss.

        Args:
            shares: Total shares to execute
            horizon: Number of periods
            risk_aversion: Override default risk aversion

        Returns:
            Tuple of (schedule, metadata)
        """
        ra = risk_aversion or self.risk_aversion
        result = self.executor.optimal_schedule(shares, horizon, ra)

        metadata = {
            "expected_cost": result.expected_cost,
            "variance": result.variance,
            "trajectory": result.trajectory,
        }

        return result.trade_schedule, metadata

    def compute_reservation_price(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float,
        gamma: float = 0.1,
    ) -> float:
        """
        Compute Avellaneda-Stoikov reservation price.

        Useful for market-making style position management.

        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            time_remaining: Time until end of session (0-1)
            gamma: Risk aversion

        Returns:
            Reservation price
        """
        return reservation_price(
            mid_price=mid_price,
            inventory=inventory,
            volatility=self.volatility,
            time_remaining=time_remaining,
            gamma=gamma,
        )

    def enhance_signal(
        self,
        raw_signal: float,
        max_position: float = 100,
        current_position: float = 0,
        spread_value: Optional[float] = None,
        execution_horizon: int = 20,
    ) -> EnhancedSignal:
        """
        Fully enhance a trading signal with all adjustments.

        Args:
            raw_signal: Original signal [-1, 1]
            max_position: Max position size
            current_position: Current position
            spread_value: Current spread (for filtering)
            execution_horizon: Periods for execution

        Returns:
            EnhancedSignal with all adjustments
        """
        # Compute impact-adjusted position
        position = self.compute_position_with_impact(
            raw_signal, max_position, current_position
        )
        trade_size = position - current_position

        # Compute execution cost
        cost_info = self.estimate_execution_cost(abs(trade_size), 1.0)

        # Generate execution schedule if trade is significant
        schedule = None
        if abs(trade_size) > max_position * 0.1:
            schedule, _ = self.optimal_execution_schedule(
                abs(trade_size), execution_horizon
            )
            if trade_size < 0:
                schedule = -schedule

        # Adjust signal based on costs
        adjusted_signal = raw_signal
        if cost_info["total_impact_bps"] > 10:
            # Reduce signal if costs are high
            adjusted_signal *= max(0.5, 1 - cost_info["total_impact_bps"] / 100)

        return EnhancedSignal(
            raw_signal=raw_signal,
            adjusted_signal=adjusted_signal,
            position_size=position,
            execution_schedule=schedule.tolist() if schedule is not None else None,
            estimated_cost_bps=cost_info["total_impact_bps"],
            spread_estimate=spread_value,
        )


def create_wti_enhancer(
    volatility: float = 0.025,
    daily_volume: float = 500_000,
) -> TradingEnhancer:
    """Create enhancer with WTI-appropriate defaults."""
    return TradingEnhancer(
        volatility=volatility,
        daily_volume=daily_volume,
        impact_eta=5e-7,  # Commodities have higher impact
        impact_gamma=5e-7,
        risk_aversion=2e-6,
    )


def create_equity_enhancer(
    volatility: float = 0.02,
    daily_volume: float = 1_000_000,
) -> TradingEnhancer:
    """Create enhancer with equity-appropriate defaults."""
    return TradingEnhancer(
        volatility=volatility,
        daily_volume=daily_volume,
        impact_eta=2.5e-7,
        impact_gamma=2.5e-7,
        risk_aversion=1e-6,
    )
