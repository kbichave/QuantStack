"""
Market Impact Models.

Models for estimating price impact of trades.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


def square_root_impact(
    order_size: float,
    daily_volume: float,
    volatility: float,
    eta: float = 0.1,
) -> float:
    """
    Square-root price impact.

    ΔP = η * σ * sign(Q) * sqrt(|Q|/V)

    Args:
        order_size: Signed order size
        daily_volume: Average daily volume
        volatility: Daily volatility
        eta: Impact coefficient

    Returns:
        Expected price impact
    """
    if daily_volume <= 0:
        return 0.0

    sign = np.sign(order_size)
    participation = abs(order_size) / daily_volume

    return eta * volatility * sign * np.sqrt(participation)


def estimate_kyle_lambda(
    price_changes: np.ndarray,
    order_flow: np.ndarray,
) -> float:
    """
    Estimate Kyle's lambda from trade data.

    Uses regression: ΔP = λ * Q + ε

    Args:
        price_changes: Array of price changes
        order_flow: Array of signed order flows

    Returns:
        Estimated lambda
    """
    if len(price_changes) < 10:
        return 0.1

    cov = np.cov(order_flow, price_changes)[0, 1]
    var = np.var(order_flow)

    return cov / var if var > 0 else 0.1


@dataclass
class ImpactParams:
    """Parameters for impact model."""

    eta: float = 0.1  # Temporary impact
    gamma: float = 0.05  # Permanent impact
    decay_rate: float = 0.1


class ImpactModel:
    """
    Comprehensive market impact model.

    Example:
        model = ImpactModel(volatility=0.02, daily_volume=1_000_000)
        impact = model.estimate(order_size=1000, execution_time=0.5)
        cost = model.execution_cost(order_size=1000, execution_time=0.5)
    """

    def __init__(
        self,
        volatility: float,
        daily_volume: float,
        params: Optional[ImpactParams] = None,
    ):
        self.volatility = volatility
        self.daily_volume = daily_volume
        self.params = params or ImpactParams()

    def estimate(
        self,
        order_size: float,
        execution_time: float = 1.0,
    ) -> Dict[str, float]:
        """Estimate total price impact."""
        if self.daily_volume <= 0 or execution_time <= 0:
            return {"permanent": 0, "temporary": 0, "total": 0}

        sign = np.sign(order_size)
        participation = abs(order_size) / self.daily_volume

        permanent = self.params.gamma * self.volatility * sign * np.sqrt(participation)

        execution_rate = abs(order_size) / (self.daily_volume * execution_time)
        temporary = self.params.eta * self.volatility * sign * np.sqrt(execution_rate)

        return {
            "permanent": permanent,
            "temporary": temporary,
            "total": permanent + temporary,
            "participation": participation,
        }

    def execution_cost(
        self,
        order_size: float,
        execution_time: float,
        price: float = 100.0,
    ) -> float:
        """Estimate total execution cost in dollars."""
        impact = self.estimate(order_size, execution_time)
        return abs(order_size) * abs(impact["total"])

    def optimal_execution_time(
        self,
        order_size: float,
        risk_aversion: float = 1e-6,
    ) -> float:
        """Calculate optimal execution time."""
        kappa = np.sqrt(risk_aversion * self.volatility**2 / (self.params.eta + 1e-10))
        optimal_T = 1.0 / (kappa + 1e-10)
        return np.clip(optimal_T, 0.1, 5.0)
