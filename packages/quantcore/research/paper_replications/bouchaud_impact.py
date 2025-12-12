"""
Bouchaud Price Impact Models.

References:
    Bouchaud, J.P., Farmer, J.D., & Lillo, F. (2009).
    "How markets slowly digest changes in supply and demand."

Key Results:
    - Price impact is concave (square-root law)
    - Impact decays over time (propagator model)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


def propagator_model(t: np.ndarray, beta: float = 0.5, tau: float = 10.0) -> np.ndarray:
    """
    Bouchaud propagator (response function).

    G(t) = G_0 * (tau / (tau + t))^beta

    Args:
        t: Time array
        beta: Decay exponent (typically 0.5-0.7)
        tau: Characteristic decay time

    Returns:
        Propagator values G(t)
    """
    return (tau / (tau + t)) ** beta


@dataclass
class ImpactResult:
    """Result of impact calculation."""

    immediate_impact: float
    permanent_impact: float
    transient_impact: float
    total_impact: float


class BouchaudImpactModel:
    """
    Full Bouchaud impact model with propagator dynamics.

    Example:
        model = BouchaudImpactModel(eta=0.1, beta=0.5, tau=10)
        impact = model.compute_impact(order_size=1000, daily_volume=1_000_000, volatility=0.02)
    """

    def __init__(
        self,
        eta: float = 0.1,
        beta: float = 0.5,
        tau: float = 10.0,
        permanent_fraction: float = 0.5,
    ):
        self.eta = eta
        self.beta = beta
        self.tau = tau
        self.permanent_fraction = permanent_fraction
        self.trade_history: List[Tuple[float, float, float]] = []

    def compute_impact(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """Compute immediate impact of a trade."""
        if daily_volume <= 0:
            return 0.0

        sign = np.sign(order_size)
        participation = abs(order_size) / daily_volume
        return self.eta * volatility * sign * np.sqrt(participation)

    def compute_trajectory(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
        horizon: int = 50,
    ) -> np.ndarray:
        """Compute impact trajectory over time."""
        immediate = self.compute_impact(order_size, daily_volume, volatility)
        times = np.arange(horizon)

        permanent = immediate * self.permanent_fraction
        initial_transient = immediate * (1 - self.permanent_fraction)

        trajectory = permanent + initial_transient * propagator_model(
            times, self.beta, self.tau
        )
        return trajectory

    def add_trade(self, time: float, order_size: float, daily_volume: float) -> None:
        """Add a trade to history for aggregate impact."""
        self.trade_history.append((time, order_size, daily_volume))

    def aggregate_impact(self, current_time: float, volatility: float) -> float:
        """Compute aggregate impact from all historical trades."""
        total_impact = 0.0

        for trade_time, order_size, daily_volume in self.trade_history:
            time_elapsed = current_time - trade_time
            if time_elapsed < 0:
                continue

            immediate = self.compute_impact(order_size, daily_volume, volatility)
            permanent = immediate * self.permanent_fraction
            transient = (
                immediate
                * (1 - self.permanent_fraction)
                * propagator_model(np.array([time_elapsed]), self.beta, self.tau)[0]
            )
            total_impact += permanent + transient

        return total_impact
