"""
Avellaneda-Stoikov Market Making Model.

Reference:
    Avellaneda, M., & Stoikov, S. (2008).
    "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.

Key Results:
    - Optimal bid/ask quotes for market makers
    - Inventory risk management through quote skewing
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class QuoteResult:
    """Optimal quote result."""

    bid_price: float
    ask_price: float
    spread: float
    reservation_price: float


def reservation_price(
    mid_price: float,
    inventory: int,
    volatility: float,
    time_remaining: float,
    gamma: float = 0.1,
) -> float:
    """
    Calculate reservation price (indifference price).

    r(s,q,t) = s - q * gamma * sigma^2 * (T-t)

    Args:
        mid_price: Current mid price (s)
        inventory: Current inventory position (q)
        volatility: Price volatility sigma
        time_remaining: Time until end of session (T-t)
        gamma: Risk aversion parameter

    Returns:
        Reservation price
    """
    adjustment = inventory * gamma * (volatility**2) * time_remaining
    return mid_price - adjustment


def optimal_spread(
    volatility: float,
    time_remaining: float,
    gamma: float = 0.1,
    kappa: float = 1.5,
) -> float:
    """
    Calculate optimal bid-ask spread.

    delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

    Args:
        volatility: Price volatility
        time_remaining: Time until end of session
        gamma: Risk aversion parameter
        kappa: Order arrival intensity parameter

    Returns:
        Optimal half-spread
    """
    inventory_risk = gamma * (volatility**2) * time_remaining
    adverse_selection = (2 / gamma) * np.log(1 + gamma / kappa)
    return inventory_risk + adverse_selection


class AvellanedaStoikovMM:
    """
    Avellaneda-Stoikov Market Maker.

    Example:
        mm = AvellanedaStoikovMM(gamma=0.1, kappa=1.5)
        quotes = mm.get_quotes(mid_price=100.0, inventory=5, volatility=0.20, time_remaining=0.5)
    """

    def __init__(
        self, gamma: float = 0.1, kappa: float = 1.5, max_inventory: int = 100
    ):
        self.gamma = gamma
        self.kappa = kappa
        self.max_inventory = max_inventory

    def get_quotes(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        time_remaining: float,
    ) -> QuoteResult:
        """Calculate optimal bid and ask quotes."""
        r = reservation_price(
            mid_price, inventory, volatility, time_remaining, self.gamma
        )
        delta = optimal_spread(volatility, time_remaining, self.gamma, self.kappa)

        return QuoteResult(
            bid_price=r - delta,
            ask_price=r + delta,
            spread=2 * delta,
            reservation_price=r,
        )

    def simulate_session(
        self,
        initial_mid: float,
        volatility: float,
        n_steps: int = 1000,
        seed: Optional[int] = None,
    ) -> dict:
        """Simulate a trading session."""
        if seed is not None:
            np.random.seed(seed)

        mid = initial_mid
        inventory = 0
        pnl = 0.0
        dt = 1.0 / n_steps
        sigma_dt = volatility * np.sqrt(dt / 252)

        for t in range(n_steps):
            time_remaining = 1.0 - t / n_steps
            quotes = self.get_quotes(mid, inventory, volatility, time_remaining)

            if (
                np.random.random() < self.kappa * dt * 0.5
                and inventory < self.max_inventory
            ):
                inventory += 1
                pnl -= quotes.bid_price

            if (
                np.random.random() < self.kappa * dt * 0.5
                and inventory > -self.max_inventory
            ):
                inventory -= 1
                pnl += quotes.ask_price

            mid += sigma_dt * np.random.randn()

        return {
            "final_pnl": pnl + inventory * mid,
            "final_inventory": inventory,
        }
