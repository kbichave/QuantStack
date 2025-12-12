"""
Market Simulator.

Combines order book, matching engine, and impact models for simulation.
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass

from quantcore.microstructure.order_book import OrderBook, Order, OrderType, Side
from quantcore.microstructure.matching_engine import MatchingEngine, Fill
from quantcore.microstructure.impact_models import ImpactModel
from quantcore.microstructure.execution_algos import ExecutionAlgo, TWAPExecutor


@dataclass
class SimulationResult:
    """Result of market simulation."""

    prices: np.ndarray
    spreads: np.ndarray
    volumes: np.ndarray
    fills: List[Fill]


class MarketSimulator:
    """
    Full market simulator with realistic dynamics.

    Example:
        sim = MarketSimulator(initial_price=100, volatility=0.02)
        result = sim.run(n_steps=1000)
        exec_result = sim.execute_algo(twap, order_size=1000, horizon=100)
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        tick_size: float = 0.01,
        daily_volume: float = 1_000_000,
    ):
        self.initial_price = initial_price
        self.volatility = volatility
        self.tick_size = tick_size
        self.daily_volume = daily_volume

        self.engine = MatchingEngine()
        self.impact_model = ImpactModel(volatility, daily_volume)
        self.current_price = initial_price

    def initialize_book(self, n_levels: int = 10, size_per_level: int = 100):
        """Initialize order book with resting liquidity."""
        self.engine = MatchingEngine()

        spread_ticks = 2
        best_bid = self.current_price - spread_ticks / 2 * self.tick_size
        best_ask = self.current_price + spread_ticks / 2 * self.tick_size

        order_id = 1

        for i in range(n_levels):
            price = best_bid - i * self.tick_size
            self.engine.book.add_order(
                Order(order_id, Side.BID, round(price, 4), size_per_level)
            )
            order_id += 1

        for i in range(n_levels):
            price = best_ask + i * self.tick_size
            self.engine.book.add_order(
                Order(order_id, Side.ASK, round(price, 4), size_per_level)
            )
            order_id += 1

    def step(self, n_orders: int = 10) -> Dict:
        """Advance simulation by one time step."""
        fills = []

        for _ in range(n_orders):
            side = Side.BID if np.random.random() < 0.5 else Side.ASK
            is_market = np.random.random() < 0.1

            if is_market:
                order = Order(0, side, 0, np.random.randint(1, 20), OrderType.MARKET)
            else:
                offset = np.random.choice(range(-3, 4)) * self.tick_size
                if side == Side.BID:
                    price = self.current_price - self.tick_size + offset
                else:
                    price = self.current_price + self.tick_size + offset
                order = Order(
                    0,
                    side,
                    round(price, 4),
                    np.random.randint(10, 100),
                    OrderType.LIMIT,
                )

            report = self.engine.submit_order(order)
            fills.extend(report.fills)

        if fills:
            net_flow = sum(
                f.quantity if f.side == Side.BID else -f.quantity for f in fills
            )
            impact = self.impact_model.estimate(net_flow, 0.01)
            self.current_price += impact["total"]

        self.current_price += self.volatility / np.sqrt(252) * np.random.randn()

        return {
            "price": self.current_price,
            "spread": self.engine.book.spread,
            "n_fills": len(fills),
        }

    def run(self, n_steps: int = 1000) -> SimulationResult:
        """Run full simulation."""
        self.initialize_book()

        prices = np.zeros(n_steps)
        spreads = np.zeros(n_steps)
        volumes = np.zeros(n_steps)
        all_fills = []

        for i in range(n_steps):
            stats = self.step()
            prices[i] = stats["price"]
            spreads[i] = stats["spread"] or 0
            volumes[i] = stats["n_fills"]
            all_fills.extend(self.engine.fills[-stats["n_fills"] :])

        return SimulationResult(
            prices=prices, spreads=spreads, volumes=volumes, fills=all_fills
        )

    def execute_algo(
        self, algo: ExecutionAlgo, order_size: float, side: Side, horizon: int
    ) -> Dict:
        """Execute an order using execution algorithm."""
        self.initialize_book(n_levels=10, size_per_level=200)

        plan = algo.create_plan(order_size, side, float(horizon))
        arrival_price = self.current_price

        fills = []
        total_executed = 0.0
        total_cost = 0.0
        slice_idx = 0

        for t in range(horizon):
            if slice_idx < len(plan.slices) and plan.slices[slice_idx].time <= t:
                qty = plan.slices[slice_idx].quantity
                order = Order(0, side, 0, qty, OrderType.MARKET)
                report = self.engine.submit_order(order)
                fills.extend(report.fills)
                total_executed += report.total_filled
                total_cost += sum(f.price * f.quantity for f in report.fills)
                slice_idx += 1

            self.step(n_orders=5)

        avg_price = total_cost / total_executed if total_executed > 0 else 0
        is_bps = (avg_price - arrival_price) / arrival_price * 10000
        if side == Side.ASK:
            is_bps = -is_bps

        return {
            "algo": plan.algo_name,
            "total_executed": total_executed,
            "avg_price": avg_price,
            "arrival_price": arrival_price,
            "implementation_shortfall_bps": is_bps,
        }
