"""
Limit Order Book Implementation.

Price-time priority order book with bid and ask sides.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq


class Side(Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Order:
    """Order representation."""

    order_id: int
    side: Side
    price: float
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    remaining: float = field(default=None)

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.quantity


@dataclass
class Level:
    """Price level in order book."""

    price: float
    orders: List[Order] = field(default_factory=list)

    @property
    def total_quantity(self) -> float:
        return sum(o.remaining for o in self.orders)


class OrderBook:
    """
    Limit Order Book with price-time priority.

    Example:
        book = OrderBook()
        book.add_order(Order(1, Side.BID, 100.0, 10))
        book.add_order(Order(2, Side.ASK, 101.0, 5))
        print(f"Spread: {book.spread}")
    """

    def __init__(self):
        self.bids: Dict[float, Level] = {}
        self.asks: Dict[float, Level] = {}
        self.orders: Dict[int, Order] = {}
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []

    @property
    def best_bid(self) -> Optional[float]:
        while self._bid_prices:
            price = -self._bid_prices[0]
            if price in self.bids and self.bids[price].total_quantity > 0:
                return price
            heapq.heappop(self._bid_prices)
        return None

    @property
    def best_ask(self) -> Optional[float]:
        while self._ask_prices:
            price = self._ask_prices[0]
            if price in self.asks and self.asks[price].total_quantity > 0:
                return price
            heapq.heappop(self._ask_prices)
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return None

    def add_order(self, order: Order) -> None:
        """Add order to book."""
        if order.side == Side.BID:
            if order.price not in self.bids:
                self.bids[order.price] = Level(order.price)
                heapq.heappush(self._bid_prices, -order.price)
            self.bids[order.price].orders.append(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = Level(order.price)
                heapq.heappush(self._ask_prices, order.price)
            self.asks[order.price].orders.append(order)

        self.orders[order.order_id] = order

    def cancel_order(self, order_id: int) -> bool:
        """Cancel order by ID."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        levels = self.bids if order.side == Side.BID else self.asks

        if order.price in levels:
            level = levels[order.price]
            level.orders = [o for o in level.orders if o.order_id != order_id]
            if not level.orders:
                del levels[order.price]

        del self.orders[order_id]
        return True

    def get_depth(self, n_levels: int = 5) -> Tuple[List[Tuple], List[Tuple]]:
        """Get order book depth."""
        bid_levels = []
        ask_levels = []

        bid_prices = sorted(self.bids.keys(), reverse=True)[:n_levels]
        for price in bid_prices:
            level = self.bids[price]
            bid_levels.append((price, level.total_quantity, len(level.orders)))

        ask_prices = sorted(self.asks.keys())[:n_levels]
        for price in ask_prices:
            level = self.asks[price]
            ask_levels.append((price, level.total_quantity, len(level.orders)))

        return bid_levels, ask_levels

    def get_imbalance(self, n_levels: int = 5) -> float:
        """Compute order imbalance at top N levels."""
        bid_vol = sum(
            self.bids[p].total_quantity
            for p in sorted(self.bids.keys(), reverse=True)[:n_levels]
        )
        ask_vol = sum(
            self.asks[p].total_quantity for p in sorted(self.asks.keys())[:n_levels]
        )

        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
