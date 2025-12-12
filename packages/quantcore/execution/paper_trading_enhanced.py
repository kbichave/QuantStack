# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced Paper Trading Engine with Order Book Simulation.

Production-grade paper trading with:
- Full order book simulation
- Market impact modeling
- Complete audit trail
- Execution quality metrics
- JSON logging for compliance review
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Thread, Event
import uuid

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.microstructure.order_book import OrderBook, Order, OrderType, Side
from quantcore.microstructure.matching_engine import (
    MatchingEngine,
    Fill,
    ExecutionReport,
)
from quantcore.microstructure.impact_models import ImpactModel, ImpactParams


@dataclass
class EnhancedPaperOrder:
    """Enhanced paper trading order with full details."""

    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: str  # "MARKET", "LIMIT"
    quantity: float
    limit_price: Optional[float] = None

    # Signal info
    signal_strength: float = 0.0
    signal_reason: str = ""

    # Execution details
    status: str = "PENDING"  # PENDING, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)

    # Timing
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Costs
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "signal_strength": self.signal_strength,
            "signal_reason": self.signal_reason,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "fills": self.fills,
            "submitted_at": (
                self.submitted_at.isoformat() if self.submitted_at else None
            ),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "commission": self.commission,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
        }


@dataclass
class OrderBookState:
    """Snapshot of order book state at a point in time."""

    timestamp: datetime
    symbol: str
    mid_price: float
    spread: float
    best_bid: float
    best_ask: float
    bid_depth_5: float  # Total volume at top 5 bid levels
    ask_depth_5: float  # Total volume at top 5 ask levels
    imbalance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "bid_depth_5": self.bid_depth_5,
            "ask_depth_5": self.ask_depth_5,
            "imbalance": self.imbalance,
        }


@dataclass
class EnhancedPaperPosition:
    """Position with full tracking."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    avg_entry_price: float
    entry_timestamp: datetime

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Execution quality
    total_slippage: float = 0.0
    total_impact: float = 0.0
    total_commission: float = 0.0

    # Order history
    order_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_slippage": self.total_slippage,
            "total_impact": self.total_impact,
            "total_commission": self.total_commission,
            "order_ids": self.order_ids,
        }


@dataclass
class ExecutionQualityMetrics:
    """Execution quality report."""

    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0

    total_volume: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_impact_bps: float = 0.0
    total_commission: float = 0.0

    # VWAP comparison
    vwap_price: float = 0.0
    avg_fill_vs_vwap_bps: float = 0.0

    # Arrival price comparison (Implementation Shortfall)
    implementation_shortfall_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnhancedPaperTradingEngine:
    """
    Enhanced paper trading engine with full order book simulation.

    Features:
    - Realistic order book with price-time priority
    - Market impact modeling (permanent + temporary)
    - Complete audit trail with JSON logs
    - Execution quality metrics
    - Compliance-ready reporting

    Example:
        engine = EnhancedPaperTradingEngine(
            initial_capital=100000,
            log_dir="paper_trading_logs/wti",
        )

        # Submit order
        order = engine.submit_order(
            symbol="WTI",
            side="BUY",
            quantity=100,
            signal_strength=0.8,
        )

        # Get execution report
        print(engine.get_execution_quality_report())

        # Save full audit log
        engine.save_audit_log()
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        daily_volume: float = 1_000_000.0,
        volatility: float = 0.02,
        tick_size: float = 0.01,
        commission_rate: float = 0.001,  # 10 bps
        log_dir: str = "paper_trading_logs",
        log_book_states: bool = True,
    ):
        """
        Initialize enhanced paper trading engine.

        Args:
            initial_capital: Starting capital
            daily_volume: Assumed daily volume for impact calculation
            volatility: Daily volatility for impact calculation
            tick_size: Minimum price increment
            commission_rate: Commission as fraction of trade value
            log_dir: Directory for audit logs
            log_book_states: Whether to log order book snapshots
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.daily_volume = daily_volume
        self.volatility = volatility
        self.tick_size = tick_size
        self.commission_rate = commission_rate
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_book_states = log_book_states

        # Microstructure components
        self.engines: Dict[str, MatchingEngine] = {}
        self.impact_model = ImpactModel(
            volatility=volatility,
            daily_volume=daily_volume,
        )

        # State
        self.positions: Dict[str, EnhancedPaperPosition] = {}
        self.orders: List[EnhancedPaperOrder] = []
        self.book_states: List[OrderBookState] = []
        self.equity_curve: List[Dict] = []

        # Metrics
        self.total_trades = 0
        self.total_volume = 0.0
        self.total_slippage = 0.0
        self.total_impact = 0.0
        self.total_commission = 0.0

        logger.info(f"Enhanced Paper Trading initialized: ${initial_capital:,.2f}")

    def _get_or_create_engine(
        self, symbol: str, current_price: float
    ) -> MatchingEngine:
        """Get or create matching engine for symbol."""
        if symbol not in self.engines:
            self.engines[symbol] = MatchingEngine()
            self._initialize_book(symbol, current_price)
        return self.engines[symbol]

    def _initialize_book(
        self,
        symbol: str,
        price: float,
        n_levels: int = 10,
        size_per_level: int = 100,
    ) -> None:
        """Initialize order book with liquidity."""
        engine = self.engines[symbol]
        engine.book = OrderBook()  # Reset

        spread_half = self.tick_size
        best_bid = price - spread_half
        best_ask = price + spread_half

        order_id = 1

        # Bid levels
        for i in range(n_levels):
            level_price = round(best_bid - i * self.tick_size, 4)
            engine.book.add_order(
                Order(
                    order_id=order_id,
                    side=Side.BID,
                    price=level_price,
                    quantity=size_per_level * (1 + i * 0.1),  # Deeper = more volume
                )
            )
            order_id += 1

        # Ask levels
        for i in range(n_levels):
            level_price = round(best_ask + i * self.tick_size, 4)
            engine.book.add_order(
                Order(
                    order_id=order_id,
                    side=Side.ASK,
                    price=level_price,
                    quantity=size_per_level * (1 + i * 0.1),
                )
            )
            order_id += 1

    def _log_book_state(self, symbol: str, timestamp: datetime) -> None:
        """Log current order book state."""
        if not self.log_book_states or symbol not in self.engines:
            return

        book = self.engines[symbol].book
        bid_depth, ask_depth = book.get_depth(5)

        bid_vol = sum(d[1] for d in bid_depth)
        ask_vol = sum(d[1] for d in ask_depth)

        state = OrderBookState(
            timestamp=timestamp,
            symbol=symbol,
            mid_price=book.mid_price or 0,
            spread=book.spread or 0,
            best_bid=book.best_bid or 0,
            best_ask=book.best_ask or 0,
            bid_depth_5=bid_vol,
            ask_depth_5=ask_vol,
            imbalance=book.get_imbalance(),
        )

        self.book_states.append(state)

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        signal_strength: float = 0.0,
        signal_reason: str = "",
    ) -> EnhancedPaperOrder:
        """
        Submit an order for paper execution.

        Args:
            symbol: Symbol to trade
            side: "BUY" or "SELL"
            quantity: Order quantity
            current_price: Current market price
            order_type: "MARKET" or "LIMIT"
            limit_price: Limit price (required for LIMIT orders)
            signal_strength: Strength of signal (0-1)
            signal_reason: Reason for the trade

        Returns:
            EnhancedPaperOrder with execution details
        """
        timestamp = datetime.now()
        order_id = str(uuid.uuid4())[:8]

        order = EnhancedPaperOrder(
            order_id=order_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            signal_strength=signal_strength,
            signal_reason=signal_reason,
            submitted_at=timestamp,
        )

        # Get or create engine
        engine = self._get_or_create_engine(symbol, current_price)

        # Log book state before execution
        self._log_book_state(symbol, timestamp)

        # Record arrival price for IS calculation
        arrival_price = engine.book.mid_price or current_price

        # Create order
        book_side = Side.BID if side == "BUY" else Side.ASK
        book_order_type = (
            OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT
        )
        book_order = Order(
            order_id=0,
            side=book_side,
            price=limit_price or 0,
            quantity=quantity,
            order_type=book_order_type,
        )

        # Execute
        report = engine.submit_order(book_order)

        # Process fills
        if report.fills:
            order.status = "FILLED" if report.total_filled >= quantity else "PARTIAL"
            order.filled_quantity = report.total_filled
            order.avg_fill_price = report.avg_price
            order.filled_at = datetime.now()

            # Record fills
            for fill in report.fills:
                order.fills.append(
                    {
                        "price": fill.price,
                        "quantity": fill.quantity,
                        "timestamp": timestamp.isoformat(),
                    }
                )

            # Calculate costs
            trade_value = report.total_filled * report.avg_price
            order.commission = trade_value * self.commission_rate

            # Slippage: difference from arrival price
            order.slippage = (
                abs(report.avg_price - arrival_price) / arrival_price * 10000
            )  # bps

            # Market impact
            impact = self.impact_model.estimate(
                order_size=quantity if side == "BUY" else -quantity,
                execution_time=0.01,
            )
            order.market_impact = abs(impact["total"]) * 10000  # bps

            # Update totals
            self.total_trades += 1
            self.total_volume += report.total_filled
            self.total_slippage += order.slippage * report.total_filled
            self.total_impact += order.market_impact * report.total_filled
            self.total_commission += order.commission

            # Update position
            self._update_position(
                symbol, side, report.total_filled, report.avg_price, order_id, order
            )

        else:
            order.status = "REJECTED" if order_type == "MARKET" else "PENDING"

        # Log book state after execution
        self._log_book_state(symbol, datetime.now())

        # Store order
        self.orders.append(order)

        logger.info(
            f"Order {order_id}: {side} {quantity} {symbol} @ {order.avg_fill_price:.2f} [{order.status}]"
        )

        return order

    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        order_id: str,
        order: EnhancedPaperOrder,
    ) -> None:
        """Update position after fill."""
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = EnhancedPaperPosition(
                symbol=symbol,
                side="LONG" if side == "BUY" else "SHORT",
                quantity=quantity,
                avg_entry_price=fill_price,
                entry_timestamp=datetime.now(),
                order_ids=[order_id],
            )
        else:
            pos = self.positions[symbol]
            pos.order_ids.append(order_id)

            # Same direction: add to position
            if (pos.side == "LONG" and side == "BUY") or (
                pos.side == "SHORT" and side == "SELL"
            ):
                # Average price update
                total_value = pos.quantity * pos.avg_entry_price + quantity * fill_price
                pos.quantity += quantity
                pos.avg_entry_price = total_value / pos.quantity

            else:
                # Opposite direction: reduce/close/reverse
                if quantity >= pos.quantity:
                    # Close or reverse
                    close_qty = pos.quantity

                    # Realize P&L
                    if pos.side == "LONG":
                        pnl = (fill_price - pos.avg_entry_price) * close_qty
                    else:
                        pnl = (pos.avg_entry_price - fill_price) * close_qty

                    pos.realized_pnl += pnl
                    self.capital += pnl - order.commission

                    remaining = quantity - close_qty
                    if remaining > 0:
                        # Reverse
                        pos.side = "LONG" if side == "BUY" else "SHORT"
                        pos.quantity = remaining
                        pos.avg_entry_price = fill_price
                        pos.entry_timestamp = datetime.now()
                    else:
                        # Closed completely
                        del self.positions[symbol]
                else:
                    # Partial close
                    if pos.side == "LONG":
                        pnl = (fill_price - pos.avg_entry_price) * quantity
                    else:
                        pnl = (pos.avg_entry_price - fill_price) * quantity

                    pos.realized_pnl += pnl
                    self.capital += pnl - order.commission
                    pos.quantity -= quantity

            # Track costs
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.total_slippage += order.slippage
                pos.total_impact += order.market_impact
                pos.total_commission += order.commission

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices and mark to market."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.current_price = price

                if pos.side == "LONG":
                    pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.avg_entry_price - price) * pos.quantity

        # Record equity curve point
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.equity_curve.append(
            {
                "timestamp": datetime.now().isoformat(),
                "capital": self.capital,
                "unrealized_pnl": total_unrealized,
                "equity": self.capital + total_unrealized,
                "n_positions": len(self.positions),
            }
        )

    def get_execution_quality_metrics(self) -> ExecutionQualityMetrics:
        """Calculate execution quality metrics."""
        metrics = ExecutionQualityMetrics()

        metrics.total_orders = len(self.orders)
        metrics.filled_orders = sum(1 for o in self.orders if o.status == "FILLED")
        metrics.rejected_orders = sum(1 for o in self.orders if o.status == "REJECTED")

        filled_orders = [o for o in self.orders if o.status == "FILLED"]
        if filled_orders:
            metrics.total_volume = sum(o.filled_quantity for o in filled_orders)
            metrics.avg_slippage_bps = (
                self.total_slippage / metrics.total_volume
                if metrics.total_volume > 0
                else 0
            )
            metrics.avg_impact_bps = (
                self.total_impact / metrics.total_volume
                if metrics.total_volume > 0
                else 0
            )
            metrics.total_commission = self.total_commission

            # VWAP
            total_value = sum(
                o.avg_fill_price * o.filled_quantity for o in filled_orders
            )
            metrics.vwap_price = total_value / metrics.total_volume

            # Implementation shortfall
            metrics.implementation_shortfall_bps = (
                metrics.avg_slippage_bps + metrics.avg_impact_bps
            )

        return metrics

    def get_execution_quality_report(self) -> str:
        """Generate human-readable execution quality report."""
        metrics = self.get_execution_quality_metrics()

        lines = [
            "=" * 60,
            "EXECUTION QUALITY REPORT",
            "=" * 60,
            "",
            f"Total Orders: {metrics.total_orders}",
            f"  Filled: {metrics.filled_orders}",
            f"  Rejected: {metrics.rejected_orders}",
            "",
            f"Total Volume: {metrics.total_volume:,.0f}",
            f"Total Commission: ${metrics.total_commission:,.2f}",
            "",
            "Execution Quality:",
            f"  Avg Slippage: {metrics.avg_slippage_bps:.2f} bps",
            f"  Avg Impact: {metrics.avg_impact_bps:.2f} bps",
            f"  Implementation Shortfall: {metrics.implementation_shortfall_bps:.2f} bps",
            "",
            f"VWAP: ${metrics.vwap_price:.2f}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    def save_audit_log(self, filename: Optional[str] = None) -> Path:
        """
        Save complete audit log to JSON file.

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_log_{timestamp}.json"

        filepath = self.log_dir / filename

        audit_data = {
            "metadata": {
                "initial_capital": self.initial_capital,
                "current_capital": self.capital,
                "created_at": datetime.now().isoformat(),
                "total_trades": self.total_trades,
                "total_volume": self.total_volume,
            },
            "execution_quality": self.get_execution_quality_metrics().to_dict(),
            "orders": [o.to_dict() for o in self.orders],
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "book_states": [s.to_dict() for s in self.book_states],
            "equity_curve": self.equity_curve,
        }

        with open(filepath, "w") as f:
            json.dump(audit_data, f, indent=2, default=str)

        logger.info(f"Audit log saved to {filepath}")
        return filepath

    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions."""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, pos in self.positions.items():
            data.append(
                {
                    "symbol": symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "total_pnl": pos.unrealized_pnl + pos.realized_pnl,
                }
            )

        return pd.DataFrame(data)

    def get_trade_log(self) -> pd.DataFrame:
        """Get DataFrame of all trades."""
        if not self.orders:
            return pd.DataFrame()

        data = []
        for order in self.orders:
            data.append(
                {
                    "order_id": order.order_id,
                    "timestamp": order.timestamp,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": order.status,
                    "fill_price": order.avg_fill_price,
                    "slippage_bps": order.slippage,
                    "impact_bps": order.market_impact,
                    "commission": order.commission,
                    "signal_strength": order.signal_strength,
                }
            )

        return pd.DataFrame(data)
