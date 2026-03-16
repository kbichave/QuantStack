# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Simulated broker for historical simulation.

Provides a paper trading broker that:
- Tracks positions and cash
- Applies slippage and commissions
- Enforces risk limits (max position, max exposure, drawdown halt)
- Records all trades with timestamps
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Position:
    """A position in a single symbol."""

    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float = 0.0
    opened_at: Optional[date] = None

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return (self.current_price - self.avg_entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class Order:
    """A trade order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    fill_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0
    created_at: Optional[date] = None
    filled_at: Optional[date] = None
    rejection_reason: Optional[str] = None


@dataclass
class Trade:
    """A completed trade record."""

    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    date: date
    order_id: str
    pnl: Optional[float] = None  # Realized P&L for closes


@dataclass
class PortfolioState:
    """Snapshot of portfolio state."""

    date: date
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    max_drawdown: float
    positions: Dict[str, Position] = field(default_factory=dict)
    exposures: Dict[str, float] = field(default_factory=dict)  # symbol -> % of equity


class SimBroker:
    """
    Simulated broker for historical backtesting.

    Features:
    - Tracks positions and cash
    - Applies configurable slippage and commissions
    - Enforces risk limits
    - Records complete trade history
    - Computes drawdown and P&L attribution

    Usage:
        broker = SimBroker(
            initial_equity=100_000,
            slippage_bps=5,
            commission_per_share=0.005,
        )

        # Update prices
        broker.update_prices({"SPY": 450.0, "QQQ": 380.0})

        # Execute trades
        order = broker.submit_order("SPY", OrderSide.BUY, 100)

        # Get portfolio state
        state = broker.get_portfolio_state(current_date)
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        max_position_pct: float = 0.20,
        max_drawdown_halt_pct: float = 0.15,
        max_leverage: float = 1.0,
        max_daily_loss_pct: float = 0.05,
    ):
        """
        Initialize simulated broker.

        Args:
            initial_equity: Starting capital
            slippage_bps: Slippage in basis points
            commission_per_share: Commission per share
            max_position_pct: Maximum position size as % of equity
            max_drawdown_halt_pct: Halt trading if drawdown exceeds this
            max_leverage: Maximum portfolio leverage (1.0 = no leverage)
            max_daily_loss_pct: Halt trading if intraday equity falls this far
                                from day-open equity (e.g. 0.05 = 5%)
        """
        self.initial_equity = initial_equity
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.max_position_pct = max_position_pct
        self.max_drawdown_halt_pct = max_drawdown_halt_pct
        self.max_leverage = max_leverage
        self.max_daily_loss_pct = max_daily_loss_pct

        # Portfolio state
        self.cash = initial_equity
        self._positions: Dict[str, Position] = {}
        self._prices: Dict[str, float] = {}

        # Performance tracking
        self.high_water_mark = initial_equity
        self.realized_pnl = 0.0
        self._current_date: Optional[date] = None

        # Daily loss limit tracking — reset each new trading day
        self._day_open_equity: float = initial_equity
        self._last_reset_date: Optional[date] = None

        # Volume and volatility per symbol — populated by update_prices() when
        # a full market snapshot is provided. Used for Almgren-Chriss slippage.
        self._volumes: Dict[str, float] = {}        # daily volume per symbol
        self._volatilities: Dict[str, float] = {}   # annualised vol per symbol

        # History
        self._orders: List[Order] = []
        self._trades: List[Trade] = []
        self._daily_snapshots: List[PortfolioState] = []

        # Audit trail — every order attempt recorded with full lifecycle details.
        # This is separate from _orders so it is never filtered/modified.
        self._order_audit: List[Dict[str, Any]] = []

        # Trading state
        self._trading_halted = False
        self._halt_reason: Optional[str] = None

        logger.info(
            f"SimBroker initialized: equity=${initial_equity:,.0f}, "
            f"slippage={slippage_bps}bps, commission=${commission_per_share}/share, "
            f"daily_loss_limit={max_daily_loss_pct:.0%}"
        )

    def update_prices(
        self,
        prices: Dict[str, float],
        current_date: date,
        *,
        volumes: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update current prices (and optionally volume/volatility) for all symbols.

        Args:
            prices: Dict of symbol -> closing price
            current_date: Current simulation date
            volumes: Optional dict of symbol -> daily volume.
                     When provided, slippage switches from flat-bps to
                     Almgren-Chriss volume-dependent model.
            volatilities: Optional dict of symbol -> annualised volatility (0–1).
                          Used by the volume slippage model for market-impact calc.
        """
        self._prices.update(prices)

        # Update volume and volatility context if provided
        if volumes:
            self._volumes.update(volumes)
        if volatilities:
            self._volatilities.update(volatilities)

        # Reset daily loss baseline on each new trading day.
        # Must happen before position prices are updated so _day_open_equity
        # reflects the portfolio value at the start of the day, not mid-day.
        if current_date != self._last_reset_date:
            self._day_open_equity = self.get_equity()
            self._last_reset_date = current_date

        self._current_date = current_date

        # Update position prices
        for symbol, position in self._positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

        # Check drawdown halt
        equity = self.get_equity()
        if equity > self.high_water_mark:
            self.high_water_mark = equity

        drawdown = (self.high_water_mark - equity) / self.high_water_mark
        if drawdown >= self.max_drawdown_halt_pct and not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = f"Max drawdown exceeded: {drawdown:.1%}"
            logger.warning(f"Trading halted: {self._halt_reason}")

    def get_equity(self) -> float:
        """Get current total equity."""
        positions_value = sum(p.market_value for p in self._positions.values())
        return self.cash + positions_value

    def get_drawdown(self) -> float:
        """Get current drawdown as decimal."""
        equity = self.get_equity()
        if self.high_water_mark == 0:
            return 0.0
        return (self.high_water_mark - equity) / self.high_water_mark

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> Order:
        """
        Submit an order for execution.

        Args:
            symbol: Symbol to trade
            side: BUY or SELL
            quantity: Number of shares
            limit_price: Optional limit price (market order if None)

        Returns:
            Order object with execution status
        """
        order = Order(
            order_id=f"ord_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            created_at=self._current_date,
        )

        # Validate order
        rejection = self._validate_order(order)
        if rejection:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = rejection
            self._orders.append(order)
            self._record_audit(order, mid_price=self._prices.get(order.symbol))
            logger.warning(f"Order rejected: {rejection}")
            return order

        # Execute order
        self._execute_order(order)
        self._orders.append(order)
        self._record_audit(order, mid_price=self._prices.get(order.symbol))

        return order

    def _validate_order(self, order: Order) -> Optional[str]:
        """Validate order against risk limits. Returns rejection reason or None."""
        # Check trading halt (drawdown halt takes precedence)
        if self._trading_halted:
            return f"Trading halted: {self._halt_reason}"

        # Daily loss limit: halt new orders if today's equity has fallen too far
        # from the day-open baseline. Only applies to new BUY orders — we still
        # allow exits (SELL) so the desk can reduce risk when the limit is hit.
        if order.side == OrderSide.BUY and self._day_open_equity > 0:
            current_equity = self.get_equity()
            daily_loss_pct = (self._day_open_equity - current_equity) / self._day_open_equity
            if daily_loss_pct >= self.max_daily_loss_pct:
                return (
                    f"Daily loss limit reached: down {daily_loss_pct:.1%} from "
                    f"day-open ${self._day_open_equity:,.0f} "
                    f"(limit: {self.max_daily_loss_pct:.0%})"
                )

        # Check price available
        if order.symbol not in self._prices:
            return f"No price available for {order.symbol}"

        price = self._prices[order.symbol]
        equity = self.get_equity()

        # For buys, check buying power
        if order.side == OrderSide.BUY:
            required_cash = price * order.quantity * (1 + self.slippage_bps / 10000)
            required_cash += self.commission_per_share * order.quantity

            if required_cash > self.cash:
                return f"Insufficient cash: need ${required_cash:,.2f}, have ${self.cash:,.2f}"

            # Check position size limit
            new_position_value = price * order.quantity
            existing_position = self._positions.get(order.symbol)
            if existing_position:
                new_position_value += existing_position.market_value

            if new_position_value / equity > self.max_position_pct:
                return f"Position would exceed {self.max_position_pct:.0%} limit"

            # Check leverage
            total_exposure = sum(p.market_value for p in self._positions.values())
            new_exposure = total_exposure + price * order.quantity
            if new_exposure / equity > self.max_leverage:
                return f"Would exceed {self.max_leverage}x leverage limit"

        # For sells, check position exists
        if order.side == OrderSide.SELL:
            position = self._positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                available = position.quantity if position else 0
                return f"Insufficient position: need {order.quantity}, have {available}"

        return None

    def _execute_order(self, order: Order) -> None:
        """Execute a validated order."""
        price = self._prices[order.symbol]

        # Apply slippage — use Almgren-Chriss volume model if volume data is
        # available for this symbol, otherwise fall back to flat basis points.
        volume = self._volumes.get(order.symbol, 0.0)
        volatility = self._volatilities.get(order.symbol, 0.0)

        if volume > 0:
            try:
                from quantcore.execution.slippage import VolumeSlippageModel

                _vol_model = VolumeSlippageModel()
                estimate = _vol_model.estimate(
                    trade_size=float(order.quantity),
                    price=price,
                    volume=volume,
                    volatility=volatility,
                    spread_bps=self.slippage_bps,  # use config bps as spread floor
                )
                effective_bps = estimate.total_slippage_bps
            except ImportError:
                effective_bps = self.slippage_bps
        else:
            effective_bps = self.slippage_bps

        slippage = price * (effective_bps / 10_000)
        if order.side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        # Calculate commission
        commission = self.commission_per_share * order.quantity

        # Update order
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.filled_at = self._current_date

        # Update cash and positions
        trade_value = fill_price * order.quantity
        realized_pnl = None

        if order.side == OrderSide.BUY:
            self.cash -= trade_value + commission
            self._add_to_position(order.symbol, order.quantity, fill_price)
        else:
            self.cash += trade_value - commission
            realized_pnl = self._reduce_position(
                order.symbol, order.quantity, fill_price
            )
            self.realized_pnl += realized_pnl

        # Record trade
        trade = Trade(
            trade_id=f"trd_{uuid.uuid4().hex[:8]}",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            date=self._current_date,
            order_id=order.order_id,
            pnl=realized_pnl,
        )
        self._trades.append(trade)

        logger.info(
            f"Executed: {order.side.value.upper()} {order.quantity} {order.symbol} "
            f"@ ${fill_price:.2f} (commission: ${commission:.2f})"
        )

    def _add_to_position(self, symbol: str, quantity: int, price: float) -> None:
        """Add to or create a position."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            # Average in
            total_cost = pos.cost_basis + (quantity * price)
            total_qty = pos.quantity + quantity
            pos.avg_entry_price = total_cost / total_qty
            pos.quantity = total_qty
            pos.current_price = price
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                opened_at=self._current_date,
            )

    def _reduce_position(self, symbol: str, quantity: int, price: float) -> float:
        """Reduce position and return realized P&L."""
        pos = self._positions[symbol]
        realized_pnl = (price - pos.avg_entry_price) * quantity

        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self._positions[symbol]

        return realized_pnl

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position in a symbol."""
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        return self.submit_order(symbol, OrderSide.SELL, position.quantity)

    def close_all_positions(self) -> List[Order]:
        """Close all open positions."""
        orders = []
        for symbol in list(self._positions.keys()):
            order = self.close_position(symbol)
            if order:
                orders.append(order)
        return orders

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()

    def get_portfolio_state(
        self, current_date: Optional[date] = None
    ) -> PortfolioState:
        """
        Get current portfolio state snapshot.

        Args:
            current_date: Date for the snapshot (default: current)

        Returns:
            PortfolioState with all metrics
        """
        if current_date is None:
            current_date = self._current_date or date.today()

        equity = self.get_equity()
        positions_value = sum(p.market_value for p in self._positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())

        # Calculate exposures
        exposures = {}
        for symbol, pos in self._positions.items():
            exposures[symbol] = pos.market_value / equity if equity > 0 else 0.0

        return PortfolioState(
            date=current_date,
            equity=equity,
            cash=self.cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            max_drawdown=self.get_drawdown(),
            positions=self._positions.copy(),
            exposures=exposures,
        )

    def save_daily_snapshot(self) -> PortfolioState:
        """Save current state as daily snapshot."""
        state = self.get_portfolio_state()
        self._daily_snapshots.append(state)
        return state

    def get_trade_history(self) -> List[Trade]:
        """Get complete trade history."""
        return self._trades.copy()

    def get_daily_snapshots(self) -> List[PortfolioState]:
        """Get all daily snapshots."""
        return self._daily_snapshots.copy()

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get equity curve data for charting."""
        return [
            {
                "date": s.date.isoformat(),
                "equity": s.equity,
                "cash": s.cash,
                "drawdown": s.max_drawdown,
            }
            for s in self._daily_snapshots
        ]

    def reset_halt(self) -> None:
        """Reset trading halt (use with caution)."""
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Trading halt reset")

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._trading_halted

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        equity = self.get_equity()
        total_return = (equity - self.initial_equity) / self.initial_equity

        # Calculate win rate
        winning_trades = [t for t in self._trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self._trades if t.pnl and t.pnl < 0]
        total_closing_trades = len(winning_trades) + len(losing_trades)
        win_rate = (
            len(winning_trades) / total_closing_trades
            if total_closing_trades > 0
            else 0
        )

        return {
            "initial_equity": self.initial_equity,
            "final_equity": equity,
            "total_return": total_return,
            "realized_pnl": self.realized_pnl,
            "max_drawdown": self.get_drawdown(),
            "total_trades": len(self._trades),
            "win_rate": win_rate,
            "is_halted": self._trading_halted,
        }

    def _record_audit(self, order: Order, mid_price: Optional[float]) -> None:
        """Append a complete order lifecycle record to the audit trail."""
        slippage_bps_actual: Optional[float] = None
        if order.fill_price is not None and mid_price and mid_price > 0:
            diff = order.fill_price - mid_price
            # For sells, slippage is negative (worse price)
            slippage_bps_actual = round((diff / mid_price) * 10_000, 2)

        self._order_audit.append(
            {
                "order_id": order.order_id,
                "date": order.created_at.isoformat() if order.created_at else None,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty_requested": order.quantity,
                "qty_filled": order.filled_quantity,
                "fill_price": order.fill_price,
                "mid_price_at_order": mid_price,
                "slippage_bps_actual": slippage_bps_actual,
                "commission": order.commission,
                "status": order.status.value,
                "rejection_reason": order.rejection_reason,
            }
        )

    def get_order_audit(self) -> List[Dict[str, Any]]:
        """Return the complete order audit trail (all attempts, fills, and rejections)."""
        return self._order_audit.copy()

    def get_trade_analytics(self) -> Dict[str, Any]:
        """
        Compute institutional-grade trade analytics beyond the basic win rate.

        Returns:
            Dict with expectancy, profit factor, avg win/loss, consecutive
            streaks, hold duration stats, and per-symbol breakdown.
        """
        closed_trades = [t for t in self._trades if t.pnl is not None]

        if not closed_trades:
            return {"error": "No closed trades available"}

        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        n_total = len(closed_trades)

        win_rate = len(wins) / n_total if n_total else 0.0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

        # Expectancy: E[P&L per trade]
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Profit factor: gross profit / gross loss
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Largest single winner and loser
        largest_win = max((t.pnl for t in wins), default=0.0)
        largest_loss = min((t.pnl for t in losses), default=0.0)

        # Max consecutive wins and losses
        max_consec_wins = max_consec_losses = cur_wins = cur_losses = 0
        for t in closed_trades:
            if t.pnl > 0:
                cur_wins += 1
                cur_losses = 0
                max_consec_wins = max(max_consec_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_consec_losses = max(max_consec_losses, cur_losses)

        # Hold duration — requires both open and close trade records.
        # We approximate using entry date from Position.opened_at if available;
        # otherwise we use consecutive same-symbol trade pairs.
        durations: List[int] = []
        open_dates: Dict[str, date] = {}
        for t in self._trades:
            if t.side == "buy":
                open_dates[t.symbol] = t.date
            elif t.side == "sell" and t.symbol in open_dates:
                hold_days = (t.date - open_dates[t.symbol]).days
                if hold_days >= 0:
                    durations.append(hold_days)
                del open_dates[t.symbol]

        avg_hold_days = sum(durations) / len(durations) if durations else 0.0
        max_hold_days = max(durations, default=0)

        # Per-symbol breakdown
        by_symbol: Dict[str, Dict] = {}
        for t in closed_trades:
            entry = by_symbol.setdefault(t.symbol, {"trades": 0, "pnl": 0.0, "wins": 0})
            entry["trades"] += 1
            entry["pnl"] += t.pnl
            if t.pnl > 0:
                entry["wins"] += 1
        for sym, entry in by_symbol.items():
            n = entry["trades"]
            entry["win_rate"] = entry["wins"] / n if n else 0.0

        return {
            "total_closed_trades": n_total,
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_loss_ratio": round(abs(avg_win / avg_loss), 4) if avg_loss != 0 else None,
            "expectancy": round(expectancy, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_hold_days": round(avg_hold_days, 1),
            "max_hold_days": max_hold_days,
            "by_symbol": by_symbol,
        }

    def __repr__(self) -> str:
        equity = self.get_equity()
        return (
            f"SimBroker(equity=${equity:,.0f}, positions={len(self._positions)}, "
            f"halted={self._trading_halted})"
        )
