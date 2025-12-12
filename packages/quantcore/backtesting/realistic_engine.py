# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Realistic Backtesting Engine with Microstructure Simulation.

Production-grade backtesting that simulates:
- Order book dynamics with price-time priority
- Market impact (square root, Kyle's lambda)
- Realistic slippage and fills
- Execution algorithms (TWAP, VWAP, IS)
- Full audit trail with order book logs
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.microstructure.order_book import OrderBook, Order, OrderType, Side
from quantcore.microstructure.matching_engine import (
    MatchingEngine,
    Fill,
    ExecutionReport,
)
from quantcore.microstructure.impact_models import (
    ImpactModel,
    ImpactParams,
    square_root_impact,
)
from quantcore.microstructure.execution_algos import (
    TWAPExecutor,
    VWAPExecutor,
    ISExecutor,
)


@dataclass
class RealisticBacktestConfig:
    """Configuration for realistic backtesting."""

    initial_capital: float = 100_000.0
    daily_volume: float = 1_000_000.0  # Average daily volume
    volatility: float = 0.02  # Daily volatility
    tick_size: float = 0.01
    n_book_levels: int = 10
    size_per_level: int = 100

    # Impact parameters
    eta: float = 0.1  # Temporary impact coefficient
    gamma: float = 0.05  # Permanent impact coefficient

    # Execution
    execution_algo: str = "market"  # "market", "twap", "vwap", "is"
    execution_horizon: int = 10  # Bars for algo execution

    # Logging
    log_order_book: bool = True
    log_fills: bool = True
    log_path: Optional[str] = None


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state."""

    timestamp: datetime
    mid_price: float
    spread: float
    best_bid: float
    best_ask: float
    bid_depth: List[Tuple[float, float, int]]  # price, qty, n_orders
    ask_depth: List[Tuple[float, float, int]]
    imbalance: float


@dataclass
class FillRecord:
    """Record of a trade execution."""

    timestamp: datetime
    order_id: int
    side: str
    price: float
    quantity: float
    impact: float
    slippage: float
    arrival_price: float


@dataclass
class RealisticBacktestResult:
    """Result from realistic backtest."""

    # Performance
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float

    # Execution quality
    total_trades: int
    total_volume: float
    avg_slippage_bps: float
    avg_impact_bps: float
    total_execution_cost: float
    implementation_shortfall: float

    # Trade stats
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float

    # Logs
    order_book_logs: List[OrderBookSnapshot] = field(default_factory=list)
    fill_logs: List[FillRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "performance": {
                "initial_capital": self.initial_capital,
                "final_capital": self.final_capital,
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
            },
            "execution_quality": {
                "total_trades": self.total_trades,
                "total_volume": self.total_volume,
                "avg_slippage_bps": self.avg_slippage_bps,
                "avg_impact_bps": self.avg_impact_bps,
                "total_execution_cost": self.total_execution_cost,
                "implementation_shortfall": self.implementation_shortfall,
            },
            "trade_stats": {
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "avg_trade_pnl": self.avg_trade_pnl,
            },
        }


class RealisticBacktestEngine:
    """
    Production-grade backtesting engine with microstructure simulation.

    Features:
    - Full order book simulation with price-time priority
    - Realistic market impact modeling (permanent + temporary)
    - Multiple execution algorithms (Market, TWAP, VWAP, IS)
    - Complete audit trail with order book snapshots
    - Execution quality metrics (slippage, IS, VWAP vs actual)

    Example:
        config = RealisticBacktestConfig(
            initial_capital=100000,
            execution_algo="twap",
            log_order_book=True,
        )
        engine = RealisticBacktestEngine(config)
        result = engine.run(signals_df, price_df)
        engine.save_logs("logs/backtest_audit.json")
    """

    def __init__(self, config: Optional[RealisticBacktestConfig] = None):
        """Initialize engine with configuration."""
        self.config = config or RealisticBacktestConfig()

        # Microstructure components
        self.engine = MatchingEngine()
        self.impact_model = ImpactModel(
            volatility=self.config.volatility,
            daily_volume=self.config.daily_volume,
            params=ImpactParams(
                eta=self.config.eta,
                gamma=self.config.gamma,
            ),
        )

        # State
        self.capital = self.config.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.current_price = 100.0

        # Logging
        self.order_book_logs: List[OrderBookSnapshot] = []
        self.fill_logs: List[FillRecord] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.config.initial_capital]

        # Execution algo instances
        self._executors = {
            "twap": TWAPExecutor(n_slices=self.config.execution_horizon),
            "vwap": VWAPExecutor(n_slices=self.config.execution_horizon),
            "is": ISExecutor(
                volatility=self.config.volatility,
                daily_volume=self.config.daily_volume,
                n_slices=self.config.execution_horizon,
            ),
        }

    def run(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> RealisticBacktestResult:
        """
        Run realistic backtest with order book simulation.

        Args:
            signals: DataFrame with signal columns (signal, signal_direction)
            price_data: OHLCV price data

        Returns:
            RealisticBacktestResult with metrics and logs
        """
        logger.info("Starting realistic backtest with order book simulation...")

        # Initialize
        self.capital = self.config.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.order_book_logs = []
        self.fill_logs = []
        self.trades = []
        self.equity_curve = [self.config.initial_capital]

        total_slippage = 0.0
        total_impact = 0.0
        total_volume = 0.0
        n_fills = 0

        # Align data
        common_idx = signals.index.intersection(price_data.index)
        if len(common_idx) == 0:
            logger.warning("No common indices between signals and price data")
            return self._create_empty_result()

        signals = signals.loc[common_idx]
        prices = price_data.loc[common_idx]

        for i in range(1, len(common_idx)):
            idx = common_idx[i]
            prev_idx = common_idx[i - 1]

            # Update price
            self.current_price = float(prices.loc[idx, "close"])

            # Initialize order book at current price
            self._initialize_book_at_price(self.current_price)

            # Log order book state
            if self.config.log_order_book:
                self._log_order_book_state(idx)

            # Get signal
            signal = signals.loc[idx].get("signal", 0)
            direction = signals.loc[idx].get("signal_direction", "NONE")

            # Convert string direction to numeric
            if isinstance(direction, str):
                direction_map = {
                    "LONG": 1,
                    "SHORT": -1,
                    "NONE": 0,
                    "NEUTRAL": 0,
                    "FLAT": 0,
                }
                direction = direction_map.get(direction.upper(), 0)

            # Determine target position
            target_position = 0
            if signal == 1 and direction == 1:
                target_position = 1
            elif signal == 1 and direction == -1:
                target_position = -1

            # Execute position change
            if target_position != self.position:
                order_size = target_position - self.position

                # Calculate arrival price before execution
                arrival_price = self.engine.book.mid_price or self.current_price

                # Execute order
                fills, exec_report = self._execute_order(order_size, idx)

                # Calculate slippage and impact
                if fills:
                    avg_fill_price = sum(f.price * f.quantity for f in fills) / sum(
                        f.quantity for f in fills
                    )
                    slippage = (
                        abs(avg_fill_price - arrival_price) / arrival_price * 10000
                    )  # bps

                    # Market impact
                    impact = self.impact_model.estimate(order_size, execution_time=0.1)
                    impact_bps = abs(impact["total"]) * 10000

                    total_slippage += slippage * sum(f.quantity for f in fills)
                    total_impact += impact_bps * sum(f.quantity for f in fills)
                    total_volume += sum(f.quantity for f in fills)
                    n_fills += len(fills)

                    # Log fills
                    if self.config.log_fills:
                        for fill in fills:
                            self.fill_logs.append(
                                FillRecord(
                                    timestamp=idx,
                                    order_id=fill.aggressor_id,
                                    side=fill.side.value,
                                    price=fill.price,
                                    quantity=fill.quantity,
                                    impact=impact_bps,
                                    slippage=slippage,
                                    arrival_price=arrival_price,
                                )
                            )

                    # Update position
                    if self.position == 0:
                        self.entry_price = avg_fill_price

                    # Record trade if closing position
                    if target_position == 0 and self.position != 0:
                        pnl = (avg_fill_price - self.entry_price) * self.position
                        pnl -= (
                            abs(order_size) * self.config.tick_size * 2
                        )  # Commission estimate
                        self.capital += pnl
                        self.trades.append(
                            {
                                "entry_price": self.entry_price,
                                "exit_price": avg_fill_price,
                                "pnl": pnl,
                                "direction": "LONG" if self.position > 0 else "SHORT",
                                "slippage_bps": slippage,
                                "impact_bps": impact_bps,
                            }
                        )

                    self.position = target_position

            # Mark to market
            if self.position != 0:
                mtm_pnl = (self.current_price - self.entry_price) * self.position
                mtm_capital = self.capital + mtm_pnl
            else:
                mtm_capital = self.capital

            self.equity_curve.append(mtm_capital)

        # Calculate result metrics
        return self._calculate_result(
            total_slippage, total_impact, total_volume, n_fills
        )

    def _initialize_book_at_price(self, price: float) -> None:
        """Initialize order book with liquidity around price."""
        self.engine = MatchingEngine()

        spread_half = self.config.tick_size
        best_bid = price - spread_half
        best_ask = price + spread_half

        order_id = 1

        # Add bid levels
        for i in range(self.config.n_book_levels):
            level_price = best_bid - i * self.config.tick_size
            self.engine.book.add_order(
                Order(
                    order_id=order_id,
                    side=Side.BID,
                    price=round(level_price, 4),
                    quantity=self.config.size_per_level,
                )
            )
            order_id += 1

        # Add ask levels
        for i in range(self.config.n_book_levels):
            level_price = best_ask + i * self.config.tick_size
            self.engine.book.add_order(
                Order(
                    order_id=order_id,
                    side=Side.ASK,
                    price=round(level_price, 4),
                    quantity=self.config.size_per_level,
                )
            )
            order_id += 1

    def _log_order_book_state(self, timestamp: datetime) -> None:
        """Log current order book state."""
        bid_depth, ask_depth = self.engine.book.get_depth(5)

        self.order_book_logs.append(
            OrderBookSnapshot(
                timestamp=timestamp,
                mid_price=self.engine.book.mid_price or 0,
                spread=self.engine.book.spread or 0,
                best_bid=self.engine.book.best_bid or 0,
                best_ask=self.engine.book.best_ask or 0,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                imbalance=self.engine.book.get_imbalance(),
            )
        )

    def _execute_order(
        self,
        order_size: int,
        timestamp: datetime,
    ) -> Tuple[List[Fill], ExecutionReport]:
        """Execute order using configured execution algorithm."""
        side = Side.BID if order_size > 0 else Side.ASK
        quantity = abs(order_size)

        if self.config.execution_algo == "market":
            # Market order
            order = Order(
                order_id=0,
                side=side,
                price=0,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
            report = self.engine.submit_order(order)
            return report.fills, report

        elif self.config.execution_algo in self._executors:
            # Use execution algorithm
            executor = self._executors[self.config.execution_algo]
            plan = executor.create_plan(
                order_size=quantity,
                side=side,
                horizon=1.0,  # Normalized horizon
            )

            # Execute slices
            all_fills = []
            for slice_item in plan.slices:
                order = Order(
                    order_id=0,
                    side=side,
                    price=0,
                    quantity=slice_item.quantity,
                    order_type=OrderType.MARKET,
                )
                report = self.engine.submit_order(order)
                all_fills.extend(report.fills)

            # Create combined report
            total_filled = sum(f.quantity for f in all_fills)
            avg_price = (
                sum(f.price * f.quantity for f in all_fills) / total_filled
                if total_filled > 0
                else 0
            )

            combined_report = ExecutionReport(
                order_id=0,
                fills=all_fills,
                total_filled=total_filled,
                avg_price=avg_price,
                remaining=0,
                status="filled",
            )
            return all_fills, combined_report

        return [], ExecutionReport(0, [], 0, 0, 0, "invalid")

    def _calculate_result(
        self,
        total_slippage: float,
        total_impact: float,
        total_volume: float,
        n_fills: int,
    ) -> RealisticBacktestResult:
        """Calculate final backtest result."""
        equity = np.array(self.equity_curve)

        # Returns and Sharpe
        if len(equity) > 1:
            returns = np.diff(equity) / (equity[:-1] + 1e-8)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-8)
        max_dd = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

        # Total return
        total_return = (
            (equity[-1] - equity[0]) / equity[0] * 100 if len(equity) > 0 else 0
        )

        # Trade stats
        total_trades = len(self.trades)
        winners = sum(1 for t in self.trades if t["pnl"] > 0)
        win_rate = winners / total_trades * 100 if total_trades > 0 else 0

        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade_pnl = (
            sum(t["pnl"] for t in self.trades) / total_trades if total_trades > 0 else 0
        )

        # Execution quality
        avg_slippage = total_slippage / total_volume if total_volume > 0 else 0
        avg_impact = total_impact / total_volume if total_volume > 0 else 0

        # Implementation shortfall (sum of slippage + impact)
        implementation_shortfall = avg_slippage + avg_impact

        return RealisticBacktestResult(
            initial_capital=self.config.initial_capital,
            final_capital=(
                equity[-1] if len(equity) > 0 else self.config.initial_capital
            ),
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=total_trades,
            total_volume=total_volume,
            avg_slippage_bps=avg_slippage,
            avg_impact_bps=avg_impact,
            total_execution_cost=total_slippage + total_impact,
            implementation_shortfall=implementation_shortfall,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            order_book_logs=self.order_book_logs,
            fill_logs=self.fill_logs,
            equity_curve=self.equity_curve,
        )

    def _create_empty_result(self) -> RealisticBacktestResult:
        """Create empty result for error cases."""
        return RealisticBacktestResult(
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,
            total_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            total_trades=0,
            total_volume=0,
            avg_slippage_bps=0,
            avg_impact_bps=0,
            total_execution_cost=0,
            implementation_shortfall=0,
            win_rate=0,
            profit_factor=0,
            avg_trade_pnl=0,
        )

    def save_logs(self, path: str) -> None:
        """Save audit logs to JSON file."""
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert logs to serializable format
        logs = {
            "metadata": {
                "initial_capital": self.config.initial_capital,
                "execution_algo": self.config.execution_algo,
                "n_book_levels": self.config.n_book_levels,
                "timestamp": datetime.now().isoformat(),
            },
            "order_book_snapshots": [
                {
                    "timestamp": str(s.timestamp),
                    "mid_price": s.mid_price,
                    "spread": s.spread,
                    "best_bid": s.best_bid,
                    "best_ask": s.best_ask,
                    "bid_depth": s.bid_depth,
                    "ask_depth": s.ask_depth,
                    "imbalance": s.imbalance,
                }
                for s in self.order_book_logs
            ],
            "fills": [
                {
                    "timestamp": str(f.timestamp),
                    "order_id": f.order_id,
                    "side": f.side,
                    "price": f.price,
                    "quantity": f.quantity,
                    "impact_bps": f.impact,
                    "slippage_bps": f.slippage,
                    "arrival_price": f.arrival_price,
                }
                for f in self.fill_logs
            ],
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2, default=str)

        logger.info(f"Saved audit logs to {log_path}")

    def get_execution_quality_report(self) -> str:
        """Generate execution quality report."""
        lines = [
            "=" * 60,
            "EXECUTION QUALITY REPORT",
            "=" * 60,
            "",
            f"Total Fills: {len(self.fill_logs)}",
            f"Total Volume: {sum(f.quantity for f in self.fill_logs):.0f}",
            "",
        ]

        if self.fill_logs:
            avg_slippage = np.mean([f.slippage for f in self.fill_logs])
            avg_impact = np.mean([f.impact for f in self.fill_logs])

            lines.extend(
                [
                    "Slippage Analysis:",
                    f"  Average: {avg_slippage:.2f} bps",
                    f"  Max: {max(f.slippage for f in self.fill_logs):.2f} bps",
                    f"  Min: {min(f.slippage for f in self.fill_logs):.2f} bps",
                    "",
                    "Market Impact:",
                    f"  Average: {avg_impact:.2f} bps",
                    "",
                    "Implementation Shortfall:",
                    f"  Total: {avg_slippage + avg_impact:.2f} bps",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)
