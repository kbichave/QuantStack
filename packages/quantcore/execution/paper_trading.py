"""
Paper trading engine for options.

Provides:
- Live data feed integration
- Real-time signal generation
- Order logging
- Performance tracking
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Thread, Event

import pandas as pd
from loguru import logger

from quantcore.strategy.base import (
    Strategy,
    MarketState,
    TargetPosition,
    PositionDirection,
    RegimeState,
)
from quantcore.data.fetcher import AlphaVantageClient
from quantcore.data.storage import DataStore
from quantcore.data.universe import UniverseManager
from quantcore.risk.options_risk import PortfolioGreeksManager, RiskState


@dataclass
class PaperOrder:
    """Paper trading order."""

    order_id: str
    timestamp: datetime
    symbol: str
    direction: str
    quantity: int
    structure_type: str
    confidence: float
    reason: str

    # Execution
    status: str = "PENDING"
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "structure_type": self.structure_type,
            "confidence": self.confidence,
            "reason": self.reason,
            "status": self.status,
            "fill_price": self.fill_price,
            "fill_timestamp": (
                self.fill_timestamp.isoformat() if self.fill_timestamp else None
            ),
        }


@dataclass
class PaperPosition:
    """Paper trading position."""

    symbol: str
    direction: str
    quantity: int
    entry_price: float
    entry_timestamp: datetime

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    # Greeks (simplified)
    delta: float = 0.0
    theta: float = 0.0


class PaperTradingEngine:
    """
    Paper trading engine for live testing.

    Features:
    - Live data feed from AlphaVantage
    - Real-time signal generation
    - Order and position tracking
    - Signal logging for review
    """

    def __init__(
        self,
        strategy: Strategy,
        universe: UniverseManager,
        fetcher: AlphaVantageClient,
        data_store: DataStore,
        initial_equity: float = 100000,
        max_trade_value: float = 1000,  # $1K max per trade
        signals_path: str = "paper_trading/signals.json",
    ):
        """
        Initialize paper trading engine.

        Args:
            strategy: Strategy to run
            universe: Universe manager
            fetcher: AlphaVantage client
            data_store: Data store for historical data
            initial_equity: Starting equity
            max_trade_value: Maximum value per trade (default $1K)
            signals_path: Path for signal logging
        """
        self.strategy = strategy
        self.universe = universe
        self.fetcher = fetcher
        self.data_store = data_store
        self.initial_equity = initial_equity
        self.max_trade_value = max_trade_value
        self.signals_path = Path(signals_path)

        # State
        self.equity = initial_equity
        self.cash = initial_equity
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: List[PaperOrder] = []
        self.signals: List[Dict] = []

        # Risk management
        self.risk_manager = PortfolioGreeksManager()

        # Control
        self._stop_event = Event()
        self._running = False

    def start(
        self,
        symbols: Optional[List[str]] = None,
        interval_seconds: int = 3600,  # 1 hour default
    ) -> None:
        """
        Start paper trading.

        Args:
            symbols: Symbols to trade (default: from universe)
            interval_seconds: Seconds between updates
        """
        if symbols is None:
            symbols = self.universe.symbols[:10]  # Limit to 10 for API calls

        logger.info(
            f"Starting paper trading: {len(symbols)} symbols, {interval_seconds}s interval"
        )

        self._running = True
        self._stop_event.clear()

        while not self._stop_event.is_set():
            try:
                self._trading_loop(symbols)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

            # Wait for next interval
            self._stop_event.wait(interval_seconds)

        self._running = False
        logger.info("Paper trading stopped")

    def stop(self) -> None:
        """Stop paper trading."""
        self._stop_event.set()

    def _trading_loop(self, symbols: List[str]) -> None:
        """Single iteration of trading loop."""
        timestamp = datetime.now()
        logger.debug(f"Trading loop: {timestamp}")

        for symbol in symbols:
            try:
                # Get current data
                quote = self._get_current_quote(symbol)
                if quote is None:
                    continue

                # Get historical features
                features = self._get_features(symbol)
                if features is None:
                    continue

                # Build market state
                market_state = self._build_market_state(symbol, quote, features)

                # Get strategy signals
                signals = self.strategy.on_bar(market_state)

                # Process signals
                for signal in signals:
                    self._process_signal(signal, quote)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Update positions
        self._update_positions()

        # Check risk limits
        self._check_risk()

        # Save signals
        self._save_signals()

    def _get_current_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for symbol."""
        try:
            data = self.fetcher.fetch_quote(symbol)
            if data.empty:
                return None

            row = data.iloc[0]
            return {
                "symbol": symbol,
                "price": row.get("price", row.get("close", 0)),
                "open": row.get("open", 0),
                "high": row.get("high", 0),
                "low": row.get("low", 0),
                "volume": row.get("volume", 0),
            }
        except Exception as e:
            logger.debug(f"Error fetching quote for {symbol}: {e}")
            return None

    def _get_features(self, symbol: str) -> Optional[Dict]:
        """Get features for symbol."""
        # Load recent data
        try:
            df = self.data_store.load_ohlcv(
                symbol=symbol,
                timeframe="1D",
                start=datetime.now() - timedelta(days=30),
            )

            if df.empty:
                return None

            # Compute basic features
            features = {
                "zscore_price": (df["close"].iloc[-1] - df["close"].mean())
                / df["close"].std(),
                "ema_alignment": (
                    1
                    if df["close"].iloc[-1] > df["close"].rolling(20).mean().iloc[-1]
                    else -1
                ),
                "rsi": self._calculate_rsi(df["close"]),
                "momentum_score": df["close"].pct_change(5).iloc[-1] * 100,
            }

            return features

        except Exception as e:
            logger.debug(f"Error getting features for {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        if loss.iloc[-1] == 0:
            return 100

        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def _build_market_state(
        self,
        symbol: str,
        quote: Dict,
        features: Dict,
    ) -> MarketState:
        """Build MarketState from current data."""
        return MarketState(
            timestamp=datetime.now(),
            bar_index=0,
            symbol=symbol,
            open=quote.get("open", quote["price"]),
            high=quote.get("high", quote["price"]),
            low=quote.get("low", quote["price"]),
            close=quote["price"],
            volume=quote.get("volume", 0),
            features=features,
            regime=None,  # Would come from regime model
            portfolio_equity=self.equity,
        )

    def _process_signal(self, signal: TargetPosition, quote: Dict) -> None:
        """Process a trading signal."""
        # Log signal
        signal_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "price": quote["price"],
        }
        self.signals.append(signal_record)

        logger.info(
            f"Signal: {signal.symbol} {signal.direction.value} "
            f"confidence={signal.confidence:.2f} reason={signal.reason}"
        )

        # Check risk before creating order
        if signal.direction != PositionDirection.FLAT:
            delta = 50 if signal.direction == PositionDirection.LONG else -50
            allowed, reason = self.risk_manager.check_new_trade(
                proposed_delta=delta,
                proposed_gamma=5,
                symbol=signal.symbol,
            )

            if not allowed:
                logger.warning(f"Trade blocked: {reason}")
                return

        # Calculate quantity from dollar value ($1K max per trade)
        position_value = self.max_trade_value * signal.confidence
        # Each option contract controls 100 shares
        # Approximate option price as ~5% of underlying
        option_price_estimate = quote["price"] * 0.05
        quantity = max(1, int(position_value / (option_price_estimate * 100)))
        quantity = min(quantity, 10)  # Cap at 10 contracts

        # Create paper order
        order = PaperOrder(
            order_id=f"PO{len(self.orders):06d}",
            timestamp=datetime.now(),
            symbol=signal.symbol,
            direction=signal.direction.value,
            quantity=quantity,
            structure_type="DIRECTIONAL",
            confidence=signal.confidence,
            reason=signal.reason,
        )

        # Simulate fill
        order.status = "FILLED"
        order.fill_price = quote["price"]
        order.fill_timestamp = datetime.now()

        self.orders.append(order)

        # Update position
        if signal.direction != PositionDirection.FLAT:
            self._update_position_from_order(order)

    def _update_position_from_order(self, order: PaperOrder) -> None:
        """Update position from filled order."""
        symbol = order.symbol

        if symbol in self.positions:
            # Modify existing position
            pos = self.positions[symbol]
            if order.direction == pos.direction:
                # Add to position
                pos.quantity += order.quantity
            else:
                # Reduce or close
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                direction=order.direction,
                quantity=order.quantity,
                entry_price=order.fill_price,
                entry_timestamp=order.fill_timestamp,
                current_price=order.fill_price,
            )

    def _update_positions(self) -> None:
        """Update all position values."""
        for symbol, pos in self.positions.items():
            try:
                quote = self._get_current_quote(symbol)
                if quote:
                    pos.current_price = quote["price"]

                    # Calculate unrealized PnL
                    if pos.direction == "LONG":
                        pos.unrealized_pnl = (
                            (pos.current_price - pos.entry_price) * pos.quantity * 100
                        )
                    else:
                        pos.unrealized_pnl = (
                            (pos.entry_price - pos.current_price) * pos.quantity * 100
                        )

            except Exception as e:
                logger.debug(f"Error updating {symbol} position: {e}")

        # Update total equity
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.equity = self.cash + total_pnl

    def _check_risk(self) -> None:
        """Check risk limits."""
        # Build positions dict for risk manager
        from quantcore.options.models import OptionsPosition

        # Update risk metrics
        # (Simplified - would use actual Greeks in production)

        if self.risk_manager.current_metrics.risk_state in [
            RiskState.BREACH,
            RiskState.CRITICAL,
        ]:
            logger.warning(
                f"Risk state: {self.risk_manager.current_metrics.risk_state}"
            )

    def _save_signals(self) -> None:
        """Save signals to file."""
        self.signals_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.signals_path, "w") as f:
            json.dump(
                {
                    "last_update": datetime.now().isoformat(),
                    "equity": self.equity,
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "direction": p.direction,
                            "quantity": p.quantity,
                            "entry_price": p.entry_price,
                            "current_price": p.current_price,
                            "unrealized_pnl": p.unrealized_pnl,
                        }
                        for p in self.positions.values()
                    ],
                    "recent_signals": self.signals[-50:],  # Last 50 signals
                    "recent_orders": [o.to_dict() for o in self.orders[-50:]],
                },
                f,
                indent=2,
                default=str,
            )

    def get_status(self) -> Dict:
        """Get current trading status."""
        return {
            "running": self._running,
            "equity": self.equity,
            "cash": self.cash,
            "num_positions": len(self.positions),
            "num_orders": len(self.orders),
            "num_signals": len(self.signals),
            "risk_state": self.risk_manager.current_metrics.risk_state.value,
        }


def run_paper_trading(
    strategy: Strategy,
    symbols: List[str],
    duration_hours: float = 1.0,
    interval_seconds: int = 60,
) -> Dict:
    """
    Convenience function to run paper trading.

    Args:
        strategy: Strategy to run
        symbols: Symbols to trade
        duration_hours: How long to run
        interval_seconds: Update interval

    Returns:
        Final status
    """
    from quantcore.data.fetcher import AlphaVantageClient
    from quantcore.data.storage import DataStore
    from quantcore.data.universe import UniverseManager

    fetcher = AlphaVantageClient()
    data_store = DataStore()
    universe = UniverseManager()

    engine = PaperTradingEngine(
        strategy=strategy,
        universe=universe,
        fetcher=fetcher,
        data_store=data_store,
    )

    # Run in background
    from threading import Thread

    def run():
        engine.start(symbols=symbols, interval_seconds=interval_seconds)

    thread = Thread(target=run, daemon=True)
    thread.start()

    # Wait for duration
    time.sleep(duration_hours * 3600)

    # Stop
    engine.stop()
    thread.join(timeout=10)

    return engine.get_status()
