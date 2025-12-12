"""
Options backtesting engine.

Key features:
- Event-driven hourly clock
- Options expiration handling
- T+1 execution (signal at T close, trade at T+1 open)
- Slippage modeling
- Greeks tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.strategy.base import (
    Strategy,
    MarketState,
    TargetPosition,
    PositionDirection,
    RegimeState,
)
from quantcore.options.models import (
    OptionContract,
    OptionLeg,
    OptionsPosition,
    OptionType,
)
from quantcore.options.contract_selector import (
    ContractSelector,
    Direction,
    VolRegime,
    TrendRegime,
)
from quantcore.options.pricing import (
    black_scholes_price,
    black_scholes_greeks,
    estimate_slippage,
)
from quantcore.risk.position_sizing import ATRPositionSizer, PositionSize


class FillType(Enum):
    """Order fill type."""

    OPEN = "OPEN"  # Fill at bar open
    CLOSE = "CLOSE"  # Fill at bar close
    VWAP = "VWAP"  # Approximated VWAP


@dataclass
class Trade:
    """Executed trade record."""

    trade_id: str
    symbol: str
    timestamp: datetime
    direction: PositionDirection
    structure_type: str
    entry_price: float
    quantity: int
    commission: float
    slippage: float

    # Position tracking
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0

    # Greeks at entry
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class BacktestState:
    """Current backtest state."""

    equity: float
    cash: float
    positions: Dict[str, OptionsPosition] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)

    # Greeks
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0

    # Tracking
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    peak_equity: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    initial_equity: float = 100000
    commission_per_contract: float = 0.65
    slippage_model: str = "parametric"  # "fixed", "parametric"
    fixed_slippage_pct: float = 0.02
    fill_type: FillType = FillType.OPEN

    # Risk limits - dollar-based sizing
    max_trade_value: float = 1000  # $1K max per trade
    max_delta_per_symbol: float = 100
    max_total_delta: float = 500
    max_daily_loss: float = 0.02  # 2% of equity

    # ATR-based position sizing
    use_atr_sizing: bool = True
    risk_per_trade_pct: float = 1.0  # 1% of equity per trade
    atr_stop_multiplier: float = 2.0  # Stop at 2x ATR from entry

    # Options-specific
    assume_european: bool = True
    auto_close_dte: int = 1  # Close at N days to expiry
    apply_theta_decay: bool = True  # Apply daily theta to positions

    # Execution timing
    trade_at_open: bool = True  # T+1 open execution


class OptionsBacktester:
    """
    Event-driven options backtesting engine.

    Key features:
    - Signal at bar T close, trade at bar T+1 open
    - Options expiration handling
    - Greeks tracking and limits
    - Slippage modeling
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.state: Optional[BacktestState] = None
        self.contract_selector = ContractSelector()

        # ATR-based position sizer
        self.position_sizer = ATRPositionSizer(
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            max_position_pct=20.0,  # Max 20% of equity per position
        )

        # Pending signals (for T+1 execution)
        self._pending_signals: List[TargetPosition] = []

        # Risk tracking
        self._daily_pnl = 0.0
        self._last_date: Optional[date] = None
        self._cumulative_theta: float = 0.0  # Track total theta paid/received

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        features: pd.DataFrame,
        options_data: Optional[Dict[str, pd.DataFrame]] = None,
        regime_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest.

        Args:
            strategy: Strategy to backtest
            data: OHLCV DataFrame for underlying
            features: Feature DataFrame
            options_data: Dict of date -> options chain DataFrame
            regime_data: Regime indicators DataFrame

        Returns:
            Backtest results dictionary
        """
        # Initialize state
        self.state = BacktestState(
            equity=self.config.initial_equity,
            cash=self.config.initial_equity,
            peak_equity=self.config.initial_equity,
        )
        self._pending_signals = []
        self._daily_pnl = 0.0
        self._last_date = None

        # Initialize strategy
        strategy.initialize()

        # Main backtest loop
        for i in range(len(data)):
            timestamp = data.index[i]
            current_date = timestamp.date() if hasattr(timestamp, "date") else timestamp

            # Reset daily PnL on new day
            if self._last_date is not None and current_date != self._last_date:
                self._daily_pnl = 0.0
            self._last_date = current_date

            # Get current bar
            bar = data.iloc[i]

            # 1. Execute pending signals (from previous bar)
            if self._pending_signals and self.config.trade_at_open:
                # Get ATR from features for position sizing
                current_atr = None
                if i < len(features) and "atr" in features.columns:
                    current_atr = features.iloc[i].get("atr")

                self._execute_pending_signals(
                    bar["open"],
                    timestamp,
                    options_data.get(str(current_date)) if options_data else None,
                    atr=current_atr,
                )

            # 2. Check expiration and close expiring positions
            self._handle_expirations(current_date, bar["close"])

            # 3. Update position values and Greeks
            self._update_positions(bar["close"], timestamp)

            # 4. Build market state
            market_state = self._build_market_state(
                timestamp=timestamp,
                bar=bar,
                bar_index=i,
                features=features.iloc[i] if i < len(features) else None,
                regime_data=(
                    regime_data.iloc[i]
                    if regime_data is not None and i < len(regime_data)
                    else None
                ),
            )

            # 5. Get strategy signals (these execute on NEXT bar)
            signals = strategy.on_bar(market_state)
            self._pending_signals = signals

            # 6. Check risk limits
            self._check_risk_limits()

            # 7. Record equity
            self._record_equity(timestamp)

        # Close all remaining positions at end
        self._close_all_positions(data.iloc[-1]["close"], data.index[-1])

        # Calculate metrics
        return self._calculate_results(data)

    def _build_market_state(
        self,
        timestamp: datetime,
        bar: pd.Series,
        bar_index: int,
        features: Optional[pd.Series],
        regime_data: Optional[pd.Series],
    ) -> MarketState:
        """Build MarketState from current data."""
        # Get regime
        regime = None
        if regime_data is not None:
            trend_regime = regime_data.get("trend_regime", "SIDEWAYS")
            vol_regime = regime_data.get("vol_regime", "MEDIUM")
            regime = RegimeState(
                trend_regime=str(trend_regime),
                vol_regime=str(vol_regime),
            )

        # Get features dict
        features_dict = {}
        if features is not None:
            features_dict = features.to_dict()

        # Get options metrics
        iv_rank = features_dict.get("iv_rank", 50)
        days_to_earnings = features_dict.get("days_to_earnings")

        return MarketState(
            timestamp=timestamp,
            bar_index=bar_index,
            symbol=bar.name if hasattr(bar, "name") else "UNKNOWN",
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar.get("volume", 0),
            features=features_dict,
            regime=regime,
            iv_rank=iv_rank,
            days_to_earnings=int(days_to_earnings) if days_to_earnings else None,
            current_delta=self.state.total_delta,
            current_gamma=self.state.total_gamma,
            current_theta=self.state.total_theta,
            current_vega=self.state.total_vega,
            unrealized_pnl=sum(
                pos.unrealized_pnl() for pos in self.state.positions.values()
            ),
            portfolio_equity=self.state.equity,
            drawdown_pct=self.state.max_drawdown,
        )

    def _execute_pending_signals(
        self,
        execution_price: float,
        timestamp: datetime,
        options_chain: Optional[pd.DataFrame],
        atr: Optional[float] = None,
    ) -> None:
        """Execute pending signals at current bar open."""
        for signal in self._pending_signals:
            # Check daily loss limit
            if self._daily_pnl < -self.config.max_daily_loss * self.state.equity:
                logger.warning("Daily loss limit reached, skipping signal")
                continue

            # Check delta limits
            if abs(self.state.total_delta) > self.config.max_total_delta:
                if (
                    signal.direction == PositionDirection.LONG
                    and self.state.total_delta > 0
                ) or (
                    signal.direction == PositionDirection.SHORT
                    and self.state.total_delta < 0
                ):
                    logger.warning("Delta limit reached, skipping signal")
                    continue

            # Execute trade with ATR for sizing
            self._execute_trade(signal, execution_price, timestamp, options_chain, atr)

        self._pending_signals = []

    def _execute_trade(
        self,
        signal: TargetPosition,
        underlying_price: float,
        timestamp: datetime,
        options_chain: Optional[pd.DataFrame],
        atr: Optional[float] = None,
    ) -> None:
        """Execute a single trade."""
        # Determine contract selection
        vol_regime = VolRegime.MEDIUM
        if hasattr(signal, "iv_rank") and signal.iv_rank is not None:
            if signal.iv_rank < 30:
                vol_regime = VolRegime.LOW
            elif signal.iv_rank > 70:
                vol_regime = VolRegime.HIGH

        # Calculate position size
        if self.config.use_atr_sizing and atr is not None and atr > 0:
            # ATR-based sizing: risk per trade / ATR = position size
            stop_distance = atr * self.config.atr_stop_multiplier
            stop_loss = (
                underlying_price - stop_distance
                if signal.direction == PositionDirection.LONG
                else underlying_price + stop_distance
            )

            position_sizing = self.position_sizer.calculate(
                equity=self.state.equity,
                entry_price=underlying_price,
                stop_loss=stop_loss,
                alignment_score=signal.confidence,  # Use confidence as alignment
            )

            # For options, we need to translate shares to contracts
            # Each contract controls 100 shares
            # Approximate option price as delta * underlying + time value
            option_price_estimate = underlying_price * 0.05  # Rough ATM estimate
            contracts_from_atr = max(
                1, int(position_sizing.notional_value / (option_price_estimate * 100))
            )

            # Also check against max trade value
            value_based_contracts = max(
                1,
                int(
                    self.config.max_trade_value
                    * signal.confidence
                    / (option_price_estimate * 100)
                ),
            )

            # Use the more conservative of the two
            position_size = min(contracts_from_atr, value_based_contracts, 10)
        else:
            # Fallback: dollar-based sizing
            position_value = self.config.max_trade_value * signal.confidence
            option_price_estimate = underlying_price * 0.05
            position_size = max(1, int(position_value / (option_price_estimate * 100)))
            position_size = min(position_size, 10)

        # Estimate entry price and slippage
        if self.config.slippage_model == "fixed":
            entry_price = underlying_price * (1 + self.config.fixed_slippage_pct / 100)
        else:
            entry_price = underlying_price  # Will be adjusted per contract

        # Calculate commission
        commission = self.config.commission_per_contract * position_size

        # Create trade record
        trade_id = f"T{len(self.state.trades):06d}"
        trade = Trade(
            trade_id=trade_id,
            symbol=signal.symbol,
            timestamp=timestamp,
            direction=signal.direction,
            structure_type="DIRECTIONAL",  # Would come from contract selector
            entry_price=entry_price,
            quantity=position_size,
            commission=commission,
            slippage=abs(entry_price - underlying_price) * position_size,
            delta=50
            * (1 if signal.direction == PositionDirection.LONG else -1)
            * position_size,
        )

        # Update state
        self.state.trades.append(trade)
        self.state.cash -= commission
        self.state.total_delta += trade.delta

        logger.debug(
            f"Executed trade: {trade_id} {signal.direction.value} {signal.symbol}"
        )

    def _handle_expirations(self, current_date: date, underlying_price: float) -> None:
        """Handle options expiration."""
        positions_to_close = []

        for pos_id, position in self.state.positions.items():
            earliest_expiry = position.earliest_expiry
            if earliest_expiry is None:
                continue

            days_to_expiry = (earliest_expiry - current_date).days

            # Close if at or past expiry, or if within auto-close window
            if days_to_expiry <= self.config.auto_close_dte:
                positions_to_close.append(pos_id)

        for pos_id in positions_to_close:
            self._close_position(pos_id, underlying_price, "EXPIRATION")

    def _update_positions(self, underlying_price: float, timestamp: datetime) -> None:
        """Update all position values and Greeks, including theta decay."""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        daily_theta_impact = 0.0

        for position in self.state.positions.values():
            # Update each leg's Greeks
            # (Simplified - in production would use actual options chain data)
            total_delta += position.net_delta
            total_gamma += position.net_gamma
            total_theta += position.net_theta
            total_vega += position.net_vega

            # Apply theta decay if enabled
            # Theta is typically negative for long options (value decay)
            # and positive for short options (value gain from decay)
            if self.config.apply_theta_decay:
                # Get daily theta impact (theta is in $/day per contract)
                position_theta = position.net_theta
                daily_theta_impact += position_theta

        # Apply theta decay to cash (theta is already in daily terms)
        if self.config.apply_theta_decay and daily_theta_impact != 0:
            self.state.cash += (
                daily_theta_impact  # Theta is neg for longs, adds to shorts
            )
            self._cumulative_theta += daily_theta_impact

            # Log significant theta impact
            if abs(daily_theta_impact) > 10:
                logger.debug(f"Theta decay: ${daily_theta_impact:.2f}")

        self.state.total_delta = total_delta
        self.state.total_gamma = total_gamma
        self.state.total_theta = total_theta
        self.state.total_vega = total_vega

    def _close_position(
        self,
        position_id: str,
        underlying_price: float,
        reason: str,
    ) -> None:
        """Close a position."""
        if position_id not in self.state.positions:
            return

        position = self.state.positions[position_id]

        # Calculate exit value (simplified)
        pnl = position.unrealized_pnl()
        commission = self.config.commission_per_contract * len(position.legs)

        # Update state
        self.state.cash += pnl - commission
        self._daily_pnl += pnl

        # Remove position
        del self.state.positions[position_id]

        logger.debug(f"Closed position {position_id}: PnL={pnl:.2f}, reason={reason}")

    def _close_all_positions(
        self,
        underlying_price: float,
        timestamp: datetime,
    ) -> None:
        """Close all open positions."""
        position_ids = list(self.state.positions.keys())
        for pos_id in position_ids:
            self._close_position(pos_id, underlying_price, "END_OF_BACKTEST")

    def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        # Check daily loss
        if self._daily_pnl < -self.config.max_daily_loss * self.state.equity:
            logger.warning("Daily loss limit breached")
            # Could force-close positions here

        # Check delta limits
        if abs(self.state.total_delta) > self.config.max_total_delta:
            logger.warning(f"Delta limit breached: {self.state.total_delta:.1f}")

    def _record_equity(self, timestamp: datetime) -> None:
        """Record equity for curve tracking."""
        # Calculate total equity
        unrealized = sum(pos.unrealized_pnl() for pos in self.state.positions.values())
        self.state.equity = self.state.cash + unrealized

        # Track peak and drawdown
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown

        self.state.equity_curve.append(self.state.equity)
        self.state.drawdown_curve.append(drawdown)

    def _calculate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate backtest results and metrics."""
        equity_curve = np.array(self.state.equity_curve)

        if len(equity_curve) < 2:
            return {"error": "Insufficient data"}

        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Basic metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annual_return = total_return * (252 / len(equity_curve))

        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Trade metrics
        trades = self.state.trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": self.state.max_drawdown,
            "num_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "final_equity": equity_curve[-1],
            "equity_curve": equity_curve.tolist(),
            "drawdown_curve": self.state.drawdown_curve,
            # Theta tracking
            "cumulative_theta": self._cumulative_theta,
            "theta_as_pct_of_pnl": (
                self._cumulative_theta / (equity_curve[-1] - equity_curve[0]) * 100
                if (equity_curve[-1] - equity_curve[0]) != 0
                else 0
            ),
        }


def run_options_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    features: pd.DataFrame,
    initial_equity: float = 100000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run options backtest.

    Args:
        strategy: Strategy to backtest
        data: OHLCV DataFrame
        features: Feature DataFrame
        initial_equity: Starting equity
        **kwargs: Additional config options

    Returns:
        Backtest results
    """
    config = BacktestConfig(initial_equity=initial_equity, **kwargs)
    backtester = OptionsBacktester(config)
    return backtester.run(strategy, data, features)
