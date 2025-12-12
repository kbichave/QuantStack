"""
Options risk management.

Provides:
- Simple margin calculator
- Portfolio Greeks manager
- Regime-based risk limits
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from loguru import logger

from quantcore.options.models import OptionsPosition, OptionType


class RiskState(Enum):
    """Risk state classification."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    BREACH = "BREACH"
    CRITICAL = "CRITICAL"


@dataclass
class GreeksLimits:
    """Greek exposure limits."""

    max_delta: float = 500
    max_gamma: float = 100
    max_theta: float = -500  # Max daily decay (negative = earning theta)
    max_vega: float = 1000

    # Per-symbol limits
    max_delta_per_symbol: float = 100
    max_gamma_per_symbol: float = 50


@dataclass
class RiskLimits:
    """Overall risk limits."""

    max_position_value: float = 50000
    max_total_exposure: float = 200000
    max_single_loss: float = 5000
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10

    # Regime-adjusted limits
    bull_delta_multiplier: float = 1.2
    bear_delta_multiplier: float = 0.8
    high_vol_position_multiplier: float = 0.7


@dataclass
class RiskMetrics:
    """Current risk metrics."""

    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0

    delta_by_symbol: Dict[str, float] = None
    gamma_by_symbol: Dict[str, float] = None

    total_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0

    risk_state: RiskState = RiskState.NORMAL

    def __post_init__(self):
        if self.delta_by_symbol is None:
            self.delta_by_symbol = {}
        if self.gamma_by_symbol is None:
            self.gamma_by_symbol = {}


class SimpleMarginCalculator:
    """
    Simple margin calculator for options positions.

    Uses broker-like approximations:
    - Long options: Premium paid
    - Short options: Max loss estimate
    - Spreads: Width minus premium
    """

    def __init__(
        self,
        margin_multiplier: float = 1.5,
        min_margin_per_contract: float = 100,
    ):
        """
        Initialize margin calculator.

        Args:
            margin_multiplier: Multiplier for margin requirements
            min_margin_per_contract: Minimum margin per contract
        """
        self.margin_multiplier = margin_multiplier
        self.min_margin_per_contract = min_margin_per_contract

    def calculate_position_margin(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> float:
        """
        Calculate margin requirement for a position.

        Args:
            position: Options position
            underlying_price: Current underlying price

        Returns:
            Required margin
        """
        total_margin = 0.0

        for leg in position.legs:
            leg_margin = self._calculate_leg_margin(
                leg.contract,
                leg.quantity,
                underlying_price,
            )
            total_margin += leg_margin

        # Apply spread reduction for defined-risk positions
        if position.is_defined_risk():
            total_margin *= 0.5  # Spreads require less margin

        return max(total_margin, self.min_margin_per_contract * len(position.legs))

    def _calculate_leg_margin(
        self,
        contract,
        quantity: int,
        underlying_price: float,
    ) -> float:
        """Calculate margin for a single leg."""
        if quantity > 0:
            # Long option: margin = premium paid
            return abs(quantity) * contract.mid * 100
        else:
            # Short option: use worst-case loss estimate
            if contract.option_type == OptionType.PUT:
                # Short put: max loss = strike - premium
                max_loss = (contract.strike - contract.mid) * 100
            else:
                # Short call: use percentage of underlying
                max_loss = underlying_price * 0.15 * 100  # 15% move

            return abs(quantity) * max_loss * self.margin_multiplier


class PortfolioGreeksManager:
    """
    Manages portfolio-level Greeks and risk limits.

    Features:
    - Real-time Greeks aggregation
    - Regime-based limit adjustments
    - Breach detection and handling
    """

    def __init__(
        self,
        greek_limits: Optional[GreeksLimits] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize Greeks manager.

        Args:
            greek_limits: Greek exposure limits
            risk_limits: Overall risk limits
        """
        self.greek_limits = greek_limits or GreeksLimits()
        self.risk_limits = risk_limits or RiskLimits()

        self.current_metrics = RiskMetrics()
        self._peak_equity = 0.0
        self._daily_starting_equity = 0.0

    def update_from_positions(
        self,
        positions: Dict[str, OptionsPosition],
        current_equity: float,
    ) -> RiskMetrics:
        """
        Update risk metrics from current positions.

        Args:
            positions: Dict of position_id -> OptionsPosition
            current_equity: Current portfolio equity

        Returns:
            Updated RiskMetrics
        """
        # Reset aggregates
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        delta_by_symbol = {}
        gamma_by_symbol = {}

        total_exposure = 0.0
        unrealized_pnl = 0.0

        for position in positions.values():
            # Aggregate Greeks
            total_delta += position.net_delta
            total_gamma += position.net_gamma
            total_theta += position.net_theta
            total_vega += position.net_vega

            # By symbol
            symbol = position.underlying
            delta_by_symbol[symbol] = (
                delta_by_symbol.get(symbol, 0) + position.net_delta
            )
            gamma_by_symbol[symbol] = (
                gamma_by_symbol.get(symbol, 0) + position.net_gamma
            )

            # Exposure and PnL
            total_exposure += abs(position.total_premium)
            unrealized_pnl += position.unrealized_pnl()

        # Update drawdown
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        drawdown_pct = (
            (self._peak_equity - current_equity) / self._peak_equity
            if self._peak_equity > 0
            else 0
        )

        # Daily PnL
        daily_pnl = (
            current_equity - self._daily_starting_equity
            if self._daily_starting_equity > 0
            else 0
        )

        # Determine risk state
        risk_state = self._determine_risk_state(
            total_delta=total_delta,
            total_gamma=total_gamma,
            drawdown_pct=drawdown_pct,
            daily_pnl=daily_pnl,
            current_equity=current_equity,
            delta_by_symbol=delta_by_symbol,
            gamma_by_symbol=gamma_by_symbol,
        )

        self.current_metrics = RiskMetrics(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            delta_by_symbol=delta_by_symbol,
            gamma_by_symbol=gamma_by_symbol,
            total_exposure=total_exposure,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            drawdown_pct=drawdown_pct,
            risk_state=risk_state,
        )

        return self.current_metrics

    def _determine_risk_state(
        self,
        total_delta: float,
        total_gamma: float,
        drawdown_pct: float,
        daily_pnl: float,
        current_equity: float,
        delta_by_symbol: Dict[str, float],
        gamma_by_symbol: Dict[str, float],
    ) -> RiskState:
        """Determine overall risk state."""
        breaches = []
        warnings = []

        # Check portfolio-level limits
        if abs(total_delta) > self.greek_limits.max_delta:
            breaches.append(f"Delta: {total_delta:.1f} > {self.greek_limits.max_delta}")
        elif abs(total_delta) > self.greek_limits.max_delta * 0.8:
            warnings.append(f"Delta approaching limit: {total_delta:.1f}")

        if abs(total_gamma) > self.greek_limits.max_gamma:
            breaches.append(f"Gamma: {total_gamma:.1f} > {self.greek_limits.max_gamma}")

        # Check per-symbol limits
        for symbol, delta in delta_by_symbol.items():
            if abs(delta) > self.greek_limits.max_delta_per_symbol:
                breaches.append(f"{symbol} delta: {delta:.1f}")

        # Check drawdown
        if drawdown_pct > self.risk_limits.max_drawdown_pct:
            breaches.append(f"Drawdown: {drawdown_pct:.1%}")
        elif drawdown_pct > self.risk_limits.max_drawdown_pct * 0.7:
            warnings.append(f"Drawdown warning: {drawdown_pct:.1%}")

        # Check daily loss
        daily_loss_pct = (
            -daily_pnl / current_equity if current_equity > 0 and daily_pnl < 0 else 0
        )
        if daily_loss_pct > self.risk_limits.max_daily_loss_pct:
            breaches.append(f"Daily loss: {daily_loss_pct:.1%}")

        if breaches:
            logger.warning(f"Risk breaches: {breaches}")
            if len(breaches) > 2:
                return RiskState.CRITICAL
            return RiskState.BREACH

        if warnings:
            logger.debug(f"Risk warnings: {warnings}")
            return RiskState.WARNING

        return RiskState.NORMAL

    def get_delta_limit(self, regime: str) -> float:
        """
        Get delta limit adjusted for regime.

        Args:
            regime: Current trend regime (BULL/BEAR/SIDEWAYS)

        Returns:
            Adjusted delta limit
        """
        base_limit = self.greek_limits.max_delta

        if regime == "BULL":
            return base_limit * self.risk_limits.bull_delta_multiplier
        elif regime == "BEAR":
            return base_limit * self.risk_limits.bear_delta_multiplier

        return base_limit

    def get_position_size_limit(self, vol_regime: str) -> float:
        """
        Get position size limit adjusted for vol regime.

        Args:
            vol_regime: Current vol regime (LOW/MEDIUM/HIGH)

        Returns:
            Multiplier for position sizing
        """
        if vol_regime == "HIGH":
            return self.risk_limits.high_vol_position_multiplier
        return 1.0

    def check_new_trade(
        self,
        proposed_delta: float,
        proposed_gamma: float,
        symbol: str,
    ) -> Tuple[bool, str]:
        """
        Check if a new trade would breach limits.

        Args:
            proposed_delta: Delta of proposed trade
            proposed_gamma: Gamma of proposed trade
            symbol: Symbol of proposed trade

        Returns:
            Tuple of (allowed, reason)
        """
        new_delta = self.current_metrics.total_delta + proposed_delta
        new_gamma = self.current_metrics.total_gamma + proposed_gamma

        if abs(new_delta) > self.greek_limits.max_delta:
            return False, f"Would breach delta limit: {new_delta:.1f}"

        if abs(new_gamma) > self.greek_limits.max_gamma:
            return False, f"Would breach gamma limit: {new_gamma:.1f}"

        # Check symbol-specific
        current_symbol_delta = self.current_metrics.delta_by_symbol.get(symbol, 0)
        new_symbol_delta = current_symbol_delta + proposed_delta

        if abs(new_symbol_delta) > self.greek_limits.max_delta_per_symbol:
            return False, f"Would breach {symbol} delta limit: {new_symbol_delta:.1f}"

        return True, "OK"

    def get_reduction_orders(self) -> List[Dict]:
        """
        Get orders needed to reduce risk to limits.

        Returns:
            List of reduction orders
        """
        orders = []

        # If in breach, need to reduce
        if self.current_metrics.risk_state in [RiskState.BREACH, RiskState.CRITICAL]:

            # Reduce delta if breached
            if abs(self.current_metrics.total_delta) > self.greek_limits.max_delta:
                excess = (
                    abs(self.current_metrics.total_delta) - self.greek_limits.max_delta
                )
                orders.append(
                    {
                        "type": "REDUCE_DELTA",
                        "target_reduction": excess,
                        "priority": "HIGH",
                    }
                )

            # Reduce per-symbol delta
            for symbol, delta in self.current_metrics.delta_by_symbol.items():
                if abs(delta) > self.greek_limits.max_delta_per_symbol:
                    excess = abs(delta) - self.greek_limits.max_delta_per_symbol
                    orders.append(
                        {
                            "type": "REDUCE_SYMBOL_DELTA",
                            "symbol": symbol,
                            "target_reduction": excess,
                            "priority": "HIGH",
                        }
                    )

        return orders

    def reset_daily(self, current_equity: float) -> None:
        """Reset daily tracking."""
        self._daily_starting_equity = current_equity
