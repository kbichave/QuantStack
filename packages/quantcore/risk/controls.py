"""
Risk controls including exposure limits and drawdown protection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.config.settings import get_settings
from quantcore.hierarchy.regime_classifier import RegimeType


class RiskStatus(Enum):
    """Overall risk status."""

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    RESTRICTED = "RESTRICTED"
    HALTED = "HALTED"


@dataclass
class RiskState:
    """Current risk state."""

    status: RiskStatus
    equity: float
    drawdown_pct: float
    daily_pnl: float
    open_trades: int
    open_exposure_pct: float
    regime: Optional[str] = None
    messages: List[str] = field(default_factory=list)

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return self.status in [RiskStatus.NORMAL, RiskStatus.CAUTION]

    def size_multiplier(self) -> float:
        """Get position size multiplier based on status."""
        multipliers = {
            RiskStatus.NORMAL: 1.0,
            RiskStatus.CAUTION: 0.5,
            RiskStatus.RESTRICTED: 0.0,
            RiskStatus.HALTED: 0.0,
        }
        return multipliers[self.status]


class ExposureManager:
    """
    Manages exposure limits.

    Controls:
    - Maximum concurrent trades
    - Maximum exposure per symbol
    - Maximum total exposure
    - Daily trade limits
    """

    def __init__(
        self,
        max_concurrent_trades: int = 5,
        max_exposure_per_symbol_pct: float = 20.0,
        max_total_exposure_pct: float = 80.0,
        max_daily_trades: int = 20,
    ):
        """
        Initialize exposure manager.

        Args:
            max_concurrent_trades: Maximum open positions
            max_exposure_per_symbol_pct: Max exposure per symbol (% of equity)
            max_total_exposure_pct: Max total exposure (% of equity)
            max_daily_trades: Maximum trades per day
        """
        self.max_concurrent_trades = max_concurrent_trades
        self.max_exposure_per_symbol_pct = max_exposure_per_symbol_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_daily_trades = max_daily_trades

        # State tracking
        self._open_positions: Dict[str, float] = {}  # symbol -> exposure
        self._daily_trades: Dict[str, int] = {}  # date -> count

    def can_open_position(
        self,
        symbol: str,
        exposure_pct: float,
        equity: float,
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Symbol to trade
            exposure_pct: Proposed exposure as % of equity
            equity: Current equity

        Returns:
            Tuple of (allowed, reason)
        """
        # Check concurrent trades
        if len(self._open_positions) >= self.max_concurrent_trades:
            return (
                False,
                f"Max concurrent trades ({self.max_concurrent_trades}) reached",
            )

        # Check symbol exposure
        current_symbol_exposure = self._open_positions.get(symbol, 0)
        new_symbol_exposure = current_symbol_exposure + exposure_pct
        if new_symbol_exposure > self.max_exposure_per_symbol_pct:
            return (
                False,
                f"Max symbol exposure ({self.max_exposure_per_symbol_pct}%) exceeded",
            )

        # Check total exposure
        total_exposure = sum(self._open_positions.values()) + exposure_pct
        if total_exposure > self.max_total_exposure_pct:
            return (
                False,
                f"Max total exposure ({self.max_total_exposure_pct}%) exceeded",
            )

        # Check daily trades
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self._daily_trades.get(today, 0)
        if daily_count >= self.max_daily_trades:
            return False, f"Max daily trades ({self.max_daily_trades}) reached"

        return True, "OK"

    def register_open(self, symbol: str, exposure_pct: float) -> None:
        """Register an opened position."""
        self._open_positions[symbol] = (
            self._open_positions.get(symbol, 0) + exposure_pct
        )

        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_trades[today] = self._daily_trades.get(today, 0) + 1

    def register_close(self, symbol: str, exposure_pct: float) -> None:
        """Register a closed position."""
        if symbol in self._open_positions:
            self._open_positions[symbol] -= exposure_pct
            if self._open_positions[symbol] <= 0:
                del self._open_positions[symbol]

    def get_total_exposure(self) -> float:
        """Get total current exposure."""
        return sum(self._open_positions.values())

    def reset_daily(self) -> None:
        """Reset daily counters."""
        today = datetime.now().strftime("%Y-%m-%d")
        # Keep only today's count
        self._daily_trades = {k: v for k, v in self._daily_trades.items() if k == today}


class DrawdownProtection:
    """
    Drawdown-based risk controls.

    Levels:
    - Soft stop: Reduce position sizes
    - Hard stop: Halt trading
    """

    def __init__(
        self,
        soft_stop_pct: float = 3.0,
        hard_stop_pct: float = 7.0,
        daily_loss_limit_pct: float = 2.0,
        recovery_threshold_pct: float = 1.0,
    ):
        """
        Initialize drawdown protection.

        Args:
            soft_stop_pct: DD % to trigger size reduction
            hard_stop_pct: DD % to halt trading
            daily_loss_limit_pct: Max daily loss %
            recovery_threshold_pct: DD recovery % before resuming
        """
        self.soft_stop_pct = soft_stop_pct
        self.hard_stop_pct = hard_stop_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.recovery_threshold_pct = recovery_threshold_pct

        # State
        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._halted: bool = False
        self._halt_time: Optional[datetime] = None

    def update(
        self,
        equity: float,
        timestamp: Optional[datetime] = None,
    ) -> RiskStatus:
        """
        Update drawdown state and get risk status.

        Args:
            equity: Current equity
            timestamp: Current timestamp

        Returns:
            RiskStatus
        """
        timestamp = timestamp or datetime.now()

        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Check if new day
        if self._day_start_equity == 0 or self._is_new_day(timestamp):
            self._day_start_equity = equity

        # Calculate drawdowns
        dd_from_peak = (self._peak_equity - equity) / self._peak_equity * 100
        daily_dd = (self._day_start_equity - equity) / self._day_start_equity * 100

        # Check recovery from halt
        if self._halted:
            if dd_from_peak <= self.hard_stop_pct - self.recovery_threshold_pct:
                self._halted = False
                logger.info("Drawdown recovered, resuming trading")
            else:
                return RiskStatus.HALTED

        # Check hard stop
        if dd_from_peak >= self.hard_stop_pct:
            self._halted = True
            self._halt_time = timestamp
            logger.warning(f"Hard stop triggered at {dd_from_peak:.2f}% drawdown")
            return RiskStatus.HALTED

        # Check daily loss limit
        if daily_dd >= self.daily_loss_limit_pct:
            logger.warning(f"Daily loss limit triggered at {daily_dd:.2f}%")
            return RiskStatus.RESTRICTED

        # Check soft stop
        if dd_from_peak >= self.soft_stop_pct:
            return RiskStatus.CAUTION

        return RiskStatus.NORMAL

    def get_drawdown(self, equity: float) -> float:
        """Get current drawdown percentage."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - equity) / self._peak_equity * 100

    def get_daily_pnl(self, equity: float) -> float:
        """Get current day P&L percentage."""
        if self._day_start_equity <= 0:
            return 0.0
        return (equity - self._day_start_equity) / self._day_start_equity * 100

    def _is_new_day(self, timestamp: datetime) -> bool:
        """Check if timestamp is a new trading day."""
        # Simple check - in production would use proper calendar
        return timestamp.hour == 9 and timestamp.minute < 35

    def reset(self, equity: float) -> None:
        """Reset drawdown tracking."""
        self._peak_equity = equity
        self._day_start_equity = equity
        self._halted = False
        self._halt_time = None


class RiskController:
    """
    Master risk controller combining all risk components.
    """

    def __init__(
        self,
        exposure_manager: Optional[ExposureManager] = None,
        drawdown_protection: Optional[DrawdownProtection] = None,
    ):
        """
        Initialize risk controller.

        Args:
            exposure_manager: Exposure manager instance
            drawdown_protection: Drawdown protection instance
        """
        settings = get_settings()

        self.exposure = exposure_manager or ExposureManager(
            max_concurrent_trades=settings.max_concurrent_trades,
        )
        self.drawdown = drawdown_protection or DrawdownProtection(
            soft_stop_pct=settings.soft_stop_drawdown_pct,
            hard_stop_pct=settings.hard_stop_drawdown_pct,
        )

        self._current_regime: Optional[RegimeType] = None

    def update_regime(self, regime: RegimeType) -> None:
        """Update current market regime."""
        self._current_regime = regime

    def get_risk_state(
        self,
        equity: float,
        timestamp: Optional[datetime] = None,
    ) -> RiskState:
        """
        Get comprehensive risk state.

        Args:
            equity: Current equity
            timestamp: Current timestamp

        Returns:
            RiskState with all details
        """
        timestamp = timestamp or datetime.now()

        # Get drawdown status
        dd_status = self.drawdown.update(equity, timestamp)
        dd_pct = self.drawdown.get_drawdown(equity)
        daily_pnl = self.drawdown.get_daily_pnl(equity)

        # Get exposure
        total_exposure = self.exposure.get_total_exposure()
        open_trades = len(self.exposure._open_positions)

        # Build messages
        messages = []

        if dd_status == RiskStatus.HALTED:
            messages.append(f"Trading halted: {dd_pct:.1f}% drawdown")
        elif dd_status == RiskStatus.RESTRICTED:
            messages.append(f"Daily loss limit reached: {daily_pnl:.1f}%")
        elif dd_status == RiskStatus.CAUTION:
            messages.append(f"Elevated drawdown: {dd_pct:.1f}%")

        if total_exposure > 60:
            messages.append(f"High exposure: {total_exposure:.1f}%")

        # Regime-based adjustments
        regime_name = self._current_regime.value if self._current_regime else None

        return RiskState(
            status=dd_status,
            equity=equity,
            drawdown_pct=dd_pct,
            daily_pnl=daily_pnl,
            open_trades=open_trades,
            open_exposure_pct=total_exposure,
            regime=regime_name,
            messages=messages,
        )

    def can_trade(
        self,
        symbol: str,
        exposure_pct: float,
        equity: float,
        direction: str = "LONG",
        timestamp: Optional[datetime] = None,
    ) -> tuple[bool, str, float]:
        """
        Check if a trade is allowed.

        Args:
            symbol: Symbol to trade
            exposure_pct: Proposed exposure
            equity: Current equity
            direction: Trade direction
            timestamp: Current timestamp

        Returns:
            Tuple of (allowed, reason, size_multiplier)
        """
        # Get risk state
        state = self.get_risk_state(equity, timestamp)

        if not state.can_trade():
            return False, f"Risk status: {state.status.value}", 0.0

        # Check exposure
        can_open, reason = self.exposure.can_open_position(symbol, exposure_pct, equity)
        if not can_open:
            return False, reason, 0.0

        # Regime-based restrictions
        if self._current_regime:
            if direction == "LONG" and self._current_regime == RegimeType.BEAR:
                return False, "Long trades blocked in BEAR regime", 0.0
            if direction == "SHORT" and self._current_regime == RegimeType.BULL:
                return False, "Short trades blocked in BULL regime", 0.0

        return True, "OK", state.size_multiplier()
