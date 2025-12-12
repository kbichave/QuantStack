"""
Position sizing with ATR-based risk and alignment scaling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger

from quantcore.config.settings import get_settings


@dataclass
class PositionSize:
    """Result of position size calculation."""

    shares: float
    notional_value: float
    risk_amount: float
    risk_pct: float
    alignment_multiplier: float

    def __str__(self):
        return f"PositionSize(shares={self.shares:.2f}, value=${self.notional_value:.2f}, risk=${self.risk_amount:.2f})"


class PositionSizer(ABC):
    """Abstract base class for position sizing."""

    @abstractmethod
    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        alignment_score: float = 1.0,
    ) -> PositionSize:
        """Calculate position size."""
        pass


class ATRPositionSizer(PositionSizer):
    """
    ATR-based position sizer with alignment scaling.

    Risk per trade = equity * risk_pct
    Position size = risk / (entry - stop_loss)
    Final size = position_size * alignment_multiplier
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 1.0,
        max_position_pct: float = 20.0,
        min_alignment_multiplier: float = 0.5,
        max_alignment_multiplier: float = 1.0,
    ):
        """
        Initialize position sizer.

        Args:
            risk_per_trade_pct: Percentage of equity to risk per trade
            max_position_pct: Maximum position as % of equity
            min_alignment_multiplier: Minimum size multiplier for low alignment
            max_alignment_multiplier: Maximum size multiplier for high alignment
        """
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.min_alignment_multiplier = min_alignment_multiplier
        self.max_alignment_multiplier = max_alignment_multiplier

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        alignment_score: float = 1.0,
    ) -> PositionSize:
        """
        Calculate position size.

        Args:
            equity: Current account equity
            entry_price: Entry price
            stop_loss: Stop loss price
            alignment_score: Cross-TF alignment score (0-1)

        Returns:
            PositionSize with all details
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            logger.warning("Invalid risk per share")
            return PositionSize(
                shares=0,
                notional_value=0,
                risk_amount=0,
                risk_pct=0,
                alignment_multiplier=1.0,
            )

        # Calculate base position size
        risk_amount = equity * (self.risk_per_trade_pct / 100)
        base_shares = risk_amount / risk_per_share

        # Calculate alignment multiplier
        alignment_multiplier = self._calculate_alignment_multiplier(alignment_score)

        # Apply alignment scaling
        adjusted_shares = base_shares * alignment_multiplier

        # Cap by max position
        max_shares = (equity * self.max_position_pct / 100) / entry_price
        final_shares = min(adjusted_shares, max_shares)

        # Calculate final values
        notional_value = final_shares * entry_price
        actual_risk = final_shares * risk_per_share
        actual_risk_pct = actual_risk / equity * 100

        return PositionSize(
            shares=final_shares,
            notional_value=notional_value,
            risk_amount=actual_risk,
            risk_pct=actual_risk_pct,
            alignment_multiplier=alignment_multiplier,
        )

    def _calculate_alignment_multiplier(self, alignment_score: float) -> float:
        """
        Calculate position size multiplier based on alignment.

        Higher alignment = larger position.
        """
        # Linear interpolation between min and max multiplier
        alignment_score = np.clip(alignment_score, 0, 1)

        multiplier = self.min_alignment_multiplier + alignment_score * (
            self.max_alignment_multiplier - self.min_alignment_multiplier
        )

        return multiplier

    def calculate_from_atr(
        self,
        equity: float,
        entry_price: float,
        atr: float,
        sl_atr_multiple: float = 1.0,
        alignment_score: float = 1.0,
    ) -> PositionSize:
        """
        Calculate position size using ATR for stop loss.

        Args:
            equity: Account equity
            entry_price: Entry price
            atr: Average True Range
            sl_atr_multiple: ATR multiple for stop loss
            alignment_score: Alignment score

        Returns:
            PositionSize
        """
        stop_loss = entry_price - (atr * sl_atr_multiple)
        return self.calculate(equity, entry_price, stop_loss, alignment_score)


class KellyCriterionSizer(PositionSizer):
    """
    Position sizer based on Kelly Criterion.

    Uses historical win rate and avg win/loss to determine optimal size.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.25,  # Use fraction of Kelly
        max_position_pct: float = 20.0,
    ):
        """
        Initialize Kelly sizer.

        Args:
            win_rate: Historical win rate
            avg_win_loss_ratio: Avg win / avg loss ratio
            kelly_fraction: Fraction of Kelly to use (conservative)
            max_position_pct: Maximum position size
        """
        self.win_rate = win_rate
        self.avg_win_loss_ratio = avg_win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        alignment_score: float = 1.0,
    ) -> PositionSize:
        """Calculate position using Kelly Criterion."""
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = 1 - p, b = win/loss ratio
        p = self.win_rate
        q = 1 - p
        b = self.avg_win_loss_ratio

        kelly_pct = (p * b - q) / b

        # Apply fraction and alignment
        adjusted_kelly = kelly_pct * self.kelly_fraction * alignment_score
        adjusted_kelly = np.clip(adjusted_kelly, 0, self.max_position_pct / 100)

        # Calculate position
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return PositionSize(
                shares=0,
                notional_value=0,
                risk_amount=0,
                risk_pct=0,
                alignment_multiplier=1.0,
            )

        risk_amount = equity * adjusted_kelly
        shares = risk_amount / risk_per_share

        # Cap by max position
        max_shares = (equity * self.max_position_pct / 100) / entry_price
        final_shares = min(shares, max_shares)

        return PositionSize(
            shares=final_shares,
            notional_value=final_shares * entry_price,
            risk_amount=final_shares * risk_per_share,
            risk_pct=adjusted_kelly * 100,
            alignment_multiplier=alignment_score,
        )
