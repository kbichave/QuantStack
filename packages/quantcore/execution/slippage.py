"""
Slippage modeling for realistic execution simulation.

Models slippage as function of:
- Volume (market impact)
- Volatility (uncertainty)
- Spread (immediate cost)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class SlippageEstimate:
    """Estimated slippage for a trade."""

    spread_cost_bps: float
    market_impact_bps: float
    volatility_cost_bps: float
    total_slippage_bps: float
    confidence: float  # 0-1, higher = more confident


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def estimate(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float,
        spread_bps: float,
    ) -> SlippageEstimate:
        """
        Estimate slippage for a trade.

        Args:
            trade_size: Number of shares/units
            price: Current price
            volume: Recent volume
            volatility: Current volatility
            spread_bps: Bid-ask spread in bps

        Returns:
            SlippageEstimate
        """
        pass


class VolumeSlippageModel(SlippageModel):
    """
    Volume-based slippage model.

    Market impact is proportional to participation rate.
    Based on Almgren-Chriss framework.
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.5,
        participation_cap: float = 0.1,
    ):
        """
        Initialize volume slippage model.

        Args:
            permanent_impact_coef: Permanent impact coefficient
            temporary_impact_coef: Temporary impact coefficient
            participation_cap: Max participation rate before linear scaling
        """
        self.permanent_impact_coef = permanent_impact_coef
        self.temporary_impact_coef = temporary_impact_coef
        self.participation_cap = participation_cap

    def estimate(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float,
        spread_bps: float,
    ) -> SlippageEstimate:
        """Estimate volume-based slippage."""
        if volume <= 0 or trade_size <= 0:
            return SlippageEstimate(
                spread_cost_bps=spread_bps,
                market_impact_bps=0,
                volatility_cost_bps=0,
                total_slippage_bps=spread_bps,
                confidence=0.5,
            )

        # Participation rate
        participation = trade_size / volume

        # Market impact (square root law)
        # Impact ∝ σ × √(trade_size / volume)
        vol_pct = volatility / 100 if volatility > 1 else volatility

        # Temporary impact (recovers)
        temp_impact = (
            self.temporary_impact_coef * vol_pct * np.sqrt(participation) * 10000
        )

        # Permanent impact (doesn't recover)
        perm_impact = self.permanent_impact_coef * vol_pct * participation * 10000

        market_impact = temp_impact + perm_impact

        # Additional cost if exceeding participation cap
        if participation > self.participation_cap:
            excess = participation / self.participation_cap
            market_impact *= 1 + np.log(excess)

        # Volatility cost (execution uncertainty)
        vol_cost = vol_pct * 0.1 * 10000  # Small fraction of vol

        total = spread_bps + market_impact + vol_cost

        # Confidence decreases with participation
        confidence = max(0.3, 1 - participation * 2)

        return SlippageEstimate(
            spread_cost_bps=spread_bps,
            market_impact_bps=market_impact,
            volatility_cost_bps=vol_cost,
            total_slippage_bps=total,
            confidence=confidence,
        )


class VolatilitySlippageModel(SlippageModel):
    """
    Volatility-based slippage model.

    Simpler model where slippage scales with volatility.
    """

    def __init__(
        self,
        vol_multiplier: float = 0.5,
        base_slippage_bps: float = 2.0,
    ):
        """
        Initialize volatility slippage model.

        Args:
            vol_multiplier: Multiplier for vol-based slippage
            base_slippage_bps: Base slippage regardless of vol
        """
        self.vol_multiplier = vol_multiplier
        self.base_slippage_bps = base_slippage_bps

    def estimate(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float,
        spread_bps: float,
    ) -> SlippageEstimate:
        """Estimate volatility-based slippage."""
        vol_pct = volatility / 100 if volatility > 1 else volatility

        # Vol-based component
        vol_slippage = vol_pct * self.vol_multiplier * 10000

        # Size adjustment
        if volume > 0:
            size_factor = 1 + max(0, trade_size / volume - 0.01) * 10
        else:
            size_factor = 2.0

        market_impact = vol_slippage * size_factor

        total = spread_bps + self.base_slippage_bps + market_impact

        return SlippageEstimate(
            spread_cost_bps=spread_bps,
            market_impact_bps=market_impact,
            volatility_cost_bps=self.base_slippage_bps,
            total_slippage_bps=total,
            confidence=0.6,
        )


class CompositeSlippageModel(SlippageModel):
    """
    Composite model combining multiple slippage estimates.
    """

    def __init__(self):
        """Initialize composite model."""
        self.volume_model = VolumeSlippageModel()
        self.volatility_model = VolatilitySlippageModel()

    def estimate(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float,
        spread_bps: float,
    ) -> SlippageEstimate:
        """Estimate composite slippage (max of models)."""
        vol_est = self.volume_model.estimate(
            trade_size, price, volume, volatility, spread_bps
        )
        volat_est = self.volatility_model.estimate(
            trade_size, price, volume, volatility, spread_bps
        )

        # Use max of estimates (conservative)
        if vol_est.total_slippage_bps > volat_est.total_slippage_bps:
            return vol_est
        return volat_est

    def estimate_from_bar(
        self,
        trade_size: float,
        bar: pd.Series,
        avg_volume: float,
    ) -> SlippageEstimate:
        """
        Estimate slippage from a bar's data.

        Args:
            trade_size: Trade size in shares
            bar: Series with OHLCV and features
            avg_volume: Average volume for spread estimation

        Returns:
            SlippageEstimate
        """
        price = bar["close"]
        volume = bar.get("volume", avg_volume)

        # Get volatility (ATR-based)
        volatility = bar.get("atr_pct", 1.0)

        # Get spread estimate
        spread_bps = bar.get("estimated_spread_bps", 5.0)

        return self.estimate(trade_size, price, volume, volatility, spread_bps)
