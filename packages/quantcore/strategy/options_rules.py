"""
Rule-based options strategies.

Implements strategies that use the existing MeanReversionRules
with options-specific contract selection.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.strategy.base import (
    Strategy,
    MarketState,
    TargetPosition,
    DataRequirements,
    PositionDirection,
    RegimeState,
)
from quantcore.options.contract_selector import (
    ContractSelector,
    Direction,
    VolRegime,
    TrendRegime,
)
from quantcore.features.options_features import classify_vol_regime


class OptionsDirectionalStrategy(Strategy):
    """
    Rule-based directional options strategy.

    Uses technical signals from existing features to determine direction,
    then selects appropriate options structure based on vol regime.
    """

    def __init__(
        self,
        name: str = "OptionsDirectional",
        zscore_threshold: float = 2.0,
        reversion_delta: float = 0.2,
        min_confidence: float = 0.3,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            zscore_threshold: Z-score threshold for entry signals
            reversion_delta: Minimum z-score change for confirmation
            min_confidence: Minimum confidence to generate signal
        """
        super().__init__(name)
        self.zscore_threshold = zscore_threshold
        self.reversion_delta = reversion_delta
        self.min_confidence = min_confidence
        self.contract_selector = ContractSelector()

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """Generate signals based on mean reversion rules."""
        # Get features
        features = state.features

        # Check for z-score entry conditions
        # Try multiple feature name patterns (timeframe-prefixed and bare names)
        zscore = features.get("1H_zscore_price", features.get("zscore_price", 0))
        # Use the lagged version or current if not available
        zscore_prev = features.get(
            "1H_zscore_price_lag1", features.get("zscore_price_prev", zscore - 0.1)
        )

        # Determine direction based on z-score stretch + reversion
        direction = PositionDirection.FLAT
        confidence = 0.0
        reason = ""

        # Long signal: oversold + starting to revert up
        if zscore_prev < -self.zscore_threshold:
            if zscore > zscore_prev + self.reversion_delta:
                direction = PositionDirection.LONG
                confidence = min(abs(zscore_prev) / 3.0, 1.0)
                reason = f"MR Long: z={zscore:.2f}, prev={zscore_prev:.2f}"

        # Short signal: overbought + starting to revert down
        elif zscore_prev > self.zscore_threshold:
            if zscore < zscore_prev - self.reversion_delta:
                direction = PositionDirection.SHORT
                confidence = min(abs(zscore_prev) / 3.0, 1.0)
                reason = f"MR Short: z={zscore:.2f}, prev={zscore_prev:.2f}"

        # Apply regime filters
        if state.regime:
            # In BEAR regime, reduce long confidence
            if (
                state.regime.trend_regime == "BEAR"
                and direction == PositionDirection.LONG
            ):
                confidence *= 0.5
                reason += " (reduced: BEAR regime)"

            # In BULL regime, reduce short confidence
            if (
                state.regime.trend_regime == "BULL"
                and direction == PositionDirection.SHORT
            ):
                confidence *= 0.5
                reason += " (reduced: BULL regime)"

        # Check confidence threshold
        if confidence < self.min_confidence:
            return []

        # Check earnings gate
        if state.days_to_earnings is not None and state.days_to_earnings <= 5:
            logger.debug(f"Earnings gate: {state.days_to_earnings} days to earnings")
            confidence *= 0.5

        return [
            TargetPosition(
                symbol=state.symbol,
                direction=direction,
                confidence=confidence,
                reason=reason,
                signal_strength=abs(zscore_prev),
            )
        ]

    def get_required_data(self) -> DataRequirements:
        """Specify data needs."""
        return DataRequirements(
            timeframes=["1H", "4H", "1D", "1W"],
            need_options_chain=True,
            need_earnings_calendar=True,
            lookback_bars=252,
        )


class OptionsMomentumStrategy(Strategy):
    """
    Momentum-based options strategy.

    Goes with the trend in trending markets,
    uses mean reversion in sideways markets.
    """

    def __init__(
        self,
        name: str = "OptionsMomentum",
        trend_threshold: float = 0.5,
        momentum_lookback: int = 20,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            trend_threshold: Threshold for trend strength
            momentum_lookback: Lookback for momentum calculation
        """
        super().__init__(name)
        self.trend_threshold = trend_threshold
        self.momentum_lookback = momentum_lookback

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """Generate signals based on momentum."""
        features = state.features

        # Get trend features - try timeframe-prefixed names first
        ema_alignment = features.get(
            "1H_ema_alignment", features.get("ema_alignment", 0)
        )
        momentum_score = features.get(
            "1H_momentum_score", features.get("momentum_score", 0)
        )
        rsi = features.get("1H_rsi", features.get("rsi", 50))

        # Determine direction
        direction = PositionDirection.FLAT
        confidence = 0.0
        reason = ""

        if state.regime and state.regime.trend_regime in ["BULL", "BEAR"]:
            # Trending market - go with momentum
            if momentum_score > self.trend_threshold:
                direction = PositionDirection.LONG
                confidence = min(momentum_score, 1.0)
                reason = f"Momentum Long: score={momentum_score:.2f}, RSI={rsi:.1f}"
            elif momentum_score < -self.trend_threshold:
                direction = PositionDirection.SHORT
                confidence = min(abs(momentum_score), 1.0)
                reason = f"Momentum Short: score={momentum_score:.2f}, RSI={rsi:.1f}"
        else:
            # Sideways market - use RSI extremes
            if rsi < 30:
                direction = PositionDirection.LONG
                confidence = (30 - rsi) / 30
                reason = f"RSI Oversold: RSI={rsi:.1f}"
            elif rsi > 70:
                direction = PositionDirection.SHORT
                confidence = (rsi - 70) / 30
                reason = f"RSI Overbought: RSI={rsi:.1f}"

        if direction == PositionDirection.FLAT:
            return []

        return [
            TargetPosition(
                symbol=state.symbol,
                direction=direction,
                confidence=confidence,
                reason=reason,
            )
        ]

    def get_required_data(self) -> DataRequirements:
        """Specify data needs."""
        return DataRequirements(
            timeframes=["1H", "4H", "1D"],
            need_options_chain=True,
            lookback_bars=100,
        )


class OptionsVolatilityStrategy(Strategy):
    """
    Volatility-based options strategy.

    Trades volatility expansion/contraction using options.
    """

    def __init__(
        self,
        name: str = "OptionsVolatility",
        iv_rank_low: float = 30,
        iv_rank_high: float = 70,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            iv_rank_low: IV rank threshold for low vol
            iv_rank_high: IV rank threshold for high vol
        """
        super().__init__(name)
        self.iv_rank_low = iv_rank_low
        self.iv_rank_high = iv_rank_high

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """Generate signals based on volatility."""
        iv_rank = state.iv_rank

        if iv_rank is None:
            return []

        features = state.features
        trend_direction = features.get(
            "1H_ema_alignment", features.get("ema_alignment", 0)
        )

        direction = PositionDirection.FLAT
        confidence = 0.0
        reason = ""

        # Low IV rank - expect vol expansion, use long options
        if iv_rank < self.iv_rank_low:
            if trend_direction > 0:
                direction = PositionDirection.LONG
                confidence = (self.iv_rank_low - iv_rank) / self.iv_rank_low
                reason = f"Low IV + Bullish: IV rank={iv_rank:.1f}"
            elif trend_direction < 0:
                direction = PositionDirection.SHORT
                confidence = (self.iv_rank_low - iv_rank) / self.iv_rank_low
                reason = f"Low IV + Bearish: IV rank={iv_rank:.1f}"

        # High IV rank - expect vol contraction
        # In high IV, we'd use spreads (defined risk) - direction based on trend
        elif iv_rank > self.iv_rank_high:
            if trend_direction > 0:
                direction = PositionDirection.LONG
                confidence = 0.5  # Lower confidence in high IV
                reason = f"High IV + Bullish (use spread): IV rank={iv_rank:.1f}"
            elif trend_direction < 0:
                direction = PositionDirection.SHORT
                confidence = 0.5
                reason = f"High IV + Bearish (use spread): IV rank={iv_rank:.1f}"

        if direction == PositionDirection.FLAT:
            return []

        return [
            TargetPosition(
                symbol=state.symbol,
                direction=direction,
                confidence=confidence,
                reason=reason,
            )
        ]

    def get_required_data(self) -> DataRequirements:
        """Specify data needs."""
        return DataRequirements(
            timeframes=["1D"],
            need_options_chain=True,
            lookback_bars=252,
        )


class OptionsRRGStrategy(Strategy):
    """
    RRG-based options strategy.

    Uses Relative Rotation Graph quadrants for cross-sectional signals.
    """

    def __init__(
        self,
        name: str = "OptionsRRG",
        favorable_quadrants: List[str] = None,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            favorable_quadrants: Quadrants favorable for longs
        """
        super().__init__(name)
        self.favorable_quadrants = favorable_quadrants or ["LEADING", "IMPROVING"]

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """Generate signals based on RRG quadrant."""
        features = state.features

        rrg_leading = features.get("rrg_leading", 0)
        rrg_improving = features.get("rrg_improving", 0)
        rrg_weakening = features.get("rrg_weakening", 0)
        rrg_lagging = features.get("rrg_lagging", 0)

        direction = PositionDirection.FLAT
        confidence = 0.0
        reason = ""

        # Leading or Improving -> Long
        if rrg_leading or rrg_improving:
            direction = PositionDirection.LONG
            rs_ratio = features.get("rs_ratio", 100)
            rs_momentum = features.get("rs_momentum", 100)
            confidence = min(((rs_ratio - 100) + (rs_momentum - 100)) / 10, 1.0)
            quadrant = "LEADING" if rrg_leading else "IMPROVING"
            reason = (
                f"RRG {quadrant}: RS ratio={rs_ratio:.1f}, momentum={rs_momentum:.1f}"
            )

        # Weakening or Lagging -> Short (or avoid)
        elif rrg_weakening or rrg_lagging:
            direction = PositionDirection.SHORT
            rs_ratio = features.get("rs_ratio", 100)
            rs_momentum = features.get("rs_momentum", 100)
            confidence = min((100 - rs_ratio + 100 - rs_momentum) / 10, 1.0)
            quadrant = "WEAKENING" if rrg_weakening else "LAGGING"
            reason = (
                f"RRG {quadrant}: RS ratio={rs_ratio:.1f}, momentum={rs_momentum:.1f}"
            )

        if direction == PositionDirection.FLAT or confidence < 0.3:
            return []

        return [
            TargetPosition(
                symbol=state.symbol,
                direction=direction,
                confidence=confidence,
                reason=reason,
            )
        ]

    def get_required_data(self) -> DataRequirements:
        """Specify data needs."""
        return DataRequirements(
            timeframes=["1D", "1W"],
            need_options_chain=True,
            lookback_bars=100,
        )
