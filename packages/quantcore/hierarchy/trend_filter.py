"""
Daily trend filter for intermediate trend direction.

Determines if the daily trend supports mean-reversion trades.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class TrendDirection(Enum):
    """Trend direction classification."""

    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass
class TrendContext:
    """Daily trend context information."""

    direction: TrendDirection
    strength: float  # 0-1 strength score
    slope: float  # Regression slope
    price_vs_ema: float  # Price distance from EMA (%)
    momentum: float  # Momentum score
    consecutive_bars: int  # Consecutive bars in direction

    def allows_long(self) -> bool:
        """Check if trend allows long MR trades."""
        # Allow longs in uptrend or neutral, not in strong downtrend
        if self.direction == TrendDirection.DOWN and self.strength > 0.7:
            return False
        return True

    def allows_short(self) -> bool:
        """Check if trend allows short MR trades."""
        # Allow shorts in downtrend or neutral, not in strong uptrend
        if self.direction == TrendDirection.UP and self.strength > 0.7:
            return False
        return True


class DailyTrendFilter:
    """
    Filter for daily trend direction.

    Analyzes daily timeframe to determine intermediate trend
    and whether it supports mean-reversion trades.
    """

    # Thresholds
    SLOPE_THRESHOLD = 0.1  # % per bar for trend detection
    PRICE_EMA_THRESHOLD = 1.0  # % distance for trend confirmation
    STRONG_TREND_THRESHOLD = 0.7

    def analyze(
        self,
        df: pd.DataFrame,
        lookback: int = 1,
    ) -> TrendContext:
        """
        Analyze daily trend.

        Args:
            df: Daily DataFrame with features
            lookback: Number of bars to consider

        Returns:
            TrendContext with analysis
        """
        if df.empty or len(df) < lookback:
            return TrendContext(
                direction=TrendDirection.NEUTRAL,
                strength=0.0,
                slope=0.0,
                price_vs_ema=0.0,
                momentum=0.0,
                consecutive_bars=0,
            )

        current = df.iloc[-1]

        # Collect signals
        signals = []

        # 1. Regression slope
        slope = self._get_slope(current)
        slope_signal = self._slope_to_signal(slope)
        signals.append(slope_signal)

        # 2. Price vs EMA
        price_vs_ema = self._get_price_vs_ema(current)
        ema_signal = self._price_ema_to_signal(price_vs_ema)
        signals.append(ema_signal)

        # 3. EMA alignment
        ema_alignment = self._get_ema_alignment(current)
        signals.append(ema_alignment)

        # 4. Momentum
        momentum = self._get_momentum(current)
        mom_signal = self._momentum_to_signal(momentum)
        signals.append(mom_signal)

        # 5. Price structure
        structure_signal = self._get_structure_signal(current)
        signals.append(structure_signal)

        # Aggregate signals
        avg_signal = np.mean([s for s in signals if s is not None])

        # Determine direction
        if avg_signal > 0.2:
            direction = TrendDirection.UP
        elif avg_signal < -0.2:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.NEUTRAL

        # Calculate strength
        strength = min(abs(avg_signal), 1.0)

        # Count consecutive bars
        consecutive = self._count_consecutive_trend(df, direction)

        return TrendContext(
            direction=direction,
            strength=strength,
            slope=slope,
            price_vs_ema=price_vs_ema,
            momentum=momentum,
            consecutive_bars=consecutive,
        )

    def _get_slope(self, row: pd.Series) -> float:
        """Get regression slope."""
        if "regression_slope" in row.index:
            return float(row["regression_slope"])
        return 0.0

    def _slope_to_signal(self, slope: float) -> float:
        """Convert slope to signal."""
        if abs(slope) < self.SLOPE_THRESHOLD:
            return 0.0
        return np.clip(slope / (self.SLOPE_THRESHOLD * 3), -1, 1)

    def _get_price_vs_ema(self, row: pd.Series) -> float:
        """Get price distance from EMA."""
        if "price_dist_ema_fast" in row.index:
            return float(row["price_dist_ema_fast"])
        return 0.0

    def _price_ema_to_signal(self, dist: float) -> float:
        """Convert price-EMA distance to signal."""
        if abs(dist) < self.PRICE_EMA_THRESHOLD:
            return 0.0
        return np.clip(dist / (self.PRICE_EMA_THRESHOLD * 3), -1, 1)

    def _get_ema_alignment(self, row: pd.Series) -> float:
        """Get EMA alignment."""
        if "ema_alignment" in row.index:
            return float(row["ema_alignment"])
        return 0.0

    def _get_momentum(self, row: pd.Series) -> float:
        """Get momentum score."""
        if "momentum_score" in row.index:
            return float(row["momentum_score"])
        return 0.0

    def _momentum_to_signal(self, momentum: float) -> float:
        """Convert momentum to signal."""
        return np.clip(momentum / 100, -1, 1)

    def _get_structure_signal(self, row: pd.Series) -> float:
        """Get trend structure signal."""
        if "trend_structure" in row.index:
            return float(row["trend_structure"])
        return 0.0

    def _count_consecutive_trend(
        self,
        df: pd.DataFrame,
        direction: TrendDirection,
    ) -> int:
        """Count consecutive bars in trend direction."""
        if len(df) < 2:
            return 1

        count = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]

            slope = row.get("regression_slope", 0)

            if direction == TrendDirection.UP and slope > 0:
                count += 1
            elif direction == TrendDirection.DOWN and slope < 0:
                count += 1
            elif (
                direction == TrendDirection.NEUTRAL
                and abs(slope) < self.SLOPE_THRESHOLD
            ):
                count += 1
            else:
                break

        return max(count, 1)

    def get_trend_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Get trend direction for entire series.

        Args:
            df: DataFrame with features

        Returns:
            Series of TrendDirection values
        """
        directions = []

        for i in range(len(df)):
            subset = df.iloc[: i + 1]
            context = self.analyze(subset)
            directions.append(context.direction.value)

        return pd.Series(directions, index=df.index, name="trend")
