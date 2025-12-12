"""
Weekly regime classifier for macro market context.

Classifies market into BULL, BEAR, or SIDEWAYS based on trend and momentum indicators.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class RegimeType(Enum):
    """Market regime classification."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class RegimeContext:
    """Weekly regime context information."""

    regime: RegimeType
    confidence: float  # 0-1 confidence score
    ema_alignment: int  # 1: bullish, -1: bearish, 0: neutral
    momentum_score: float  # Aggregate momentum
    volatility_regime: int  # -1: low, 0: normal, 1: high
    rrg_quadrant: Optional[str] = None
    bars_in_regime: int = 0

    def allows_long(self) -> bool:
        """Check if regime allows long mean-reversion trades."""
        return self.regime in [RegimeType.BULL, RegimeType.SIDEWAYS]

    def allows_short(self) -> bool:
        """Check if regime allows short mean-reversion trades."""
        return self.regime in [RegimeType.BEAR, RegimeType.SIDEWAYS]


class WeeklyRegimeClassifier:
    """
    Classifier for weekly market regime.

    Uses multiple indicators to determine if market is in:
    - BULL: Uptrend, positive momentum
    - BEAR: Downtrend, negative momentum
    - SIDEWAYS: Range-bound, mixed signals
    """

    # Thresholds for classification
    BULLISH_ZSCORE = -0.5  # Z-score above this is bullish
    BEARISH_ZSCORE = 0.5  # Z-score below this is bearish
    MOMENTUM_THRESHOLD = 0.3

    def classify(
        self,
        df: pd.DataFrame,
        lookback: int = 1,
    ) -> RegimeContext:
        """
        Classify current weekly regime.

        Args:
            df: Weekly DataFrame with features computed
            lookback: Number of recent bars to consider

        Returns:
            RegimeContext with classification
        """
        if df.empty or len(df) < lookback:
            return RegimeContext(
                regime=RegimeType.SIDEWAYS,
                confidence=0.0,
                ema_alignment=0,
                momentum_score=0.0,
                volatility_regime=0,
            )

        # Get recent data
        recent = df.iloc[-lookback:]
        current = df.iloc[-1]

        # Collect signals
        signals = []

        # 1. EMA Alignment
        ema_alignment = self._get_ema_alignment(current)
        signals.append(ema_alignment)

        # 2. Price vs EMA position
        price_position = self._get_price_position(current)
        signals.append(price_position)

        # 3. Momentum (RSI, MACD)
        momentum = self._get_momentum_signal(current)
        signals.append(momentum)

        # 4. Trend structure
        trend_structure = self._get_trend_structure(current)
        signals.append(trend_structure)

        # 5. RRG quadrant (if available)
        rrg_signal, rrg_quadrant = self._get_rrg_signal(current)
        if rrg_signal is not None:
            signals.append(rrg_signal)

        # Aggregate signals
        avg_signal = np.mean([s for s in signals if s is not None])

        # Classify regime
        if avg_signal > self.MOMENTUM_THRESHOLD:
            regime = RegimeType.BULL
        elif avg_signal < -self.MOMENTUM_THRESHOLD:
            regime = RegimeType.BEAR
        else:
            regime = RegimeType.SIDEWAYS

        # Calculate confidence
        confidence = min(abs(avg_signal), 1.0)

        # Count bars in current regime
        bars_in_regime = self._count_regime_duration(df, regime)

        # Get volatility regime
        vol_regime = int(current.get("vol_regime", 0))

        return RegimeContext(
            regime=regime,
            confidence=confidence,
            ema_alignment=int(np.sign(ema_alignment)),
            momentum_score=float(current.get("momentum_score", 0)),
            volatility_regime=vol_regime,
            rrg_quadrant=rrg_quadrant,
            bars_in_regime=bars_in_regime,
        )

    def _get_ema_alignment(self, row: pd.Series) -> float:
        """Get EMA alignment signal (-1 to 1)."""
        if "ema_alignment" in row.index:
            return float(row["ema_alignment"])
        return 0.0

    def _get_price_position(self, row: pd.Series) -> float:
        """Get price position signal based on z-score."""
        if "zscore_price" in row.index:
            zscore = row["zscore_price"]
            # Normalize to -1 to 1
            return float(np.clip(zscore / 2, -1, 1))
        return 0.0

    def _get_momentum_signal(self, row: pd.Series) -> float:
        """Get momentum signal from RSI and MACD."""
        signals = []

        if "rsi" in row.index:
            # RSI: 30-70 range normalized
            rsi = row["rsi"]
            rsi_signal = (rsi - 50) / 50  # -1 to 1
            signals.append(rsi_signal)

        if "macd_cross" in row.index:
            signals.append(float(row["macd_cross"]))

        if "momentum_score" in row.index:
            # Already normalized
            signals.append(float(row["momentum_score"]) / 100)

        return np.mean(signals) if signals else 0.0

    def _get_trend_structure(self, row: pd.Series) -> float:
        """Get trend structure signal."""
        if "trend_structure" in row.index:
            return float(row["trend_structure"])
        return 0.0

    def _get_rrg_signal(self, row: pd.Series) -> tuple[Optional[float], Optional[str]]:
        """Get RRG-based signal."""
        quadrant = None
        signal = None

        if "rrg_quadrant" in row.index:
            quadrant = row["rrg_quadrant"]
            signal_map = {
                "LEADING": 1.0,
                "WEAKENING": 0.3,
                "LAGGING": -1.0,
                "IMPROVING": -0.3,
            }
            signal = signal_map.get(quadrant, 0.0)

        return signal, quadrant

    def _count_regime_duration(
        self,
        df: pd.DataFrame,
        current_regime: RegimeType,
    ) -> int:
        """Count how many bars we've been in current regime."""
        if len(df) < 2:
            return 1

        count = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]

            # Simple classification for historical bars
            ema_alignment = row.get("ema_alignment", 0)

            if current_regime == RegimeType.BULL and ema_alignment > 0:
                count += 1
            elif current_regime == RegimeType.BEAR and ema_alignment < 0:
                count += 1
            elif current_regime == RegimeType.SIDEWAYS and ema_alignment == 0:
                count += 1
            else:
                break

        return max(count, 1)

    def get_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify regime for entire series.

        Useful for backtesting and analysis.

        Args:
            df: DataFrame with features

        Returns:
            Series of RegimeType values
        """
        regimes = []

        for i in range(len(df)):
            subset = df.iloc[: i + 1]
            context = self.classify(subset)
            regimes.append(context.regime.value)

        return pd.Series(regimes, index=df.index, name="regime")
