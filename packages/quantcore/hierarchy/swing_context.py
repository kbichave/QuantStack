"""
4H swing context analyzer.

Identifies swing phases (impulse vs correction) for optimal entry timing.
Now wave-aware: integrates Elliott Wave-style context when available.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger


class SwingPhase(Enum):
    """Swing phase classification."""

    IMPULSE_UP = "IMPULSE_UP"  # Strong move up
    CORRECTION_DOWN = "CORRECTION_DOWN"  # Pullback in uptrend
    IMPULSE_DOWN = "IMPULSE_DOWN"  # Strong move down
    CORRECTION_UP = "CORRECTION_UP"  # Bounce in downtrend
    CONSOLIDATION = "CONSOLIDATION"  # No clear direction


@dataclass
class SwingContext:
    """4H swing context information with optional wave context."""

    phase: SwingPhase
    near_swing_low: bool
    near_swing_high: bool
    swing_strength: float  # 0-1
    bars_since_swing: int
    correction_depth: float  # % retracement
    trend_exhaustion: bool

    # Wave-aware extensions (optional, filled when wave features available)
    wave_role: str = "none"  # From WaveRole enum
    wave_stage: int = -1  # 1-5 for impulse, 10-12 for ABC
    wave_conf: float = 0.0  # Wave pattern confidence
    prob_impulse_up: float = 0.0  # Probability of impulse up structure
    prob_impulse_down: float = 0.0  # Probability of impulse down structure
    prob_corr_down: float = 0.0  # Probability of corrective down
    prob_corr_up: float = 0.0  # Probability of corrective up
    is_late_impulse: bool = False  # In wave 4/5 (caution zone)

    @property
    def is_corrective_down(self) -> bool:
        """Check if in corrective down structure (wave-aware)."""
        return (
            self.wave_role == "corr_down"
            or self.prob_corr_down > 0.6
            or self.phase == SwingPhase.CORRECTION_DOWN
        )

    @property
    def is_corrective_up(self) -> bool:
        """Check if in corrective up structure (wave-aware)."""
        return (
            self.wave_role == "corr_up"
            or self.prob_corr_up > 0.6
            or self.phase == SwingPhase.CORRECTION_UP
        )

    def optimal_for_long_mr(self) -> bool:
        """Check if optimal for long mean-reversion entry (wave-aware)."""
        # Wave-aware check first
        if self.wave_conf > 0.4:
            # In corrective down = ideal for long MR
            if self.is_corrective_down and not self.is_late_impulse:
                return True
            # In impulse wave 2 or 4 = also good for MR
            if self.wave_stage in [2, 4] and self.prob_impulse_up > 0.5:
                return True

        # Fallback to standard swing logic
        return (
            self.near_swing_low
            or self.phase == SwingPhase.CORRECTION_DOWN
            or (self.phase == SwingPhase.CONSOLIDATION and self.correction_depth > 30)
        )

    def optimal_for_short_mr(self) -> bool:
        """Check if optimal for short mean-reversion entry (wave-aware)."""
        if self.wave_conf > 0.4:
            if self.is_corrective_up and not self.is_late_impulse:
                return True
            if self.wave_stage in [2, 4] and self.prob_impulse_down > 0.5:
                return True

        return (
            self.near_swing_high
            or self.phase == SwingPhase.CORRECTION_UP
            or (self.phase == SwingPhase.CONSOLIDATION and self.correction_depth > 30)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase.value,
            "near_swing_low": self.near_swing_low,
            "near_swing_high": self.near_swing_high,
            "swing_strength": self.swing_strength,
            "bars_since_swing": self.bars_since_swing,
            "correction_depth": self.correction_depth,
            "trend_exhaustion": self.trend_exhaustion,
            "wave_role": self.wave_role,
            "wave_stage": self.wave_stage,
            "wave_conf": self.wave_conf,
            "is_corrective_down": self.is_corrective_down,
            "is_corrective_up": self.is_corrective_up,
            "is_late_impulse": self.is_late_impulse,
        }


class SwingContextAnalyzer:
    """
    Analyzer for 4H swing context.

    Determines the current swing phase and whether we're near
    probable swing points for optimal entry timing.

    Now wave-aware: incorporates EW-style wave labels when available.
    """

    # Thresholds
    NEAR_SWING_BARS = 5  # Bars to consider "near" a swing
    MIN_CORRECTION_DEPTH = 20  # Minimum % for correction
    EXHAUSTION_BARS = 6  # Consecutive bars for exhaustion

    def analyze(
        self,
        df: pd.DataFrame,
        trend_context: Optional["TrendContext"] = None,
        include_wave_context: bool = True,
    ) -> SwingContext:
        """
        Analyze 4H swing context with optional wave awareness.

        Args:
            df: 4H DataFrame with features (optionally with wave features)
            trend_context: Optional daily trend context for reference
            include_wave_context: Whether to include wave features if available

        Returns:
            SwingContext with analysis (including wave context if available)
        """
        if df.empty or len(df) < 5:
            return SwingContext(
                phase=SwingPhase.CONSOLIDATION,
                near_swing_low=False,
                near_swing_high=False,
                swing_strength=0.0,
                bars_since_swing=0,
                correction_depth=0.0,
                trend_exhaustion=False,
            )

        current = df.iloc[-1]

        # Check swing proximity
        near_swing_low = self._is_near_swing_low(df, current)
        near_swing_high = self._is_near_swing_high(df, current)

        # Calculate bars since swing
        bars_since_low = self._bars_since_swing(df, "probable_swing_low")
        bars_since_high = self._bars_since_swing(df, "probable_swing_high")
        bars_since_swing = min(bars_since_low, bars_since_high)

        # Determine swing phase
        phase = self._classify_phase(df, current, trend_context)

        # Calculate correction depth
        correction_depth = self._calculate_correction_depth(df, phase)

        # Check trend exhaustion
        trend_exhaustion = self._check_exhaustion(current)

        # Calculate swing strength
        swing_strength = self._calculate_swing_strength(df, phase)

        # Extract wave context if available
        wave_role = "none"
        wave_stage = -1
        wave_conf = 0.0
        prob_impulse_up = 0.0
        prob_impulse_down = 0.0
        prob_corr_down = 0.0
        prob_corr_up = 0.0
        is_late_impulse = False

        if include_wave_context and "wave_role" in df.columns:
            wave_role = str(current.get("wave_role", "none"))
            wave_stage = int(current.get("wave_stage", -1))
            wave_conf = float(current.get("wave_conf", 0.0))
            prob_impulse_up = float(current.get("prob_impulse_up", 0.0))
            prob_impulse_down = float(current.get("prob_impulse_down", 0.0))
            prob_corr_down = float(current.get("prob_corr_down", 0.0))
            prob_corr_up = float(current.get("prob_corr_up", 0.0))

            # Late impulse = wave 4/5 or terminal impulse
            is_late_impulse = wave_role in [
                "impulse_up_terminal",
                "impulse_down_terminal",
            ] or wave_stage in [4, 5]

        return SwingContext(
            phase=phase,
            near_swing_low=near_swing_low,
            near_swing_high=near_swing_high,
            swing_strength=swing_strength,
            bars_since_swing=bars_since_swing,
            correction_depth=correction_depth,
            trend_exhaustion=trend_exhaustion,
            wave_role=wave_role,
            wave_stage=wave_stage,
            wave_conf=wave_conf,
            prob_impulse_up=prob_impulse_up,
            prob_impulse_down=prob_impulse_down,
            prob_corr_down=prob_corr_down,
            prob_corr_up=prob_corr_up,
            is_late_impulse=is_late_impulse,
        )

    def _is_near_swing_low(self, df: pd.DataFrame, current: pd.Series) -> bool:
        """Check if we're near a probable swing low."""
        # Recent swing low signal
        if "probable_swing_low" in current.index:
            if current["probable_swing_low"] == 1:
                return True

        # Or recent swing low within lookback
        if "probable_swing_low" in df.columns:
            recent_swings = df["probable_swing_low"].iloc[-self.NEAR_SWING_BARS :]
            if recent_swings.sum() > 0:
                return True

        # Or near recent low
        if "dist_to_swing_low" in current.index:
            if current["dist_to_swing_low"] < 1.0:  # Within 1%
                return True

        return False

    def _is_near_swing_high(self, df: pd.DataFrame, current: pd.Series) -> bool:
        """Check if we're near a probable swing high."""
        if "probable_swing_high" in current.index:
            if current["probable_swing_high"] == 1:
                return True

        if "probable_swing_high" in df.columns:
            recent_swings = df["probable_swing_high"].iloc[-self.NEAR_SWING_BARS :]
            if recent_swings.sum() > 0:
                return True

        if "dist_to_swing_high" in current.index:
            if current["dist_to_swing_high"] < 1.0:
                return True

        return False

    def _bars_since_swing(self, df: pd.DataFrame, column: str) -> int:
        """Count bars since last swing signal."""
        if column not in df.columns:
            return 999

        swing_signals = df[column]
        last_swing_idx = swing_signals[swing_signals == 1].index

        if len(last_swing_idx) == 0:
            return 999

        last_swing = last_swing_idx[-1]
        bars = len(df.loc[last_swing:]) - 1
        return bars

    def _classify_phase(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        trend_context: Optional["TrendContext"],
    ) -> SwingPhase:
        """Classify current swing phase."""
        # Get recent price action
        recent_returns = df["close"].pct_change().iloc[-5:]
        avg_return = recent_returns.mean()

        # Get momentum
        momentum = current.get("momentum_score", 0)

        # Get trend structure
        structure = current.get("trend_structure", 0)

        # EMA alignment
        ema_alignment = current.get("ema_alignment", 0)

        # Classify based on signals
        if avg_return > 0.002 and momentum > 20 and ema_alignment > 0:
            return SwingPhase.IMPULSE_UP
        elif avg_return < -0.002 and momentum < -20 and ema_alignment < 0:
            return SwingPhase.IMPULSE_DOWN
        elif avg_return < -0.001 and ema_alignment > 0:
            # Pullback in uptrend
            return SwingPhase.CORRECTION_DOWN
        elif avg_return > 0.001 and ema_alignment < 0:
            # Bounce in downtrend
            return SwingPhase.CORRECTION_UP
        else:
            return SwingPhase.CONSOLIDATION

    def _calculate_correction_depth(
        self,
        df: pd.DataFrame,
        phase: SwingPhase,
    ) -> float:
        """Calculate correction depth as percentage of prior move."""
        if len(df) < 20:
            return 0.0

        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()

        current_close = df["close"].iloc[-1]
        recent_high = high_20.iloc[-1]
        recent_low = low_20.iloc[-1]

        range_size = recent_high - recent_low
        if range_size <= 0:
            return 0.0

        if phase in [SwingPhase.CORRECTION_DOWN, SwingPhase.IMPULSE_DOWN]:
            # How far from high
            return (recent_high - current_close) / range_size * 100
        else:
            # How far from low
            return (current_close - recent_low) / range_size * 100

    def _check_exhaustion(self, current: pd.Series) -> bool:
        """Check if current trend is showing exhaustion."""
        uptrend_exhaust = current.get("uptrend_exhaustion", 0) == 1
        downtrend_exhaust = current.get("downtrend_exhaustion", 0) == 1
        return uptrend_exhaust or downtrend_exhaust

    def _calculate_swing_strength(
        self,
        df: pd.DataFrame,
        phase: SwingPhase,
    ) -> float:
        """Calculate strength of current swing move."""
        if len(df) < 10:
            return 0.5

        # Recent volatility-adjusted move
        recent_returns = df["close"].pct_change().iloc[-5:].sum()
        atr_pct = df["atr_pct"].iloc[-1] if "atr_pct" in df.columns else 1.0

        if atr_pct <= 0:
            return 0.5

        strength = abs(recent_returns * 100) / atr_pct
        return float(np.clip(strength / 3, 0, 1))  # Normalize

    def get_phase_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Get swing phase for entire series.

        Args:
            df: DataFrame with features

        Returns:
            Series of SwingPhase values
        """
        phases = []

        for i in range(len(df)):
            subset = df.iloc[: i + 1]
            context = self.analyze(subset)
            phases.append(context.phase.value)

        return pd.Series(phases, index=df.index, name="swing_phase")
