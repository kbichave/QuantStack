"""
Wave-aware context analyzer for 4H timeframe.

Provides wave phase summaries for alignment checks and signal generation.
Integrates with swing_context to add Elliott Wave-style interpretation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.waves import (
    WaveFeatures,
    WaveRole,
    WaveConfig,
    WAVE_CONFIGS,
    SwingPoint,
    SwingLeg,
)
from quantcore.config.timeframes import Timeframe


@dataclass
class WaveContextSummary:
    """
    Summary of current wave context for a single timeframe.

    Used by alignment logic and signal generation.
    """

    # Current wave classification
    wave_role: str  # From WaveRole enum
    wave_stage: int  # 1-5 for impulse, 10-12 for ABC
    wave_conf: float  # Pattern confidence 0-1

    # Partial pattern probabilities
    prob_impulse_up: float
    prob_impulse_down: float
    prob_corr_down: float
    prob_corr_up: float

    # Derived flags for easy filtering
    is_corrective_down: bool  # In downward correction
    is_corrective_up: bool  # In upward correction
    is_impulse_active: bool  # In an impulse wave
    is_late_impulse: bool  # In wave 4 or 5 (caution zone)

    # Additional context
    bars_in_current_wave: int  # How long in this wave
    current_leg_ret_pct: float  # Return of current leg

    def for_long_mr(self) -> bool:
        """Check if wave context favors long mean-reversion."""
        # Ideal: corrective down within larger uptrend
        if self.is_corrective_down:
            return True
        if self.prob_corr_down > 0.6:
            return True
        # Also allow consolidation within impulse up
        if self.prob_impulse_up > 0.5 and self.wave_stage in [2, 4]:
            return True
        return False

    def for_short_mr(self) -> bool:
        """Check if wave context favors short mean-reversion."""
        if self.is_corrective_up:
            return True
        if self.prob_corr_up > 0.6:
            return True
        if self.prob_impulse_down > 0.5 and self.wave_stage in [2, 4]:
            return True
        return False

    def caution_for_long(self) -> bool:
        """Check if should be cautious about long entries."""
        # Late impulse up = likely exhaustion
        if self.is_late_impulse and self.prob_impulse_up > 0.5:
            return True
        # Strong impulse down = not corrective, don't catch falling knife
        if self.prob_impulse_down > 0.7:
            return True
        return False

    def caution_for_short(self) -> bool:
        """Check if should be cautious about short entries."""
        if self.is_late_impulse and self.prob_impulse_down > 0.5:
            return True
        if self.prob_impulse_up > 0.7:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "wave_role": self.wave_role,
            "wave_stage": self.wave_stage,
            "wave_conf": self.wave_conf,
            "prob_impulse_up": self.prob_impulse_up,
            "prob_impulse_down": self.prob_impulse_down,
            "prob_corr_down": self.prob_corr_down,
            "prob_corr_up": self.prob_corr_up,
            "is_corrective_down": self.is_corrective_down,
            "is_corrective_up": self.is_corrective_up,
            "is_impulse_active": self.is_impulse_active,
            "is_late_impulse": self.is_late_impulse,
            "for_long_mr": self.for_long_mr(),
            "for_short_mr": self.for_short_mr(),
            "caution_long": self.caution_for_long(),
            "caution_short": self.caution_for_short(),
        }


class WaveContextAnalyzer:
    """
    Analyzes wave context from price data.

    Primary use is 4H timeframe, but can be applied to Daily for
    higher-level wave structure.
    """

    def __init__(self, timeframe: Timeframe = Timeframe.H4):
        """
        Initialize wave context analyzer.

        Args:
            timeframe: Timeframe for wave analysis (default 4H)
        """
        self.timeframe = timeframe
        self.wave_features = WaveFeatures(timeframe)
        self.config = WAVE_CONFIGS.get(timeframe, WaveConfig())

    def analyze(self, df: pd.DataFrame) -> WaveContextSummary:
        """
        Analyze current wave context from DataFrame.

        Args:
            df: DataFrame with OHLCV (and optionally precomputed wave features)

        Returns:
            WaveContextSummary with current wave state
        """
        if df.empty or len(df) < 10:
            return self._empty_context()

        # Check if wave features already computed
        if "wave_role" not in df.columns:
            df = self.wave_features.compute(df)

        current = df.iloc[-1]

        # Extract wave state
        wave_role = str(current.get("wave_role", "none"))
        wave_stage = int(current.get("wave_stage", -1))
        wave_conf = float(current.get("wave_conf", 0.0))

        prob_impulse_up = float(current.get("prob_impulse_up", 0.0))
        prob_impulse_down = float(current.get("prob_impulse_down", 0.0))
        prob_corr_down = float(current.get("prob_corr_down", 0.0))
        prob_corr_up = float(current.get("prob_corr_up", 0.0))

        # Derive flags
        is_corrective_down = (
            wave_role == WaveRole.CORR_DOWN.value or prob_corr_down > 0.6
        )

        is_corrective_up = wave_role == WaveRole.CORR_UP.value or prob_corr_up > 0.6

        is_impulse_active = wave_role in [
            WaveRole.IMPULSE_UP.value,
            WaveRole.IMPULSE_DOWN.value,
            WaveRole.IMPULSE_UP_TERMINAL.value,
            WaveRole.IMPULSE_DOWN_TERMINAL.value,
        ]

        is_late_impulse = wave_role in [
            WaveRole.IMPULSE_UP_TERMINAL.value,
            WaveRole.IMPULSE_DOWN_TERMINAL.value,
        ] or wave_stage in [4, 5]

        # Calculate bars in current wave
        bars_in_wave = self._count_bars_in_wave(df)

        # Current leg return
        current_leg_ret = self._get_current_leg_return(df)

        return WaveContextSummary(
            wave_role=wave_role,
            wave_stage=wave_stage,
            wave_conf=wave_conf,
            prob_impulse_up=prob_impulse_up,
            prob_impulse_down=prob_impulse_down,
            prob_corr_down=prob_corr_down,
            prob_corr_up=prob_corr_up,
            is_corrective_down=is_corrective_down,
            is_corrective_up=is_corrective_up,
            is_impulse_active=is_impulse_active,
            is_late_impulse=is_late_impulse,
            bars_in_current_wave=bars_in_wave,
            current_leg_ret_pct=current_leg_ret,
        )

    def _empty_context(self) -> WaveContextSummary:
        """Return empty context when insufficient data."""
        return WaveContextSummary(
            wave_role="none",
            wave_stage=-1,
            wave_conf=0.0,
            prob_impulse_up=0.0,
            prob_impulse_down=0.0,
            prob_corr_down=0.0,
            prob_corr_up=0.0,
            is_corrective_down=False,
            is_corrective_up=False,
            is_impulse_active=False,
            is_late_impulse=False,
            bars_in_current_wave=0,
            current_leg_ret_pct=0.0,
        )

    def _count_bars_in_wave(self, df: pd.DataFrame) -> int:
        """Count how many bars we've been in the current wave stage."""
        if "wave_stage" not in df.columns:
            return 0

        current_stage = df["wave_stage"].iloc[-1]
        if current_stage == -1:
            return 0

        # Count backwards until stage changes
        count = 0
        for i in range(len(df) - 1, -1, -1):
            if df["wave_stage"].iloc[i] == current_stage:
                count += 1
            else:
                break

        return count

    def _get_current_leg_return(self, df: pd.DataFrame) -> float:
        """Get return of current swing leg."""
        # Find recent swing point
        swings, legs = self.wave_features.get_swings_and_legs(df)

        if not legs:
            return 0.0

        # Return of last completed leg
        return legs[-1].ret_pct * 100

    def get_wave_context_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get wave context for entire series.

        Useful for backtesting - creates context columns for each bar.

        Args:
            df: DataFrame with OHLCV

        Returns:
            DataFrame with wave context columns added
        """
        result = df.copy()

        # Compute wave features if not present
        if "wave_role" not in result.columns:
            result = self.wave_features.compute(result)

        # Derive additional context columns
        result["wave_is_corr_down"] = (
            (result["wave_role"] == WaveRole.CORR_DOWN.value)
            | (result["prob_corr_down"] > 0.6)
        ).astype(int)

        result["wave_is_corr_up"] = (
            (result["wave_role"] == WaveRole.CORR_UP.value)
            | (result["prob_corr_up"] > 0.6)
        ).astype(int)

        result["wave_is_late_impulse"] = (
            (
                result["wave_role"].isin(
                    [
                        WaveRole.IMPULSE_UP_TERMINAL.value,
                        WaveRole.IMPULSE_DOWN_TERMINAL.value,
                    ]
                )
            )
            | (result["wave_stage"].isin([4, 5]))
        ).astype(int)

        # Favorable for MR long: corrective down, not late impulse
        result["wave_favor_long_mr"] = (
            result["wave_is_corr_down"] & ~result["wave_is_late_impulse"].astype(bool)
        ).astype(int)

        # Favorable for MR short: corrective up, not late impulse
        result["wave_favor_short_mr"] = (
            result["wave_is_corr_up"] & ~result["wave_is_late_impulse"].astype(bool)
        ).astype(int)

        return result


class MultiTimeframeWaveContext:
    """
    Combines wave context from multiple timeframes.

    Primary: 4H for wave structure
    Secondary: Daily for larger wave frame
    """

    def __init__(self):
        """Initialize multi-TF wave context."""
        self.h4_analyzer = WaveContextAnalyzer(Timeframe.H4)
        self.d1_analyzer = WaveContextAnalyzer(Timeframe.D1)

    def analyze(
        self,
        df_h4: pd.DataFrame,
        df_d1: Optional[pd.DataFrame] = None,
    ) -> Dict[str, WaveContextSummary]:
        """
        Analyze wave context for both timeframes.

        Args:
            df_h4: 4H DataFrame
            df_d1: Optional Daily DataFrame

        Returns:
            Dict with H4 and D1 context summaries
        """
        result = {}

        # Primary: 4H context
        result["H4"] = self.h4_analyzer.analyze(df_h4)

        # Secondary: D1 context
        if df_d1 is not None and not df_d1.empty:
            result["D1"] = self.d1_analyzer.analyze(df_d1)
        else:
            result["D1"] = self.h4_analyzer._empty_context()

        return result

    def get_combined_signal_quality(
        self,
        contexts: Dict[str, WaveContextSummary],
        direction: str,
    ) -> float:
        """
        Get combined wave quality score for signal direction.

        Args:
            contexts: Dict with H4 and D1 context
            direction: "LONG" or "SHORT"

        Returns:
            Quality score 0-1
        """
        h4 = contexts.get("H4")
        d1 = contexts.get("D1")

        if h4 is None:
            return 0.5

        score = 0.5

        if direction == "LONG":
            # 4H corrective down is ideal
            if h4.for_long_mr():
                score += 0.25
            if h4.caution_for_long():
                score -= 0.2

            # Daily context bonus
            if d1 is not None:
                if d1.prob_impulse_up > 0.5:  # Daily impulse up = good for long MR
                    score += 0.15
                if d1.is_corrective_down:  # Larger correction = more caution
                    score -= 0.1

        else:  # SHORT
            if h4.for_short_mr():
                score += 0.25
            if h4.caution_for_short():
                score -= 0.2

            if d1 is not None:
                if d1.prob_impulse_down > 0.5:
                    score += 0.15
                if d1.is_corrective_up:
                    score -= 0.1

        return max(0.0, min(1.0, score))
