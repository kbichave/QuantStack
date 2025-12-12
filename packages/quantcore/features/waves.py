"""
Elliott Wave-style wave grammar on top of swing detection.

Implements:
- ZigZag swing detection (ATR-based)
- Wave pattern detection (impulse/correction)
- Partial pattern scoring for live execution
- Wave feature computation for ML
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class WaveRole(Enum):
    """Classification of wave role in larger structure."""

    IMPULSE_UP = "impulse_up"
    IMPULSE_DOWN = "impulse_down"
    IMPULSE_UP_TERMINAL = "impulse_up_terminal"  # Wave 5
    IMPULSE_DOWN_TERMINAL = "impulse_down_terminal"  # Wave 5
    CORR_UP = "corr_up"  # Corrective bounce
    CORR_DOWN = "corr_down"  # Corrective pullback
    ABC_COMPLETE = "abc_complete"
    NONE = "none"


@dataclass
class SwingPoint:
    """A swing high or low point in price series."""

    idx: int  # Index position in DataFrame
    time: pd.Timestamp  # Timestamp
    price: float  # Price at swing point
    direction: Literal["up", "down"]  # Leg direction TO this swing

    def __repr__(self) -> str:
        return f"Swing({self.direction}→{self.price:.4f}@{self.idx})"


@dataclass
class SwingLeg:
    """A leg between two consecutive swing points."""

    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    direction: Literal["up", "down"]
    ret_pct: float  # Percentage return (signed)
    length_bars: int  # Duration in bars

    @property
    def abs_ret_pct(self) -> float:
        return abs(self.ret_pct)

    def __repr__(self) -> str:
        return (
            f"Leg({self.direction}: {self.ret_pct:+.2%} over {self.length_bars} bars)"
        )


@dataclass
class WavePattern:
    """Detected wave pattern (impulse or correction)."""

    type: Literal["impulse_up", "impulse_down", "abc_up", "abc_down"]
    leg_indices: List[int]  # Indices in legs list
    start_idx: int  # DataFrame index where pattern starts
    end_idx: int  # DataFrame index where pattern ends
    confidence: float  # Pattern quality score 0-1
    stages: List[int]  # Wave stages for each leg (1-5 or A-C encoded)

    def __repr__(self) -> str:
        return f"WavePattern({self.type}, conf={self.confidence:.2f})"


@dataclass
class WaveConfig:
    """Configuration for wave detection thresholds."""

    # ZigZag swing detection
    atr_mult: float = 1.5  # ATR multiple for swing reversal
    min_swing_bars: int = 3  # Minimum bars for swing to be valid

    # Impulse pattern constraints
    min_impulse_pct: float = 0.01  # 1% minimum impulse move
    max_wave2_retrace: float = 1.0  # Wave 2 can't retrace 100% of wave 1
    max_wave4_retrace: float = 1.0  # Wave 4 can't retrace 100% of wave 3
    min_wave3_vs_wave1: float = 1.0  # Wave 3 >= Wave 1 size
    max_wave5_vs_wave3: float = 1.5  # Wave 5 not excessively larger than 3
    wave2_ideal_retrace: Tuple[float, float] = (0.382, 0.618)  # Fib range
    wave4_ideal_retrace: Tuple[float, float] = (0.236, 0.50)  # Fib range

    # ABC correction constraints
    min_corr_pct: float = 0.005  # 0.5% minimum correction move
    max_b_wave_retrace: float = 1.0  # B wave can't exceed A wave high
    c_wave_min_vs_a: float = 0.618  # C wave at least 61.8% of A

    # Pattern scoring weights
    fib_weight: float = 0.3  # Weight for Fibonacci alignment
    structure_weight: float = 0.4  # Weight for proper structure
    length_weight: float = 0.3  # Weight for appropriate leg lengths

    # Partial pattern thresholds
    partial_min_legs: int = 2  # Min legs to score partial pattern


# Default configs by timeframe
WAVE_CONFIGS = {
    Timeframe.W1: WaveConfig(atr_mult=2.0, min_swing_bars=2),
    Timeframe.D1: WaveConfig(atr_mult=1.5, min_swing_bars=3),
    Timeframe.H4: WaveConfig(atr_mult=1.5, min_swing_bars=3),
    Timeframe.H1: WaveConfig(atr_mult=1.2, min_swing_bars=4),
}


class SwingDetector:
    """
    ZigZag-style swing detector using ATR-based thresholds.

    Produces a list of alternating swing highs and lows from price data.
    No lookahead: each swing is confirmed only after sufficient reversal.
    """

    def __init__(self, config: WaveConfig):
        self.config = config

    def detect(self, df: pd.DataFrame, atr_column: str = "atr") -> List[SwingPoint]:
        """
        Detect swing points in OHLCV data.

        Args:
            df: DataFrame with OHLCV and ATR column
            atr_column: Name of ATR column

        Returns:
            List of SwingPoint in chronological order, alternating up/down
        """
        if len(df) < 5:
            return []

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Handle ATR
        if atr_column in df.columns:
            atr = df[atr_column].values
        else:
            atr = self._compute_atr(df)

        times = df.index.to_list()
        n = len(df)

        swings: List[SwingPoint] = []

        # Initialize - find first significant move
        last_idx = 0
        last_price = closes[0]
        direction: Optional[Literal["up", "down"]] = None

        for i in range(1, n):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            threshold = self.config.atr_mult * atr[i]

            # Determine initial direction
            if direction is None:
                move = closes[i] - last_price
                if abs(move) >= threshold:
                    direction = "up" if move > 0 else "down"
                    last_idx = 0
                    last_price = closes[0]
                continue

            if direction == "up":
                # In up leg: track new highs
                if highs[i] >= last_price:
                    last_idx = i
                    last_price = highs[i]
                else:
                    # Check reversal down
                    move_down = last_price - lows[i]
                    if move_down >= threshold:
                        # Validate swing
                        if (
                            last_idx - (swings[-1].idx if swings else 0)
                            >= self.config.min_swing_bars
                        ):
                            swings.append(
                                SwingPoint(
                                    idx=last_idx,
                                    time=times[last_idx],
                                    price=last_price,
                                    direction="up",
                                )
                            )
                        # Start new down leg
                        direction = "down"
                        last_idx = i
                        last_price = lows[i]

            else:  # direction == "down"
                # In down leg: track new lows
                if lows[i] <= last_price:
                    last_idx = i
                    last_price = lows[i]
                else:
                    # Check reversal up
                    move_up = highs[i] - last_price
                    if move_up >= threshold:
                        if (
                            last_idx - (swings[-1].idx if swings else 0)
                            >= self.config.min_swing_bars
                        ):
                            swings.append(
                                SwingPoint(
                                    idx=last_idx,
                                    time=times[last_idx],
                                    price=last_price,
                                    direction="down",
                                )
                            )
                        direction = "up"
                        last_idx = i
                        last_price = highs[i]

        return swings

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Compute ATR if not available."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]

        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # EMA of true range
        atr = np.zeros(len(df))
        atr[:period] = np.nan
        atr[period - 1] = np.mean(tr[:period])

        multiplier = 2 / (period + 1)
        for i in range(period, len(df)):
            atr[i] = (tr[i] * multiplier) + (atr[i - 1] * (1 - multiplier))

        return atr

    def build_legs(self, swings: List[SwingPoint]) -> List[SwingLeg]:
        """Convert swing points to leg objects."""
        legs = []
        for i in range(1, len(swings)):
            sp_prev = swings[i - 1]
            sp_curr = swings[i]

            ret_pct = (
                (sp_curr.price - sp_prev.price) / sp_prev.price
                if sp_prev.price != 0
                else 0
            )

            legs.append(
                SwingLeg(
                    start_idx=sp_prev.idx,
                    end_idx=sp_curr.idx,
                    start_price=sp_prev.price,
                    end_price=sp_curr.price,
                    direction=sp_curr.direction,
                    ret_pct=ret_pct,
                    length_bars=sp_curr.idx - sp_prev.idx,
                )
            )

        return legs


class WaveLabeler:
    """
    Labels swing leg sequences with Elliott Wave-style patterns.

    Detects:
    - 5-wave impulse structures (up and down)
    - 3-wave corrections (ABC patterns)
    - Partial patterns for live execution
    """

    def __init__(self, config: WaveConfig):
        self.config = config

    def detect_impulse_up(self, legs: List[SwingLeg]) -> List[WavePattern]:
        """
        Detect impulse-up patterns (5 legs: up, down, up, down, up).

        Returns list of detected patterns with confidence scores.
        """
        patterns = []
        n = len(legs)

        for i in range(0, n - 4):  # Need 5 consecutive legs
            L1, L2, L3, L4, L5 = legs[i : i + 5]

            # Check directions: up, down, up, down, up
            if not (
                L1.direction == "up"
                and L2.direction == "down"
                and L3.direction == "up"
                and L4.direction == "down"
                and L5.direction == "up"
            ):
                continue

            # Size constraints
            if L1.ret_pct <= self.config.min_impulse_pct:
                continue
            if L3.ret_pct <= self.config.min_impulse_pct:
                continue

            # Wave 2 retrace check
            if L2.ret_pct >= 0:  # Should be negative
                continue
            w2_retrace = abs(L2.ret_pct) / abs(L1.ret_pct) if L1.ret_pct != 0 else 999
            if w2_retrace > self.config.max_wave2_retrace:
                continue

            # Wave 4 retrace check
            if L4.ret_pct >= 0:  # Should be negative
                continue
            w4_retrace = abs(L4.ret_pct) / abs(L3.ret_pct) if L3.ret_pct != 0 else 999
            if w4_retrace > self.config.max_wave4_retrace:
                continue

            # Wave 3 should be at least as big as wave 1
            if abs(L3.ret_pct) < self.config.min_wave3_vs_wave1 * abs(L1.ret_pct):
                continue

            # Calculate confidence based on Fibonacci alignment
            confidence = self._score_impulse_up(L1, L2, L3, L4, L5)

            patterns.append(
                WavePattern(
                    type="impulse_up",
                    leg_indices=list(range(i, i + 5)),
                    start_idx=L1.start_idx,
                    end_idx=L5.end_idx,
                    confidence=confidence,
                    stages=[1, 2, 3, 4, 5],
                )
            )

        return patterns

    def detect_impulse_down(self, legs: List[SwingLeg]) -> List[WavePattern]:
        """Detect impulse-down patterns (5 legs: down, up, down, up, down)."""
        patterns = []
        n = len(legs)

        for i in range(0, n - 4):
            L1, L2, L3, L4, L5 = legs[i : i + 5]

            # Check directions: down, up, down, up, down
            if not (
                L1.direction == "down"
                and L2.direction == "up"
                and L3.direction == "down"
                and L4.direction == "up"
                and L5.direction == "down"
            ):
                continue

            # Size constraints (use abs for down moves)
            if abs(L1.ret_pct) <= self.config.min_impulse_pct:
                continue
            if abs(L3.ret_pct) <= self.config.min_impulse_pct:
                continue

            # Wave 2 retrace (should be positive bounce)
            if L2.ret_pct <= 0:
                continue
            w2_retrace = abs(L2.ret_pct) / abs(L1.ret_pct) if L1.ret_pct != 0 else 999
            if w2_retrace > self.config.max_wave2_retrace:
                continue

            # Wave 4 retrace
            if L4.ret_pct <= 0:
                continue
            w4_retrace = abs(L4.ret_pct) / abs(L3.ret_pct) if L3.ret_pct != 0 else 999
            if w4_retrace > self.config.max_wave4_retrace:
                continue

            # Wave 3 >= wave 1
            if abs(L3.ret_pct) < self.config.min_wave3_vs_wave1 * abs(L1.ret_pct):
                continue

            confidence = self._score_impulse_down(L1, L2, L3, L4, L5)

            patterns.append(
                WavePattern(
                    type="impulse_down",
                    leg_indices=list(range(i, i + 5)),
                    start_idx=L1.start_idx,
                    end_idx=L5.end_idx,
                    confidence=confidence,
                    stages=[1, 2, 3, 4, 5],
                )
            )

        return patterns

    def detect_abc_down(self, legs: List[SwingLeg]) -> List[WavePattern]:
        """Detect ABC correction down (3 legs: down, up, down)."""
        patterns = []
        n = len(legs)

        for i in range(0, n - 2):
            LA, LB, LC = legs[i : i + 3]

            # Directions: down, up, down
            if not (
                LA.direction == "down"
                and LB.direction == "up"
                and LC.direction == "down"
            ):
                continue

            # Size constraints
            if abs(LA.ret_pct) <= self.config.min_corr_pct:
                continue

            # B wave shouldn't exceed A wave start (no overlap in typical ABC)
            b_retrace = abs(LB.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 999
            if b_retrace > self.config.max_b_wave_retrace:
                continue

            # C wave at least 61.8% of A
            if abs(LC.ret_pct) < self.config.c_wave_min_vs_a * abs(LA.ret_pct):
                continue

            confidence = self._score_abc(LA, LB, LC)

            patterns.append(
                WavePattern(
                    type="abc_down",
                    leg_indices=list(range(i, i + 3)),
                    start_idx=LA.start_idx,
                    end_idx=LC.end_idx,
                    confidence=confidence,
                    stages=[10, 11, 12],  # A=10, B=11, C=12 encoding
                )
            )

        return patterns

    def detect_abc_up(self, legs: List[SwingLeg]) -> List[WavePattern]:
        """Detect ABC correction up (3 legs: up, down, up)."""
        patterns = []
        n = len(legs)

        for i in range(0, n - 2):
            LA, LB, LC = legs[i : i + 3]

            # Directions: up, down, up
            if not (
                LA.direction == "up" and LB.direction == "down" and LC.direction == "up"
            ):
                continue

            if abs(LA.ret_pct) <= self.config.min_corr_pct:
                continue

            b_retrace = abs(LB.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 999
            if b_retrace > self.config.max_b_wave_retrace:
                continue

            if abs(LC.ret_pct) < self.config.c_wave_min_vs_a * abs(LA.ret_pct):
                continue

            confidence = self._score_abc(LA, LB, LC)

            patterns.append(
                WavePattern(
                    type="abc_up",
                    leg_indices=list(range(i, i + 3)),
                    start_idx=LA.start_idx,
                    end_idx=LC.end_idx,
                    confidence=confidence,
                    stages=[10, 11, 12],
                )
            )

        return patterns

    def _score_impulse_up(
        self, L1: SwingLeg, L2: SwingLeg, L3: SwingLeg, L4: SwingLeg, L5: SwingLeg
    ) -> float:
        """Score impulse-up pattern quality."""
        score = 0.5  # Base score for valid structure

        # Fibonacci alignment for wave 2 (ideal: 38.2%-61.8%)
        w2_retrace = abs(L2.ret_pct) / abs(L1.ret_pct) if L1.ret_pct != 0 else 0
        if (
            self.config.wave2_ideal_retrace[0]
            <= w2_retrace
            <= self.config.wave2_ideal_retrace[1]
        ):
            score += 0.15 * self.config.fib_weight

        # Fibonacci alignment for wave 4 (ideal: 23.6%-50%)
        w4_retrace = abs(L4.ret_pct) / abs(L3.ret_pct) if L3.ret_pct != 0 else 0
        if (
            self.config.wave4_ideal_retrace[0]
            <= w4_retrace
            <= self.config.wave4_ideal_retrace[1]
        ):
            score += 0.15 * self.config.fib_weight

        # Wave 3 being the largest is classic
        waves_up = [abs(L1.ret_pct), abs(L3.ret_pct), abs(L5.ret_pct)]
        if abs(L3.ret_pct) == max(waves_up):
            score += 0.2 * self.config.structure_weight

        # Proportional leg lengths
        lengths = [
            L1.length_bars,
            L2.length_bars,
            L3.length_bars,
            L4.length_bars,
            L5.length_bars,
        ]
        avg_len = np.mean(lengths)
        len_variance = np.var(lengths) / (avg_len**2) if avg_len > 0 else 1
        if len_variance < 0.5:  # Low variance = consistent rhythm
            score += 0.1 * self.config.length_weight

        return min(score, 1.0)

    def _score_impulse_down(
        self, L1: SwingLeg, L2: SwingLeg, L3: SwingLeg, L4: SwingLeg, L5: SwingLeg
    ) -> float:
        """Score impulse-down pattern quality."""
        # Mirror of impulse_up scoring
        score = 0.5

        w2_retrace = abs(L2.ret_pct) / abs(L1.ret_pct) if L1.ret_pct != 0 else 0
        if (
            self.config.wave2_ideal_retrace[0]
            <= w2_retrace
            <= self.config.wave2_ideal_retrace[1]
        ):
            score += 0.15 * self.config.fib_weight

        w4_retrace = abs(L4.ret_pct) / abs(L3.ret_pct) if L3.ret_pct != 0 else 0
        if (
            self.config.wave4_ideal_retrace[0]
            <= w4_retrace
            <= self.config.wave4_ideal_retrace[1]
        ):
            score += 0.15 * self.config.fib_weight

        waves_down = [abs(L1.ret_pct), abs(L3.ret_pct), abs(L5.ret_pct)]
        if abs(L3.ret_pct) == max(waves_down):
            score += 0.2 * self.config.structure_weight

        lengths = [
            L1.length_bars,
            L2.length_bars,
            L3.length_bars,
            L4.length_bars,
            L5.length_bars,
        ]
        avg_len = np.mean(lengths)
        len_variance = np.var(lengths) / (avg_len**2) if avg_len > 0 else 1
        if len_variance < 0.5:
            score += 0.1 * self.config.length_weight

        return min(score, 1.0)

    def _score_abc(self, LA: SwingLeg, LB: SwingLeg, LC: SwingLeg) -> float:
        """Score ABC correction pattern."""
        score = 0.5

        # B wave ideal retrace: 50-61.8%
        b_retrace = abs(LB.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 0
        if 0.5 <= b_retrace <= 0.618:
            score += 0.2

        # C wave often equals A wave (1.0 ratio)
        c_vs_a = abs(LC.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 0
        if 0.9 <= c_vs_a <= 1.1:
            score += 0.2
        elif 0.618 <= c_vs_a <= 1.618:  # Within Fib bounds
            score += 0.1

        return min(score, 1.0)

    def detect_all_patterns(self, legs: List[SwingLeg]) -> List[WavePattern]:
        """Detect all pattern types and return sorted by start index."""
        all_patterns = []

        all_patterns.extend(self.detect_impulse_up(legs))
        all_patterns.extend(self.detect_impulse_down(legs))
        all_patterns.extend(self.detect_abc_down(legs))
        all_patterns.extend(self.detect_abc_up(legs))

        # Sort by start index
        all_patterns.sort(key=lambda p: p.start_idx)

        return all_patterns


class PartialPatternScorer:
    """
    Scores partial patterns for live execution.

    Evaluates whether current swing sequence looks like an in-progress
    impulse or correction, returning probability scores.
    """

    def __init__(self, config: WaveConfig):
        self.config = config

    def score_partial_impulse_up(self, legs: List[SwingLeg]) -> float:
        """
        Score how much the recent legs look like an in-progress impulse up.

        Returns probability 0-1 that we're in an impulse up structure.
        """
        if len(legs) < self.config.partial_min_legs:
            return 0.0

        # Take last 4 legs max
        recent = legs[-4:] if len(legs) >= 4 else legs

        score = 0.0

        # Check for alternating pattern starting with up
        expected_dirs = ["up", "down", "up", "down"][: len(recent)]
        actual_dirs = [leg.direction for leg in recent]

        if actual_dirs != expected_dirs:
            return 0.0

        # Base score for correct direction sequence
        score = 0.3 + 0.1 * len(recent)

        # Check wave relationships
        if len(recent) >= 2:
            L1 = recent[0]
            L2 = recent[1]

            # L1 should be substantial impulse
            if L1.ret_pct >= self.config.min_impulse_pct:
                score += 0.1

            # L2 should be proper retrace
            w2_retrace = abs(L2.ret_pct) / abs(L1.ret_pct) if L1.ret_pct != 0 else 0
            if w2_retrace < self.config.max_wave2_retrace:
                score += 0.1
                if (
                    self.config.wave2_ideal_retrace[0]
                    <= w2_retrace
                    <= self.config.wave2_ideal_retrace[1]
                ):
                    score += 0.1

        if len(recent) >= 3:
            L1, L2, L3 = recent[0], recent[1], recent[2]
            # Wave 3 developing - should be strong
            if abs(L3.ret_pct) >= abs(L1.ret_pct):
                score += 0.15

        return min(score, 1.0)

    def score_partial_impulse_down(self, legs: List[SwingLeg]) -> float:
        """Score partial impulse down pattern."""
        if len(legs) < self.config.partial_min_legs:
            return 0.0

        recent = legs[-4:] if len(legs) >= 4 else legs

        score = 0.0
        expected_dirs = ["down", "up", "down", "up"][: len(recent)]
        actual_dirs = [leg.direction for leg in recent]

        if actual_dirs != expected_dirs:
            return 0.0

        score = 0.3 + 0.1 * len(recent)

        if len(recent) >= 2:
            L1, L2 = recent[0], recent[1]
            if abs(L1.ret_pct) >= self.config.min_impulse_pct:
                score += 0.1

            w2_retrace = abs(L2.ret_pct) / abs(L1.ret_pct) if L1.ret_pct != 0 else 0
            if w2_retrace < self.config.max_wave2_retrace:
                score += 0.1
                if (
                    self.config.wave2_ideal_retrace[0]
                    <= w2_retrace
                    <= self.config.wave2_ideal_retrace[1]
                ):
                    score += 0.1

        if len(recent) >= 3:
            L1, L2, L3 = recent[0], recent[1], recent[2]
            if abs(L3.ret_pct) >= abs(L1.ret_pct):
                score += 0.15

        return min(score, 1.0)

    def score_partial_corr_down(self, legs: List[SwingLeg]) -> float:
        """Score partial correction down (pullback in uptrend)."""
        if len(legs) < 1:
            return 0.0

        recent = legs[-3:] if len(legs) >= 3 else legs

        # For correction down, expect: down, up, down (A, B, C)
        expected_dirs = ["down", "up", "down"][: len(recent)]
        actual_dirs = [leg.direction for leg in recent]

        if actual_dirs != expected_dirs:
            # Check if just starting correction (single down leg)
            if len(recent) == 1 and recent[0].direction == "down":
                return 0.3  # Early correction signal
            return 0.0

        score = 0.4

        if len(recent) >= 2:
            LA, LB = recent[0], recent[1]
            # B wave retrace
            b_retrace = abs(LB.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 0
            if b_retrace < self.config.max_b_wave_retrace:
                score += 0.2
                if 0.5 <= b_retrace <= 0.618:
                    score += 0.1

        if len(recent) >= 3:
            LA, LB, LC = recent[0], recent[1], recent[2]
            c_vs_a = abs(LC.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 0
            if c_vs_a >= self.config.c_wave_min_vs_a:
                score += 0.15

        return min(score, 1.0)

    def score_partial_corr_up(self, legs: List[SwingLeg]) -> float:
        """Score partial correction up (bounce in downtrend)."""
        if len(legs) < 1:
            return 0.0

        recent = legs[-3:] if len(legs) >= 3 else legs

        expected_dirs = ["up", "down", "up"][: len(recent)]
        actual_dirs = [leg.direction for leg in recent]

        if actual_dirs != expected_dirs:
            if len(recent) == 1 and recent[0].direction == "up":
                return 0.3
            return 0.0

        score = 0.4

        if len(recent) >= 2:
            LA, LB = recent[0], recent[1]
            b_retrace = abs(LB.ret_pct) / abs(LA.ret_pct) if LA.ret_pct != 0 else 0
            if b_retrace < self.config.max_b_wave_retrace:
                score += 0.2
                if 0.5 <= b_retrace <= 0.618:
                    score += 0.1

        return min(score, 1.0)


class WaveFeatures(FeatureBase):
    """
    Compute wave-related features for ML and signal generation.

    Outputs:
    - wave_role: Current bar's wave role classification
    - wave_stage: Wave number (1-5 or A/B/C encoded)
    - wave_conf: Confidence in current pattern
    - prob_impulse_up: Probability of in-progress impulse up
    - prob_impulse_down: Probability of in-progress impulse down
    - prob_corr_down: Probability of corrective down structure
    - prob_corr_up: Probability of corrective up structure
    """

    def __init__(self, timeframe: Timeframe):
        super().__init__(timeframe)
        self.config = WAVE_CONFIGS.get(timeframe, WaveConfig())
        self.swing_detector = SwingDetector(self.config)
        self.wave_labeler = WaveLabeler(self.config)
        self.partial_scorer = PartialPatternScorer(self.config)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute wave features."""
        result = df.copy()
        n = len(result)

        # Initialize columns
        result["wave_role"] = "none"
        result["wave_stage"] = -1
        result["wave_conf"] = 0.0
        result["prob_impulse_up"] = 0.0
        result["prob_impulse_down"] = 0.0
        result["prob_corr_down"] = 0.0
        result["prob_corr_up"] = 0.0

        if n < 10:
            return result

        # Detect swings
        swings = self.swing_detector.detect(result)
        if len(swings) < 2:
            return result

        # Build legs
        legs = self.swing_detector.build_legs(swings)
        if not legs:
            return result

        # Detect complete patterns
        patterns = self.wave_labeler.detect_all_patterns(legs)

        # Assign wave labels from complete patterns
        result = self._assign_pattern_labels(result, legs, patterns)

        # Compute partial pattern scores for each bar
        result = self._compute_partial_scores(result, legs)

        logger.debug(
            f"Wave features: {len(swings)} swings, {len(patterns)} patterns detected"
        )

        return result

    def _assign_pattern_labels(
        self,
        df: pd.DataFrame,
        legs: List[SwingLeg],
        patterns: List[WavePattern],
    ) -> pd.DataFrame:
        """Assign wave labels from detected patterns."""
        # Create arrays for labels
        n = len(df)
        wave_stage = np.full(n, -1, dtype=int)
        wave_role = np.array(["none"] * n, dtype=object)
        wave_conf = np.zeros(n)

        # Role mapping for each pattern type and stage
        ROLE_MAPPING = {
            "impulse_up": {
                1: WaveRole.IMPULSE_UP.value,
                2: WaveRole.CORR_DOWN.value,
                3: WaveRole.IMPULSE_UP.value,
                4: WaveRole.CORR_DOWN.value,
                5: WaveRole.IMPULSE_UP_TERMINAL.value,
            },
            "impulse_down": {
                1: WaveRole.IMPULSE_DOWN.value,
                2: WaveRole.CORR_UP.value,
                3: WaveRole.IMPULSE_DOWN.value,
                4: WaveRole.CORR_UP.value,
                5: WaveRole.IMPULSE_DOWN_TERMINAL.value,
            },
            "abc_down": {
                10: WaveRole.CORR_DOWN.value,  # A
                11: WaveRole.CORR_UP.value,  # B
                12: WaveRole.CORR_DOWN.value,  # C
            },
            "abc_up": {
                10: WaveRole.CORR_UP.value,  # A
                11: WaveRole.CORR_DOWN.value,  # B
                12: WaveRole.CORR_UP.value,  # C
            },
        }

        for p in patterns:
            role_map = ROLE_MAPPING.get(p.type, {})

            for offset, leg_idx in enumerate(p.leg_indices):
                if leg_idx >= len(legs):
                    continue

                leg = legs[leg_idx]
                stage = p.stages[offset]
                role = role_map.get(stage, WaveRole.NONE.value)

                # Propagate labels to all bars in this leg
                start = leg.start_idx
                end = min(leg.end_idx + 1, n)

                for bar_idx in range(start, end):
                    # Keep higher confidence pattern
                    if p.confidence > wave_conf[bar_idx]:
                        wave_stage[bar_idx] = stage
                        wave_role[bar_idx] = role
                        wave_conf[bar_idx] = p.confidence

        df["wave_stage"] = wave_stage
        df["wave_role"] = wave_role
        df["wave_conf"] = wave_conf

        return df

    def _compute_partial_scores(
        self,
        df: pd.DataFrame,
        legs: List[SwingLeg],
    ) -> pd.DataFrame:
        """Compute partial pattern probabilities for each bar."""
        n = len(df)

        prob_impulse_up = np.zeros(n)
        prob_impulse_down = np.zeros(n)
        prob_corr_down = np.zeros(n)
        prob_corr_up = np.zeros(n)

        # For efficiency, compute at leg boundaries and forward fill
        leg_ends = sorted(set([leg.end_idx for leg in legs]))

        for leg_end in leg_ends:
            if leg_end >= n:
                continue

            # Get legs up to this point
            legs_to_here = [l for l in legs if l.end_idx <= leg_end]
            if not legs_to_here:
                continue

            # Score partial patterns
            p_imp_up = self.partial_scorer.score_partial_impulse_up(legs_to_here)
            p_imp_down = self.partial_scorer.score_partial_impulse_down(legs_to_here)
            p_corr_down = self.partial_scorer.score_partial_corr_down(legs_to_here)
            p_corr_up = self.partial_scorer.score_partial_corr_up(legs_to_here)

            prob_impulse_up[leg_end] = p_imp_up
            prob_impulse_down[leg_end] = p_imp_down
            prob_corr_down[leg_end] = p_corr_down
            prob_corr_up[leg_end] = p_corr_up

        # Forward fill between leg boundaries
        last_values = (0, 0, 0, 0)
        for i in range(n):
            if (
                prob_impulse_up[i] > 0
                or prob_impulse_down[i] > 0
                or prob_corr_down[i] > 0
                or prob_corr_up[i] > 0
            ):
                last_values = (
                    prob_impulse_up[i],
                    prob_impulse_down[i],
                    prob_corr_down[i],
                    prob_corr_up[i],
                )
            else:
                prob_impulse_up[i] = last_values[0]
                prob_impulse_down[i] = last_values[1]
                prob_corr_down[i] = last_values[2]
                prob_corr_up[i] = last_values[3]

        df["prob_impulse_up"] = prob_impulse_up
        df["prob_impulse_down"] = prob_impulse_down
        df["prob_corr_down"] = prob_corr_down
        df["prob_corr_up"] = prob_corr_up

        return df

    def get_feature_names(self) -> List[str]:
        """Return wave feature names."""
        return [
            "wave_role",
            "wave_stage",
            "wave_conf",
            "prob_impulse_up",
            "prob_impulse_down",
            "prob_corr_down",
            "prob_corr_up",
        ]

    def get_swings_and_legs(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[SwingPoint], List[SwingLeg]]:
        """
        Get raw swings and legs for analysis.

        Useful for visualization and research.
        """
        swings = self.swing_detector.detect(df)
        legs = self.swing_detector.build_legs(swings)
        return swings, legs

    def get_patterns(self, df: pd.DataFrame) -> List[WavePattern]:
        """Get detected wave patterns for analysis."""
        swings = self.swing_detector.detect(df)
        legs = self.swing_detector.build_legs(swings)
        return self.wave_labeler.detect_all_patterns(legs)
