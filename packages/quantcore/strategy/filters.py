"""
Filters for signal validation.

Includes RRG-based and swing-based filters.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class FilterResult:
    """Result of filter evaluation."""

    passed: bool
    reason: str = ""
    score: float = 1.0  # 0-1 filter quality score


class RRGFilter:
    """
    Filter based on Relative Rotation Graph quadrant.

    For LONG trades:
    - Allow: LEADING, IMPROVING quadrants
    - Block: LAGGING quadrant (persistent underperformer)
    - Cautious: WEAKENING (allow with reduced confidence)

    For SHORT trades:
    - Allow: LAGGING, WEAKENING quadrants
    - Block: LEADING quadrant
    - Cautious: IMPROVING
    """

    # Quadrant configurations
    LONG_ALLOWED = ["LEADING", "IMPROVING"]
    LONG_BLOCKED = ["LAGGING"]
    LONG_CAUTIOUS = ["WEAKENING"]

    SHORT_ALLOWED = ["LAGGING", "WEAKENING"]
    SHORT_BLOCKED = ["LEADING"]
    SHORT_CAUTIOUS = ["IMPROVING"]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize RRG filter.

        Args:
            strict_mode: If True, block cautious quadrants too
        """
        self.strict_mode = strict_mode

    def check_long(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> FilterResult:
        """
        Check if RRG allows long trade.

        Args:
            df: DataFrame with RRG features
            idx: Bar index to check

        Returns:
            FilterResult
        """
        if "rrg_quadrant" not in df.columns:
            # No RRG data - pass by default
            return FilterResult(passed=True, reason="No RRG data", score=0.5)

        quadrant = df.iloc[idx].get("rrg_quadrant", None)

        if quadrant is None or pd.isna(quadrant):
            return FilterResult(passed=True, reason="No quadrant", score=0.5)

        if quadrant in self.LONG_ALLOWED:
            return FilterResult(passed=True, reason=f"RRG {quadrant}", score=1.0)
        elif quadrant in self.LONG_BLOCKED:
            return FilterResult(
                passed=False, reason=f"RRG {quadrant} blocks long", score=0.0
            )
        elif quadrant in self.LONG_CAUTIOUS:
            if self.strict_mode:
                return FilterResult(
                    passed=False, reason=f"RRG {quadrant} (strict)", score=0.3
                )
            return FilterResult(
                passed=True, reason=f"RRG {quadrant} (cautious)", score=0.5
            )

        return FilterResult(passed=True, reason="Unknown quadrant", score=0.5)

    def check_short(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> FilterResult:
        """
        Check if RRG allows short trade.

        Args:
            df: DataFrame with RRG features
            idx: Bar index to check

        Returns:
            FilterResult
        """
        if "rrg_quadrant" not in df.columns:
            return FilterResult(passed=True, reason="No RRG data", score=0.5)

        quadrant = df.iloc[idx].get("rrg_quadrant", None)

        if quadrant is None or pd.isna(quadrant):
            return FilterResult(passed=True, reason="No quadrant", score=0.5)

        if quadrant in self.SHORT_ALLOWED:
            return FilterResult(passed=True, reason=f"RRG {quadrant}", score=1.0)
        elif quadrant in self.SHORT_BLOCKED:
            return FilterResult(
                passed=False, reason=f"RRG {quadrant} blocks short", score=0.0
            )
        elif quadrant in self.SHORT_CAUTIOUS:
            if self.strict_mode:
                return FilterResult(
                    passed=False, reason=f"RRG {quadrant} (strict)", score=0.3
                )
            return FilterResult(
                passed=True, reason=f"RRG {quadrant} (cautious)", score=0.5
            )

        return FilterResult(passed=True, reason="Unknown quadrant", score=0.5)


class SwingFilter:
    """
    Filter based on market structure swing points.

    For LONG trades:
    - Prefer: Near probable swing low
    - Allow: Recent swing low within lookback

    For SHORT trades:
    - Prefer: Near probable swing high
    - Allow: Recent swing high within lookback
    """

    def __init__(
        self,
        lookback_bars: int = 5,
        require_swing: bool = False,
    ):
        """
        Initialize swing filter.

        Args:
            lookback_bars: Bars to look back for recent swing
            require_swing: If True, require swing for signal
        """
        self.lookback_bars = lookback_bars
        self.require_swing = require_swing

    def check_long(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> FilterResult:
        """
        Check if swing structure supports long trade.

        Args:
            df: DataFrame with swing features
            idx: Bar index

        Returns:
            FilterResult
        """
        current = df.iloc[idx]

        # Check current bar swing low
        if "probable_swing_low" in df.columns:
            if current.get("probable_swing_low", 0) == 1:
                return FilterResult(passed=True, reason="At swing low", score=1.0)

        # Check recent swing low
        if "probable_swing_low" in df.columns:
            start_idx = max(0, idx - self.lookback_bars)
            recent = df.iloc[start_idx : idx + 1]["probable_swing_low"]
            if recent.sum() > 0:
                bars_ago = len(recent) - recent.values[::-1].argmax() - 1
                score = 1.0 - (bars_ago / self.lookback_bars) * 0.5
                return FilterResult(
                    passed=True,
                    reason=f"Swing low {bars_ago} bars ago",
                    score=score,
                )

        # Check near support
        if "near_support" in df.columns:
            if current.get("near_support", 0) == 1:
                return FilterResult(passed=True, reason="Near support", score=0.7)

        # No swing found
        if self.require_swing:
            return FilterResult(passed=False, reason="No recent swing low", score=0.0)

        return FilterResult(passed=True, reason="No swing filter", score=0.3)

    def check_short(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> FilterResult:
        """
        Check if swing structure supports short trade.

        Args:
            df: DataFrame with swing features
            idx: Bar index

        Returns:
            FilterResult
        """
        current = df.iloc[idx]

        # Check current bar swing high
        if "probable_swing_high" in df.columns:
            if current.get("probable_swing_high", 0) == 1:
                return FilterResult(passed=True, reason="At swing high", score=1.0)

        # Check recent swing high
        if "probable_swing_high" in df.columns:
            start_idx = max(0, idx - self.lookback_bars)
            recent = df.iloc[start_idx : idx + 1]["probable_swing_high"]
            if recent.sum() > 0:
                bars_ago = len(recent) - recent.values[::-1].argmax() - 1
                score = 1.0 - (bars_ago / self.lookback_bars) * 0.5
                return FilterResult(
                    passed=True,
                    reason=f"Swing high {bars_ago} bars ago",
                    score=score,
                )

        # Check near resistance
        if "near_resistance" in df.columns:
            if current.get("near_resistance", 0) == 1:
                return FilterResult(passed=True, reason="Near resistance", score=0.7)

        if self.require_swing:
            return FilterResult(passed=False, reason="No recent swing high", score=0.0)

        return FilterResult(passed=True, reason="No swing filter", score=0.3)


class WaveFilter:
    """
    Filter based on Elliott Wave-style wave context.

    For LONG trades:
    - Allow: Corrective down phases (pullback in uptrend)
    - Allow: Wave 2/4 within impulse up
    - Block: Late impulse (wave 5) - potential exhaustion

    For SHORT trades:
    - Allow: Corrective up phases (bounce in downtrend)
    - Allow: Wave 2/4 within impulse down
    - Block: Late impulse (wave 5)
    """

    # Probability thresholds
    CORR_THRESHOLD = 0.6  # Minimum prob for corrective signal
    IMPULSE_THRESHOLD = 0.5  # Minimum prob for impulse context
    LATE_IMPULSE_CAUTION = 0.7  # Penalty for late impulse

    def __init__(
        self,
        require_wave_context: bool = False,
        block_late_impulse: bool = False,
    ):
        """
        Initialize wave filter.

        Args:
            require_wave_context: If True, require wave features for signal
            block_late_impulse: If True, hard block on wave 4/5
        """
        self.require_wave_context = require_wave_context
        self.block_late_impulse = block_late_impulse

    def check_long(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> FilterResult:
        """
        Check if wave context supports long trade.

        Args:
            df: DataFrame with wave features
            idx: Bar index

        Returns:
            FilterResult
        """
        # Check if wave features available
        if "wave_role" not in df.columns or "prob_corr_down" not in df.columns:
            if self.require_wave_context:
                return FilterResult(passed=False, reason="No wave context", score=0.0)
            return FilterResult(passed=True, reason="No wave data", score=0.5)

        current = df.iloc[idx]

        wave_role = str(current.get("wave_role", "none"))
        wave_stage = int(current.get("wave_stage", -1))
        wave_conf = float(current.get("wave_conf", 0.0))
        prob_corr_down = float(current.get("prob_corr_down", 0.0))
        prob_impulse_up = float(current.get("prob_impulse_up", 0.0))

        # Check for late impulse (caution zone)
        is_late_impulse = wave_role in ["impulse_up_terminal"] or wave_stage in [4, 5]

        if is_late_impulse and self.block_late_impulse:
            return FilterResult(
                passed=False,
                reason=f"Late impulse (stage {wave_stage})",
                score=0.0,
            )

        # Ideal: corrective down within larger uptrend
        if wave_role == "corr_down" and wave_conf > 0.4:
            score = 0.8 + (wave_conf * 0.2)
            return FilterResult(
                passed=True,
                reason=f"Wave corr_down (conf={wave_conf:.2f})",
                score=min(score, 1.0),
            )

        # Good: high probability of corrective down
        if prob_corr_down >= self.CORR_THRESHOLD:
            return FilterResult(
                passed=True,
                reason=f"Prob corr_down={prob_corr_down:.2f}",
                score=0.7 + (prob_corr_down - 0.6) * 0.5,
            )

        # Acceptable: wave 2/4 in impulse up structure
        if wave_stage in [2, 4] and prob_impulse_up >= self.IMPULSE_THRESHOLD:
            return FilterResult(
                passed=True,
                reason=f"Impulse wave {wave_stage}",
                score=0.65,
            )

        # Caution: late impulse
        if is_late_impulse:
            return FilterResult(
                passed=True,
                reason=f"Late impulse (stage {wave_stage})",
                score=self.LATE_IMPULSE_CAUTION * 0.5,
            )

        # Default: no clear wave context
        return FilterResult(passed=True, reason="No clear wave signal", score=0.5)

    def check_short(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> FilterResult:
        """Check if wave context supports short trade."""
        if "wave_role" not in df.columns or "prob_corr_up" not in df.columns:
            if self.require_wave_context:
                return FilterResult(passed=False, reason="No wave context", score=0.0)
            return FilterResult(passed=True, reason="No wave data", score=0.5)

        current = df.iloc[idx]

        wave_role = str(current.get("wave_role", "none"))
        wave_stage = int(current.get("wave_stage", -1))
        wave_conf = float(current.get("wave_conf", 0.0))
        prob_corr_up = float(current.get("prob_corr_up", 0.0))
        prob_impulse_down = float(current.get("prob_impulse_down", 0.0))

        is_late_impulse = wave_role in ["impulse_down_terminal"] or wave_stage in [4, 5]

        if is_late_impulse and self.block_late_impulse:
            return FilterResult(
                passed=False,
                reason=f"Late impulse (stage {wave_stage})",
                score=0.0,
            )

        if wave_role == "corr_up" and wave_conf > 0.4:
            score = 0.8 + (wave_conf * 0.2)
            return FilterResult(
                passed=True,
                reason=f"Wave corr_up (conf={wave_conf:.2f})",
                score=min(score, 1.0),
            )

        if prob_corr_up >= self.CORR_THRESHOLD:
            return FilterResult(
                passed=True,
                reason=f"Prob corr_up={prob_corr_up:.2f}",
                score=0.7 + (prob_corr_up - 0.6) * 0.5,
            )

        if wave_stage in [2, 4] and prob_impulse_down >= self.IMPULSE_THRESHOLD:
            return FilterResult(
                passed=True,
                reason=f"Impulse wave {wave_stage}",
                score=0.65,
            )

        if is_late_impulse:
            return FilterResult(
                passed=True,
                reason=f"Late impulse (stage {wave_stage})",
                score=self.LATE_IMPULSE_CAUTION * 0.5,
            )

        return FilterResult(passed=True, reason="No clear wave signal", score=0.5)


class CombinedFilter:
    """
    Combines multiple filters with configurable weights.

    Now includes optional wave filter for EW-style context.
    """

    def __init__(
        self,
        rrg_filter: Optional[RRGFilter] = None,
        swing_filter: Optional[SwingFilter] = None,
        wave_filter: Optional[WaveFilter] = None,
        min_combined_score: float = 0.4,
    ):
        """
        Initialize combined filter.

        Args:
            rrg_filter: RRG filter instance
            swing_filter: Swing filter instance
            wave_filter: Wave filter instance (optional)
            min_combined_score: Minimum score to pass
        """
        self.rrg_filter = rrg_filter or RRGFilter()
        self.swing_filter = swing_filter or SwingFilter()
        self.wave_filter = wave_filter  # Optional
        self.min_combined_score = min_combined_score

        # Weights for combining scores (adjusted when wave filter active)
        if wave_filter is not None:
            self.weights = {
                "rrg": 0.3,
                "swing": 0.4,
                "wave": 0.3,
            }
        else:
            self.weights = {
                "rrg": 0.4,
                "swing": 0.6,
            }

    def check(
        self,
        df: pd.DataFrame,
        direction: str,
        idx: int = -1,
    ) -> FilterResult:
        """
        Run all filters and combine results.

        Args:
            df: DataFrame with features
            direction: "LONG" or "SHORT"
            idx: Bar index

        Returns:
            Combined FilterResult
        """
        results = {}

        # RRG filter
        if direction == "LONG":
            results["rrg"] = self.rrg_filter.check_long(df, idx)
        else:
            results["rrg"] = self.rrg_filter.check_short(df, idx)

        # Swing filter
        if direction == "LONG":
            results["swing"] = self.swing_filter.check_long(df, idx)
        else:
            results["swing"] = self.swing_filter.check_short(df, idx)

        # Wave filter (if enabled)
        if self.wave_filter is not None:
            if direction == "LONG":
                results["wave"] = self.wave_filter.check_long(df, idx)
            else:
                results["wave"] = self.wave_filter.check_short(df, idx)

        # Check for hard blocks
        for name, result in results.items():
            if not result.passed and result.score == 0.0:
                return FilterResult(
                    passed=False,
                    reason=f"Blocked by {name}: {result.reason}",
                    score=0.0,
                )

        # Calculate combined score
        combined_score = sum(
            self.weights.get(name, 0) * result.score for name, result in results.items()
        )

        # Check minimum score
        passed = combined_score >= self.min_combined_score

        reasons = [f"{name}={result.score:.2f}" for name, result in results.items()]

        return FilterResult(
            passed=passed,
            reason=f"Combined: {', '.join(reasons)}",
            score=combined_score,
        )

    def get_wave_bonus(
        self,
        df: pd.DataFrame,
        direction: str,
        idx: int = -1,
    ) -> float:
        """
        Get wave context bonus for signal confidence.

        Use this to boost confidence when wave context is favorable.

        Args:
            df: DataFrame with wave features
            direction: "LONG" or "SHORT"
            idx: Bar index

        Returns:
            Bonus value 0.0 to 0.15
        """
        if self.wave_filter is None:
            return 0.0

        if direction == "LONG":
            result = self.wave_filter.check_long(df, idx)
        else:
            result = self.wave_filter.check_short(df, idx)

        # High-quality wave context gives bonus
        if result.score > 0.7 and result.passed:
            return 0.1 + (result.score - 0.7) * 0.5  # 0.1 to 0.15

        return 0.0
