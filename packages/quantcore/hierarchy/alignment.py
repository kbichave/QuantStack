"""
Cross-timeframe alignment checker.

Validates if all timeframes agree and calculates alignment score.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.config.timeframes import Timeframe, TIMEFRAME_HIERARCHY
from quantcore.hierarchy.regime_classifier import RegimeContext, RegimeType
from quantcore.hierarchy.trend_filter import TrendContext, TrendDirection
from quantcore.hierarchy.swing_context import SwingContext, SwingPhase


@dataclass
class AlignmentResult:
    """Result of cross-timeframe alignment check."""

    aligned: bool
    score: float  # 0-1 alignment score
    direction: Literal["LONG", "SHORT", "NEUTRAL"]

    # Individual timeframe contexts
    weekly_context: Optional[RegimeContext] = None
    daily_context: Optional[TrendContext] = None
    h4_context: Optional[SwingContext] = None

    # Breakdown of alignment factors
    factors: Dict[str, float] = field(default_factory=dict)

    # Reasons for misalignment
    rejection_reasons: list = field(default_factory=list)

    def __post_init__(self):
        if self.factors is None:
            self.factors = {}
        if self.rejection_reasons is None:
            self.rejection_reasons = []


class HierarchicalAlignment:
    """
    Checker for multi-timeframe alignment.

    Validates that higher timeframes support the trade direction
    and calculates an overall alignment score.
    """

    # Minimum score thresholds
    MIN_ALIGNMENT_SCORE = 0.5
    HIGH_ALIGNMENT_THRESHOLD = 0.8

    def check_long_alignment(
        self,
        weekly_ctx: Optional[RegimeContext] = None,
        daily_ctx: Optional[TrendContext] = None,
        h4_ctx: Optional[SwingContext] = None,
    ) -> AlignmentResult:
        """
        Check if all timeframes support a long mean-reversion trade.

        Args:
            weekly_ctx: Weekly regime context
            daily_ctx: Daily trend context
            h4_ctx: 4H swing context

        Returns:
            AlignmentResult with alignment details
        """
        factors = {}
        rejection_reasons = []

        # Weekly regime check
        weekly_score = 0.5  # Neutral default
        if weekly_ctx:
            if weekly_ctx.regime == RegimeType.BULL:
                weekly_score = 1.0
            elif weekly_ctx.regime == RegimeType.SIDEWAYS:
                weekly_score = 0.7
            elif weekly_ctx.regime == RegimeType.BEAR:
                weekly_score = 0.2
                rejection_reasons.append("Weekly regime is BEAR")

            # RRG bonus
            if weekly_ctx.rrg_quadrant in ["LEADING", "IMPROVING"]:
                weekly_score = min(weekly_score + 0.1, 1.0)
            elif weekly_ctx.rrg_quadrant == "LAGGING":
                weekly_score = max(weekly_score - 0.2, 0.0)
                rejection_reasons.append("RRG quadrant is LAGGING")

        factors["weekly"] = weekly_score

        # Daily trend check
        daily_score = 0.5
        if daily_ctx:
            if daily_ctx.direction == TrendDirection.UP:
                daily_score = 1.0
            elif daily_ctx.direction == TrendDirection.NEUTRAL:
                daily_score = 0.7
            elif daily_ctx.direction == TrendDirection.DOWN:
                if daily_ctx.strength > 0.7:
                    daily_score = 0.1
                    rejection_reasons.append("Strong daily downtrend")
                else:
                    daily_score = 0.4

        factors["daily"] = daily_score

        # 4H swing check (now wave-aware)
        h4_score = 0.5
        if h4_ctx:
            # Wave-aware scoring (if available)
            if h4_ctx.wave_conf > 0.4:
                # Corrective down within impulse up = ideal for long MR
                if h4_ctx.is_corrective_down and not h4_ctx.is_late_impulse:
                    h4_score = 0.9
                elif h4_ctx.wave_stage in [2, 4] and h4_ctx.prob_impulse_up > 0.5:
                    h4_score = 0.85  # Wave 2/4 pullback in impulse
                elif h4_ctx.is_late_impulse:
                    h4_score = 0.4
                    rejection_reasons.append("4H in late impulse (wave 4/5)")

            # Standard swing scoring (fallback/supplement)
            if h4_ctx.near_swing_low:
                h4_score = max(h4_score, 1.0)
            elif h4_ctx.phase == SwingPhase.CORRECTION_DOWN:
                h4_score = max(h4_score, 0.8)
            elif h4_ctx.phase == SwingPhase.CONSOLIDATION:
                h4_score = max(h4_score, 0.6)
            elif h4_ctx.phase == SwingPhase.IMPULSE_DOWN:
                h4_score = min(h4_score, 0.3)
                if not h4_ctx.trend_exhaustion:
                    rejection_reasons.append("4H in impulse down without exhaustion")

            # Exhaustion bonus
            if h4_ctx.trend_exhaustion:
                h4_score = min(h4_score + 0.2, 1.0)

        factors["h4"] = h4_score

        # Calculate overall score (weighted average)
        weights = {"weekly": 0.35, "daily": 0.35, "h4": 0.30}
        overall_score = sum(factors[k] * weights[k] for k in factors)

        # Determine if aligned
        aligned = (
            overall_score >= self.MIN_ALIGNMENT_SCORE and len(rejection_reasons) == 0
        )

        return AlignmentResult(
            aligned=aligned,
            score=overall_score,
            direction="LONG",
            weekly_context=weekly_ctx,
            daily_context=daily_ctx,
            h4_context=h4_ctx,
            factors=factors,
            rejection_reasons=rejection_reasons,
        )

    def check_short_alignment(
        self,
        weekly_ctx: Optional[RegimeContext] = None,
        daily_ctx: Optional[TrendContext] = None,
        h4_ctx: Optional[SwingContext] = None,
    ) -> AlignmentResult:
        """
        Check if all timeframes support a short mean-reversion trade.

        Args:
            weekly_ctx: Weekly regime context
            daily_ctx: Daily trend context
            h4_ctx: 4H swing context

        Returns:
            AlignmentResult with alignment details
        """
        factors = {}
        rejection_reasons = []

        # Weekly regime check (inverse of long)
        weekly_score = 0.5
        if weekly_ctx:
            if weekly_ctx.regime == RegimeType.BEAR:
                weekly_score = 1.0
            elif weekly_ctx.regime == RegimeType.SIDEWAYS:
                weekly_score = 0.7
            elif weekly_ctx.regime == RegimeType.BULL:
                weekly_score = 0.2
                rejection_reasons.append("Weekly regime is BULL")

            if weekly_ctx.rrg_quadrant in ["LAGGING", "WEAKENING"]:
                weekly_score = min(weekly_score + 0.1, 1.0)
            elif weekly_ctx.rrg_quadrant == "LEADING":
                weekly_score = max(weekly_score - 0.2, 0.0)
                rejection_reasons.append("RRG quadrant is LEADING")

        factors["weekly"] = weekly_score

        # Daily trend check
        daily_score = 0.5
        if daily_ctx:
            if daily_ctx.direction == TrendDirection.DOWN:
                daily_score = 1.0
            elif daily_ctx.direction == TrendDirection.NEUTRAL:
                daily_score = 0.7
            elif daily_ctx.direction == TrendDirection.UP:
                if daily_ctx.strength > 0.7:
                    daily_score = 0.1
                    rejection_reasons.append("Strong daily uptrend")
                else:
                    daily_score = 0.4

        factors["daily"] = daily_score

        # 4H swing check (now wave-aware)
        h4_score = 0.5
        if h4_ctx:
            # Wave-aware scoring (if available)
            if h4_ctx.wave_conf > 0.4:
                # Corrective up within impulse down = ideal for short MR
                if h4_ctx.is_corrective_up and not h4_ctx.is_late_impulse:
                    h4_score = 0.9
                elif h4_ctx.wave_stage in [2, 4] and h4_ctx.prob_impulse_down > 0.5:
                    h4_score = 0.85  # Wave 2/4 bounce in impulse down
                elif h4_ctx.is_late_impulse:
                    h4_score = 0.4
                    rejection_reasons.append("4H in late impulse (wave 4/5)")

            # Standard swing scoring (fallback/supplement)
            if h4_ctx.near_swing_high:
                h4_score = max(h4_score, 1.0)
            elif h4_ctx.phase == SwingPhase.CORRECTION_UP:
                h4_score = max(h4_score, 0.8)
            elif h4_ctx.phase == SwingPhase.CONSOLIDATION:
                h4_score = max(h4_score, 0.6)
            elif h4_ctx.phase == SwingPhase.IMPULSE_UP:
                h4_score = min(h4_score, 0.3)
                if not h4_ctx.trend_exhaustion:
                    rejection_reasons.append("4H in impulse up without exhaustion")

            if h4_ctx.trend_exhaustion:
                h4_score = min(h4_score + 0.2, 1.0)

        factors["h4"] = h4_score

        # Calculate overall score
        weights = {"weekly": 0.35, "daily": 0.35, "h4": 0.30}
        overall_score = sum(factors[k] * weights[k] for k in factors)

        aligned = (
            overall_score >= self.MIN_ALIGNMENT_SCORE and len(rejection_reasons) == 0
        )

        return AlignmentResult(
            aligned=aligned,
            score=overall_score,
            direction="SHORT",
            weekly_context=weekly_ctx,
            daily_context=daily_ctx,
            h4_context=h4_ctx,
            factors=factors,
            rejection_reasons=rejection_reasons,
        )

    def get_best_direction(
        self,
        weekly_ctx: Optional[RegimeContext] = None,
        daily_ctx: Optional[TrendContext] = None,
        h4_ctx: Optional[SwingContext] = None,
    ) -> AlignmentResult:
        """
        Get the best-aligned trade direction.

        Args:
            weekly_ctx: Weekly regime context
            daily_ctx: Daily trend context
            h4_ctx: 4H swing context

        Returns:
            AlignmentResult for the better-aligned direction
        """
        long_result = self.check_long_alignment(weekly_ctx, daily_ctx, h4_ctx)
        short_result = self.check_short_alignment(weekly_ctx, daily_ctx, h4_ctx)

        if long_result.score > short_result.score:
            return long_result
        elif short_result.score > long_result.score:
            return short_result
        else:
            # Neutral
            return AlignmentResult(
                aligned=False,
                score=0.5,
                direction="NEUTRAL",
                weekly_context=weekly_ctx,
                daily_context=daily_ctx,
                h4_context=h4_ctx,
                factors={"neutral": 0.5},
                rejection_reasons=["No clear directional bias"],
            )

    def is_high_alignment(self, result: AlignmentResult) -> bool:
        """Check if alignment score is high."""
        return result.score >= self.HIGH_ALIGNMENT_THRESHOLD
