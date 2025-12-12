"""
Hierarchical signal cascade for multi-timeframe signal generation.

Implements the top-down filtering: Weekly → Daily → 4H → 1H
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Literal, List
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS
from quantcore.hierarchy.regime_classifier import WeeklyRegimeClassifier, RegimeContext
from quantcore.hierarchy.trend_filter import DailyTrendFilter, TrendContext
from quantcore.hierarchy.swing_context import SwingContextAnalyzer, SwingContext
from quantcore.hierarchy.alignment import HierarchicalAlignment, AlignmentResult


@dataclass
class Signal:
    """Trading signal with full context."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float  # ML probability
    alignment_score: float  # Cross-TF alignment (0-1)

    # Timeframe breakdown
    tf_breakdown: Dict[str, str] = field(default_factory=dict)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    timeframe: Timeframe = Timeframe.H1
    atr: float = 0.0

    # Risk metrics
    risk_reward_ratio: float = 0.0
    position_size_multiplier: float = 1.0

    def __post_init__(self):
        if self.tf_breakdown is None:
            self.tf_breakdown = {}

        # Calculate risk/reward
        if self.direction == "LONG":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit

        if risk > 0:
            self.risk_reward_ratio = reward / risk

        # Position size multiplier based on alignment
        self.position_size_multiplier = 0.5 + 0.5 * self.alignment_score


@dataclass
class CascadeResult:
    """Result of cascade evaluation."""

    passed: bool
    signal: Optional[Signal] = None
    weekly_ctx: Optional[RegimeContext] = None
    daily_ctx: Optional[TrendContext] = None
    h4_ctx: Optional[SwingContext] = None
    alignment: Optional[AlignmentResult] = None
    rejection_stage: Optional[str] = None
    rejection_reason: Optional[str] = None


class SignalCascade:
    """
    Hierarchical signal cascade for generating filtered signals.

    Flow:
    1. Check Weekly regime → filter if unfavorable
    2. Check Daily trend → filter if against trade direction
    3. Check 4H swing context → filter if not near swing
    4. Check 1H rules → generate signal if all pass
    """

    def __init__(
        self,
        probability_threshold: float = 0.6,
        min_alignment_score: float = 0.5,
    ):
        """
        Initialize signal cascade.

        Args:
            probability_threshold: Minimum ML probability to generate signal
            min_alignment_score: Minimum alignment score required
        """
        self.probability_threshold = probability_threshold
        self.min_alignment_score = min_alignment_score

        # Initialize analyzers
        self.regime_classifier = WeeklyRegimeClassifier()
        self.trend_filter = DailyTrendFilter()
        self.swing_analyzer = SwingContextAnalyzer()
        self.alignment_checker = HierarchicalAlignment()

    def evaluate(
        self,
        symbol: str,
        data: Dict[Timeframe, pd.DataFrame],
        direction: Literal["LONG", "SHORT"],
        ml_probability: Optional[float] = None,
    ) -> CascadeResult:
        """
        Evaluate cascade for a potential signal.

        Args:
            symbol: Stock symbol
            data: Multi-timeframe data with features
            direction: Trade direction to evaluate
            ml_probability: Optional ML model probability

        Returns:
            CascadeResult with signal if all filters pass
        """
        # Step 1: Weekly regime
        weekly_ctx = None
        if Timeframe.W1 in data and not data[Timeframe.W1].empty:
            weekly_ctx = self.regime_classifier.classify(data[Timeframe.W1])

            # Check regime allows direction
            if direction == "LONG" and not weekly_ctx.allows_long():
                return CascadeResult(
                    passed=False,
                    weekly_ctx=weekly_ctx,
                    rejection_stage="weekly",
                    rejection_reason=f"Weekly regime {weekly_ctx.regime.value} blocks long trades",
                )
            if direction == "SHORT" and not weekly_ctx.allows_short():
                return CascadeResult(
                    passed=False,
                    weekly_ctx=weekly_ctx,
                    rejection_stage="weekly",
                    rejection_reason=f"Weekly regime {weekly_ctx.regime.value} blocks short trades",
                )

        # Step 2: Daily trend
        daily_ctx = None
        if Timeframe.D1 in data and not data[Timeframe.D1].empty:
            daily_ctx = self.trend_filter.analyze(data[Timeframe.D1])

            if direction == "LONG" and not daily_ctx.allows_long():
                return CascadeResult(
                    passed=False,
                    weekly_ctx=weekly_ctx,
                    daily_ctx=daily_ctx,
                    rejection_stage="daily",
                    rejection_reason="Strong daily downtrend blocks long trades",
                )
            if direction == "SHORT" and not daily_ctx.allows_short():
                return CascadeResult(
                    passed=False,
                    weekly_ctx=weekly_ctx,
                    daily_ctx=daily_ctx,
                    rejection_stage="daily",
                    rejection_reason="Strong daily uptrend blocks short trades",
                )

        # Step 3: 4H swing context
        h4_ctx = None
        if Timeframe.H4 in data and not data[Timeframe.H4].empty:
            h4_ctx = self.swing_analyzer.analyze(data[Timeframe.H4], daily_ctx)

            if direction == "LONG" and not h4_ctx.optimal_for_long_mr():
                return CascadeResult(
                    passed=False,
                    weekly_ctx=weekly_ctx,
                    daily_ctx=daily_ctx,
                    h4_ctx=h4_ctx,
                    rejection_stage="h4",
                    rejection_reason="4H not optimal for long MR (not near swing low)",
                )
            if direction == "SHORT" and not h4_ctx.optimal_for_short_mr():
                return CascadeResult(
                    passed=False,
                    weekly_ctx=weekly_ctx,
                    daily_ctx=daily_ctx,
                    h4_ctx=h4_ctx,
                    rejection_stage="h4",
                    rejection_reason="4H not optimal for short MR (not near swing high)",
                )

        # Step 4: Check alignment
        if direction == "LONG":
            alignment = self.alignment_checker.check_long_alignment(
                weekly_ctx, daily_ctx, h4_ctx
            )
        else:
            alignment = self.alignment_checker.check_short_alignment(
                weekly_ctx, daily_ctx, h4_ctx
            )

        if alignment.score < self.min_alignment_score:
            return CascadeResult(
                passed=False,
                weekly_ctx=weekly_ctx,
                daily_ctx=daily_ctx,
                h4_ctx=h4_ctx,
                alignment=alignment,
                rejection_stage="alignment",
                rejection_reason=f"Alignment score {alignment.score:.2f} below threshold {self.min_alignment_score}",
            )

        # Step 5: Check ML probability (if provided)
        if ml_probability is not None and ml_probability < self.probability_threshold:
            return CascadeResult(
                passed=False,
                weekly_ctx=weekly_ctx,
                daily_ctx=daily_ctx,
                h4_ctx=h4_ctx,
                alignment=alignment,
                rejection_stage="ml_probability",
                rejection_reason=f"ML probability {ml_probability:.2f} below threshold {self.probability_threshold}",
            )

        # All filters passed - generate signal
        signal = self._create_signal(
            symbol=symbol,
            direction=direction,
            data=data,
            weekly_ctx=weekly_ctx,
            daily_ctx=daily_ctx,
            h4_ctx=h4_ctx,
            alignment=alignment,
            ml_probability=ml_probability,
        )

        return CascadeResult(
            passed=True,
            signal=signal,
            weekly_ctx=weekly_ctx,
            daily_ctx=daily_ctx,
            h4_ctx=h4_ctx,
            alignment=alignment,
        )

    def _create_signal(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        data: Dict[Timeframe, pd.DataFrame],
        weekly_ctx: Optional[RegimeContext],
        daily_ctx: Optional[TrendContext],
        h4_ctx: Optional[SwingContext],
        alignment: AlignmentResult,
        ml_probability: Optional[float],
    ) -> Signal:
        """Create a Signal object from cascade results."""
        # Get 1H data for entry/TP/SL calculation
        h1_data = data.get(Timeframe.H1, pd.DataFrame())

        if h1_data.empty:
            raise ValueError("No 1H data available for signal generation")

        current = h1_data.iloc[-1]
        params = TIMEFRAME_PARAMS[Timeframe.H1]

        entry_price = float(current["close"])
        atr = float(current.get("atr", entry_price * 0.01))  # Default 1% if no ATR

        if direction == "LONG":
            take_profit = entry_price + (params.tp_atr_multiple * atr)
            stop_loss = entry_price - (params.sl_atr_multiple * atr)
        else:
            take_profit = entry_price - (params.tp_atr_multiple * atr)
            stop_loss = entry_price + (params.sl_atr_multiple * atr)

        # Build timeframe breakdown
        tf_breakdown = {}
        if weekly_ctx:
            tf_breakdown["W1"] = weekly_ctx.regime.value
        if daily_ctx:
            tf_breakdown["D1"] = daily_ctx.direction.value
        if h4_ctx:
            tf_breakdown["H4"] = h4_ctx.phase.value
        tf_breakdown["H1"] = "TRIGGERED"

        return Signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            confidence=ml_probability or 0.5,
            alignment_score=alignment.score,
            tf_breakdown=tf_breakdown,
            generated_at=datetime.now(),
            timeframe=Timeframe.H1,
            atr=atr,
        )

    def generate_signals(
        self,
        symbol: str,
        data: Dict[Timeframe, pd.DataFrame],
        check_long: bool = True,
        check_short: bool = True,
        ml_probability_long: Optional[float] = None,
        ml_probability_short: Optional[float] = None,
    ) -> List[Signal]:
        """
        Generate all valid signals for a symbol.

        Args:
            symbol: Stock symbol
            data: Multi-timeframe data
            check_long: Whether to check long signals
            check_short: Whether to check short signals
            ml_probability_long: ML probability for long trades
            ml_probability_short: ML probability for short trades

        Returns:
            List of valid signals (0, 1, or 2)
        """
        signals = []

        if check_long:
            result = self.evaluate(symbol, data, "LONG", ml_probability_long)
            if result.passed and result.signal:
                signals.append(result.signal)

        if check_short:
            result = self.evaluate(symbol, data, "SHORT", ml_probability_short)
            if result.passed and result.signal:
                signals.append(result.signal)

        return signals
