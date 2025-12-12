# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Timeframe (MTF) Context Builder.

Provides hierarchical market context across timeframes:
    Weekly (Macro Regime) → Daily (Intermediate Trend) → 4H (Swing Context) → 1H (Execution)

Each higher timeframe provides context for the lower timeframe's decision making.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Import from quantcore hierarchy modules
try:
    from quantcore.hierarchy.regime_classifier import (
        RegimeType,
        RegimeContext,
        WeeklyRegimeClassifier,
    )
    from quantcore.hierarchy.trend_filter import (
        TrendDirection,
        TrendContext,
        DailyTrendFilter,
    )
    from quantcore.hierarchy.swing_context import (
        SwingPhase,
        SwingContext,
        SwingContextAnalyzer,
    )
    from quantcore.hierarchy.alignment import HierarchicalAlignment, AlignmentResult
    from quantcore.config.timeframes import (
        Timeframe,
        TIMEFRAME_HIERARCHY,
        TIMEFRAME_PARAMS,
    )

    QUANTCORE_AVAILABLE = True
except ImportError:
    QUANTCORE_AVAILABLE = False

    # Define fallback enums
    class RegimeType(Enum):
        BULL = "BULL"
        BEAR = "BEAR"
        SIDEWAYS = "SIDEWAYS"

    class TrendDirection(Enum):
        UP = "UP"
        DOWN = "DOWN"
        NEUTRAL = "NEUTRAL"

    class SwingPhase(Enum):
        IMPULSE_UP = "IMPULSE_UP"
        CORRECTION_DOWN = "CORRECTION_DOWN"
        IMPULSE_DOWN = "IMPULSE_DOWN"
        CORRECTION_UP = "CORRECTION_UP"
        CONSOLIDATION = "CONSOLIDATION"


# =============================================================================
# MTF CONTEXT DATACLASS
# =============================================================================


@dataclass
class MTFContext:
    """
    Multi-timeframe market context for trading decisions.

    Provides a complete hierarchical view of market conditions:
    - Weekly: Macro regime (BULL/BEAR/SIDEWAYS)
    - Daily: Intermediate trend direction and strength
    - 4H: Swing phase and optimal entry timing
    - 1H: Execution-level setup and triggers

    The alignment_score indicates how well all timeframes agree on direction.
    Higher scores = higher confidence trades.
    """

    # Symbol this context is for
    symbol: str

    # Timestamp of this context
    timestamp: datetime

    # Weekly macro regime
    weekly_regime: RegimeType = RegimeType.SIDEWAYS
    weekly_confidence: float = 0.5
    weekly_bars_in_regime: int = 0

    # Daily trend
    daily_trend: TrendDirection = TrendDirection.NEUTRAL
    daily_strength: float = 0.5
    daily_momentum: float = 0.0
    daily_price_vs_ema: float = 0.0  # % distance from 50 EMA

    # 4H swing context
    h4_swing_phase: SwingPhase = SwingPhase.CONSOLIDATION
    h4_near_swing_low: bool = False
    h4_near_swing_high: bool = False
    h4_swing_strength: float = 0.5
    h4_correction_depth: float = 0.0  # % retracement
    h4_trend_exhaustion: bool = False

    # 1H execution setup
    h1_setup: str = "none"  # "long_trigger", "short_trigger", "waiting", "none"
    h1_rsi: float = 50.0
    h1_momentum: float = 0.0
    h1_volume_surge: bool = False

    # Cross-timeframe alignment
    alignment_score: float = 0.5  # 0-1, how well TFs agree
    alignment_direction: Literal["LONG", "SHORT", "NEUTRAL"] = "NEUTRAL"
    alignment_factors: Dict[str, float] = field(default_factory=dict)

    # Trading implications
    trade_bias: Literal["long", "short", "neutral"] = "neutral"
    suggested_holding_period: Literal["intraday", "swing", "position"] = "swing"
    position_scale: float = 1.0  # 0-1 scaling based on alignment

    # Rejection reasons if any
    rejection_reasons: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived fields."""
        if not self.alignment_factors:
            self.alignment_factors = {}
        if not self.rejection_reasons:
            self.rejection_reasons = []

    @property
    def allows_long(self) -> bool:
        """Check if MTF context supports long trades."""
        # Weekly regime must allow longs
        if self.weekly_regime == RegimeType.BEAR and self.weekly_confidence > 0.7:
            return False
        # Daily trend shouldn't be strong down
        if self.daily_trend == TrendDirection.DOWN and self.daily_strength > 0.7:
            return False
        return True

    @property
    def allows_short(self) -> bool:
        """Check if MTF context supports short trades."""
        # Weekly regime must allow shorts
        if self.weekly_regime == RegimeType.BULL and self.weekly_confidence > 0.7:
            return False
        # Daily trend shouldn't be strong up
        if self.daily_trend == TrendDirection.UP and self.daily_strength > 0.7:
            return False
        return True

    @property
    def is_high_conviction(self) -> bool:
        """Check if this is a high-conviction setup (strong alignment)."""
        return self.alignment_score >= 0.7

    @property
    def optimal_for_trend_following(self) -> bool:
        """Check if context favors trend-following strategies."""
        return (
            self.weekly_regime in [RegimeType.BULL, RegimeType.BEAR]
            and self.daily_strength > 0.5
            and self.h4_swing_phase in [SwingPhase.IMPULSE_UP, SwingPhase.IMPULSE_DOWN]
        )

    @property
    def optimal_for_mean_reversion(self) -> bool:
        """Check if context favors mean-reversion strategies."""
        return (
            self.weekly_regime == RegimeType.SIDEWAYS
            or self.h4_swing_phase
            in [SwingPhase.CORRECTION_DOWN, SwingPhase.CORRECTION_UP]
            or (self.h4_near_swing_low or self.h4_near_swing_high)
        )

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get suggested strategy weights based on MTF context.

        Returns dict of strategy -> weight (0-1).
        """
        weights = {
            "TrendFollower": 0.2,
            "MeanReversion": 0.2,
            "MomentumTrader": 0.2,
            "BreakoutTrader": 0.2,
            "VolatilityTrader": 0.2,
        }

        # Adjust based on regime
        if self.weekly_regime == RegimeType.BULL:
            weights["TrendFollower"] = 0.35
            weights["MomentumTrader"] = 0.25
            weights["MeanReversion"] = 0.15
        elif self.weekly_regime == RegimeType.BEAR:
            weights["TrendFollower"] = 0.30
            weights["VolatilityTrader"] = 0.25
            weights["MeanReversion"] = 0.20
        else:  # SIDEWAYS
            weights["MeanReversion"] = 0.35
            weights["BreakoutTrader"] = 0.25
            weights["TrendFollower"] = 0.10

        # Adjust for swing phase
        if self.h4_swing_phase in [
            SwingPhase.CORRECTION_DOWN,
            SwingPhase.CORRECTION_UP,
        ]:
            weights["MeanReversion"] = min(0.40, weights["MeanReversion"] + 0.10)

        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def to_prompt_context(self) -> str:
        """
        Format MTF context for LLM agent prompts.

        Returns a clear, structured summary for agent decision-making.
        """
        return f"""
═══════════════════════════════════════════════════════════════════════════════
                    MULTI-TIMEFRAME CONTEXT: {self.symbol}
═══════════════════════════════════════════════════════════════════════════════

WEEKLY (Macro Regime):
  • Regime: {self.weekly_regime.value}
  • Confidence: {self.weekly_confidence:.0%}
  • Duration: {self.weekly_bars_in_regime} weeks in current regime

DAILY (Intermediate Trend):
  • Direction: {self.daily_trend.value}
  • Strength: {self.daily_strength:.0%}
  • Momentum: {self.daily_momentum:+.2f}
  • Price vs EMA: {self.daily_price_vs_ema:+.1%}

4H (Swing Context):
  • Phase: {self.h4_swing_phase.value}
  • Near Swing Low: {'YES' if self.h4_near_swing_low else 'NO'}
  • Near Swing High: {'YES' if self.h4_near_swing_high else 'NO'}
  • Correction Depth: {self.h4_correction_depth:.1%}
  • Exhaustion Warning: {'YES' if self.h4_trend_exhaustion else 'NO'}

1H (Execution):
  • Setup: {self.h1_setup}
  • RSI: {self.h1_rsi:.1f}
  • Volume Surge: {'YES' if self.h1_volume_surge else 'NO'}

═══════════════════════════════════════════════════════════════════════════════
                    CROSS-TIMEFRAME ALIGNMENT
═══════════════════════════════════════════════════════════════════════════════

  • Alignment Score: {self.alignment_score:.0%}
  • Direction Bias: {self.alignment_direction}
  • Trade Bias: {self.trade_bias.upper()}
  • Suggested Holding: {self.suggested_holding_period.upper()}
  • Position Scale: {self.position_scale:.0%}
  • Allows Long: {'YES' if self.allows_long else 'NO'}
  • Allows Short: {'YES' if self.allows_short else 'NO'}
  • High Conviction: {'YES' if self.is_high_conviction else 'NO'}

{self._format_rejection_reasons()}
═══════════════════════════════════════════════════════════════════════════════
"""

    def _format_rejection_reasons(self) -> str:
        """Format any rejection reasons."""
        if not self.rejection_reasons:
            return ""
        return "CAUTION:\n" + "\n".join(f"  ⚠ {r}" for r in self.rejection_reasons)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "weekly_regime": self.weekly_regime.value,
            "weekly_confidence": self.weekly_confidence,
            "daily_trend": self.daily_trend.value,
            "daily_strength": self.daily_strength,
            "h4_swing_phase": self.h4_swing_phase.value,
            "h4_near_swing_low": self.h4_near_swing_low,
            "h4_near_swing_high": self.h4_near_swing_high,
            "h1_setup": self.h1_setup,
            "alignment_score": self.alignment_score,
            "alignment_direction": self.alignment_direction,
            "trade_bias": self.trade_bias,
            "suggested_holding_period": self.suggested_holding_period,
            "allows_long": self.allows_long,
            "allows_short": self.allows_short,
        }


# =============================================================================
# MTF CONTEXT BUILDER
# =============================================================================


class MTFContextBuilder:
    """
    Builder for constructing MTFContext from multi-timeframe data.

    Uses quantcore hierarchy modules when available, otherwise
    computes basic context from raw price data.
    """

    def __init__(self):
        """Initialize builder with hierarchy analyzers."""
        self._regime_classifier = None
        self._trend_filter = None
        self._swing_analyzer = None
        self._alignment_checker = None

        if QUANTCORE_AVAILABLE:
            try:
                self._regime_classifier = WeeklyRegimeClassifier()
                self._trend_filter = DailyTrendFilter()
                self._swing_analyzer = SwingContextAnalyzer()
                self._alignment_checker = HierarchicalAlignment()
                logger.info(
                    "MTFContextBuilder initialized with quantcore hierarchy modules"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize quantcore modules: {e}")
        else:
            logger.warning(
                "quantcore hierarchy modules not available, using basic analysis"
            )

    def build(
        self,
        symbol: str,
        mtf_data: Dict[str, pd.DataFrame],
        timestamp: Optional[datetime] = None,
    ) -> MTFContext:
        """
        Build MTFContext from multi-timeframe data.

        Args:
            symbol: The symbol to build context for
            mtf_data: Dict with keys "1H", "4H", "D1", "W1" containing DataFrames
            timestamp: Timestamp for this context (default: now)

        Returns:
            MTFContext with all timeframe analysis
        """
        timestamp = timestamp or datetime.now()

        ctx = MTFContext(symbol=symbol, timestamp=timestamp)

        try:
            # Analyze each timeframe
            self._analyze_weekly(ctx, mtf_data.get("W1"))
            self._analyze_daily(ctx, mtf_data.get("D1"))
            self._analyze_4h(ctx, mtf_data.get("4H"))
            self._analyze_1h(ctx, mtf_data.get("1H"))

            # Compute cross-timeframe alignment
            self._compute_alignment(ctx)

            # Determine trade bias and holding period
            self._compute_trade_implications(ctx)

        except Exception as e:
            logger.error(f"Error building MTF context for {symbol}: {e}")
            ctx.rejection_reasons.append(f"Context build error: {str(e)}")

        return ctx

    def _analyze_weekly(self, ctx: MTFContext, df: Optional[pd.DataFrame]) -> None:
        """Analyze weekly timeframe for macro regime."""
        if df is None or df.empty:
            return

        if self._regime_classifier:
            try:
                regime_ctx = self._regime_classifier.classify(df)
                ctx.weekly_regime = regime_ctx.regime
                ctx.weekly_confidence = regime_ctx.confidence
                ctx.weekly_bars_in_regime = regime_ctx.bars_in_regime
                return
            except Exception as e:
                logger.warning(f"Regime classifier failed: {e}")

        # Fallback: Basic regime detection from price vs MA
        self._basic_regime_analysis(ctx, df)

    def _basic_regime_analysis(self, ctx: MTFContext, df: pd.DataFrame) -> None:
        """Basic regime analysis when quantcore not available."""
        if len(df) < 20:
            return

        # Compute basic indicators
        close = df["close"].values if "close" in df.columns else df["Close"].values

        # 10-week and 20-week SMAs
        sma10 = np.mean(close[-10:])
        sma20 = np.mean(close[-20:])
        current_price = close[-1]

        # Determine regime
        if current_price > sma10 > sma20:
            ctx.weekly_regime = RegimeType.BULL
            ctx.weekly_confidence = 0.7
        elif current_price < sma10 < sma20:
            ctx.weekly_regime = RegimeType.BEAR
            ctx.weekly_confidence = 0.7
        else:
            ctx.weekly_regime = RegimeType.SIDEWAYS
            ctx.weekly_confidence = 0.5

    def _analyze_daily(self, ctx: MTFContext, df: Optional[pd.DataFrame]) -> None:
        """Analyze daily timeframe for intermediate trend."""
        if df is None or df.empty:
            return

        if self._trend_filter:
            try:
                trend_ctx = self._trend_filter.analyze(df)
                ctx.daily_trend = trend_ctx.direction
                ctx.daily_strength = trend_ctx.strength
                ctx.daily_momentum = trend_ctx.momentum
                ctx.daily_price_vs_ema = trend_ctx.price_vs_ema
                return
            except Exception as e:
                logger.warning(f"Trend filter failed: {e}")

        # Fallback: Basic trend analysis
        self._basic_trend_analysis(ctx, df)

    def _basic_trend_analysis(self, ctx: MTFContext, df: pd.DataFrame) -> None:
        """Basic trend analysis when quantcore not available."""
        if len(df) < 50:
            return

        close = df["close"].values if "close" in df.columns else df["Close"].values

        # 20 and 50 day EMAs
        ema20 = self._compute_ema(close, 20)
        ema50 = self._compute_ema(close, 50)
        current_price = close[-1]

        # Price vs EMA
        ctx.daily_price_vs_ema = (current_price - ema50) / ema50

        # Momentum (10-day ROC)
        if len(close) >= 10:
            ctx.daily_momentum = close[-1] / close[-10] - 1

        # Determine trend
        if current_price > ema20 > ema50:
            ctx.daily_trend = TrendDirection.UP
            ctx.daily_strength = min(1.0, abs(ctx.daily_price_vs_ema) * 5)
        elif current_price < ema20 < ema50:
            ctx.daily_trend = TrendDirection.DOWN
            ctx.daily_strength = min(1.0, abs(ctx.daily_price_vs_ema) * 5)
        else:
            ctx.daily_trend = TrendDirection.NEUTRAL
            ctx.daily_strength = 0.3

    def _analyze_4h(self, ctx: MTFContext, df: Optional[pd.DataFrame]) -> None:
        """Analyze 4H timeframe for swing context."""
        if df is None or df.empty:
            return

        if self._swing_analyzer:
            try:
                swing_ctx = self._swing_analyzer.analyze(df)
                ctx.h4_swing_phase = swing_ctx.phase
                ctx.h4_near_swing_low = swing_ctx.near_swing_low
                ctx.h4_near_swing_high = swing_ctx.near_swing_high
                ctx.h4_swing_strength = swing_ctx.swing_strength
                ctx.h4_correction_depth = swing_ctx.correction_depth
                ctx.h4_trend_exhaustion = swing_ctx.trend_exhaustion
                return
            except Exception as e:
                logger.warning(f"Swing analyzer failed: {e}")

        # Fallback: Basic swing analysis
        self._basic_swing_analysis(ctx, df)

    def _basic_swing_analysis(self, ctx: MTFContext, df: pd.DataFrame) -> None:
        """Basic swing analysis when quantcore not available."""
        if len(df) < 20:
            return

        close = df["close"].values if "close" in df.columns else df["Close"].values
        high = df["high"].values if "high" in df.columns else df["High"].values
        low = df["low"].values if "low" in df.columns else df["Low"].values

        # Recent swing high/low (last 20 bars)
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        current = close[-1]

        # Distance to swing points
        range_size = recent_high - recent_low
        if range_size > 0:
            dist_to_low = (current - recent_low) / range_size
            dist_to_high = (recent_high - current) / range_size

            ctx.h4_near_swing_low = dist_to_low < 0.2
            ctx.h4_near_swing_high = dist_to_high < 0.2
            ctx.h4_correction_depth = 1 - dist_to_low  # How much retraced

        # Determine phase (simplified)
        mom = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0

        if mom > 0.01 and ctx.daily_trend == TrendDirection.UP:
            ctx.h4_swing_phase = SwingPhase.IMPULSE_UP
        elif mom < -0.01 and ctx.daily_trend == TrendDirection.DOWN:
            ctx.h4_swing_phase = SwingPhase.IMPULSE_DOWN
        elif mom < 0 and ctx.daily_trend == TrendDirection.UP:
            ctx.h4_swing_phase = SwingPhase.CORRECTION_DOWN
        elif mom > 0 and ctx.daily_trend == TrendDirection.DOWN:
            ctx.h4_swing_phase = SwingPhase.CORRECTION_UP
        else:
            ctx.h4_swing_phase = SwingPhase.CONSOLIDATION

    def _analyze_1h(self, ctx: MTFContext, df: Optional[pd.DataFrame]) -> None:
        """Analyze 1H timeframe for execution setup."""
        if df is None or df.empty:
            return

        close = df["close"].values if "close" in df.columns else df["Close"].values
        volume = (
            df["volume"].values
            if "volume" in df.columns
            else df.get("Volume", pd.Series([0] * len(df))).values
        )

        # Compute RSI
        ctx.h1_rsi = self._compute_rsi(close, 14)

        # Momentum
        if len(close) >= 5:
            ctx.h1_momentum = close[-1] / close[-5] - 1

        # Volume surge (volume > 2x average)
        if len(volume) >= 20:
            avg_vol = np.mean(volume[-20:])
            ctx.h1_volume_surge = volume[-1] > 2 * avg_vol if avg_vol > 0 else False

        # Determine setup
        ctx.h1_setup = self._determine_1h_setup(ctx)

    def _determine_1h_setup(self, ctx: MTFContext) -> str:
        """Determine 1H execution setup based on all context."""
        # Long trigger conditions
        if ctx.allows_long and ctx.h1_rsi < 35 and ctx.h4_near_swing_low:
            return "long_trigger"

        # Short trigger conditions
        if ctx.allows_short and ctx.h1_rsi > 65 and ctx.h4_near_swing_high:
            return "short_trigger"

        # Waiting for setup
        if ctx.h4_swing_phase in [SwingPhase.CORRECTION_DOWN, SwingPhase.CORRECTION_UP]:
            return "waiting"

        return "none"

    def _compute_alignment(self, ctx: MTFContext) -> None:
        """Compute cross-timeframe alignment score."""
        scores = []
        factors = {}

        # Weekly-Daily alignment
        if (
            ctx.weekly_regime == RegimeType.BULL
            and ctx.daily_trend == TrendDirection.UP
        ):
            factors["weekly_daily"] = 1.0
        elif (
            ctx.weekly_regime == RegimeType.BEAR
            and ctx.daily_trend == TrendDirection.DOWN
        ):
            factors["weekly_daily"] = 1.0
        elif ctx.weekly_regime == RegimeType.SIDEWAYS:
            factors["weekly_daily"] = 0.5
        else:
            factors["weekly_daily"] = 0.3
        scores.append(factors["weekly_daily"])

        # Daily-4H alignment
        if ctx.daily_trend == TrendDirection.UP and ctx.h4_swing_phase in [
            SwingPhase.IMPULSE_UP,
            SwingPhase.CORRECTION_DOWN,
        ]:
            factors["daily_4h"] = 0.8
        elif ctx.daily_trend == TrendDirection.DOWN and ctx.h4_swing_phase in [
            SwingPhase.IMPULSE_DOWN,
            SwingPhase.CORRECTION_UP,
        ]:
            factors["daily_4h"] = 0.8
        elif ctx.h4_swing_phase == SwingPhase.CONSOLIDATION:
            factors["daily_4h"] = 0.4
        else:
            factors["daily_4h"] = 0.5
        scores.append(factors["daily_4h"])

        # 4H-1H alignment
        if (ctx.h4_near_swing_low and ctx.h1_rsi < 40) or (
            ctx.h4_near_swing_high and ctx.h1_rsi > 60
        ):
            factors["4h_1h"] = 0.9
        elif ctx.h1_setup in ["long_trigger", "short_trigger"]:
            factors["4h_1h"] = 0.8
        else:
            factors["4h_1h"] = 0.5
        scores.append(factors["4h_1h"])

        # Overall alignment
        ctx.alignment_score = np.mean(scores)
        ctx.alignment_factors = factors

        # Determine alignment direction
        if (
            ctx.weekly_regime == RegimeType.BULL
            and ctx.daily_trend == TrendDirection.UP
        ):
            ctx.alignment_direction = "LONG"
        elif (
            ctx.weekly_regime == RegimeType.BEAR
            and ctx.daily_trend == TrendDirection.DOWN
        ):
            ctx.alignment_direction = "SHORT"
        else:
            ctx.alignment_direction = "NEUTRAL"

    def _compute_trade_implications(self, ctx: MTFContext) -> None:
        """Compute trading implications from MTF context."""
        # Trade bias
        if ctx.alignment_direction == "LONG" and ctx.allows_long:
            ctx.trade_bias = "long"
        elif ctx.alignment_direction == "SHORT" and ctx.allows_short:
            ctx.trade_bias = "short"
        else:
            ctx.trade_bias = "neutral"

        # Position scale based on alignment
        ctx.position_scale = ctx.alignment_score

        # Suggested holding period
        if ctx.weekly_confidence > 0.7 and ctx.alignment_score > 0.8:
            ctx.suggested_holding_period = "position"  # Weeks
        elif ctx.daily_strength > 0.5 and ctx.alignment_score > 0.6:
            ctx.suggested_holding_period = "swing"  # Days
        else:
            ctx.suggested_holding_period = "intraday"  # Hours

        # Add rejection reasons
        if not ctx.allows_long and not ctx.allows_short:
            ctx.rejection_reasons.append(
                "Neither long nor short allowed by MTF context"
            )
        if ctx.h4_trend_exhaustion:
            ctx.rejection_reasons.append("4H trend showing exhaustion signs")
        if ctx.alignment_score < 0.4:
            ctx.rejection_reasons.append("Low cross-timeframe alignment")

    @staticmethod
    def _compute_ema(data: np.ndarray, period: int) -> float:
        """Compute EMA of the last value."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0

        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _compute_rsi(data: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if len(data) < period + 1:
            return 50.0

        deltas = np.diff(data[-(period + 1) :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# =============================================================================
# SINGLETON BUILDER
# =============================================================================

_builder_instance: Optional[MTFContextBuilder] = None


def get_mtf_context_builder() -> MTFContextBuilder:
    """Get singleton MTFContextBuilder instance."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = MTFContextBuilder()
    return _builder_instance


def build_mtf_context(
    symbol: str,
    mtf_data: Dict[str, pd.DataFrame],
    timestamp: Optional[datetime] = None,
) -> MTFContext:
    """
    Convenience function to build MTF context.

    Args:
        symbol: Symbol to analyze
        mtf_data: Dict with "1H", "4H", "D1", "W1" DataFrames
        timestamp: Optional timestamp

    Returns:
        MTFContext instance
    """
    builder = get_mtf_context_builder()
    return builder.build(symbol, mtf_data, timestamp)
