"""
Ensemble signal combiner for options trading.

Combines signals from multiple technical analysis strategies with configurable weights:
- Mean Reversion (z-score stretch + reversion)
- Momentum (RSI + MACD)
- RRG Quadrant (relative strength)
- Gann Levels (support/resistance proximity)
- Candlestick Patterns (reversal signals)
- Wave Context (Elliott wave stage)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class SignalDirection(Enum):
    """Signal direction."""

    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class SignalResult:
    """Result from a single strategy signal."""

    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    reason: str = ""


@dataclass
class EnsembleSignal:
    """Combined ensemble signal result."""

    direction: SignalDirection
    confidence: float
    component_signals: Dict[str, SignalResult] = field(default_factory=dict)
    weighted_score: float = 0.0
    reason: str = ""


@dataclass
class EnsembleWeights:
    """Configurable weights for each signal source."""

    mean_reversion: float = 0.25
    momentum: float = 0.20
    rrg: float = 0.15
    gann: float = 0.15
    candlestick: float = 0.10
    wave: float = 0.15

    def __post_init__(self):
        """Validate and normalize weights."""
        total = (
            self.mean_reversion
            + self.momentum
            + self.rrg
            + self.gann
            + self.candlestick
            + self.wave
        )
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.mean_reversion /= total
            self.momentum /= total
            self.rrg /= total
            self.gann /= total
            self.candlestick /= total
            self.wave /= total

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_reversion": self.mean_reversion,
            "momentum": self.momentum,
            "rrg": self.rrg,
            "gann": self.gann,
            "candlestick": self.candlestick,
            "wave": self.wave,
        }


class MeanReversionSignal:
    """
    Mean reversion signal generator.

    Uses z-score stretch + reversion confirmation.
    """

    def __init__(
        self,
        zscore_threshold: float = 2.0,
        reversion_delta: float = 0.2,
    ):
        self.zscore_threshold = zscore_threshold
        self.reversion_delta = reversion_delta

    def generate(self, features: Dict[str, float]) -> SignalResult:
        """Generate mean reversion signal from features."""
        # Get z-score (try multiple feature names)
        zscore = features.get("mr_zscore", features.get("zscore_price", 0))
        zscore_change = features.get("mr_zscore_change", 0)

        # Check for long signal (from features)
        mr_long = features.get("mr_long_signal", 0)
        mr_short = features.get("mr_short_signal", 0)

        if mr_long > 0:
            confidence = min(abs(zscore) / 3.0, 1.0)
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=confidence,
                reason=f"MR Long: zscore={zscore:.2f}, change={zscore_change:.2f}",
            )

        if mr_short > 0:
            confidence = min(abs(zscore) / 3.0, 1.0)
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=confidence,
                reason=f"MR Short: zscore={zscore:.2f}, change={zscore_change:.2f}",
            )

        # Manual check if signals not pre-computed
        if zscore < -self.zscore_threshold and zscore_change > self.reversion_delta:
            confidence = min(abs(zscore) / 3.0, 1.0)
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=confidence,
                reason=f"MR Long (manual): zscore={zscore:.2f}",
            )

        if zscore > self.zscore_threshold and zscore_change < -self.reversion_delta:
            confidence = min(abs(zscore) / 3.0, 1.0)
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=confidence,
                reason=f"MR Short (manual): zscore={zscore:.2f}",
            )

        return SignalResult(
            direction=SignalDirection.FLAT, confidence=0.0, reason="No MR signal"
        )


class MomentumSignal:
    """
    Momentum signal generator.

    Uses RSI + MACD crossover.
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate(self, features: Dict[str, float]) -> SignalResult:
        """Generate momentum signal from features."""
        # Get RSI (try multiple names)
        rsi = None
        for key in ["1H_rsi_14", "rsi_14", "rsi"]:
            if key in features:
                rsi = features[key]
                break

        if rsi is None:
            rsi = 50  # Neutral default

        # Get MACD histogram
        macd = features.get("1H_macd", features.get("macd", 0))
        macd_signal = features.get("1H_macd_signal", features.get("macd_signal", 0))
        macd_hist = macd - macd_signal

        # Get momentum score if available
        momentum_score = features.get(
            "momentum_score", features.get("1H_momentum_score", 0)
        )

        # Long signal: RSI oversold + bullish MACD
        if rsi < self.rsi_oversold and macd_hist > 0:
            confidence = (self.rsi_oversold - rsi) / self.rsi_oversold
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=min(confidence, 1.0),
                reason=f"Momentum Long: RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}",
            )

        # Short signal: RSI overbought + bearish MACD
        if rsi > self.rsi_overbought and macd_hist < 0:
            confidence = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=min(confidence, 1.0),
                reason=f"Momentum Short: RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}",
            )

        # Use momentum score as secondary signal
        if abs(momentum_score) > 0.5:
            direction = (
                SignalDirection.LONG if momentum_score > 0 else SignalDirection.SHORT
            )
            return SignalResult(
                direction=direction,
                confidence=min(abs(momentum_score), 1.0) * 0.5,  # Reduced confidence
                reason=f"Momentum score: {momentum_score:.2f}",
            )

        return SignalResult(
            direction=SignalDirection.FLAT,
            confidence=0.0,
            reason=f"No momentum signal: RSI={rsi:.1f}",
        )


class RRGSignal:
    """
    RRG (Relative Rotation Graph) signal generator.

    Uses quadrant position for relative strength signals.
    """

    def generate(self, features: Dict[str, float]) -> SignalResult:
        """Generate RRG signal from features."""
        # Check RRG quadrant indicators
        rrg_leading = features.get("rrg_leading", features.get("1H_rrg_leading", 0))
        rrg_improving = features.get(
            "rrg_improving", features.get("1H_rrg_improving", 0)
        )
        rrg_weakening = features.get(
            "rrg_weakening", features.get("1H_rrg_weakening", 0)
        )
        rrg_lagging = features.get("rrg_lagging", features.get("1H_rrg_lagging", 0))

        # Get RS ratio and momentum
        rs_ratio = features.get("rs_ratio", features.get("1H_rs_ratio", 100))
        rs_momentum = features.get("rs_momentum", features.get("1H_rs_momentum", 100))

        # Long: Leading or Improving quadrant
        if rrg_leading > 0 or rrg_improving > 0:
            confidence = min(((rs_ratio - 100) + (rs_momentum - 100)) / 20 + 0.5, 1.0)
            quadrant = "Leading" if rrg_leading > 0 else "Improving"
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=max(0.3, confidence),
                reason=f"RRG {quadrant}: RS={rs_ratio:.1f}, Mom={rs_momentum:.1f}",
            )

        # Short: Weakening or Lagging quadrant
        if rrg_weakening > 0 or rrg_lagging > 0:
            confidence = min((100 - rs_ratio + 100 - rs_momentum) / 20 + 0.5, 1.0)
            quadrant = "Weakening" if rrg_weakening > 0 else "Lagging"
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=max(0.3, confidence),
                reason=f"RRG {quadrant}: RS={rs_ratio:.1f}, Mom={rs_momentum:.1f}",
            )

        # Use RRG long favorable flag if available
        rrg_long_favorable = features.get(
            "rrg_long_favorable", features.get("1H_rrg_long_favorable", 0)
        )
        if rrg_long_favorable > 0:
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=0.4,
                reason="RRG favorable for longs",
            )

        return SignalResult(
            direction=SignalDirection.FLAT, confidence=0.0, reason="No RRG signal"
        )


class GannSignal:
    """
    Gann-based signal generator.

    Uses swing points, retracement levels, and price-time analysis.
    """

    def generate(self, features: Dict[str, float]) -> SignalResult:
        """Generate Gann signal from features."""
        # Get Gann features
        gann_oversold = features.get(
            "gann_oversold", features.get("1H_gann_oversold", 0)
        )
        gann_overbought = features.get(
            "gann_overbought", features.get("1H_gann_overbought", 0)
        )
        gann_near_level = features.get(
            "gann_near_any_level", features.get("1H_gann_near_any_level", 0)
        )

        # Retracement levels
        near_382 = features.get(
            "gann_retracement_382", features.get("1H_gann_retracement_382", 0)
        )
        near_500 = features.get(
            "gann_retracement_500", features.get("1H_gann_retracement_500", 0)
        )
        near_618 = features.get(
            "gann_retracement_618", features.get("1H_gann_retracement_618", 0)
        )

        # Range position (0-1 scale)
        range_position = features.get(
            "gann_range_position", features.get("1H_gann_range_position", 0.5)
        )

        # Gann angles
        vs_1x1_low = features.get(
            "gann_vs_1x1_low", features.get("1H_gann_vs_1x1_low", 0)
        )
        vs_1x1_high = features.get(
            "gann_vs_1x1_high", features.get("1H_gann_vs_1x1_high", 0)
        )

        # Long: Near support level + oversold
        if gann_oversold > 0 and gann_near_level > 0:
            confidence = 0.7 if (near_618 > 0 or near_500 > 0) else 0.5
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=confidence,
                reason=f"Gann Long: oversold at support, range_pos={range_position:.2f}",
            )

        # Long: Price above 1x1 from low (bullish Gann angle)
        if vs_1x1_low > 0.02:  # 2% above 1x1 line
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=0.4,
                reason=f"Gann Long: above 1x1 angle from low",
            )

        # Short: Near resistance level + overbought
        if gann_overbought > 0 and gann_near_level > 0:
            confidence = 0.7 if (near_382 > 0 or near_500 > 0) else 0.5
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=confidence,
                reason=f"Gann Short: overbought at resistance, range_pos={range_position:.2f}",
            )

        # Short: Price below 1x1 from high (bearish Gann angle)
        if vs_1x1_high < -0.02:  # 2% below 1x1 line
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=0.4,
                reason=f"Gann Short: below 1x1 angle from high",
            )

        return SignalResult(
            direction=SignalDirection.FLAT, confidence=0.0, reason="No Gann signal"
        )


class CandlestickSignal:
    """
    Candlestick pattern signal generator.

    Uses recognized candlestick patterns for reversal signals.
    """

    def generate(self, features: Dict[str, float]) -> SignalResult:
        """Generate candlestick signal from features."""
        # Get candlestick pattern counts
        bullish_count = features.get(
            "cdl_bullish_count", features.get("1H_cdl_bullish_count", 0)
        )
        bearish_count = features.get(
            "cdl_bearish_count", features.get("1H_cdl_bearish_count", 0)
        )
        net_signal = features.get(
            "cdl_net_signal", features.get("1H_cdl_net_signal", 0)
        )

        # Specific patterns
        double_bottom = features.get(
            "cdl_double_bottom", features.get("1H_cdl_double_bottom", 0)
        )

        # Strong bullish patterns
        if double_bottom > 0:
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=0.7,
                reason="Candlestick: Double bottom pattern",
            )

        # Net signal approach
        if net_signal > 0 and bullish_count >= 2:
            confidence = min(bullish_count / 5, 1.0)
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=max(0.4, confidence),
                reason=f"Candlestick Long: {bullish_count} bullish patterns",
            )

        if net_signal < 0 and bearish_count >= 2:
            confidence = min(bearish_count / 5, 1.0)
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=max(0.4, confidence),
                reason=f"Candlestick Short: {bearish_count} bearish patterns",
            )

        # Single pattern with net signal
        if abs(net_signal) > 0.5:
            direction = (
                SignalDirection.LONG if net_signal > 0 else SignalDirection.SHORT
            )
            return SignalResult(
                direction=direction,
                confidence=0.3,
                reason=f"Candlestick net signal: {net_signal:.2f}",
            )

        return SignalResult(
            direction=SignalDirection.FLAT,
            confidence=0.0,
            reason="No candlestick signal",
        )


class WaveSignal:
    """
    Elliott Wave context signal generator.

    Uses wave stage and probabilities for trend context.
    """

    def generate(self, features: Dict[str, float]) -> SignalResult:
        """Generate wave signal from features."""
        # Get wave probabilities
        prob_impulse_up = features.get(
            "prob_impulse_up", features.get("4H_prob_impulse_up", 0)
        )
        prob_impulse_down = features.get(
            "prob_impulse_down", features.get("4H_prob_impulse_down", 0)
        )
        prob_corr_up = features.get("prob_corr_up", features.get("4H_prob_corr_up", 0))
        prob_corr_down = features.get(
            "prob_corr_down", features.get("4H_prob_corr_down", 0)
        )

        # Wave stage and confidence
        wave_conf = features.get("wave_conf", features.get("4H_wave_conf", 0))

        # Strong impulse up signal
        if prob_impulse_up > 0.6 and wave_conf > 0.5:
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=prob_impulse_up,
                reason=f"Wave Long: impulse up prob={prob_impulse_up:.2f}",
            )

        # Correction up (buying the dip in uptrend)
        if prob_corr_up > 0.6 and wave_conf > 0.4:
            return SignalResult(
                direction=SignalDirection.LONG,
                confidence=prob_corr_up * 0.8,  # Slightly reduced
                reason=f"Wave Long: correction up prob={prob_corr_up:.2f}",
            )

        # Strong impulse down signal
        if prob_impulse_down > 0.6 and wave_conf > 0.5:
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=prob_impulse_down,
                reason=f"Wave Short: impulse down prob={prob_impulse_down:.2f}",
            )

        # Correction down (shorting the rally in downtrend)
        if prob_corr_down > 0.6 and wave_conf > 0.4:
            return SignalResult(
                direction=SignalDirection.SHORT,
                confidence=prob_corr_down * 0.8,
                reason=f"Wave Short: correction down prob={prob_corr_down:.2f}",
            )

        return SignalResult(
            direction=SignalDirection.FLAT, confidence=0.0, reason="No wave signal"
        )


class EnsembleSignalGenerator:
    """
    Ensemble signal generator combining multiple strategies.

    Combines signals from mean reversion, momentum, RRG, Gann,
    candlestick patterns, and wave analysis with configurable weights.
    """

    def __init__(
        self,
        weights: Optional[EnsembleWeights] = None,
        min_confidence: float = 0.3,
        min_agreement: int = 2,
        zscore_threshold: float = 2.0,
        reversion_delta: float = 0.2,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        """
        Initialize ensemble generator.

        Args:
            weights: Signal weights (default: equal weighting)
            min_confidence: Minimum confidence to generate signal
            min_agreement: Minimum number of agreeing strategies
            zscore_threshold: Z-score threshold for mean reversion
            reversion_delta: Delta for reversion confirmation
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        self.weights = weights or EnsembleWeights()
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement

        # Initialize individual signal generators
        self.mean_reversion = MeanReversionSignal(zscore_threshold, reversion_delta)
        self.momentum = MomentumSignal(rsi_oversold, rsi_overbought)
        self.rrg = RRGSignal()
        self.gann = GannSignal()
        self.candlestick = CandlestickSignal()
        self.wave = WaveSignal()

    def generate_signal(self, features: Dict[str, float]) -> EnsembleSignal:
        """
        Generate ensemble signal from features.

        Args:
            features: Dictionary of feature values

        Returns:
            EnsembleSignal with combined direction and confidence
        """
        # Generate individual signals
        signals = {
            "mean_reversion": self.mean_reversion.generate(features),
            "momentum": self.momentum.generate(features),
            "rrg": self.rrg.generate(features),
            "gann": self.gann.generate(features),
            "candlestick": self.candlestick.generate(features),
            "wave": self.wave.generate(features),
        }

        weights = self.weights.to_dict()

        # Calculate weighted score (-1 to +1)
        weighted_score = 0.0
        total_confidence = 0.0
        long_count = 0
        short_count = 0
        reasons = []

        for name, signal in signals.items():
            weight = weights.get(name, 0)

            if signal.direction == SignalDirection.LONG:
                weighted_score += weight * signal.confidence
                long_count += 1
                reasons.append(f"{name}: LONG ({signal.confidence:.2f})")
            elif signal.direction == SignalDirection.SHORT:
                weighted_score -= weight * signal.confidence
                short_count += 1
                reasons.append(f"{name}: SHORT ({signal.confidence:.2f})")

            total_confidence += weight * signal.confidence

        # Determine final direction
        if weighted_score > 0 and long_count >= self.min_agreement:
            direction = SignalDirection.LONG
            confidence = min(weighted_score / 0.5, 1.0)  # Normalize
        elif weighted_score < 0 and short_count >= self.min_agreement:
            direction = SignalDirection.SHORT
            confidence = min(abs(weighted_score) / 0.5, 1.0)
        else:
            direction = SignalDirection.FLAT
            confidence = 0.0

        # Check minimum confidence
        if confidence < self.min_confidence:
            direction = SignalDirection.FLAT
            confidence = 0.0

        return EnsembleSignal(
            direction=direction,
            confidence=confidence,
            component_signals=signals,
            weighted_score=weighted_score,
            reason=" | ".join(reasons) if reasons else "No signals",
        )

    def generate_signals_bulk(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for entire DataFrame.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            DataFrame with signal, confidence columns
        """
        results = []

        for idx in features_df.index:
            row = features_df.loc[idx]
            features = row.to_dict()

            signal = self.generate_signal(features)

            results.append(
                {
                    "timestamp": idx,
                    "signal": signal.direction.value,
                    "confidence": signal.confidence,
                    "weighted_score": signal.weighted_score,
                    "reason": signal.reason,
                }
            )

        result_df = pd.DataFrame(results)
        if "timestamp" in result_df.columns:
            result_df = result_df.set_index("timestamp")

        return result_df

    def get_signal_breakdown(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Get detailed breakdown of signals from each strategy.

        Args:
            features: Feature dictionary

        Returns:
            Dictionary with signal details per strategy
        """
        signal = self.generate_signal(features)

        breakdown = {}
        for name, result in signal.component_signals.items():
            breakdown[name] = {
                "direction": result.direction.name,
                "confidence": result.confidence,
                "reason": result.reason,
                "weight": self.weights.to_dict().get(name, 0),
            }

        breakdown["ensemble"] = {
            "direction": signal.direction.name,
            "confidence": signal.confidence,
            "weighted_score": signal.weighted_score,
        }

        return breakdown


def create_ensemble_from_config(config: Dict) -> EnsembleSignalGenerator:
    """
    Create ensemble generator from configuration dictionary.

    Args:
        config: Configuration with weights and thresholds

    Returns:
        Configured EnsembleSignalGenerator
    """
    weights_config = config.get("weights", {})
    weights = EnsembleWeights(
        mean_reversion=weights_config.get("mean_reversion", 0.25),
        momentum=weights_config.get("momentum", 0.20),
        rrg=weights_config.get("rrg", 0.15),
        gann=weights_config.get("gann", 0.15),
        candlestick=weights_config.get("candlestick", 0.10),
        wave=weights_config.get("wave", 0.15),
    )

    return EnsembleSignalGenerator(
        weights=weights,
        min_confidence=config.get("min_confidence", 0.3),
        min_agreement=config.get("min_agreement", 2),
        zscore_threshold=config.get("zscore_threshold", 2.0),
        reversion_delta=config.get("reversion_delta", 0.2),
        rsi_oversold=config.get("rsi_oversold", 30),
        rsi_overbought=config.get("rsi_overbought", 70),
    )
