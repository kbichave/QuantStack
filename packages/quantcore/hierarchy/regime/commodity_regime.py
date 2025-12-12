"""
Commodity regime detector integrating HMM, changepoint, and TFT models.

Provides unified regime detection for commodity trading.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.hierarchy.regime.hmm_model import (
    HMMRegimeModel,
    HMMRegimeState,
    HMMRegimeResult,
)
from quantcore.hierarchy.regime.changepoint import (
    BayesianChangepointDetector,
    ChangepointResult,
)
from quantcore.hierarchy.regime.tft_regime import (
    TFTRegimeModel,
    TFTRegimeState,
    TFTRegimeResult,
)


class CommodityRegimeType(Enum):
    """Unified commodity regime types."""

    INVENTORY_DRIVEN = "INVENTORY_DRIVEN"  # EIA/storage dominates
    MACRO_DRIVEN = "MACRO_DRIVEN"  # Risk-on/off, global growth
    USD_DRIVEN = "USD_DRIVEN"  # Dollar dominates
    VOLATILITY_DRIVEN = "VOLATILITY_DRIVEN"  # Vol regime dominates
    SEASONAL = "SEASONAL"  # Seasonal patterns dominate
    TRANSITION = "TRANSITION"  # Regime transition in progress


@dataclass
class CommodityRegimeResult:
    """Unified commodity regime result."""

    primary_regime: CommodityRegimeType
    regime_confidence: float
    hmm_result: Optional[HMMRegimeResult]
    changepoint_result: Optional[ChangepointResult]
    tft_result: Optional[TFTRegimeResult]
    regime_stability: float
    expected_regime_duration: int
    regime_context: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "primary_regime": self.primary_regime.value,
            "regime_confidence": self.regime_confidence,
            "hmm_state": self.hmm_result.state.name if self.hmm_result else None,
            "changepoint_prob": (
                self.changepoint_result.regime_change_probability
                if self.changepoint_result
                else 0
            ),
            "tft_regime": (
                self.tft_result.predicted_regime.name if self.tft_result else None
            ),
            "regime_stability": self.regime_stability,
            "expected_duration": self.expected_regime_duration,
            **self.regime_context,
        }


class CommodityRegimeDetector:
    """
    Unified commodity regime detector.

    Combines three detection methods:
    1. HMM: Statistical regime classification (vol/trend states)
    2. Changepoint: Detects regime transitions
    3. TFT: ML-based regime prediction with interpretability

    Additional context:
    - Event proximity (EIA, OPEC)
    - Cross-asset regime indicators
    - Curve regime (contango/backwardation)
    - Seasonal regime
    """

    def __init__(
        self,
        use_hmm: bool = True,
        use_changepoint: bool = True,
        use_tft: bool = True,
        hmm_lookback: int = 252,
        changepoint_hazard: float = 1 / 250,
        tft_lookback: int = 60,
    ):
        """
        Initialize commodity regime detector.

        Args:
            use_hmm: Whether to use HMM model
            use_changepoint: Whether to use changepoint detection
            use_tft: Whether to use TFT model
            hmm_lookback: Lookback for HMM training
            changepoint_hazard: Hazard rate for changepoint
            tft_lookback: Lookback for TFT model
        """
        self.use_hmm = use_hmm
        self.use_changepoint = use_changepoint
        self.use_tft = use_tft

        # Initialize models
        if use_hmm:
            self.hmm = HMMRegimeModel(lookback=hmm_lookback)
        else:
            self.hmm = None

        if use_changepoint:
            self.changepoint = BayesianChangepointDetector(
                hazard_rate=changepoint_hazard
            )
        else:
            self.changepoint = None

        if use_tft:
            self.tft = TFTRegimeModel(lookback=tft_lookback)
        else:
            self.tft = None

        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "CommodityRegimeDetector":
        """
        Fit regime detection models.

        Args:
            df: DataFrame with OHLCV data and features

        Returns:
            Self for chaining
        """
        if self.hmm is not None:
            logger.info("Fitting HMM model...")
            self.hmm.fit(df)

        if self.tft is not None:
            logger.info("Fitting TFT model...")
            self.tft.fit(df)

        self.is_fitted = True
        logger.info("Commodity regime detector fitted")

        return self

    def detect(self, df: pd.DataFrame) -> CommodityRegimeResult:
        """
        Detect current regime.

        Args:
            df: DataFrame with OHLCV data and commodity features

        Returns:
            CommodityRegimeResult with unified regime
        """
        # Get results from each model
        hmm_result = self.hmm.predict(df) if self.hmm is not None else None
        cp_result = (
            self.changepoint.detect(df) if self.changepoint is not None else None
        )
        tft_result = self.tft.predict(df) if self.tft is not None else None

        # Extract context from features
        regime_context = self._extract_context(df)

        # Combine results to determine primary regime
        primary_regime, confidence = self._determine_primary_regime(
            hmm_result, cp_result, tft_result, regime_context
        )

        # Calculate stability
        stability = self._calculate_stability(hmm_result, cp_result, tft_result)

        # Expected duration
        expected_duration = self._estimate_duration(hmm_result, primary_regime)

        return CommodityRegimeResult(
            primary_regime=primary_regime,
            regime_confidence=confidence,
            hmm_result=hmm_result,
            changepoint_result=cp_result,
            tft_result=tft_result,
            regime_stability=stability,
            expected_regime_duration=expected_duration,
            regime_context=regime_context,
        )

    def _extract_context(self, df: pd.DataFrame) -> Dict:
        """Extract regime context from features."""
        if len(df) == 0:
            return {}

        current = df.iloc[-1]
        context = {}

        # Event context
        context["eia_proximity"] = float(current.get("eia_proximity", 0))
        context["is_eia_day"] = int(current.get("is_eia_day", 0))
        context["high_event_risk"] = int(current.get("high_event_risk", 0))

        # Macro context
        context["vix_zscore"] = float(current.get("vix_zscore", 0))
        context["risk_off"] = int(current.get("risk_off", 0))

        # USD context
        context["usd_zscore"] = float(current.get("usd_zscore", 0))
        context["wti_usd_corr"] = float(current.get("wti_usd_corr", 0))

        # Volatility context
        context["vol_regime"] = int(current.get("vol_regime", 1))
        context["vol_zscore"] = float(current.get("vol_zscore", 0))

        # Curve context
        context["is_contango"] = int(current.get("is_contango", 0))
        context["curve_slope"] = float(current.get("curve_slope", 0))

        # Seasonal context
        context["is_driving_season"] = int(current.get("is_driving_season", 0))
        context["is_heating_season"] = int(current.get("is_heating_season", 0))

        return context

    def _determine_primary_regime(
        self,
        hmm_result: Optional[HMMRegimeResult],
        cp_result: Optional[ChangepointResult],
        tft_result: Optional[TFTRegimeResult],
        context: Dict,
    ) -> tuple:
        """Determine primary regime from all sources."""

        scores = {
            CommodityRegimeType.INVENTORY_DRIVEN: 0.0,
            CommodityRegimeType.MACRO_DRIVEN: 0.0,
            CommodityRegimeType.USD_DRIVEN: 0.0,
            CommodityRegimeType.VOLATILITY_DRIVEN: 0.0,
            CommodityRegimeType.SEASONAL: 0.0,
            CommodityRegimeType.TRANSITION: 0.0,
        }

        # Check for regime transition (changepoint)
        if cp_result is not None:
            if cp_result.regime_change_probability > 0.3:
                scores[
                    CommodityRegimeType.TRANSITION
                ] += cp_result.regime_change_probability

        # Inventory-driven (event proximity)
        eia_proximity = context.get("eia_proximity", 0)
        high_event = context.get("high_event_risk", 0)
        if eia_proximity > 0.5 or high_event == 1:
            scores[CommodityRegimeType.INVENTORY_DRIVEN] += 0.4 + eia_proximity * 0.3

        # Macro-driven (VIX/risk-off)
        vix_zscore = abs(context.get("vix_zscore", 0))
        risk_off = context.get("risk_off", 0)
        if vix_zscore > 1.0 or risk_off == 1:
            scores[CommodityRegimeType.MACRO_DRIVEN] += 0.3 + min(vix_zscore / 3, 0.4)

        # USD-driven (strong USD correlation)
        usd_corr = abs(context.get("wti_usd_corr", 0))
        usd_zscore = abs(context.get("usd_zscore", 0))
        if usd_corr > 0.5 and usd_zscore > 1.0:
            scores[CommodityRegimeType.USD_DRIVEN] += 0.3 + usd_corr * 0.3

        # Volatility-driven (high vol regime)
        vol_regime = context.get("vol_regime", 1)
        vol_zscore = abs(context.get("vol_zscore", 0))
        if vol_regime == 2 or vol_zscore > 1.5:
            scores[CommodityRegimeType.VOLATILITY_DRIVEN] += 0.4 + min(
                vol_zscore / 4, 0.3
            )

        # Add HMM contribution
        if hmm_result is not None:
            hmm_state = hmm_result.state
            if hmm_state in [
                HMMRegimeState.HIGH_VOL_BULL,
                HMMRegimeState.HIGH_VOL_BEAR,
            ]:
                scores[CommodityRegimeType.VOLATILITY_DRIVEN] += (
                    0.2 * hmm_result.regime_stability
                )
            else:
                # Low vol states favor other regimes
                scores[CommodityRegimeType.MACRO_DRIVEN] += (
                    0.1 * hmm_result.regime_stability
                )

        # Add TFT contribution
        if tft_result is not None:
            tft_regime = tft_result.predicted_regime
            tft_conf = tft_result.confidence

            if tft_regime == TFTRegimeState.VOLATILE:
                scores[CommodityRegimeType.VOLATILITY_DRIVEN] += 0.2 * tft_conf
            elif tft_regime == TFTRegimeState.RANGING:
                scores[CommodityRegimeType.SEASONAL] += 0.1 * tft_conf

        # Seasonal (if in seasonal period and no dominant regime)
        is_seasonal = context.get("is_driving_season", 0) or context.get(
            "is_heating_season", 0
        )
        if is_seasonal:
            scores[CommodityRegimeType.SEASONAL] += 0.2

        # Normalize and find max
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        # If transition has high score, use it
        if scores[CommodityRegimeType.TRANSITION] > 0.3:
            return (
                CommodityRegimeType.TRANSITION,
                scores[CommodityRegimeType.TRANSITION],
            )

        return primary, confidence

    def _calculate_stability(
        self,
        hmm_result: Optional[HMMRegimeResult],
        cp_result: Optional[ChangepointResult],
        tft_result: Optional[TFTRegimeResult],
    ) -> float:
        """Calculate regime stability."""
        stability_scores = []

        if hmm_result is not None:
            stability_scores.append(hmm_result.regime_stability)

        if cp_result is not None:
            # Lower changepoint probability = higher stability
            stability_scores.append(1.0 - cp_result.regime_change_probability)

        if tft_result is not None:
            stability_scores.append(tft_result.confidence)

        if len(stability_scores) == 0:
            return 0.5

        return float(np.mean(stability_scores))

    def _estimate_duration(
        self,
        hmm_result: Optional[HMMRegimeResult],
        regime: CommodityRegimeType,
    ) -> int:
        """Estimate expected regime duration (in bars)."""
        # Base duration by regime type
        base_durations = {
            CommodityRegimeType.INVENTORY_DRIVEN: 5,  # Short (event-driven)
            CommodityRegimeType.MACRO_DRIVEN: 20,  # Medium
            CommodityRegimeType.USD_DRIVEN: 15,  # Medium
            CommodityRegimeType.VOLATILITY_DRIVEN: 10,  # Short (vol mean reverts)
            CommodityRegimeType.SEASONAL: 60,  # Long (seasonal)
            CommodityRegimeType.TRANSITION: 3,  # Very short
        }

        base = base_durations.get(regime, 10)

        # Adjust based on HMM duration estimate
        if hmm_result is not None:
            hmm_duration = int(hmm_result.expected_duration)
            return int((base + hmm_duration) / 2)

        return base

    def get_regime_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime classification for entire series.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with regime columns
        """
        # Get HMM series
        hmm_series = self.hmm.predict_series(df) if self.hmm is not None else None

        # Get changepoint probability series
        cp_series = (
            self.changepoint.get_changepoint_probability_series(df)
            if self.changepoint is not None
            else None
        )

        # Build result DataFrame
        result = pd.DataFrame(index=df.index)

        if hmm_series is not None:
            result["hmm_regime"] = hmm_series

        if cp_series is not None:
            result["changepoint_prob"] = cp_series

        # Simple rolling regime detection
        result["primary_regime"] = self._rolling_regime_detection(df)

        return result

    def _rolling_regime_detection(self, df: pd.DataFrame) -> pd.Series:
        """Simple rolling regime detection for series."""
        regimes = []

        window = 20
        for i in range(len(df)):
            if i < window:
                regimes.append("MACRO_DRIVEN")
                continue

            # Look at recent context
            recent = df.iloc[max(0, i - window) : i + 1].iloc[-1]

            # Simple rules
            if recent.get("high_event_risk", 0) == 1:
                regimes.append("INVENTORY_DRIVEN")
            elif abs(recent.get("vix_zscore", 0)) > 1.5:
                regimes.append("MACRO_DRIVEN")
            elif abs(recent.get("usd_zscore", 0)) > 1.5:
                regimes.append("USD_DRIVEN")
            elif recent.get("vol_regime", 1) == 2:
                regimes.append("VOLATILITY_DRIVEN")
            elif recent.get("is_driving_season", 0) or recent.get(
                "is_heating_season", 0
            ):
                regimes.append("SEASONAL")
            else:
                regimes.append("MACRO_DRIVEN")

        return pd.Series(regimes, index=df.index, name="primary_regime")
