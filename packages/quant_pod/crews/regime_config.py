# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Regime-adaptive crew configuration.

Different market regimes call for different tool selection, pod activation,
position sizing, and confidence thresholds. This module maps the RegimeDetector
output to crew-level configuration so the right pods fire for the right regimes.

Design:
  - Regime determines WHICH ICs and pods are active (not agent prompts)
  - Size multiplier scales output of RiskGate position sizing down when conviction
    is structurally lower (ranging, high-vol)
  - Confidence threshold sets the floor: agents must clear this before execution
  - Notes are injected into crew context so agents understand why certain tools
    are more prominent

Regimes from RegimeDetector:
  trend:      trending_up | trending_down | ranging | unknown
  volatility: low | normal | high | extreme
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RegimeCrewConfig:
    """
    Crew configuration derived from current market regime.

    Consumed by TradingDayFlow.run_crew_analysis() to override the default
    profile before kicking off the crew.
    """

    # ICs to ADD on top of the default equities profile
    additional_ics: List[str] = field(default_factory=list)

    # ICs to REMOVE from the default equities profile (e.g. suppress trend IC in ranging)
    suppressed_ics: List[str] = field(default_factory=list)

    # Pod managers to ADD on top of default equities profile
    additional_pods: List[str] = field(default_factory=list)

    # Scale applied to RiskGate approved quantity: 0.25 (crisis) – 1.0 (full)
    # Final size = RiskGate.approved_quantity * size_multiplier
    size_multiplier: float = 1.0

    # Minimum confidence an agent must express before execution is allowed.
    # Pairs with the calibration tracker: if calibration says agent is overconfident,
    # the effective threshold is raised further in execute_trades().
    confidence_threshold: float = 0.55

    # Notes injected at the top of crew inputs under "regime_guidance" key.
    # Agents read this and should adjust their reasoning accordingly.
    regime_notes: str = ""

    # Regime labels for logging
    trend_regime: str = "unknown"
    volatility_regime: str = "normal"


# =============================================================================
# REGIME MAPPING
# =============================================================================

# Per-regime config table.
# Keys are (trend_regime, volatility_regime) tuples.
# The resolver does a two-pass lookup: exact match first, then volatility-only fallback.
_REGIME_CONFIGS: Dict[tuple, RegimeCrewConfig] = {

    # ---- TRENDING UP --------------------------------------------------------
    ("trending_up", "low"): RegimeCrewConfig(
        additional_ics=[],
        suppressed_ics=["statarb_ic"],   # Stat-arb underperforms in strong trends
        size_multiplier=1.0,
        confidence_threshold=0.55,
        regime_notes=(
            "REGIME: Strong uptrend, low volatility. "
            "Favor momentum and breakout signals. "
            "Mean-reversion and stat-arb setups are likely to underperform — "
            "do not prioritize them. Full position size approved."
        ),
    ),
    ("trending_up", "normal"): RegimeCrewConfig(
        additional_ics=[],
        suppressed_ics=["statarb_ic"],
        size_multiplier=1.0,
        confidence_threshold=0.55,
        regime_notes=(
            "REGIME: Uptrend, normal volatility. "
            "Standard trend-following approach. "
            "Prioritize trend_momentum and structure_levels analysis."
        ),
    ),
    ("trending_up", "high"): RegimeCrewConfig(
        additional_ics=["options_vol_ic"],  # Options vol IC valuable in high-vol uptrend
        suppressed_ics=[],
        size_multiplier=0.75,
        confidence_threshold=0.60,
        regime_notes=(
            "REGIME: Uptrend but elevated volatility. "
            "Momentum signals valid but wider swings expected. "
            "Reduce position size by 25%. "
            "Options vol IC activated — consider whether IV supports directional trades."
        ),
    ),
    ("trending_up", "extreme"): RegimeCrewConfig(
        additional_ics=["options_vol_ic"],
        suppressed_ics=[],
        size_multiplier=0.40,
        confidence_threshold=0.70,
        regime_notes=(
            "REGIME: Uptrend with extreme volatility. "
            "High risk of sharp reversals despite trend direction. "
            "Require strong IC consensus (≥70% confidence) before executing. "
            "Size reduced to 40% of normal. Options pod elevated."
        ),
    ),

    # ---- TRENDING DOWN ------------------------------------------------------
    ("trending_down", "low"): RegimeCrewConfig(
        additional_ics=[],
        suppressed_ics=["statarb_ic"],
        size_multiplier=0.60,
        confidence_threshold=0.60,
        regime_notes=(
            "REGIME: Downtrend, low volatility. "
            "Avoid new long entries unless reversal confirmed by structure levels. "
            "Risk pod should flag existing long exposure for review. "
            "Size at 60% — do not add to losers."
        ),
    ),
    ("trending_down", "normal"): RegimeCrewConfig(
        additional_ics=[],
        suppressed_ics=["statarb_ic"],
        size_multiplier=0.60,
        confidence_threshold=0.60,
        regime_notes=(
            "REGIME: Downtrend, normal volatility. "
            "Defensive posture. Only high-confidence mean-reversion or confirmed "
            "reversals warrant new longs. Size at 60%."
        ),
    ),
    ("trending_down", "high"): RegimeCrewConfig(
        additional_ics=["options_vol_ic"],
        suppressed_ics=["statarb_ic"],
        size_multiplier=0.35,
        confidence_threshold=0.65,
        regime_notes=(
            "REGIME: Downtrend with high volatility. "
            "Bear market / correction conditions. "
            "New long positions require strong consensus (≥65% confidence). "
            "Protective puts and hedges should be considered. Size at 35%."
        ),
    ),
    ("trending_down", "extreme"): RegimeCrewConfig(
        additional_ics=["options_vol_ic"],
        suppressed_ics=["statarb_ic", "trend_momentum_ic"],
        size_multiplier=0.20,
        confidence_threshold=0.75,
        regime_notes=(
            "REGIME: Downtrend with extreme volatility — crisis conditions. "
            "All new position entries require full IC agreement (≥75% confidence). "
            "Risk pod must approve explicitly. Size at 20% of normal. "
            "Options pod prioritized for hedging analysis."
        ),
    ),

    # ---- RANGING / SIDEWAYS -------------------------------------------------
    ("ranging", "low"): RegimeCrewConfig(
        additional_ics=["statarb_ic"],   # Stat-arb thrives in ranging markets
        suppressed_ics=["trend_momentum_ic"],
        size_multiplier=0.80,
        confidence_threshold=0.58,
        regime_notes=(
            "REGIME: Ranging market, low volatility. "
            "Mean-reversion and stat-arb setups preferred over momentum. "
            "Trend-following signals are likely false breakouts — discount them. "
            "Structure levels (support/resistance) are the primary edge."
        ),
    ),
    ("ranging", "normal"): RegimeCrewConfig(
        additional_ics=["statarb_ic"],
        suppressed_ics=["trend_momentum_ic"],
        size_multiplier=0.80,
        confidence_threshold=0.58,
        regime_notes=(
            "REGIME: Ranging market, normal volatility. "
            "Mean-reversion at extremes. Avoid chasing breakouts until confirmed. "
            "Stat-arb IC elevated. Trend momentum IC suppressed."
        ),
    ),
    ("ranging", "high"): RegimeCrewConfig(
        additional_ics=["options_vol_ic", "statarb_ic"],
        suppressed_ics=["trend_momentum_ic"],
        size_multiplier=0.55,
        confidence_threshold=0.62,
        regime_notes=(
            "REGIME: Ranging with high volatility (choppy). "
            "Most directional signals will whipsaw. "
            "Stat-arb and volatility plays are the primary opportunity. "
            "Reduce size to 55%. Require clear structure level confluence."
        ),
    ),

    # ---- UNKNOWN FALLBACK ---------------------------------------------------
    ("unknown", "normal"): RegimeCrewConfig(
        additional_ics=[],
        suppressed_ics=[],
        size_multiplier=0.50,
        confidence_threshold=0.65,
        regime_notes=(
            "REGIME: Unknown / insufficient data. "
            "Conservative posture until regime clarifies. "
            "Half-size positions only. Require 65% confidence minimum."
        ),
    ),
}

# Volatility-only fallback when exact (trend, vol) pair not in table
_VOLATILITY_FALLBACK: Dict[str, RegimeCrewConfig] = {
    "high": RegimeCrewConfig(
        additional_ics=["options_vol_ic"],
        suppressed_ics=[],
        size_multiplier=0.60,
        confidence_threshold=0.62,
        regime_notes="REGIME: High volatility detected. Reducing size to 60%, activating options vol IC.",
    ),
    "extreme": RegimeCrewConfig(
        additional_ics=["options_vol_ic"],
        suppressed_ics=[],
        size_multiplier=0.25,
        confidence_threshold=0.70,
        regime_notes="REGIME: Extreme volatility. Crisis-mode sizing (25%). Full IC consensus required.",
    ),
}


# =============================================================================
# PUBLIC API
# =============================================================================


def get_regime_crew_config(
    trend_regime: str,
    volatility_regime: str,
) -> RegimeCrewConfig:
    """
    Return the crew configuration appropriate for the given regime.

    Lookup order:
      1. Exact (trend_regime, volatility_regime) match
      2. (trend_regime, "normal") fallback
      3. Volatility-only fallback for high/extreme volatility
      4. Conservative default

    Args:
        trend_regime:      One of "trending_up", "trending_down", "ranging", "unknown"
        volatility_regime: One of "low", "normal", "high", "extreme"

    Returns:
        RegimeCrewConfig with active/suppressed ICs, size_multiplier, threshold, notes
    """
    trend = trend_regime or "unknown"
    vol = volatility_regime or "normal"

    config = (
        _REGIME_CONFIGS.get((trend, vol))
        or _REGIME_CONFIGS.get((trend, "normal"))
        or _VOLATILITY_FALLBACK.get(vol)
        or RegimeCrewConfig(
            size_multiplier=0.50,
            confidence_threshold=0.65,
            regime_notes=f"REGIME: Unrecognized ({trend}/{vol}). Conservative defaults applied.",
        )
    )
    config.trend_regime = trend
    config.volatility_regime = vol
    return config


def apply_regime_config_to_inputs(
    inputs: Dict[str, Any],
    config: RegimeCrewConfig,
    base_ics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Merge regime config into crew kickoff inputs.

    Adds regime_guidance (notes), adjusted_ics (for crews that support
    dynamic IC selection), size_multiplier, and confidence_threshold.

    Args:
        inputs:    Existing crew inputs dict (modified in place + returned)
        config:    RegimeCrewConfig from get_regime_crew_config()
        base_ics:  Default IC list to adjust (from PROFILE_DEFAULTS)

    Returns:
        Updated inputs dict
    """
    inputs["regime_guidance"] = config.regime_notes
    inputs["size_multiplier"] = config.size_multiplier
    inputs["confidence_threshold"] = config.confidence_threshold

    if base_ics is not None:
        adjusted = [ic for ic in base_ics if ic not in config.suppressed_ics]
        for ic in config.additional_ics:
            if ic not in adjusted:
                adjusted.append(ic)
        inputs["active_ics"] = adjusted

    return inputs
