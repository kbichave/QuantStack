"""Deterministic rule-based market regime classification.

This module contains pure classification logic — no DB writes, no EventBus calls.
The caller (supervisor/nodes.py) handles persistence and event publishing.

Classification rules (applied in order):
  1. ADX < 20                                → ranging
  2. ADX > 25 AND spy_20d_return > 0.03      → trending_up
  3. ADX > 25 AND spy_20d_return < -0.03     → trending_down
  4. Everything else (20 ≤ ADX ≤ 25)        → unknown

Confidence is linearly interpolated toward 0.5 when ADX is within 3 points
of either threshold (ranging boundary at 20, trending boundary at 25+).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


_ADX_RANGING_THRESHOLD = 20.0
_ADX_TRENDING_THRESHOLD = 25.0
_RETURN_THRESHOLD = 0.03      # 3% 20-day return required for trending classification
_CONFIDENCE_ZONE = 3.0        # ADX must be this far from a boundary for full confidence


@dataclass(frozen=True)
class RegimeInputs:
    adx: float              # 14-period ADX for SPY
    spy_20d_return: float   # SPY 20-day return as decimal (e.g. 0.05 for 5%)
    vix_level: float
    breadth_score: float    # fraction of S&P 500 constituents above 50-day MA (0–1)
    previous_regime: str | None


@dataclass(frozen=True)
class RegimeClassification:
    regime: str             # trending_up | trending_down | ranging | unknown
    regime_change: bool
    confidence: float       # 0–1
    detected_at: datetime


def _compute_confidence(adx: float) -> float:
    """
    Compute confidence based on ADX distance from classification boundaries.

    Returns 1.0 when ADX is clearly above or below all thresholds.
    Linearly interpolates toward 0.5 within _CONFIDENCE_ZONE of any boundary.
    """
    # Distance from the ranging boundary (20) and the trending boundary (25)
    dist_from_ranging = abs(adx - _ADX_RANGING_THRESHOLD)
    dist_from_trending = abs(adx - _ADX_TRENDING_THRESHOLD)
    min_dist = min(dist_from_ranging, dist_from_trending)

    if min_dist >= _CONFIDENCE_ZONE:
        return 1.0

    # Linear interpolation: 0 distance → 0.5, _CONFIDENCE_ZONE distance → 1.0
    return 0.5 + 0.5 * (min_dist / _CONFIDENCE_ZONE)


def classify_regime(inputs: RegimeInputs) -> RegimeClassification:
    """
    Deterministic rule-based regime classification.

    Rules (applied in order):
      1. ADX < 20                                  → ranging
      2. ADX > 25 AND spy_20d_return > +3%         → trending_up
      3. ADX > 25 AND spy_20d_return < -3%         → trending_down
      4. Everything else                            → unknown

    Confidence: 1.0 when ADX is clearly above or below thresholds (< 20 or > 28).
    Confidence is linearly interpolated toward 0.5 when ADX is within 3 points
    of either threshold boundary.

    Regime change: True when new regime differs from previous_regime, or previous is None.
    """
    adx = inputs.adx
    ret = inputs.spy_20d_return

    if adx < _ADX_RANGING_THRESHOLD:
        regime = "ranging"
    elif adx > _ADX_TRENDING_THRESHOLD and ret > _RETURN_THRESHOLD:
        regime = "trending_up"
    elif adx > _ADX_TRENDING_THRESHOLD and ret < -_RETURN_THRESHOLD:
        regime = "trending_down"
    else:
        regime = "unknown"

    regime_change = inputs.previous_regime is None or regime != inputs.previous_regime
    confidence = _compute_confidence(adx)

    return RegimeClassification(
        regime=regime,
        regime_change=regime_change,
        confidence=confidence,
        detected_at=datetime.now(timezone.utc),
    )
