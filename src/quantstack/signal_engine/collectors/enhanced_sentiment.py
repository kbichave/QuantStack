"""Enhanced sentiment aggregation with source weighting and time decay.

Aggregates analyst, news, and social sentiment into a composite score.
Not a standalone collector — called as post-processing in _build_brief().

Source weights (calibrated from historical predictive value):
  analyst_actions : 0.5
  news_sentiment  : 0.3
  social_sentiment: 0.2

Time decay: signals from last 24h weighted 2x vs signals from 7+ days ago.
"""

from __future__ import annotations

from typing import Any

# Source weights
_WEIGHT_ANALYST = 0.5
_WEIGHT_NEWS = 0.3
_WEIGHT_SOCIAL = 0.2


def _time_decay_multiplier(age_hours: float) -> float:
    """Compute time decay multiplier for a signal.

    weight = max(0.1, 2.0 - (t / 168) * 1.9)
    2.0x at t=0, 1.0x at ~88h, 0.1x at 168h (7 days).
    """
    return max(0.1, 2.0 - (age_hours / 168.0) * 1.9)


def compute_enhanced_sentiment(
    analyst_score: float | None = None,
    news_score: float | None = None,
    social_score: float | None = None,
    analyst_age_hours: float = 12.0,
    news_age_hours: float = 12.0,
    social_age_hours: float = 12.0,
) -> dict[str, Any]:
    """Aggregate multiple sentiment sources with weighting and time decay.

    Returns dict with composite_sentiment, source_count, agreement_rate,
    confidence, dominant_direction.
    """
    sources: list[tuple[float, float, float]] = []  # (score, weight, decay)

    if analyst_score is not None:
        decay = _time_decay_multiplier(analyst_age_hours)
        sources.append((analyst_score, _WEIGHT_ANALYST, decay))

    if news_score is not None:
        decay = _time_decay_multiplier(news_age_hours)
        sources.append((news_score, _WEIGHT_NEWS, decay))

    if social_score is not None:
        decay = _time_decay_multiplier(social_age_hours)
        sources.append((social_score, _WEIGHT_SOCIAL, decay))

    source_count = len(sources)

    if source_count == 0:
        return {
            "composite_sentiment": 0.5,
            "source_count": 0,
            "agreement_rate": 0.0,
            "confidence": 0.0,
            "dominant_direction": "neutral",
        }

    # Renormalize weights to available sources
    total_weight = sum(w * d for _, w, d in sources)
    if total_weight == 0:
        total_weight = 1.0

    composite = sum(score * weight * decay for score, weight, decay in sources) / total_weight

    # Agreement rate: how aligned are the sources?
    # 1.0 = all same direction, 0.0 = completely mixed
    if source_count >= 2:
        scores = [s for s, _, _ in sources]
        directions = [1 if s > 0.5 else (-1 if s < 0.5 else 0) for s in scores]
        if all(d == directions[0] for d in directions):
            agreement_rate = 1.0
        else:
            # Fraction of sources agreeing with majority
            from collections import Counter
            counts = Counter(directions)
            majority_count = counts.most_common(1)[0][1]
            agreement_rate = majority_count / len(directions)
    else:
        agreement_rate = 1.0  # Single source trivially agrees with itself

    # Confidence scoring
    if source_count >= 3 and agreement_rate >= 0.8:
        confidence = 0.9
    elif source_count >= 2 and agreement_rate >= 0.6:
        confidence = 0.6
    elif source_count == 1:
        confidence = 0.3
    else:
        confidence = 0.2

    # Dominant direction
    if composite > 0.6:
        dominant = "bullish"
    elif composite < 0.4:
        dominant = "bearish"
    else:
        dominant = "neutral"

    return {
        "composite_sentiment": round(composite, 4),
        "source_count": source_count,
        "agreement_rate": round(agreement_rate, 4),
        "confidence": confidence,
        "dominant_direction": dominant,
    }
