"""Earnings catalyst options: IV premium decision matrix.

Takes earnings analyst output (IV premium ratio, directional conviction)
and recommends an options play for upcoming earnings events.
"""

from __future__ import annotations


def recommend_earnings_options_play(
    iv_premium_ratio: float,
    directional_conviction: str | None,
    sue_history: float | None,
) -> dict | None:
    """Recommend an options play for an upcoming earnings event.

    Decision matrix:
    - IV premium < 1.2 + conviction: directional options
    - IV premium > 1.5 + no conviction: iron condor (sell premium)
    - IV premium 1.2-1.5: no play (ambiguous edge)
    - Missing/zero IV data: no play

    Returns:
        Dict with play_type, rationale, confidence. Or None for no play.
    """
    if iv_premium_ratio <= 0:
        return None

    if 1.2 <= iv_premium_ratio <= 1.5:
        return None

    if iv_premium_ratio < 1.2 and directional_conviction:
        if directional_conviction == "bullish":
            play_type = "directional_calls"
            rationale = f"Low IV premium ({iv_premium_ratio:.2f}) with bullish conviction"
        elif directional_conviction == "bearish":
            play_type = "directional_puts"
            rationale = f"Low IV premium ({iv_premium_ratio:.2f}) with bearish conviction"
        else:
            return None

        confidence = min(1.0, 0.5 + (abs(sue_history or 0) * 0.1))
        return {
            "play_type": play_type,
            "rationale": rationale,
            "confidence": confidence,
            "iv_premium_ratio": iv_premium_ratio,
        }

    if iv_premium_ratio > 1.5 and not directional_conviction:
        return {
            "play_type": "iron_condor",
            "rationale": f"High IV premium ({iv_premium_ratio:.2f}) — sell premium, collect IV crush",
            "confidence": min(1.0, 0.4 + (iv_premium_ratio - 1.5) * 0.3),
            "iv_premium_ratio": iv_premium_ratio,
        }

    return None
