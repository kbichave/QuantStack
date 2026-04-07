"""Regime flip detection and response logic.

Provides severity classification and stop-tightening math for positions
whose market regime has changed since entry.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Opposite-direction pairs (severe mismatch)
_OPPOSITES = {
    ("trending_up", "trending_down"),
    ("trending_down", "trending_up"),
}


def classify_regime_flip(entry_regime: str, current_regime: str) -> str | None:
    """Classify the severity of a regime change.

    Returns "severe", "moderate", or None (no actionable flip).
    """
    if entry_regime == current_regime:
        return None
    if entry_regime == "unknown":
        return None  # can't assess flip from unknown entry
    if (entry_regime, current_regime) in _OPPOSITES:
        return "severe"
    if current_regime == "unknown":
        return "moderate"
    # Lateral flip (trending->ranging, ranging->trending)
    if entry_regime != current_regime:
        return "moderate"
    return None


def compute_tightened_stop(
    current_price: float,
    stop_price: float | None,
    entry_atr: float,
    side: str = "long",
) -> float:
    """Compute a tightened stop price with floor enforcement.

    Halves the current stop distance, enforced at max(2*ATR, 1% of price).
    If stop_price is None, sets a new stop at the floor distance.
    """
    floor = max(2.0 * entry_atr, 0.01 * current_price)

    if stop_price is None:
        if side == "long":
            return current_price - floor
        return current_price + floor

    if side == "long":
        distance = current_price - stop_price
        new_distance = distance * 0.5
        new_distance = max(new_distance, floor)
        return current_price - new_distance
    else:
        distance = stop_price - current_price
        new_distance = distance * 0.5
        new_distance = max(new_distance, floor)
        return current_price + new_distance


def generate_regime_flip_actions(
    symbol: str,
    side: str,
    quantity: int,
    entry_regime: str,
    current_regime: str,
    current_price: float,
    stop_price: float | None,
    entry_atr: float,
) -> dict:
    """Determine actions for a regime flip.

    Returns a dict with:
      - severity: "severe" | "moderate" | None
      - exit_order: dict or None (for severe)
      - new_stop: float or None (for moderate)
    """
    severity = classify_regime_flip(entry_regime, current_regime)
    if severity is None:
        return {"severity": None, "exit_order": None, "new_stop": None}

    result: dict = {"severity": severity, "exit_order": None, "new_stop": None}

    if severity == "severe":
        result["exit_order"] = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "reason": "regime_flip_severe",
            "entry_regime": entry_regime,
            "current_regime": current_regime,
        }
        # Also set a stop if there isn't one (belt and suspenders)
        if stop_price is None:
            result["new_stop"] = compute_tightened_stop(
                current_price, None, entry_atr, side,
            )

    if severity == "moderate":
        result["new_stop"] = compute_tightened_stop(
            current_price, stop_price, entry_atr, side,
        )

    return result
