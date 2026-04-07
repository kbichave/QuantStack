"""Monthly threshold tuner.

Adjusts decision thresholds based on observed false-rejection and
false-acceptance rates.  Changes are bounded by the floor/ceiling defined
in ``thresholds.yaml`` to prevent runaway relaxation or tightening.
"""

from __future__ import annotations

from quantstack.meta.config import (
    get_threshold,
    get_threshold_bounds,
    set_threshold,
    save_thresholds,
)

_STEP = 0.05


def tune_threshold(
    name: str,
    false_rejection_rate: float,
    false_acceptance_rate: float,
) -> float | None:
    """Adjust the threshold named *name* and return the new value, or None if unchanged.

    Rules:
      - false_rejection_rate > 0.20 --> lower by 0.05 (respecting floor)
      - false_acceptance_rate > 0.30 --> raise by 0.05 (respecting ceiling)
      - Otherwise: no change
    """
    current = get_threshold(name)
    floor, ceiling = get_threshold_bounds(name)

    if false_rejection_rate > 0.20:
        new_value = max(floor, current - _STEP)
        if new_value != current:
            set_threshold(name, new_value)
            return new_value
        return None

    if false_acceptance_rate > 0.30:
        new_value = min(ceiling, current + _STEP)
        if new_value != current:
            set_threshold(name, new_value)
            return new_value
        return None

    return None


def run_monthly_tuning(outcome_data: list[dict]) -> list[dict]:
    """Run tuning across all thresholds given observed outcome data.

    Each entry in *outcome_data* should have keys:
      - ``threshold_name`` (str)
      - ``false_rejection_rate`` (float)
      - ``false_acceptance_rate`` (float)

    Returns a list of dicts describing applied changes.
    """
    changes: list[dict] = []
    for entry in outcome_data:
        name = entry["threshold_name"]
        before = get_threshold(name)
        result = tune_threshold(
            name,
            entry["false_rejection_rate"],
            entry["false_acceptance_rate"],
        )
        if result is not None:
            changes.append(
                {
                    "threshold_name": name,
                    "before": before,
                    "after": result,
                }
            )
    if changes:
        save_thresholds()
    return changes
