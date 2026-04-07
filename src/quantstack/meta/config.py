"""Centralized threshold configuration for meta agents.

Thresholds are stored in ``thresholds.yaml`` alongside this module.  Each
threshold has a current ``value`` plus hard ``floor`` and ``ceiling`` bounds
that meta-tuning must respect.  This prevents runaway self-modification from
relaxing quality gates below safe minimums or tightening them to the point
where nothing passes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).parent / "thresholds.yaml"


def _load_thresholds() -> dict[str, dict]:
    """Read thresholds from the YAML file on disk."""
    with open(_YAML_PATH) as fh:
        return yaml.safe_load(fh)


_THRESHOLDS: dict[str, dict] = _load_thresholds()


def get_threshold(name: str) -> float:
    """Return the current value for *name*, raising KeyError if unknown."""
    return float(_THRESHOLDS[name]["value"])


def get_threshold_bounds(name: str) -> tuple[float, float]:
    """Return ``(floor, ceiling)`` for the named threshold."""
    entry = _THRESHOLDS[name]
    return float(entry["floor"]), float(entry["ceiling"])


def set_threshold(name: str, value: float) -> None:
    """Set *name* to *value*, clamped to its floor/ceiling bounds."""
    entry = _THRESHOLDS[name]
    floor = float(entry["floor"])
    ceiling = float(entry["ceiling"])
    clamped = max(floor, min(ceiling, value))
    _THRESHOLDS[name]["value"] = clamped


def save_thresholds() -> None:
    """Persist current in-memory thresholds back to the YAML file."""
    with open(_YAML_PATH, "w") as fh:
        yaml.safe_dump(_THRESHOLDS, fh, default_flow_style=False)
