"""Data models for threshold calibration results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a threshold calibration computation."""

    threshold_name: str
    value: float
    confidence_interval: tuple[float, float]
    sample_size: int
    methodology: str
    is_fallback: bool = False
