"""
Temporal lookahead bias detector.

Complements the statistical leakage tests in ``leakage.py`` by checking
*metadata-based* temporal validity: does a feature's publication date fall
before the signal timestamp?

The two approaches are orthogonal:

* ``leakage.py`` — statistical: shift features, permute labels, KS drift.
* This module  — metadata: use PIT registry delays to flag violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from loguru import logger

from quantstack.core.features.pit_registry import PUBLICATION_DELAYS


@dataclass(frozen=True)
class LookaheadViolation:
    """A single look-ahead bias violation."""

    feature_name: str
    signal_time: datetime
    known_since: datetime
    violation_days: float


@dataclass(frozen=True)
class FeatureTimestamp:
    """Temporal metadata for one feature observation."""

    feature_name: str
    known_since: datetime


def check_lookahead(
    features: list[FeatureTimestamp],
    signal_time: datetime,
) -> list[LookaheadViolation]:
    """Flag features whose ``known_since`` is after *signal_time*.

    Args:
        features: List of feature timestamps to validate.
        signal_time: The point in time the signal is generated.

    Returns:
        List of violations (empty if all features are clean).
    """
    violations: list[LookaheadViolation] = []
    for ft in features:
        if ft.known_since > signal_time:
            delta = (ft.known_since - signal_time).total_seconds() / 86400
            violations.append(
                LookaheadViolation(
                    feature_name=ft.feature_name,
                    signal_time=signal_time,
                    known_since=ft.known_since,
                    violation_days=round(delta, 2),
                )
            )
    return violations


def check_feature_availability(
    df: pd.DataFrame,
    feature_sources: dict[str, str],
) -> list[LookaheadViolation]:
    """Scan a feature DataFrame for PIT violations using the registry.

    For each feature column, the publication delay for its data source is
    looked up in :data:`PUBLICATION_DELAYS`.  If the delay is non-zero, we
    check whether the feature appears to be available *before* the delay
    window would allow.

    In a correctly shifted DataFrame (via ``pit_registry.shift_to_available``),
    this check should find zero violations.  If it finds violations, it means
    the feature pipeline omitted the PIT shift.

    Args:
        df: Feature DataFrame with a DatetimeIndex.
        feature_sources: Mapping of ``{column_name: source_key}`` where
            source_key is a key in ``PUBLICATION_DELAYS``.

    Returns:
        List of violations.
    """
    violations: list[LookaheadViolation] = []

    for col, source in feature_sources.items():
        if col not in df.columns:
            continue

        delay_days = PUBLICATION_DELAYS.get(source, 0)
        if delay_days == 0:
            continue

        # Approximate trading-day shift that *should* have been applied.
        expected_shift = max(1, int(delay_days / 1.4) + 1)

        # If the feature has non-NaN values within the first `expected_shift`
        # rows of the DataFrame, it's likely being used before it's known.
        first_valid_idx = df[col].first_valid_index()
        if first_valid_idx is None:
            continue

        first_valid_pos = df.index.get_loc(first_valid_idx)
        if first_valid_pos < expected_shift:
            violations.append(
                LookaheadViolation(
                    feature_name=col,
                    signal_time=df.index[first_valid_pos],
                    known_since=df.index[min(first_valid_pos + expected_shift, len(df) - 1)],
                    violation_days=float(delay_days),
                )
            )
            logger.warning(
                f"Lookahead violation: {col} ({source}) available at row "
                f"{first_valid_pos}, expected shift >= {expected_shift}"
            )

    return violations
