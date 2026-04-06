# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Execution Quality Scoring — maps mean absolute forecast error to a position
sizing scalar.

The quality scalar adjusts order quantities in the risk gate based on
historical execution quality.  Symbols with tight fills get a bonus (1.1x),
while symbols with consistently poor fills are penalised (down to 0.5x).

Threshold table:
    mean_abs_error < 5 bps   -> 1.1  (excellent — slight bonus)
    5 <= error <= 15 bps     -> 1.0  (acceptable — no adjustment)
    15 < error <= 30 bps     -> 0.7  (poor — reduce size)
    error > 30 bps           -> 0.5  (very poor — significant reduction)
"""

from __future__ import annotations


def compute_quality_scalar(mean_abs_error_bps: float) -> float:
    """Convert mean absolute forecast error (bps) into a position sizing scalar.

    Args:
        mean_abs_error_bps: Mean absolute error between pre-trade forecast
            and realised execution cost, in basis points.

    Returns:
        A float multiplier to apply to proposed order quantity.
    """
    if mean_abs_error_bps < 5:
        return 1.1
    if mean_abs_error_bps <= 15:
        return 1.0
    if mean_abs_error_bps <= 30:
        return 0.7
    return 0.5
