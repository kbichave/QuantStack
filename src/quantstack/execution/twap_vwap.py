# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
TWAP/VWAP scheduling strategies — child order generation from parent orders.

Given an AlgoParentOrder with a defined execution window (start_time to end_time),
these functions produce lists of ChildOrder objects with scheduled times and
target quantities.  TWAP distributes evenly; VWAP weights by a volume profile
(supplied or synthetic U-shaped).

Randomisation (jitter on timing, variation on quantities) makes the execution
less predictable to adversarial observers while preserving the total-quantity
invariant: sum(child.target_quantity) == parent.total_quantity.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from quantstack.execution.algo_scheduler import AlgoParentOrder, ChildOrder


def synthetic_volume_profile(num_buckets: int) -> list[float]:
    """Generate a normalised U-shaped intraday volume profile.

    The U-shape mirrors typical equity-market volume: heavier at open and
    close, lighter midday.  Formula: weight[i] = 1 + 0.5 * cos(2*pi*i/(N-1))
    for N >= 2.  Single-bucket edge case returns [1.0].

    Returns a list of floats that sum to 1.0.
    """
    if num_buckets <= 0:
        return []
    if num_buckets == 1:
        return [1.0]

    raw = [
        1.0 + 0.5 * math.cos(2 * math.pi * i / (num_buckets - 1))
        for i in range(num_buckets)
    ]
    total = sum(raw)
    return [w / total for w in raw]


def plan_twap_children(
    parent: AlgoParentOrder,
    bucket_minutes: int = 5,
    rng: random.Random | None = None,
) -> list[ChildOrder]:
    """Slice a parent order into equal-time TWAP children.

    Each child gets approximately equal quantity with +/- 10% random variation
    (total preserved).  Scheduled times include +/- 20% jitter within each
    bucket so the order flow is less predictable.

    Args:
        parent: The parent algo order to slice.
        bucket_minutes: Width of each time bucket in minutes.
        rng: Optional seeded Random instance for reproducibility.

    Returns:
        List of ChildOrder with status="pending" and correctly formatted IDs.
    """
    if rng is None:
        rng = random.Random()

    window_seconds = (parent.end_time - parent.start_time).total_seconds()
    bucket_seconds = bucket_minutes * 60
    num_buckets = max(1, int(window_seconds / bucket_seconds))

    base_qty = parent.total_quantity // num_buckets
    remainder = parent.total_quantity - base_qty * num_buckets

    # Build raw quantities with +/- 10% variation, preserving total
    raw_quantities = [base_qty] * num_buckets
    # Distribute remainder to last child initially
    raw_quantities[-1] += remainder

    # Apply variation: perturb each child by up to 10% of base_qty, then
    # reconcile the last child to hit the exact total.
    if num_buckets > 1 and base_qty > 0:
        max_delta = max(1, int(base_qty * 0.10))
        for i in range(num_buckets - 1):
            delta = rng.randint(-max_delta, max_delta)
            raw_quantities[i] += delta
            # Clamp to at least 1 share
            raw_quantities[i] = max(1, raw_quantities[i])
        # Reconcile: last child absorbs the difference
        raw_quantities[-1] = parent.total_quantity - sum(raw_quantities[:-1])
        # Ensure last child is positive (edge case with very small orders)
        if raw_quantities[-1] < 1:
            # Steal from the largest other child
            largest_idx = max(range(num_buckets - 1), key=lambda j: raw_quantities[j])
            shortfall = 1 - raw_quantities[-1]
            raw_quantities[largest_idx] -= shortfall
            raw_quantities[-1] = 1

    children: list[ChildOrder] = []
    jitter_max_seconds = bucket_seconds * 0.20

    for i in range(num_buckets):
        bucket_start = parent.start_time + timedelta(seconds=i * bucket_seconds)
        jitter = timedelta(seconds=rng.uniform(-jitter_max_seconds, jitter_max_seconds))
        scheduled = bucket_start + jitter
        # Clamp within the execution window
        scheduled = max(parent.start_time, min(scheduled, parent.end_time))

        children.append(
            ChildOrder(
                child_id=f"{parent.parent_order_id}-C{i + 1:03d}",
                parent_id=parent.parent_order_id,
                scheduled_time=scheduled,
                target_quantity=raw_quantities[i],
            )
        )

    return children


def plan_vwap_children(
    parent: AlgoParentOrder,
    volume_profile: list[float] | None = None,
    bucket_minutes: int = 5,
    rng: random.Random | None = None,
) -> list[ChildOrder]:
    """Slice a parent order into VWAP-weighted children.

    If a historical volume_profile is provided, child quantities are
    proportional to each bucket's weight.  Otherwise a synthetic U-shaped
    curve is used (heavier at open/close, lighter midday).

    Args:
        parent: The parent algo order to slice.
        volume_profile: Optional list of floats (one per bucket).  Need not
            sum to 1.0 — they are normalised internally.
        bucket_minutes: Width of each time bucket in minutes.
        rng: Optional seeded Random instance for reproducibility.

    Returns:
        List of ChildOrder with status="pending" and correctly formatted IDs.
    """
    if rng is None:
        rng = random.Random()

    window_seconds = (parent.end_time - parent.start_time).total_seconds()
    bucket_seconds = bucket_minutes * 60
    num_buckets = max(1, int(window_seconds / bucket_seconds))

    # Resolve profile
    if volume_profile is not None:
        # Resample or truncate to match num_buckets
        profile = list(volume_profile[:num_buckets])
        # Pad with the mean if profile is shorter than num_buckets
        if len(profile) < num_buckets:
            mean_weight = sum(profile) / len(profile) if profile else 1.0
            profile.extend([mean_weight] * (num_buckets - len(profile)))
    else:
        profile = synthetic_volume_profile(num_buckets)

    # Normalise
    total_weight = sum(profile)
    if total_weight <= 0:
        # Degenerate case: fall back to uniform
        profile = [1.0 / num_buckets] * num_buckets
        total_weight = 1.0
    else:
        profile = [w / total_weight for w in profile]

    # Allocate quantities proportionally, rounding down, then fix remainder
    raw_quantities = [int(parent.total_quantity * w) for w in profile]
    allocated = sum(raw_quantities)
    remainder = parent.total_quantity - allocated

    # Distribute remainder one share at a time to the buckets with the
    # largest fractional parts (greedy rounding).
    fractional_parts = [
        (parent.total_quantity * profile[i]) - raw_quantities[i]
        for i in range(num_buckets)
    ]
    indices_by_fraction = sorted(
        range(num_buckets), key=lambda j: fractional_parts[j], reverse=True
    )
    for k in range(remainder):
        raw_quantities[indices_by_fraction[k]] += 1

    # Build children
    children: list[ChildOrder] = []
    jitter_max_seconds = bucket_seconds * 0.20

    for i in range(num_buckets):
        bucket_start = parent.start_time + timedelta(seconds=i * bucket_seconds)
        jitter = timedelta(seconds=rng.uniform(-jitter_max_seconds, jitter_max_seconds))
        scheduled = bucket_start + jitter
        scheduled = max(parent.start_time, min(scheduled, parent.end_time))

        children.append(
            ChildOrder(
                child_id=f"{parent.parent_order_id}-C{i + 1:03d}",
                parent_id=parent.parent_order_id,
                scheduled_time=scheduled,
                target_quantity=raw_quantities[i],
            )
        )

    return children
