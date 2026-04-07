# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TWAP/VWAP child-order scheduling (section 08)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from quantstack.execution.algo_scheduler import AlgoParentOrder, ChildOrder
from quantstack.execution.twap_vwap import (
    plan_twap_children,
    plan_vwap_children,
    synthetic_volume_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parent(
    total_quantity: int = 1000,
    window_minutes: int = 30,
    algo_type: str = "twap",
    symbol: str = "SPY",
    side: str = "buy",
) -> AlgoParentOrder:
    start = datetime(2026, 4, 6, 9, 30, 0)
    end = start + timedelta(minutes=window_minutes)
    return AlgoParentOrder(
        parent_order_id="P-TEST-001",
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        algo_type=algo_type,
        start_time=start,
        end_time=end,
        arrival_price=450.00,
    )


# ---------------------------------------------------------------------------
# Synthetic volume profile
# ---------------------------------------------------------------------------


class TestSyntheticVolumeProfile:
    def test_sums_to_one(self) -> None:
        profile = synthetic_volume_profile(12)
        assert pytest.approx(sum(profile), abs=1e-9) == 1.0

    def test_u_shape_ends_higher_than_middle(self) -> None:
        profile = synthetic_volume_profile(10)
        mid = len(profile) // 2
        assert profile[0] > profile[mid]
        assert profile[-1] > profile[mid]

    def test_single_bucket(self) -> None:
        assert synthetic_volume_profile(1) == [1.0]

    def test_two_buckets(self) -> None:
        profile = synthetic_volume_profile(2)
        assert len(profile) == 2
        assert pytest.approx(sum(profile), abs=1e-9) == 1.0

    def test_zero_buckets(self) -> None:
        assert synthetic_volume_profile(0) == []


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------


class TestPlanTwapChildren:
    def test_correct_number_of_children(self) -> None:
        parent = _make_parent(total_quantity=1000, window_minutes=30)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        # 30 min / 5 min = 6 buckets
        assert len(children) == 6

    def test_total_quantity_preserved(self) -> None:
        parent = _make_parent(total_quantity=1000, window_minutes=30)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        assert sum(c.target_quantity for c in children) == 1000

    def test_approximate_equal_distribution(self) -> None:
        parent = _make_parent(total_quantity=1000, window_minutes=30)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        base = 1000 // 6  # ~166
        for child in children:
            # Within 20% of base to allow for variation + remainder
            assert child.target_quantity > 0
            assert abs(child.target_quantity - base) < base * 0.30

    def test_quantity_variation_exists(self) -> None:
        """Children should not all have exactly the same quantity (randomisation)."""
        parent = _make_parent(total_quantity=1000, window_minutes=30)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        quantities = [c.target_quantity for c in children]
        # With 6 children and variation, not all should be identical
        assert len(set(quantities)) > 1

    def test_jitter_within_bounds(self) -> None:
        """Scheduled times should differ from exact bucket starts but stay in window."""
        parent = _make_parent(total_quantity=1000, window_minutes=30)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        bucket_seconds = 5 * 60
        max_jitter = bucket_seconds * 0.20

        for i, child in enumerate(children):
            exact_start = parent.start_time + timedelta(seconds=i * bucket_seconds)
            delta = abs((child.scheduled_time - exact_start).total_seconds())
            assert delta <= max_jitter + 0.01  # small float tolerance
            # Must be within execution window
            assert child.scheduled_time >= parent.start_time
            assert child.scheduled_time <= parent.end_time

    def test_scheduled_times_span_window(self) -> None:
        parent = _make_parent(total_quantity=1000, window_minutes=30)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        times = sorted(c.scheduled_time for c in children)
        # First child near start, last child near end (bucket + jitter)
        assert (times[0] - parent.start_time).total_seconds() < 5 * 60
        # Last bucket starts 5 min before end; with 20% jitter it can be up to 6 min away
        assert (parent.end_time - times[-1]).total_seconds() < 7 * 60

    def test_odd_quantity_remainder(self) -> None:
        """Indivisible quantity still sums correctly."""
        parent = _make_parent(total_quantity=7, window_minutes=15)
        children = plan_twap_children(parent, bucket_minutes=5, rng=random.Random(42))
        assert len(children) == 3
        assert sum(c.target_quantity for c in children) == 7


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


class TestPlanVwapChildren:
    def test_total_quantity_preserved(self) -> None:
        parent = _make_parent(total_quantity=1000, window_minutes=30, algo_type="vwap")
        children = plan_vwap_children(parent, bucket_minutes=5, rng=random.Random(42))
        assert sum(c.target_quantity for c in children) == 1000

    def test_correct_number_of_children(self) -> None:
        parent = _make_parent(total_quantity=1000, window_minutes=30, algo_type="vwap")
        children = plan_vwap_children(parent, bucket_minutes=5, rng=random.Random(42))
        assert len(children) == 6

    def test_synthetic_profile_u_shape(self) -> None:
        """With no explicit profile, open/close buckets should be larger than midday."""
        parent = _make_parent(total_quantity=1000, window_minutes=30, algo_type="vwap")
        children = plan_vwap_children(parent, bucket_minutes=5, rng=random.Random(42))
        quantities = [c.target_quantity for c in children]
        mid = len(quantities) // 2
        # First and last should be larger than the middle bucket
        assert quantities[0] > quantities[mid]
        assert quantities[-1] > quantities[mid]

    def test_custom_volume_profile(self) -> None:
        """Explicit profile weights should be respected proportionally."""
        parent = _make_parent(total_quantity=100, window_minutes=15, algo_type="vwap")
        # 3 buckets: heavily weighted toward bucket 0
        profile = [8.0, 1.0, 1.0]
        children = plan_vwap_children(
            parent, volume_profile=profile, bucket_minutes=5, rng=random.Random(42)
        )
        assert len(children) == 3
        assert sum(c.target_quantity for c in children) == 100
        # Bucket 0 should get ~80% of the quantity
        assert children[0].target_quantity >= 70

    def test_fallback_to_synthetic_when_no_profile(self) -> None:
        parent = _make_parent(total_quantity=500, window_minutes=30, algo_type="vwap")
        children = plan_vwap_children(parent, volume_profile=None, rng=random.Random(42))
        # Should still produce valid children
        assert len(children) == 6
        assert sum(c.target_quantity for c in children) == 500

    def test_profile_shorter_than_buckets_padded(self) -> None:
        """If profile has fewer entries than buckets, it should be padded."""
        parent = _make_parent(total_quantity=100, window_minutes=30, algo_type="vwap")
        profile = [1.0, 2.0]  # Only 2 entries for 6 buckets
        children = plan_vwap_children(
            parent, volume_profile=profile, bucket_minutes=5, rng=random.Random(42)
        )
        assert len(children) == 6
        assert sum(c.target_quantity for c in children) == 100


# ---------------------------------------------------------------------------
# Invariants (shared across TWAP and VWAP)
# ---------------------------------------------------------------------------


class TestChildOrderInvariants:
    @pytest.fixture(params=["twap", "vwap"])
    def children(self, request: pytest.FixtureRequest) -> list[ChildOrder]:
        parent = _make_parent(
            total_quantity=500, window_minutes=30, algo_type=request.param
        )
        rng = random.Random(42)
        if request.param == "twap":
            return plan_twap_children(parent, bucket_minutes=5, rng=rng)
        return plan_vwap_children(parent, bucket_minutes=5, rng=rng)

    def test_all_statuses_pending(self, children: list[ChildOrder]) -> None:
        for child in children:
            assert child.status == "pending"

    def test_parent_id_matches(self, children: list[ChildOrder]) -> None:
        for child in children:
            assert child.parent_id == "P-TEST-001"

    def test_child_id_format(self, children: list[ChildOrder]) -> None:
        for i, child in enumerate(children):
            expected = f"P-TEST-001-C{i + 1:03d}"
            assert child.child_id == expected

    def test_all_quantities_positive(self, children: list[ChildOrder]) -> None:
        for child in children:
            assert child.target_quantity > 0

    def test_filled_quantity_zero(self, children: list[ChildOrder]) -> None:
        for child in children:
            assert child.filled_quantity == 0

    def test_no_broker_order_id(self, children: list[ChildOrder]) -> None:
        for child in children:
            assert child.broker_order_id is None
