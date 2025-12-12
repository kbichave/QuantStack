# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Policy Store - Manages strategy policy snapshots.

Tracks the evolution of strategy weights and thresholds over time,
supporting the learning/adaptation cycle of QuantArena.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class PolicySnapshot(BaseModel):
    """A snapshot of strategy policy at a point in time."""

    effective_date: date
    pod_weights: Dict[str, float] = Field(
        default_factory=dict, description="Weight assigned to each strategy pod"
    )
    thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Risk/entry thresholds"
    )
    comment: str = ""

    class Config:
        arbitrary_types_allowed = True


class PolicyStore:
    """
    Manages policy evolution for historical simulation.

    Responsibilities:
    - Store policy snapshots with effective dates
    - Retrieve current policy as of a given date
    - Determine when policy updates should occur

    Usage:
        store = PolicyStore(knowledge_store)

        # Get current policy
        policy = store.get_current(date.today())

        # Check if update needed
        if store.should_update(current_date, "monthly"):
            new_policy = evolution_agent.propose_update(...)
            store.save(new_policy)
    """

    def __init__(self, knowledge_store: Any):
        """
        Initialize policy store.

        Args:
            knowledge_store: KnowledgeStore instance for persistence
        """
        self.store = knowledge_store
        self._cache: Dict[str, PolicySnapshot] = {}
        self._last_update_date: Optional[date] = None

        logger.info("PolicyStore initialized")

    def get_current(self, as_of_date: date) -> Optional[PolicySnapshot]:
        """
        Get the current policy as of a date.

        Args:
            as_of_date: Date to get policy for

        Returns:
            PolicySnapshot or None if no policy exists
        """
        # Check cache
        cache_key = as_of_date.isoformat()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query from store
        if self.store is None:
            return self._default_policy(as_of_date)

        try:
            snapshot_dict = self.store.get_latest_policy(as_of_date)

            if snapshot_dict is None:
                return self._default_policy(as_of_date)

            # Convert to PolicySnapshot
            snapshot = PolicySnapshot(
                effective_date=snapshot_dict.get("effective_date", as_of_date),
                pod_weights=snapshot_dict.get("pod_weights", {}),
                thresholds=snapshot_dict.get("thresholds", {}),
                comment=snapshot_dict.get("comment", ""),
            )

            self._cache[cache_key] = snapshot
            return snapshot

        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            return self._default_policy(as_of_date)

    def _default_policy(self, as_of_date: date) -> PolicySnapshot:
        """Create default policy when none exists."""
        return PolicySnapshot(
            effective_date=as_of_date,
            pod_weights={
                "trend_following": 0.5,
                "mean_reversion": 0.5,
            },
            thresholds={
                "min_confidence": 0.5,
                "max_position_pct": 0.20,
            },
            comment="Default policy",
        )

    def should_update(self, current_date: date, frequency: str) -> bool:
        """
        Check if policy should be updated.

        Args:
            current_date: Current simulation date
            frequency: Update frequency ("monthly", "quarterly", "never")

        Returns:
            True if update should occur
        """
        if frequency == "never":
            return False

        # First day of simulation always updates
        if self._last_update_date is None:
            return True

        if frequency == "monthly":
            # Update on first trading day of month
            return current_date.month != self._last_update_date.month

        elif frequency == "quarterly":
            # Update on first trading day of quarter
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (self._last_update_date.month - 1) // 3
            return (
                current_quarter != last_quarter
                or current_date.year != self._last_update_date.year
            )

        return False

    def save(self, snapshot: PolicySnapshot) -> None:
        """
        Save a policy snapshot.

        Args:
            snapshot: PolicySnapshot to save
        """
        if self.store is None:
            logger.warning("No store configured, policy not persisted")
            return

        try:
            self.store.save_policy_snapshot(
                {
                    "effective_date": snapshot.effective_date,
                    "pod_weights": snapshot.pod_weights,
                    "thresholds": snapshot.thresholds,
                    "comment": snapshot.comment,
                }
            )

            # Update tracking
            self._last_update_date = snapshot.effective_date

            # Clear cache
            self._cache.clear()

            logger.info(
                f"Policy saved: {snapshot.effective_date}, "
                f"weights={snapshot.pod_weights}"
            )

        except Exception as e:
            logger.error(f"Failed to save policy: {e}")

    def get_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[PolicySnapshot]:
        """
        Get policy evolution history.

        Args:
            start_date: Start of range
            end_date: End of range

        Returns:
            List of PolicySnapshots in chronological order
        """
        if self.store is None:
            return []

        try:
            snapshots_dict = self.store.load_policy_snapshots(
                start_date=start_date,
                end_date=end_date,
            )

            return [
                PolicySnapshot(
                    effective_date=s.get("effective_date"),
                    pod_weights=s.get("pod_weights", {}),
                    thresholds=s.get("thresholds", {}),
                    comment=s.get("comment", ""),
                )
                for s in snapshots_dict
            ]

        except Exception as e:
            logger.error(f"Failed to load policy history: {e}")
            return []

    def get_pod_weight(self, pod_name: str, as_of_date: date) -> float:
        """
        Get weight for a specific pod.

        Args:
            pod_name: Name of strategy pod
            as_of_date: Date to get weight for

        Returns:
            Weight (0.0-1.0) or 0.5 if not found
        """
        policy = self.get_current(as_of_date)
        if policy is None:
            return 0.5

        return policy.pod_weights.get(pod_name, 0.5)
