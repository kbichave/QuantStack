# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for AgentSkill IC/ICIR metrics and SkillTracker.record_ic() — Sprint 2.

Uses a PostgreSQL KnowledgeStore connection via pg_conn().
"""

from __future__ import annotations

import pytest
from quantstack.db import pg_conn
from quantstack.learning.skill_tracker import AgentSkill, SkillTracker

# ---------------------------------------------------------------------------
# AgentSkill: IC / ICIR property tests (no DB required)
# ---------------------------------------------------------------------------


class TestAgentSkillICProperties:
    def test_ic_empty_observations(self):
        skill = AgentSkill(agent_id="test_agent")
        assert skill.ic == 0.0

    def test_ic_mean_computed(self):
        skill = AgentSkill(agent_id="a", ic_observations=[0.1, 0.2, 0.3])
        assert abs(skill.ic - 0.2) < 1e-9

    def test_ic_std_less_than_two_obs(self):
        skill = AgentSkill(agent_id="a", ic_observations=[0.1])
        assert skill.ic_std == 0.0

    def test_icir_zero_when_no_variance(self):
        # All same IC → std ≈ 0 → ICIR = 0
        skill = AgentSkill(agent_id="a", ic_observations=[0.1, 0.1, 0.1])
        assert skill.icir == 0.0

    def test_icir_positive_for_consistent_signal(self):
        # IC observations with positive mean and some variance
        skill = AgentSkill(
            agent_id="a",
            ic_observations=[
                0.05,
                0.06,
                0.04,
                0.07,
                0.05,
                0.06,
                0.04,
                0.05,
                0.06,
                0.07,
            ],
        )
        assert skill.icir > 0

    def test_rolling_ic_window(self):
        obs = list(range(40))  # 0..39
        skill = AgentSkill(agent_id="a", ic_observations=[float(x) for x in obs])
        # Last 30 observations are 10..39, mean = 24.5
        assert abs(skill.rolling_ic(30) - 24.5) < 1e-6

    def test_ic_trend_insufficient_data(self):
        skill = AgentSkill(agent_id="a", ic_observations=[0.1] * 15)
        assert skill.ic_trend() == "INSUFFICIENT_DATA"

    def test_ic_trend_improving(self):
        # Prior 10 = [0.01..0.10], Recent 10 = [0.20..0.29] — recent >> prior
        prior = [0.01 * i for i in range(1, 11)]
        recent = [0.20 + 0.01 * i for i in range(10)]
        skill = AgentSkill(agent_id="a", ic_observations=prior + recent)
        assert skill.ic_trend() == "IMPROVING"

    def test_ic_trend_decaying(self):
        # Prior 10 = [0.20..0.29], Recent 10 = [0.01..0.10] — recent << prior
        prior = [0.20 + 0.01 * i for i in range(10)]
        recent = [0.01 * i for i in range(1, 11)]
        skill = AgentSkill(agent_id="a", ic_observations=prior + recent)
        assert skill.ic_trend() == "DECAYING"

    def test_ic_trend_stable(self):
        # Same value in both windows
        obs = [0.05] * 30
        skill = AgentSkill(agent_id="a", ic_observations=obs)
        assert skill.ic_trend() == "STABLE"


# ---------------------------------------------------------------------------
# SkillTracker.record_ic — persistence to PostgreSQL
# ---------------------------------------------------------------------------


class _MinimalKnowledgeStore:
    """Minimal KnowledgeStore stand-in backed by a PostgreSQL connection."""

    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def tracker() -> SkillTracker:
    with pg_conn() as conn:
        store = _MinimalKnowledgeStore(conn)
        yield SkillTracker(store)


class TestSkillTrackerRecordIC:
    def test_record_ic_updates_observations(self, tracker):
        skill = tracker.record_ic("agent_a", 0.05)
        assert len(skill.ic_observations) == 1
        assert abs(skill.ic_observations[0] - 0.05) < 1e-9

    def test_record_ic_multiple(self, tracker):
        for v in [0.1, 0.2, 0.3]:
            tracker.record_ic("agent_b", v)
        skill = tracker._get_skill("agent_b")
        assert len(skill.ic_observations) == 3

    def test_record_ic_nan_ignored(self, tracker):
        skill = tracker.record_ic("agent_c", float("nan"))
        assert len(skill.ic_observations) == 0

    def test_record_ic_inf_ignored(self, tracker):
        skill = tracker.record_ic("agent_d", float("inf"))
        assert len(skill.ic_observations) == 0

    def test_ic_mean_after_recording(self, tracker):
        for v in [0.1, 0.3]:
            tracker.record_ic("agent_e", v)
        skill = tracker._get_skill("agent_e")
        assert abs(skill.ic - 0.2) < 1e-9

    def test_icir_computable_after_observations(self, tracker):
        for v in [0.05, 0.06, 0.04, 0.07, 0.05]:
            tracker.record_ic("agent_f", v)
        skill = tracker._get_skill("agent_f")
        # Should have positive ICIR given consistent positive IC values
        assert skill.icir >= 0.0


# ---------------------------------------------------------------------------
# SkillTracker.needs_retraining IC criterion
# ---------------------------------------------------------------------------


class TestNeedsRetrainingIC:
    def test_no_observations_returns_false(self, tracker):
        assert tracker.needs_retraining("new_agent") is False

    def test_negative_rolling_ic_triggers_retraining(self, tracker):
        """Rolling IC < 0 with ≥ 20 observations → needs retraining."""
        for _ in range(25):
            tracker.record_ic("bad_agent", -0.05)
        assert tracker.needs_retraining("bad_agent") is True

    def test_positive_ic_does_not_trigger_retraining(self, tracker):
        for _ in range(25):
            tracker.record_ic("good_agent", 0.05)
        assert tracker.needs_retraining("good_agent") is False

    def test_fewer_than_20_obs_does_not_trigger_ic_criterion(self, tracker):
        for _ in range(15):
            tracker.record_ic("new_agent2", -0.05)
        # Not enough observations for IC criterion
        assert tracker.needs_retraining("new_agent2") is False
