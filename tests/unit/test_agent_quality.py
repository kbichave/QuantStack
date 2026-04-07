"""Tests for agent quality evaluation — degradation detection and confidence formatting."""

from __future__ import annotations

import pytest

from quantstack.learning.agent_quality import (
    AgentQualityReport,
    evaluate_agent_quality,
    format_agent_confidence,
    get_degraded_agents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skill(signal_count: int, winning_signals: int, confidence_adjustment: float = 1.0) -> dict:
    return {
        "signal_count": signal_count,
        "winning_signals": winning_signals,
        "confidence_adjustment": confidence_adjustment,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWinRateComputation:
    def test_win_rate_computation(self):
        """Agent with 30 signals and 18 wins should have win_rate == 0.6."""
        skills = {"alpha_agent": _skill(30, 18)}
        reports = evaluate_agent_quality(skills, min_trades=30, alert_threshold=0.40)
        assert len(reports) == 1
        assert reports[0].win_rate == 0.6


class TestDegradationDetection:
    def test_degradation_event_below_40_pct(self):
        """Agent with 33% win rate (10/30) should be flagged as degraded."""
        skills = {"weak_agent": _skill(30, 10)}
        reports = evaluate_agent_quality(skills, min_trades=30, alert_threshold=0.40)
        assert len(reports) == 1
        assert reports[0].is_degraded is True

    def test_no_alert_at_40_pct(self):
        """Agent with exactly 40% win rate (12/30) should NOT be flagged."""
        skills = {"ok_agent": _skill(30, 12)}
        reports = evaluate_agent_quality(skills, min_trades=30, alert_threshold=0.40)
        assert len(reports) == 1
        assert reports[0].is_degraded is False


class TestResearchTaskContext:
    def test_research_task_queued_on_degradation(self):
        """Degraded agent list includes context needed for research task queuing."""
        skills = {"weak_agent": _skill(30, 10)}
        degraded = get_degraded_agents(skills, min_trades=30, alert_threshold=0.40)
        assert len(degraded) == 1
        entry = degraded[0]
        assert entry["agent_id"] == "weak_agent"
        assert entry["win_rate"] == pytest.approx(10 / 30, abs=0.001)
        assert entry["signal_count"] == 30
        assert entry["task_type"] == "agent_prompt_investigation"


class TestColdStart:
    def test_cold_start_fewer_than_30_trades(self):
        """Agent with only 15 signals (even at 20% win rate) should NOT be evaluated."""
        skills = {"new_agent": _skill(15, 3)}
        reports = evaluate_agent_quality(skills, min_trades=30, alert_threshold=0.40)
        assert len(reports) == 0
        degraded = get_degraded_agents(skills, min_trades=30, alert_threshold=0.40)
        assert len(degraded) == 0


class TestFormatAgentConfidence:
    def test_format_agent_confidence_with_data(self):
        """Format function should list agents with their confidence label."""
        skills = {
            "alpha": _skill(50, 30, confidence_adjustment=1.2),
            "beta": _skill(50, 25, confidence_adjustment=0.9),
            "gamma": _skill(50, 15, confidence_adjustment=0.6),
        }
        result = format_agent_confidence(skills, min_signals=5)
        assert "alpha" in result
        assert "reliable" in result
        assert "beta" in result
        assert "cautious" in result
        assert "gamma" in result
        assert "degraded" in result
        assert result.startswith("Agent confidence:")

    def test_format_agent_confidence_empty(self):
        """Empty skills dict returns empty string."""
        assert format_agent_confidence({}, min_signals=5) == ""
