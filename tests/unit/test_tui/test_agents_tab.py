"""Tests for Agents tab widgets."""
from datetime import datetime

from quantstack.tui.queries.agents import (
    AgentSkill,
    CalibrationRecord,
    CycleHistory,
    GraphActivity,
)
from quantstack.tui.widgets.agents import AgentScorecardWidget, GraphActivityWidget


class TestGraphActivityWidget:
    def test_renders_activity(self):
        w = GraphActivityWidget()
        now = datetime.now()
        w.update_view({
            "activity": [
                GraphActivity("research", "scan", "scanner", 10, now, 5),
                GraphActivity("trading", "monitor", "risk_mgr", 5, now, 3),
            ],
            "history": [
                CycleHistory("research", 9, 45.0, "scanner", 12),
            ],
        })

    def test_handles_empty(self):
        w = GraphActivityWidget()
        w.update_view({"activity": [], "history": []})

    def test_handles_none(self):
        w = GraphActivityWidget()
        w.update_view(None)


class TestAgentScorecardWidget:
    def test_renders_skills(self):
        w = AgentScorecardWidget()
        w.update_view({
            "skills": [AgentSkill("scanner", 0.72, 0.65, 120.0, 0.15, "improving")],
            "calibration": [CalibrationRecord("scanner", 0.8, 0.6, True)],
            "prompts": [],
        })

    def test_flags_overconfident(self):
        w = AgentScorecardWidget()
        w.update_view({
            "skills": [AgentSkill("scanner", 0.72, 0.65, 120.0, 0.15, "stable")],
            "calibration": [CalibrationRecord("scanner", 0.9, 0.5, True)],
            "prompts": [],
        })

    def test_handles_empty(self):
        w = AgentScorecardWidget()
        w.update_view({"skills": [], "calibration": [], "prompts": []})

    def test_handles_none(self):
        w = AgentScorecardWidget()
        w.update_view(None)
