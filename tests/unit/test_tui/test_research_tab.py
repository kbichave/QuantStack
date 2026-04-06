"""Tests for Research tab widgets."""
from datetime import datetime

from quantstack.tui.queries.research import (
    AlphaProgram,
    Breakthrough,
    BugRecord,
    ConceptDrift,
    MlExperiment,
    ResearchQueueItem,
    ResearchWip,
    TradeReflection,
)
from quantstack.tui.widgets.research import (
    BugStatusWidget,
    DiscoveriesWidget,
    MLExperimentsWidget,
    ReflectionsWidget,
    ResearchQueueWidget,
)


class TestResearchQueueWidget:
    def test_renders_wip_and_queue(self):
        w = ResearchQueueWidget()
        w.update_view({
            "wip": [ResearchWip("AAPL", "swing", "agent1", datetime.now(), 15.0)],
            "queue": [ResearchQueueItem("scan", "pending", "Momentum scan", 1)],
        })

    def test_handles_empty(self):
        w = ResearchQueueWidget()
        w.update_view({"wip": [], "queue": []})

    def test_handles_none(self):
        w = ResearchQueueWidget()
        w.update_view(None)


class TestMLExperimentsWidget:
    def test_renders_experiments(self):
        w = MLExperimentsWidget()
        w.update_view({
            "experiments": [
                MlExperiment("exp-001", datetime.now(), "lgbm", "AAPL", 0.72, 42, "promoted"),
            ],
            "drift": [],
        })

    def test_renders_drift_alerts(self):
        w = MLExperimentsWidget()
        w.update_view({
            "experiments": [
                MlExperiment("exp-001", datetime.now(), "lgbm", "AAPL", 0.72, 42, "promoted"),
            ],
            "drift": [ConceptDrift("AAPL", 0.65, 0.72, 0.07)],
        })

    def test_handles_empty(self):
        w = MLExperimentsWidget()
        w.update_view({"experiments": [], "drift": []})

    def test_handles_none(self):
        w = MLExperimentsWidget()
        w.update_view(None)


class TestDiscoveriesWidget:
    def test_renders_programs_and_breakthroughs(self):
        w = DiscoveriesWidget()
        w.update_view({
            "programs": [AlphaProgram("Stat Arb hypothesis", "active", 5, "Pair correlations found")],
            "breakthroughs": [Breakthrough("vol_skew_ratio", 0.85)],
        })

    def test_handles_empty(self):
        w = DiscoveriesWidget()
        w.update_view({"programs": [], "breakthroughs": []})

    def test_handles_none(self):
        w = DiscoveriesWidget()
        w.update_view(None)


class TestReflectionsWidget:
    def test_renders_reflections(self):
        w = ReflectionsWidget()
        w.update_view([
            TradeReflection("AAPL", 3.5, "Entered too late, missed initial move", datetime.now()),
        ])

    def test_handles_empty(self):
        w = ReflectionsWidget()
        w.update_view([])

    def test_handles_none(self):
        w = ReflectionsWidget()
        w.update_view(None)


class TestBugStatusWidget:
    def test_renders_bugs(self):
        w = BugStatusWidget()
        w.update_view([
            BugRecord("bug-001", "fetch_ohlcv", "open", "Timeout on AAPL", datetime.now()),
        ])

    def test_handles_empty(self):
        w = BugStatusWidget()
        w.update_view([])

    def test_handles_none(self):
        w = BugStatusWidget()
        w.update_view(None)
