"""Tests for the weekend parallel research coordinator."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from quantstack.research.weekend_runner import (
    StreamResult,
    WeekendResearchState,
    build_weekend_graph,
    fan_out_streams,
    is_weekend_window,
    run_stream,
    synthesis_node,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Time window tests
# ---------------------------------------------------------------------------

class TestWeekendWindow:
    """Runner time window: Friday 20:00 ET to Monday 04:00 ET."""

    def test_friday_before_window(self):
        friday_19 = datetime(2026, 4, 3, 19, 59, tzinfo=ET)  # Friday 19:59
        assert not is_weekend_window(friday_19)

    def test_friday_at_start(self):
        friday_20 = datetime(2026, 4, 3, 20, 0, tzinfo=ET)  # Friday 20:00
        assert is_weekend_window(friday_20)

    def test_saturday(self):
        saturday = datetime(2026, 4, 4, 12, 0, tzinfo=ET)
        assert is_weekend_window(saturday)

    def test_sunday(self):
        sunday = datetime(2026, 4, 5, 15, 0, tzinfo=ET)
        assert is_weekend_window(sunday)

    def test_monday_before_end(self):
        monday_3am = datetime(2026, 4, 6, 3, 59, tzinfo=ET)
        assert is_weekend_window(monday_3am)

    def test_monday_at_end(self):
        monday_4am = datetime(2026, 4, 6, 4, 0, tzinfo=ET)
        assert not is_weekend_window(monday_4am)

    def test_wednesday(self):
        wednesday = datetime(2026, 4, 1, 12, 0, tzinfo=ET)
        assert not is_weekend_window(wednesday)

    def test_naive_datetime_treated_as_et(self):
        """Naive datetimes should be interpreted as ET."""
        saturday_naive = datetime(2026, 4, 4, 12, 0)
        assert is_weekend_window(saturday_naive)


# ---------------------------------------------------------------------------
# State model tests
# ---------------------------------------------------------------------------

class TestStreamResult:
    """Each stream has isolated state via StreamResult model validation."""

    def test_valid_stream_result(self):
        sr = StreamResult(
            stream_name="factor_mining",
            findings=[{"type": "factor_candidate", "ic_estimate": 0.04}],
            experiments_run=1,
            cost_usd=0.50,
            errors=[],
        )
        assert sr.stream_name == "factor_mining"
        assert len(sr.findings) == 1
        assert sr.cost_usd == 0.50

    def test_defaults(self):
        sr = StreamResult(stream_name="test")
        assert sr.findings == []
        assert sr.experiments_run == 0
        assert sr.cost_usd == 0.0
        assert sr.errors == []

    def test_error_isolation(self):
        """Errors in one StreamResult don't affect others."""
        sr1 = StreamResult(stream_name="a", errors=["boom"])
        sr2 = StreamResult(stream_name="b", errors=[])
        assert sr1.errors == ["boom"]
        assert sr2.errors == []


class TestWeekendResearchState:
    """Results merge via reducer (list append)."""

    def test_default_budget(self):
        state = WeekendResearchState()
        assert state.budget_remaining == 50.0

    def test_results_list_starts_empty(self):
        state = WeekendResearchState()
        assert state.weekend_research_results == []


# ---------------------------------------------------------------------------
# Fan-out tests
# ---------------------------------------------------------------------------

class TestFanOut:
    """Weekend runner spawns exactly 4 parallel streams."""

    def test_fan_out_returns_4_sends(self):
        state = WeekendResearchState()
        sends = fan_out_streams(state)
        assert len(sends) == 4

    def test_fan_out_send_types(self):
        from langgraph.types import Send

        state = WeekendResearchState()
        sends = fan_out_streams(state)
        for s in sends:
            assert isinstance(s, Send)

    def test_fan_out_stream_names(self):
        state = WeekendResearchState()
        sends = fan_out_streams(state)
        names = [s.arg["stream_name"] for s in sends]
        assert set(names) == {
            "factor_mining",
            "regime_research",
            "cross_asset_signals",
            "portfolio_construction",
        }

    def test_fan_out_passes_budget(self):
        state = WeekendResearchState(budget_remaining=25.0)
        sends = fan_out_streams(state)
        for s in sends:
            assert s.arg["budget_remaining"] == 25.0


# ---------------------------------------------------------------------------
# Individual stream tests
# ---------------------------------------------------------------------------

class TestFactorMining:
    """Factor mining extracts testable factor."""

    def test_returns_findings(self):
        result = run_stream({"stream_name": "factor_mining"})
        results = result["weekend_research_results"]
        assert len(results) == 1
        sr = StreamResult(**results[0])
        assert sr.stream_name == "factor_mining"
        assert len(sr.findings) >= 1
        assert sr.findings[0]["type"] == "factor_candidate"


class TestRegimeResearch:
    """Regime research labels regimes."""

    def test_returns_regime_labels(self):
        result = run_stream({"stream_name": "regime_research"})
        results = result["weekend_research_results"]
        sr = StreamResult(**results[0])
        assert sr.stream_name == "regime_research"
        assert len(sr.findings) >= 1
        assert sr.findings[0]["type"] == "regime_label"


class TestCrossAssetSignals:
    """Cross-asset computes lead-lag."""

    def test_returns_lead_lag(self):
        result = run_stream({"stream_name": "cross_asset_signals"})
        results = result["weekend_research_results"]
        sr = StreamResult(**results[0])
        assert sr.stream_name == "cross_asset_signals"
        assert len(sr.findings) >= 1
        assert sr.findings[0]["type"] == "lead_lag_pair"
        assert "lead" in sr.findings[0]
        assert "lag" in sr.findings[0]


class TestPortfolioConstruction:
    """Portfolio construction compares optimizers."""

    def test_returns_optimizer_comparison(self):
        result = run_stream({"stream_name": "portfolio_construction"})
        results = result["weekend_research_results"]
        sr = StreamResult(**results[0])
        assert sr.stream_name == "portfolio_construction"
        assert len(sr.findings) >= 1
        assert sr.findings[0]["type"] == "optimizer_comparison"
        assert "optimizers" in sr.findings[0]


# ---------------------------------------------------------------------------
# Error isolation tests
# ---------------------------------------------------------------------------

class TestErrorIsolation:
    """Individual stream failure doesn't crash the runner."""

    def test_unknown_stream_returns_error(self):
        result = run_stream({"stream_name": "nonexistent"})
        sr = StreamResult(**result["weekend_research_results"][0])
        assert len(sr.errors) == 1
        assert "Unknown stream" in sr.errors[0]

    def test_crashing_stream_returns_error(self):
        with patch(
            "quantstack.research.weekend_runner._STREAM_CONFIGS",
            [{"stream_name": "boom", "runner": lambda s: 1 / 0}],
        ):
            # Re-import not needed; run_stream reads _STREAM_CONFIGS at call time
            # but we patched the module-level list, so we call run_stream directly
            from quantstack.research.weekend_runner import run_stream as rs

            result = rs({"stream_name": "boom"})
            sr = StreamResult(**result["weekend_research_results"][0])
            assert len(sr.errors) == 1
            assert "crashed" in sr.errors[0]


# ---------------------------------------------------------------------------
# Synthesis tests
# ---------------------------------------------------------------------------

class TestSynthesis:
    """Synthesis produces prioritized research tasks."""

    def test_synthesis_creates_tasks_from_results(self):
        state = WeekendResearchState(
            weekend_research_results=[
                {
                    "stream_name": "factor_mining",
                    "findings": [{"type": "factor", "status": "pending_validation"}],
                    "experiments_run": 1,
                    "cost_usd": 2.0,
                    "errors": [],
                },
                {
                    "stream_name": "regime_research",
                    "findings": [{"type": "regime", "status": "pending_validation"}],
                    "experiments_run": 1,
                    "cost_usd": 3.0,
                    "errors": [],
                },
            ],
        )
        result = synthesis_node(state)
        tasks = result["synthesis_tasks"]
        assert len(tasks) == 2
        assert all(t["action"] == "validate_and_backtest" for t in tasks)
        assert all(t["priority"] == "high" for t in tasks)

    def test_synthesis_deducts_cost(self):
        state = WeekendResearchState(
            budget_remaining=50.0,
            weekend_research_results=[
                {
                    "stream_name": "test",
                    "findings": [],
                    "experiments_run": 0,
                    "cost_usd": 10.0,
                    "errors": [],
                },
            ],
        )
        result = synthesis_node(state)
        assert result["budget_remaining"] == 40.0

    def test_synthesis_with_empty_results(self):
        state = WeekendResearchState(weekend_research_results=[])
        result = synthesis_node(state)
        assert result["synthesis_tasks"] == []

    def test_synthesis_logs_errors(self):
        state = WeekendResearchState(
            weekend_research_results=[
                {
                    "stream_name": "broken",
                    "findings": [],
                    "experiments_run": 0,
                    "cost_usd": 0.0,
                    "errors": ["something went wrong"],
                },
            ],
        )
        with patch("quantstack.research.weekend_runner.logger") as mock_logger:
            synthesis_node(state)
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Graph construction test
# ---------------------------------------------------------------------------

class TestBuildGraph:
    """build_weekend_graph produces a valid StateGraph."""

    def test_graph_compiles(self):
        graph = build_weekend_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_has_expected_nodes(self):
        graph = build_weekend_graph()
        # StateGraph nodes dict includes our named nodes
        assert "run_stream" in graph.nodes
        assert "synthesis" in graph.nodes
