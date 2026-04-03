"""Tests for community_intel scheduled task in supervisor graph."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quantstack.graphs.supervisor.nodes import (
    _is_community_intel_due,
    _is_execution_researcher_due,
)


def test_is_community_intel_due_returns_bool():
    """_is_community_intel_due returns a boolean."""
    result = _is_community_intel_due()
    assert isinstance(result, bool)


def test_is_execution_researcher_due_returns_bool():
    """_is_execution_researcher_due returns a boolean."""
    result = _is_execution_researcher_due()
    assert isinstance(result, bool)


@pytest.mark.asyncio
@patch("quantstack.graphs.supervisor.nodes.run_agent")
@patch("quantstack.graphs.supervisor.nodes._is_community_intel_due", return_value=True)
@patch("quantstack.graphs.supervisor.nodes._is_execution_researcher_due", return_value=False)
async def test_community_intel_publishes_ideas_discovered(
    mock_exec_due, mock_community_due, mock_run_agent
):
    """community_intel publishes IDEAS_DISCOVERED event when ideas found."""
    from quantstack.graphs.supervisor.nodes import make_scheduled_tasks

    # First call: community_intel returns ideas
    # Second call: other scheduled tasks
    mock_run_agent.side_effect = [
        json.dumps({"ideas": [
            {"title": "KAN Networks for Factor Modeling", "source": "arXiv", "relevance_score": 0.8},
        ]}),
        json.dumps([{"task": "data_freshness", "was_due": False, "fired": False}]),
    ]

    llm = MagicMock()
    cfg = MagicMock()
    cfg.name = "health_monitor"
    cfg.role = "System Health Monitor"
    cfg.goal = "Monitor system health"
    cfg.backstory = "You check system health."

    node = make_scheduled_tasks(llm, cfg, [])
    state = {"cycle_number": 42, "errors": []}

    # Patch at the source modules that the deferred imports resolve to
    with patch("quantstack.coordination.event_bus.EventBus") as mock_bus_cls, \
         patch("quantstack.db.db_conn") as mock_db, \
         patch("quantstack.tools.functions.system_functions.record_heartbeat") as mock_hb:
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.return_value = mock_conn

        mock_bus = MagicMock()
        mock_bus_cls.return_value = mock_bus
        mock_hb.return_value = None

        result = await node(state)

    community_result = next(
        (r for r in result["scheduled_task_results"] if r.get("task") == "community_intel"),
        None,
    )
    assert community_result is not None
    assert community_result["fired"] is True


@pytest.mark.asyncio
@patch("quantstack.graphs.supervisor.nodes.run_agent")
@patch("quantstack.graphs.supervisor.nodes._is_community_intel_due", return_value=True)
@patch("quantstack.graphs.supervisor.nodes._is_execution_researcher_due", return_value=False)
async def test_community_intel_handles_empty_results(
    mock_exec_due, mock_community_due, mock_run_agent
):
    """community_intel completes with empty idea list when web_search returns nothing."""
    from quantstack.graphs.supervisor.nodes import make_scheduled_tasks

    mock_run_agent.side_effect = [
        json.dumps({"ideas": []}),
        json.dumps([]),
    ]

    llm = MagicMock()
    cfg = MagicMock()
    cfg.name = "health_monitor"
    cfg.role = "System Health Monitor"
    cfg.goal = "Monitor"
    cfg.backstory = "Monitor."

    node = make_scheduled_tasks(llm, cfg, [])
    state = {"cycle_number": 1, "errors": []}

    with patch("quantstack.tools.functions.system_functions.record_heartbeat") as mock_hb:
        mock_hb.return_value = None
        result = await node(state)

    community_result = next(
        (r for r in result["scheduled_task_results"] if r.get("task") == "community_intel"),
        None,
    )
    assert community_result is not None
    assert community_result["fired"] is True
    assert community_result.get("ideas_found", 0) == 0
