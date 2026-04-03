"""Tests for execution_researcher scheduled task in supervisor graph."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quantstack.graphs.supervisor.nodes import _is_execution_researcher_due


def test_is_execution_researcher_due_returns_bool():
    """execution_researcher due check returns a boolean."""
    result = _is_execution_researcher_due()
    assert isinstance(result, bool)


@pytest.mark.asyncio
@patch("quantstack.graphs.supervisor.nodes.run_agent")
@patch("quantstack.graphs.supervisor.nodes._is_community_intel_due", return_value=False)
@patch("quantstack.graphs.supervisor.nodes._is_execution_researcher_due", return_value=True)
async def test_execution_researcher_fires_monthly(
    mock_exec_due, mock_community_due, mock_run_agent
):
    """execution_researcher fires on 1st business day of month."""
    from quantstack.graphs.supervisor.nodes import make_scheduled_tasks

    mock_run_agent.side_effect = [
        json.dumps({
            "report": "Monthly execution audit: avg shortfall 2.1 bps",
            "avg_shortfall_bps": 2.1,
            "worst_executions": [{"symbol": "AAPL", "shortfall_bps": 8.5}],
            "recommendations": ["Use TWAP for orders > 0.1% ADV"],
        }),
        json.dumps([]),
    ]

    llm = MagicMock()
    cfg = MagicMock()
    cfg.name = "health_monitor"
    cfg.role = "System Health Monitor"
    cfg.goal = "Monitor"
    cfg.backstory = "Monitor."

    node = make_scheduled_tasks(llm, cfg, [])
    state = {"cycle_number": 100, "errors": []}

    with patch("quantstack.knowledge.store.KnowledgeStore") as mock_ks, \
         patch("quantstack.tools.functions.system_functions.record_heartbeat") as mock_hb:
        mock_store = MagicMock()
        mock_ks.return_value = mock_store
        mock_hb.return_value = None

        result = await node(state)

    exec_result = next(
        (r for r in result["scheduled_task_results"] if r.get("task") == "execution_researcher"),
        None,
    )
    assert exec_result is not None
    assert exec_result["fired"] is True


@pytest.mark.asyncio
@patch("quantstack.graphs.supervisor.nodes.run_agent")
@patch("quantstack.graphs.supervisor.nodes._is_community_intel_due", return_value=False)
@patch("quantstack.graphs.supervisor.nodes._is_execution_researcher_due", return_value=True)
async def test_execution_researcher_stores_in_knowledge_base(
    mock_exec_due, mock_community_due, mock_run_agent
):
    """execution quality report stored in knowledge base."""
    from quantstack.graphs.supervisor.nodes import make_scheduled_tasks

    mock_run_agent.side_effect = [
        json.dumps({"report": "Execution quality report", "avg_shortfall_bps": 1.5}),
        json.dumps([]),
    ]

    llm = MagicMock()
    cfg = MagicMock()
    cfg.name = "health_monitor"
    cfg.role = "Monitor"
    cfg.goal = "Monitor"
    cfg.backstory = "Monitor."

    node = make_scheduled_tasks(llm, cfg, [])
    state = {"cycle_number": 100, "errors": []}

    with patch("quantstack.knowledge.store.KnowledgeStore") as mock_ks, \
         patch("quantstack.tools.functions.system_functions.record_heartbeat") as mock_hb:
        mock_store = MagicMock()
        mock_ks.return_value = mock_store
        mock_hb.return_value = None

        result = await node(state)

    # KnowledgeStore was constructed and add_entry was called
    mock_ks.assert_called_once()
    mock_store.add_entry.assert_called_once()


@pytest.mark.asyncio
@patch("quantstack.graphs.supervisor.nodes.run_agent")
@patch("quantstack.graphs.supervisor.nodes._is_community_intel_due", return_value=False)
@patch("quantstack.graphs.supervisor.nodes._is_execution_researcher_due", return_value=True)
async def test_execution_researcher_handles_empty_fills(
    mock_exec_due, mock_community_due, mock_run_agent
):
    """execution_researcher handles no fills gracefully."""
    from quantstack.graphs.supervisor.nodes import make_scheduled_tasks

    mock_run_agent.side_effect = [
        json.dumps({"report": "No fills in period", "avg_shortfall_bps": 0}),
        json.dumps([]),
    ]

    llm = MagicMock()
    cfg = MagicMock()
    cfg.name = "health_monitor"
    cfg.role = "Monitor"
    cfg.goal = "Monitor"
    cfg.backstory = "Monitor."

    node = make_scheduled_tasks(llm, cfg, [])
    state = {"cycle_number": 100, "errors": []}

    with patch("quantstack.knowledge.store.KnowledgeStore") as mock_ks, \
         patch("quantstack.tools.functions.system_functions.record_heartbeat") as mock_hb:
        mock_store = MagicMock()
        mock_ks.return_value = mock_store
        mock_hb.return_value = None

        result = await node(state)

    exec_result = next(
        (r for r in result["scheduled_task_results"] if r.get("task") == "execution_researcher"),
        None,
    )
    assert exec_result is not None
    assert exec_result["fired"] is True
