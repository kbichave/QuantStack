"""Integration tests for tool search observability in LangFuse."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from quantstack.observability.tracing import (
    trace_tool_search_event,
    trace_tool_search_fallback,
    trace_tool_search_miss_event,
    trace_tool_discovery_event,
)
from quantstack.observability.tool_search_metrics import (
    ToolSearchMetrics,
    compute_tool_search_accuracy,
    _zeroed_metrics,
)


@pytest.fixture
def mock_langfuse():
    """Patch _get_langfuse() to return a mock Langfuse client."""
    mock_lf = MagicMock()
    with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
        yield mock_lf


@pytest.fixture
def mock_langfuse_metrics():
    """Patch _get_langfuse() in tool_search_metrics module."""
    mock_lf = MagicMock()
    with patch("quantstack.observability.tool_search_metrics._get_langfuse", return_value=mock_lf):
        yield mock_lf


class TestToolSearchTraceEmission:
    """Verify LangFuse callback handler emits correct trace events."""

    def test_search_event_logged_with_query_and_count(self, mock_langfuse):
        """A search event creates a trace with query, result count, and tool names."""
        trace_tool_search_event(
            agent_name="daily_planner",
            query="signal brief",
            result_count=3,
            tool_names_returned=["signal_brief", "multi_symbol_brief", "get_sentiment"],
        )
        mock_langfuse.trace.assert_called_once()
        call_kwargs = mock_langfuse.trace.call_args
        assert call_kwargs.kwargs["name"] == "tool_search:search"
        metadata = call_kwargs.kwargs["metadata"]
        assert metadata["agent"] == "daily_planner"
        assert metadata["result_count"] == 3
        assert "tool_search" in call_kwargs.kwargs["tags"]

    def test_discovery_event_logged(self, mock_langfuse):
        """A discovery event creates a trace with the discovered tool name."""
        trace_tool_discovery_event(
            agent_name="risk_analyst",
            tool_name="compute_risk_metrics",
            search_query="risk metrics",
        )
        mock_langfuse.trace.assert_called_once()
        call_kwargs = mock_langfuse.trace.call_args
        assert call_kwargs.kwargs["name"] == "tool_search:discovery"
        assert "discovery" in call_kwargs.kwargs["tags"]

    def test_miss_event_logged(self, mock_langfuse):
        """A miss event creates a trace with query and reason."""
        trace_tool_search_miss_event(
            agent_name="entry_scanner",
            query="nonexistent_tool",
            reason="no results",
        )
        mock_langfuse.trace.assert_called_once()
        call_kwargs = mock_langfuse.trace.call_args
        assert call_kwargs.kwargs["name"] == "tool_search:miss"
        assert "miss" in call_kwargs.kwargs["tags"]

    def test_fallback_event_logged(self, mock_langfuse):
        """A fallback event creates a trace with error and tool count."""
        trace_tool_search_fallback(
            agent_name="daily_planner",
            error="beta header rejected",
            tools_loaded=12,
        )
        mock_langfuse.trace.assert_called_once()
        call_kwargs = mock_langfuse.trace.call_args
        assert call_kwargs.kwargs["name"] == "tool_search:fallback"
        assert "fallback" in call_kwargs.kwargs["tags"]

    def test_trace_noop_when_langfuse_unavailable(self):
        """When _get_langfuse() returns None, trace functions are no-ops."""
        with patch("quantstack.observability.tracing._get_langfuse", return_value=None):
            # Should not raise
            trace_tool_search_event("agent", "query", 0, [])
            trace_tool_discovery_event("agent", "tool", "query")
            trace_tool_search_miss_event("agent", "query", "reason")
            trace_tool_search_fallback("agent", "error", 0)

    def test_trace_handles_langfuse_exception(self, mock_langfuse):
        """If LangFuse raises, trace functions catch and don't propagate."""
        mock_langfuse.trace.side_effect = RuntimeError("langfuse down")
        # Should not raise
        trace_tool_search_event("agent", "query", 0, [])
        trace_tool_discovery_event("agent", "tool", "query")
        trace_tool_search_miss_event("agent", "query", "reason")
        trace_tool_search_fallback("agent", "error", 0)


class TestToolSearchMetrics:
    """Verify accuracy metrics computation from trace data."""

    def test_handles_empty_trace_set(self, mock_langfuse_metrics):
        """Returns zeroed metrics when no traces exist."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_langfuse_metrics.fetch_traces.return_value = mock_result

        metrics = compute_tool_search_accuracy(timedelta(hours=24))
        assert metrics.total_searches == 0
        assert metrics.search_hit_rate == 0.0
        assert metrics.fallback_rate == 0.0

    def test_handles_langfuse_unavailable(self):
        """Returns zeroed metrics when LangFuse is not configured."""
        with patch("quantstack.observability.tool_search_metrics._get_langfuse", return_value=None):
            metrics = compute_tool_search_accuracy(timedelta(hours=24))
        assert metrics.total_searches == 0
        assert isinstance(metrics, ToolSearchMetrics)

    def test_handles_fetch_exception(self, mock_langfuse_metrics):
        """Returns zeroed metrics when fetch_traces raises."""
        mock_langfuse_metrics.fetch_traces.side_effect = RuntimeError("connection refused")
        metrics = compute_tool_search_accuracy(timedelta(hours=24))
        assert metrics.total_searches == 0

    def test_zeroed_metrics_preserves_time_window(self):
        """_zeroed_metrics returns the correct time_window."""
        window = timedelta(hours=6)
        m = _zeroed_metrics(window)
        assert m.time_window == window
        assert m.search_hit_rate == 0.0
        assert m.top_missed_tools == []
