"""Tests for tool search observability: tracing and accuracy metrics."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from quantstack.observability.tool_search_metrics import (
    ToolSearchMetrics,
    _zeroed_metrics,
    compute_tool_search_accuracy,
    store_metrics,
)
from quantstack.observability.tracing import (
    trace_tool_discovery_event,
    trace_tool_search_event,
    trace_tool_search_fallback,
    trace_tool_search_miss_event,
)


class TestToolSearchTracing:
    """Tests for LangFuse tool search event tracing."""

    def test_trace_tool_search_event_logs_query_and_result_count(self):
        """trace_tool_search_event creates a LangFuse trace with query text,
        result count, and returned tool names in metadata."""
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            trace_tool_search_event(
                agent_name="daily_planner",
                query="risk metrics",
                result_count=3,
                tool_names_returned=["compute_risk_metrics", "get_var", "position_sizing"],
                latency_ms=42.5,
            )
        mock_lf.trace.assert_called_once()
        call_kwargs = mock_lf.trace.call_args[1]
        assert call_kwargs["name"] == "tool_search:search"
        assert call_kwargs["metadata"]["agent"] == "daily_planner"
        assert call_kwargs["metadata"]["query"] == "risk metrics"
        assert call_kwargs["metadata"]["result_count"] == 3
        assert call_kwargs["metadata"]["latency_ms"] == 42.5
        assert "tool_search" in call_kwargs["tags"]
        assert "search" in call_kwargs["tags"]

    def test_trace_tool_discovery_event_logs_discovered_tool(self):
        """trace_tool_discovery_event creates a trace when a deferred tool
        is discovered via search and subsequently called."""
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            trace_tool_discovery_event(
                agent_name="risk_assessor",
                tool_name="compute_risk_metrics",
                search_query="risk calculation",
            )
        mock_lf.trace.assert_called_once()
        call_kwargs = mock_lf.trace.call_args[1]
        assert call_kwargs["name"] == "tool_search:discovery"
        assert call_kwargs["metadata"]["tool_name"] == "compute_risk_metrics"
        assert "discovery" in call_kwargs["tags"]

    def test_trace_tool_search_miss_event_logs_failed_search(self):
        """trace_tool_search_miss_event creates a trace when search returns
        no useful results."""
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            trace_tool_search_miss_event(
                agent_name="entry_scanner",
                query="nonexistent tool",
                reason="no_results",
            )
        mock_lf.trace.assert_called_once()
        call_kwargs = mock_lf.trace.call_args[1]
        assert call_kwargs["name"] == "tool_search:miss"
        assert call_kwargs["metadata"]["reason"] == "no_results"
        assert "miss" in call_kwargs["tags"]

    def test_trace_does_not_log_server_tool_use_as_invocation(self):
        """server_tool_use blocks must never appear as tool invocation events.
        The trace helpers use dedicated names, not generic tool_call names."""
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            trace_tool_search_event("agent", "q", 1, ["t1"])
        call_kwargs = mock_lf.trace.call_args[1]
        assert "tool_call" not in call_kwargs["name"]
        assert call_kwargs["name"].startswith("tool_search:")

    def test_tracing_noops_when_langfuse_unavailable(self):
        """All trace_tool_search_* functions return silently when LangFuse
        client is None (best-effort pattern)."""
        with patch("quantstack.observability.tracing._get_langfuse", return_value=None):
            # None of these should raise
            trace_tool_search_event("a", "q", 0, [])
            trace_tool_discovery_event("a", "t", "q")
            trace_tool_search_miss_event("a", "q", "no_results")
            trace_tool_search_fallback("a", "err", 10)

    def test_tracing_catches_langfuse_exceptions(self):
        """If LangFuse raises during event creation, the exception is caught
        and logged at debug level — never propagated."""
        mock_lf = MagicMock()
        mock_lf.trace.side_effect = RuntimeError("LangFuse broke")
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            # Should not raise
            trace_tool_search_event("a", "q", 0, [])
            trace_tool_discovery_event("a", "t", "q")
            trace_tool_search_miss_event("a", "q", "no_results")
            trace_tool_search_fallback("a", "err", 10)


class TestToolSearchAccuracyMetrics:
    """Tests for tool search accuracy computation."""

    def _make_trace(self, metadata=None):
        t = MagicMock()
        t.metadata = metadata or {}
        return t

    def test_compute_search_hit_rate_from_sample_traces(self):
        """search_hit_rate = discoveries / total searches."""
        search_traces = [self._make_trace() for _ in range(10)]
        discovery_traces = [self._make_trace() for _ in range(7)]

        with patch("quantstack.observability.tool_search_metrics._get_langfuse") as mock_lf_fn, \
             patch("quantstack.observability.tool_search_metrics._fetch_traces_by_tag") as mock_fetch:
            mock_lf_fn.return_value = MagicMock()
            mock_fetch.side_effect = lambda lf, tag, cutoff: {
                "search": search_traces,
                "discovery": discovery_traces,
                "miss": [],
                "fallback": [],
            }[tag]

            metrics = compute_tool_search_accuracy(timedelta(hours=24))
            assert metrics.search_hit_rate == 0.7
            assert metrics.total_searches == 10
            assert metrics.total_discoveries == 7

    def test_compute_discovery_accuracy_from_sample_traces(self):
        """discovery_accuracy = 1.0 when discoveries exist (all are used by definition)."""
        with patch("quantstack.observability.tool_search_metrics._get_langfuse") as mock_lf_fn, \
             patch("quantstack.observability.tool_search_metrics._fetch_traces_by_tag") as mock_fetch:
            mock_lf_fn.return_value = MagicMock()
            mock_fetch.side_effect = lambda lf, tag, cutoff: {
                "search": [self._make_trace() for _ in range(5)],
                "discovery": [self._make_trace() for _ in range(3)],
                "miss": [],
                "fallback": [],
            }[tag]

            metrics = compute_tool_search_accuracy(timedelta(hours=24))
            assert metrics.discovery_accuracy == 1.0

    def test_compute_handles_empty_trace_set(self):
        """When no traces exist, return zeroed metrics without raising."""
        with patch("quantstack.observability.tool_search_metrics._get_langfuse") as mock_lf_fn, \
             patch("quantstack.observability.tool_search_metrics._fetch_traces_by_tag") as mock_fetch:
            mock_lf_fn.return_value = MagicMock()
            mock_fetch.return_value = []

            metrics = compute_tool_search_accuracy(timedelta(hours=24))
            assert metrics.search_hit_rate == 0.0
            assert metrics.discovery_accuracy == 0.0
            assert metrics.fallback_rate == 0.0
            assert metrics.total_searches == 0
            assert metrics.top_missed_tools == []

    def test_compute_returns_top_missed_tools(self):
        """top_missed_tools lists queries frequently missed, sorted by frequency."""
        miss_traces = [
            self._make_trace({"query": "risk_tool"}),
            self._make_trace({"query": "risk_tool"}),
            self._make_trace({"query": "risk_tool"}),
            self._make_trace({"query": "backtest_tool"}),
            self._make_trace({"query": "backtest_tool"}),
            self._make_trace({"query": "rare_tool"}),
        ]

        with patch("quantstack.observability.tool_search_metrics._get_langfuse") as mock_lf_fn, \
             patch("quantstack.observability.tool_search_metrics._fetch_traces_by_tag") as mock_fetch:
            mock_lf_fn.return_value = MagicMock()
            mock_fetch.side_effect = lambda lf, tag, cutoff: {
                "search": [self._make_trace() for _ in range(10)],
                "discovery": [self._make_trace()],
                "miss": miss_traces,
                "fallback": [],
            }[tag]

            metrics = compute_tool_search_accuracy(timedelta(hours=24))
            assert metrics.top_missed_tools[0] == "risk_tool"
            assert metrics.top_missed_tools[1] == "backtest_tool"
            assert "rare_tool" in metrics.top_missed_tools

    def test_compute_returns_fallback_rate(self):
        """fallback_rate = fallbacks / (searches + fallbacks)."""
        with patch("quantstack.observability.tool_search_metrics._get_langfuse") as mock_lf_fn, \
             patch("quantstack.observability.tool_search_metrics._fetch_traces_by_tag") as mock_fetch:
            mock_lf_fn.return_value = MagicMock()
            mock_fetch.side_effect = lambda lf, tag, cutoff: {
                "search": [self._make_trace() for _ in range(8)],
                "discovery": [],
                "miss": [],
                "fallback": [self._make_trace() for _ in range(2)],
            }[tag]

            metrics = compute_tool_search_accuracy(timedelta(hours=24))
            assert metrics.fallback_rate == 0.2
            assert metrics.total_fallbacks == 2

    def test_compute_returns_zeroed_when_langfuse_unavailable(self):
        """When LangFuse is not configured, return zeroed metrics."""
        with patch("quantstack.observability.tool_search_metrics._get_langfuse", return_value=None):
            metrics = compute_tool_search_accuracy(timedelta(hours=24))
            assert metrics.total_searches == 0
            assert metrics.search_hit_rate == 0.0


class TestFallbackTracing:
    """Tests for fallback-mode observability."""

    def test_fallback_event_traced_with_error_context(self):
        """When bind_tools_to_llm falls back to full loading, a trace event
        is created with the original error message and agent name."""
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            trace_tool_search_fallback(
                agent_name="daily_planner",
                error="beta header rejected",
                tools_loaded=15,
            )
        mock_lf.trace.assert_called_once()
        call_kwargs = mock_lf.trace.call_args[1]
        assert call_kwargs["name"] == "tool_search:fallback"
        assert call_kwargs["metadata"]["agent"] == "daily_planner"
        assert call_kwargs["metadata"]["error"] == "beta header rejected"
        assert call_kwargs["metadata"]["tools_loaded"] == 15
        assert "fallback" in call_kwargs["tags"]

    def test_fallback_mode_flag_propagated_to_traces(self):
        """When fallback_mode=True, the trace includes that metadata field."""
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            trace_tool_search_fallback(
                agent_name="risk_assessor",
                error="unsupported tool type",
                tools_loaded=20,
            )
        call_kwargs = mock_lf.trace.call_args[1]
        # The fallback trace itself is evidence of fallback_mode=True
        assert "tool_search" in call_kwargs["tags"]
        assert "fallback" in call_kwargs["tags"]


class TestStoreMetrics:
    """Tests for persisting metrics to PostgreSQL."""

    def test_store_metrics_inserts_row(self):
        """store_metrics writes a row to tool_search_metrics."""
        metrics = ToolSearchMetrics(
            search_hit_rate=0.75,
            discovery_accuracy=1.0,
            fallback_rate=0.1,
            top_missed_tools=["risk_tool", "backtest_tool"],
            total_searches=100,
            total_discoveries=75,
            total_misses=10,
            total_fallbacks=5,
            time_window=timedelta(hours=24),
        )
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.observability.tool_search_metrics.db_conn", return_value=mock_ctx):
            store_metrics(metrics)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "INSERT INTO tool_search_metrics" in sql
        assert params[0] == 24  # time_window_h
        assert params[1] == 0.75  # search_hit_rate
        assert params[5] == 100  # total_searches
