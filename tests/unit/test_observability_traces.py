"""Tests for WI trace helpers in tracing.py (Section 10)."""

from unittest.mock import MagicMock, patch


class TestTraceQualityEvaluation:
    def test_creates_trace_with_scores(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_quality_evaluation
            trace_quality_evaluation(
                trade_id=42,
                scores={"overall_score": 0.75, "risk": 0.6},
                model_used="heavy",
                latency_ms=123.4,
            )
            mock_lf.trace.assert_called_once()
            kw = mock_lf.trace.call_args[1]
            assert kw["name"] == "trade_quality_evaluation"
            assert kw["metadata"]["trade_id"] == 42
            assert kw["metadata"]["overall_score"] == 0.75
            assert kw["metadata"]["model_used"] == "heavy"
            assert kw["metadata"]["latency_ms"] == 123.4
            assert "quality" in kw["tags"]
            assert "evaluation" in kw["tags"]

    def test_noops_when_langfuse_unavailable(self):
        with patch("quantstack.observability.tracing._get_langfuse", return_value=None):
            from quantstack.observability.tracing import trace_quality_evaluation
            trace_quality_evaluation(trade_id=1, scores={}, model_used="x", latency_ms=0)


class TestTraceThinkingEnabled:
    def test_creates_trace_with_config(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_thinking_enabled
            trace_thinking_enabled(
                agent_name="risk_analyst",
                thinking_config={"type": "adaptive"},
                model_id="claude-sonnet-4-20250514",
            )
            kw = mock_lf.trace.call_args[1]
            assert kw["name"] == "thinking_enabled"
            assert kw["metadata"]["agent_name"] == "risk_analyst"
            assert kw["metadata"]["thinking_config"] == {"type": "adaptive"}
            assert "thinking" in kw["tags"]
            assert "llm" in kw["tags"]


class TestTraceParallelBranchTiming:
    def test_creates_trace_with_duration(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_parallel_branch_timing
            trace_parallel_branch_timing(
                graph_name="trading",
                branch_name="portfolio_review",
                duration_seconds=5.2,
            )
            kw = mock_lf.trace.call_args[1]
            assert kw["name"] == "parallel_branch_timing"
            assert kw["metadata"]["branch_name"] == "portfolio_review"
            assert kw["metadata"]["duration_seconds"] == 5.2
            assert "parallel" in kw["tags"]


class TestTraceFanoutWorker:
    def test_creates_trace_success(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_fanout_worker
            trace_fanout_worker(
                symbol="AAPL",
                worker_index=0,
                duration_seconds=3.1,
                success=True,
            )
            kw = mock_lf.trace.call_args[1]
            assert kw["name"] == "fanout_worker"
            assert kw["metadata"]["symbol"] == "AAPL"
            assert kw["metadata"]["success"] is True
            assert kw["metadata"]["error"] is None
            assert "fanout" in kw["tags"]

    def test_creates_trace_failure_with_error(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_fanout_worker
            trace_fanout_worker(
                symbol="MSFT",
                worker_index=1,
                duration_seconds=0.5,
                success=False,
                error="timeout",
            )
            kw = mock_lf.trace.call_args[1]
            assert kw["metadata"]["success"] is False
            assert kw["metadata"]["error"] == "timeout"


class TestTraceHypothesisLoop:
    def test_creates_trace_with_loop_metrics(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_hypothesis_loop
            trace_hypothesis_loop(
                loop_count=2,
                final_confidence=0.82,
                max_attempts_hit=False,
            )
            kw = mock_lf.trace.call_args[1]
            assert kw["name"] == "hypothesis_loop"
            assert kw["metadata"]["loop_count"] == 2
            assert kw["metadata"]["final_confidence"] == 0.82
            assert kw["metadata"]["max_attempts_hit"] is False
            assert "hypothesis" in kw["tags"]
            assert "self_critique" in kw["tags"]

    def test_max_attempts_flag(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_hypothesis_loop
            trace_hypothesis_loop(loop_count=3, final_confidence=0.4, max_attempts_hit=True)
            kw = mock_lf.trace.call_args[1]
            assert kw["metadata"]["max_attempts_hit"] is True


class TestTraceHelpersExceptionSafety:
    """All trace helpers must silently swallow exceptions."""

    def test_quality_eval_swallows_exception(self):
        mock_lf = MagicMock()
        mock_lf.trace.side_effect = RuntimeError("langfuse down")
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_quality_evaluation
            trace_quality_evaluation(trade_id=1, scores={}, model_used="x", latency_ms=0)

    def test_thinking_swallows_exception(self):
        mock_lf = MagicMock()
        mock_lf.trace.side_effect = RuntimeError("langfuse down")
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_thinking_enabled
            trace_thinking_enabled(agent_name="x", thinking_config={}, model_id="x")

    def test_fanout_swallows_exception(self):
        mock_lf = MagicMock()
        mock_lf.trace.side_effect = RuntimeError("langfuse down")
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_fanout_worker
            trace_fanout_worker(symbol="X", worker_index=0, duration_seconds=0, success=True)

    def test_hypothesis_loop_swallows_exception(self):
        mock_lf = MagicMock()
        mock_lf.trace.side_effect = RuntimeError("langfuse down")
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_hypothesis_loop
            trace_hypothesis_loop(loop_count=1, final_confidence=0.5, max_attempts_hit=False)

    def test_parallel_branch_swallows_exception(self):
        mock_lf = MagicMock()
        mock_lf.trace.side_effect = RuntimeError("langfuse down")
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_parallel_branch_timing
            trace_parallel_branch_timing(graph_name="x", branch_name="y", duration_seconds=0)
