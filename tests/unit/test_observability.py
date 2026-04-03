"""Tests for observability layer (Section 10: LangGraph migration).

Tests use mocked Langfuse — no running server needed.
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "quantstack"


class TestSetupInstrumentation:
    """Tests for instrumentation.py."""

    def test_raises_without_secret_key(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        from quantstack.observability.instrumentation import setup_instrumentation
        with pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"):
            setup_instrumentation()

    def test_raises_without_public_key(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        from quantstack.observability.instrumentation import setup_instrumentation
        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            setup_instrumentation()

    def test_no_crewai_instrumentor_import(self):
        """setup_instrumentation must not reference CrewAIInstrumentor."""
        from quantstack.observability import instrumentation
        source = inspect.getsource(instrumentation)
        assert "CrewAIInstrumentor" not in source
        assert "openinference.instrumentation.crewai" not in source

    def test_initializes_langfuse_client(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        import quantstack.observability.tracing as tracing_mod
        tracing_mod._init_attempted = False
        tracing_mod._langfuse = None

        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing.Langfuse", return_value=mock_lf):
            from quantstack.observability.instrumentation import setup_instrumentation
            setup_instrumentation()
            assert tracing_mod._langfuse is mock_lf


class TestLangfuseTraceContext:
    """Tests for langfuse_trace_context context manager."""

    def test_yields_trace_object(self):
        mock_lf = MagicMock()
        mock_trace = MagicMock()
        mock_lf.trace.return_value = mock_trace
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.instrumentation import langfuse_trace_context
            with langfuse_trace_context("test-session", ["trading"]) as trace:
                assert trace is mock_trace

    def test_passes_session_id_and_tags(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.instrumentation import langfuse_trace_context
            with langfuse_trace_context("cycle-42", ["research", "paper"], name="research_cycle"):
                pass
            mock_lf.trace.assert_called_once_with(
                name="research_cycle",
                session_id="cycle-42",
                tags=["research", "paper"],
            )

    def test_yields_none_when_langfuse_unavailable(self):
        with patch("quantstack.observability.tracing._get_langfuse", return_value=None):
            from quantstack.observability.instrumentation import langfuse_trace_context
            with langfuse_trace_context("test", ["trading"]) as trace:
                assert trace is None

    def test_marks_trace_success(self):
        mock_lf = MagicMock()
        mock_trace = MagicMock()
        mock_lf.trace.return_value = mock_trace
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.instrumentation import langfuse_trace_context
            with langfuse_trace_context("test", ["trading"]):
                pass
            mock_trace.update.assert_called_once_with(status_message="success")


class TestFlushTraces:
    """Tests for flush_util.py."""

    def test_flush_calls_shutdown(self):
        from quantstack.observability.flush_util import flush_traces
        with patch("quantstack.observability.tracing.shutdown") as mock_shutdown:
            flush_traces()
            mock_shutdown.assert_called_once()


class TestBusinessEventTraces:
    """Tests for business event trace helpers (merged from crew_tracing.py)."""

    def test_provider_failover_creates_trace(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_provider_failover
            trace_provider_failover("bedrock", "anthropic", "timeout", "heavy")
            mock_lf.trace.assert_called_once()
            call_kwargs = mock_lf.trace.call_args[1]
            assert call_kwargs["name"] == "provider_failover"
            assert call_kwargs["metadata"]["original_provider"] == "bedrock"
            assert call_kwargs["metadata"]["fallback_provider"] == "anthropic"

    def test_strategy_lifecycle_includes_reasoning(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_strategy_lifecycle
            trace_strategy_lifecycle(
                "strat-001", "promote",
                "Strong Sharpe ratio over 20 days",
                {"sharpe": 2.1, "win_rate": 0.65},
            )
            call_kwargs = mock_lf.trace.call_args[1]
            assert "Strong Sharpe" in call_kwargs["metadata"]["reasoning"]

    def test_safety_boundary_trigger_logged(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import trace_safety_boundary_trigger
            trace_safety_boundary_trigger(
                "SPY",
                {"size_pct": 0.20},
                "max_position_pct",
                0.15,
            )
            call_kwargs = mock_lf.trace.call_args[1]
            assert call_kwargs["name"] == "safety_boundary_trigger"
            assert call_kwargs["metadata"]["gate_limit"] == "max_position_pct"

    def test_noop_when_langfuse_unavailable(self):
        with patch("quantstack.observability.tracing._get_langfuse", return_value=None):
            from quantstack.observability.tracing import trace_provider_failover
            trace_provider_failover("bedrock", "anthropic", "err", "heavy")


class TestShutdown:
    """Tests for tracing.shutdown()."""

    def test_shutdown_calls_client_shutdown(self):
        mock_lf = MagicMock()
        with patch("quantstack.observability.tracing._get_langfuse", return_value=mock_lf):
            from quantstack.observability.tracing import shutdown
            shutdown()
            mock_lf.shutdown.assert_called_once()

    def test_shutdown_noop_without_client(self):
        with patch("quantstack.observability.tracing._get_langfuse", return_value=None):
            from quantstack.observability.tracing import shutdown
            shutdown()  # Should not raise


class TestCrewTracingDeleted:
    """Verify crew_tracing.py no longer exists."""

    def test_crew_tracing_file_deleted(self):
        path = SRC_ROOT / "observability" / "crew_tracing.py"
        assert not path.exists(), f"crew_tracing.py should be deleted: {path}"


class TestNoCrewAIReferences:
    """Verify no code imports openinference.instrumentation.crewai."""

    def test_no_crewai_instrumentation_imports(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-r", "openinference.instrumentation.crewai",
             str(SRC_ROOT)],
            capture_output=True, text=True,
        )
        assert result.stdout == "", (
            f"Found CrewAI instrumentation references:\n{result.stdout}"
        )
