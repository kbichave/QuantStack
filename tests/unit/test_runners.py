"""Tests for the async runner refactor (Section 11).

All graph, health utilities, and DB are mocked. No external services required.
"""

import asyncio
import inspect
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


class AsyncStubShutdown:
    """Shutdown stub that stops after N cycle completions."""

    def __init__(self, max_cycles: int):
        self._max = max_cycles
        self._count = 0

    @property
    def should_stop(self) -> bool:
        return self._count >= self._max

    def tick(self):
        self._count += 1


# --- Market Hours Detection ---


class TestIsMarketHours:
    def test_true_during_nyse_hours(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 10, 0, tzinfo=ET)
        assert is_market_hours(dt) is True

    def test_false_on_saturday(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 4, 12, 0, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_false_on_sunday(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 5, 12, 0, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_false_on_new_years_day(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 1, 1, 12, 0, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_false_before_open(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 9, 29, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_true_at_open(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 9, 30, tzinfo=ET)
        assert is_market_hours(dt) is True

    def test_false_at_close(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 16, 0, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_true_just_before_close(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 15, 59, tzinfo=ET)
        assert is_market_hours(dt) is True

    def test_false_after_close(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 16, 1, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_false_on_christmas(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 12, 25, 12, 0, tzinfo=ET)
        assert is_market_hours(dt) is False

    def test_naive_datetime_treated_as_et(self):
        from quantstack.runners import is_market_hours
        dt = datetime(2026, 4, 1, 10, 0)
        assert is_market_hours(dt) is True


# --- Cycle Interval Selection ---


class TestGetCycleInterval:
    def test_trading_market_hours(self):
        from quantstack.runners import get_cycle_interval
        with patch("quantstack.runners.is_market_hours", return_value=True):
            assert get_cycle_interval("trading") == 300

    def test_trading_after_hours_weekday(self):
        from quantstack.runners import get_cycle_interval
        with patch("quantstack.runners.is_market_hours", return_value=False), \
             patch("quantstack.runners._is_weekend", return_value=False):
            assert get_cycle_interval("trading") == 1800

    def test_trading_weekend(self):
        from quantstack.runners import get_cycle_interval
        with patch("quantstack.runners.is_market_hours", return_value=False), \
             patch("quantstack.runners._is_weekend", return_value=True):
            assert get_cycle_interval("trading") is None

    def test_research_market_hours(self):
        from quantstack.runners import get_cycle_interval
        with patch("quantstack.runners.is_market_hours", return_value=True):
            assert get_cycle_interval("research") == 600

    def test_research_weekend(self):
        from quantstack.runners import get_cycle_interval
        with patch("quantstack.runners.is_market_hours", return_value=False), \
             patch("quantstack.runners._is_weekend", return_value=True):
            assert get_cycle_interval("research") == 7200

    def test_supervisor_always_300(self):
        from quantstack.runners import get_cycle_interval
        with patch("quantstack.runners.is_market_hours", return_value=False), \
             patch("quantstack.runners._is_weekend", return_value=True):
            assert get_cycle_interval("supervisor") == 300


# --- Runner Entry Points ---


class TestRunnerEntryPoints:
    """Verify each runner's main() uses asyncio.run()."""

    def test_run_loop_is_coroutine(self):
        from quantstack.runners.trading_runner import run_loop
        assert inspect.iscoroutinefunction(run_loop)

    def test_trading_runner_has_async_main(self):
        from quantstack.runners.trading_runner import async_main
        assert inspect.iscoroutinefunction(async_main)

    def test_research_runner_has_async_main(self):
        from quantstack.runners.research_runner import async_main
        assert inspect.iscoroutinefunction(async_main)

    def test_supervisor_runner_has_async_main(self):
        from quantstack.runners.supervisor_runner import async_main
        assert inspect.iscoroutinefunction(async_main)


# --- Async Run Loop ---


class TestAsyncRunLoop:
    """Tests for the shared async run_loop() function."""

    @pytest.mark.asyncio
    async def test_rebuilds_graph_each_cycle(self):
        """Graph is rebuilt each cycle via graph_builder callable."""
        from quantstack.runners.trading_runner import run_loop

        shutdown = AsyncStubShutdown(max_cycles=3)
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"errors": [], "decisions": []})
        build_count = 0

        def graph_builder():
            nonlocal build_count
            build_count += 1
            shutdown.tick()
            return mock_graph

        def state_builder(cycle_number):
            return {"cycle_number": cycle_number, "errors": [], "decisions": []}

        with patch("quantstack.runners.trading_runner.get_cycle_interval", return_value=0), \
             patch("quantstack.runners.trading_runner.write_heartbeat"), \
             patch("quantstack.runners.trading_runner.save_checkpoint"), \
             patch("quantstack.runners.trading_runner.langfuse_trace_context") as mock_trace:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)
            await run_loop(graph_builder, state_builder, shutdown, graph_name="trading")

        assert build_count == 3

    @pytest.mark.asyncio
    async def test_passes_thread_id_in_config(self):
        """graph.ainvoke() receives thread_id in configurable."""
        from quantstack.runners.trading_runner import run_loop

        shutdown = AsyncStubShutdown(max_cycles=1)
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"errors": [], "decisions": []})

        def graph_builder():
            shutdown.tick()
            return mock_graph

        def state_builder(cycle_number):
            return {"cycle_number": cycle_number, "errors": [], "decisions": []}

        with patch("quantstack.runners.trading_runner.get_cycle_interval", return_value=0), \
             patch("quantstack.runners.trading_runner.write_heartbeat"), \
             patch("quantstack.runners.trading_runner.save_checkpoint"), \
             patch("quantstack.runners.trading_runner.langfuse_trace_context") as mock_trace:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)
            await run_loop(graph_builder, state_builder, shutdown, graph_name="trading")

        call_config = mock_graph.ainvoke.call_args[1]["config"]
        assert "configurable" in call_config
        assert "thread_id" in call_config["configurable"]
        assert "trading" in call_config["configurable"]["thread_id"]

    @pytest.mark.asyncio
    async def test_stops_on_shutdown_flag(self):
        """run_loop exits when shutdown.should_stop is True."""
        from quantstack.runners.trading_runner import run_loop

        shutdown = AsyncStubShutdown(max_cycles=0)  # already stopped
        graph_builder = MagicMock()
        state_builder = MagicMock()

        with patch("quantstack.runners.trading_runner.get_cycle_interval", return_value=300):
            await run_loop(graph_builder, state_builder, shutdown, graph_name="test")

        graph_builder.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_graph_exception(self):
        """Exceptions from graph.ainvoke() are caught and logged."""
        from quantstack.runners.trading_runner import run_loop

        shutdown = AsyncStubShutdown(max_cycles=1)
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM failed"))

        def graph_builder():
            shutdown.tick()
            return mock_graph

        def state_builder(cycle_number):
            return {"cycle_number": cycle_number, "errors": [], "decisions": []}

        with patch("quantstack.runners.trading_runner.get_cycle_interval", return_value=0), \
             patch("quantstack.runners.trading_runner.write_heartbeat") as mock_hb, \
             patch("quantstack.runners.trading_runner.save_checkpoint") as mock_cp, \
             patch("quantstack.runners.trading_runner.langfuse_trace_context") as mock_trace:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)
            await run_loop(graph_builder, state_builder, shutdown, graph_name="trading")

        # Heartbeat NOT written on failure
        mock_hb.assert_not_called()
        # Checkpoint IS written with error status
        mock_cp.assert_called_once()
        assert mock_cp.call_args[0][3] == "error"

    @pytest.mark.asyncio
    async def test_writes_heartbeat_on_success(self):
        """Heartbeat written after successful graph invocation."""
        from quantstack.runners.trading_runner import run_loop

        shutdown = AsyncStubShutdown(max_cycles=1)
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"errors": [], "decisions": []})

        def graph_builder():
            shutdown.tick()
            return mock_graph

        def state_builder(cycle_number):
            return {"cycle_number": cycle_number, "errors": [], "decisions": []}

        with patch("quantstack.runners.trading_runner.get_cycle_interval", return_value=0), \
             patch("quantstack.runners.trading_runner.write_heartbeat") as mock_hb, \
             patch("quantstack.runners.trading_runner.save_checkpoint"), \
             patch("quantstack.runners.trading_runner.langfuse_trace_context") as mock_trace:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)
            await run_loop(graph_builder, state_builder, shutdown, graph_name="trading")

        mock_hb.assert_called_once_with("trading")

    @pytest.mark.asyncio
    async def test_consecutive_failure_tracking(self):
        """After 3 consecutive failures, critical log should be emitted."""
        from quantstack.runners.trading_runner import run_loop

        shutdown = AsyncStubShutdown(max_cycles=4)
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("always fails"))

        def graph_builder():
            shutdown.tick()
            return mock_graph

        def state_builder(cycle_number):
            return {"cycle_number": cycle_number, "errors": [], "decisions": []}

        with patch("quantstack.runners.trading_runner.get_cycle_interval", return_value=0), \
             patch("quantstack.runners.trading_runner.write_heartbeat"), \
             patch("quantstack.runners.trading_runner.save_checkpoint"), \
             patch("quantstack.runners.trading_runner.langfuse_trace_context") as mock_trace, \
             patch("quantstack.runners.trading_runner.logger") as mock_logger:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)
            await run_loop(graph_builder, state_builder, shutdown, graph_name="trading")

        # logger.critical should be called at least once (after 3rd failure)
        critical_calls = [c for c in mock_logger.critical.call_args_list
                          if "consecutive" in str(c)]
        assert len(critical_calls) >= 1


# --- GracefulShutdown Async ---


class TestGracefulShutdownAsync:
    """Verify GracefulShutdown works in async context."""

    def test_install_async_method_exists(self):
        from quantstack.health.shutdown import GracefulShutdown
        gs = GracefulShutdown()
        assert hasattr(gs, "install_async")

    @pytest.mark.asyncio
    async def test_should_stop_defaults_false(self):
        from quantstack.health.shutdown import GracefulShutdown
        gs = GracefulShutdown()
        assert gs.should_stop is False

    @pytest.mark.asyncio
    async def test_handle_async_signal_sets_stop(self):
        """Calling _handle_async_signal sets should_stop."""
        import signal
        from quantstack.health.shutdown import GracefulShutdown
        gs = GracefulShutdown()
        gs._handle_async_signal(signal.SIGTERM.value)
        assert gs.should_stop is True
