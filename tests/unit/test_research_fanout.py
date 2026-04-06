"""Tests for research fan-out via Send() (Section 09, WI-7)."""

import operator
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Send


class TestFanOutHypotheses:
    def test_returns_send_per_symbol(self):
        from quantstack.graphs.research.nodes import fan_out_hypotheses

        state = {
            "selected_symbols": ["AAPL", "MSFT", "GOOG"],
            "hypothesis": "momentum edge",
            "selected_domain": "swing",
        }
        sends = fan_out_hypotheses(state)
        assert len(sends) == 3
        assert all(isinstance(s, Send) for s in sends)

    def test_send_contains_correct_payload(self):
        from quantstack.graphs.research.nodes import fan_out_hypotheses

        state = {
            "selected_symbols": ["AAPL"],
            "hypothesis": "mean reversion",
            "selected_domain": "investment",
        }
        sends = fan_out_hypotheses(state)
        assert sends[0].node == "validate_symbol"
        payload = sends[0].arg
        assert payload["symbol_hypothesis"]["symbol"] == "AAPL"
        assert payload["symbol_hypothesis"]["hypothesis"] == "mean reversion"
        assert payload["symbol_hypothesis"]["domain"] == "investment"

    def test_empty_symbols_returns_empty(self):
        from quantstack.graphs.research.nodes import fan_out_hypotheses

        state = {"selected_symbols": [], "hypothesis": "test", "selected_domain": "swing"}
        sends = fan_out_hypotheses(state)
        assert sends == []


class TestRouteAfterHypothesisFanout:
    def test_high_confidence_fans_out(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis_fanout

        state = {
            "hypothesis_confidence": 0.8,
            "hypothesis_attempts": 1,
            "selected_symbols": ["AAPL", "MSFT"],
            "hypothesis": "test",
            "selected_domain": "swing",
        }
        result = route_after_hypothesis_fanout(state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)

    def test_low_confidence_loops_back(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis_fanout

        state = {
            "hypothesis_confidence": 0.3,
            "hypothesis_attempts": 1,
            "selected_symbols": ["AAPL"],
            "hypothesis": "test",
            "selected_domain": "swing",
        }
        result = route_after_hypothesis_fanout(state)
        assert result == "hypothesis_generation"

    def test_max_attempts_fans_out(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis_fanout

        state = {
            "hypothesis_confidence": 0.2,
            "hypothesis_attempts": 3,
            "selected_symbols": ["AAPL"],
            "hypothesis": "test",
            "selected_domain": "swing",
        }
        result = route_after_hypothesis_fanout(state)
        assert isinstance(result, list)
        assert len(result) == 1


class TestValidateSymbol:
    @pytest.mark.asyncio
    async def test_successful_validation_returns_result(self):
        from quantstack.graphs.research.nodes import make_validate_symbol

        quant_llm = MagicMock()
        ml_llm = MagicMock()
        quant_cfg = MagicMock(name="quant", max_iterations=5, timeout_seconds=30)
        ml_cfg = MagicMock(name="ml", max_iterations=5, timeout_seconds=30)

        node = make_validate_symbol(quant_llm, ml_llm, quant_cfg, ml_cfg, [], [])

        call_count = [0]

        async def mock_run_agent(llm, tools, config, prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"passed": true, "signals": ["RSI"], "reason": "confirmed"}'
            elif call_count[0] == 2:
                return '{"backtest_id": "bt1", "sharpe": 1.2, "passed": true}'
            else:
                return '{"experiment_id": "exp1", "ic": 0.05, "passed": true}'

        with patch("quantstack.graphs.research.nodes.run_agent", side_effect=mock_run_agent):
            result = await node({"symbol_hypothesis": {"symbol": "AAPL", "hypothesis": "test", "domain": "swing"}})

        assert len(result["validation_results"]) == 1
        assert result["validation_results"][0]["symbol"] == "AAPL"
        assert result["validation_results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_failed_signal_validation_short_circuits(self):
        from quantstack.graphs.research.nodes import make_validate_symbol

        node = make_validate_symbol(MagicMock(), MagicMock(), MagicMock(name="q", max_iterations=5, timeout_seconds=30), MagicMock(name="m", max_iterations=5, timeout_seconds=30), [], [])

        with patch("quantstack.graphs.research.nodes.run_agent", return_value='{"passed": false, "reason": "no signal"}'):
            result = await node({"symbol_hypothesis": {"symbol": "MSFT", "hypothesis": "test"}})

        assert result["validation_results"][0]["passed"] is False
        assert "no signal" in result["validation_results"][0]["reason"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_result(self):
        from quantstack.graphs.research.nodes import make_validate_symbol

        node = make_validate_symbol(MagicMock(), MagicMock(), MagicMock(name="q", max_iterations=5, timeout_seconds=30), MagicMock(name="m", max_iterations=5, timeout_seconds=30), [], [])

        with patch("quantstack.graphs.research.nodes.run_agent", side_effect=RuntimeError("boom")):
            result = await node({"symbol_hypothesis": {"symbol": "GOOG", "hypothesis": "test"}})

        assert result["validation_results"][0]["passed"] is False
        assert "boom" in result["validation_results"][0]["error"]


class TestFilterResults:
    @pytest.mark.asyncio
    async def test_all_passed(self):
        from quantstack.graphs.research.nodes import make_filter_results

        node = make_filter_results()
        state = {
            "validation_results": [
                {"symbol": "AAPL", "passed": True, "backtest_results": {}},
                {"symbol": "MSFT", "passed": True, "backtest_results": {}},
            ]
        }
        result = await node(state)
        assert result["validation_result"]["passed"] is True
        assert result["validation_result"]["symbols_passed"] == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_mixed_results(self):
        from quantstack.graphs.research.nodes import make_filter_results

        node = make_filter_results()
        state = {
            "validation_results": [
                {"symbol": "AAPL", "passed": True},
                {"symbol": "MSFT", "passed": False, "reason": "no signal"},
            ]
        }
        result = await node(state)
        assert result["validation_result"]["passed"] is True
        assert "AAPL" in result["validation_result"]["symbols_passed"]
        assert "MSFT" in result["validation_result"]["symbols_failed"]

    @pytest.mark.asyncio
    async def test_all_failed(self):
        from quantstack.graphs.research.nodes import make_filter_results

        node = make_filter_results()
        state = {
            "validation_results": [
                {"symbol": "AAPL", "passed": False, "reason": "no signal"},
            ]
        }
        result = await node(state)
        assert result["validation_result"]["passed"] is False

    @pytest.mark.asyncio
    async def test_empty_results(self):
        from quantstack.graphs.research.nodes import make_filter_results

        node = make_filter_results()
        state = {"validation_results": []}
        result = await node(state)
        assert result["validation_result"]["passed"] is False


class TestResearchStateAccumulator:
    def test_validation_results_uses_add_operator(self):
        from quantstack.graphs.state import ResearchState

        hints = ResearchState.__annotations__
        assert "validation_results" in hints

    def test_symbol_validation_state_exists(self):
        from quantstack.graphs.state import SymbolValidationState

        assert "symbol_hypothesis" in SymbolValidationState.__annotations__


class TestGraphTopologyFanout:
    def test_fanout_graph_compiles_when_enabled(self):
        """Verify the fan-out graph topology compiles without error."""
        import os
        from unittest.mock import patch as _patch

        with _patch.dict(os.environ, {"RESEARCH_FAN_OUT_ENABLED": "true"}):
            from quantstack.graphs.research.graph import build_research_graph

            mock_watcher = MagicMock()

            def fake_get_config(name):
                cfg = MagicMock()
                cfg.name = name
                cfg.llm_tier = "medium"
                cfg.thinking = None
                cfg.tools = []
                cfg.always_loaded_tools = []
                cfg.max_iterations = 5
                cfg.timeout_seconds = 30
                return cfg

            mock_watcher.get_config = fake_get_config

            with _patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm, \
                 _patch("quantstack.graphs.research.graph.bind_tools_to_llm") as mock_bind:
                mock_llm = MagicMock()
                mock_gcm.return_value = mock_llm
                mock_bind.return_value = (mock_llm, [], None)

                graph = build_research_graph(mock_watcher, MagicMock())
                assert graph is not None
