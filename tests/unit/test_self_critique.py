"""Tests for hypothesis self-critique loop (Section 08, WI-8)."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStateInitialization:
    @pytest.mark.asyncio
    async def test_context_load_initializes_critique_state(self):
        from quantstack.graphs.research.nodes import make_context_load

        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"summary": "test"}'))

        cfg = MagicMock()
        cfg.name = "test"
        cfg.max_iterations = 5
        cfg.timeout_seconds = 30

        node = make_context_load(llm, cfg, [])
        with patch("quantstack.graphs.research.nodes.run_agent", return_value='{"summary": "test"}'):
            result = await node({"cycle_number": 1, "regime": "unknown"})

        assert result["hypothesis_attempts"] == 0
        assert result["hypothesis_confidence"] == 0.0
        assert result["hypothesis_critique"] == ""

    @pytest.mark.asyncio
    async def test_hypothesis_generation_increments_attempts(self):
        from quantstack.graphs.research.nodes import make_hypothesis_generation

        llm = MagicMock()
        cfg = MagicMock()
        cfg.name = "test"
        cfg.max_iterations = 5
        cfg.timeout_seconds = 30

        node = make_hypothesis_generation(llm, cfg, [])

        with patch("quantstack.graphs.research.nodes.run_agent", return_value='{"hypothesis": "test"}'):
            result1 = await node({"hypothesis_attempts": 0, "selected_domain": "swing", "selected_symbols": []})
            assert result1["hypothesis_attempts"] == 1

            result2 = await node({"hypothesis_attempts": 1, "selected_domain": "swing", "selected_symbols": []})
            assert result2["hypothesis_attempts"] == 2


class TestCritiqueNode:
    def test_hypothesis_critic_config_exists(self):
        from quantstack.graphs.config import load_agent_configs

        yaml_path = Path(__file__).resolve().parents[2] / "src" / "quantstack" / "graphs" / "research" / "config" / "agents.yaml"
        configs = load_agent_configs(yaml_path)
        assert "hypothesis_critic" in configs

    def test_hypothesis_critic_is_heavy(self):
        from quantstack.graphs.config import load_agent_configs

        yaml_path = Path(__file__).resolve().parents[2] / "src" / "quantstack" / "graphs" / "research" / "config" / "agents.yaml"
        configs = load_agent_configs(yaml_path)
        assert configs["hypothesis_critic"].llm_tier == "heavy"

    @pytest.mark.asyncio
    async def test_critique_returns_confidence_and_critique(self):
        from quantstack.graphs.research.nodes import make_hypothesis_critique

        node = make_hypothesis_critique(MagicMock(), MagicMock(name="critic", max_iterations=5, timeout_seconds=30), [])
        with patch("quantstack.graphs.research.nodes.run_agent", return_value='{"confidence": 0.4, "critique": "Lacks specificity"}'):
            result = await node({"hypothesis": "test", "hypothesis_attempts": 1})

        assert isinstance(result["hypothesis_confidence"], float)
        assert result["hypothesis_confidence"] == 0.4
        assert "Lacks specificity" in result["hypothesis_critique"]

    @pytest.mark.asyncio
    async def test_critique_clears_critique_when_confident(self):
        from quantstack.graphs.research.nodes import make_hypothesis_critique

        node = make_hypothesis_critique(MagicMock(), MagicMock(name="critic", max_iterations=5, timeout_seconds=30), [])
        with patch("quantstack.graphs.research.nodes.run_agent", return_value='{"confidence": 0.85, "critique": "Good"}'):
            result = await node({"hypothesis": "test", "hypothesis_attempts": 1})

        assert result["hypothesis_confidence"] == 0.85
        assert result["hypothesis_critique"] == ""


class TestConditionalRouting:
    def test_high_confidence_routes_forward(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis

        state = {"hypothesis_confidence": 0.8, "hypothesis_attempts": 1}
        assert route_after_hypothesis(state) == "signal_validation"

    def test_low_confidence_low_attempts_loops(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis

        state = {"hypothesis_confidence": 0.4, "hypothesis_attempts": 1}
        assert route_after_hypothesis(state) == "hypothesis_generation"

    def test_low_confidence_max_retries_routes_forward(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis

        state = {"hypothesis_confidence": 0.4, "hypothesis_attempts": 3}
        assert route_after_hypothesis(state) == "signal_validation"

    def test_env_threshold_override(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis

        os.environ["HYPOTHESIS_CONFIDENCE_THRESHOLD"] = "0.5"
        try:
            state = {"hypothesis_confidence": 0.6, "hypothesis_attempts": 1}
            assert route_after_hypothesis(state) == "signal_validation"
        finally:
            del os.environ["HYPOTHESIS_CONFIDENCE_THRESHOLD"]

    def test_max_3_total_attempts(self):
        from quantstack.graphs.research.nodes import route_after_hypothesis

        # attempts 0,1,2 with low confidence -> loop
        for a in [0, 1, 2]:
            assert route_after_hypothesis({"hypothesis_confidence": 0.2, "hypothesis_attempts": a}) == "hypothesis_generation"

        # attempts 3 -> forward regardless
        assert route_after_hypothesis({"hypothesis_confidence": 0.2, "hypothesis_attempts": 3}) == "signal_validation"


class TestLoopBehavior:
    @pytest.mark.asyncio
    async def test_hypothesis_gen_uses_critique_in_prompt(self):
        from quantstack.graphs.research.nodes import make_hypothesis_generation

        node = make_hypothesis_generation(MagicMock(), MagicMock(name="test", max_iterations=5, timeout_seconds=30), [])

        captured_prompt = []

        async def capture_run_agent(llm, tools, config, prompt):
            captured_prompt.append(prompt)
            return '{"hypothesis": "revised"}'

        with patch("quantstack.graphs.research.nodes.run_agent", side_effect=capture_run_agent):
            await node({
                "hypothesis_critique": "Lacks specificity on entry conditions",
                "hypothesis_attempts": 1,
                "selected_domain": "swing",
                "selected_symbols": ["AAPL"],
            })

        assert "Lacks specificity on entry conditions" in captured_prompt[0]
