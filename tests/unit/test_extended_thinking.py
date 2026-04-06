"""Tests for extended thinking support (Section 06).

Tests cover:
- AgentConfig thinking field
- Per-agent LLM instantiation in graph builders
- Model instantiation with thinking config
- Target node verification in agents.yaml
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── 6.1 AgentConfig Thinking Field ──────────────────────────────────────────


class TestAgentConfigThinking:
    def test_accepts_thinking_config(self):
        from quantstack.graphs.config import AgentConfig

        cfg = AgentConfig(
            name="test",
            role="test role",
            goal="test goal",
            backstory="test backstory",
            llm_tier="heavy",
            thinking={"type": "adaptive"},
        )
        assert cfg.thinking == {"type": "adaptive"}

    def test_thinking_none_by_default(self):
        from quantstack.graphs.config import AgentConfig

        cfg = AgentConfig(
            name="test",
            role="test role",
            goal="test goal",
            backstory="test backstory",
            llm_tier="heavy",
        )
        assert cfg.thinking is None

    def test_load_agent_configs_reads_thinking(self, tmp_path):
        from quantstack.graphs.config import load_agent_configs

        yaml_content = """
test_agent:
  role: "Test Role"
  goal: "Test Goal"
  backstory: "Test"
  llm_tier: heavy
  thinking:
    type: adaptive
"""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(yaml_content)
        configs = load_agent_configs(yaml_file)
        assert configs["test_agent"].thinking == {"type": "adaptive"}

    def test_load_agent_configs_thinking_none_when_absent(self, tmp_path):
        from quantstack.graphs.config import load_agent_configs

        yaml_content = """
test_agent:
  role: "Test Role"
  goal: "Test Goal"
  backstory: "Test"
  llm_tier: heavy
"""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(yaml_content)
        configs = load_agent_configs(yaml_file)
        assert configs["test_agent"].thinking is None


# ── 6.2 Per-Agent LLM Instantiation ────────────────────────────────────────


def _make_config_watcher(tmp_path, yaml_content):
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    from quantstack.graphs.config_watcher import ConfigWatcher

    watcher = ConfigWatcher(yaml_file)
    return watcher


TRADING_YAML = """
daily_planner:
  role: "Planner"
  goal: "Plan"
  backstory: "Planner"
  llm_tier: medium
  tools: [signal_brief]

position_monitor:
  role: "Monitor"
  goal: "Monitor"
  backstory: "Monitor"
  llm_tier: medium
  tools: [signal_brief]

exit_evaluator:
  role: "Exit Evaluator"
  goal: "Evaluate exits"
  backstory: "Evaluates"
  llm_tier: medium
  tools: [signal_brief]

trade_debater:
  role: "Debater"
  goal: "Debate"
  backstory: "Debater"
  llm_tier: heavy
  tools: [signal_brief]

fund_manager:
  role: "FM"
  goal: "Review"
  backstory: "FM"
  llm_tier: heavy
  tools: [fetch_portfolio]

options_analyst:
  role: "Options"
  goal: "Options"
  backstory: "Options"
  llm_tier: heavy
  tools: [fetch_portfolio]

trade_reflector:
  role: "Reflector"
  goal: "Reflect"
  backstory: "Reflector"
  llm_tier: medium
  tools: [signal_brief]

market_intel:
  role: "Intel"
  goal: "Intel"
  backstory: "Intel"
  llm_tier: medium
  tools: [signal_brief]

earnings_analyst:
  role: "Earnings"
  goal: "Earnings"
  backstory: "Earnings"
  llm_tier: medium
  tools: [signal_brief]
"""


class TestPerAgentLLMTrading:
    def test_each_agent_gets_own_llm(self, tmp_path):
        """build_trading_graph calls get_chat_model once per agent."""
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph

        watcher = _make_config_watcher(tmp_path, TRADING_YAML)
        try:
            with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
                mock_gcm.return_value = MagicMock()
                build_trading_graph(watcher, MemorySaver())

            # 9 agents = 9 calls to get_chat_model
            assert mock_gcm.call_count == 9
        finally:
            watcher.stop()

    def test_non_thinking_agents_get_none(self, tmp_path):
        """Agents without thinking config pass thinking=None."""
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph

        watcher = _make_config_watcher(tmp_path, TRADING_YAML)
        try:
            with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
                mock_gcm.return_value = MagicMock()
                build_trading_graph(watcher, MemorySaver())

            # All calls should have thinking=None (no thinking agents in this fixture)
            none_calls = [
                c for c in mock_gcm.call_args_list
                if c.kwargs.get("thinking") is None
            ]
            assert len(none_calls) == 9
        finally:
            watcher.stop()


RESEARCH_YAML = """
quant_researcher:
  role: "Quant"
  goal: "Research"
  backstory: "Quant"
  llm_tier: heavy
  thinking:
    type: adaptive
  tools: [signal_brief]

ml_scientist:
  role: "ML"
  goal: "ML"
  backstory: "ML"
  llm_tier: heavy
  tools: [compute_features]

hypothesis_critic:
  role: "Critic"
  goal: "Critique"
  backstory: "Critic"
  llm_tier: medium
  tools: [search_knowledge_base]
"""


class TestPerAgentLLMResearch:
    def test_each_agent_gets_own_llm(self, tmp_path):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.research.graph import build_research_graph

        watcher = _make_config_watcher(tmp_path, RESEARCH_YAML)
        try:
            with patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm:
                mock_gcm.return_value = MagicMock()
                build_research_graph(watcher, MemorySaver())

            assert mock_gcm.call_count == 3  # quant + ml + critic
        finally:
            watcher.stop()

    def test_quant_researcher_gets_thinking(self, tmp_path):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.research.graph import build_research_graph

        watcher = _make_config_watcher(tmp_path, RESEARCH_YAML)
        try:
            with patch("quantstack.graphs.research.graph.get_chat_model") as mock_gcm:
                mock_gcm.return_value = MagicMock()
                build_research_graph(watcher, MemorySaver())

            thinking_calls = [
                c for c in mock_gcm.call_args_list
                if c.kwargs.get("thinking") == {"type": "adaptive"}
            ]
            assert len(thinking_calls) == 1
        finally:
            watcher.stop()


# ── 6.3 Model Instantiation with Thinking ──────────────────────────────────


class TestModelInstantiationThinking:
    def test_get_chat_model_max_tokens_with_thinking(self):
        from quantstack.llm.provider import get_chat_model

        with patch("quantstack.llm.provider.get_model_with_fallback", return_value="anthropic/claude-sonnet-4"):
            with patch("quantstack.llm.provider._instantiate_chat_model") as mock_inst:
                mock_inst.return_value = MagicMock()
                get_chat_model("heavy", thinking={"type": "adaptive"})

                config = mock_inst.call_args[0][0]
                assert config.max_tokens >= 8000

    def test_get_chat_model_standard_tokens_without_thinking(self):
        from quantstack.llm.provider import get_chat_model

        with patch("quantstack.llm.provider.get_model_with_fallback", return_value="anthropic/claude-sonnet-4"):
            with patch("quantstack.llm.provider._instantiate_chat_model") as mock_inst:
                mock_inst.return_value = MagicMock()
                get_chat_model("heavy", thinking=None)

                config = mock_inst.call_args[0][0]
                assert config.max_tokens == 4096

    def test_anthropic_thinking_budget_default(self):
        """When budget_tokens not specified, default to 5000."""
        from quantstack.llm.config import ModelConfig

        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4",
            tier="heavy",
            max_tokens=8192,
            thinking={"type": "adaptive"},
        )

        with patch("quantstack.llm.provider.ChatAnthropic", create=True) as mock_cls:
            with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_cls)}):
                from quantstack.llm.provider import _instantiate_chat_model
                _instantiate_chat_model(config)

                kwargs = mock_cls.call_args[1]
                assert "thinking" in kwargs
                assert kwargs["thinking"]["budget_tokens"] == 5000

    def test_anthropic_thinking_preserves_custom_budget(self):
        """When budget_tokens IS specified, preserve it."""
        from quantstack.llm.config import ModelConfig

        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4",
            tier="heavy",
            max_tokens=8192,
            thinking={"type": "adaptive", "budget_tokens": 10000},
        )

        with patch("quantstack.llm.provider.ChatAnthropic", create=True) as mock_cls:
            with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_cls)}):
                from quantstack.llm.provider import _instantiate_chat_model
                _instantiate_chat_model(config)

                kwargs = mock_cls.call_args[1]
                assert kwargs["thinking"]["budget_tokens"] == 10000

    def test_non_anthropic_ignores_thinking(self):
        """Non-anthropic providers silently ignore thinking config."""
        from quantstack.llm.config import ModelConfig

        config = ModelConfig(
            provider="openai",
            model_id="gpt-4o",
            tier="heavy",
            max_tokens=8192,
            thinking={"type": "adaptive"},
        )

        with patch("quantstack.llm.provider.ChatOpenAI", create=True) as mock_cls:
            with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_cls)}):
                from quantstack.llm.provider import _instantiate_chat_model
                _instantiate_chat_model(config)

                kwargs = mock_cls.call_args[1]
                assert "thinking" not in kwargs


# ── 6.4 Target Node Verification ───────────────────────────────────────────


class TestTargetNodeConfig:
    def _load_production_configs(self, graph_name):
        from quantstack.graphs.config import load_agent_configs

        yaml_path = (
            Path(__file__).resolve().parent.parent.parent
            / "src" / "quantstack" / "graphs" / graph_name / "config" / "agents.yaml"
        )
        return load_agent_configs(yaml_path)

    def test_quant_researcher_has_thinking(self):
        configs = self._load_production_configs("research")
        assert configs["quant_researcher"].thinking == {"type": "adaptive"}

    def test_trade_debater_no_thinking(self):
        configs = self._load_production_configs("trading")
        assert configs["trade_debater"].thinking is None

    def test_daily_planner_no_thinking(self):
        configs = self._load_production_configs("trading")
        assert configs["daily_planner"].thinking is None

    def test_exit_evaluator_no_thinking(self):
        configs = self._load_production_configs("trading")
        assert configs["exit_evaluator"].thinking is None
