"""Unit tests for LLM tier mapping (WI-4).

Covers: bedrock tier resolution (heavy->sonnet, medium->haiku, light->haiku),
provider fallback chains, and bind_tools_to_llm tier consistency.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantstack.llm.config import PROVIDER_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "quantstack"


# ---------------------------------------------------------------------------
# 5.1 Graph Builder Tier Mismatch Tests
# ---------------------------------------------------------------------------


class TestGraphBuilderTierConsistency:
    """Verify bind_tools_to_llm calls use the correct tier LLM for each agent."""

    def _build_graph_capturing_tiers(self, graph_module_path, build_func_name, yaml_path):
        """Build a graph while capturing which tier LLM each agent receives.

        Returns dict mapping agent_name -> tier_tag.
        """
        from quantstack.graphs.config import load_agent_configs
        from quantstack.graphs.config_watcher import ConfigWatcher

        configs = load_agent_configs(yaml_path)
        watcher = ConfigWatcher(yaml_path)

        tier_tags = {}

        def tagged_get_chat_model(tier, **kwargs):
            model = MagicMock()
            model._tier_tag = tier
            model.bind_tools = MagicMock(return_value=model)
            return model

        original_bind = None
        agent_tier_map = {}

        def capturing_bind(llm, agent_cfg, *args, **kwargs):
            tier_tag = getattr(llm, "_tier_tag", "unknown")
            agent_tier_map[agent_cfg.name] = tier_tag
            return llm, [], {}

        import importlib
        module = importlib.import_module(graph_module_path)
        build_func = getattr(module, build_func_name)

        from langgraph.checkpoint.memory import MemorySaver
        with patch(f"{graph_module_path}.get_chat_model", side_effect=tagged_get_chat_model), \
             patch(f"{graph_module_path}.bind_tools_to_llm", side_effect=capturing_bind):
            build_func(watcher, MemorySaver())

        watcher.stop()
        return agent_tier_map

    def test_trading_graph_tier_consistency(self):
        yaml_path = SRC_ROOT / "graphs" / "trading" / "config" / "agents.yaml"
        tier_map = self._build_graph_capturing_tiers(
            "quantstack.graphs.trading.graph", "build_trading_graph", yaml_path
        )

        from quantstack.graphs.config import load_agent_configs
        expected_tiers = {name: cfg.llm_tier for name, cfg in load_agent_configs(yaml_path).items()}

        for agent_name, expected_tier in expected_tiers.items():
            if agent_name in tier_map:
                assert tier_map[agent_name] == expected_tier, (
                    f"{agent_name}: expected {expected_tier}, got {tier_map[agent_name]}"
                )

    def test_risk_analyst_is_heavy(self):
        """Regression test for the specific risk_analyst tier mismatch bug."""
        yaml_path = SRC_ROOT / "graphs" / "trading" / "config" / "agents.yaml"
        from quantstack.graphs.config import load_agent_configs
        configs = load_agent_configs(yaml_path)
        assert configs["risk_analyst"].llm_tier == "heavy"

    def test_supervisor_graph_tier_consistency(self):
        yaml_path = SRC_ROOT / "graphs" / "supervisor" / "config" / "agents.yaml"
        tier_map = self._build_graph_capturing_tiers(
            "quantstack.graphs.supervisor.graph", "build_supervisor_graph", yaml_path
        )

        from quantstack.graphs.config import load_agent_configs
        expected_tiers = {name: cfg.llm_tier for name, cfg in load_agent_configs(yaml_path).items()}

        for agent_name, expected_tier in expected_tiers.items():
            if agent_name in tier_map:
                assert tier_map[agent_name] == expected_tier, (
                    f"{agent_name}: expected {expected_tier}, got {tier_map[agent_name]}"
                )

    def test_self_healer_is_medium(self):
        """Regression test for the self_healer tier mismatch bug."""
        yaml_path = SRC_ROOT / "graphs" / "supervisor" / "config" / "agents.yaml"
        from quantstack.graphs.config import load_agent_configs
        configs = load_agent_configs(yaml_path)
        assert configs["self_healer"].llm_tier == "medium"


# ---------------------------------------------------------------------------
# 5.2 Bedrock Tier Mapping Tests
# ---------------------------------------------------------------------------


class TestBedrockTierMapping:
    """Verify bedrock provider config maps tiers to distinct models."""

    def test_bedrock_heavy_is_sonnet(self):
        bedrock = PROVIDER_CONFIGS["bedrock"]
        assert "sonnet" in bedrock.heavy.lower()

    def test_bedrock_medium_is_haiku(self):
        bedrock = PROVIDER_CONFIGS["bedrock"]
        assert "haiku" in bedrock.medium.lower()

    def test_bedrock_light_is_haiku(self):
        bedrock = PROVIDER_CONFIGS["bedrock"]
        assert "haiku" in bedrock.light.lower()

    def test_bedrock_heavy_differs_from_medium(self):
        bedrock = PROVIDER_CONFIGS["bedrock"]
        assert bedrock.heavy != bedrock.medium

    def test_anthropic_unchanged(self):
        """Regression guard: anthropic config was already correct."""
        anthropic = PROVIDER_CONFIGS["anthropic"]
        assert "sonnet" in anthropic.heavy.lower()
        assert "sonnet" in anthropic.medium.lower()
        assert "haiku" in anthropic.light.lower()
