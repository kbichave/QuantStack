"""BM25 accuracy harness: verify deferred tools are discoverable via search.

Run before deploying each graph to catch description quality regressions.
These tests do NOT call the Anthropic API. They validate that tool descriptions
contain the vocabulary agents use, which is a proxy for BM25 discoverability.
"""

import pathlib

import pytest

from quantstack.graphs.config import load_agent_configs
from quantstack.tools.registry import TOOL_REGISTRY

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantstack"

# Sample search queries derived from agent backstories/goals.
# Each query keyword should appear in at least one of the agent's tool descriptions.
AGENT_SEARCH_QUERIES = {
    "daily_planner": ["regime", "signal brief", "portfolio", "equity curve"],
    "risk_analyst": ["VaR", "position sizing", "drawdown", "stress test", "risk limits"],
    "entry_scanner": ["momentum", "entry", "technical", "screen"],
    "quant_researcher": ["backtest", "strategy registry", "signal", "information coefficient"],
    "ml_scientist": ["train model", "feature", "drift", "prediction"],
    "options_analyst": ["Greeks", "implied volatility", "IV surface", "option"],
    "trade_debater": ["signal", "portfolio", "sentiment", "knowledge"],
    "fund_manager": ["portfolio", "risk", "strategy performance"],
    "position_monitor": ["position", "exit", "stop", "portfolio", "alert"],
}


def _load_all_agent_configs():
    """Load agent configs from all three graph YAML files."""
    configs = {}
    for graph in ("trading", "research", "supervisor"):
        yaml_path = SRC_ROOT / "graphs" / graph / "config" / "agents.yaml"
        if yaml_path.exists():
            configs.update(load_agent_configs(yaml_path))
    return configs


class TestBM25Discoverability:
    """Verify tool descriptions contain vocabulary matching agent search queries."""

    @pytest.fixture
    def tool_descriptions(self):
        return {name: tool.description.lower() for name, tool in TOOL_REGISTRY.items()}

    @pytest.fixture
    def agent_configs(self):
        return _load_all_agent_configs()

    def test_each_agent_tool_has_matching_vocabulary(self, tool_descriptions, agent_configs):
        """For each agent in AGENT_SEARCH_QUERIES, verify that at least one of
        the agent's tools has a description containing at least one keyword."""
        failures = []
        for agent_name, queries in AGENT_SEARCH_QUERIES.items():
            if agent_name not in agent_configs:
                continue
            cfg = agent_configs[agent_name]
            agent_tool_descs = [
                tool_descriptions[t] for t in cfg.tools if t in tool_descriptions
            ]
            if not agent_tool_descs:
                continue

            for query in queries:
                query_lower = query.lower()
                # Check if any tool description contains this query term
                found = any(query_lower in desc for desc in agent_tool_descs)
                if not found:
                    # Check individual words as fallback
                    words = query_lower.split()
                    found = any(
                        any(w in desc for w in words) for desc in agent_tool_descs
                    )
                if not found:
                    failures.append(f"{agent_name}: no tool matches query '{query}'")

        if failures:
            pytest.fail(
                f"BM25 vocabulary gaps ({len(failures)} total):\n"
                + "\n".join(failures[:20])
            )

    def test_no_tool_has_empty_description(self, tool_descriptions):
        """All tools in the registry must have non-empty descriptions."""
        empty = [n for n, d in tool_descriptions.items() if not d.strip()]
        assert not empty, f"Tools with empty descriptions: {empty}"

    def test_tool_descriptions_meet_minimum_length(self, tool_descriptions):
        """Descriptions shorter than 20 chars are insufficient for BM25."""
        short = {n: d for n, d in tool_descriptions.items() if len(d.strip()) < 20}
        assert not short, (
            f"Tools with descriptions <20 chars: {list(short.keys())}"
        )

    def test_deferred_tools_have_rich_descriptions(self, tool_descriptions, agent_configs):
        """Deferred tools (in tools but not in always_loaded_tools) need
        especially rich descriptions since they rely entirely on BM25."""
        thin_deferred = []
        for name, cfg in agent_configs.items():
            if not cfg.always_loaded_tools:
                continue
            always_set = set(cfg.always_loaded_tools)
            for tool_name in cfg.tools:
                if tool_name in always_set:
                    continue
                desc = tool_descriptions.get(tool_name, "")
                if len(desc) < 80:
                    thin_deferred.append(f"{name}/{tool_name}: {len(desc)} chars")

        assert not thin_deferred, (
            f"Deferred tools with thin descriptions (<80 chars):\n"
            + "\n".join(thin_deferred)
        )
