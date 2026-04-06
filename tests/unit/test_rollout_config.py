"""Tests for rollout config: always_loaded_tools assignments (Section 08)."""

from pathlib import Path

import pytest

from quantstack.graphs.config import load_agent_configs

_GRAPHS_ROOT = Path(__file__).resolve().parents[2] / "src" / "quantstack" / "graphs"


def _load_configs(graph_name: str):
    yaml_path = _GRAPHS_ROOT / graph_name / "config" / "agents.yaml"
    return load_agent_configs(yaml_path)


class TestSupervisorRollout:

    def test_strategy_promoter_has_always_loaded(self):
        configs = _load_configs("supervisor")
        sp = configs["strategy_promoter"]
        assert len(sp.always_loaded_tools) > 0
        assert "fetch_strategy_registry" in sp.always_loaded_tools
        assert "get_strategy" in sp.always_loaded_tools

    def test_self_healer_has_always_loaded(self):
        configs = _load_configs("supervisor")
        sh = configs["self_healer"]
        assert len(sh.always_loaded_tools) > 0
        assert "check_system_status" in sh.always_loaded_tools

    def test_health_monitor_no_always_loaded(self):
        """<=5 tools, no benefit from deferred loading."""
        configs = _load_configs("supervisor")
        hm = configs["health_monitor"]
        assert len(hm.always_loaded_tools) == 0


class TestResearchRollout:

    def test_quant_researcher_has_always_loaded(self):
        configs = _load_configs("research")
        qr = configs["quant_researcher"]
        assert len(qr.always_loaded_tools) >= 4
        assert "signal_brief" in qr.always_loaded_tools
        assert "search_knowledge_base" in qr.always_loaded_tools

    def test_ml_scientist_has_always_loaded(self):
        configs = _load_configs("research")
        ms = configs["ml_scientist"]
        assert len(ms.always_loaded_tools) >= 4
        assert "train_model" in ms.always_loaded_tools
        assert "search_knowledge_base" in ms.always_loaded_tools

    def test_community_intel_no_always_loaded(self):
        """<=5 tools, no benefit from deferred loading."""
        configs = _load_configs("research")
        ci = configs["community_intel"]
        assert len(ci.always_loaded_tools) == 0


class TestTradingRollout:

    def test_daily_planner_has_always_loaded(self):
        configs = _load_configs("trading")
        dp = configs["daily_planner"]
        assert len(dp.always_loaded_tools) >= 3
        assert "signal_brief" in dp.always_loaded_tools
        assert "fetch_portfolio" in dp.always_loaded_tools

    def test_executor_has_always_loaded(self):
        configs = _load_configs("trading")
        ex = configs["executor"]
        assert len(ex.always_loaded_tools) >= 3
        assert "execute_order" in ex.always_loaded_tools
        assert "check_system_status" in ex.always_loaded_tools

    def test_exit_evaluator_has_always_loaded(self):
        configs = _load_configs("trading")
        ee = configs["exit_evaluator"]
        assert len(ee.always_loaded_tools) >= 3
        assert "fetch_portfolio" in ee.always_loaded_tools
        assert "search_knowledge_base" in ee.always_loaded_tools
        assert "create_exit_signal" in ee.always_loaded_tools


class TestCrossGraphRules:

    @pytest.fixture(params=["supervisor", "research", "trading"])
    def configs(self, request):
        return _load_configs(request.param)

    def test_always_loaded_is_subset_of_tools(self, configs):
        for name, cfg in configs.items():
            always = set(cfg.always_loaded_tools)
            tools = set(cfg.tools)
            assert always.issubset(tools), (
                f"{name}: always_loaded has {always - tools} not in tools"
            )

    def test_no_agent_exceeds_10_always_loaded(self, configs):
        for name, cfg in configs.items():
            assert len(cfg.always_loaded_tools) <= 10, (
                f"{name}: {len(cfg.always_loaded_tools)} always_loaded (max 10)"
            )

    def test_agents_with_5_or_fewer_tools_skip_always_loaded(self, configs):
        # domain_researcher has 5 base tools but 30+ via domain_tool_sets — needs always_loaded
        skip = {"domain_researcher"}
        for name, cfg in configs.items():
            if name in skip:
                continue
            if len(cfg.tools) <= 5:
                assert len(cfg.always_loaded_tools) == 0, (
                    f"{name}: has {len(cfg.tools)} tools but always_loaded is set"
                )

    def test_agents_with_more_than_5_tools_have_always_loaded(self, configs):
        for name, cfg in configs.items():
            # health_monitor is an exception - simple enough that deferred loading adds overhead
            if len(cfg.tools) > 5 and name != "health_monitor":
                assert len(cfg.always_loaded_tools) > 0, (
                    f"{name}: has {len(cfg.tools)} tools but no always_loaded_tools"
                )

    def test_self_healer_tier_upgraded(self):
        configs = _load_configs("supervisor")
        assert configs["self_healer"].llm_tier == "medium"
