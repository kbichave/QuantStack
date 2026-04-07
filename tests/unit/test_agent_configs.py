"""
Tests for agent YAML configuration correctness.

Loads YAML files directly; does not instantiate the graph runtime.
Tests run as part of CI to catch YAML regressions.
"""

import yaml
import pytest
from pathlib import Path

# Paths to agent config files
TRADING_YAML = Path("src/quantstack/graphs/trading/config/agents.yaml")
RESEARCH_YAML = Path("src/quantstack/graphs/research/config/agents.yaml")
SUPERVISOR_YAML = Path("src/quantstack/graphs/supervisor/config/agents.yaml")


@pytest.fixture(scope="module")
def trading_agents():
    return yaml.safe_load(TRADING_YAML.read_text())


@pytest.fixture(scope="module")
def research_agents():
    return yaml.safe_load(RESEARCH_YAML.read_text())


@pytest.fixture(scope="module")
def supervisor_agents():
    return yaml.safe_load(SUPERVISOR_YAML.read_text())


# --- position_monitor split ---

def test_position_monitor_no_write_tools(trading_agents):
    """position_monitor must have no tools that create/update/write to DB."""
    pm = trading_agents["position_monitor"]
    write_prefixes = ("create_", "update_", "write_", "add_")
    write_tools = [
        t for t in pm["tools"]
        if any(t.startswith(p) for p in write_prefixes)
    ]
    assert write_tools == [], f"position_monitor has write tools: {write_tools}"


def test_exit_evaluator_exists(trading_agents):
    """exit_evaluator must exist as a separate agent."""
    assert "exit_evaluator" in trading_agents


def test_exit_evaluator_has_write_tools(trading_agents):
    """exit_evaluator must have create_exit_signal and update_position_stops."""
    tools = trading_agents["exit_evaluator"]["tools"]
    assert "create_exit_signal" in tools
    assert "update_position_stops" in tools


def test_exit_evaluator_has_no_assessment_duplication(trading_agents):
    """
    exit_evaluator should not duplicate heavy data-gathering tools
    already used by position_monitor. It consumes PositionAssessment output.
    """
    # exit_evaluator should NOT have compute_alpha_decay (position_monitor does that)
    assert "compute_alpha_decay" not in trading_agents["exit_evaluator"]["tools"]


# --- fund_manager tool additions ---

def test_fund_manager_has_fetch_portfolio(trading_agents):
    assert "fetch_portfolio" in trading_agents["fund_manager"]["tools"]


def test_fund_manager_has_compute_risk_metrics(trading_agents):
    assert "compute_risk_metrics" in trading_agents["fund_manager"]["tools"]


def test_fund_manager_has_search_knowledge_base(trading_agents):
    assert "search_knowledge_base" in trading_agents["fund_manager"]["tools"]


# --- risk_analyst removal ---

def test_risk_analyst_removed(trading_agents):
    """risk_analyst LLM agent must be removed — replaced by deterministic sizing."""
    assert "risk_analyst" not in trading_agents


# --- domain_researcher consolidation ---

def test_domain_researcher_exists(research_agents):
    assert "domain_researcher" in research_agents


def test_old_domain_researchers_removed(research_agents):
    """The three old per-domain agents must no longer exist."""
    assert "equity_investment_researcher" not in research_agents
    assert "equity_swing_researcher" not in research_agents
    assert "options_researcher" not in research_agents


def test_domain_researcher_has_domain_tool_sets(research_agents):
    dr = research_agents["domain_researcher"]
    assert "domain_tool_sets" in dr
    assert "investment" in dr["domain_tool_sets"]
    assert "swing" in dr["domain_tool_sets"]
    assert "options" in dr["domain_tool_sets"]


def test_domain_researcher_investment_tools_loaded(research_agents):
    """Investment domain must include fundamentals-oriented tools."""
    tools = research_agents["domain_researcher"]["domain_tool_sets"]["investment"]
    assert "fetch_fundamentals" in tools
    assert "get_analyst_estimates" in tools
    assert "get_institutional_accumulation" in tools


def test_domain_researcher_swing_tools_loaded(research_agents):
    """Swing domain must include technical signal tools."""
    tools = research_agents["domain_researcher"]["domain_tool_sets"]["swing"]
    assert "signal_brief" in tools
    assert "get_put_call_ratio" in tools
    assert "compute_technical_indicators" in tools


def test_domain_researcher_options_tools_loaded(research_agents):
    """Options domain must include options-specific tools."""
    tools = research_agents["domain_researcher"]["domain_tool_sets"]["options"]
    assert "fetch_options_chain" in tools
    assert "compute_greeks" in tools
    assert "compute_implied_vol" in tools


def test_domain_researcher_always_loaded_tools(research_agents):
    """All domains share run_backtest, run_walkforward, search_knowledge_base, register_strategy."""
    always = research_agents["domain_researcher"]["always_loaded_tools"]
    assert "run_backtest" in always
    assert "run_walkforward" in always
    assert "search_knowledge_base" in always
    assert "register_strategy" in always


# --- portfolio_risk_monitor in supervisor ---

def test_portfolio_risk_monitor_exists(supervisor_agents):
    assert "portfolio_risk_monitor" in supervisor_agents


def test_portfolio_risk_monitor_tier(supervisor_agents):
    """portfolio_risk_monitor should run at medium tier — it's a weekly scheduled agent."""
    assert supervisor_agents["portfolio_risk_monitor"]["llm_tier"] == "medium"


# --- EventType enum ---

def test_eventtype_has_ic_decay():
    from quantstack.coordination.event_bus import EventType
    assert hasattr(EventType, "IC_DECAY")
    assert EventType.IC_DECAY.value == "ic_decay"


def test_eventtype_has_regime_change():
    from quantstack.coordination.event_bus import EventType
    assert hasattr(EventType, "REGIME_CHANGE")
    assert EventType.REGIME_CHANGE.value == "regime_change"


def test_eventtype_has_risk_alert():
    from quantstack.coordination.event_bus import EventType
    assert hasattr(EventType, "RISK_ALERT")
    assert EventType.RISK_ALERT.value == "risk_alert"


# --- EWF tool wiring (section 08) ---

def test_trade_debater_has_get_ewf_analysis_tools(trading_agents):
    tools = trading_agents["trade_debater"]["tools"]
    always = trading_agents["trade_debater"]["always_loaded_tools"]
    assert "get_ewf_analysis" in tools
    assert "get_ewf_analysis" in always

def test_trade_debater_has_get_ewf_blue_box_setups(trading_agents):
    tools = trading_agents["trade_debater"]["tools"]
    always = trading_agents["trade_debater"]["always_loaded_tools"]
    assert "get_ewf_blue_box_setups" in tools
    assert "get_ewf_blue_box_setups" in always

def test_daily_planner_has_blue_box_tool(trading_agents):
    tools = trading_agents["daily_planner"]["tools"]
    always = trading_agents["daily_planner"]["always_loaded_tools"]
    assert "get_ewf_blue_box_setups" in tools
    assert "get_ewf_blue_box_setups" in always

def test_position_monitor_has_ewf_analysis(trading_agents):
    tools = trading_agents["position_monitor"]["tools"]
    always = trading_agents["position_monitor"]["always_loaded_tools"]
    assert "get_ewf_analysis" in tools
    assert "get_ewf_analysis" in always

def test_exit_evaluator_has_ewf_analysis(trading_agents):
    tools = trading_agents["exit_evaluator"]["tools"]
    assert "get_ewf_analysis" in tools

def test_domain_researcher_swing_has_ewf(research_agents):
    swing_tools = research_agents["domain_researcher"]["domain_tool_sets"]["swing"]
    assert "get_ewf_analysis" in swing_tools

def test_domain_researcher_tools_has_ewf(research_agents):
    tools = research_agents["domain_researcher"]["tools"]
    assert "get_ewf_analysis" in tools

def test_quant_researcher_has_ewf(research_agents):
    tools = research_agents["quant_researcher"]["tools"]
    always = research_agents["quant_researcher"]["always_loaded_tools"]
    assert "get_ewf_analysis" in tools
    assert "get_ewf_analysis" in always

# --- Section 04: Tier reclassification ---

VALID_AGENT_TIERS = {"heavy", "medium", "light"}


@pytest.mark.parametrize("yaml_path", [TRADING_YAML, RESEARCH_YAML, SUPERVISOR_YAML])
def test_all_agents_have_explicit_llm_tier(yaml_path):
    """Every agent in every agents.yaml must have an explicit llm_tier field."""
    config = yaml.safe_load(yaml_path.read_text())
    for agent_name, agent_cfg in config.items():
        assert "llm_tier" in agent_cfg, (
            f"{yaml_path.name}: agent '{agent_name}' is missing llm_tier"
        )


@pytest.mark.parametrize("yaml_path", [TRADING_YAML, RESEARCH_YAML, SUPERVISOR_YAML])
def test_all_tier_values_valid(yaml_path):
    """All llm_tier values must be in {heavy, medium, light}."""
    config = yaml.safe_load(yaml_path.read_text())
    for agent_name, agent_cfg in config.items():
        tier = agent_cfg.get("llm_tier")
        if tier is not None:
            assert tier in VALID_AGENT_TIERS, (
                f"{yaml_path.name}: agent '{agent_name}' has invalid tier '{tier}'. "
                f"Expected one of {VALID_AGENT_TIERS}"
            )


def test_get_chat_model_rejects_unrecognized_tier(monkeypatch):
    """get_chat_model raises ValueError on unrecognized tier."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from quantstack.llm.provider import get_chat_model
    with pytest.raises(ValueError, match="Unknown tier"):
        get_chat_model("nonexistent_tier")


def test_get_model_rejects_unrecognized_tier(monkeypatch):
    """get_model raises ValueError on unrecognized tier."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from quantstack.llm.provider import get_model
    with pytest.raises(ValueError, match="Unknown tier"):
        get_model("nonexistent_tier")


def test_ewf_tool_names_resolve_in_registry():
    from quantstack.tools.registry import TOOL_REGISTRY
    for yaml_path in [TRADING_YAML, RESEARCH_YAML]:
        config = yaml.safe_load(yaml_path.read_text())
        for agent_name, agent_cfg in config.items():
            all_tools = agent_cfg.get("tools", []) + agent_cfg.get("always_loaded_tools", [])
            for domain_tools in agent_cfg.get("domain_tool_sets", {}).values():
                all_tools.extend(domain_tools)
            for tool_name in all_tools:
                if tool_name.startswith("get_ewf"):
                    assert tool_name in TOOL_REGISTRY, (
                        f"{yaml_path.name}: agent '{agent_name}' references tool "
                        f"'{tool_name}' not found in TOOL_REGISTRY"
                    )
