"""Tests for CrewAI agent YAML configurations."""

import yaml
import pytest
from pathlib import Path

CREWS_DIR = Path("src/quantstack/crews")

VALID_TIERS = {"{heavy_model}", "{medium_model}", "{light_model}"}


def load_agents_yaml(crew_name: str) -> dict:
    """Load and parse agents.yaml for a given crew."""
    path = CREWS_DIR / crew_name / "config" / "agents.yaml"
    assert path.exists(), f"Missing agents.yaml at {path}"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("crew_name", ["trading", "research", "supervisor"])
def test_agents_yaml_is_valid_yaml(crew_name):
    """Each crew's agents.yaml must parse as valid YAML."""
    agents = load_agents_yaml(crew_name)
    assert isinstance(agents, dict), "agents.yaml must be a YAML mapping"


@pytest.mark.parametrize("crew_name", ["trading", "research", "supervisor"])
def test_each_agent_has_required_fields(crew_name):
    """Every agent must define role, goal, backstory, and llm."""
    agents = load_agents_yaml(crew_name)
    for agent_id, config in agents.items():
        for field in ("role", "goal", "backstory", "llm"):
            assert field in config, f"Agent '{agent_id}' in {crew_name} missing '{field}'"


@pytest.mark.parametrize("crew_name", ["trading", "research", "supervisor"])
def test_agent_llm_field_references_valid_tier(crew_name):
    """The llm field must reference a tier variable that the crew injects at runtime."""
    agents = load_agents_yaml(crew_name)
    for agent_id, config in agents.items():
        assert config["llm"] in VALID_TIERS, (
            f"Agent '{agent_id}' llm='{config['llm']}' not in {VALID_TIERS}"
        )


@pytest.mark.parametrize("crew_name", ["trading", "research", "supervisor"])
def test_no_agent_allows_delegation(crew_name):
    """No agent should have allow_delegation=true (prevents circular delegation)."""
    agents = load_agents_yaml(crew_name)
    for agent_id, config in agents.items():
        assert config.get("allow_delegation", False) is False, (
            f"Agent '{agent_id}' must not allow delegation"
        )


def test_trading_crew_has_all_required_agents():
    """TradingCrew must define exactly 10 agents."""
    agents = load_agents_yaml("trading")
    expected = {
        "daily_planner", "position_monitor", "trade_debater",
        "risk_analyst", "fund_manager", "options_analyst",
        "earnings_analyst", "market_intel", "trade_reflector", "executor",
    }
    assert set(agents.keys()) == expected


def test_research_crew_has_all_required_agents():
    """ResearchCrew must define exactly 4 agents."""
    agents = load_agents_yaml("research")
    expected = {
        "quant_researcher", "ml_scientist", "strategy_rd", "community_intel",
    }
    assert set(agents.keys()) == expected


def test_supervisor_crew_has_all_required_agents():
    """SupervisorCrew must define exactly 3 agents."""
    agents = load_agents_yaml("supervisor")
    expected = {
        "health_monitor", "self_healer", "strategy_promoter",
    }
    assert set(agents.keys()) == expected


def test_risk_analyst_backstory_mentions_reasoning():
    """Risk analyst must reason about risk, not check hardcoded thresholds."""
    agents = load_agents_yaml("trading")
    backstory = agents["risk_analyst"]["backstory"].lower()
    assert "reason" in backstory, "Risk analyst backstory must mention reasoning"


def test_fund_manager_backstory_mentions_correlation_and_concentration():
    """Fund manager backstory must address correlation and concentration risk."""
    agents = load_agents_yaml("trading")
    backstory = agents["fund_manager"]["backstory"].lower()
    assert "correlation" in backstory
    assert "concentration" in backstory
