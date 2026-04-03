"""Tests for agent configuration system (Section 03)."""
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import pytest
import yaml

VALID_YAML = """
quant_researcher:
  role: "Senior Quantitative Researcher"
  goal: "Discover alpha strategies"
  backstory: "Expert researcher"
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - fetch_market_data

ml_scientist:
  role: "Machine Learning Scientist"
  goal: "Design training experiments"
  backstory: "ML expert"
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - train_model
"""


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_loads_from_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config import load_agent_configs
        configs = load_agent_configs(yaml_file)
        assert "quant_researcher" in configs
        cfg = configs["quant_researcher"]
        assert cfg.name == "quant_researcher"
        assert cfg.role == "Senior Quantitative Researcher"
        assert cfg.llm_tier == "heavy"
        assert cfg.max_iterations == 20
        assert cfg.tools == ("signal_brief", "fetch_market_data")

    def test_rejects_missing_required_fields(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("""
bad_agent:
  goal: "something"
  backstory: "something"
  llm_tier: heavy
""")
        from quantstack.graphs.config import load_agent_configs
        with pytest.raises(ValueError, match="role"):
            load_agent_configs(yaml_file)

    def test_rejects_invalid_llm_tier(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("""
agent:
  role: "Test"
  goal: "Test"
  backstory: "Test"
  llm_tier: turbo
""")
        from quantstack.graphs.config import load_agent_configs
        with pytest.raises(ValueError, match="llm_tier"):
            load_agent_configs(yaml_file)

    def test_returns_dict_of_agent_configs(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config import AgentConfig, load_agent_configs
        configs = load_agent_configs(yaml_file)
        assert len(configs) == 2
        for name, cfg in configs.items():
            assert isinstance(cfg, AgentConfig)

    def test_raises_on_duplicate_agent_names(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("""
agent_a:
  role: "First"
  goal: "First"
  backstory: "First"
  llm_tier: heavy

agent_a:
  role: "Second"
  goal: "Second"
  backstory: "Second"
  llm_tier: medium
""")
        from quantstack.graphs.config import load_agent_configs
        with pytest.raises(ValueError, match="Duplicate"):
            load_agent_configs(yaml_file)

    def test_tool_references_cross_validate(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config import load_agent_configs, validate_tool_references
        configs = load_agent_configs(yaml_file)
        registry = {"signal_brief": lambda: None, "fetch_market_data": lambda: None}
        # ml_scientist references "train_model" which is NOT in registry
        with pytest.raises(ValueError, match="train_model"):
            validate_tool_references(configs, registry)

    def test_frozen_config(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config import load_agent_configs
        configs = load_agent_configs(yaml_file)
        with pytest.raises(AttributeError):
            configs["quant_researcher"].role = "Changed"


class TestConfigWatcher:
    """Tests for ConfigWatcher hot-reload."""

    def test_get_config_returns_current(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config_watcher import ConfigWatcher
        watcher = ConfigWatcher(yaml_file)
        cfg = watcher.get_config("quant_researcher")
        assert cfg.role == "Senior Quantitative Researcher"
        watcher.stop()

    def test_get_all_configs(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config_watcher import ConfigWatcher
        watcher = ConfigWatcher(yaml_file)
        configs = watcher.get_all_configs()
        assert len(configs) == 2
        watcher.stop()

    def test_missing_agent_raises_key_error(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config_watcher import ConfigWatcher
        watcher = ConfigWatcher(yaml_file)
        with pytest.raises(KeyError):
            watcher.get_config("nonexistent")
        watcher.stop()

    def test_reload_at_cycle_boundary_only(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config_watcher import ConfigWatcher
        watcher = ConfigWatcher(yaml_file)

        # Modify YAML
        updated = VALID_YAML.replace("Senior Quantitative Researcher", "Updated Role")
        yaml_file.write_text(updated)
        # Manually stage reload
        watcher._stage_reload()

        # Before apply, should still have old config
        assert watcher.get_config("quant_researcher").role == "Senior Quantitative Researcher"

        # After apply, should have new config
        assert watcher.apply_pending_reload() is True
        assert watcher.get_config("quant_researcher").role == "Updated Role"
        watcher.stop()

    def test_reload_is_atomic(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config_watcher import ConfigWatcher
        watcher = ConfigWatcher(yaml_file)

        errors = []

        def reader():
            for _ in range(100):
                try:
                    cfg = watcher.get_config("quant_researcher")
                    assert cfg.role is not None
                    assert cfg.goal is not None
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()

        # Trigger reload mid-read
        updated = VALID_YAML.replace("Senior Quantitative Researcher", "Atomic Test")
        yaml_file.write_text(updated)
        watcher._stage_reload()
        watcher.apply_pending_reload()

        for t in threads:
            t.join()

        assert not errors, f"Concurrent reads failed: {errors}"
        watcher.stop()

    @pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP not available")
    def test_sighup_reload(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(VALID_YAML)
        from quantstack.graphs.config_watcher import ConfigWatcher
        watcher = ConfigWatcher(yaml_file)
        watcher.register_sighup_handler()

        # Modify YAML and send SIGHUP
        updated = VALID_YAML.replace("Senior Quantitative Researcher", "SIGHUP Updated")
        yaml_file.write_text(updated)
        os.kill(os.getpid(), signal.SIGHUP)

        # Apply pending reload
        assert watcher.apply_pending_reload() is True
        assert watcher.get_config("quant_researcher").role == "SIGHUP Updated"
        watcher.stop()


class TestProductionYamlFiles:
    """Validate the actual agent YAML files in the project."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = PROJECT_ROOT / "src" / "quantstack"

    @pytest.mark.parametrize("graph", ["research", "trading", "supervisor"])
    def test_yaml_file_exists(self, graph):
        path = self.SRC_ROOT / "graphs" / graph / "config" / "agents.yaml"
        assert path.is_file(), f"Missing {path}"

    @pytest.mark.parametrize("graph", ["research", "trading", "supervisor"])
    def test_yaml_loads_successfully(self, graph):
        from quantstack.graphs.config import load_agent_configs
        path = self.SRC_ROOT / "graphs" / graph / "config" / "agents.yaml"
        configs = load_agent_configs(path)
        assert len(configs) > 0

    def test_research_agents(self):
        from quantstack.graphs.config import load_agent_configs
        path = self.SRC_ROOT / "graphs" / "research" / "config" / "agents.yaml"
        configs = load_agent_configs(path)
        assert "quant_researcher" in configs
        assert "ml_scientist" in configs
        assert "strategy_rd" in configs
        assert "community_intel" in configs

    def test_trading_agents(self):
        from quantstack.graphs.config import load_agent_configs
        path = self.SRC_ROOT / "graphs" / "trading" / "config" / "agents.yaml"
        configs = load_agent_configs(path)
        expected = {
            "daily_planner", "position_monitor", "trade_debater",
            "risk_analyst", "fund_manager", "options_analyst",
            "earnings_analyst", "market_intel", "trade_reflector", "executor",
        }
        assert expected.issubset(configs.keys())

    def test_supervisor_agents(self):
        from quantstack.graphs.config import load_agent_configs
        path = self.SRC_ROOT / "graphs" / "supervisor" / "config" / "agents.yaml"
        configs = load_agent_configs(path)
        assert "health_monitor" in configs
        assert "self_healer" in configs
        assert "strategy_promoter" in configs
