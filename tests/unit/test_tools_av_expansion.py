"""Tests for AV data expansion tools and agent config."""
import re

import pytest
import yaml
from pydantic import ValidationError


class TestToolDefinitions:
    def test_get_put_call_ratio_has_valid_description(self):
        from quantstack.tools.langchain.data_tools import get_put_call_ratio

        assert get_put_call_ratio.description
        assert len(get_put_call_ratio.description) > 20

    def test_get_earnings_momentum_has_valid_description(self):
        from quantstack.tools.langchain.data_tools import get_earnings_momentum

        assert get_earnings_momentum.description

    def test_get_commodity_signals_has_valid_description(self):
        from quantstack.tools.langchain.data_tools import get_commodity_signals

        assert get_commodity_signals.description

    def test_get_forex_rates_has_valid_description(self):
        from quantstack.tools.langchain.data_tools import get_forex_rates

        assert get_forex_rates.description

    def test_check_listing_status_has_valid_description(self):
        from quantstack.tools.langchain.data_tools import check_listing_status

        assert check_listing_status.description


class TestInputModels:
    def test_put_call_ratio_input_defaults(self):
        from quantstack.tools.models import PutCallRatioInput

        m = PutCallRatioInput(symbol="AAPL")
        assert m.lookback_days == 30

    def test_put_call_ratio_input_requires_symbol(self):
        from quantstack.tools.models import PutCallRatioInput

        with pytest.raises(ValidationError):
            PutCallRatioInput()

    def test_earnings_momentum_input_defaults(self):
        from quantstack.tools.models import EarningsMomentumInput

        m = EarningsMomentumInput(symbol="AAPL")
        assert m.quarters == 8

    def test_commodity_signals_input_defaults(self):
        from quantstack.tools.models import CommoditySignalsInput

        m = CommoditySignalsInput()
        assert m.lookback_days == 60


class TestToolRegistration:
    def test_all_five_tools_in_registry(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        expected = [
            "get_put_call_ratio",
            "get_earnings_momentum",
            "get_commodity_signals",
            "get_forex_rates",
            "check_listing_status",
        ]
        for name in expected:
            assert name in TOOL_REGISTRY, f"{name} not in TOOL_REGISTRY"

    def test_tool_names_are_snake_case(self):
        expected = [
            "get_put_call_ratio",
            "get_earnings_momentum",
            "get_commodity_signals",
            "get_forex_rates",
            "check_listing_status",
        ]
        for name in expected:
            assert re.match(r"^[a-z][a-z0-9_]*$", name), f"{name} not snake_case"


class TestAgentYAMLConfigs:
    def test_research_yaml_valid(self):
        from pathlib import Path

        p = Path("src/quantstack/graphs/research/config/agents.yaml")
        data = yaml.safe_load(p.read_text())
        assert isinstance(data, dict)

    def test_trading_yaml_valid(self):
        from pathlib import Path

        p = Path("src/quantstack/graphs/trading/config/agents.yaml")
        data = yaml.safe_load(p.read_text())
        assert isinstance(data, dict)

    def test_supervisor_yaml_valid(self):
        from pathlib import Path

        p = Path("src/quantstack/graphs/supervisor/config/agents.yaml")
        data = yaml.safe_load(p.read_text())
        assert isinstance(data, dict)

    def test_research_has_earnings_and_commodity_tools(self):
        from pathlib import Path

        p = Path("src/quantstack/graphs/research/config/agents.yaml")
        data = yaml.safe_load(p.read_text())
        all_tools = []
        for agent_name, agent_conf in data.items():
            if isinstance(agent_conf, dict) and "tools" in agent_conf:
                all_tools.extend(agent_conf["tools"])
        assert "get_earnings_momentum" in all_tools
        assert "get_commodity_signals" in all_tools

    def test_trading_has_pcr_and_earnings_tools(self):
        from pathlib import Path

        p = Path("src/quantstack/graphs/trading/config/agents.yaml")
        data = yaml.safe_load(p.read_text())
        all_tools = []
        for agent_name, agent_conf in data.items():
            if isinstance(agent_conf, dict) and "tools" in agent_conf:
                all_tools.extend(agent_conf["tools"])
        assert "get_put_call_ratio" in all_tools
        assert "get_earnings_momentum" in all_tools

    def test_supervisor_has_listing_status_tool(self):
        from pathlib import Path

        p = Path("src/quantstack/graphs/supervisor/config/agents.yaml")
        data = yaml.safe_load(p.read_text())
        all_tools = []
        for agent_name, agent_conf in data.items():
            if isinstance(agent_conf, dict) and "tools" in agent_conf:
                all_tools.extend(agent_conf["tools"])
        assert "check_listing_status" in all_tools
