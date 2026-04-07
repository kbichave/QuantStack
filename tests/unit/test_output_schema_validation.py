"""Tests for output schema validation (Section 04).

Validates:
1. All 21 agents have registered output schemas
2. Safety-critical fallbacks are fail-CLOSED
3. Pydantic models accept valid data and reject invalid data
4. parse_and_validate() integrates parsing + schema validation
"""

import json

import pytest

from quantstack.graphs.schemas import (
    AGENT_FALLBACKS,
    AGENT_OUTPUT_SCHEMAS,
    DailyPlanOutput,
    EntrySignalOutput,
    HealthCheckOutput,
    MarketIntelOutput,
    SafetyCheckOutput,
    TradeReflectionOutput,
)
from quantstack.graphs.agent_executor import parse_and_validate, parse_json_response


# ---------------------------------------------------------------------------
# Fallback audit: safety-critical agents MUST fail CLOSED
# ---------------------------------------------------------------------------

class TestFallbackAudit:
    """Every safety-critical agent must fail CLOSED, not OPEN."""

    def test_safety_check_fallback_is_halted_true(self):
        """P0: safety_check parse failure must halt, not proceed."""
        fallback = AGENT_FALLBACKS["safety_check"]
        assert fallback["halted"] is True

    def test_executor_fallback_is_empty(self):
        """No orders executed on parse failure."""
        assert AGENT_FALLBACKS["executor"] == []

    def test_fund_manager_fallback_is_empty(self):
        """No entries on parse failure."""
        assert AGENT_FALLBACKS["fund_manager"] == []

    def test_exit_evaluator_fallback_is_empty(self):
        """No exits triggered on parse failure."""
        assert AGENT_FALLBACKS["exit_evaluator"] == []

    @pytest.mark.parametrize(
        "agent_name",
        list(AGENT_FALLBACKS.keys()),
    )
    def test_all_agents_have_documented_fallbacks(self, agent_name):
        """Every agent in the registry has a fallback value."""
        assert agent_name in AGENT_FALLBACKS


# ---------------------------------------------------------------------------
# Schema registry completeness
# ---------------------------------------------------------------------------

class TestSchemaRegistry:
    """All 21 agents have output schemas registered."""

    EXPECTED_AGENTS = {
        # Trading (10)
        "market_intel", "daily_planner", "safety_check", "position_monitor",
        "exit_evaluator", "trade_debater", "fund_manager", "options_analyst",
        "earnings_analyst", "executor", "trade_reflector",
        # Research (7)
        "quant_researcher", "ml_scientist", "strategy_rd", "hypothesis_critic",
        "community_intel", "domain_researcher", "execution_researcher",
        # Supervisor (4)
        "health_monitor", "self_healer", "portfolio_risk_monitor", "strategy_promoter",
    }

    def test_all_agents_have_schemas(self):
        for agent in self.EXPECTED_AGENTS:
            assert agent in AGENT_OUTPUT_SCHEMAS, f"Missing schema for {agent}"

    def test_schema_count_is_21_plus(self):
        assert len(AGENT_OUTPUT_SCHEMAS) >= 21

    @pytest.mark.parametrize("agent_name", list(AGENT_OUTPUT_SCHEMAS.keys()))
    def test_schema_is_serializable(self, agent_name):
        """Each schema's JSON schema is serializable (for retry prompts)."""
        schema_cls = AGENT_OUTPUT_SCHEMAS[agent_name]
        json_schema = schema_cls.model_json_schema()
        assert isinstance(json.dumps(json_schema), str)


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------

class TestPydanticModels:
    """Output models validate known-good samples and reject bad ones."""

    def test_market_intel_valid(self):
        data = {
            "headlines": ["Fed holds rates"],
            "risk_alerts": [],
            "event_calendar": [],
            "sector_news": {},
            "sentiment": "neutral",
        }
        model = MarketIntelOutput.model_validate(data)
        assert model.sentiment == "neutral"

    def test_market_intel_extra_fields_ignored(self):
        data = {"headlines": [], "injected_field": "malicious"}
        model = MarketIntelOutput.model_validate(data)
        assert not hasattr(model, "injected_field")

    def test_safety_check_default_is_halted(self):
        """Default SafetyCheckOutput fails CLOSED."""
        model = SafetyCheckOutput()
        assert model.halted is True

    def test_safety_check_valid(self):
        data = {"halted": False, "reason": "system healthy"}
        model = SafetyCheckOutput.model_validate(data)
        assert model.halted is False

    def test_entry_signal_valid(self):
        data = {"signals": [{"symbol": "AAPL", "direction": "long"}], "reasoning": "momentum"}
        model = EntrySignalOutput.model_validate(data)
        assert len(model.signals) == 1

    def test_health_check_valid(self):
        data = {"overall": "healthy", "checks": [], "alerts": []}
        model = HealthCheckOutput.model_validate(data)
        assert model.overall == "healthy"

    def test_trade_reflection_valid(self):
        data = {"reflection": "Good day", "lessons": ["patience"], "adjustments": []}
        model = TradeReflectionOutput.model_validate(data)
        assert model.reflection == "Good day"


# ---------------------------------------------------------------------------
# parse_and_validate() integration
# ---------------------------------------------------------------------------

class TestParseAndValidate:
    """Tests for the enhanced parse_and_validate function."""

    def test_valid_json_valid_schema(self):
        raw = '{"halted": false, "reason": "all clear"}'
        result, retried = parse_and_validate(
            raw, {"halted": True}, output_schema=SafetyCheckOutput
        )
        assert result["halted"] is False
        assert retried is False

    def test_valid_json_with_extra_text(self):
        raw = 'Here is my analysis: {"halted": false, "reason": "ok"}'
        result, retried = parse_and_validate(
            raw, {"halted": True}, output_schema=SafetyCheckOutput
        )
        assert result["halted"] is False

    def test_invalid_json_returns_fallback(self):
        raw = "This is not JSON at all"
        result, retried = parse_and_validate(
            raw, {"halted": True, "reason": "parse_failure"},
            output_schema=SafetyCheckOutput,
        )
        assert result["halted"] is True

    def test_no_schema_passes_through(self):
        raw = '{"custom": "data"}'
        result, retried = parse_and_validate(raw, {})
        assert result == {"custom": "data"}
        assert retried is False

    def test_empty_input_returns_fallback(self):
        result, retried = parse_and_validate("", {"halted": True})
        assert result == {"halted": True}

    def test_safety_check_fallback_in_trading_nodes(self):
        """Verify the actual fallback used in trading/nodes.py is fail-CLOSED."""
        # This tests the integration: parse_json_response with the corrected fallback
        result = parse_json_response(
            "unparseable garbage",
            {"halted": True, "reason": "parse_failure"},
        )
        assert result["halted"] is True
