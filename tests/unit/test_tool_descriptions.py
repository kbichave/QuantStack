"""Tests for tool description validation (Section 02)."""

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from scripts.validate_tool_descriptions import validate_tool_description


def _make_tool(name: str, description: str, args_schema=None) -> BaseTool:
    """Create a minimal BaseTool mock for testing validation."""
    t = MagicMock(spec=BaseTool)
    t.name = name
    t.description = description
    t.args_schema = args_schema
    return t


class _GoodSchema(BaseModel):
    symbol: str = Field(description="Ticker symbol (e.g., AAPL)")
    lookback: int = Field(default=20, description="Number of bars to analyze")


class _BadSchema(BaseModel):
    symbol: str  # No description
    lookback: int = 20  # No description


class TestValidateToolDescription:

    def test_empty_description_flagged(self):
        t = _make_tool("compute_risk", "")
        violations = validate_tool_description(t)
        assert any("too short" in v for v in violations)

    def test_short_description_flagged(self):
        t = _make_tool("compute_risk", "Compute risk metrics.")
        violations = validate_tool_description(t)
        assert any("too short" in v for v in violations)

    def test_valid_description_passes(self):
        t = _make_tool(
            "compute_risk",
            "Calculate portfolio risk metrics including VaR, CVaR, and maximum drawdown. "
            "Use when assessing position-level or portfolio-level risk exposure. "
            "Returns risk decomposition with confidence intervals.",
            args_schema=_GoodSchema,
        )
        violations = validate_tool_description(t)
        assert violations == []

    def test_argument_descriptions_checked(self):
        t = _make_tool(
            "compute_risk",
            "Calculate portfolio risk metrics including VaR, CVaR, and maximum drawdown. "
            "Use when assessing risk. Returns risk decomposition.",
            args_schema=_BadSchema,
        )
        violations = validate_tool_description(t)
        assert any("argument 'symbol' missing description" in v for v in violations)

    def test_all_args_described_passes(self):
        t = _make_tool(
            "compute_risk",
            "Calculate portfolio risk metrics including VaR, CVaR, and maximum drawdown. "
            "Use when assessing risk. Returns risk decomposition.",
            args_schema=_GoodSchema,
        )
        violations = validate_tool_description(t)
        assert not any("argument" in v for v in violations)

    def test_generic_description_flagged(self):
        t = _make_tool("compute_risk", "Compute risk.")
        violations = validate_tool_description(t)
        assert len(violations) > 0  # At minimum, too short

    def test_lacks_actionable_guidance_flagged(self):
        """A long description without actionable phrases is flagged."""
        long_desc = "This tool does various things with market data and generates several outputs for the portfolio across multiple asset classes and time horizons."
        t = _make_tool("compute_risk", long_desc)
        violations = validate_tool_description(t)
        assert any("actionable guidance" in v for v in violations)


class TestAllRegisteredToolsPass:
    """Integration test — validates all tools in TOOL_REGISTRY after rewrite."""

    def test_all_tools_pass_description_validation(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        failures = {}
        for name, tool_obj in TOOL_REGISTRY.items():
            violations = validate_tool_description(tool_obj)
            if violations:
                failures[name] = violations

        if failures:
            msg_lines = [f"\n{len(failures)} tool(s) failed description validation:"]
            for name, viols in sorted(failures.items()):
                msg_lines.append(f"  {name}: {'; '.join(viols)}")
            pytest.fail("\n".join(msg_lines))
