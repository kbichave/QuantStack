"""Tests for Section 12: Risk & Safety — Programmatic Safety Boundary."""

import pytest

from quantstack.core.risk.safety_gate import (
    SafetyGate,
    SafetyGateLimits,
    RiskDecision,
    RiskVerdict,
)


@pytest.fixture()
def gate():
    return SafetyGate()


@pytest.fixture()
def base_context():
    return {
        "total_equity": 25000.0,
        "cash_available": 12000.0,
        "daily_pnl": -100.0,
        "daily_pnl_pct": -0.004,
        "gross_exposure_pct": 0.80,
        "net_exposure_pct": 0.60,
        "adv": 5_000_000,
        "options_premium_pct": 0.02,
        "current_regime": "trending_up",
        "vix_level": 18.5,
    }


class TestSafetyGateRejects:
    """Safety gate must reject unsafe LLM recommendations."""

    def test_rejects_position_above_15_pct(self, gate, base_context):
        decision = RiskDecision(
            symbol="AAPL", recommended_size_pct=20.0,
            reasoning="High conviction", confidence=0.9,
        )
        verdict = gate.validate(decision, base_context)
        assert verdict.approved is False
        assert verdict.violation_rule == "max_position_size"

    def test_rejects_daily_loss_above_3_pct(self, gate, base_context):
        base_context["daily_pnl"] = -800.0
        base_context["daily_pnl_pct"] = -0.032
        decision = RiskDecision(
            symbol="AAPL", recommended_size_pct=5.0,
            reasoning="Small position", confidence=0.7,
        )
        verdict = gate.validate(decision, base_context)
        assert verdict.approved is False
        assert verdict.violation_rule == "daily_loss_halt"

    def test_rejects_low_adv(self, gate, base_context):
        base_context["adv"] = 150_000
        decision = RiskDecision(
            symbol="TINY", recommended_size_pct=5.0,
            reasoning="Good setup", confidence=0.8,
        )
        verdict = gate.validate(decision, base_context)
        assert verdict.approved is False
        assert verdict.violation_rule == "min_liquidity"

    def test_rejects_gross_exposure_above_200_pct(self, gate, base_context):
        base_context["gross_exposure_pct"] = 1.90
        decision = RiskDecision(
            symbol="SPY", recommended_size_pct=15.0,
            reasoning="Big move expected", confidence=0.95,
        )
        verdict = gate.validate(decision, base_context)
        assert verdict.approved is False
        assert verdict.violation_rule == "max_gross_exposure"

    def test_rejects_options_premium_above_10_pct(self, gate, base_context):
        base_context["options_premium_pct"] = 0.112
        decision = RiskDecision(
            symbol="TSLA", recommended_size_pct=5.0,
            reasoning="Options play", confidence=0.6,
        )
        verdict = gate.validate(decision, base_context)
        assert verdict.approved is False
        assert verdict.violation_rule == "max_options_premium"


class TestSafetyGatePasses:
    """Safety gate must approve valid recommendations."""

    def test_passes_valid_recommendation(self, gate, base_context):
        decision = RiskDecision(
            symbol="AAPL", recommended_size_pct=8.0,
            reasoning="Moderate conviction, good setup", confidence=0.75,
        )
        verdict = gate.validate(decision, base_context)
        assert verdict.approved is True
        assert verdict.violations == []


class TestRiskDecisionSchema:
    """Risk decision data model validation."""

    def test_risk_decision_has_required_fields(self):
        d = RiskDecision(
            symbol="AAPL", recommended_size_pct=5.0,
            reasoning="Good setup", confidence=0.8,
        )
        assert d.symbol == "AAPL"
        assert d.recommended_size_pct == 5.0
        assert d.reasoning == "Good setup"
        assert d.confidence == 0.8
        assert d.approved is True

    def test_risk_verdict_defaults(self):
        v = RiskVerdict(approved=True)
        assert v.violations == []
        assert v.violation_rule is None


class TestSafetyGateLimits:
    """Verify default limits match spec."""

    def test_default_limits(self):
        limits = SafetyGateLimits()
        assert limits.max_position_pct == 0.15
        assert limits.daily_loss_halt_pct == 0.03
        assert limits.min_adv == 200_000
        assert limits.max_gross_exposure_pct == 2.00
        assert limits.max_options_premium_pct == 0.10

    def test_custom_limits(self):
        limits = SafetyGateLimits(max_position_pct=0.10, min_adv=500_000)
        gate = SafetyGate(limits=limits)
        decision = RiskDecision(
            symbol="AAPL", recommended_size_pct=12.0,
            reasoning="Test", confidence=0.5,
        )
        context = {
            "total_equity": 25000, "daily_pnl_pct": 0,
            "gross_exposure_pct": 0.5, "adv": 1_000_000,
            "options_premium_pct": 0.01,
        }
        verdict = gate.validate(decision, context)
        assert verdict.approved is False
        assert verdict.violation_rule == "max_position_size"
