"""Programmatic safety boundary for LLM-reasoned risk decisions.

The LLM risk agent reasons about position sizing and trade approval.
This module validates every LLM decision against hard outer limits
that prevent catastrophic outcomes from hallucination or prompt injection.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyGateLimits:
    """Outer envelope limits for LLM-reasoned risk decisions.

    Intentionally wider than what a reasonable LLM would recommend.
    These catch anomalies, not constrain normal operation.
    """
    max_position_pct: float = 0.15
    daily_loss_halt_pct: float = 0.03
    min_adv: int = 200_000
    max_gross_exposure_pct: float = 2.00
    max_options_premium_pct: float = 0.10


@dataclass
class RiskDecision:
    """Structured output from the LLM risk agent."""
    symbol: str
    recommended_size_pct: float
    reasoning: str
    confidence: float
    approved: bool = True


@dataclass
class RiskVerdict:
    """Result of safety gate validation."""
    approved: bool
    violations: list[str] = field(default_factory=list)
    violation_rule: str | None = None


class SafetyGate:
    """Programmatic safety boundary around LLM risk decisions.

    Validates every RiskDecision against hard outer limits before
    allowing execution.
    """

    def __init__(self, limits: SafetyGateLimits | None = None) -> None:
        self.limits = limits or SafetyGateLimits()

    def validate(
        self,
        decision: RiskDecision,
        portfolio_context: dict,
    ) -> RiskVerdict:
        """Check an LLM risk decision against hard safety limits.

        Args:
            decision: The LLM's risk recommendation.
            portfolio_context: Full portfolio state from get_portfolio_context_tool.

        Returns:
            RiskVerdict with approved=True if all checks pass,
            or approved=False with violation details.
        """
        violations: list[str] = []
        equity = portfolio_context.get("total_equity", 0)

        # Daily loss halt — deterministic, checked first
        daily_pnl_pct = abs(portfolio_context.get("daily_pnl_pct", 0))
        if daily_pnl_pct >= self.limits.daily_loss_halt_pct:
            return RiskVerdict(
                approved=False,
                violations=[
                    f"Daily loss {daily_pnl_pct:.2%} exceeds halt threshold "
                    f"{self.limits.daily_loss_halt_pct:.2%}"
                ],
                violation_rule="daily_loss_halt",
            )

        # Max position size
        size_pct = decision.recommended_size_pct / 100.0  # convert from percentage
        if size_pct > self.limits.max_position_pct:
            violations.append(
                f"Position size {size_pct:.2%} exceeds max {self.limits.max_position_pct:.2%}"
            )
            return RiskVerdict(
                approved=False,
                violations=violations,
                violation_rule="max_position_size",
            )

        # Min liquidity (ADV)
        adv = portfolio_context.get("adv", portfolio_context.get("average_daily_volume", 0))
        if adv and adv < self.limits.min_adv:
            violations.append(
                f"ADV {adv:,.0f} below minimum {self.limits.min_adv:,.0f}"
            )
            return RiskVerdict(
                approved=False,
                violations=violations,
                violation_rule="min_liquidity",
            )

        # Max gross exposure
        current_exposure = portfolio_context.get("gross_exposure_pct", 0)
        proposed_exposure = current_exposure + size_pct
        if proposed_exposure > self.limits.max_gross_exposure_pct:
            violations.append(
                f"Gross exposure {proposed_exposure:.2%} exceeds max "
                f"{self.limits.max_gross_exposure_pct:.2%}"
            )
            return RiskVerdict(
                approved=False,
                violations=violations,
                violation_rule="max_gross_exposure",
            )

        # Max options premium at risk
        options_premium_pct = portfolio_context.get("options_premium_pct", 0)
        if options_premium_pct > self.limits.max_options_premium_pct:
            violations.append(
                f"Options premium at risk {options_premium_pct:.2%} exceeds max "
                f"{self.limits.max_options_premium_pct:.2%}"
            )
            return RiskVerdict(
                approved=False,
                violations=violations,
                violation_rule="max_options_premium",
            )

        return RiskVerdict(approved=True)
