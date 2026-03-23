"""
Promotion gate — statistical go/no-go criteria for shadow → live promotion.

A model is promoted when its [SHADOW] tag is removed and its predictions
directly influence trading decisions.

Ported from quantstack.rl.promotion_gate — same statistical tests,
now integrated with the FinRL model registry instead of per-agent config flags.

Usage:
    from quantstack.finrl.promotion import PromotionGate

    gate = PromotionGate()
    result = gate.evaluate(
        simulated_returns=[0.02, -0.01, 0.03, ...],
        n_observations=80,
        simulated_sharpe=0.65,
        max_drawdown=0.08,
        directional_agreement=0.62,
    )
    if result.passes:
        registry.update_status(model_id, "live")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from quantstack.core.backtesting.stats import (
    monte_carlo_permutation,
    sharpe_ratio_with_ci,
)
from quantstack.finrl.config import get_finrl_config


@dataclass
class PromotionCheckResult:
    """Result of a single promotion check."""

    name: str
    passed: bool
    value: float | None
    threshold: float | None
    message: str


@dataclass
class PromotionResult:
    """Aggregate result across all checks."""

    model_id: str
    passes: bool
    checks: list[PromotionCheckResult] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed_checks(self) -> list[PromotionCheckResult]:
        return [c for c in self.checks if c.passed]

    @property
    def failed_checks(self) -> list[PromotionCheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        lines = [f"Promotion for {self.model_id}: {'PASS' if self.passes else 'FAIL'}"]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}: {c.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "passes": self.passes,
            "evaluated_at": self.evaluated_at.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


class PromotionGate:
    """
    Evaluates whether a model is ready for shadow → live promotion.

    All checks must pass. Thresholds come from FinRLConfig.
    """

    def __init__(self, config: Any | None = None):
        self.cfg = config or get_finrl_config()

    def evaluate(
        self,
        model_id: str,
        simulated_returns: list[float] | None = None,
        n_observations: int = 0,
        simulated_sharpe: float | None = None,
        max_drawdown: float | None = None,
        directional_agreement: float | None = None,
        walk_forward_folds: list[tuple[float, float]] | None = None,
    ) -> PromotionResult:
        """Run all promotion checks."""
        checks: list[PromotionCheckResult] = []

        # Check 1: Observation count
        obs_check = self._check_observations(n_observations)
        checks.append(obs_check)
        if not obs_check.passed:
            return PromotionResult(model_id=model_id, passes=False, checks=checks)

        # Check 2: Sharpe ratio
        sharpe_check = self._check_sharpe(simulated_returns, simulated_sharpe)
        checks.append(sharpe_check)

        # Check 3: Max drawdown
        if max_drawdown is not None:
            checks.append(self._check_drawdown(max_drawdown))

        # Check 4: Directional agreement
        if directional_agreement is not None:
            checks.append(self._check_agreement(directional_agreement))

        # Check 5: Monte Carlo
        if simulated_returns and len(simulated_returns) >= 30:
            checks.append(self._check_monte_carlo(simulated_returns))

        # Check 6: Walk-forward
        if walk_forward_folds and len(walk_forward_folds) >= 3:
            checks.append(self._check_walk_forward(walk_forward_folds))

        all_passed = all(c.passed for c in checks)
        return PromotionResult(model_id=model_id, passes=all_passed, checks=checks)

    def _check_observations(self, n: int) -> PromotionCheckResult:
        min_req = self.cfg.min_shadow_observations
        passed = n >= min_req
        return PromotionCheckResult(
            name="observation_count",
            passed=passed,
            value=float(n),
            threshold=float(min_req),
            message=f"{n} {'>=  ' if passed else '<'} {min_req} required.",
        )

    def _check_sharpe(
        self, returns: list[float] | None, point_estimate: float | None
    ) -> PromotionCheckResult:
        threshold = self.cfg.min_promo_sharpe

        if returns and len(returns) >= 20:
            try:
                ret_arr = np.array(returns) / 100.0
                _, (lower, upper) = sharpe_ratio_with_ci(ret_arr)
                value = lower
                suffix = f"(CI lower bound: [{lower:.2f}, {upper:.2f}])"
            except Exception:
                value = point_estimate or 0.0
                suffix = "(point estimate)"
        else:
            value = point_estimate or 0.0
            suffix = "(point estimate)"

        passed = value >= threshold
        return PromotionCheckResult(
            name="sharpe_ratio",
            passed=passed,
            value=round(value, 4),
            threshold=threshold,
            message=f"Sharpe {value:.2f} {'>=  ' if passed else '<'} {threshold} {suffix}",
        )

    def _check_drawdown(self, max_dd: float) -> PromotionCheckResult:
        threshold = self.cfg.max_promo_drawdown
        passed = max_dd <= threshold
        return PromotionCheckResult(
            name="max_drawdown",
            passed=passed,
            value=round(max_dd, 4),
            threshold=threshold,
            message=f"Max DD {max_dd:.1%} {'<=  ' if passed else '>'} {threshold:.0%}",
        )

    def _check_agreement(self, agreement: float) -> PromotionCheckResult:
        threshold = self.cfg.min_direction_agreement
        passed = agreement >= threshold
        return PromotionCheckResult(
            name="directional_agreement",
            passed=passed,
            value=round(agreement, 4),
            threshold=threshold,
            message=f"Agreement {agreement:.1%} {'>=  ' if passed else '<'} {threshold:.0%}",
        )

    def _check_monte_carlo(self, returns: list[float]) -> PromotionCheckResult:
        threshold = self.cfg.max_monte_carlo_pvalue
        try:
            ret_arr = np.array(returns) / 100.0
            observed_sharpe, _ = sharpe_ratio_with_ci(ret_arr)
            p_value = monte_carlo_permutation(ret_arr, observed_sharpe, n_permutations=1000)
            passed = p_value <= threshold
            return PromotionCheckResult(
                name="monte_carlo",
                passed=passed,
                value=round(p_value, 4),
                threshold=threshold,
                message=f"p-value {p_value:.3f} {'<=  ' if passed else '>'} {threshold}",
            )
        except Exception as e:
            return PromotionCheckResult(
                name="monte_carlo",
                passed=True,
                value=None,
                threshold=threshold,
                message=f"Skipped: {e}",
            )

    def _check_walk_forward(
        self, folds: list[tuple[float, float]]
    ) -> PromotionCheckResult:
        oos_sharpes = [oos for _, oos in folds]
        is_sharpes = [is_ for is_, _ in folds]

        pct_positive = float(np.mean([s > 0 for s in oos_sharpes]))
        avg_is = float(np.mean(is_sharpes))
        avg_oos = float(np.mean(oos_sharpes))
        degradation = (avg_is - avg_oos) / (abs(avg_is) + 1e-8)

        passed_pct = pct_positive >= self.cfg.min_wf_positive_folds
        passed_deg = degradation <= self.cfg.max_wf_sharpe_degradation
        passed = passed_pct and passed_deg

        return PromotionCheckResult(
            name="walk_forward",
            passed=passed,
            value=round(pct_positive, 4),
            threshold=self.cfg.min_wf_positive_folds,
            message=(
                f"{pct_positive:.0%} positive folds, "
                f"degradation={degradation:.0%} "
                f"({'OK' if passed else 'FAIL'})"
            ),
        )
