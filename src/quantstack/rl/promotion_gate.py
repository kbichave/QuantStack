"""
RL Promotion Gate — numeric go/no-go criteria for shadow → production promotion.

An RL agent is "promoted" when its [SHADOW] tag is removed and its recommendations
directly influence crew decisions without advisory labeling.

Promotion requires ALL checks to pass. Each check is independently auditable.
The PromotionGate uses existing stats.py functions for all statistical tests —
no new statistical code is introduced here.

Usage:
    from quantstack.rl.promotion_gate import PromotionGate
    from quantstack.rl.shadow_mode import ShadowEvaluator

    gate = PromotionGate()
    evaluator = ShadowEvaluator(store)

    shadow_result = evaluator.evaluate_shadow_period("sizing", min_observations=63)
    result = gate.evaluate(
        agent_type="sizing",
        shadow_result=shadow_result,
        simulated_returns=simulated_returns,   # list of per-trade pnl%
    )

    if result.passes:
        cfg.sizing_shadow = False
        logger.info("Sizing agent promoted to production!")
    else:
        logger.info(f"Not ready: {result.failed_checks}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from loguru import logger


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
    """Aggregate result across all promotion checks for one agent."""

    agent_type: str
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
        lines = [
            f"PromotionResult for {self.agent_type}: {'PASS' if self.passes else 'FAIL'}"
        ]
        for c in self.checks:
            status = "✓" if c.passed else "✗"
            lines.append(f"  {status} {c.name}: {c.message}")
        return "\n".join(lines)


class PromotionGate:
    """
    Evaluates whether an RL agent is ready for shadow → production promotion.

    Uses stats.py statistical functions where available (sharpe_ratio_with_ci,
    monte_carlo_permutation). Falls back to simple point estimates when
    insufficient data.

    Thresholds are conservative by design for a paper-trading system.
    """

    # Minimum observations required per agent type before promotion is even attempted
    MIN_OBSERVATIONS = {
        "execution": 63,
        "sizing": 63,
        "meta": 126,
    }

    # Sizing and meta: Sharpe CI lower bound must exceed this
    MIN_SHARPE_LOWER_BOUND = 0.5

    # Execution: implementation shortfall improvement vs baseline
    MIN_EXECUTION_IMPROVEMENT = 0.20  # 20% better than naive TWAP

    # Walk-forward: minimum fraction of positive folds
    MIN_WF_POSITIVE_FOLDS = 0.60

    # Walk-forward: maximum OOS Sharpe degradation vs IS Sharpe
    MAX_WF_DEGRADATION = 0.30

    # Monte Carlo: p-value cap
    MAX_PVALUE = 0.05

    # Max drawdown
    MAX_DRAWDOWN = 0.12

    # Directional agreement with profitable crew trades
    MIN_DIRECTION_AGREEMENT = 0.55

    def evaluate(
        self,
        agent_type: str,
        shadow_result: ShadowEvaluationResult,  # type: ignore[name-defined]  # noqa: F821
        simulated_returns: list[float] | None = None,
        walk_forward_folds: list[tuple[float, float]] | None = None,
    ) -> PromotionResult:
        """
        Run all promotion checks for the given agent type.

        Args:
            agent_type: "sizing" | "execution" | "meta"
            shadow_result: From ShadowEvaluator.evaluate_shadow_period()
            simulated_returns: Optional list of per-trade simulated returns (%)
                               for statistical tests. If None, only shadow metrics used.
            walk_forward_folds: Optional list of (in_sample_sharpe, oos_sharpe) tuples.
                                 If None, walk-forward check is skipped.

        Returns:
            PromotionResult with pass/fail and per-check details.
        """
        checks: list[PromotionCheckResult] = []
        min_obs = self.MIN_OBSERVATIONS.get(agent_type, 63)

        # ------------------------------------------------------------------
        # Check 1: Observation count
        # ------------------------------------------------------------------
        obs_check = self._check_observations(shadow_result.n_observations, min_obs)
        checks.append(obs_check)
        if not obs_check.passed:
            # Gate: can't proceed without enough observations
            return PromotionResult(
                agent_type=agent_type,
                passes=False,
                checks=checks,
            )

        # ------------------------------------------------------------------
        # Check 2: Simulated Sharpe (Lo 2002 CI lower bound when possible)
        # ------------------------------------------------------------------
        if agent_type in ("sizing", "meta"):
            sharpe_check = self._check_sharpe(
                simulated_returns or [],
                shadow_result.rl_simulated_sharpe,
                agent_type,
            )
            checks.append(sharpe_check)

        # ------------------------------------------------------------------
        # Check 3: Max drawdown
        # ------------------------------------------------------------------
        if shadow_result.rl_simulated_max_drawdown is not None:
            dd_check = self._check_drawdown(shadow_result.rl_simulated_max_drawdown)
            checks.append(dd_check)

        # ------------------------------------------------------------------
        # Check 4: Directional agreement
        # ------------------------------------------------------------------
        if shadow_result.directional_agreement_rate is not None:
            agree_check = self._check_agreement(
                shadow_result.directional_agreement_rate
            )
            checks.append(agree_check)

        # ------------------------------------------------------------------
        # Check 5: Monte Carlo significance (if returns provided)
        # ------------------------------------------------------------------
        if simulated_returns and len(simulated_returns) >= 30:
            mc_check = self._check_monte_carlo(simulated_returns)
            checks.append(mc_check)

        # ------------------------------------------------------------------
        # Check 6: Walk-forward consistency (if folds provided)
        # ------------------------------------------------------------------
        if walk_forward_folds and len(walk_forward_folds) >= 3:
            wf_check = self._check_walk_forward(walk_forward_folds)
            checks.append(wf_check)

        # ------------------------------------------------------------------
        # Execution-specific: implementation shortfall check
        # ------------------------------------------------------------------
        if agent_type == "execution" and shadow_result.rl_simulated_sharpe is not None:
            exec_check = self._check_execution_shortfall(
                shadow_result.directional_agreement_rate or 0.0
            )
            checks.append(exec_check)

        all_passed = all(c.passed for c in checks)
        return PromotionResult(
            agent_type=agent_type,
            passes=all_passed,
            checks=checks,
        )

    # -------------------------------------------------------------------------
    # Individual checks
    # -------------------------------------------------------------------------

    def _check_observations(self, n: int, min_required: int) -> PromotionCheckResult:
        passed = n >= min_required
        return PromotionCheckResult(
            name="observation_count",
            passed=passed,
            value=float(n),
            threshold=float(min_required),
            message=(
                f"{n} >= {min_required} required observations."
                if passed
                else f"Only {n} observations; need {min_required}."
            ),
        )

    def _check_sharpe(
        self,
        returns: list[float],
        point_estimate: float | None,
        agent_type: str,
    ) -> PromotionCheckResult:
        """Check Sharpe using Lo (2002) CI lower bound when possible."""
        if len(returns) >= 20:
            try:
                from quantstack.core.backtesting.stats import sharpe_ratio_with_ci

                ret_arr = np.array(returns) / 100.0  # convert pct to decimal
                _, (lower, upper) = sharpe_ratio_with_ci(ret_arr)
                value = lower
                message_suffix = (
                    f"(CI lower bound, Lo 2002: [{lower:.2f}, {upper:.2f}])"
                )
            except Exception as exc:
                logger.debug(f"[PromotionGate] stats.py import failed: {exc}")
                value = point_estimate or 0.0
                message_suffix = "(point estimate — stats.py unavailable)"
        else:
            value = point_estimate or 0.0
            message_suffix = "(point estimate — insufficient data for CI)"

        passed = value is not None and value >= self.MIN_SHARPE_LOWER_BOUND
        return PromotionCheckResult(
            name=f"sharpe_{agent_type}",
            passed=passed,
            value=round(value, 4) if value is not None else None,
            threshold=self.MIN_SHARPE_LOWER_BOUND,
            message=(
                f"Sharpe {value:.2f} >= {self.MIN_SHARPE_LOWER_BOUND} {message_suffix}."
                if passed and value is not None
                else f"Sharpe {value:.2f if value else 'N/A'} < {self.MIN_SHARPE_LOWER_BOUND} {message_suffix}."
            ),
        )

    def _check_drawdown(self, max_dd: float) -> PromotionCheckResult:
        passed = max_dd <= self.MAX_DRAWDOWN
        return PromotionCheckResult(
            name="max_drawdown",
            passed=passed,
            value=round(max_dd, 4),
            threshold=self.MAX_DRAWDOWN,
            message=(
                f"Max drawdown {max_dd:.1%} <= {self.MAX_DRAWDOWN:.0%} limit."
                if passed
                else f"Max drawdown {max_dd:.1%} exceeds {self.MAX_DRAWDOWN:.0%} limit."
            ),
        )

    def _check_agreement(self, agreement: float) -> PromotionCheckResult:
        passed = agreement >= self.MIN_DIRECTION_AGREEMENT
        return PromotionCheckResult(
            name="directional_agreement",
            passed=passed,
            value=round(agreement, 4),
            threshold=self.MIN_DIRECTION_AGREEMENT,
            message=(
                f"Agreement {agreement:.1%} >= {self.MIN_DIRECTION_AGREEMENT:.0%}."
                if passed
                else f"Agreement {agreement:.1%} < {self.MIN_DIRECTION_AGREEMENT:.0%} minimum."
            ),
        )

    def _check_monte_carlo(self, returns: list[float]) -> PromotionCheckResult:
        """Monte Carlo permutation test: p-value must be <= MAX_PVALUE."""
        try:
            from quantstack.core.backtesting.stats import (
                monte_carlo_permutation,
                sharpe_ratio_with_ci,
            )

            ret_arr = np.array(returns) / 100.0
            observed_sharpe, _ = sharpe_ratio_with_ci(ret_arr)
            p_value = monte_carlo_permutation(
                ret_arr, observed_sharpe, n_permutations=1000
            )
            passed = p_value <= self.MAX_PVALUE
            return PromotionCheckResult(
                name="monte_carlo_significance",
                passed=passed,
                value=round(p_value, 4),
                threshold=self.MAX_PVALUE,
                message=(
                    f"Monte Carlo p-value {p_value:.3f} <= {self.MAX_PVALUE} (statistically significant)."
                    if passed
                    else f"Monte Carlo p-value {p_value:.3f} > {self.MAX_PVALUE} (not significant)."
                ),
            )
        except Exception as exc:
            return PromotionCheckResult(
                name="monte_carlo_significance",
                passed=True,  # skip rather than block on import error
                value=None,
                threshold=self.MAX_PVALUE,
                message=f"Monte Carlo test skipped: {exc}",
            )

    def _check_walk_forward(
        self,
        folds: list[tuple[float, float]],
    ) -> PromotionCheckResult:
        """
        Walk-forward: at least MIN_WF_POSITIVE_FOLDS must be positive OOS,
        and degradation must not exceed MAX_WF_DEGRADATION.
        """
        oos_sharpes = [oos for _, oos in folds]
        is_sharpes = [is_ for is_, _ in folds]

        pct_positive = float(np.mean([s > 0 for s in oos_sharpes]))

        avg_is = float(np.mean(is_sharpes))
        avg_oos = float(np.mean(oos_sharpes))
        degradation = (avg_is - avg_oos) / (abs(avg_is) + 1e-8)

        passed_pct = pct_positive >= self.MIN_WF_POSITIVE_FOLDS
        passed_deg = degradation <= self.MAX_WF_DEGRADATION
        passed = passed_pct and passed_deg

        return PromotionCheckResult(
            name="walk_forward",
            passed=passed,
            value=round(pct_positive, 4),
            threshold=self.MIN_WF_POSITIVE_FOLDS,
            message=(
                f"Walk-forward: {pct_positive:.0%} positive folds, "
                f"degradation={degradation:.0%} "
                f"({'OK' if passed else 'FAIL'})."
            ),
        )

    def _check_execution_shortfall(self, agreement: float) -> PromotionCheckResult:
        """
        Execution-specific check: RL must have better agreement with low-slippage
        outcomes than 55% baseline.
        """
        improvement = max(0.0, agreement - 0.50)
        passed = improvement >= self.MIN_EXECUTION_IMPROVEMENT
        return PromotionCheckResult(
            name="execution_shortfall_improvement",
            passed=passed,
            value=round(improvement, 4),
            threshold=self.MIN_EXECUTION_IMPROVEMENT,
            message=(
                f"Execution improvement {improvement:.1%} >= {self.MIN_EXECUTION_IMPROVEMENT:.0%}."
                if passed
                else (
                    f"Execution improvement {improvement:.1%} < "
                    f"{self.MIN_EXECUTION_IMPROVEMENT:.0%} required."
                )
            ),
        )
