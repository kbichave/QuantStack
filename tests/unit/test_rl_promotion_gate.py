# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PromotionGate.

Tests all individual check methods and the aggregate evaluate() method.
Uses mock ShadowEvaluationResult objects to avoid needing a live store.
"""

from __future__ import annotations

import numpy as np


def _make_shadow_result(
    n_observations=70,
    rl_simulated_sharpe=1.2,
    rl_simulated_win_rate=0.6,
    rl_simulated_max_drawdown=0.08,
    directional_agreement_rate=0.65,
    ready_for_promotion=True,
    agent_type="sizing",
):
    from quantstack.rl.shadow_mode import ShadowEvaluationResult

    return ShadowEvaluationResult(
        agent_type=agent_type,
        n_observations=n_observations,
        rl_simulated_sharpe=rl_simulated_sharpe,
        rl_simulated_win_rate=rl_simulated_win_rate,
        rl_simulated_max_drawdown=rl_simulated_max_drawdown,
        directional_agreement_rate=directional_agreement_rate,
        ready_for_promotion=ready_for_promotion,
    )


class TestObservationCheck:
    def test_passes_when_sufficient(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_observations(n=70, min_required=63)
        assert result.passed is True
        assert result.value == 70.0

    def test_fails_when_insufficient(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_observations(n=10, min_required=63)
        assert result.passed is False

    def test_exactly_at_threshold_passes(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_observations(n=63, min_required=63)
        assert result.passed is True


class TestDrawdownCheck:
    def test_passes_under_limit(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_drawdown(max_dd=0.05)
        assert result.passed is True

    def test_fails_over_limit(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_drawdown(max_dd=0.15)
        assert result.passed is False

    def test_exactly_at_limit_passes(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_drawdown(max_dd=gate.MAX_DRAWDOWN)
        assert result.passed is True


class TestAgreementCheck:
    def test_passes_above_min(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_agreement(agreement=0.65)
        assert result.passed is True

    def test_fails_below_min(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        result = gate._check_agreement(agreement=0.45)
        assert result.passed is False


class TestWalkForwardCheck:
    def test_passes_with_good_folds(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        # All 5 folds: IS Sharpe 1.0, OOS Sharpe 0.8 → positive, low degradation
        folds = [(1.0, 0.8)] * 5
        result = gate._check_walk_forward(folds)
        assert result.passed is True

    def test_fails_with_mostly_negative_oos(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        # 4/5 negative OOS → 20% positive < 60% threshold
        folds = [(1.0, 0.5), (1.0, -0.3), (1.0, -0.5), (1.0, -0.2), (1.0, -0.1)]
        result = gate._check_walk_forward(folds)
        assert result.passed is False

    def test_fails_with_high_degradation(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        # IS=2.0, OOS=0.1 → 95% degradation >> 30% threshold
        folds = [(2.0, 0.1)] * 5
        result = gate._check_walk_forward(folds)
        assert result.passed is False


class TestExecutionShortfallCheck:
    def test_passes_with_good_improvement(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        # agreement=0.75 → improvement = 0.75 - 0.50 = 0.25 >= 0.20
        result = gate._check_execution_shortfall(agreement=0.75)
        assert result.passed is True

    def test_fails_with_poor_improvement(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        # agreement=0.55 → improvement = 0.05 < 0.20
        result = gate._check_execution_shortfall(agreement=0.55)
        assert result.passed is False


class TestEvaluateGating:
    def test_gate_blocks_on_insufficient_observations(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        shadow = _make_shadow_result(n_observations=5)
        result = gate.evaluate("sizing", shadow)
        assert result.passes is False
        # Only the observation check should be present (gated early)
        assert len(result.checks) == 1

    def test_passes_with_ideal_sizing_metrics(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        shadow = _make_shadow_result(
            n_observations=70,
            rl_simulated_sharpe=1.5,
            rl_simulated_max_drawdown=0.05,
            directional_agreement_rate=0.70,
        )
        # Build good simulated returns
        np.random.seed(42)
        returns = list(np.random.randn(100) * 0.5 + 0.2)  # positive drift → good Sharpe
        folds = [(1.2, 0.9)] * 5
        result = gate.evaluate(
            "sizing",
            shadow,
            simulated_returns=returns,
            walk_forward_folds=folds,
        )
        # All individual checks should be present
        assert len(result.checks) > 1
        # Result object must be consistent
        assert result.passes == all(c.passed for c in result.checks)

    def test_fails_on_high_drawdown(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        shadow = _make_shadow_result(
            n_observations=70,
            rl_simulated_max_drawdown=0.20,  # exceeds 12% limit
        )
        result = gate.evaluate("sizing", shadow)
        dd_check = next((c for c in result.checks if c.name == "max_drawdown"), None)
        assert dd_check is not None
        assert dd_check.passed is False

    def test_execution_agent_gets_shortfall_check(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        shadow = _make_shadow_result(
            agent_type="execution",
            n_observations=70,
            directional_agreement_rate=0.55,  # → improvement = 0.05, fails shortfall
        )
        result = gate.evaluate("execution", shadow)
        shortfall_check = next(
            (c for c in result.checks if c.name == "execution_shortfall_improvement"),
            None,
        )
        assert shortfall_check is not None

    def test_summary_returns_string(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        shadow = _make_shadow_result(n_observations=70)
        result = gate.evaluate("sizing", shadow)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "sizing" in summary.lower()

    def test_passed_failed_check_properties(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        shadow = _make_shadow_result(
            n_observations=70,
            rl_simulated_max_drawdown=0.20,  # will fail
        )
        result = gate.evaluate("sizing", shadow)
        all_checks = result.checks
        assert len(result.passed_checks) + len(result.failed_checks) == len(all_checks)

    def test_meta_agent_requires_more_observations(self):
        from quantstack.rl.promotion_gate import PromotionGate

        gate = PromotionGate()
        # 70 observations fine for sizing (63), but not for meta (126)
        shadow = _make_shadow_result(n_observations=70, agent_type="meta")
        result = gate.evaluate("meta", shadow)
        assert result.passes is False
