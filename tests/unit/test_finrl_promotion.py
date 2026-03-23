"""Tests for quantstack.finrl.promotion.PromotionGate."""

import pytest

from quantstack.finrl.promotion import PromotionGate, PromotionResult


class TestPromotionGate:
    def setup_method(self):
        self.gate = PromotionGate()

    def test_insufficient_observations_fails(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=10,
        )
        assert not result.passes
        assert result.failed_checks[0].name == "observation_count"

    def test_sufficient_observations_passes_check(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=0.8,
        )
        # Obs check passes
        obs_check = next(c for c in result.checks if c.name == "observation_count")
        assert obs_check.passed

    def test_low_sharpe_fails(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=0.1,  # below 0.5 threshold
        )
        sharpe_check = next(c for c in result.checks if c.name == "sharpe_ratio")
        assert not sharpe_check.passed

    def test_high_sharpe_passes(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=1.5,
        )
        sharpe_check = next(c for c in result.checks if c.name == "sharpe_ratio")
        assert sharpe_check.passed

    def test_high_drawdown_fails(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=1.0,
            max_drawdown=0.20,  # above 0.12 threshold
        )
        dd_check = next(c for c in result.checks if c.name == "max_drawdown")
        assert not dd_check.passed

    def test_low_drawdown_passes(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=1.0,
            max_drawdown=0.05,
        )
        dd_check = next(c for c in result.checks if c.name == "max_drawdown")
        assert dd_check.passed

    def test_all_checks_pass(self):
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=1.5,
            max_drawdown=0.05,
            directional_agreement=0.70,
        )
        assert result.passes

    def test_walk_forward_check(self):
        # OOS close to IS → low degradation
        folds = [(1.0, 0.9), (0.8, 0.7), (1.1, 1.0), (0.9, 0.8)]
        result = self.gate.evaluate(
            model_id="test",
            n_observations=100,
            simulated_sharpe=1.0,
            walk_forward_folds=folds,
        )
        wf_check = next((c for c in result.checks if c.name == "walk_forward"), None)
        assert wf_check is not None
        assert wf_check.passed  # all OOS > 0, degradation < 30%

    def test_to_dict(self):
        result = self.gate.evaluate(model_id="test", n_observations=100, simulated_sharpe=1.0)
        d = result.to_dict()
        assert "model_id" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_summary(self):
        result = self.gate.evaluate(model_id="test", n_observations=100, simulated_sharpe=1.0)
        s = result.summary()
        assert "test" in s
        assert "PASS" in s or "FAIL" in s
