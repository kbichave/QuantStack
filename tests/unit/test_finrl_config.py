"""Tests for quantstack.finrl.config."""

import os

import pytest

from quantstack.finrl.config import FinRLConfig, get_finrl_config, reset_finrl_config


class TestFinRLConfig:
    def setup_method(self):
        reset_finrl_config()

    def test_default_config(self):
        cfg = FinRLConfig()
        assert cfg.config_version == "2.0.0"
        assert cfg.shadow_mode_enabled is True
        assert cfg.default_algorithm == "ppo"
        assert cfg.default_total_timesteps == 100_000

    def test_path_expansion(self):
        cfg = FinRLConfig()
        assert "~" not in cfg.checkpoint_base_path

    def test_singleton(self):
        cfg1 = get_finrl_config()
        cfg2 = get_finrl_config()
        assert cfg1 is cfg2

    def test_reset(self):
        cfg1 = get_finrl_config()
        reset_finrl_config()
        cfg2 = get_finrl_config()
        assert cfg1 is not cfg2

    def test_env_override(self, monkeypatch):
        reset_finrl_config()
        monkeypatch.setenv("FINRL_DEFAULT_ALGORITHM", "sac")
        cfg = get_finrl_config()
        assert cfg.default_algorithm == "sac"
        reset_finrl_config()

    def test_promotion_thresholds(self):
        cfg = FinRLConfig()
        assert cfg.min_shadow_observations == 63
        assert cfg.min_promo_sharpe == 0.5
        assert cfg.max_promo_drawdown == 0.12

    def test_ensemble_defaults(self):
        cfg = FinRLConfig()
        assert "ppo" in cfg.ensemble_algorithms
        assert "a2c" in cfg.ensemble_algorithms
