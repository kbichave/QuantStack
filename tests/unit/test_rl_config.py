# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for RLProductionConfig.

Covers: defaults, env-var overrides, validator, singleton, reset.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestRLProductionConfigDefaults:
    def test_import(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert cfg is not None

    def test_feature_flags_default_true(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert cfg.enable_execution_rl is True
        assert cfg.enable_sizing_rl is True
        assert cfg.enable_meta_rl is True

    def test_shadow_mode_default_true(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert cfg.execution_shadow is True
        assert cfg.sizing_shadow is True
        assert cfg.meta_shadow is True

    def test_safety_bounds_sane(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert cfg.max_updates_per_day >= 1
        assert cfg.min_replay_buffer_size >= 1
        assert 0.0 < cfg.max_param_change_norm <= 1.0
        assert cfg.degradation_threshold > 0.0

    def test_feature_dims(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert cfg.execution_state_dim == 8
        assert cfg.sizing_state_dim == 10

    def test_promotion_thresholds(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert cfg.min_shadow_observations_execution > 0
        assert cfg.min_shadow_observations_sizing > 0
        assert cfg.min_shadow_observations_meta > cfg.min_shadow_observations_sizing
        assert 0.0 < cfg.max_monte_carlo_pvalue < 1.0
        assert 0.0 < cfg.max_promo_drawdown < 1.0

    def test_checkpoint_paths_are_paths(self):
        from quantcore.rl.config import RLProductionConfig
        cfg = RLProductionConfig()
        assert isinstance(cfg.sizing_checkpoint_path, Path)
        assert isinstance(cfg.execution_checkpoint_path, Path)
        assert isinstance(cfg.meta_checkpoint_path, Path)


class TestRLProductionConfigEnvOverrides:
    def test_env_override_updates_per_day(self):
        from quantcore.rl.config import RLProductionConfig
        with patch.dict(os.environ, {"QUANTRL_MAX_UPDATES_PER_DAY": "3"}):
            cfg = RLProductionConfig()
            assert cfg.max_updates_per_day == 3

    def test_env_override_disable_sizing(self):
        from quantcore.rl.config import RLProductionConfig
        with patch.dict(os.environ, {"QUANTRL_ENABLE_SIZING_RL": "false"}):
            cfg = RLProductionConfig()
            assert cfg.enable_sizing_rl is False

    def test_env_override_shadow_off(self):
        from quantcore.rl.config import RLProductionConfig
        with patch.dict(os.environ, {"QUANTRL_SIZING_SHADOW": "false"}):
            cfg = RLProductionConfig()
            assert cfg.sizing_shadow is False


class TestGetRLConfig:
    def setup_method(self):
        from quantcore.rl.config import reset_rl_config
        reset_rl_config()

    def teardown_method(self):
        from quantcore.rl.config import reset_rl_config
        reset_rl_config()

    def test_returns_instance(self):
        from quantcore.rl.config import get_rl_config
        cfg = get_rl_config()
        assert cfg is not None

    def test_singleton_same_object(self):
        from quantcore.rl.config import get_rl_config
        cfg1 = get_rl_config()
        cfg2 = get_rl_config()
        assert cfg1 is cfg2

    def test_reset_creates_new_instance(self):
        from quantcore.rl.config import get_rl_config, reset_rl_config
        cfg1 = get_rl_config()
        reset_rl_config()
        cfg2 = get_rl_config()
        assert cfg1 is not cfg2
