"""Tests for custom Gymnasium environments in quantstack.finrl.environments."""

import numpy as np
import pytest

from quantstack.finrl.environments import (
    AlphaSelectionEnv,
    ExecutionEnv,
    SizingEnv,
)


class TestExecutionEnv:
    """ExecutionEnv: order execution optimization."""

    def test_reset_returns_obs_and_info(self):
        env = ExecutionEnv(seed=42)
        obs, info = env.reset()
        assert obs.shape == (8,)
        assert obs.dtype == np.float32

    def test_step_returns_5_tuple(self):
        env = ExecutionEnv(seed=42)
        env.reset()
        obs, reward, done, truncated, info = env.step(1)
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)

    def test_market_order_completes_execution(self):
        env = ExecutionEnv(total_quantity=100, time_horizon=10, seed=42)
        env.reset()
        # Keep executing market orders until done
        for _ in range(20):
            obs, reward, done, truncated, info = env.step(4)  # MARKET
            if done:
                break
        assert done

    def test_wait_action_no_execution(self):
        env = ExecutionEnv(seed=42)
        env.reset()
        obs, reward, done, truncated, info = env.step(0)  # WAIT
        assert info["executed_qty"] == 0.0

    def test_obs_values_in_bounds(self):
        env = ExecutionEnv(seed=42)
        obs, _ = env.reset()
        # qty_frac and time_frac should be in [0, 1]
        assert 0 <= obs[0] <= 1
        assert 0 <= obs[1] <= 1

    def test_action_space_valid(self):
        env = ExecutionEnv()
        assert env.action_space.n == 5

    def test_observation_space_shape(self):
        env = ExecutionEnv()
        assert env.observation_space.shape == (8,)


class TestSizingEnv:
    """SizingEnv: dynamic position sizing."""

    def test_reset_returns_obs(self):
        env = SizingEnv(seed=42)
        obs, info = env.reset()
        assert obs.shape == (10,)
        assert obs.dtype == np.float32

    def test_step_returns_5_tuple(self):
        env = SizingEnv(seed=42)
        env.reset()
        action = np.array([0.5], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == (10,)
        assert "equity" in info

    def test_drawdown_terminates(self):
        env = SizingEnv(
            initial_equity=100, max_drawdown_limit=0.01, max_steps=1000, seed=42
        )
        env.reset()
        done = False
        for _ in range(200):
            action = np.array([1.0], dtype=np.float32)
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert done or truncated

    def test_action_space_continuous(self):
        env = SizingEnv()
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0
        assert env.action_space.high[0] == 1


class TestAlphaSelectionEnv:
    """AlphaSelectionEnv: alpha signal weighting."""

    def test_reset_returns_obs(self):
        env = AlphaSelectionEnv(seed=42)
        obs, info = env.reset()
        expected_dim = 4 + 4 * len(env.alpha_names) + 4
        assert obs.shape == (expected_dim,)

    def test_step_returns_5_tuple(self):
        env = AlphaSelectionEnv(seed=42)
        env.reset()
        obs, reward, done, truncated, info = env.step(0)
        assert "selected_alpha" in info
        assert "selected_return" in info

    def test_no_trade_action(self):
        env = AlphaSelectionEnv(seed=42)
        env.reset()
        no_trade_action = env.n_alphas  # last action = no trade
        _, _, _, _, info = env.step(no_trade_action)
        assert info["selected_alpha"] == "NO_TRADE"

    def test_truncation_at_max_steps(self):
        env = AlphaSelectionEnv(max_steps=5, seed=42)
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                break
        assert truncated

    def test_custom_alpha_names(self):
        names = ["alpha_a", "alpha_b", "alpha_c"]
        env = AlphaSelectionEnv(alpha_names=names, seed=42)
        assert env.n_alphas == 3
        assert env.action_space.n == 4  # 3 alphas + no-trade

    def test_with_real_alpha_returns(self):
        returns = {
            "momentum": [0.01, -0.005, 0.003] * 10,
            "mean_reversion": [-0.002, 0.008, 0.001] * 10,
        }
        env = AlphaSelectionEnv(
            alpha_names=["momentum", "mean_reversion"],
            alpha_returns=returns,
            seed=42,
        )
        obs, _ = env.reset()
        assert obs.shape[0] == 4 + 4 * 2 + 4
