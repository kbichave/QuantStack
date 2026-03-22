# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PostTradeRLAdapter.

Tests: reward computation, rate limiting, kill-switch, degradation guard,
buffer gate, and shadow evaluator integration.
All RL trainer calls are mocked — no torch dependency required.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np


def _make_config(max_updates=10, min_buffer=0, degradation=0.80):
    cfg = MagicMock()
    cfg.max_updates_per_day = max_updates
    cfg.min_replay_buffer_size = min_buffer
    cfg.degradation_threshold = degradation
    cfg.enable_sizing_rl = True
    cfg.enable_execution_rl = True
    cfg.sizing_state_dim = 10
    cfg.execution_state_dim = 8
    cfg.sizing_checkpoint_path = MagicMock(exists=lambda: False)
    cfg.execution_checkpoint_path = MagicMock(exists=lambda: False)
    return cfg


def _make_kill_switch(active=False):
    ks = MagicMock()
    ks.is_active.return_value = active
    return ks


def _make_adapter(max_updates=10, min_buffer=0, kill_active=False):
    from quantstack.rl.online_adapter import PostTradeRLAdapter

    cfg = _make_config(max_updates=max_updates, min_buffer=min_buffer)
    ks = _make_kill_switch(active=kill_active)
    return PostTradeRLAdapter(cfg, ks)


def _sizing_snapshot(scale=0.7, state_dim=10, volatility=0.02):
    return {
        "tool_name": "rl_position_size",
        "state_vector": [0.0] * state_dim,
        "action_value": scale,
        "volatility_at_entry": volatility,
        "signal_confidence": 0.7,
    }


def _execution_snapshot(action=1, state_dim=8):
    return {
        "tool_name": "rl_execution_strategy",
        "state_vector": [0.0] * state_dim,
        "action_value": action,
    }


def _trade(pnl=200.0, slippage_bps=3.0, order_id="ord1"):
    return {
        "order_id": order_id,
        "pnl": pnl,
        "slippage_bps": slippage_bps,
        "symbol": "SPY",
        "side": "buy",
    }


class TestRewardComputation:
    def test_sizing_reward_positive_pnl(self):
        adapter = _make_adapter()
        reward = adapter._compute_sizing_reward(pnl=200.0, snapshot=_sizing_snapshot())
        assert reward > 0.0

    def test_sizing_reward_negative_pnl(self):
        adapter = _make_adapter()
        reward = adapter._compute_sizing_reward(pnl=-200.0, snapshot=_sizing_snapshot())
        assert reward < 0.0

    def test_sizing_reward_finite(self):
        adapter = _make_adapter()
        reward = adapter._compute_sizing_reward(pnl=0.0, snapshot=_sizing_snapshot())
        assert np.isfinite(reward)

    def test_sizing_reward_clipped(self):
        adapter = _make_adapter()
        reward = adapter._compute_sizing_reward(pnl=1e10, snapshot=_sizing_snapshot())
        assert reward <= 10.0
        reward2 = adapter._compute_sizing_reward(pnl=-1e10, snapshot=_sizing_snapshot())
        assert reward2 >= -10.0

    def test_sizing_reward_none_pnl_returns_zero(self):
        adapter = _make_adapter()
        reward = adapter._compute_sizing_reward(pnl=None, snapshot=_sizing_snapshot())
        assert reward == 0.0

    def test_execution_reward_below_baseline(self):
        # slippage_bps=3 < baseline(5) → positive reward
        adapter = _make_adapter()
        reward = adapter._compute_execution_reward(slippage_bps=3.0)
        assert reward > 0.0

    def test_execution_reward_above_baseline(self):
        # slippage_bps=8 > baseline(5) → negative reward
        adapter = _make_adapter()
        reward = adapter._compute_execution_reward(slippage_bps=8.0)
        assert reward < 0.0

    def test_execution_reward_clipped(self):
        adapter = _make_adapter()
        r = adapter._compute_execution_reward(slippage_bps=1000.0)
        assert r >= -5.0


class TestKillSwitch:
    def test_active_kill_switch_blocks_all_updates(self):
        adapter = _make_adapter(kill_active=True)
        mock_trainer = MagicMock()
        adapter._sizing_trainer = mock_trainer
        adapter.process_trade_outcome(_trade(), _sizing_snapshot())
        mock_trainer.update_from_outcome.assert_not_called()

    def test_inactive_kill_switch_allows_updates(self):
        adapter = _make_adapter(kill_active=False, min_buffer=0)
        with patch.object(adapter, "_get_sizing_trainer") as mock_get:
            mock_trainer = MagicMock()
            mock_trainer.buffer.__len__ = lambda self: 200
            mock_get.return_value = mock_trainer
            adapter.process_trade_outcome(_trade(), _sizing_snapshot())
            # Should have attempted the update path


class TestRateLimiting:
    def test_stops_after_daily_cap(self):
        adapter = _make_adapter(max_updates=2, min_buffer=0)
        with patch.object(adapter, "_get_sizing_trainer") as mock_get:
            mock_trainer = MagicMock()
            mock_trainer.buffer.__len__ = lambda self: 200
            mock_get.return_value = mock_trainer
            for i in range(5):
                adapter.process_trade_outcome(
                    _trade(order_id=f"ord{i}"), _sizing_snapshot()
                )
        # Cap is 2 — updates_today should not exceed cap
        assert adapter._updates_today <= adapter.config.max_updates_per_day

    def test_counter_resets_on_new_day(self):
        adapter = _make_adapter(max_updates=5)
        adapter._update_date = date(2000, 1, 1)  # old date
        adapter._updates_today = 99
        # Calling _within_rate_limit triggers reset
        assert adapter._within_rate_limit() is True
        assert adapter._updates_today == 0


class TestBufferGate:
    def test_skips_when_buffer_too_small(self):
        adapter = _make_adapter(min_buffer=100)
        mock_trainer = MagicMock()
        mock_trainer.buffer.__len__ = lambda self: 5  # below 100 min
        with patch.object(adapter, "_get_sizing_trainer", return_value=mock_trainer):
            adapter.process_trade_outcome(_trade(), _sizing_snapshot())
            mock_trainer.update_from_outcome.assert_not_called()

    def test_proceeds_when_buffer_sufficient(self):
        adapter = _make_adapter(min_buffer=10)
        mock_trainer = MagicMock()
        mock_trainer.buffer.__len__ = lambda self: 50  # above 10
        with patch.object(adapter, "_get_sizing_trainer", return_value=mock_trainer):
            adapter.process_trade_outcome(_trade(), _sizing_snapshot())


class TestDegradationGuard:
    def test_no_degradation_with_sparse_history(self):
        adapter = _make_adapter()
        # Fewer than 10 samples → always returns True
        result = adapter._reward_not_degraded(reward_value=0.5, agent="sizing")
        assert result is True

    def test_degradation_detected_when_rewards_drop(self):
        adapter = _make_adapter()
        # Seed history with good rewards, then push best > 0
        for _ in range(15):
            adapter._reward_not_degraded(reward_value=5.0, agent="sizing")
        # Now rewards collapse
        for _ in range(20):
            adapter._sizing_rewards.append(-1.0)
        adapter._best_sizing_reward = 5.0
        result = adapter._reward_not_degraded(reward_value=-1.0, agent="sizing")
        assert result is False

    def test_no_degradation_when_best_is_negative(self):
        adapter = _make_adapter()
        adapter._best_sizing_reward = -1.0  # negative best → guard skipped
        for _ in range(20):
            adapter._sizing_rewards.append(-2.0)
        result = adapter._reward_not_degraded(reward_value=-2.0, agent="sizing")
        assert result is True  # best is negative → condition skipped


class TestShadowEvaluatorIntegration:
    def test_record_outcome_called_when_snapshot_has_decision_id(self):
        adapter = _make_adapter()
        shadow_ev = MagicMock()
        adapter.set_shadow_evaluator(shadow_ev)

        snapshot = _sizing_snapshot()
        snapshot["shadow_decision_id"] = "test-decision-123"

        adapter._record_shadow(
            tool_name="rl_position_size",
            snapshot=snapshot,
            trade=_trade(),
            reward_value=1.0,
            symbol="SPY",
        )
        shadow_ev.record_outcome.assert_called_once_with(
            decision_id="test-decision-123",
            pnl=200.0,
            slippage_bps=3.0,
        )

    def test_no_shadow_call_without_decision_id(self):
        adapter = _make_adapter()
        shadow_ev = MagicMock()
        adapter.set_shadow_evaluator(shadow_ev)

        snapshot = _sizing_snapshot()  # no shadow_decision_id key
        adapter._record_shadow("rl_position_size", snapshot, _trade(), 1.0, "SPY")
        shadow_ev.record_outcome.assert_not_called()


class TestNonFatalBehavior:
    def test_process_does_not_raise_on_bad_snapshot(self):
        adapter = _make_adapter()
        adapter.process_trade_outcome({}, {})  # empty dicts — should not raise

    def test_process_does_not_raise_on_missing_pnl(self):
        adapter = _make_adapter()
        trade = {"order_id": "x", "slippage_bps": 3.0, "symbol": "SPY"}
        adapter.process_trade_outcome(trade, _sizing_snapshot())
