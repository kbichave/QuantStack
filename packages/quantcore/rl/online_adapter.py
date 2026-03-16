"""
PostTradeRLAdapter — wires trade outcomes to OnlineRLTrainer.update_from_outcome().

This is the missing feedback loop between TradingDayFlow and the RL agents.
When a trade closes, the adapter reconstructs the (state, action, reward, next_state)
tuple from the pre-trade snapshot that each RL tool saved at call time, then pushes
it to the appropriate OnlineRLTrainer.

Design constraints:
- Non-fatal: every exception is caught and logged; flow execution never blocked.
- Rate-limited: at most max_updates_per_day online updates per calendar day.
- Kill-switch aware: all updates skipped when kill_switch.is_active().
- Reward separation: sizing reward = risk-adjusted PnL; execution reward = -slippage_bps.
- Buffer gate: OnlineRLTrainer.update_from_outcome() is skipped unless replay buffer
  has >= min_replay_buffer_size entries (prevents updates on a cold buffer).
- Degradation guard: if rolling 20-trade eval reward drops below 80% of best seen,
  online updates are paused and a warning is emitted.

Usage (from TradingDayFlow._run_post_trade_learning()):
    adapter = PostTradeRLAdapter(config, kill_switch)
    for trade in self.state.executed_trades:
        snapshot = rl_session_store.load(trade.get("order_id"))
        if snapshot:
            adapter.process_trade_outcome(trade, snapshot)
"""

from __future__ import annotations

import json
from collections import deque
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from quantcore.rl.base import Action, Reward, State
from quantcore.rl.config import RLProductionConfig


class PostTradeRLAdapter:
    """
    Bridges trade outcomes back to OnlineRLTrainer instances.

    Each agent type (sizing, execution) has its own trainer.  The adapter
    is responsible for:
      1. Reconstructing RL (state, action, reward, next_state) from trade + snapshot.
      2. Enforcing all safety bounds before calling trainer.update_from_outcome().
      3. Logging all updates to the shadow evaluator so the PromotionGate has data.
    """

    def __init__(
        self,
        config: RLProductionConfig,
        kill_switch: Any,  # quant_pod.execution.kill_switch.KillSwitch
    ) -> None:
        self.config = config
        self._kill_switch = kill_switch

        # Per-day rate limiting
        self._update_date: Optional[date] = None
        self._updates_today: int = 0

        # Rolling reward history for degradation guard (last 20 trades per agent)
        self._sizing_rewards: deque = deque(maxlen=20)
        self._execution_rewards: deque = deque(maxlen=20)
        self._best_sizing_reward: float = -float("inf")
        self._best_execution_reward: float = -float("inf")

        # Lazy-load trainers to avoid import-time torch dependency
        self._sizing_trainer = None
        self._execution_trainer = None
        self._shadow_evaluator = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_trade_outcome(
        self,
        trade: Dict[str, Any],
        pre_trade_snapshot: Dict[str, Any],
    ) -> None:
        """
        Process a completed trade and push RL updates.

        Args:
            trade: Executed trade dict from TradingDayFlow.state.executed_trades.
                   Must have keys: order_id, pnl (optional), slippage_bps (optional),
                   symbol, side, confidence.
            pre_trade_snapshot: Dict saved by RLTool._run() at call time.
                   Must have: tool_name, state_vector (list[float]), action_value,
                   signal_confidence, volatility_at_entry.
        """
        try:
            self._process(trade, pre_trade_snapshot)
        except Exception as exc:
            logger.warning(f"[RL Adapter] process_trade_outcome failed (non-fatal): {exc}")

    def set_shadow_evaluator(self, evaluator: Any) -> None:
        """Attach a ShadowEvaluator for recording decisions during shadow period."""
        self._shadow_evaluator = evaluator

    # -------------------------------------------------------------------------
    # Internal logic
    # -------------------------------------------------------------------------

    def _process(
        self,
        trade: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> None:
        # Safety gates — checked before any update
        if self._kill_switch_active():
            return
        if not self._within_rate_limit():
            return

        tool_name = snapshot.get("tool_name", "")
        pnl = trade.get("pnl")
        slippage_bps = trade.get("slippage_bps", 0.0) or 0.0
        order_id = trade.get("order_id", "")
        symbol = trade.get("symbol")

        # Route to appropriate agent based on tool_name
        if tool_name == "rl_position_size" and self.config.enable_sizing_rl:
            reward_value = self._compute_sizing_reward(pnl, snapshot)
            self._update_sizing_agent(snapshot, reward_value)
            self._record_shadow(tool_name, snapshot, trade, reward_value, symbol)
            self._updates_today += 1

        elif tool_name == "rl_execution_strategy" and self.config.enable_execution_rl:
            reward_value = self._compute_execution_reward(slippage_bps)
            self._update_execution_agent(snapshot, reward_value)
            self._record_shadow(tool_name, snapshot, trade, reward_value, symbol)
            self._updates_today += 1

    def _compute_sizing_reward(
        self,
        pnl: Optional[float],
        snapshot: Dict[str, Any],
    ) -> float:
        """
        Risk-adjusted PnL as sizing reward.

        Divides raw PnL by volatility at entry to produce a Sharpe-like signal.
        A negative PnL with a high-confidence, oversized position is penalized more.
        """
        if pnl is None:
            return 0.0

        vol = float(snapshot.get("volatility_at_entry", 0.02))
        vol = max(vol, 1e-4)  # floor to avoid div/0
        risk_adj_pnl = pnl / (vol * 1000)  # scale to ~[-5, 5] range

        # Additional penalty when RL recommended large size but trade lost
        action_value = float(snapshot.get("action_value", 0.5))
        if pnl < 0 and action_value > 0.5:
            risk_adj_pnl *= 1.5  # amplify penalty for overconfident bad call

        return float(np.clip(risk_adj_pnl, -10.0, 10.0))

    def _compute_execution_reward(self, slippage_bps: float) -> float:
        """
        Negative implementation shortfall as execution reward.

        Lower slippage = higher reward. Baseline TWAP ≈ 5 bps.
        """
        baseline_bps = 5.0
        improvement = baseline_bps - slippage_bps
        # Scale: every bps better than baseline = +0.2 reward
        return float(np.clip(improvement * 0.2, -5.0, 5.0))

    def _update_sizing_agent(
        self,
        snapshot: Dict[str, Any],
        reward_value: float,
    ) -> None:
        """Push (s, a, r, s') to sizing OnlineRLTrainer."""
        trainer = self._get_sizing_trainer()
        if trainer is None:
            return

        if not self._buffer_ready(trainer):
            logger.debug("[RL Adapter] Sizing buffer not ready — skipping update")
            return

        if not self._reward_not_degraded(reward_value, "sizing"):
            return

        state_vec = snapshot.get("state_vector", [])
        if not state_vec:
            return

        state = State(features=np.array(state_vec, dtype=np.float32))
        action = Action(value=float(snapshot.get("action_value", 0.5)))
        reward = Reward(value=reward_value)
        # next_state: zero vector — we don't have post-trade market state
        next_state = State(features=np.zeros(len(state_vec), dtype=np.float32))

        trainer.update_from_outcome(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
        )
        logger.debug(f"[RL Adapter] Sizing update: reward={reward_value:.3f}")

    def _update_execution_agent(
        self,
        snapshot: Dict[str, Any],
        reward_value: float,
    ) -> None:
        """Push (s, a, r, s') to execution OnlineRLTrainer."""
        trainer = self._get_execution_trainer()
        if trainer is None:
            return

        if not self._buffer_ready(trainer):
            logger.debug("[RL Adapter] Execution buffer not ready — skipping update")
            return

        if not self._reward_not_degraded(reward_value, "execution"):
            return

        state_vec = snapshot.get("state_vector", [])
        if not state_vec:
            return

        state = State(features=np.array(state_vec, dtype=np.float32))
        action = Action(value=int(snapshot.get("action_value", 1)))
        reward = Reward(value=reward_value)
        next_state = State(features=np.zeros(len(state_vec), dtype=np.float32))

        trainer.update_from_outcome(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
        )
        logger.debug(f"[RL Adapter] Execution update: reward={reward_value:.3f}")

    def _record_shadow(
        self,
        tool_name: str,
        snapshot: Dict[str, Any],
        trade: Dict[str, Any],
        reward_value: float,
        symbol: Optional[str],
    ) -> None:
        """Update ShadowEvaluator with trade outcome if attached."""
        if self._shadow_evaluator is None:
            return
        decision_id = snapshot.get("shadow_decision_id")
        if not decision_id:
            return
        try:
            self._shadow_evaluator.record_outcome(
                decision_id=decision_id,
                pnl=trade.get("pnl"),
                slippage_bps=trade.get("slippage_bps"),
            )
        except Exception as exc:
            logger.debug(f"[RL Adapter] shadow record_outcome failed (non-fatal): {exc}")

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------

    def _kill_switch_active(self) -> bool:
        try:
            if self._kill_switch and self._kill_switch.is_active():
                logger.info("[RL Adapter] Kill switch active — skipping RL update")
                return True
        except Exception:
            pass
        return False

    def _within_rate_limit(self) -> bool:
        today = date.today()
        if self._update_date != today:
            self._update_date = today
            self._updates_today = 0
        if self._updates_today >= self.config.max_updates_per_day:
            logger.debug(
                f"[RL Adapter] Daily update cap reached "
                f"({self.config.max_updates_per_day}/day) — skipping"
            )
            return False
        return True

    def _buffer_ready(self, trainer: Any) -> bool:
        """Check that trainer's replay buffer has enough samples."""
        try:
            buf_size = len(trainer.buffer)
            return buf_size >= self.config.min_replay_buffer_size
        except Exception:
            return False

    def _reward_not_degraded(self, reward_value: float, agent: str) -> bool:
        """
        Rolling degradation guard: if recent rewards < 80% of best seen, pause.
        Returns False when degraded (caller should skip the update).
        """
        if agent == "sizing":
            history = self._sizing_rewards
            best_attr = "_best_sizing_reward"
        else:
            history = self._execution_rewards
            best_attr = "_best_execution_reward"

        history.append(reward_value)
        if len(history) < 10:
            # Not enough history to judge degradation
            return True

        rolling_avg = float(np.mean(history))
        best = getattr(self, best_attr)
        if rolling_avg > best:
            setattr(self, best_attr, rolling_avg)
            return True

        threshold = best * self.config.degradation_threshold
        if rolling_avg < threshold and best > 0:
            logger.warning(
                f"[RL Adapter] {agent} agent degraded: "
                f"rolling_avg={rolling_avg:.3f} < {self.config.degradation_threshold:.0%} "
                f"of best={best:.3f} — pausing online updates"
            )
            return False

        return True

    # -------------------------------------------------------------------------
    # Lazy trainer loading
    # -------------------------------------------------------------------------

    def _get_sizing_trainer(self) -> Optional[Any]:
        if self._sizing_trainer is not None:
            return self._sizing_trainer
        try:
            from quantcore.rl.sizing.agent import SizingRLAgent
            from quantcore.rl.training import OnlineRLTrainer, TrainingConfig

            ckpt = self.config.sizing_checkpoint_path
            agent = SizingRLAgent(
                state_dim=self.config.sizing_state_dim,
                action_dim=1,
            )
            if ckpt.exists():
                agent.load_checkpoint(str(ckpt))
            cfg = TrainingConfig(batch_size=32)
            self._sizing_trainer = OnlineRLTrainer(agent=agent, config=cfg)
            logger.info("[RL Adapter] Sizing OnlineRLTrainer initialized")
        except Exception as exc:
            logger.debug(f"[RL Adapter] Sizing trainer init failed (non-fatal): {exc}")
        return self._sizing_trainer

    def _get_execution_trainer(self) -> Optional[Any]:
        if self._execution_trainer is not None:
            return self._execution_trainer
        try:
            from quantcore.rl.execution.agent import ExecutionRLAgent
            from quantcore.rl.training import OnlineRLTrainer, TrainingConfig

            ckpt = self.config.execution_checkpoint_path
            agent = ExecutionRLAgent(
                state_dim=self.config.execution_state_dim,
                n_actions=5,
            )
            if ckpt.exists():
                agent.load_checkpoint(str(ckpt))
            cfg = TrainingConfig(batch_size=32)
            self._execution_trainer = OnlineRLTrainer(agent=agent, config=cfg)
            logger.info("[RL Adapter] Execution OnlineRLTrainer initialized")
        except Exception as exc:
            logger.debug(f"[RL Adapter] Execution trainer init failed (non-fatal): {exc}")
        return self._execution_trainer
