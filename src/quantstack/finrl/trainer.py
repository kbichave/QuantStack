"""
Trainer — thin wrapper around FinRL's DRLAgent and DRLEnsembleAgent.

Handles: algorithm selection, hyperparameter passing, checkpoint management,
and integration with the model registry.

Usage:
    from quantstack.finrl.trainer import FinRLTrainer

    trainer = FinRLTrainer()
    result = trainer.train(env, algorithm="ppo", total_timesteps=100_000)
    trainer.evaluate(model_path, test_env)
"""

from __future__ import annotations

import json
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from quantstack.finrl.config import get_finrl_config


@dataclass
class TrainResult:
    """Result of a training run."""

    model_id: str
    algorithm: str
    checkpoint_path: str
    total_timesteps: int
    training_time_s: float
    final_reward: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of model evaluation."""

    model_id: str
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    total_trades: int
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)


class FinRLTrainer:
    """
    Trains RL models using stable-baselines3 algorithms.

    For stock trading / portfolio envs, delegates to FinRL's DRLAgent.
    For custom envs (ExecutionEnv, SizingEnv, AlphaSelectionEnv),
    uses stable-baselines3 directly.
    """

    ALGORITHM_MAP = {
        "ppo": "PPO",
        "a2c": "A2C",
        "sac": "SAC",
        "td3": "TD3",
        "ddpg": "DDPG",
        "dqn": "DQN",
    }

    def __init__(self, config: Any | None = None):
        self.config = config or get_finrl_config()

    def train(
        self,
        env: gym.Env,
        algorithm: str = "ppo",
        total_timesteps: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        net_arch: list[int] | None = None,
        model_name: str | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> TrainResult:
        """
        Train a model on the given Gymnasium environment.

        Returns TrainResult with model_id, checkpoint path, and metrics.
        """
        algo_name = algorithm.lower()
        if algo_name not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {list(self.ALGORITHM_MAP.keys())}"
            )

        ts = total_timesteps or self.config.default_total_timesteps
        lr = learning_rate or self.config.default_learning_rate
        bs = batch_size or self.config.default_batch_size
        arch = net_arch or self.config.default_net_arch

        algo_cls = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG, "dqn": DQN}[
            algo_name
        ]

        # Build policy kwargs
        policy_kwargs = {"net_arch": arch}
        extra = hyperparams or {}

        # Determine policy type
        if isinstance(env.action_space, gym.spaces.Discrete):
            policy = "MlpPolicy"
        else:
            policy = "MlpPolicy"

        # SAC/TD3/DDPG require continuous action space
        if algo_name in ("sac", "td3", "ddpg") and isinstance(
            env.action_space, gym.spaces.Discrete
        ):
            raise ValueError(
                f"{algo_name.upper()} requires continuous action space, "
                f"but env has Discrete({env.action_space.n})."
            )

        # DQN requires discrete action space
        if algo_name == "dqn" and not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN requires discrete action space.")

        model_kwargs = {
            "policy": policy,
            "env": env,
            "learning_rate": lr,
            "verbose": 0,
            "policy_kwargs": policy_kwargs,
        }

        # Add batch_size where supported
        if algo_name in ("ppo", "dqn"):
            model_kwargs["batch_size"] = bs
        elif algo_name in ("sac", "td3", "ddpg"):
            model_kwargs["batch_size"] = bs

        model_kwargs.update(extra)

        logger.info(
            f"[FinRLTrainer] Training {algo_name.upper()} for {ts} timesteps "
            f"(lr={lr}, batch={bs}, arch={arch})"
        )

        start = _time.time()
        model = algo_cls(**model_kwargs)
        model.learn(total_timesteps=ts)
        elapsed = _time.time() - start

        # Save checkpoint
        model_id = model_name or f"finrl_{algo_name}_{uuid.uuid4().hex[:8]}"
        ckpt_dir = Path(self.config.checkpoint_base_path) / model_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / "model")
        model.save(ckpt_path)

        # Save metadata
        meta = {
            "model_id": model_id,
            "algorithm": algo_name,
            "total_timesteps": ts,
            "learning_rate": lr,
            "batch_size": bs,
            "net_arch": arch,
            "hyperparams": extra,
            "training_time_s": round(elapsed, 2),
            "created_at": datetime.utcnow().isoformat(),
        }
        (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info(
            f"[FinRLTrainer] Training complete: {model_id} in {elapsed:.1f}s → {ckpt_path}"
        )

        return TrainResult(
            model_id=model_id,
            algorithm=algo_name,
            checkpoint_path=ckpt_path,
            total_timesteps=ts,
            training_time_s=round(elapsed, 2),
            final_reward=0.0,  # SB3 doesn't expose this directly
            metrics=meta,
        )

    def evaluate(
        self,
        model_path: str,
        env: gym.Env,
        algorithm: str = "ppo",
        n_episodes: int = 10,
    ) -> EvalResult:
        """
        Evaluate a trained model on a test environment.

        Runs n_episodes and computes aggregate metrics.
        """
        algo_cls = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG, "dqn": DQN}[
            algorithm.lower()
        ]

        model = algo_cls.load(model_path, env=env)

        all_rewards = []
        all_equity = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            ep_equity = [getattr(env, "initial_equity", 100_000)]

            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                if "equity" in info:
                    ep_equity.append(info["equity"])

            all_rewards.append(ep_reward)
            if len(ep_equity) > 1:
                all_equity.extend(ep_equity)

        # Compute metrics
        rewards_arr = np.array(all_rewards)
        sharpe = float(
            np.mean(rewards_arr) / (np.std(rewards_arr) + 1e-8) * np.sqrt(252)
        )

        if all_equity:
            eq = np.array(all_equity)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / (peak + 1e-8)
            max_dd = float(dd.max())
            total_ret = float((eq[-1] - eq[0]) / eq[0]) if eq[0] > 0 else 0.0
        else:
            max_dd = 0.0
            total_ret = float(np.mean(rewards_arr))

        return EvalResult(
            model_id=Path(model_path).parent.name,
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            total_return=round(total_ret, 4),
            win_rate=round(float(np.mean(rewards_arr > 0)), 4),
            total_trades=n_episodes,
            equity_curve=all_equity[-100:] if all_equity else [],
        )

    def predict(
        self,
        model_path: str,
        obs: np.ndarray,
        algorithm: str = "ppo",
        deterministic: bool = True,
    ) -> tuple[Any, float]:
        """
        Get a single prediction from a trained model.

        Returns (action, confidence).
        """
        algo_cls = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG, "dqn": DQN}[
            algorithm.lower()
        ]

        model = algo_cls.load(model_path)
        action, _states = model.predict(obs, deterministic=deterministic)

        # Confidence: use action probability for discrete, value estimate for continuous
        confidence = 0.5  # default
        try:
            if hasattr(model.policy, "predict_values"):
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
                value = model.policy.predict_values(obs_tensor)
                confidence = float(torch.sigmoid(value).item())
        except Exception:
            pass

        return action, confidence

    def train_ensemble(
        self,
        env_train: gym.Env,
        env_val: gym.Env,
        algorithms: list[str] | None = None,
        total_timesteps: int | None = None,
        model_name: str | None = None,
    ) -> TrainResult:
        """
        Train multiple algorithms and select the best by validation Sharpe.

        Simplified walk-forward: train each algo, evaluate on validation env,
        pick the best.
        """
        algos = algorithms or self.config.ensemble_algorithms
        ts = total_timesteps or self.config.default_total_timesteps

        best_result = None
        best_sharpe = -np.inf
        results = {}

        for algo in algos:
            try:
                result = self.train(env_train, algorithm=algo, total_timesteps=ts)
                eval_result = self.evaluate(
                    result.checkpoint_path, env_val, algorithm=algo, n_episodes=5
                )
                results[algo] = {
                    "train": result,
                    "eval": eval_result,
                    "sharpe": eval_result.sharpe_ratio,
                }
                logger.info(
                    f"[FinRLTrainer] Ensemble: {algo.upper()} → "
                    f"Sharpe={eval_result.sharpe_ratio:.3f}"
                )
                if eval_result.sharpe_ratio > best_sharpe:
                    best_sharpe = eval_result.sharpe_ratio
                    best_result = result
            except Exception as e:
                logger.warning(f"[FinRLTrainer] Ensemble: {algo} failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("All ensemble algorithms failed to train.")

        best_result.metrics["ensemble_results"] = {
            algo: {"sharpe": r["sharpe"]} for algo, r in results.items()
        }
        best_result.metrics["ensemble_winner"] = best_result.algorithm

        if model_name:
            best_result.model_id = model_name

        return best_result
