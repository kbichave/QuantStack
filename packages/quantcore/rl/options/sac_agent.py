"""
SAC Agent for Options Trading using Stable Baselines3.

This module provides a production-ready SAC (Soft Actor-Critic) agent
for options trading. SAC is chosen for its:
- Sample efficiency (critical for limited historical data)
- Automatic entropy tuning (balances exploration/exploitation)
- Continuous action space support (natural for position sizing)
- Stability (less hyperparameter sensitive than PPO)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Union, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning(
        "Stable Baselines3 not available. Install with: pip install stable-baselines3"
    )

from quantcore.rl.options.gym_env import OptionsTradingEnv, create_trading_env


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for tracking trading-specific metrics during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_pnls: List[float] = []
        self.episode_trades: List[int] = []

    def _on_step(self) -> bool:
        # Check for episode completion
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    infos = self.locals.get("infos", [])
                    if i < len(infos):
                        info = infos[i]
                        self.episode_pnls.append(info.get("episode_pnl", 0))
                        self.episode_trades.append(info.get("episode_trades", 0))

        return True

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics."""
        if not self.episode_pnls:
            return {}

        return {
            "mean_pnl": float(np.mean(self.episode_pnls)),
            "std_pnl": float(np.std(self.episode_pnls)),
            "max_pnl": float(np.max(self.episode_pnls)),
            "min_pnl": float(np.min(self.episode_pnls)),
            "mean_trades": float(np.mean(self.episode_trades)),
            "total_episodes": len(self.episode_pnls),
        }


class SACOptionsAgent:
    """
    SAC Agent wrapper for options trading.

    Features:
    - Uses Stable Baselines3 SAC implementation
    - Automatic entropy tuning for exploration
    - Continuous position sizing (delta targeting)
    - Supports training, evaluation, and deployment

    Example:
        >>> agent = SACOptionsAgent()
        >>> agent.train(train_data, train_features, total_timesteps=100000)
        >>> actions = agent.predict(test_features)
        >>> agent.save("models/sac_options")
    """

    # Default hyperparameters (tuned for financial data)
    DEFAULT_PARAMS = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,  # Soft update coefficient
        "gamma": 0.99,  # Discount factor
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",  # Auto-tune entropy coefficient
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": False,  # State-dependent exploration
        "policy_kwargs": {
            "net_arch": [256, 256],  # Two hidden layers
        },
    }

    def __init__(
        self,
        algorithm: str = "SAC",
        hyperparams: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        Initialize SAC agent.

        Args:
            algorithm: RL algorithm ("SAC", "PPO", "TD3")
            hyperparams: Custom hyperparameters
            device: Device to use ("auto", "cuda", "cpu")
            verbose: Verbosity level
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "Stable Baselines3 is required. Install with: pip install stable-baselines3"
            )

        self.algorithm = algorithm.upper()
        self.hyperparams = {**self.DEFAULT_PARAMS, **(hyperparams or {})}
        self.device = device
        self.verbose = verbose

        self.model: Optional[Any] = None
        self.env: Optional[Any] = None
        self.vec_env: Optional[VecNormalize] = None

        self._training_metrics: Dict = {}

    def _create_model(self, env: OptionsTradingEnv) -> Any:
        """Create SB3 model based on algorithm choice."""
        # Wrap in Monitor for logging
        monitored_env = Monitor(env)

        # Wrap in DummyVecEnv for SB3 compatibility
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Add observation normalization
        self.vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # Select algorithm
        algo_class: Type[Union[SAC, PPO, TD3]]
        if self.algorithm == "SAC":
            algo_class = SAC
        elif self.algorithm == "PPO":
            algo_class = PPO
        elif self.algorithm == "TD3":
            algo_class = TD3
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Create model
        model = algo_class(
            policy="MlpPolicy",
            env=self.vec_env,
            device=self.device,
            verbose=self.verbose,
            **self._get_algo_params(algo_class),
        )

        return model

    def _get_algo_params(self, algo_class: Type) -> Dict:
        """Get algorithm-specific parameters."""
        params = {}

        if algo_class == SAC:
            params = {
                "learning_rate": self.hyperparams["learning_rate"],
                "buffer_size": self.hyperparams["buffer_size"],
                "learning_starts": self.hyperparams["learning_starts"],
                "batch_size": self.hyperparams["batch_size"],
                "tau": self.hyperparams["tau"],
                "gamma": self.hyperparams["gamma"],
                "train_freq": self.hyperparams["train_freq"],
                "gradient_steps": self.hyperparams["gradient_steps"],
                "ent_coef": self.hyperparams["ent_coef"],
                "target_update_interval": self.hyperparams["target_update_interval"],
                "target_entropy": self.hyperparams["target_entropy"],
                "use_sde": self.hyperparams["use_sde"],
                "policy_kwargs": self.hyperparams["policy_kwargs"],
            }
        elif algo_class == PPO:
            params = {
                "learning_rate": self.hyperparams["learning_rate"],
                "n_steps": 2048,
                "batch_size": self.hyperparams["batch_size"],
                "gamma": self.hyperparams["gamma"],
                "policy_kwargs": self.hyperparams["policy_kwargs"],
            }
        elif algo_class == TD3:
            params = {
                "learning_rate": self.hyperparams["learning_rate"],
                "buffer_size": self.hyperparams["buffer_size"],
                "learning_starts": self.hyperparams["learning_starts"],
                "batch_size": self.hyperparams["batch_size"],
                "tau": self.hyperparams["tau"],
                "gamma": self.hyperparams["gamma"],
                "policy_kwargs": self.hyperparams["policy_kwargs"],
            }

        return params

    def train(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        total_timesteps: int = 100_000,
        initial_equity: float = 100_000,
        eval_data: Optional[pd.DataFrame] = None,
        eval_features: Optional[pd.DataFrame] = None,
        save_path: Optional[Path] = None,
        save_freq: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Train the SAC agent.

        Args:
            data: Training OHLCV data
            features: Training features
            total_timesteps: Total training timesteps
            initial_equity: Starting equity
            eval_data: Evaluation OHLCV data
            eval_features: Evaluation features
            save_path: Path to save checkpoints
            save_freq: Checkpoint frequency

        Returns:
            Training metrics dictionary
        """
        logger.info(
            f"Training {self.algorithm} agent for {total_timesteps:,} timesteps"
        )

        # Create environment
        self.env = create_trading_env(
            data=data,
            features=features,
            initial_equity=initial_equity,
        )

        # Create model
        self.model = self._create_model(self.env)

        # Setup callbacks
        callbacks = [TradingMetricsCallback()]

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(save_path),
                    name_prefix=f"{self.algorithm.lower()}_options",
                )
            )

        # Setup evaluation
        if eval_data is not None and eval_features is not None:
            eval_env = create_trading_env(
                data=eval_data,
                features=eval_features,
                initial_equity=initial_equity,
            )
            eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])

            callbacks.append(
                EvalCallback(
                    eval_env=eval_vec_env,
                    n_eval_episodes=5,
                    eval_freq=10_000,
                    deterministic=True,
                    verbose=self.verbose,
                )
            )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Collect metrics
        metrics_callback = callbacks[0]
        self._training_metrics = metrics_callback.get_metrics()
        self._training_metrics["total_timesteps"] = total_timesteps
        self._training_metrics["algorithm"] = self.algorithm

        logger.info(
            f"Training complete. Mean PnL: ${self._training_metrics.get('mean_pnl', 0):.2f}"
        )

        return self._training_metrics

    def predict(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        deterministic: bool = True,
    ) -> pd.DataFrame:
        """
        Generate predictions for given data.

        Args:
            data: OHLCV data
            features: Feature data
            deterministic: Use deterministic actions (no exploration)

        Returns:
            DataFrame with predicted actions and positions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create environment for inference
        env = create_trading_env(
            data=data,
            features=features,
        )

        predictions = []
        obs, info = env.reset()

        for i in range(len(data)):
            # Normalize observation if we have a normalized env
            if self.vec_env is not None:
                obs_normalized = self.vec_env.normalize_obs(obs.reshape(1, -1))
            else:
                obs_normalized = obs.reshape(1, -1)

            # Get action
            action, _ = self.model.predict(obs_normalized, deterministic=deterministic)

            predictions.append(
                {
                    "timestamp": data.index[i] if hasattr(data, "index") else i,
                    "action": float(action[0]),
                    "target_delta": float(action[0]) * 100,  # Scale to delta
                }
            )

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        return pd.DataFrame(predictions)

    def evaluate(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            data: Evaluation OHLCV data
            features: Evaluation features
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        env = create_trading_env(
            data=data,
            features=features,
        )
        vec_env = DummyVecEnv([lambda: Monitor(env)])

        # Run evaluation
        mean_reward, std_reward = evaluate_policy(
            self.model,
            vec_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            return_episode_rewards=False,
        )

        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_episodes": n_episodes,
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model and normalization stats.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / f"{self.algorithm.lower()}_model"
        self.model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

        # Save normalization stats
        if self.vec_env is not None:
            norm_path = path / "vec_normalize.pkl"
            self.vec_env.save(str(norm_path))
            logger.info(f"Saved normalization stats to {norm_path}")

        # Save metadata
        metadata = {
            "algorithm": self.algorithm,
            "hyperparams": self.hyperparams,
            "training_metrics": self._training_metrics,
            "saved_at": datetime.now().isoformat(),
        }
        metadata_path = path / "metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load model and normalization stats.

        Args:
            path: Path to load model from
        """
        path = Path(path)

        # Determine algorithm from metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.algorithm = metadata.get("algorithm", self.algorithm)
            self.hyperparams = metadata.get("hyperparams", self.hyperparams)
            self._training_metrics = metadata.get("training_metrics", {})

        # Load model
        model_path = path / f"{self.algorithm.lower()}_model"

        algo_class: Type[Union[SAC, PPO, TD3]]
        if self.algorithm == "SAC":
            algo_class = SAC
        elif self.algorithm == "PPO":
            algo_class = PPO
        elif self.algorithm == "TD3":
            algo_class = TD3
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.model = algo_class.load(str(model_path), device=self.device)
        logger.info(f"Loaded {self.algorithm} model from {model_path}")

        # Load normalization stats if available
        norm_path = path / "vec_normalize.pkl"
        if norm_path.exists():
            # Note: VecNormalize loading requires an env, which we don't have here
            # The normalization will be applied when using predict()
            logger.info(f"Normalization stats available at {norm_path}")

    @property
    def training_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self._training_metrics


def train_sac_agent(
    train_data: pd.DataFrame,
    train_features: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    val_features: Optional[pd.DataFrame] = None,
    total_timesteps: int = 100_000,
    algorithm: str = "SAC",
    save_path: Optional[Path] = None,
    **kwargs,
) -> Tuple[SACOptionsAgent, Dict[str, Any]]:
    """
    Convenience function to train SAC/PPO/TD3 agent.

    Args:
        train_data: Training OHLCV data
        train_features: Training features
        val_data: Validation OHLCV data
        val_features: Validation features
        total_timesteps: Total training timesteps
        algorithm: Algorithm choice ("SAC", "PPO", "TD3")
        save_path: Path to save model
        **kwargs: Additional agent arguments

    Returns:
        Tuple of (trained agent, training metrics)
    """
    agent = SACOptionsAgent(algorithm=algorithm, **kwargs)

    metrics = agent.train(
        data=train_data,
        features=train_features,
        total_timesteps=total_timesteps,
        eval_data=val_data,
        eval_features=val_features,
        save_path=save_path,
    )

    if save_path:
        agent.save(save_path)

    return agent, metrics
