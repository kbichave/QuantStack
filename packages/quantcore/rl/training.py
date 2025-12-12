"""
RL Training infrastructure.

Provides training loops, logging, and evaluation for RL agents.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from loguru import logger

from quantcore.rl.base import (
    RLAgent,
    RLEnvironment,
    State,
    Action,
    Reward,
    Experience,
    ReplayBuffer,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for RL training."""

    # Training
    total_timesteps: int = 100000
    batch_size: int = 64
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1

    # Exploration
    exploration_initial: float = 1.0
    exploration_final: float = 0.01
    exploration_fraction: float = 0.1

    # Target network
    target_update_interval: int = 100
    tau: float = 0.005

    # Replay buffer
    buffer_size: int = 100000
    prioritized_replay: bool = False

    # Evaluation
    eval_freq: int = 1000
    n_eval_episodes: int = 5

    # Logging
    log_interval: int = 100
    tensorboard_log: Optional[str] = None

    # Checkpointing
    save_freq: int = 10000
    save_path: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: Dict[str, List[float]] = field(default_factory=lambda: {})
    eval_rewards: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add_episode(self, reward: float, length: int) -> None:
        """Add episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.timestamps.append(time.time())

    def add_loss(self, name: str, value: float) -> None:
        """Add loss value."""
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(value)

    def get_recent_reward(self, n: int = 100) -> float:
        """Get average of recent episode rewards."""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-n:])

    def get_recent_length(self, n: int = 100) -> float:
        """Get average of recent episode lengths."""
        if not self.episode_lengths:
            return 0.0
        return np.mean(self.episode_lengths[-n:])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "eval_rewards": self.eval_rewards,
        }


class RLTrainer:
    """
    Trainer for RL agents.

    Handles:
    - Training loop
    - Experience collection
    - Logging
    - Evaluation
    - Checkpointing
    """

    def __init__(
        self,
        agent: RLAgent,
        env: RLEnvironment,
        config: Optional[TrainingConfig] = None,
        eval_env: Optional[RLEnvironment] = None,
    ):
        """
        Initialize trainer.

        Args:
            agent: RL agent to train
            env: Training environment
            config: Training configuration
            eval_env: Evaluation environment (uses env if None)
        """
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or env
        self.config = config or TrainingConfig()

        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            prioritized=self.config.prioritized_replay,
        )

        # Metrics
        self.metrics = TrainingMetrics()

        # State
        self.current_step = 0
        self.current_episode = 0
        self.exploration_rate = self.config.exploration_initial

    def train(self, callback: Optional[Callable] = None) -> TrainingMetrics:
        """
        Run training loop.

        Args:
            callback: Optional callback called after each step

        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {self.config.total_timesteps} timesteps")

        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.config.total_timesteps):
            self.current_step = step

            # Update exploration rate
            self._update_exploration_rate()

            # Select action
            action = self.agent.select_action(state, explore=True)

            # Apply exploration noise/epsilon (if applicable)
            action = self._apply_exploration(action)

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )
            self.buffer.push(experience)

            # Update state
            state = next_state
            episode_reward += reward.value
            episode_length += 1

            # Train agent
            if (
                step >= self.config.learning_starts
                and step % self.config.train_freq == 0
            ):
                for _ in range(self.config.gradient_steps):
                    if self.buffer.is_ready(self.config.batch_size):
                        experiences = self.buffer.sample(self.config.batch_size)
                        loss_info = self.agent.update(experiences)

                        for name, value in loss_info.items():
                            self.metrics.add_loss(name, value)

            # Episode done
            if done:
                self.metrics.add_episode(episode_reward, episode_length)
                self.current_episode += 1

                # Reset
                state = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            # Logging
            if step % self.config.log_interval == 0 and step > 0:
                self._log_progress(step)

            # Evaluation
            if step % self.config.eval_freq == 0 and step > 0:
                eval_reward = self.evaluate()
                self.metrics.eval_rewards.append(eval_reward)

            # Checkpointing
            if self.config.save_path and step % self.config.save_freq == 0 and step > 0:
                self.save_checkpoint(step)

            # Callback
            if callback:
                callback(self, step)

        logger.info("Training complete")
        return self.metrics

    def evaluate(self, n_episodes: Optional[int] = None) -> float:
        """
        Evaluate agent.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Mean evaluation reward
        """
        n_episodes = n_episodes or self.config.n_eval_episodes

        self.agent.eval()
        rewards = []

        for _ in range(n_episodes):
            state = self.eval_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self.agent.select_action(state, explore=False)
                state, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward.value

            rewards.append(episode_reward)

        self.agent.train()

        mean_reward = np.mean(rewards)
        logger.info(
            f"Evaluation: mean reward = {mean_reward:.2f} (+/- {np.std(rewards):.2f})"
        )

        return mean_reward

    def _update_exploration_rate(self) -> None:
        """Update exploration rate based on schedule."""
        fraction = min(
            1.0,
            self.current_step
            / (self.config.total_timesteps * self.config.exploration_fraction),
        )
        self.exploration_rate = self.config.exploration_initial + fraction * (
            self.config.exploration_final - self.config.exploration_initial
        )

    def _apply_exploration(self, action: Action) -> Action:
        """Apply exploration to action."""
        # For discrete actions, epsilon-greedy is handled in agent
        # For continuous actions, noise is added in agent
        # This method can be overridden for custom exploration
        return action

    def _log_progress(self, step: int) -> None:
        """Log training progress."""
        recent_reward = self.metrics.get_recent_reward(100)
        recent_length = self.metrics.get_recent_length(100)

        loss_str = ""
        for name, values in self.metrics.losses.items():
            if values:
                loss_str += f", {name}={np.mean(values[-100:]):.4f}"

        logger.info(
            f"Step {step}/{self.config.total_timesteps} | "
            f"Episodes: {self.current_episode} | "
            f"Reward: {recent_reward:.2f} | "
            f"Length: {recent_length:.1f} | "
            f"Exploration: {self.exploration_rate:.3f}"
            f"{loss_str}"
        )

    def save_checkpoint(self, step: int) -> None:
        """Save training checkpoint."""
        if not self.config.save_path:
            return

        save_dir = Path(self.config.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save agent
        agent_path = save_dir / f"agent_step_{step}.pt"
        self.agent.save(str(agent_path))

        # Save metrics
        metrics_path = save_dir / f"metrics_step_{step}.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics.to_dict(), f)

        # Save config
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f)

        logger.info(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        save_dir = Path(path)

        # Find latest agent checkpoint
        agent_files = list(save_dir.glob("agent_step_*.pt"))
        if agent_files:
            latest = max(agent_files, key=lambda f: int(f.stem.split("_")[-1]))
            self.agent.load(str(latest))
            self.current_step = int(latest.stem.split("_")[-1])
            logger.info(f"Loaded checkpoint from step {self.current_step}")


class OnlineRLTrainer(RLTrainer):
    """
    Online RL trainer for live trading scenarios.

    Differs from standard trainer:
    - No replay buffer (learns from single experiences)
    - Processes streaming data
    - Adapts to non-stationary environments
    """

    def __init__(
        self,
        agent: RLAgent,
        env: RLEnvironment,
        config: Optional[TrainingConfig] = None,
    ):
        super().__init__(agent, env, config)

        # Online learning doesn't use large replay buffer
        self.buffer = ReplayBuffer(capacity=1000)

    def step(self, state: State) -> Action:
        """
        Take single step in online mode.

        Args:
            state: Current market state

        Returns:
            Action to take
        """
        action = self.agent.select_action(state, explore=self.agent.training)
        return action

    def update_from_outcome(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        done: bool,
    ) -> Dict[str, float]:
        """
        Update agent from single outcome.

        Args:
            state: State when action was taken
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            Loss information
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.buffer.push(experience)

        # Update if enough samples
        if self.buffer.is_ready(min(self.config.batch_size, len(self.buffer))):
            experiences = self.buffer.sample(
                min(self.config.batch_size, len(self.buffer))
            )
            return self.agent.update(experiences)

        return {}


# ============================================================================
# Evaluation Utilities
# ============================================================================


def evaluate_trading_agent(
    agent: RLAgent,
    env: RLEnvironment,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    Evaluate trading agent with trading-specific metrics.

    Args:
        agent: Agent to evaluate
        env: Environment to evaluate in
        n_episodes: Number of episodes

    Returns:
        Dictionary of metrics
    """
    agent.eval()

    total_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False

        returns = []
        equity_curve = [1.0]

        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, done, info = env.step(action)

            returns.append(reward.value)
            equity_curve.append(equity_curve[-1] * (1 + reward.value))

        # Calculate metrics
        total_return = equity_curve[-1] / equity_curve[0] - 1
        total_returns.append(total_return)

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            sharpe_ratios.append(sharpe)

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdowns.append(np.max(drawdown))

        # Win rate
        win_rate = np.mean([r > 0 for r in returns]) if returns else 0
        win_rates.append(win_rate)

    agent.train()

    return {
        "mean_return": np.mean(total_returns),
        "std_return": np.std(total_returns),
        "mean_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0,
        "mean_max_drawdown": np.mean(max_drawdowns),
        "mean_win_rate": np.mean(win_rates),
    }
