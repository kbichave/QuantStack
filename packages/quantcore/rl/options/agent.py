"""
RL Agent for options trading direction prediction.

Implements the hybrid architecture:
- Agent outputs direction + confidence
- Sizing and contract selection are deterministic
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import random
from collections import deque

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. RL agent will use random policy.")

from quantcore.rl.base import RLAgent, Action, Experience
from quantcore.rl.options.environment import OptionsAction, OptionsEnvironment
from quantcore.strategy.base import (
    Strategy,
    MarketState,
    TargetPosition,
    DataRequirements,
    PositionDirection,
)


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:

    class DQNNetwork(nn.Module):
        """Deep Q-Network for direction prediction."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x):
            return self.network(x)


class DirectionAgent(RLAgent, Strategy):
    """
    RL agent that outputs direction and confidence.

    Implements both RLAgent interface for training and
    Strategy interface for live trading.

    Architecture:
    - DQN with experience replay
    - Epsilon-greedy exploration
    - Target network for stability
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        name: str = "DirectionAgent",
    ):
        """
        Initialize direction agent.

        Args:
            state_dim: State dimension
            action_dim: Number of actions (5 for direction agent)
            hidden_dim: Hidden layer size
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            name: Agent name
        """
        Strategy.__init__(self, name)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(
                self.device
            )
            self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(
                self.device
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        else:
            self.device = None
            self.policy_net = None
            self.target_net = None
            self.optimizer = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Training tracking
        self.training_steps = 0
        self.episode_count = 0

        # For Strategy interface
        self._last_action = OptionsAction.FLAT

    def select_action(self, state: np.ndarray, training: bool = True) -> Action:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state vector
            training: Whether in training mode

        Returns:
            Selected action
        """
        if not TORCH_AVAILABLE:
            # Random policy fallback
            action_idx = random.randint(0, self.action_dim - 1)
            return Action(value=action_idx)

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax(dim=1).item()

        self._last_action = OptionsAction(action_idx)
        return Action(value=action_idx)

    def train_step(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            Training metrics
        """
        # Store experience
        exp = Experience(
            state=state,
            action=action.value,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.replay_buffer.push(exp)

        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.batch_size:
            return {"buffer_size": len(self.replay_buffer)}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1

        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q.mean().item(),
            "buffer_size": len(self.replay_buffer),
        }

    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        Update agent from batch of experiences (required by RLAgent interface).

        Args:
            experiences: List of Experience objects

        Returns:
            Training metrics
        """
        # Add experiences to replay buffer
        for exp in experiences:
            self.replay_buffer.push(exp)

        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.batch_size:
            return {"buffer_size": len(self.replay_buffer)}

        # Sample batch and train
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1

        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q.mean().item(),
            "buffer_size": len(self.replay_buffer),
        }

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """
        Strategy interface: generate target positions.

        Args:
            state: Market state

        Returns:
            List of target positions
        """
        # Convert market state to feature vector
        features = state.to_feature_vector()

        # Pad/truncate to expected dimension
        if len(features) < self.state_dim:
            features = np.pad(features, (0, self.state_dim - len(features)))
        elif len(features) > self.state_dim:
            features = features[: self.state_dim]

        # Select action
        action = self.select_action(features, training=False)
        action_enum = OptionsAction(action.value)

        # Convert to target position
        direction, confidence = self._action_to_position(action_enum)

        if direction == PositionDirection.FLAT:
            return []

        return [
            TargetPosition(
                symbol=state.symbol,
                direction=direction,
                confidence=confidence,
                reason=f"RL action: {action_enum.name}",
            )
        ]

    def _action_to_position(
        self,
        action: OptionsAction,
    ) -> Tuple[PositionDirection, float]:
        """Convert action to position direction and confidence."""
        mapping = {
            OptionsAction.STRONG_LONG: (PositionDirection.LONG, 0.8),
            OptionsAction.WEAK_LONG: (PositionDirection.LONG, 0.4),
            OptionsAction.FLAT: (PositionDirection.FLAT, 0.0),
            OptionsAction.WEAK_SHORT: (PositionDirection.SHORT, 0.4),
            OptionsAction.STRONG_SHORT: (PositionDirection.SHORT, 0.8),
        }
        return mapping.get(action, (PositionDirection.FLAT, 0.0))

    def get_required_data(self) -> DataRequirements:
        """Specify data needs."""
        return DataRequirements(
            timeframes=["1H", "4H", "1D", "1W"],
            need_options_chain=True,
            need_earnings_calendar=True,
            lookback_bars=252,
        )

    def save(self, path: str) -> None:
        """Save model weights."""
        if TORCH_AVAILABLE and self.policy_net is not None:
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "training_steps": self.training_steps,
                    "episode_count": self.episode_count,
                },
                path,
            )
            logger.info(f"Saved agent to {path}")

    def load(self, path: str) -> None:
        """Load model weights."""
        if TORCH_AVAILABLE and self.policy_net is not None:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.training_steps = checkpoint["training_steps"]
            self.episode_count = checkpoint["episode_count"]
            logger.info(f"Loaded agent from {path}")


def train_direction_agent(
    env: OptionsEnvironment,
    agent: DirectionAgent,
    train_pct: float = 0.7,
    num_episodes: int = 100,
    max_steps_per_episode: int = 252,
    log_freq: int = 10,
) -> Dict[str, Any]:
    """
    Train direction agent on environment with proper train/test split.

    CRITICAL: Only trains on first train_pct of data. Test data is HOLDOUT.

    Args:
        env: Options environment
        agent: Direction agent to train
        train_pct: Percentage of data to use for training (default 70%)
        num_episodes: Number of episodes
        max_steps_per_episode: Max steps per episode
        log_freq: Episodes between logging

    Returns:
        Training metrics including test evaluation
    """
    # Calculate train/test split indices
    total_steps = len(env.data)
    train_end = int(total_steps * train_pct)

    assert train_end > 100, "Insufficient training data"
    assert train_end < total_steps - 50, "Insufficient test data"

    logger.info(
        f"RL Data split: train={train_end} steps, test={total_steps - train_end} steps"
    )
    logger.info(f"Train dates: {env.data.index[0]} to {env.data.index[train_end-1]}")
    logger.info(
        f"Test dates: {env.data.index[train_end]} to {env.data.index[-1]} (HOLDOUT)"
    )

    # Create training environment (subset of data)
    train_data = env.data.iloc[:train_end].copy()
    train_features = env.features.iloc[:train_end].copy()

    train_env = OptionsEnvironment(
        data=train_data,
        features=train_features,
        initial_equity=env.initial_equity,
        max_holding_days=env.max_holding_days,
        forced_exit_dte=env.forced_exit_dte,
    )

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "losses": [],
        "epsilons": [],
    }

    # Training loop - only uses train_env
    for episode in range(num_episodes):
        state = train_env.reset()
        episode_reward = 0
        episode_losses = []

        # Limit steps to training data
        max_steps = min(max_steps_per_episode, train_end - 1)

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state.features, training=True)

            # Take step
            next_state, reward, done, info = train_env.step(action)

            # Train
            train_metrics = agent.train_step(
                state.features,
                action,
                reward.value,
                next_state.features,
                done,
            )

            if "loss" in train_metrics:
                episode_losses.append(train_metrics["loss"])

            episode_reward += reward.value
            state = next_state

            if done:
                break

        agent.episode_count += 1

        # Record metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(step + 1)
        metrics["epsilons"].append(agent.epsilon)
        if episode_losses:
            metrics["losses"].append(np.mean(episode_losses))

        # Log
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(metrics["episode_rewards"][-log_freq:])
            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"reward={avg_reward:.2f}, epsilon={agent.epsilon:.3f}"
            )

    # Evaluate on TEST data (HOLDOUT - no training)
    test_data = env.data.iloc[train_end:].copy()
    test_features = env.features.iloc[train_end:].copy()

    test_env = OptionsEnvironment(
        data=test_data,
        features=test_features,
        initial_equity=env.initial_equity,
        max_holding_days=env.max_holding_days,
        forced_exit_dte=env.forced_exit_dte,
    )

    test_metrics = evaluate_agent(agent, test_env)

    metrics["train_steps"] = train_end
    metrics["test_steps"] = total_steps - train_end
    metrics["test_reward"] = test_metrics["total_reward"]
    metrics["test_final_equity"] = test_metrics["final_equity"]

    logger.info(
        f"Test evaluation: reward={test_metrics['total_reward']:.2f}, "
        f"final_equity={test_metrics['final_equity']:.2f}"
    )

    return metrics


def evaluate_agent(
    agent: DirectionAgent,
    env: OptionsEnvironment,
) -> Dict[str, float]:
    """
    Evaluate trained agent on environment (no training updates).

    Args:
        agent: Trained agent
        env: Environment to evaluate on

    Returns:
        Evaluation metrics
    """
    state = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        # Select action without exploration (training=False)
        action = agent.select_action(state.features, training=False)

        # Take step (no training)
        next_state, reward, done, info = env.step(action)

        total_reward += reward.value
        state = next_state
        steps += 1

        if done or steps >= len(env.data) - 1:
            break

    return {
        "total_reward": total_reward,
        "final_equity": info.get("equity", env.initial_equity),
        "steps": steps,
    }
