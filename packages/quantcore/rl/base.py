"""
Base classes for Reinforcement Learning agents.

Provides foundational abstractions for all RL layers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import random
from loguru import logger

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. RL agents will use fallback implementations."
    )


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class State:
    """
    Represents the state observed by an RL agent.

    Attributes:
        features: Feature vector (numpy array)
        timestamp: Time of observation
        metadata: Additional context (regime, volatility, etc.)
    """

    features: np.ndarray
    timestamp: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tensor(self) -> "torch.Tensor":
        """Convert to PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        return torch.FloatTensor(self.features)

    @property
    def dim(self) -> int:
        """State dimension."""
        return len(self.features)


@dataclass
class Action:
    """
    Represents an action taken by an RL agent.

    Attributes:
        value: Action value (can be discrete index or continuous vector)
        action_type: Type of action (for multi-action spaces)
        metadata: Additional action context
    """

    value: Union[int, float, np.ndarray]
    action_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_discrete(self) -> bool:
        """Check if action is discrete."""
        return isinstance(self.value, (int, np.integer))

    @property
    def is_continuous(self) -> bool:
        """Check if action is continuous."""
        return isinstance(self.value, (float, np.floating, np.ndarray))


@dataclass
class Reward:
    """
    Represents the reward received after taking an action.

    Attributes:
        value: Reward value
        components: Breakdown of reward components
        info: Additional information
    """

    value: float
    components: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value


@dataclass
class Experience:
    """
    A single experience tuple (s, a, r, s', done).

    Used for replay buffer storage.
    """

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Replay Buffer
# ============================================================================


class ReplayBuffer:
    """
    Experience replay buffer for RL training.

    Stores experiences and provides random sampling for training.
    Supports prioritized experience replay (optional).
    """

    def __init__(
        self,
        capacity: int = 100000,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            prioritized: Whether to use prioritized replay
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta

        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, experience: Experience, priority: Optional[float] = None) -> None:
        """
        Add experience to buffer.

        Args:
            experience: Experience tuple
            priority: Priority for prioritized replay (uses max if None)
        """
        self.buffer.append(experience)

        if self.prioritized:
            p = priority if priority is not None else self.max_priority
            self.priorities.append(p)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of Experience objects
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        if self.prioritized:
            return self._prioritized_sample(batch_size)
        else:
            return random.sample(list(self.buffer), batch_size)

    def _prioritized_sample(self, batch_size: int) -> List[Experience]:
        """Sample with prioritization."""
        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for experiences."""
        if not self.prioritized:
            return

        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size


# ============================================================================
# Base Agent Class
# ============================================================================


class RLAgent(ABC):
    """
    Abstract base class for all RL agents.

    Subclasses must implement:
    - select_action: Choose action given state
    - update: Learn from experience
    - get_state_dim: Return state dimension
    - get_action_dim: Return action dimension
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        """
        Initialize RL agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device to use (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device

        self.training = True
        self.episode_count = 0
        self.step_count = 0

    @abstractmethod
    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select action given state.

        Args:
            state: Current state
            explore: Whether to explore (add noise/epsilon)

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        Update agent from experiences.

        Args:
            experiences: Batch of experiences

        Returns:
            Dictionary of loss values and metrics
        """
        pass

    def train(self) -> None:
        """Set agent to training mode."""
        self.training = True

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False

    def save(self, path: str) -> None:
        """Save agent to file."""
        pass

    def load(self, path: str) -> None:
        """Load agent from file."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "device": self.device,
        }


# ============================================================================
# Base Environment Class
# ============================================================================


class RLEnvironment(ABC):
    """
    Abstract base class for RL environments.

    Follows OpenAI Gym-style interface.
    """

    def __init__(self):
        """Initialize environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

    @abstractmethod
    def reset(self) -> State:
        """
        Reset environment to initial state.

        Returns:
            Initial state
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]:
        """
        Take action in environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return state dimension."""
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        """Return action dimension."""
        pass

    def render(self) -> None:
        """Render environment (optional)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass


# ============================================================================
# Neural Network Building Blocks (if PyTorch available)
# ============================================================================

if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """Multi-layer perceptron for RL agents."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = [256, 256],
            activation: str = "relu",
            output_activation: Optional[str] = None,
        ):
            super().__init__()

            layers = []
            prev_dim = input_dim

            # Hidden layers
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))

            if output_activation == "tanh":
                layers.append(nn.Tanh())
            elif output_activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif output_activation == "softmax":
                layers.append(nn.Softmax(dim=-1))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    class DuelingNetwork(nn.Module):
        """Dueling DQN architecture."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = [256, 256],
        ):
            super().__init__()

            # Shared feature extraction
            self.feature = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
            )

            # Value stream
            self.value = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1),
            )

            # Advantage stream
            self.advantage = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature(x)
            value = self.value(features)
            advantage = self.advantage(features)

            # Q = V + (A - mean(A))
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q

    class ActorCritic(nn.Module):
        """Actor-Critic network for policy gradient methods."""

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int] = [256, 256],
            continuous: bool = False,
        ):
            super().__init__()

            self.continuous = continuous

            # Shared features
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(),
            )

            # Actor (policy)
            if continuous:
                self.actor_mean = nn.Sequential(
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], action_dim),
                    nn.Tanh(),
                )
                self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            else:
                self.actor = nn.Sequential(
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], action_dim),
                    nn.Softmax(dim=-1),
                )

            # Critic (value)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1),
            )

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning policy and value."""
            shared = self.shared(state)

            if self.continuous:
                mean = self.actor_mean(shared)
                std = self.actor_log_std.exp()
                policy = (mean, std)
            else:
                policy = self.actor(shared)

            value = self.critic(shared)

            return policy, value

        def get_action(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Get action and log probability."""
            shared = self.shared(state)

            if self.continuous:
                mean = self.actor_mean(shared)
                std = self.actor_log_std.exp()

                if deterministic:
                    action = mean
                else:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()

                log_prob = (
                    torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
                )
            else:
                probs = self.actor(shared)

                if deterministic:
                    action = probs.argmax(dim=-1)
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()

                log_prob = torch.log(
                    probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8
                )

            return action, log_prob


# ============================================================================
# Utility Functions
# ============================================================================


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    """Soft update of target network parameters."""
    if not TORCH_AVAILABLE:
        return

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Hard update (copy) of target network parameters."""
    if not TORCH_AVAILABLE:
        return

    target.load_state_dict(source.state_dict())


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
    normalize: bool = True,
) -> np.ndarray:
    """Compute discounted returns."""
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = np.array(returns)

    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    gae = 0

    values = list(values) + [next_value]

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return np.array(advantages), np.array(returns)
