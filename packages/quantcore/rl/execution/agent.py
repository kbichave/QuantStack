"""
Execution RL Agent.

DQN-based agent for optimal order execution.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from quantcore.rl.base import (
    RLAgent,
    State,
    Action,
    Experience,
    ReplayBuffer,
    MLP,
    DuelingNetwork,
    soft_update,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F


class ExecutionRLAgent(RLAgent):
    """
    DQN agent for execution optimization.

    Uses Dueling DQN architecture with:
    - Separate value and advantage streams
    - Double DQN for stable learning
    - Prioritized experience replay (optional)

    Actions:
    - 0: Wait
    - 1: Small limit (10%)
    - 2: Medium limit (25%)
    - 3: Large limit (50%)
    - 4: Market order (100%)
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 5,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        tau: float = 0.005,
        double_dqn: bool = True,
        dueling: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize execution agent.

        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: Target network update frequency
            tau: Soft update coefficient
            double_dqn: Use Double DQN
            dueling: Use Dueling architecture
            device: Device to use
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma, device)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.hidden_dims = hidden_dims

        self._build_networks()

    def _build_networks(self) -> None:
        """Build neural networks."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using fallback agent")
            self.q_network = None
            self.target_network = None
            self.optimizer = None
            return

        # Q-network
        if self.dueling:
            self.q_network = DuelingNetwork(
                self.state_dim,
                self.action_dim,
                self.hidden_dims,
            ).to(self.device)
        else:
            self.q_network = MLP(
                self.state_dim,
                self.action_dim,
                self.hidden_dims,
            ).to(self.device)

        # Target network
        if self.dueling:
            self.target_network = DuelingNetwork(
                self.state_dim,
                self.action_dim,
                self.hidden_dims,
            ).to(self.device)
        else:
            self.target_network = MLP(
                self.state_dim,
                self.action_dim,
                self.hidden_dims,
            ).to(self.device)

        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select action using epsilon-greedy.

        Args:
            state: Current state
            explore: Whether to explore

        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if explore and self.training and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
            return Action(value=action_idx, action_type="random")

        # Greedy action
        if TORCH_AVAILABLE and self.q_network is not None:
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax(dim=-1).item()
        else:
            # Fallback: simple heuristic
            action_idx = self._heuristic_action(state)

        return Action(value=action_idx, action_type="greedy")

    def _heuristic_action(self, state: State) -> int:
        """Heuristic action when no neural network available."""
        remaining_qty_frac = state.features[0]
        remaining_time_frac = state.features[1]

        # Urgency-based heuristic
        urgency = remaining_qty_frac / max(remaining_time_frac, 0.1)

        if urgency > 2.0:
            return 4  # Market order (urgent)
        elif urgency > 1.0:
            return 3  # Large limit
        elif urgency > 0.5:
            return 2  # Medium limit
        elif remaining_time_frac > 0.5:
            return 0  # Wait (plenty of time)
        else:
            return 1  # Small limit

    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        Update agent from experiences.

        Args:
            experiences: Batch of experiences

        Returns:
            Loss information
        """
        if not TORCH_AVAILABLE or self.q_network is None:
            return {"loss": 0.0}

        # Prepare batch
        states = torch.FloatTensor(
            np.array([e.state.features for e in experiences])
        ).to(self.device)

        actions = (
            torch.LongTensor([e.action.value for e in experiences])
            .unsqueeze(-1)
            .to(self.device)
        )

        rewards = torch.FloatTensor([e.reward.value for e in experiences]).to(
            self.device
        )

        next_states = torch.FloatTensor(
            np.array([e.next_state.features for e in experiences])
        ).to(self.device)

        dones = torch.FloatTensor([float(e.done) for e in experiences]).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions).squeeze(-1)

        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network for action selection
                next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
                next_q = (
                    self.target_network(next_states).gather(1, next_actions).squeeze(-1)
                )
            else:
                next_q = self.target_network(next_states).max(dim=-1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            soft_update(self.target_network, self.q_network, self.tau)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def save(self, path: str) -> None:
        """Save agent to file."""
        if not TORCH_AVAILABLE or self.q_network is None:
            return

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
                "config": self.get_config(),
            },
            path,
        )
        logger.info(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent from file."""
        if not TORCH_AVAILABLE or self.q_network is None:
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.step_count = checkpoint.get("step_count", 0)
        logger.info(f"Agent loaded from {path}")

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_update_freq": self.target_update_freq,
                "tau": self.tau,
                "double_dqn": self.double_dqn,
                "dueling": self.dueling,
                "hidden_dims": self.hidden_dims,
            }
        )
        return config
