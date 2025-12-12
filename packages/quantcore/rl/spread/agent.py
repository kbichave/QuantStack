"""
Spread Arbitrage RL Agent.

DQN agent for WTI-Brent spread trading.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from quantcore.rl.base import (
    RLAgent,
    State,
    Action,
    Experience,
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


class SpreadArbitrageAgent(RLAgent):
    """
    DQN agent for spread arbitrage.

    Specializes in:
    - Mean reversion on WTI-Brent spread
    - Z-score based entry/exit
    - Regime-aware trading

    Actions:
    - 0: Close position
    - 1: Small long spread (25%)
    - 2: Full long spread (100%)
    - 3: Small short spread (25%)
    - 4: Full short spread (100%)
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 5,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        tau: float = 0.005,
        zscore_entry_threshold: float = 1.5,
        zscore_exit_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize spread arbitrage agent.

        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration
            epsilon_end: Final exploration
            epsilon_decay: Exploration decay
            target_update_freq: Target network update frequency
            tau: Soft update coefficient
            zscore_entry_threshold: Z-score threshold for entry
            zscore_exit_threshold: Z-score threshold for exit
            device: Device to use
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma, device)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.zscore_entry_threshold = zscore_entry_threshold
        self.zscore_exit_threshold = zscore_exit_threshold
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

        # Use dueling architecture
        self.q_network = DuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dims,
        ).to(self.device)

        self.target_network = DuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_dims,
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select action with spread-specific logic.

        Args:
            state: Current state
            explore: Whether to explore

        Returns:
            Selected action
        """
        # Epsilon-greedy with spread heuristics
        if explore and self.training and np.random.random() < self.epsilon:
            # Use informed exploration based on z-score
            zscore = state.features[0] * 3  # Denormalize
            position_dir = state.features[4]

            if zscore < -self.zscore_entry_threshold and position_dir <= 0:
                # Oversold, no long position -> favor long
                action_idx = np.random.choice([1, 2], p=[0.6, 0.4])
            elif zscore > self.zscore_entry_threshold and position_dir >= 0:
                # Overbought, no short position -> favor short
                action_idx = np.random.choice([3, 4], p=[0.6, 0.4])
            elif abs(zscore) < self.zscore_exit_threshold and position_dir != 0:
                # Near mean, have position -> consider closing
                action_idx = np.random.choice(
                    [0, 1, 2, 3, 4], p=[0.5, 0.125, 0.125, 0.125, 0.125]
                )
            else:
                action_idx = np.random.randint(self.action_dim)

            return Action(value=action_idx, action_type="explore")

        # Greedy action from Q-network
        if TORCH_AVAILABLE and self.q_network is not None:
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax(dim=-1).item()
        else:
            action_idx = self._heuristic_action(state)

        return Action(value=action_idx, action_type="greedy")

    def _heuristic_action(self, state: State) -> int:
        """Heuristic action when no network available."""
        zscore = state.features[0] * 3  # Denormalize
        position_dir = state.features[4]
        position_size = state.features[5]
        bars_held = state.features[7] * 50  # Denormalize

        # Entry logic
        if position_dir == 0:
            if zscore < -self.zscore_entry_threshold:
                return 2 if zscore < -2 else 1  # Long spread
            elif zscore > self.zscore_entry_threshold:
                return 4 if zscore > 2 else 3  # Short spread
            return 0  # Stay flat

        # Exit logic
        if position_dir > 0:  # Long spread
            if zscore > -self.zscore_exit_threshold or bars_held > 30:
                return 0  # Close
        elif position_dir < 0:  # Short spread
            if zscore < self.zscore_exit_threshold or bars_held > 30:
                return 0  # Close

        # Hold
        return 1 if position_dir > 0 else 3

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

        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
            next_q = (
                self.target_network(next_states).gather(1, next_actions).squeeze(-1)
            )
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
                "zscore_entry_threshold": self.zscore_entry_threshold,
                "zscore_exit_threshold": self.zscore_exit_threshold,
                "hidden_dims": self.hidden_dims,
            }
        )
        return config
