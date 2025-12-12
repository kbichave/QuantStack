"""
Alpha Selection RL Agent.

Contextual bandit / DQN agent for alpha selection.
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
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F


class AlphaSelectionAgent(RLAgent):
    """
    Agent for selecting which alpha to follow.

    Uses:
    - Thompson Sampling / Upper Confidence Bound for exploration
    - DQN-style learning for contextual decisions
    - Regime-conditioned policy

    Actions:
    - Select one of N alphas (discrete)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        ucb_coef: float = 2.0,
        use_ucb: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize alpha selection agent.

        Args:
            state_dim: State dimension
            action_dim: Number of alphas + optional no-trade
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration
            epsilon_end: Final exploration
            epsilon_decay: Exploration decay
            ucb_coef: UCB exploration coefficient
            use_ucb: Use UCB exploration (vs epsilon-greedy)
            device: Device to use
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma, device)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.ucb_coef = ucb_coef
        self.use_ucb = use_ucb
        self.hidden_dims = hidden_dims

        # UCB statistics
        self.action_counts = np.zeros(action_dim)
        self.action_rewards = np.zeros(action_dim)

        self._build_networks()

    def _build_networks(self) -> None:
        """Build neural networks."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using fallback agent")
            self.q_network = None
            self.optimizer = None
            return

        self.q_network = MLP(
            self.state_dim,
            self.action_dim,
            self.hidden_dims,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select alpha using UCB or epsilon-greedy.

        Args:
            state: Current state
            explore: Whether to explore

        Returns:
            Selected action
        """
        if TORCH_AVAILABLE and self.q_network is not None:
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy().flatten()
        else:
            q_values = self._heuristic_values(state)

        if explore and self.training:
            if self.use_ucb:
                action_idx = self._ucb_selection(q_values)
            else:
                action_idx = self._epsilon_greedy(q_values)
        else:
            action_idx = int(np.argmax(q_values))

        return Action(value=action_idx, action_type="discrete")

    def _epsilon_greedy(self, q_values: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        return int(np.argmax(q_values))

    def _ucb_selection(self, q_values: np.ndarray) -> int:
        """UCB action selection."""
        total_counts = np.sum(self.action_counts) + 1

        ucb_values = q_values.copy()

        for a in range(self.action_dim):
            if self.action_counts[a] == 0:
                ucb_values[a] = float("inf")  # Try untried actions
            else:
                # UCB bonus
                bonus = self.ucb_coef * np.sqrt(
                    np.log(total_counts) / self.action_counts[a]
                )
                ucb_values[a] += bonus

        return int(np.argmax(ucb_values))

    def _heuristic_values(self, state: State) -> np.ndarray:
        """Heuristic Q-values when no network available."""
        # Use per-alpha Sharpe from state as proxy
        values = np.zeros(self.action_dim)

        # Extract per-alpha Sharpe from state (assuming known structure)
        # State: [regime(4), alpha_features(4*n), market_features(4)]
        n_alphas = (len(state.features) - 8) // 4

        for i in range(min(n_alphas, self.action_dim)):
            start_idx = 4 + i * 4
            if start_idx < len(state.features):
                sharpe = state.features[start_idx]
                values[i] = sharpe

        return values

    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        Update agent from experiences.

        Args:
            experiences: Batch of experiences

        Returns:
            Loss information
        """
        if not TORCH_AVAILABLE or self.q_network is None:
            # Update UCB statistics
            for exp in experiences:
                action = int(exp.action.value)
                reward = exp.reward.value
                self.action_counts[action] += 1
                self.action_rewards[action] += reward
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
            next_q = self.q_network(next_states).max(dim=-1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update UCB statistics
        for exp in experiences:
            action = int(exp.action.value)
            reward = exp.reward.value
            self.action_counts[action] += 1
            self.action_rewards[action] += reward

        self.step_count += 1

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def get_alpha_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each alpha."""
        stats = {}
        for i in range(self.action_dim):
            count = self.action_counts[i]
            total_reward = self.action_rewards[i]
            avg_reward = total_reward / count if count > 0 else 0

            stats[f"alpha_{i}"] = {
                "count": count,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
            }
        return stats

    def save(self, path: str) -> None:
        """Save agent to file."""
        if not TORCH_AVAILABLE or self.q_network is None:
            return

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "action_counts": self.action_counts,
                "action_rewards": self.action_rewards,
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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.action_counts = checkpoint.get("action_counts", np.zeros(self.action_dim))
        self.action_rewards = checkpoint.get(
            "action_rewards", np.zeros(self.action_dim)
        )
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
                "ucb_coef": self.ucb_coef,
                "use_ucb": self.use_ucb,
                "hidden_dims": self.hidden_dims,
            }
        )
        return config
