"""
Position Sizing RL Agent.

PPO-based agent for dynamic position sizing.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from loguru import logger

from quantcore.rl.base import (
    RLAgent,
    State,
    Action,
    Experience,
    ActorCritic,
    compute_gae,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal


class SizingRLAgent(RLAgent):
    """
    PPO agent for position sizing.

    Uses Proximal Policy Optimization with:
    - Continuous action space (scale factor 0-1)
    - Clipped objective for stable training
    - Generalized Advantage Estimation (GAE)

    Output:
    - Position scale factor in [0, 1]
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize sizing agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension (1 for scale factor)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            n_epochs: PPO epochs per update
            batch_size: Mini-batch size
            device: Device to use
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma, device)

        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims

        self._build_networks()

        # Trajectory buffer
        self.trajectory: List[Dict] = []

    def _build_networks(self) -> None:
        """Build neural networks."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using fallback agent")
            self.actor_critic = None
            self.optimizer = None
            return

        self.actor_critic = ActorCritic(
            self.state_dim,
            self.action_dim,
            self.hidden_dims,
            continuous=True,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select action using policy.

        Args:
            state: Current state
            explore: Whether to sample (vs deterministic)

        Returns:
            Selected action
        """
        if TORCH_AVAILABLE and self.actor_critic is not None:
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
                action, log_prob = self.actor_critic.get_action(
                    state_tensor,
                    deterministic=not explore,
                )

                # Clamp to [0, 1]
                action = torch.clamp(action, 0, 1)
                action_np = action.cpu().numpy().flatten()

                # Store for trajectory
                if explore and self.training:
                    (policy, value) = self.actor_critic(state_tensor)
                    self._store_transition(
                        state=state,
                        action=action_np,
                        log_prob=log_prob.cpu().numpy(),
                        value=value.cpu().numpy().flatten()[0],
                    )
        else:
            # Fallback: heuristic sizing
            action_np = self._heuristic_action(state)

        return Action(value=action_np, action_type="continuous")

    def _heuristic_action(self, state: State) -> np.ndarray:
        """Heuristic action when no neural network available."""
        confidence = state.features[0]
        direction = state.features[1]
        volatility = state.features[2]
        drawdown = state.features[3]

        # Base scale on confidence
        scale = confidence

        # Reduce in high volatility
        if volatility > 0.7:
            scale *= 0.5

        # Reduce in drawdown
        if drawdown > 0.5:
            scale *= 0.5

        # No position if neutral signal
        if abs(direction) < 0.5:
            scale = 0.0

        return np.array([np.clip(scale, 0, 1)])

    def _store_transition(
        self,
        state: State,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: float,
    ) -> None:
        """Store transition in trajectory buffer."""
        self.trajectory.append(
            {
                "state": state.features,
                "action": action,
                "log_prob": log_prob,
                "value": value,
            }
        )

    def complete_trajectory(
        self,
        rewards: List[float],
        dones: List[bool],
        final_value: float,
    ) -> None:
        """
        Complete trajectory with rewards.

        Args:
            rewards: List of rewards
            dones: List of done flags
            final_value: Value estimate of final state
        """
        if len(self.trajectory) == 0:
            return

        values = [t["value"] for t in self.trajectory]

        # Compute advantages and returns
        advantages, returns = compute_gae(
            rewards,
            values,
            final_value,
            self.gamma,
            self.gae_lambda,
        )

        # Add to trajectory
        for i, t in enumerate(self.trajectory):
            if i < len(advantages):
                t["advantage"] = advantages[i]
                t["return"] = returns[i]
                t["reward"] = rewards[i]
                t["done"] = dones[i]

    def update(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        Update agent using PPO.

        Args:
            experiences: Not used directly (uses trajectory buffer)

        Returns:
            Loss information
        """
        if not TORCH_AVAILABLE or self.actor_critic is None:
            return {"loss": 0.0}

        if len(self.trajectory) == 0:
            return {"loss": 0.0}

        # Filter complete transitions
        complete_traj = [t for t in self.trajectory if "advantage" in t]

        if len(complete_traj) < self.batch_size:
            return {"loss": 0.0}

        # Prepare data
        states = torch.FloatTensor(np.array([t["state"] for t in complete_traj])).to(
            self.device
        )

        actions = torch.FloatTensor(np.array([t["action"] for t in complete_traj])).to(
            self.device
        )

        old_log_probs = torch.FloatTensor(
            np.array([t["log_prob"] for t in complete_traj])
        ).to(self.device)

        advantages = torch.FloatTensor(
            np.array([t["advantage"] for t in complete_traj])
        ).to(self.device)

        returns = torch.FloatTensor(np.array([t["return"] for t in complete_traj])).to(
            self.device
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0

        n_samples = len(complete_traj)
        indices = np.arange(n_samples)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                (mean, std), values = self.actor_critic(batch_states)

                # Action distribution
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().mean()

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze())
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.item()

        # Clear trajectory
        self.trajectory = []

        n_updates = self.n_epochs * (n_samples // self.batch_size + 1)

        return {
            "loss": total_loss / n_updates,
            "policy_loss": policy_loss_sum / n_updates,
            "value_loss": value_loss_sum / n_updates,
            "entropy": entropy_sum / n_updates,
        }

    def save(self, path: str) -> None:
        """Save agent to file."""
        if not TORCH_AVAILABLE or self.actor_critic is None:
            return

        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "config": self.get_config(),
            },
            path,
        )
        logger.info(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent from file."""
        if not TORCH_AVAILABLE or self.actor_critic is None:
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
        logger.info(f"Agent loaded from {path}")

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        config = super().get_config()
        config.update(
            {
                "gae_lambda": self.gae_lambda,
                "clip_ratio": self.clip_ratio,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "hidden_dims": self.hidden_dims,
            }
        )
        return config
