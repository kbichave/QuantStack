"""
Gymnasium-compatible environment for options trading.

Wraps the options trading logic in a Gymnasium interface
for use with Stable Baselines3 algorithms (SAC, PPO, TD3).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger

# Suppress gym deprecation warning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    try:
        # Fallback to gym if gymnasium not available
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
        logger.info("Using legacy gym instead of gymnasium")
    except ImportError:
        GYM_AVAILABLE = False
        logger.warning(
            "Neither gymnasium nor gym available. Install with: pip install gymnasium"
        )


class OptionsTradingEnv(gym.Env):
    """
    Gymnasium environment for options trading with continuous action space.

    Observation Space:
        - Technical features (normalized)
        - Position state (delta, gamma, theta, unrealized PnL)
        - Regime indicators

    Action Space (Continuous):
        - action[0]: Position direction and size (-1 to +1)
                     -1 = max short, 0 = flat, +1 = max long

    Reward:
        - PnL change from price movement
        - Transaction costs
        - Risk penalties (position size, gamma near events)
    """

    metadata = {"render_modes": ["human"]}

    # Reward function parameters
    LAMBDA_POSITION = 0.001  # Penalty for holding
    LAMBDA_GAMMA = 0.01  # Penalty for gamma near earnings
    LAMBDA_TAIL = 0.05  # Penalty for exceeding limits

    # Position limits
    MAX_DELTA = 100
    MAX_GAMMA = 50

    # Transaction costs
    COMMISSION_PER_CONTRACT = 0.65
    SPREAD_PCT = 0.02  # 2% of option price

    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        initial_equity: float = 100000,
        max_holding_days: int = 30,
        window_size: int = 1,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize options trading environment.

        Args:
            data: OHLCV DataFrame for underlying
            features: Feature DataFrame (aligned with data)
            initial_equity: Starting equity
            max_holding_days: Max days to hold position
            window_size: Number of past observations to include
            render_mode: Rendering mode
        """
        super().__init__()

        if not GYM_AVAILABLE:
            raise ImportError(
                "Gymnasium is required. Install with: pip install gymnasium"
            )

        self.data = data.copy()
        self.features = features.copy()
        self.initial_equity = initial_equity
        self.max_holding_days = max_holding_days
        self.window_size = window_size
        self.render_mode = render_mode

        # Calculate observation dimension
        self._feature_dim = len(self.features.columns)
        self._position_state_dim = 4  # delta, gamma, theta, unrealized_pnl
        self._obs_dim = self._feature_dim + self._position_state_dim

        # Define observation space (normalized features + position state)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Continuous action space: position direction and size
        # -1 = max short, 0 = flat, +1 = max long
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # State tracking
        self._current_step = 0
        self._equity = initial_equity
        self._position_delta = 0.0
        self._position_gamma = 0.0
        self._position_theta = 0.0
        self._entry_price = 0.0
        self._entry_step = 0
        self._unrealized_pnl = 0.0

        # Episode tracking
        self._episode_pnl = 0.0
        self._episode_trades = 0
        self._episode_rewards = []

        # Precompute normalized features
        self._normalized_features = self._normalize_features()

    def _normalize_features(self) -> np.ndarray:
        """Normalize features for neural network input."""
        features_array = self.features.values.astype(np.float32)

        # Replace NaN with 0
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize (mean 0, std 1) per feature
        mean = np.nanmean(features_array, axis=0, keepdims=True)
        std = np.nanstd(features_array, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        normalized = (features_array - mean) / std

        # Clip extreme values
        normalized = np.clip(normalized, -10, 10)

        return normalized.astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self._current_step = 0
        self._equity = self.initial_equity
        self._position_delta = 0.0
        self._position_gamma = 0.0
        self._position_theta = 0.0
        self._entry_price = 0.0
        self._entry_step = 0
        self._unrealized_pnl = 0.0
        self._episode_pnl = 0.0
        self._episode_trades = 0
        self._episode_rewards = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Continuous action in [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Extract action value
        action_value = float(np.clip(action[0], -1.0, 1.0))

        # Convert action to target delta
        target_delta = action_value * self.MAX_DELTA

        # Calculate reward components
        pnl_change = self._calculate_pnl_change()
        transaction_cost = self._calculate_transaction_cost(target_delta)
        position_penalty = self._calculate_position_penalty()
        gamma_penalty = self._calculate_gamma_penalty()
        tail_penalty = self._calculate_tail_penalty()

        # Total reward
        reward = float(
            pnl_change
            - transaction_cost
            - position_penalty
            - gamma_penalty
            - tail_penalty
        )

        # Update position
        self._update_position(target_delta)

        # Advance step
        self._current_step += 1

        # Track episode
        self._episode_rewards.append(reward)

        # Check termination
        terminated = self._current_step >= len(self.data) - 1
        truncated = False

        # Check forced exit (max holding)
        if self._should_force_exit():
            self._close_position()

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self._current_step >= len(self._normalized_features):
            features = np.zeros(self._feature_dim, dtype=np.float32)
        else:
            features = self._normalized_features[self._current_step]

        # Position state (normalized)
        position_state = np.array(
            [
                self._position_delta / self.MAX_DELTA,
                self._position_gamma / self.MAX_GAMMA,
                self._position_theta / 100,
                self._unrealized_pnl / 10000,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([features, position_state])

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get current info dict."""
        return {
            "step": self._current_step,
            "equity": self._equity,
            "position_delta": self._position_delta,
            "unrealized_pnl": self._unrealized_pnl,
            "episode_pnl": self._episode_pnl,
            "episode_trades": self._episode_trades,
        }

    def _calculate_pnl_change(self) -> float:
        """Calculate PnL change from price movement."""
        if self._current_step < 1 or self._position_delta == 0:
            return 0.0

        price_change = (
            self.data["close"].iloc[self._current_step]
            - self.data["close"].iloc[self._current_step - 1]
        )

        # PnL from delta
        pnl = self._position_delta * price_change

        # Add theta decay (negative for long options)
        pnl += self._position_theta

        self._unrealized_pnl += pnl
        self._episode_pnl += pnl

        return pnl

    def _calculate_transaction_cost(self, target_delta: float) -> float:
        """Calculate transaction costs for position change."""
        delta_change = abs(target_delta - self._position_delta)

        if delta_change < 5:  # Threshold to avoid tiny trades
            return 0.0

        # Estimate cost based on position change
        position_value = delta_change * self.data["close"].iloc[self._current_step]
        cost = position_value * self.SPREAD_PCT + self.COMMISSION_PER_CONTRACT

        return cost

    def _calculate_position_penalty(self) -> float:
        """Calculate penalty for holding position."""
        realized_vol = self._get_realized_vol()
        return self.LAMBDA_POSITION * abs(self._position_delta) * realized_vol

    def _calculate_gamma_penalty(self) -> float:
        """Calculate penalty for gamma exposure near events."""
        if self._current_step >= len(self.features):
            return 0.0

        row = self.features.iloc[self._current_step]
        days_to_earnings = row.get("earn_days_to", 999)

        if pd.notna(days_to_earnings) and days_to_earnings < 5:
            return self.LAMBDA_GAMMA * abs(self._position_gamma)
        return 0.0

    def _calculate_tail_penalty(self) -> float:
        """Calculate penalty for exceeding risk limits."""
        penalty = 0.0

        if abs(self._position_delta) > self.MAX_DELTA:
            excess = abs(self._position_delta) - self.MAX_DELTA
            penalty += self.LAMBDA_TAIL * excess

        if abs(self._position_gamma) > self.MAX_GAMMA:
            excess = abs(self._position_gamma) - self.MAX_GAMMA
            penalty += self.LAMBDA_TAIL * excess

        return penalty

    def _update_position(self, target_delta: float) -> None:
        """Update position based on target delta."""
        old_delta = self._position_delta

        # Smooth transition to target
        self._position_delta = 0.7 * self._position_delta + 0.3 * target_delta

        # Clip to limits
        self._position_delta = np.clip(
            self._position_delta, -self.MAX_DELTA, self.MAX_DELTA
        )

        # Update gamma/theta based on position
        self._position_gamma = abs(self._position_delta) * 0.1
        self._position_theta = -abs(self._position_delta) * 0.5

        # Track new entries
        if abs(old_delta) < 5 and abs(self._position_delta) >= 5:
            self._entry_step = self._current_step
            self._entry_price = self.data["close"].iloc[self._current_step]
            self._episode_trades += 1

    def _should_force_exit(self) -> bool:
        """Check if position should be force-closed."""
        if abs(self._position_delta) < 5:
            return False

        # Check max holding period
        if self._current_step - self._entry_step >= self.max_holding_days:
            return True

        return False

    def _close_position(self) -> None:
        """Close current position."""
        # Realize PnL
        self._equity += self._unrealized_pnl

        # Reset position
        self._position_delta = 0.0
        self._position_gamma = 0.0
        self._position_theta = 0.0
        self._entry_step = 0
        self._entry_price = 0.0
        self._unrealized_pnl = 0.0

    def _get_realized_vol(self) -> float:
        """Get realized volatility."""
        if self._current_step < 20:
            return 0.2  # Default

        returns = (
            self.data["close"]
            .pct_change()
            .iloc[self._current_step - 20 : self._current_step]
        )
        return float(returns.std() * np.sqrt(252))

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode == "human":
            print(
                f"Step: {self._current_step}, Equity: ${self._equity:.2f}, "
                f"Delta: {self._position_delta:.1f}, PnL: ${self._episode_pnl:.2f}"
            )


def create_trading_env(
    data: pd.DataFrame,
    features: pd.DataFrame,
    initial_equity: float = 100000,
    **kwargs,
) -> OptionsTradingEnv:
    """
    Factory function to create trading environment.

    Args:
        data: OHLCV DataFrame
        features: Feature DataFrame
        initial_equity: Starting equity
        **kwargs: Additional arguments for environment

    Returns:
        OptionsTradingEnv instance
    """
    return OptionsTradingEnv(
        data=data,
        features=features,
        initial_equity=initial_equity,
        **kwargs,
    )
