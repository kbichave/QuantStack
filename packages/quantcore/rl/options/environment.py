"""
RL Environment for options trading.

Implements the hybrid architecture:
- RL agent outputs direction + confidence
- Environment handles market simulation
- Explicit reward function with risk penalties
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.rl.base import RLEnvironment, State, Action, Reward


class OptionsAction(Enum):
    """Discrete action space for options trading."""

    STRONG_LONG = 0  # High confidence long
    WEAK_LONG = 1  # Low confidence long
    FLAT = 2  # No position
    WEAK_SHORT = 3  # Low confidence short
    STRONG_SHORT = 4  # High confidence short


@dataclass
class OptionsState:
    """State representation for options RL."""

    # Underlying features
    features: np.ndarray

    # Regime context (one-hot)
    trend_regime: np.ndarray  # [bull, bear, sideways]
    vol_regime: np.ndarray  # [low, medium, high]

    # Options features
    iv_rank: float
    iv_percentile: float
    days_to_earnings: int

    # Position state
    current_delta: float
    current_gamma: float
    current_theta: float
    unrealized_pnl: float

    def to_vector(self) -> np.ndarray:
        """Convert to flat feature vector."""
        return np.concatenate(
            [
                self.features,
                self.trend_regime,
                self.vol_regime,
                np.array(
                    [
                        self.iv_rank / 100,
                        self.iv_percentile / 100,
                        min(self.days_to_earnings, 90) / 90,
                        self.current_delta / 100,
                        self.current_gamma / 10,
                        self.current_theta / 100,
                        self.unrealized_pnl / 10000,
                    ]
                ),
            ]
        )


class OptionsEnvironment(RLEnvironment):
    """
    RL Environment for options trading.

    Features:
    - Event-driven simulation
    - Explicit reward function with risk penalties
    - Position tracking
    - Transaction cost modeling
    """

    # Reward function parameters - tuned for trend reversal trading
    LAMBDA_POSITION = 0.0005  # Lower penalty for holding (encourage longer trades)
    LAMBDA_GAMMA = 0.01  # Penalty for gamma exposure
    LAMBDA_TAIL = 0.05  # Penalty for exceeding limits
    LAMBDA_REVERSAL = 0.5  # Bonus for catching reversals
    LAMBDA_MOMENTUM = 0.1  # Bonus for momentum alignment

    # Position limits
    MAX_DELTA = 100
    MAX_GAMMA = 50

    # Transaction costs
    COMMISSION_PER_CONTRACT = 0.65
    SPREAD_PCT = 0.05  # 5% of option price

    # Trend reversal thresholds
    REVERSAL_ATR_MULT = 1.5  # ATR multiple to count as reversal catch
    MOMENTUM_LOOKBACK = 5  # Bars to check momentum direction

    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        initial_equity: float = 100000,
        max_holding_days: int = 30,
        forced_exit_dte: int = 5,
    ):
        """
        Initialize options environment.

        Args:
            data: OHLCV DataFrame for underlying
            features: Feature DataFrame
            initial_equity: Starting equity
            max_holding_days: Max days to hold position
            forced_exit_dte: Force exit N days before expiry
        """
        self.data = data
        self.features = features
        self.initial_equity = initial_equity
        self.max_holding_days = max_holding_days
        self.forced_exit_dte = forced_exit_dte

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

    def reset(self) -> State:
        """Reset environment to initial state."""
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

        return self._get_current_state()

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get action value
        if isinstance(action.value, int):
            action_enum = OptionsAction(action.value)
        else:
            action_enum = OptionsAction.FLAT

        # Convert action to direction and confidence
        direction, confidence = self._action_to_direction(action_enum)

        # Calculate reward components
        pnl_change = self._calculate_pnl_change()
        transaction_cost = self._calculate_transaction_cost(direction, confidence)
        position_penalty = self._calculate_position_penalty()
        gamma_penalty = self._calculate_gamma_penalty()
        tail_penalty = self._calculate_tail_penalty()

        # Trend reversal bonuses
        reversal_bonus = self._calculate_reversal_bonus(direction)
        momentum_bonus = self._calculate_momentum_bonus(direction)

        # Total reward - tuned for trend reversal trading
        reward_value = (
            pnl_change
            - transaction_cost
            - position_penalty
            - gamma_penalty
            - tail_penalty
            + reversal_bonus
            + momentum_bonus
        )

        # Update position
        self._update_position(direction, confidence)

        # Advance step
        self._current_step += 1

        # Check termination
        done = self._current_step >= len(self.data) - 1

        # Check forced exit (max holding or near expiry)
        if self._should_force_exit():
            self._close_position()

        # Get next state
        next_state = self._get_current_state()

        # Build reward
        reward = Reward(
            value=reward_value,
            components={
                "pnl_change": pnl_change,
                "transaction_cost": transaction_cost,
                "position_penalty": position_penalty,
                "gamma_penalty": gamma_penalty,
                "tail_penalty": tail_penalty,
                "reversal_bonus": reversal_bonus,
                "momentum_bonus": momentum_bonus,
            },
        )

        info = {
            "equity": self._equity,
            "position_delta": self._position_delta,
            "episode_pnl": self._episode_pnl,
            "episode_trades": self._episode_trades,
        }

        return next_state, reward, done, info

    def _get_current_state(self) -> State:
        """Get current state."""
        if self._current_step >= len(self.features):
            # Return zero state if past end
            return State(
                features=np.zeros(self.get_state_dim()),
                timestamp=None,
            )

        # Get features for current step
        row = self.features.iloc[self._current_step]
        features = row.values.astype(np.float32)

        # Handle NaN
        features = np.nan_to_num(features, nan=0.0)

        # Add position state
        position_state = np.array(
            [
                self._position_delta / 100,
                self._position_gamma / 10,
                self._position_theta / 100,
                self._unrealized_pnl / 10000,
            ],
            dtype=np.float32,
        )

        full_features = np.concatenate([features, position_state])

        return State(
            features=full_features,
            timestamp=(
                self.data.index[self._current_step]
                if self._current_step < len(self.data)
                else None
            ),
            metadata={
                "step": self._current_step,
                "equity": self._equity,
                "position_delta": self._position_delta,
            },
        )

    def _action_to_direction(self, action: OptionsAction) -> Tuple[int, float]:
        """
        Convert action to direction and confidence.

        Returns:
            Tuple of (direction: -1/0/+1, confidence: 0-1)
        """
        mapping = {
            OptionsAction.STRONG_LONG: (1, 0.8),
            OptionsAction.WEAK_LONG: (1, 0.4),
            OptionsAction.FLAT: (0, 0.0),
            OptionsAction.WEAK_SHORT: (-1, 0.4),
            OptionsAction.STRONG_SHORT: (-1, 0.8),
        }
        return mapping.get(action, (0, 0.0))

    def _calculate_pnl_change(self) -> float:
        """Calculate PnL change from price movement."""
        if self._current_step < 1 or self._position_delta == 0:
            return 0.0

        price_change = (
            self.data["close"].iloc[self._current_step]
            - self.data["close"].iloc[self._current_step - 1]
        )

        # Approximate PnL from delta
        pnl = self._position_delta * price_change

        # Add theta decay
        pnl += self._position_theta

        self._unrealized_pnl += pnl
        self._episode_pnl += pnl

        return pnl

    def _calculate_transaction_cost(self, direction: int, confidence: float) -> float:
        """Calculate transaction costs for position change."""
        # If no change, no cost
        current_direction = (
            1 if self._position_delta > 0 else (-1 if self._position_delta < 0 else 0)
        )

        if direction == current_direction:
            return 0.0

        # Estimate cost based on position size and spread
        position_value = (
            abs(self._position_delta) * self.data["close"].iloc[self._current_step]
        )
        cost = position_value * self.SPREAD_PCT + self.COMMISSION_PER_CONTRACT

        return cost

    def _calculate_position_penalty(self) -> float:
        """Calculate penalty for holding position."""
        realized_vol = self._get_realized_vol()
        return self.LAMBDA_POSITION * abs(self._position_delta) * realized_vol

    def _calculate_gamma_penalty(self) -> float:
        """Calculate penalty for gamma exposure near events."""
        # Get days to earnings from features if available
        days_to_earnings = 999
        if self._current_step < len(self.features):
            row = self.features.iloc[self._current_step]
            if "days_to_earnings" in row.index:
                days_to_earnings = row["days_to_earnings"]

        if days_to_earnings < 5:
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

    def _calculate_reversal_bonus(self, direction: int) -> float:
        """
        Calculate bonus for catching a trend reversal.

        Rewards positions that are correctly aligned when significant
        price moves occur (catching the reversal).
        """
        if self._position_delta == 0 or self._current_step < 2:
            return 0.0

        # Check if we caught a significant move
        if self._entry_step == 0 or self._entry_price == 0:
            return 0.0

        current_price = self.data["close"].iloc[self._current_step]
        price_change = current_price - self._entry_price

        # Get ATR for threshold
        atr = self._get_atr_at_step(self._entry_step)
        if atr <= 0:
            return 0.0

        # Reward if position is in profit and move is significant
        position_direction = 1 if self._position_delta > 0 else -1
        aligned_move = price_change * position_direction

        if aligned_move > atr * self.REVERSAL_ATR_MULT:
            # We caught a reversal! Scale bonus by move size
            move_size = aligned_move / atr
            return self.LAMBDA_REVERSAL * min(move_size, 3.0)  # Cap at 3x ATR

        return 0.0

    def _calculate_momentum_bonus(self, direction: int) -> float:
        """
        Calculate bonus for momentum alignment.

        Rewards positions that are aligned with recent price momentum,
        encouraging trend-following within reversals.
        """
        if direction == 0 or self._current_step < self.MOMENTUM_LOOKBACK:
            return 0.0

        # Calculate recent momentum
        recent_prices = self.data["close"].iloc[
            self._current_step - self.MOMENTUM_LOOKBACK : self._current_step + 1
        ]
        momentum = recent_prices.iloc[-1] - recent_prices.iloc[0]

        # Check alignment
        momentum_direction = 1 if momentum > 0 else (-1 if momentum < 0 else 0)

        if direction == momentum_direction and momentum_direction != 0:
            # Position aligned with momentum - small bonus
            return self.LAMBDA_MOMENTUM * abs(momentum) / recent_prices.iloc[0] * 100
        elif direction == -momentum_direction and momentum_direction != 0:
            # Counter-momentum (potential reversal entry) - smaller bonus
            return (
                self.LAMBDA_MOMENTUM * 0.5 * abs(momentum) / recent_prices.iloc[0] * 100
            )

        return 0.0

    def _get_atr_at_step(self, step: int, period: int = 14) -> float:
        """Get ATR at a specific step."""
        if step < period:
            return self.data["close"].iloc[: step + 1].std() if step > 0 else 0.01

        high = self.data["high"].iloc[step - period : step + 1]
        low = self.data["low"].iloc[step - period : step + 1]
        close = self.data["close"].iloc[step - period : step + 1]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.mean()

    def _update_position(self, direction: int, confidence: float) -> None:
        """Update position based on action."""
        target_delta = direction * confidence * 50  # Scale to reasonable delta

        # Smooth transition
        self._position_delta = 0.8 * self._position_delta + 0.2 * target_delta

        # Update gamma/theta based on position
        self._position_gamma = abs(self._position_delta) * 0.1
        self._position_theta = -abs(self._position_delta) * 0.5

        if direction != 0 and self._entry_step == 0:
            self._entry_step = self._current_step
            self._entry_price = self.data["close"].iloc[self._current_step]
            self._episode_trades += 1

    def _should_force_exit(self) -> bool:
        """Check if position should be force-closed."""
        if self._position_delta == 0:
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
        return returns.std() * np.sqrt(252)

    def get_state_dim(self) -> int:
        """Get state dimension."""
        if len(self.features) > 0:
            return len(self.features.columns) + 4  # +4 for position state
        return 100  # Default

    def get_action_dim(self) -> int:
        """Get action dimension."""
        return len(OptionsAction)
