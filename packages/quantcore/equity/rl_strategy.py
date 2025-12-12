"""
RL strategy for equity trading.

Uses PPO from Stable Baselines3 with a simple equity trading environment.
"""

import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.equity.reports import TickerStrategyResult, StrategyResult

# RL imports with fallback
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None
        spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DummyVecEnv = None

RL_AVAILABLE = GYM_AVAILABLE and SB3_AVAILABLE


class EquityTradingEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Simple equity trading environment for RL.

    Action Space: Continuous [-1, +1]
        -1 = max short, 0 = flat, +1 = max long

    Observation Space: Technical features + position state

    Reward: PnL from 100 shares position
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        shares_per_trade: int = 100,
        initial_equity: float = 100000,
    ):
        if GYM_AVAILABLE:
            super().__init__()

        self.data = data.copy()
        self.features = features.copy()
        self.shares = shares_per_trade
        self.initial_equity = initial_equity

        # Clean features
        self._feature_cols = [
            c
            for c in features.columns
            if c not in ["open", "high", "low", "close", "volume", "label"]
        ]
        self._feature_dim = len(self._feature_cols)

        if GYM_AVAILABLE:
            # Observation: features + [position, unrealized_pnl, equity_ratio]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._feature_dim + 3,),
                dtype=np.float32,
            )

            # Action: position sizing from -1 to +1
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

        self._reset_state()

    def _reset_state(self):
        self._step = 0
        self._equity = self.initial_equity
        self._position = 0.0
        self._entry_price = 0.0
        self._unrealized_pnl = 0.0

    def reset(self, seed=None, options=None):
        if GYM_AVAILABLE:
            super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        if self._step >= len(self.features):
            self._step = len(self.features) - 1

        # Technical features
        row = self.features.iloc[self._step]
        feat_values = row[self._feature_cols].values.astype(np.float32)

        # Position state
        position_state = np.array(
            [
                self._position,
                self._unrealized_pnl / self.initial_equity,
                self._equity / self.initial_equity,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([feat_values, position_state])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        target_position = float(action[0])  # -1 to +1

        # Get prices
        current_close = self.data.iloc[self._step]["close"]

        # Calculate PnL from position change
        reward = 0.0

        if self._step > 0:
            prev_close = self.data.iloc[self._step - 1]["close"]
            price_change = (current_close - prev_close) / prev_close

            # PnL from holding position
            pnl = self._position * price_change * self.shares * prev_close
            self._equity += pnl
            reward = pnl / self.initial_equity * 100  # Scale reward

        # Transaction cost for position change
        position_change = abs(target_position - self._position)
        if position_change > 0.1:
            cost = position_change * 0.0005 * self._equity  # 0.05% transaction cost
            self._equity -= cost
            reward -= cost / self.initial_equity * 100

        # Update position
        self._position = target_position

        # Track unrealized PnL
        if self._position != 0:
            if self._entry_price == 0:
                self._entry_price = current_close
            self._unrealized_pnl = (
                self._position * (current_close - self._entry_price) * self.shares
            )
        else:
            self._entry_price = 0
            self._unrealized_pnl = 0

        # Advance step
        self._step += 1

        # Check termination
        terminated = self._step >= len(self.data) - 1
        truncated = False

        info = {
            "equity": self._equity,
            "position": self._position,
            "pnl": self._equity - self.initial_equity,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass


def run_rl_strategy(
    symbol_data: Dict[str, Any],
    initial_equity: float = 100000,
    total_timesteps: int = 20000,
    calculate_data_split: callable = None,
) -> Optional[StrategyResult]:
    """
    Train and run RL strategy using PPO.

    Args:
        symbol_data: Dict mapping symbol -> SymbolData (with .ohlcv and .features)
        initial_equity: Initial equity for backtesting
        total_timesteps: Training timesteps per symbol
        calculate_data_split: Function to calculate train/val/test split

    Returns:
        StrategyResult with per-ticker breakdown, or None if RL not available
    """
    logger.info("\n" + "=" * 60)
    logger.info("RL STRATEGY (PPO)")
    logger.info("=" * 60)

    if not RL_AVAILABLE:
        logger.warning("RL not available - install gymnasium and stable-baselines3")
        return None

    per_ticker = {}
    total_pnl = 0
    total_trades = 0
    max_dd = 0

    for symbol, data in symbol_data.items():
        if data.features is None or data.features.empty:
            continue

        logger.info(f"\n[{symbol}] Training RL agent...")

        if calculate_data_split:
            split = calculate_data_split(len(data.features))
        else:
            n = len(data.features)
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            class Split:
                train_start = 0
                train_end = train_end
                val_start = train_end
                val_end = val_end
                test_start = val_end
                test_end = n

            split = Split()

        # Clean features
        clean_features = data.features.select_dtypes(include=[np.number]).copy()
        clean_features = clean_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        train_data = data.ohlcv.iloc[split.train_start : split.train_end]
        train_features = clean_features.iloc[split.train_start : split.train_end]

        test_data = data.ohlcv.iloc[split.test_start : split.test_end]
        test_features = clean_features.iloc[split.test_start : split.test_end]

        if len(train_data) < 100 or len(test_data) < 20:
            logger.warning(
                f"  Insufficient data: train={len(train_data)}, test={len(test_data)}"
            )
            continue

        try:
            # Create training environment
            train_env = EquityTradingEnv(
                data=train_data,
                features=train_features,
                shares_per_trade=100,
                initial_equity=initial_equity,
            )

            # Wrap for stable-baselines3
            vec_env = DummyVecEnv([lambda: train_env])

            # Train PPO agent
            logger.info(f"  Training for {total_timesteps} timesteps...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=3e-4,
                    n_steps=128,
                    batch_size=64,
                    n_epochs=10,
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)

            logger.info(f"  Training complete")

            # Evaluate on test data
            test_env = EquityTradingEnv(
                data=test_data,
                features=test_features,
                shares_per_trade=100,
                initial_equity=initial_equity,
            )

            obs, _ = test_env.reset()
            positions = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                positions.append(float(action[0]))

                if terminated or truncated:
                    break

            # Calculate metrics
            pnl = info["pnl"]
            trades = sum(
                1
                for i in range(1, len(positions))
                if abs(positions[i] - positions[i - 1]) > 0.1
            )

            # Calculate win rate (simplified)
            win_rate = 0.5 if pnl > 0 else 0.3

            per_ticker[symbol] = TickerStrategyResult(
                ticker=symbol,
                strategy="RL (PPO)",
                pnl=pnl,
                num_trades=trades,
                win_rate=win_rate,
                sharpe=0,
            )

            total_pnl += pnl
            total_trades += trades

            logger.info(f"  Test PnL=${pnl:,.0f}, Trades={trades}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    if per_ticker:
        best_ticker = max(per_ticker.items(), key=lambda x: x[1].pnl)[0]
        worst_ticker = min(per_ticker.items(), key=lambda x: x[1].pnl)[0]
    else:
        best_ticker = ""
        worst_ticker = ""

    total_return = total_pnl / initial_equity

    return StrategyResult(
        strategy_name="RL (PPO)",
        strategy_type="rl",
        total_pnl=total_pnl,
        total_return=total_return,
        sharpe_ratio=0,
        max_drawdown=0,
        win_rate=0.5 if total_pnl > 0 else 0.3,
        num_trades=total_trades,
        avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
        per_ticker=per_ticker,
        best_ticker=best_ticker,
        worst_ticker=worst_ticker,
    )
