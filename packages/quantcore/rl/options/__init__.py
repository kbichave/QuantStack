"""
RL module for options trading.

Provides:
- OptionsEnvironment: Environment for options trading (discrete actions)
- DirectionAgent: RL agent that outputs direction + confidence (DQN)
- OptionsTradingEnv: Gymnasium-compatible environment (continuous actions)
- SACOptionsAgent: SAC/PPO/TD3 agent using Stable Baselines3
"""

from quantcore.rl.options.environment import OptionsEnvironment
from quantcore.rl.options.agent import DirectionAgent

# Try to import Gymnasium-based components (require extra dependencies)
try:
    from quantcore.rl.options.gym_env import OptionsTradingEnv, create_trading_env
    from quantcore.rl.options.sac_agent import SACOptionsAgent, train_sac_agent

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    OptionsTradingEnv = None
    create_trading_env = None
    SACOptionsAgent = None
    train_sac_agent = None

__all__ = [
    "OptionsEnvironment",
    "DirectionAgent",
    "OptionsTradingEnv",
    "create_trading_env",
    "SACOptionsAgent",
    "train_sac_agent",
]
