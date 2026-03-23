"""
RL module for options trading.

Provides:
- OptionsEnvironment: Environment for options trading (discrete actions)
- DirectionAgent: RL agent that outputs direction + confidence (DQN)
- OptionsTradingEnv: Gymnasium-compatible environment (continuous actions)
- SACOptionsAgent: SAC/PPO/TD3 agent using Stable Baselines3
"""

from quantstack.rl.options.agent import DirectionAgent
from quantstack.rl.options.environment import OptionsEnvironment

from quantstack.rl.options.gym_env import OptionsTradingEnv, create_trading_env
from quantstack.rl.options.sac_agent import SACOptionsAgent, train_sac_agent

__all__ = [
    "OptionsEnvironment",
    "DirectionAgent",
    "OptionsTradingEnv",
    "create_trading_env",
    "SACOptionsAgent",
    "train_sac_agent",
]
