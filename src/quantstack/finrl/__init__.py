"""
FinRL-based reinforcement learning for trading.

Replaces custom RL implementations with FinRL's DRLAgent/DRLEnsembleAgent
backed by stable-baselines3 algorithms (PPO, A2C, SAC, TD3, DDPG).

Custom Gymnasium environments handle domain-specific use cases:
  - ExecutionEnv: Order execution optimization
  - SizingEnv: Dynamic position sizing
  - AlphaSelectionEnv: Alpha signal weighting
  - PortfolioEnv: ML-based portfolio allocation (uses FinRL built-in)

All functionality exposed via LangChain tools in quantstack.tools.langchain.finrl_tools.
"""

from quantstack.finrl.config import FinRLConfig, get_finrl_config
from quantstack.finrl.model_registry import ModelRegistry

__all__ = [
    "FinRLConfig",
    "get_finrl_config",
    "ModelRegistry",
]
