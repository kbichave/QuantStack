"""
Reinforcement Learning layers for trading.

Implements 4 RL layers:
1. Execution RL - Order execution optimization (HRT-style)
2. Position Sizing RL - Dynamic position scaling
3. Alpha Selection RL - Meta-policy for alpha weighting
4. Spread Arbitrage RL - Spread trading optimization
"""

from quantstack.rl.base import (
    Action,
    Experience,
    ReplayBuffer,
    Reward,
    RLAgent,
    RLEnvironment,
    State,
)
from quantstack.rl.orchestrator import RLOrchestrator
from quantstack.rl.training import RLTrainer

__all__ = [
    # Base classes
    "RLAgent",
    "RLEnvironment",
    "State",
    "Action",
    "Reward",
    "Experience",
    "ReplayBuffer",
    # Training
    "RLTrainer",
    # Orchestration
    "RLOrchestrator",
]
