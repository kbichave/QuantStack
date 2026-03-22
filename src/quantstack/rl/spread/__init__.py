"""
Spread Arbitrage RL - WTI-Brent spread trading.

Uses RL to optimize spread trading based on z-score and regime.
"""

from quantstack.rl.spread.agent import SpreadArbitrageAgent
from quantstack.rl.spread.environment import SpreadEnvironment

# Alias for backwards compatibility
SpreadArbitrageEnvironment = SpreadEnvironment

__all__ = [
    "SpreadArbitrageAgent",
    "SpreadEnvironment",
    "SpreadArbitrageEnvironment",
]
