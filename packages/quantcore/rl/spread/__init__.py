"""
Spread Arbitrage RL - WTI-Brent spread trading.

Uses RL to optimize spread trading based on z-score and regime.
"""

from quantcore.rl.spread.agent import SpreadArbitrageAgent
from quantcore.rl.spread.environment import SpreadEnvironment

# Alias for backwards compatibility
SpreadArbitrageEnvironment = SpreadEnvironment

__all__ = [
    "SpreadArbitrageAgent",
    "SpreadEnvironment",
    "SpreadArbitrageEnvironment",
]
