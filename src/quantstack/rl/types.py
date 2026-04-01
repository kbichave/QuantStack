"""
Shared RL domain types.

Kept separate from base.py (which holds the abstract RLEnvironment/Agent
machinery) to avoid a sizing.environment ↔ data_bridge circular dependency:
both modules need TradingSignal but neither should import from the other.
"""

from dataclasses import dataclass


@dataclass
class TradingSignal:
    """Trading signal passed from alpha layer to the sizing RL environment."""

    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0-1
    expected_return: float
    alpha_name: str
