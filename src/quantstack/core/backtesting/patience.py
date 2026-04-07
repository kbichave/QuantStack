"""3-window backtest patience protocol (AR-9).

Enforces multi-window validation to prevent premature hypothesis rejection.
A hypothesis must pass validation in 3 distinct market windows before
full acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatienceConfig:
    """Configuration for multi-window validation."""

    full_start: str = "2020-01-01"
    recent_months: int = 12
    stressed_start: str = "2020-03-01"
    stressed_end: str = "2020-06-30"


@dataclass
class WindowResult:
    """Result from a single validation window."""

    window_name: str  # "full", "recent", "stressed"
    passed: bool
    sharpe: float
    max_drawdown: float
    ic: float


def evaluate_patience(results: list[WindowResult]) -> str:
    """Determine hypothesis status from multi-window results.

    Returns
    -------
    'accepted' if all 3 windows pass.
    'provisional' if exactly 2 of 3 pass (lower confidence, smaller position sizing).
    'rejected' if fewer than 2 pass.
    """
    passed_count = sum(1 for r in results if r.passed)
    if passed_count >= 3:
        return "accepted"
    if passed_count == 2:
        return "provisional"
    return "rejected"
