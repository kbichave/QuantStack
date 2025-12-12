"""Agent entrypoints and deprecated wrappers for QuantPod."""

from __future__ import annotations

import warnings

from quant_pod.agents.regime_detector import RegimeDetectorAgent
from quant_pod.crews.schemas import AnalysisNote, DailyBrief, TradeDecision


class SuperTrader:
    """Deprecated placeholder kept for backward compatibility in tests."""

    def __init__(self) -> None:
        warnings.warn(
            "SuperTrader is deprecated; use TradingCrew.super_trader instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def decide(self, *args, **kwargs):
        raise NotImplementedError(
            "SuperTrader.decide is deprecated; use TradingCrew.super_trader()."
        )


def create_all_pods():
    """Deprecated placeholder for legacy API."""
    raise NotImplementedError(
        "create_all_pods is deprecated; use TradingCrew().crew() instead."
    )


def get_super_trader():
    """Deprecated placeholder for legacy API."""
    raise NotImplementedError(
        "get_super_trader is deprecated; use TradingCrew.super_trader instead."
    )


__all__ = [
    "RegimeDetectorAgent",
    "SuperTrader",
    "create_all_pods",
    "get_super_trader",
    # Schema re-exports for convenience
    "TradeDecision",
    "DailyBrief",
    "AnalysisNote",
]
