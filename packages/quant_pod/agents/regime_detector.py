"""Simple RegimeDetectorAgent stub used for tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RegimeDetectorAgent:
    """
    Lightweight regime detector placeholder.

    The production system uses an LLM-backed CrewAI agent. For tests we only
    need a deterministic, import-safe implementation that exposes the expected
    attributes.
    """

    ADX_TRENDING_THRESHOLD = 25
    VOL_LOW_THRESHOLD = 25
    VOL_NORMAL_THRESHOLD = 75

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self.symbols = list(symbols) if symbols else ["SPY"]
        # Placeholder for the underlying CrewAI agent
        self.agent: Dict[str, Any] = {
            "name": "regime_detector",
            "symbols": self.symbols,
        }

    def detect_regime(self, symbol: str, timeframe: str = "daily") -> Dict[str, Any]:
        """
        Return a deterministic regime classification for testing.

        Args:
            symbol: Ticker symbol to classify
            timeframe: Timeframe string (unused in stub)
        """
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "trend_regime": "trending_up",
            "volatility_regime": "normal",
            "confidence": 0.6,
            "adx": self.ADX_TRENDING_THRESHOLD,
            "atr_percentile": 50,
        }
