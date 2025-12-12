"""Expectancy calculations for trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from quant_pod.knowledge.models import StructureType, TradeRecord
from quant_pod.knowledge.store import KnowledgeStore


@dataclass
class ExpectancyResult:
    """Simple container for expectancy metrics."""

    sample_size: int
    win_rate: float
    expectancy: float
    avg_win: float
    avg_loss: float


class ExpectancyEngine:
    """Compute expectancy metrics using trades stored in KnowledgeStore."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def _get_trades(
        self, structure: Optional[StructureType] = None
    ) -> List[TradeRecord]:
        trades = [t for t in self.store.get_trades(limit=500) if t.pnl is not None]
        if structure:
            trades = [t for t in trades if t.structure_type == structure]
        return trades

    def calculate_expectancy(
        self, structure: Optional[StructureType] = None
    ) -> ExpectancyResult:
        trades = self._get_trades(structure)
        if not trades:
            return ExpectancyResult(0, 0.0, 0.0, 0.0, 0.0)

        wins = [t.pnl for t in trades if t.pnl is not None and t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl is not None and t.pnl <= 0]

        win_rate = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        return ExpectancyResult(
            sample_size=len(trades),
            win_rate=win_rate,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

    def get_kelly_fraction(self, kelly_mode: str = "standard") -> float:
        """Compute a simple Kelly fraction from historical expectancy."""
        result = self.calculate_expectancy()
        if result.sample_size == 0 or result.avg_loss == 0 or result.avg_win == 0:
            return 0.0

        edge = result.win_rate - (1 - result.win_rate) / (
            result.avg_win / result.avg_loss
        )
        edge = max(0.0, edge)  # guard against negative edge

        if kelly_mode.lower() == "half":
            edge *= 0.5
        return min(1.0, edge)

    def get_trade_quality_score(
        self, expected_win_rate: float, expected_risk_reward: float
    ) -> dict:
        """
        Provide a quick quality score for a prospective trade.

        Returns a dict with quality_score (0-100), expected_ev, and a
        coarse recommendation label.
        """
        expected_ev = expected_win_rate * expected_risk_reward - (1 - expected_win_rate)
        quality_score = max(0.0, min(100.0, (expected_ev + 1.0) * 50.0))

        if quality_score >= 75:
            recommendation = "PURSUE"
        elif quality_score >= 50:
            recommendation = "CONSIDER"
        elif quality_score >= 25:
            recommendation = "AVOID"
        else:
            recommendation = "STRONG_AVOID"

        return {
            "quality_score": quality_score,
            "expected_ev": expected_ev,
            "recommendation": recommendation,
        }
