"""Structure-level performance statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from quant_pod.knowledge.models import StructureType, TradeRecord
from quant_pod.knowledge.store import KnowledgeStore


@dataclass
class StructureStatsSummary:
    """Aggregated statistics for a single structure type."""

    structure_type: StructureType
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    expectancy: float


class StructureStats:
    """Compute trade statistics grouped by option structure."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def _get_trades(
        self, structure_type: Optional[StructureType] = None
    ) -> List[TradeRecord]:
        trades = self.store.get_trades(limit=500)
        if structure_type:
            trades = [t for t in trades if t.structure_type == structure_type]
        return trades

    def get_structure_stats(
        self, structure_type: StructureType
    ) -> StructureStatsSummary:
        trades = self._get_trades(structure_type)
        total = len(trades)
        wins = [t for t in trades if (t.pnl or 0) > 0]
        losses = [t for t in trades if (t.pnl or 0) < 0]

        win_rate = len(wins) / total if total else 0.0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        return StructureStatsSummary(
            structure_type=structure_type,
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            expectancy=expectancy,
        )

    def get_best_structures(self, min_trades: int = 5) -> List[StructureStatsSummary]:
        """Return structures that meet the trade threshold sorted by expectancy."""
        summaries: List[StructureStatsSummary] = []
        for structure in StructureType:
            trades = self._get_trades(structure)
            if len(trades) < min_trades:
                continue
            summaries.append(self.get_structure_stats(structure))

        summaries.sort(key=lambda s: (s.expectancy, s.win_rate), reverse=True)
        return summaries

    def get_structure_recommendation(self, direction: str = "LONG") -> StructureType:
        """Return a simple recommendation based on historical performance."""
        best = self.get_best_structures(min_trades=1)
        if best:
            return best[0].structure_type

        # Fallback defaults
        if direction.upper() == "SHORT":
            return StructureType.PUT_SPREAD
        return StructureType.CALL_SPREAD
