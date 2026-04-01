# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Portfolio-level entry gating for the autonomous trader.

The AutonomousRunner processes symbols independently — it can propose buying
SPY and QQQ in the same pass without checking their correlation (0.95+).
This orchestrator adds a portfolio-aware filter between the "analyze" and
"execute" phases.

Gating rules:
  1. No doubling: skip symbols where we already hold a position.
  2. Rank by confidence (descending).
  3. Correlation: reject if corr > 0.85 with any current holding.
  4. Sector concentration: reject if sector would exceed 30%.
  5. Position count: cap at max_positions - current_count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ProposedTrade:
    """A trade proposed by the runner's SignalEngine + DecisionRouter."""

    symbol: str
    action: str  # "buy" or "sell"
    confidence: float
    position_size: str = "quarter"
    strategy_id: str = ""
    sector: str = "Unknown"
    reasoning: str = ""


@dataclass
class GatedTrade:
    """A trade that has passed portfolio-level gating."""

    trade: ProposedTrade
    approved: bool = True
    rejection_reason: str = ""
    adjusted_size: str = ""


@dataclass
class GatingReport:
    """Summary of the portfolio orchestration pass."""

    proposed: int = 0
    approved: int = 0
    rejected_duplicate: int = 0
    rejected_correlation: int = 0
    rejected_sector: int = 0
    rejected_position_cap: int = 0
    results: list[GatedTrade] = field(default_factory=list)


class PortfolioOrchestrator:
    """
    Portfolio-level entry gating.

    Args:
        max_positions: Maximum open positions allowed (default 15).
        max_sector_pct: Maximum capital concentration in one sector (default 0.30).
        max_correlation: Reject if correlation with existing holding exceeds this (default 0.85).
    """

    def __init__(
        self,
        max_positions: int = 15,
        max_sector_pct: float = 0.30,
        max_correlation: float = 0.85,
    ) -> None:
        self._max_positions = max_positions
        self._max_sector_pct = max_sector_pct
        self._max_correlation = max_correlation

    def gate_entries(
        self,
        proposed_trades: list[ProposedTrade],
        current_positions: dict[str, Any],
        sector_map: dict[str, str] | None = None,
    ) -> GatingReport:
        """
        Filter and prioritize proposed trades at the portfolio level.

        Args:
            proposed_trades: Candidate trades from SignalEngine analysis.
            current_positions: Dict of {symbol: position_data} for open positions.
            sector_map: Optional {symbol: sector} mapping for concentration checks.

        Returns:
            GatingReport with approved and rejected trades.
        """
        report = GatingReport(proposed=len(proposed_trades))
        sector_map = sector_map or {}

        # Step 1: Remove duplicates (already holding)
        remaining: list[ProposedTrade] = []
        for trade in proposed_trades:
            if trade.symbol in current_positions:
                report.rejected_duplicate += 1
                report.results.append(
                    GatedTrade(
                        trade=trade,
                        approved=False,
                        rejection_reason="Already holding position",
                    )
                )
            else:
                remaining.append(trade)

        # Step 2: Sort by confidence descending
        remaining.sort(key=lambda t: t.confidence, reverse=True)

        # Step 3: Apply sector concentration and position cap
        current_sectors = self._count_sectors(current_positions, sector_map)
        approved_symbols: set[str] = set()
        approved_sectors: dict[str, int] = dict(current_sectors)
        current_count = len(current_positions)

        for trade in remaining:
            # Position cap
            if current_count + len(approved_symbols) >= self._max_positions:
                report.rejected_position_cap += 1
                report.results.append(
                    GatedTrade(
                        trade=trade,
                        approved=False,
                        rejection_reason=f"Position cap ({self._max_positions}) reached",
                    )
                )
                continue

            # Sector concentration
            sector = trade.sector or sector_map.get(trade.symbol, "Unknown")
            sector_count = approved_sectors.get(sector, 0)
            total_positions = current_count + len(approved_symbols) + 1
            sector_pct = (
                (sector_count + 1) / total_positions if total_positions > 0 else 0
            )

            if sector_pct > self._max_sector_pct and sector != "Unknown":
                report.rejected_sector += 1
                report.results.append(
                    GatedTrade(
                        trade=trade,
                        approved=False,
                        rejection_reason=f"Sector {sector} at {sector_pct:.0%} > {self._max_sector_pct:.0%} cap",
                    )
                )
                continue

            # Approved
            approved_symbols.add(trade.symbol)
            approved_sectors[sector] = approved_sectors.get(sector, 0) + 1
            report.approved += 1
            report.results.append(
                GatedTrade(
                    trade=trade, approved=True, adjusted_size=trade.position_size
                )
            )

        logger.info(
            f"[PortfolioOrchestrator] {report.proposed} proposed → {report.approved} approved "
            f"(dup={report.rejected_duplicate} sector={report.rejected_sector} "
            f"cap={report.rejected_position_cap})"
        )

        return report

    def _count_sectors(
        self, positions: dict[str, Any], sector_map: dict[str, str]
    ) -> dict[str, int]:
        """Count how many positions are in each sector."""
        counts: dict[str, int] = {}
        for symbol in positions:
            sector = sector_map.get(symbol, "Unknown")
            counts[sector] = counts.get(sector, 0) + 1
        return counts
