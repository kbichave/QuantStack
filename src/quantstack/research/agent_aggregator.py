# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Result aggregation for parallel research agents.

In BLITZ mode, the orchestrator spawns multiple domain agents in parallel (e.g., 3 symbols × 3 domains = 9 agents).
Each agent returns a standardized AgentResult. The ResearchAggregator combines these results into a cross-domain summary.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AgentResult:
    """Standardized output from a research agent."""
    symbol: str
    domain: str  # 'investment', 'swing', 'options'
    status: str  # 'success', 'failure', 'locked', 'needs_more_data'
    strategies_registered: List[str] = field(default_factory=list)
    models_trained: List[str] = field(default_factory=list)
    hypotheses_tested: int = 0
    breakthrough_features: List[str] = field(default_factory=list)
    thesis_status: str = "unknown"  # 'intact', 'weakening', 'broken', 'unknown'
    thesis_summary: str = ""
    conflicts: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class ResearchAggregator:
    """Aggregates results from parallel domain agents into cross-domain summary."""

    def aggregate(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Aggregate parallel agent results into cross-domain summary.

        Args:
            results: List of AgentResult objects from parallel agents

        Returns:
            Dictionary with:
                - symbols_complete: symbols with all 3 domains successfully researched
                - symbols_partial: symbols with 1-2 domains completed
                - total_strategies: sum of strategies registered
                - total_models: sum of models trained
                - total_hypotheses: sum of hypotheses tested
                - breakthrough_features: features appearing in 2+ domains
                - conflicts: cross-domain thesis conflicts (bullish investment + bearish swing)
                - domain_coverage: {symbol: {domain: status}}
        """
        # Group by symbol
        by_symbol: Dict[str, Dict[str, AgentResult]] = {}
        for r in results:
            if r.symbol not in by_symbol:
                by_symbol[r.symbol] = {}
            by_symbol[r.symbol][r.domain] = r

        # Identify complete vs partial coverage
        symbols_complete = []
        symbols_partial = []
        for symbol, domains in by_symbol.items():
            success_count = sum(1 for d in domains.values() if d.status == 'success')
            if success_count == 3:
                symbols_complete.append(symbol)
            elif success_count > 0:
                symbols_partial.append(symbol)

        # Detect cross-domain conflicts (thesis divergence)
        conflicts = []
        for symbol, domains in by_symbol.items():
            theses = {d: domains[d].thesis_status for d in domains if domains[d].status == 'success'}
            if len(theses) >= 2:
                # Check for conflicts: intact vs broken, or intact vs weakening
                statuses = set(theses.values())
                if 'intact' in statuses and 'broken' in statuses:
                    conflicts.append(f"{symbol}: conflicting theses across domains — {theses}")
                elif 'intact' in statuses and 'weakening' in statuses:
                    conflicts.append(f"{symbol}: thesis weakening in some domains — {theses}")

        # Aggregate metrics (only from successful runs)
        successful_results = [r for r in results if r.status == 'success']
        total_strategies = sum(len(r.strategies_registered) for r in successful_results)
        total_models = sum(len(r.models_trained) for r in successful_results)
        total_hypotheses = sum(r.hypotheses_tested for r in successful_results)

        # Breakthrough features (appear in 2+ domains)
        feature_counts: Dict[str, int] = {}
        for r in successful_results:
            for feat in r.breakthrough_features:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        top_features = sorted(
            [f for f, cnt in feature_counts.items() if cnt >= 2],
            key=lambda f: feature_counts[f],
            reverse=True
        )

        # Domain coverage matrix
        domain_coverage = {
            symbol: {
                domain: result.status
                for domain, result in domains.items()
            }
            for symbol, domains in by_symbol.items()
        }

        # Compute success rate by domain
        domain_success_rate = {}
        for domain in ['investment', 'swing', 'options']:
            domain_results = [r for r in results if r.domain == domain]
            if domain_results:
                success_count = sum(1 for r in domain_results if r.status == 'success')
                domain_success_rate[domain] = success_count / len(domain_results)
            else:
                domain_success_rate[domain] = 0.0

        return {
            "symbols_complete": symbols_complete,
            "symbols_partial": symbols_partial,
            "total_strategies": total_strategies,
            "total_models": total_models,
            "total_hypotheses": total_hypotheses,
            "breakthrough_features": top_features,
            "conflicts": conflicts,
            "domain_coverage": domain_coverage,
            "domain_success_rate": domain_success_rate,
            "agents_spawned": len(results),
            "agents_succeeded": len(successful_results),
        }

    def format_summary(self, aggregated: Dict[str, Any]) -> str:
        """
        Format aggregated results into human-readable summary.

        Args:
            aggregated: Output from aggregate()

        Returns:
            Formatted summary string
        """
        lines = [
            "=== BLITZ Mode Research Summary ===",
            f"Agents: {aggregated['agents_spawned']} spawned, {aggregated['agents_succeeded']} succeeded",
            "",
            f"Complete Coverage: {len(aggregated['symbols_complete'])} symbols",
            f"  {', '.join(aggregated['symbols_complete']) if aggregated['symbols_complete'] else '(none)'}",
            "",
            f"Partial Coverage: {len(aggregated['symbols_partial'])} symbols",
            f"  {', '.join(aggregated['symbols_partial']) if aggregated['symbols_partial'] else '(none)'}",
            "",
            f"Output: {aggregated['total_strategies']} strategies, {aggregated['total_models']} models, "
            f"{aggregated['total_hypotheses']} hypotheses tested",
            "",
            f"Breakthrough Features (2+ domains): {len(aggregated['breakthrough_features'])}",
        ]

        if aggregated['breakthrough_features']:
            for feat in aggregated['breakthrough_features'][:5]:  # Show top 5
                lines.append(f"  - {feat}")

        lines.append("")
        lines.append("Domain Success Rates:")
        for domain, rate in aggregated['domain_success_rate'].items():
            lines.append(f"  {domain}: {rate:.0%}")

        if aggregated['conflicts']:
            lines.append("")
            lines.append(f"CONFLICTS ({len(aggregated['conflicts'])}):")
            for conflict in aggregated['conflicts']:
                lines.append(f"  - {conflict}")

        return "\n".join(lines)
