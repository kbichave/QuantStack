# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Execution Researcher Pod — optimizes HOW to execute and construct the portfolio.

Analyzes TCA results, strategy correlations, factor exposure, and position
sizing effectiveness. Produces actionable recommendations that the trading
pipeline can apply deterministically.

Monthly cycle. Runs after enough trade data accumulates.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import duckdb
from loguru import logger

from quant_pod.research.context import ResearchContext
from quant_pod.research.schemas import (
    ExecutionRecommendation,
    ExecutionResearchPlan,
)


_SYSTEM_PROMPT = """You are a senior execution researcher at a systematic trading firm.
You optimize trade execution quality and portfolio construction.

Your domain expertise:
- Almgren-Chriss optimal execution: market impact = k × σ × sqrt(Q/V)
- VWAP/TWAP strategies: when to use each based on urgency and volume profile
- Time-of-day effects: first/last 30 minutes have wider spreads and higher impact
- Kelly criterion: f* = (bp - q) / b where b=odds, p=win probability, q=1-p
- Fractional Kelly: use f*/2 or f*/3 in practice (reduces drawdown, small Sharpe cost)
- Portfolio construction: HRP (Hierarchical Risk Parity) for strategy allocation
- Factor exposure: decompose returns into market, size, value, momentum, quality factors
- Strategy correlation: >0.7 pairwise correlation = redundant, reduce combined allocation

Your approach:
1. Measure before recommending. Every recommendation has supporting data.
2. Focus on the BIGGEST leak first. A 5bps timing improvement beats a 0.5bps algo change.
3. Correlation warnings are high priority — concentrated portfolios blow up.
4. Factor exposure warnings are critical — single-factor strategies amplify drawdowns.

Output format: Respond with ONLY valid JSON matching the ExecutionResearchPlan schema."""


class ExecutionResearcher:
    """
    Execution research pod — TCA analysis, correlation, factor exposure.

    Primarily rule-based (TCA analysis is data-driven, not creative).
    LLM used only for nuanced portfolio construction recommendations.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._context = ResearchContext(conn)

    async def generate_plan(self) -> ExecutionResearchPlan:
        """Analyze execution quality and portfolio construction."""
        context = self._context.build_execution_researcher_context()

        # Execution analysis is primarily rule-based
        plan = self._analyze_execution(context)

        # Optionally enhance with LLM for portfolio recommendations
        llm_recs = await self._llm_portfolio_analysis(context)
        if llm_recs:
            plan.recommendations.extend(llm_recs)

        self._persist_plan(plan)

        logger.info(
            f"[ExecutionResearcher] Generated plan: "
            f"{len(plan.recommendations)} recommendations"
        )
        return plan

    def _analyze_execution(self, context: dict) -> ExecutionResearchPlan:
        """Rule-based execution analysis from TCA and P&L data."""
        recommendations: list[ExecutionRecommendation] = []

        # 1. TCA: time-of-day slippage analysis
        tca_recs = self._analyze_tca()
        recommendations.extend(tca_recs)

        # 2. Strategy correlation analysis
        corr_recs, correlations = self._analyze_strategy_correlation()
        recommendations.extend(corr_recs)

        # 3. Strategy P&L concentration
        pnl = context.get("strategy_pnl", [])
        if pnl and len(pnl) >= 2:
            total_pnl = sum(abs(s["total_pnl"]) for s in pnl)
            if total_pnl > 0:
                top_pnl_pct = abs(pnl[0]["total_pnl"]) / total_pnl * 100
                if top_pnl_pct > 60:
                    recommendations.append(ExecutionRecommendation(
                        recommendation_type="factor_exposure",
                        description=f"Strategy '{pnl[0]['strategy_id']}' accounts for {top_pnl_pct:.0f}% of absolute P&L — high concentration risk",
                        action="Reduce allocation to top strategy by 20% and redistribute to others",
                        evidence=f"Top strategy P&L=${pnl[0]['total_pnl']:.0f}, total absolute P&L=${total_pnl:.0f}",
                        impact_estimate="Reduces single-strategy drawdown risk by ~30%",
                        priority="high",
                    ))

        # 4. Average slippage
        avg_slip = self._get_avg_slippage()

        plan = ExecutionResearchPlan(
            plan_id=f"execplan_{uuid.uuid4().hex[:8]}",
            recommendations=recommendations,
            strategy_correlations=correlations,
            avg_slippage_bps=avg_slip,
            context_summary=f"Analyzed {len(pnl)} strategies, avg slippage {avg_slip:.1f}bps",
        )
        return plan

    def _analyze_tca(self) -> list[ExecutionRecommendation]:
        """Analyze TCA for timing and slippage patterns."""
        recs: list[ExecutionRecommendation] = []
        try:
            from quantcore.execution.tca_storage import TCAStore
            store = TCAStore()
            stats = store.get_aggregate_stats(lookback_days=30)
            store.close()

            if stats.get("trade_count", 0) < 5:
                return recs

            avg_slip = stats.get("avg_slippage_bps", 0)
            if avg_slip > 5:
                recs.append(ExecutionRecommendation(
                    recommendation_type="slippage_pattern",
                    description=f"Average slippage is {avg_slip:.1f}bps — above 5bps target",
                    action="Review order sizing vs ADV and consider TWAP for orders > 0.5% of ADV",
                    evidence=f"{stats['trade_count']} trades, median={stats.get('median_slippage_bps', 0):.1f}bps",
                    impact_estimate=f"Reducing from {avg_slip:.1f} to 3bps saves ~{(avg_slip - 3) * stats['trade_count'] * 20:.0f} annually at $20k avg notional",
                    priority="high" if avg_slip > 10 else "medium",
                ))
        except Exception:
            pass
        return recs

    def _analyze_strategy_correlation(self) -> tuple[list[ExecutionRecommendation], dict[str, float]]:
        """Compute pairwise strategy return correlations."""
        recs: list[ExecutionRecommendation] = []
        correlations: dict[str, float] = {}

        try:
            rows = self._conn.execute(
                """
                SELECT strategy_id, date, realized_pnl
                FROM strategy_daily_pnl
                WHERE date >= ?
                ORDER BY strategy_id, date
                """,
                [date.today() - timedelta(days=60)],
            ).fetchall()

            if not rows:
                return recs, correlations

            # Build returns dict per strategy
            import numpy as np

            by_strat: dict[str, dict[date, float]] = {}
            for r in rows:
                by_strat.setdefault(r[0], {})[r[1]] = r[2]

            strats = list(by_strat.keys())
            if len(strats) < 2:
                return recs, correlations

            # Pairwise correlation on overlapping dates
            for i, s1 in enumerate(strats):
                for s2 in strats[i + 1:]:
                    common = sorted(set(by_strat[s1]) & set(by_strat[s2]))
                    if len(common) < 10:
                        continue
                    r1 = np.array([by_strat[s1][d] for d in common])
                    r2 = np.array([by_strat[s2][d] for d in common])
                    if r1.std() == 0 or r2.std() == 0:
                        continue
                    corr = float(np.corrcoef(r1, r2)[0, 1])
                    key = f"{s1}|{s2}"
                    correlations[key] = round(corr, 3)

                    if abs(corr) > 0.7:
                        recs.append(ExecutionRecommendation(
                            recommendation_type="correlation_warning",
                            description=f"Strategies '{s1}' and '{s2}' have {corr:.2f} correlation — near-redundant",
                            action=f"Reduce combined allocation by 30% or merge into single strategy",
                            evidence=f"60-day pairwise correlation = {corr:.3f} on {len(common)} common trading days",
                            impact_estimate="Reduces correlated drawdown risk",
                            priority="critical" if abs(corr) > 0.85 else "high",
                        ))
        except Exception as exc:
            logger.debug(f"[ExecutionResearcher] Correlation analysis failed: {exc}")

        return recs, correlations

    def _get_avg_slippage(self) -> float:
        """Get average slippage from TCA storage."""
        try:
            from quantcore.execution.tca_storage import TCAStore
            store = TCAStore()
            stats = store.get_aggregate_stats(lookback_days=30)
            store.close()
            return stats.get("avg_slippage_bps", 0.0)
        except Exception:
            return 0.0

    async def _llm_portfolio_analysis(self, context: dict) -> list[ExecutionRecommendation]:
        """Optional LLM enhancement for portfolio construction insights."""
        # For now, rule-based is sufficient for execution research.
        # LLM can be added when we have enough data for nuanced analysis.
        return []

    def _persist_plan(self, plan: ExecutionResearchPlan) -> None:
        """Store the execution research plan in DuckDB."""
        try:
            self._conn.execute(
                """
                INSERT INTO research_plans (plan_id, pod_name, plan_type, plan_json, context_summary)
                VALUES (?, 'execution_researcher', 'monthly', ?, ?)
                """,
                [plan.plan_id, plan.model_dump_json(), plan.context_summary],
            )
        except Exception as exc:
            logger.warning(f"[ExecutionResearcher] Failed to persist plan: {exc}")
