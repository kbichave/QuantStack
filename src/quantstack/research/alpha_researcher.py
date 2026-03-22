# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Alpha Researcher Pod — decides WHAT to research.

The senior quant researcher who maintains a coherent multi-week research
program, analyzes why experiments fail, and generates hypotheses that the
deterministic pipeline validates.

This is NOT a wrapper that calls an LLM and hopes for the best. The LLM
provides creativity (hypothesis generation, failure diagnosis); the code
provides rigor (schema validation, dead-end tracking, experiment counting).

If the LLM hallucinates a bad hypothesis, the backtest catches it.
If the LLM produces invalid output, Pydantic rejects it.
If the LLM is unavailable, the pod falls back to rule-based hypothesis
generation from known patterns (SHAP features, regime gaps).

Weekly cycle:
    Monday:    Analyze (load context, identify patterns)
    Tuesday:   Hypothesize (generate research plan)
    Wed-Thu:   Execute (deterministic — AlphaDiscoveryEngine)
    Friday:    Review (failure analysis, update program)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import duckdb
from loguru import logger

from quantstack.research.context import ResearchContext
from quantstack.research.schemas import (
    FailureAnalysis,
    Hypothesis,
    ResearchPlan,
)


# Domain knowledge injected into the LLM's system prompt.
_SYSTEM_PROMPT = """You are a senior quantitative researcher at a systematic trading firm.
You maintain a coherent multi-week research program for equity trading strategies.

Your domain expertise:
- Factor investing: Fama-French 5-factor model, momentum (Jegadeesh & Titman),
  quality (Novy-Marx), value (Lakonishok, Shleifer, Vishny)
- Mean-reversion: Poterba & Summers (1988), Lo & MacKinlay (1988)
- Regime switching: Hamilton (1989), Ang & Bekaert (2002)
- Market microstructure: Kyle's lambda, Amihud illiquidity, VPIN
- Statistical arbitrage: Avellaneda & Lee (2010), cointegration-based pairs
- Overfitting in finance: Bailey & Lopez de Prado (2014), Harvey & Liu (2020)

Your approach:
1. You analyze RESULTS, not just metrics. When Sharpe is 0.3, you ask WHY.
2. You maintain 3-5 active investigations. Each investigation is a multi-week
   research direction with a clear thesis.
3. After 3 experiments with no improvement, you ABANDON the investigation
   with a documented reason, and start a new one.
4. You NEVER repeat dead ends. If a thesis was abandoned, it stays abandoned
   unless fundamentally new evidence emerges.
5. You build on what works. If RSI+regime is the top SHAP feature in 4 models,
   you explore RSI×regime interactions, not unrelated signals.

Available indicators for strategy rules:
Technical: rsi_14, macd_hist, adx_14, bb_pct, sma_20, sma_50, sma_200, stoch_k, cci
Volume: volume_ratio, obv_slope, vwap_deviation
Regime: trend_regime, volatility_regime, regime_confidence
Fundamental: fund_pe_ratio, fund_roe, fund_debt_to_equity
Flow: insider_net_90d, institutional_change_pct
Earnings: earn_days_to, earn_surprise
ML: ml_prediction, ml_confidence

Output format: You MUST respond with valid JSON matching the ResearchPlan schema.
No prose, no markdown, no explanation outside the JSON."""

_MAX_ACTIVE_INVESTIGATIONS = 5
_ABANDON_AFTER_FAILURES = 3


class AlphaResearcher:
    """
    Alpha research pod — generates hypotheses and maintains research program.

    The pod has two modes:
    1. LLM-assisted: Uses Claude/Groq for creative hypothesis generation
       and nuanced failure analysis.
    2. Rule-based fallback: When LLM is unavailable, generates hypotheses
       from known patterns (SHAP features, regime gaps, factor decay).

    Both modes produce the same output schema (ResearchPlan).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._context = ResearchContext(conn)

    async def generate_plan(self) -> ResearchPlan:
        """
        Generate a research plan: hypotheses, failure analyses, program updates.

        Tries LLM-assisted generation first, falls back to rule-based.
        """
        context = self._context.build_alpha_researcher_context()

        # Try LLM-assisted generation
        plan = await self._llm_generate(context)

        if plan is None:
            # Fallback: rule-based hypothesis generation
            plan = self._rule_based_generate(context)

        # Persist the plan
        self._persist_plan(plan)

        # Update research program from plan
        self._update_program(plan)

        logger.info(
            f"[AlphaResearcher] Generated plan: "
            f"{len(plan.hypotheses)} hypotheses, "
            f"{len(plan.failure_analyses)} failure analyses, "
            f"{len(plan.investigations_to_abandon)} abandonments"
        )
        return plan

    # ── LLM-assisted generation ──────────────────────────────────────────

    async def _llm_generate(self, context: dict) -> ResearchPlan | None:
        """Try to generate a plan via LLM. Returns None on any failure."""
        try:
            prompt = self._build_prompt(context)
            response = await self._call_llm(prompt)

            if response is None:
                return None

            # Parse and validate
            return self._parse_response(response)

        except Exception as exc:
            logger.warning(f"[AlphaResearcher] LLM generation failed: {exc}")
            return None

    def _build_prompt(self, context: dict) -> str:
        """Build the user prompt with full research context."""
        sections = []

        # Recent experiments
        experiments = context.get("experiments", [])
        if experiments:
            sections.append("## Recent Experiments (last 30 days)")
            for exp in experiments[:10]:
                sections.append(
                    f"- {exp['id']}: {exp['symbol']} {exp['model_type']} "
                    f"AUC={exp.get('test_auc', '?')} verdict={exp.get('verdict', '?')} "
                    f"features={exp.get('top_features', [])}"
                )
                if exp.get("failure_analysis"):
                    sections.append(f"  FAILURE: {exp['failure_analysis']}")

        # Strategy P&L
        pnl = context.get("strategy_pnl", [])
        if pnl:
            sections.append("\n## Strategy P&L (last 30 days)")
            for s in pnl:
                sections.append(
                    f"- {s['strategy_id']}: P&L=${s['total_pnl']:.0f} "
                    f"win_rate={s['win_rate']:.0%} trades={s['total_trades']}"
                )

        # Active investigations
        investigations = context.get("active_investigations", [])
        if investigations:
            sections.append("\n## Active Investigations")
            for inv in investigations:
                sections.append(
                    f"- [{inv['id']}] {inv['thesis']} "
                    f"(experiments={inv['experiments_run']}, best_sharpe={inv.get('best_oos_sharpe', '?')})"
                )
                if inv.get("next_steps"):
                    sections.append(f"  Next: {inv['next_steps']}")

        # Dead ends
        dead_ends = context.get("dead_ends", [])
        if dead_ends:
            sections.append("\n## Dead Ends (DO NOT REPEAT)")
            for de in dead_ends[:10]:
                sections.append(f"- {de['thesis']}: {de['reason']}")

        # Breakthrough features
        breakthroughs = context.get("breakthrough_features", [])
        if breakthroughs:
            sections.append("\n## Breakthrough Features (high-value signals)")
            for bf in breakthroughs[:8]:
                sections.append(
                    f"- {bf['feature']}: importance={bf['avg_importance']:.3f} "
                    f"occurrences={bf['occurrences']} regimes={bf.get('regimes', '?')}"
                )

        # Current regime
        regime = context.get("regime", {})
        sections.append(
            f"\n## Current Regime: SPY trend={regime.get('spy_trend', '?')} "
            f"vol={regime.get('spy_vol', '?')}"
        )

        # Equity summary
        equity = context.get("equity_summary", {})
        if equity:
            sections.append(
                f"\n## Portfolio: equity=${equity.get('current_equity', '?')} "
                f"sharpe_30d={equity.get('sharpe_30d', '?')} "
                f"max_dd={equity.get('max_drawdown_30d', '?')}%"
            )

        # Strategy breaker states
        breakers = context.get("breaker_states", [])
        if breakers:
            sections.append("\n## Tripped/Scaled Strategies")
            for b in breakers:
                sections.append(
                    f"- {b['strategy_id']}: {b['status']} "
                    f"(losses={b['consecutive_losses']}, dd={b['drawdown_pct']}%)"
                )

        sections.append(
            "\n## Task\n"
            "Based on the above context, produce a ResearchPlan with:\n"
            "1. 3-5 ranked hypotheses to test (prioritize regime gaps and decaying features)\n"
            "2. Failure analyses for any experiments that failed without clear diagnosis\n"
            "3. List of investigations to abandon (if any have 3+ failures)\n"
            "4. List of breakthrough features to track\n\n"
            "Respond with ONLY valid JSON matching the ResearchPlan schema."
        )

        return "\n".join(sections)

    async def _call_llm(self, prompt: str) -> str | None:
        """Call the LLM for research plan generation."""
        import asyncio

        try:
            from quantstack.llm_config import get_llm_for_role

            model = get_llm_for_role("research")
            if not model:
                model = "groq/llama-3.3-70b-versatile"

            import litellm

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                ),
                timeout=30.0,
            )
            return response.choices[0].message.content

        except Exception as exc:
            logger.warning(f"[AlphaResearcher] LLM call failed: {exc}")
            return None

    def _parse_response(self, response: str) -> ResearchPlan | None:
        """Parse LLM response into a validated ResearchPlan."""
        try:
            # Strip markdown fences if present
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            data = json.loads(text)

            # Assign IDs if missing
            for h in data.get("hypotheses", []):
                if not h.get("hypothesis_id"):
                    h["hypothesis_id"] = f"hyp_{uuid.uuid4().hex[:8]}"

            plan = ResearchPlan(**data)
            plan.plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            return plan

        except Exception as exc:
            logger.warning(f"[AlphaResearcher] Response parse failed: {exc}")
            return None

    # ── Rule-based fallback ──────────────────────────────────────────────

    def _rule_based_generate(self, context: dict) -> ResearchPlan:
        """
        Generate hypotheses from known patterns when LLM is unavailable.

        Priority order:
        1. Regime gaps (regimes with no profitable strategy)
        2. Decaying features (top SHAP features losing importance)
        3. Breakthrough feature interactions
        """
        hypotheses: list[Hypothesis] = []

        # 1. Regime gaps
        live_strategies = context.get("live_strategies", [])
        covered_regimes = set()
        for s in live_strategies:
            affinity = s.get("regime_affinity") or []
            if isinstance(affinity, list):
                covered_regimes.update(affinity)

        all_regimes = {"trending_up", "trending_down", "ranging"}
        gaps = all_regimes - covered_regimes

        for regime in gaps:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_gap_{regime}_{uuid.uuid4().hex[:4]}",
                    thesis=f"Mean-reversion strategy for {regime} regime using RSI + Bollinger Bands",
                    source="regime_gap",
                    target_regimes=[regime],
                    target_symbols=["SPY", "QQQ", "AAPL"],
                    entry_rules=(
                        [
                            {
                                "indicator": "rsi_14",
                                "condition": "less_than",
                                "value": 35,
                            },
                            {
                                "indicator": "bb_pct",
                                "condition": "less_than",
                                "value": 0.2,
                            },
                        ]
                        if regime == "ranging"
                        else [
                            {
                                "indicator": "macd_hist",
                                "condition": "crosses_above",
                                "value": 0,
                            },
                            {
                                "indicator": "adx_14",
                                "condition": "greater_than",
                                "value": 25,
                            },
                        ]
                    ),
                    exit_rules=[
                        {
                            "indicator": "rsi_14",
                            "condition": "greater_than",
                            "value": 55,
                        },
                    ],
                    priority=1,
                    rationale=f"Regime '{regime}' has no active strategy — basic coverage needed",
                )
            )

        # 2. Breakthrough feature interactions
        breakthroughs = context.get("breakthrough_features", [])
        if len(breakthroughs) >= 2:
            top_two = breakthroughs[:2]
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_interact_{uuid.uuid4().hex[:4]}",
                    thesis=(
                        f"Interaction between {top_two[0]['feature']} and {top_two[1]['feature']} "
                        f"captures non-linear alpha"
                    ),
                    source="feature_interaction",
                    target_regimes=["trending_up", "ranging"],
                    target_symbols=["SPY", "AAPL", "MSFT"],
                    feature_tiers=["technical", "fundamentals"],
                    priority=3,
                    rationale=(
                        f"Both features appear in {top_two[0]['occurrences']}+ winning strategies — "
                        f"their interaction may capture additional signal"
                    ),
                )
            )

        # 3. Strategy P&L analysis — what's decaying?
        pnl = context.get("strategy_pnl", [])
        decaying = [s for s in pnl if s["total_pnl"] < 0 and s["total_trades"] > 5]
        if decaying:
            worst = decaying[0]
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_decay_{uuid.uuid4().hex[:4]}",
                    thesis=(
                        f"Strategy '{worst['strategy_id']}' is decaying (P&L=${worst['total_pnl']:.0f}) — "
                        f"investigate alternative entry timing or feature set"
                    ),
                    source="factor_decay",
                    target_regimes=["trending_up", "ranging"],
                    target_symbols=["SPY"],
                    priority=2,
                    rationale=f"Negative P&L with {worst['total_trades']} trades suggests systematic issue",
                )
            )

        # Sort by priority
        hypotheses.sort(key=lambda h: h.priority)

        plan = ResearchPlan(
            plan_id=f"plan_rule_{uuid.uuid4().hex[:8]}",
            hypotheses=hypotheses[:5],
            context_summary="Rule-based plan (LLM unavailable)",
        )
        return plan

    # ── Persistence ──────────────────────────────────────────────────────

    def _persist_plan(self, plan: ResearchPlan) -> None:
        """Store the research plan in DuckDB."""
        try:
            self._conn.execute(
                """
                INSERT INTO research_plans (plan_id, pod_name, plan_type, plan_json, context_summary)
                VALUES (?, 'alpha_researcher', 'weekly', ?, ?)
                """,
                [plan.plan_id, plan.model_dump_json(), plan.context_summary],
            )
        except Exception as exc:
            logger.warning(f"[AlphaResearcher] Failed to persist plan: {exc}")

    def _update_program(self, plan: ResearchPlan) -> None:
        """Update the alpha_research_program table from the plan."""
        # Register new hypotheses as investigations
        for h in plan.hypotheses:
            try:
                self._conn.execute(
                    """
                    INSERT INTO alpha_research_program
                        (investigation_id, thesis, status, priority, source,
                         target_regimes, target_symbols)
                    VALUES (?, ?, 'active', ?, ?, ?, ?)
                    ON CONFLICT (investigation_id) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP,
                        priority = excluded.priority,
                        next_steps = 'Updated by plan ' || ?
                    """,
                    [
                        h.hypothesis_id,
                        h.thesis,
                        h.priority,
                        h.source,
                        json.dumps(h.target_regimes),
                        json.dumps(h.target_symbols),
                        plan.plan_id,
                    ],
                )
            except Exception as exc:
                logger.debug(f"[AlphaResearcher] Investigation upsert failed: {exc}")

        # Abandon investigations
        for inv_id in plan.investigations_to_abandon:
            try:
                self._conn.execute(
                    """
                    UPDATE alpha_research_program
                    SET status = 'abandoned', dead_end_reason = 'Abandoned by plan ' || ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE investigation_id = ? AND status = 'active'
                    """,
                    [plan.plan_id, inv_id],
                )
            except Exception:
                pass

        # Track breakthrough features
        for feat in plan.breakthrough_features:
            try:
                self._conn.execute(
                    """
                    INSERT INTO breakthrough_features (feature_name, occurrence_count, avg_shap_importance)
                    VALUES (?, 1, 0.0)
                    ON CONFLICT (feature_name) DO UPDATE SET
                        occurrence_count = breakthrough_features.occurrence_count + 1,
                        last_seen = CURRENT_TIMESTAMP
                    """,
                    [feat],
                )
            except Exception:
                pass
