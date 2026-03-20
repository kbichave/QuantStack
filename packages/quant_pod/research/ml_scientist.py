# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ML Scientist Pod — decides HOW to train models.

Analyzes model performance history, feature tier effectiveness, and concept
drift to design targeted ML experiments. Handles:
- Which models need retraining (stale, drifted)
- Which feature tiers produce best OOS performance
- Which hyperparameters to tune
- Champion vs challenger model comparison
- Feature ablation studies

Unlike the Alpha Researcher (which decides WHAT to research), the ML Scientist
decides HOW to model the signals that the researcher identifies.

Bi-weekly cycle. Runs after Alpha Researcher has identified new hypotheses.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import duckdb
from loguru import logger

from quant_pod.research.context import ResearchContext
from quant_pod.research.schemas import MLExperiment, MLExperimentPlan


_SYSTEM_PROMPT = """You are a senior ML engineer at a systematic trading firm.
You design and evaluate ML experiments for equity price prediction models.

Your domain expertise:
- Gradient boosting: LightGBM (dart mode, GOSS sampling), XGBoost (alpha regularization),
  CatBoost (ordered boosting for time series)
- Feature importance: SHAP values measure contribution, NOT causality. A high-SHAP
  feature may be spurious if it fails Granger causality test.
- Label engineering: Event-based (ATR TP/SL), wave-based (Elliott wave completion),
  multi-horizon (predicting both 1-day and 5-day returns)
- Time-series CV: Purged K-Fold with embargo (Lopez de Prado). NEVER shuffle.
- Concept drift: PSI (Population Stability Index) on feature distributions.
  Feature drift precedes model degradation by ~5-10 trading days.
- Ensemble calibration: Platt scaling, isotonic regression. Raw tree probabilities
  are often miscalibrated — calibrate before combining.

Your approach:
1. ONE variable at a time. Don't change model + features + params simultaneously.
2. Always compare to baseline. A new model must beat the current champion OOS.
3. Feature ablation before addition. Remove bottom 20% SHAP features first —
   less noise often beats more signal.
4. Respect the causal filter. If CausalFilter drops a feature, don't add it back
   manually. The feature may correlate with returns in-sample but not cause them.
5. Log EVERYTHING. Every experiment, every metric, every decision.

Output format: Respond with ONLY valid JSON matching the MLExperimentPlan schema."""

_STALE_MODEL_DAYS = 30
_DRIFT_RETRAIN_THRESHOLD = 0.10  # PSI > 0.10 triggers retrain


class MLScientist:
    """
    ML research pod — designs and evaluates model training experiments.

    Modes:
    1. LLM-assisted: Uses Groq/Claude for nuanced experiment design
    2. Rule-based fallback: Deterministic retraining rules
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._context = ResearchContext(conn)

    async def generate_plan(self) -> MLExperimentPlan:
        """Generate an ML experiment plan from model performance analysis."""
        context = self._context.build_ml_scientist_context()

        plan = await self._llm_generate(context)
        if plan is None:
            plan = self._rule_based_generate(context)

        self._persist_plan(plan)

        logger.info(
            f"[MLScientist] Generated plan: "
            f"{len(plan.experiments)} experiments, "
            f"{len(plan.retrain_symbols)} retrains, "
            f"{len(plan.feature_drops)} feature drops"
        )
        return plan

    async def _llm_generate(self, context: dict) -> MLExperimentPlan | None:
        """Try LLM-assisted experiment design."""
        try:
            prompt = self._build_prompt(context)
            response = await self._call_llm(prompt)
            if response is None:
                return None
            return self._parse_response(response)
        except Exception as exc:
            logger.warning(f"[MLScientist] LLM generation failed: {exc}")
            return None

    def _build_prompt(self, context: dict) -> str:
        """Build prompt with ML-specific context."""
        sections = []

        # Model status
        models = context.get("model_status", [])
        if models:
            sections.append("## Trained Models")
            for m in models:
                sections.append(
                    f"- {m['symbol']}: {m['model_type']} AUC={m.get('test_auc', '?')} "
                    f"age={m['age_days']}d {'STALE' if m.get('stale') else 'OK'} "
                    f"top_features={m.get('top_features', [])[:3]}"
                )
        else:
            sections.append("## No trained models exist yet — need initial training run")

        # Recent experiments
        experiments = context.get("experiments", [])
        if experiments:
            sections.append("\n## Recent Experiments")
            for exp in experiments[:8]:
                sections.append(
                    f"- {exp['id']}: {exp['symbol']} {exp['model_type']} "
                    f"AUC={exp.get('test_auc', '?')} CV={exp.get('cv_auc_mean', '?')} "
                    f"verdict={exp.get('verdict', '?')}"
                )

        # Breakthrough features
        breakthroughs = context.get("breakthrough_features", [])
        if breakthroughs:
            sections.append("\n## High-Value Features")
            for bf in breakthroughs[:5]:
                sections.append(
                    f"- {bf['feature']}: avg_importance={bf['avg_importance']:.3f} "
                    f"in {bf['occurrences']} models"
                )

        # Portfolio performance
        equity = context.get("equity_summary", {})
        if equity:
            sections.append(
                f"\n## Portfolio: sharpe_30d={equity.get('sharpe_30d', '?')} "
                f"max_dd={equity.get('max_drawdown_30d', '?')}%"
            )

        sections.append(
            "\n## Task\n"
            "Design 3-5 ML experiments to improve model performance.\n"
            "Priority: retrain stale/drifted models > feature ablation > architecture changes.\n"
            "One variable at a time. Always compare to baseline.\n"
            "Respond with ONLY valid JSON matching the MLExperimentPlan schema."
        )

        return "\n".join(sections)

    async def _call_llm(self, prompt: str) -> str | None:
        """Call LLM for experiment plan generation."""
        import asyncio

        try:
            from quant_pod.llm_config import get_llm_for_role

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
                    temperature=0.2,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                ),
                timeout=25.0,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning(f"[MLScientist] LLM call failed: {exc}")
            return None

    def _parse_response(self, response: str) -> MLExperimentPlan | None:
        """Parse LLM response into validated MLExperimentPlan."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

            data = json.loads(text.strip())

            for exp in data.get("experiments", []):
                if not exp.get("experiment_id"):
                    exp["experiment_id"] = f"mlexp_{uuid.uuid4().hex[:8]}"

            plan = MLExperimentPlan(**data)
            plan.plan_id = f"mlplan_{uuid.uuid4().hex[:8]}"
            return plan
        except Exception as exc:
            logger.warning(f"[MLScientist] Parse failed: {exc}")
            return None

    def _rule_based_generate(self, context: dict) -> MLExperimentPlan:
        """Deterministic experiment generation when LLM is unavailable."""
        experiments: list[MLExperiment] = []
        retrain_symbols: list[str] = []

        # 1. Retrain stale models
        models = context.get("model_status", [])
        for m in models:
            if m.get("stale") or m.get("age_days", 0) > _STALE_MODEL_DAYS:
                retrain_symbols.append(m["symbol"])
                experiments.append(MLExperiment(
                    experiment_id=f"mlexp_retrain_{m['symbol']}_{uuid.uuid4().hex[:4]}",
                    experiment_type="retrain",
                    symbol=m["symbol"],
                    hypothesis=f"Model for {m['symbol']} is {m.get('age_days', '?')} days old — retraining with latest data should improve OOS AUC",
                    success_criteria="New model AUC > current champion AUC",
                    failure_analysis_plan="If AUC drops, check for regime shift or feature drift",
                    config={"model_type": m.get("model_type", "lightgbm")},
                    priority=1,
                ))

        # 2. If no stale models, try feature ablation on worst performer
        if not experiments and models:
            worst = min(models, key=lambda m: m.get("test_auc", 1.0))
            experiments.append(MLExperiment(
                experiment_id=f"mlexp_ablation_{worst['symbol']}_{uuid.uuid4().hex[:4]}",
                experiment_type="feature_ablation",
                symbol=worst["symbol"],
                hypothesis=f"Removing bottom 20% SHAP features from {worst['symbol']} model will reduce noise and improve OOS AUC",
                success_criteria="AUC improves by >0.01 OOS",
                failure_analysis_plan="If AUC drops, the removed features carried real signal despite low importance",
                config={"ablation_pct": 0.2, "model_type": worst.get("model_type", "lightgbm")},
                priority=3,
            ))

        # 3. If no models exist at all, train initial models
        if not models:
            for symbol in ["SPY", "AAPL", "MSFT"]:
                experiments.append(MLExperiment(
                    experiment_id=f"mlexp_initial_{symbol}_{uuid.uuid4().hex[:4]}",
                    experiment_type="retrain",
                    symbol=symbol,
                    hypothesis=f"Initial model training for {symbol} with technical + fundamental features",
                    success_criteria="AUC > 0.55 OOS (better than random)",
                    failure_analysis_plan="If AUC < 0.55, the feature set may not capture signal for this symbol",
                    config={"model_type": "lightgbm", "feature_tiers": ["technical", "fundamentals"]},
                    priority=1,
                ))

        plan = MLExperimentPlan(
            plan_id=f"mlplan_rule_{uuid.uuid4().hex[:8]}",
            experiments=experiments[:5],
            retrain_symbols=retrain_symbols,
            context_summary="Rule-based plan (LLM unavailable)",
        )
        return plan

    def _persist_plan(self, plan: MLExperimentPlan) -> None:
        """Store the ML experiment plan in DuckDB."""
        try:
            self._conn.execute(
                """
                INSERT INTO research_plans (plan_id, pod_name, plan_type, plan_json, context_summary)
                VALUES (?, 'ml_scientist', 'biweekly', ?, ?)
                """,
                [plan.plan_id, plan.model_dump_json(), plan.context_summary],
            )
        except Exception as exc:
            logger.warning(f"[MLScientist] Failed to persist plan: {exc}")
