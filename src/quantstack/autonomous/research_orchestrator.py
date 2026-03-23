# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ResearchOrchestrator — autonomous research loop coordinator.

Wires the 3 research pods + AlphaDiscoveryEngine + ML pipeline into a
single scheduled lifecycle. No humans.

Schedule:
    NIGHTLY (after market close, 16:05 ET):
        1. EquityTracker.snapshot_daily()
        2. BenchmarkTracker.update_benchmark("SPY")
        3. Watchdog.run_once()
        4. AlphaResearcher.generate_plan() → hypotheses
        5. AlphaDiscoveryEngine.run(hypotheses) → draft strategies
        6. Log results

    WEEKLY (Saturday):
        7. MLScientist.generate_plan() → experiments
        8. Execute ML experiments (train/retrain/tune)
        9. WeightLearner.learn_weights()
        10. Strategy validation + promotion/retirement

    MONTHLY (1st Saturday):
        11. ExecutionResearcher.generate_plan() → recommendations
        12. Full model retraining for all symbols
        13. Concept drift check

Usage:
    orchestrator = ResearchOrchestrator(conn)
    await orchestrator.run_nightly()
    await orchestrator.run_weekly()
    await orchestrator.run_monthly()
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import duckdb
from loguru import logger

from quantstack.alpha_discovery.engine import AlphaDiscoveryEngine
from quantstack.autonomous.judge import HypothesisJudge
from quantstack.autonomous.strategy_lifecycle import StrategyLifecycle
from quantstack.autonomous.watchdog import Watchdog
from quantstack.ml.training_service import train_model
from quantstack.optimization.textgrad_loop import TextGradOptimizer
from quantstack.performance.benchmark import BenchmarkTracker
from quantstack.performance.equity_tracker import EquityTracker
from quantstack.performance.weight_learner import WeightLearner
from quantstack.research.alpha_researcher import AlphaResearcher
from quantstack.research.execution_researcher import ExecutionResearcher
from quantstack.research.ml_scientist import MLScientist


@dataclass
class OrchestratorReport:
    """Result of an orchestration run."""

    run_type: str  # "nightly", "weekly", "monthly"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    hypotheses_generated: int = 0
    experiments_designed: int = 0
    strategies_drafted: int = 0
    models_trained: int = 0
    recommendations: int = 0
    elapsed_s: float = 0.0


class ResearchOrchestrator:
    """
    Autonomous research loop — nightly/weekly/monthly schedule.

    Args:
        conn: DuckDB connection (write-enabled).
        watchlist: Default symbols for research. Can be overridden per-run.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        watchlist: list[str] | None = None,
    ) -> None:
        self._conn = conn
        self._watchlist = watchlist or [
            "SPY",
            "QQQ",
            "AAPL",
            "MSFT",
            "NVDA",
            "XOM",
            "JPM",
            "UNH",
            "AMZN",
            "GOOGL",
        ]

    async def run_nightly(self) -> OrchestratorReport:
        """
        Nightly research cycle:
        1. Snapshot equity and benchmark
        2. Run Alpha Researcher → hypotheses
        3. Feed hypotheses to AlphaDiscoveryEngine → draft strategies
        """
        report = OrchestratorReport(run_type="nightly")
        t0 = time.monotonic()

        # Step 1: Equity snapshot
        try:
            tracker = EquityTracker(self._conn)
            tracker.snapshot_daily()
            report.steps_completed.append("equity_snapshot")
        except Exception as exc:
            logger.error(f"[Orchestrator] Equity snapshot failed: {exc}")
            report.steps_failed.append(f"equity_snapshot: {exc}")

        # Step 2: Benchmark update
        try:
            bench = BenchmarkTracker(self._conn)
            bench.update_benchmark("SPY")
            report.steps_completed.append("benchmark_update")
        except Exception as exc:
            logger.warning(f"[Orchestrator] Benchmark update failed: {exc}")
            report.steps_failed.append(f"benchmark_update: {exc}")

        # Step 3: Watchdog health check
        try:
            wdog = Watchdog(self._conn)
            health = await wdog.run_once()
            if health.overall_status == "CRITICAL":
                logger.critical(
                    "[Orchestrator] CRITICAL health — aborting nightly research"
                )
                report.steps_failed.append("watchdog: CRITICAL — research aborted")
                report.finished_at = datetime.now(timezone.utc)
                report.elapsed_s = time.monotonic() - t0
                return report
            report.steps_completed.append(f"watchdog_{health.overall_status}")
        except Exception as exc:
            logger.warning(f"[Orchestrator] Watchdog failed: {exc}")
            report.steps_failed.append(f"watchdog: {exc}")

        # Step 4: Alpha Researcher → hypotheses
        try:
            researcher = AlphaResearcher(self._conn)
            plan = await researcher.generate_plan()
            report.hypotheses_generated = len(plan.hypotheses)
            report.steps_completed.append(
                f"alpha_researcher_{len(plan.hypotheses)}_hypotheses"
            )
        except Exception as exc:
            logger.error(f"[Orchestrator] Alpha Researcher failed: {exc}")
            report.steps_failed.append(f"alpha_researcher: {exc}")
            plan = None

        # Step 4.5: Hypothesis Judge — gate before expensive backtests (QuantAgent inner loop)
        if plan and plan.hypotheses:
            try:
                judge = HypothesisJudge(self._conn)
                approved = []
                rejected = 0
                for hyp in plan.hypotheses:
                    hyp_dict = {
                        "name": getattr(hyp, "name", getattr(hyp, "hypothesis_id", "unknown")),
                        "description": getattr(hyp, "description", getattr(hyp, "thesis", "")),
                        "features": getattr(hyp, "features", []),
                        "regime_target": getattr(hyp, "regime_target", ""),
                        "parameters": getattr(hyp, "parameters", {}),
                        "entry_rules": getattr(hyp, "entry_rules", []),
                    }
                    verdict = judge.review(hyp_dict)
                    if verdict.approved:
                        approved.append(hyp)
                    else:
                        rejected += 1
                plan.hypotheses = approved
                if rejected:
                    logger.info(
                        f"[Orchestrator] Judge: {rejected} hypotheses rejected, "
                        f"{len(approved)} approved for backtesting"
                    )
                report.steps_completed.append(f"judge_{len(approved)}ok_{rejected}rejected")
            except Exception as exc:
                logger.warning(f"[Orchestrator] Hypothesis Judge failed (non-blocking): {exc}")
                report.steps_failed.append(f"judge: {exc}")

        # Step 5: AlphaDiscoveryEngine with approved hypotheses
        if plan and plan.hypotheses:
            try:
                strategies_found = await self._run_discovery(plan)
                report.strategies_drafted = strategies_found
                report.steps_completed.append(
                    f"discovery_{strategies_found}_strategies"
                )
            except Exception as exc:
                logger.error(f"[Orchestrator] Discovery failed: {exc}")
                report.steps_failed.append(f"discovery: {exc}")

        # Step 6: TextGrad — critique losing trades + failed research (TextGrad, Nature 2024)
        try:
            textgrad = TextGradOptimizer(self._conn)
            trade_proposals = textgrad.run_daily(date.today())
            if trade_proposals:
                report.steps_completed.append(f"textgrad_{len(trade_proposals)}_proposals")
        except Exception as exc:
            logger.debug(f"[Orchestrator] TextGrad failed (non-blocking): {exc}")

        report.finished_at = datetime.now(timezone.utc)
        report.elapsed_s = time.monotonic() - t0

        logger.info(
            f"[Orchestrator] Nightly complete: "
            f"{len(report.steps_completed)} steps OK, {len(report.steps_failed)} failed, "
            f"{report.hypotheses_generated} hypotheses, {report.strategies_drafted} drafts, "
            f"{report.elapsed_s:.0f}s"
        )
        return report

    async def run_weekly(self) -> OrchestratorReport:
        """
        Weekly research cycle (Saturday):
        1. ML Scientist → experiment plan
        2. Execute ML experiments
        3. Learn synthesis weights
        4. Validate live strategies
        """
        report = OrchestratorReport(run_type="weekly")
        t0 = time.monotonic()

        # Step 1: ML Scientist
        try:
            scientist = MLScientist(self._conn)
            ml_plan = await scientist.generate_plan()
            report.experiments_designed = len(ml_plan.experiments)
            report.steps_completed.append(
                f"ml_scientist_{len(ml_plan.experiments)}_experiments"
            )
        except Exception as exc:
            logger.error(f"[Orchestrator] ML Scientist failed: {exc}")
            report.steps_failed.append(f"ml_scientist: {exc}")
            ml_plan = None

        # Step 2: Execute ML experiments
        if ml_plan and ml_plan.experiments:
            try:
                trained = await self._execute_ml_experiments(ml_plan)
                report.models_trained = trained
                report.steps_completed.append(f"ml_training_{trained}_models")
            except Exception as exc:
                logger.error(f"[Orchestrator] ML training failed: {exc}")
                report.steps_failed.append(f"ml_training: {exc}")

        # Step 3: Learn synthesis weights
        try:
            learner = WeightLearner(self._conn)
            learner.learn_weights(lookback_days=90)
            report.steps_completed.append("weight_learning")
        except Exception as exc:
            logger.warning(f"[Orchestrator] Weight learning failed: {exc}")
            report.steps_failed.append(f"weight_learning: {exc}")

        # Step 4: Strategy validation
        try:
            lifecycle = StrategyLifecycle(self._conn)
            monthly_report = await lifecycle.run_monthly()
            report.steps_completed.append(
                f"validation_{len(monthly_report.retirements)}_retired"
            )
        except Exception as exc:
            logger.warning(f"[Orchestrator] Strategy validation failed: {exc}")
            report.steps_failed.append(f"validation: {exc}")

        # NOTE: OPRO weekly prompt evolution (steps 5, 5.5) removed — premature.
        # Fitness scoring on <10 trades/week is noise. Re-enable after 500+ trades.
        # See docs/OPTIMIZATION.md "When to Revisit".

        report.finished_at = datetime.now(timezone.utc)
        report.elapsed_s = time.monotonic() - t0

        logger.info(
            f"[Orchestrator] Weekly complete: "
            f"{len(report.steps_completed)} steps OK, {len(report.steps_failed)} failed, "
            f"{report.experiments_designed} experiments, {report.models_trained} trained, "
            f"{report.elapsed_s:.0f}s"
        )
        return report

    async def run_monthly(self) -> OrchestratorReport:
        """
        Monthly research cycle (1st Saturday):
        1. Execution Researcher → recommendations
        2. Full model retraining
        3. Concept drift check
        """
        report = OrchestratorReport(run_type="monthly")
        t0 = time.monotonic()

        # Step 1: Execution Researcher
        try:
            exec_researcher = ExecutionResearcher(self._conn)
            exec_plan = await exec_researcher.generate_plan()
            report.recommendations = len(exec_plan.recommendations)
            report.steps_completed.append(
                f"execution_researcher_{len(exec_plan.recommendations)}_recs"
            )
        except Exception as exc:
            logger.error(f"[Orchestrator] Execution Researcher failed: {exc}")
            report.steps_failed.append(f"execution_researcher: {exc}")

        # Step 2: Full retraining for all watchlist symbols
        try:
            trained = await self._retrain_all()
            report.models_trained = trained
            report.steps_completed.append(f"full_retrain_{trained}_models")
        except Exception as exc:
            logger.error(f"[Orchestrator] Full retraining failed: {exc}")
            report.steps_failed.append(f"full_retrain: {exc}")

        report.finished_at = datetime.now(timezone.utc)
        report.elapsed_s = time.monotonic() - t0

        logger.info(
            f"[Orchestrator] Monthly complete: "
            f"{len(report.steps_completed)} steps OK, {len(report.steps_failed)} failed, "
            f"{report.recommendations} recommendations, {report.models_trained} retrained, "
            f"{report.elapsed_s:.0f}s"
        )
        return report

    # ── Internal execution methods ──────────────────────────────────────

    async def _run_discovery(self, plan: Any) -> int:
        """Run AlphaDiscoveryEngine with researcher hypotheses."""
        try:
            engine = AlphaDiscoveryEngine(conn=self._conn)

            # Extract target symbols from hypotheses (deduplicated)
            symbols = list(
                {
                    sym
                    for h in plan.hypotheses
                    for sym in (h.target_symbols or self._watchlist[:5])
                }
            )[:10]

            result = await engine.run(
                symbols=symbols,
                dry_run=False,
            )
            return result.registered + result.gp_registered

        except Exception as exc:
            logger.error(f"[Orchestrator] AlphaDiscoveryEngine failed: {exc}")
            return 0

    async def _execute_ml_experiments(self, plan: Any) -> int:
        """Execute ML experiments from the scientist's plan."""
        trained = 0

        for exp in plan.experiments:
            try:
                config = exp.config or {}
                result = await train_model(
                    symbol=exp.symbol,
                    model_type=config.get("model_type", "lightgbm"),
                    feature_tiers=config.get("feature_tiers", ["technical"]),
                    apply_causal_filter=True,
                )

                if result and result.get("success"):
                    trained += 1

                    # Log experiment
                    self._log_experiment(exp, result)

            except Exception as exc:
                logger.warning(
                    f"[Orchestrator] Experiment {exp.experiment_id} failed: {exc}"
                )
                self._log_experiment(exp, {"success": False, "error": str(exc)})

        return trained

    async def _retrain_all(self) -> int:
        """Retrain models for all watchlist symbols."""
        trained = 0
        for symbol in self._watchlist:
            try:
                result = await train_model(
                    symbol=symbol,
                    model_type="lightgbm",
                    feature_tiers=["technical", "fundamentals"],
                    apply_causal_filter=True,
                )
                if result and result.get("success"):
                    trained += 1
            except Exception as exc:
                logger.debug(f"[Orchestrator] Retrain {symbol} failed: {exc}")
        return trained

    def _log_experiment(self, exp: Any, result: dict) -> None:
        """Log an ML experiment to the experiments table.

        Also updates HypothesisJudge knowledge base with outcome (QuantAgent outer loop).
        """
        try:
            exp_id = (
                getattr(exp, "experiment_id", "") or f"mlexp_{_uuid.uuid4().hex[:8]}"
            )
            self._conn.execute(
                """
                INSERT INTO ml_experiments
                    (experiment_id, symbol, model_type, feature_tiers,
                     test_auc, cv_auc_mean, top_features, verdict, hypothesis_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (experiment_id) DO NOTHING
                """,
                [
                    exp_id,
                    getattr(exp, "symbol", ""),
                    (getattr(exp, "config", {}) or {}).get("model_type", "lightgbm"),
                    json.dumps(
                        (getattr(exp, "config", {}) or {}).get("feature_tiers", [])
                    ),
                    result.get("test_auc"),
                    result.get("cv_auc_mean"),
                    json.dumps(result.get("top_features", [])),
                    "champion" if result.get("success") else "failed",
                    getattr(exp, "hypothesis_id", None),
                ],
            )
        except Exception as exc:
            logger.debug(f"[Orchestrator] Experiment logging failed: {exc}")

        # QuantAgent outer loop: update judge knowledge base with outcome
        try:
            judge = HypothesisJudge(self._conn)
            strategy_id = getattr(exp, "hypothesis_id", exp_id)
            judge.update_knowledge(strategy_id, {
                "sharpe": result.get("sharpe", 0),
                "win_rate": result.get("test_auc", 0),
                "n_trades": result.get("n_trades", 0),
            })
        except Exception as exc:
            logger.debug(f"[Orchestrator] Judge knowledge update failed: {exc}")
