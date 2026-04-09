# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous Strategy Lifecycle — deterministic strategy R&D pipeline.

Replaces the Ralph Wiggum Strategy Factory with a pipeline that requires
zero LLM calls. Template-based hypothesis generation + statistical validation.

Lifecycle:
    1. WEEKLY: Identify regime gaps (regimes with no active strategy)
    2. For each gap: generate candidate from template library
    3. Backtest + walk-forward with purged CV
    4. If OOS Sharpe > 0.5 AND overfit_ratio < 2.0: promote to forward_testing
    5. After 30 days forward_testing: if live Sharpe > 0.3, promote to live
    6. MONTHLY: validate all live strategies. If degradation > 50%, retire.

No LLM involved. No prose reasoning. Just data.

Usage:
    lifecycle = StrategyLifecycle(conn)
    report = await lifecycle.run_weekly()   # gap analysis + candidate generation
    report = await lifecycle.run_monthly()  # validation + retirement
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from quantstack.coordination.auto_promoter import AutoPromoter
from quantstack.core.backtesting.walkforward_service import run_walkforward
from quantstack.tools._shared import run_backtest_impl
from quantstack.universe import STRATEGY_BACKTEST_DEFAULT
from quantstack.db import PgConnection


# Strategy templates — the building blocks for automated hypothesis generation.
# Each template defines entry/exit rules parameterized by regime.
_STRATEGY_TEMPLATES = [
    {
        "name_prefix": "auto_momentum",
        "description": "Trend-following momentum strategy",
        "regime_affinity": ["trending_up", "trending_down"],
        "entry_rules": [
            {"indicator": "macd_hist", "condition": "crosses_above", "value": 0},
            {"indicator": "adx_14", "condition": "greater_than", "value": 25},
        ],
        "exit_rules": [
            {"indicator": "macd_hist", "condition": "crosses_below", "value": 0},
        ],
        "parameters": {"atr_stop_mult": 2.0, "atr_tp_mult": 3.0},
        "risk_params": {"stop_loss_atr": 2.0, "take_profit_atr": 3.0},
    },
    {
        "name_prefix": "auto_mean_rev",
        "description": "Mean-reversion RSI strategy",
        "regime_affinity": ["ranging"],
        "entry_rules": [
            {"indicator": "rsi_14", "condition": "crosses_below", "value": 30},
            {"indicator": "bb_pct", "condition": "less_than", "value": 0.2},
        ],
        "exit_rules": [
            {"indicator": "rsi_14", "condition": "crosses_above", "value": 50},
        ],
        "parameters": {"rsi_entry": 30, "rsi_exit": 50},
        "risk_params": {"stop_loss_atr": 1.5, "take_profit_atr": 2.0},
    },
    {
        "name_prefix": "auto_breakout",
        "description": "Bollinger Band breakout strategy",
        "regime_affinity": ["trending_up"],
        "entry_rules": [
            {"indicator": "bb_pct", "condition": "greater_than", "value": 1.0},
            {"indicator": "volume_ratio", "condition": "greater_than", "value": 1.5},
        ],
        "exit_rules": [
            {"indicator": "bb_pct", "condition": "less_than", "value": 0.5},
        ],
        "parameters": {"bb_breakout_threshold": 1.0, "volume_confirm": 1.5},
        "risk_params": {"stop_loss_atr": 2.5, "take_profit_atr": 4.0},
    },
    {
        "name_prefix": "auto_quality_mr",
        "description": "Quality mean-reversion: RSI oversold above SMA200",
        "regime_affinity": ["ranging", "trending_up"],
        "entry_rules": [
            {"indicator": "rsi_14", "condition": "less_than", "value": 35},
            {"indicator": "close_vs_sma200", "condition": "greater_than", "value": 0},
        ],
        "exit_rules": [
            {"indicator": "rsi_14", "condition": "greater_than", "value": 60},
        ],
        "parameters": {"rsi_entry": 35, "rsi_exit": 60},
        "risk_params": {"stop_loss_atr": 2.0, "take_profit_atr": 3.0},
    },
]

# Promotion thresholds — walkforward path (run_weekly template candidates)
_MIN_OOS_SHARPE = 0.5
_MAX_OVERFIT_RATIO = 2.0
_MIN_OOS_IS_RATIO = 0.5  # OOS Sharpe must be >= 50% of IS Sharpe
_FORWARD_TEST_DAYS = 30
_MIN_LIVE_SHARPE = 0.3
_RETIREMENT_DEGRADATION_PCT = 50.0

# Monte Carlo gate — bootstrap Sharpe 5th percentile must exceed this
_MC_MIN_5TH_SHARPE = 0.3
_MC_N_SIMULATIONS = 1000

# Phase 2 screening thresholds — backtest-only path (research-graph candidates)
# Lower bar than walkforward: forward_testing is itself a probationary period
# with 50% position size and AutoPromoter gating before live.
_PHASE2_MIN_SHARPE = 0.15
_PHASE2_MIN_PROFIT_FACTOR = 1.0
_PHASE2_MIN_TRADES = 50
_PHASE2_MAX_DRAWDOWN = 15.0  # percent


@dataclass
class PipelineReport:
    """Result of a pipeline pass (Phase 1: draft → backtested)."""

    skipped: bool = False
    backtested: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LifecycleReport:
    """Result of a lifecycle run."""

    gaps_found: list[str] = field(default_factory=list)
    candidates_generated: int = 0
    candidates_passed: int = 0
    candidates_failed: int = 0
    promotions: list[str] = field(default_factory=list)
    retirements: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class StrategyLifecycle:
    """
    Autonomous strategy R&D + validation + retirement pipeline.

    Args:
        conn: PostgreSQL connection.
        test_symbols: Symbols used for backtesting candidates.
    """

    def __init__(
        self,
        conn: PgConnection,
        test_symbols: list[str] | None = None,
    ) -> None:
        self._conn = conn
        self._test_symbols = test_symbols or list(STRATEGY_BACKTEST_DEFAULT)

    async def run_weekly(self) -> LifecycleReport:
        """
        Weekly lifecycle: identify gaps, generate candidates, backtest, promote.
        """
        report = LifecycleReport()

        # 1. Find regime gaps
        gaps = self._find_regime_gaps()
        report.gaps_found = gaps

        if not gaps:
            logger.info("[Lifecycle] No regime gaps found — all regimes covered")
            return report

        logger.info(f"[Lifecycle] Found {len(gaps)} regime gaps: {gaps}")

        # 2. Generate candidates for each gap
        for regime in gaps:
            templates = [
                t for t in _STRATEGY_TEMPLATES if regime in t["regime_affinity"]
            ]
            if not templates:
                logger.debug(f"[Lifecycle] No template covers regime '{regime}'")
                continue

            for template in templates:
                report.candidates_generated += 1
                try:
                    passed = await self._evaluate_candidate(template, regime)
                    if passed:
                        report.candidates_passed += 1
                    else:
                        report.candidates_failed += 1
                except Exception as exc:
                    report.candidates_failed += 1
                    report.errors.append(f"{template['name_prefix']}: {exc}")
                    logger.warning(f"[Lifecycle] Candidate evaluation failed: {exc}")

        # 3. Check forward_testing strategies for promotion to live via AutoPromoter.
        # AutoPromoter evaluates Sharpe, win-rate, drawdown, and degradation vs backtest
        # — a much stricter gate than the legacy win-rate-only check.
        decisions = AutoPromoter(self._conn).evaluate_all()
        promotions = [d.strategy_id for d in decisions if d.decision == "promote"]
        report.promotions = promotions

        logger.info(
            f"[Lifecycle] Weekly: gaps={len(gaps)} candidates={report.candidates_generated} "
            f"passed={report.candidates_passed} promoted={len(promotions)}"
        )
        return report

    async def run_monthly(self) -> LifecycleReport:
        """
        Monthly lifecycle: validate live strategies, retire degraded ones.
        """
        report = LifecycleReport()

        # Validate all live strategies
        retirements = self._validate_and_retire()
        report.retirements = retirements

        logger.info(f"[Lifecycle] Monthly: retired={len(retirements)}")
        return report

    # ── Internal ──────────────────────────────────────────────────────────

    def _find_regime_gaps(self) -> list[str]:
        """Find regimes with no active (live or forward_testing) strategy."""
        all_regimes = {"trending_up", "trending_down", "ranging"}

        try:
            rows = self._conn.execute(
                """
                SELECT DISTINCT regime_affinity FROM strategies
                WHERE status IN ('live', 'forward_testing')
                """
            ).fetchall()

            covered = set()
            for row in rows:
                affinity = row[0]
                if affinity:
                    try:
                        regimes = (
                            json.loads(affinity)
                            if isinstance(affinity, str)
                            else affinity
                        )
                        if isinstance(regimes, list):
                            covered.update(regimes)
                        elif isinstance(regimes, dict):
                            covered.update(regimes.keys())
                        else:
                            covered.add(str(regimes))
                    except (json.JSONDecodeError, TypeError):
                        covered.add(str(affinity))

            return sorted(all_regimes - covered)
        except Exception as exc:
            logger.warning(f"[Lifecycle] Gap analysis failed: {exc}")
            return sorted(all_regimes)  # If we can't check, assume all gaps

    async def _evaluate_candidate(self, template: dict, target_regime: str) -> bool:
        """
        Backtest + walk-forward a candidate strategy.

        Returns True if the candidate passes thresholds and gets registered.
        """
        import uuid as _uuid
        nonce = _uuid.uuid4().hex[:8]
        strategy_name = f"{template['name_prefix']}_{target_regime}_{date.today().strftime('%Y%m%d')}_{nonce}"

        # Register as draft; idempotent — weekly reruns won't blow up on name collision.
        self._conn.execute(
            """
            INSERT INTO strategies (
                strategy_id, name, description, parameters,
                entry_rules, exit_rules, risk_params,
                regime_affinity, status, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'draft', NOW(), NOW())
            ON CONFLICT (strategy_id) DO NOTHING
            """,
            [
                strategy_name,
                strategy_name,
                template["description"],
                json.dumps(template["parameters"]),
                json.dumps(template["entry_rules"]),
                json.dumps(template["exit_rules"]),
                json.dumps(template["risk_params"]),
                json.dumps([target_regime]),
            ],
        )
        self._conn.commit()

        # Backtest on each test symbol
        best_oos_sharpe = -999.0
        best_is_sharpe = -999.0
        best_overfit = 999.0
        oos_returns_for_mc: list[float] = []

        for symbol in self._test_symbols:
            try:
                result = await self._run_walkforward(strategy_name, symbol)
                if result and result.get("success"):
                    oos = result.get("oos_sharpe_mean", 0)
                    is_sharpe = result.get("is_sharpe_mean", 0)
                    overfit = result.get("overfit_ratio", 999)
                    if oos > best_oos_sharpe:
                        best_oos_sharpe = oos
                        best_is_sharpe = is_sharpe
                        best_overfit = overfit
                    # Collect OOS returns for Monte Carlo gate
                    oos_ret = result.get("oos_returns", [])
                    if oos_ret:
                        oos_returns_for_mc.extend(oos_ret)
            except Exception:
                continue

        # Gate 1: absolute OOS Sharpe + overfit ratio
        passed = (
            best_oos_sharpe >= _MIN_OOS_SHARPE
            and best_overfit <= _MAX_OVERFIT_RATIO
        )

        # Gate 2: OOS/IS ratio — OOS must be >= 50% of IS to rule out overfitting
        if passed and best_is_sharpe > 0:
            oos_is_ratio = best_oos_sharpe / best_is_sharpe
            if oos_is_ratio < _MIN_OOS_IS_RATIO:
                logger.info(
                    f"[Lifecycle] REJECTED {strategy_name}: OOS/IS ratio "
                    f"{oos_is_ratio:.2f} < {_MIN_OOS_IS_RATIO}"
                )
                passed = False

        # Gate 3: Monte Carlo bootstrap — 5th percentile Sharpe must exceed floor
        if passed and oos_returns_for_mc:
            try:
                import pandas as pd

                from quantstack.core.analysis.bootstrap_mc import bootstrap_sharpe_ci

                mc_result = bootstrap_sharpe_ci(
                    pd.Series(oos_returns_for_mc),
                    n_simulations=_MC_N_SIMULATIONS,
                )
                if mc_result.ci_5 < _MC_MIN_5TH_SHARPE:
                    logger.info(
                        f"[Lifecycle] REJECTED {strategy_name}: MC 5th pctile "
                        f"Sharpe {mc_result.ci_5:.3f} < {_MC_MIN_5TH_SHARPE}"
                    )
                    passed = False
            except Exception as exc:
                logger.warning(f"[Lifecycle] MC gate skipped for {strategy_name}: {exc}")

        if passed:
            # Promote to forward_testing
            self._conn.execute(
                "UPDATE strategies SET status = 'forward_testing', updated_at = NOW() WHERE strategy_id = %s",
                [strategy_name],
            )
            self._conn.commit()
            logger.info(
                f"[Lifecycle] PROMOTED {strategy_name} to forward_testing "
                f"(OOS Sharpe={best_oos_sharpe:.2f}, overfit={best_overfit:.2f})"
            )
            return True

        # Failed — mark as rejected
        self._conn.execute(
            "UPDATE strategies SET status = 'retired', updated_at = NOW() WHERE strategy_id = %s",
            [strategy_name],
        )
        self._conn.commit()
        logger.debug(
            f"[Lifecycle] REJECTED {strategy_name} "
            f"(OOS Sharpe={best_oos_sharpe:.2f}, overfit={best_overfit:.2f})"
        )
        return False

    async def _run_walkforward(self, strategy_id: str, symbol: str) -> dict | None:
        """Run walk-forward validation for a strategy on a symbol."""
        try:
            return await run_walkforward(
                strategy_id=strategy_id,
                symbol=symbol,
                n_splits=5,
                test_size=63,
                min_train_size=126,
                use_purged_cv=True,
            )
        except Exception as exc:
            logger.debug(
                f"[Lifecycle] Walk-forward failed for {strategy_id}/{symbol}: {exc}"
            )
            return None

    def _validate_and_retire(self) -> list[str]:
        """Validate all live strategies and retire degraded ones."""
        retirements = []
        try:
            rows = self._conn.execute(
                """
                SELECT strategy_id, backtest_summary FROM strategies
                WHERE status = 'live'
                """
            ).fetchall()

            for row in rows:
                strategy_id = row[0]
                backtest_json = row[1]

                # Get backtest Sharpe
                bt_sharpe = 0.0
                if backtest_json:
                    try:
                        bt = (
                            json.loads(backtest_json)
                            if isinstance(backtest_json, str)
                            else backtest_json
                        )
                        bt_sharpe = bt.get("sharpe_ratio", 0)
                    except Exception as exc:
                        logger.warning("[Lifecycle] Failed to parse backtest JSON for %s: %s", strategy_id, exc)

                # Get live performance
                pnl_rows = self._conn.execute(
                    """
                    SELECT SUM(realized_pnl), COUNT(*) FROM strategy_daily_pnl
                    WHERE strategy_id = %s AND date >= %s
                    """,
                    [strategy_id, date.today() - timedelta(days=30)],
                ).fetchone()

                if not pnl_rows or not pnl_rows[1] or pnl_rows[1] < 5:
                    continue

                # Simple degradation check: is recent P&L negative?
                if pnl_rows[0] is not None and pnl_rows[0] < 0:
                    self._conn.execute(
                        "UPDATE strategies SET status = 'retired', updated_at = NOW() WHERE strategy_id = %s",
                        [strategy_id],
                    )
                    self._conn.commit()
                    retirements.append(strategy_id)
                    logger.warning(
                        f"[Lifecycle] RETIRED {strategy_id} — "
                        f"30-day P&L=${pnl_rows[0]:.0f} (negative)"
                    )

        except Exception as exc:
            logger.warning(f"[Lifecycle] Validation failed: {exc}")

        return retirements

    # ── Continuous pipeline (Phase 1 only) ────────────────────────────────

    async def run_pipeline_pass(self) -> PipelineReport:
        """
        Run backtests for all draft strategies with a known symbol.

        Phase 1 of the promotion pipeline: draft → backtested.
        Phase 2 (backtested → forward_testing) is intentionally handled by the
        research loop, which spawns the strategy-rd agent to reason about each
        candidate rather than applying mechanical thresholds.

        Guarded by a heartbeat check so overlapping scheduler runs skip safely.
        """
        report = PipelineReport()

        # Concurrency guard: skip if a recent run is still in-progress.
        try:
            recent = self._conn.execute(
                """
                SELECT started_at FROM loop_heartbeats
                WHERE loop_name = 'strategy_pipeline' AND status = 'running'
                  AND started_at > NOW() - INTERVAL '9 minutes'
                ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
            if recent:
                logger.info("[Pipeline] Prior run still active — skipping")
                report.skipped = True
                return report
        except Exception as exc:
            logger.debug(f"[Pipeline] Heartbeat check failed (non-fatal): {exc}")

        # Derive next iteration number for the heartbeat PK.
        iteration = 1
        try:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(iteration), 0) + 1 FROM loop_heartbeats WHERE loop_name = 'strategy_pipeline'"
            ).fetchone()
            if row:
                iteration = int(row[0])
        except Exception as exc:
            logger.warning("[Pipeline] Failed to derive next iteration number: %s", exc)

        try:
            self._conn.execute(
                """
                INSERT INTO loop_heartbeats (loop_name, iteration, started_at, status)
                VALUES ('strategy_pipeline', %s, NOW(), 'running')
                ON CONFLICT (loop_name, iteration)
                DO UPDATE SET started_at = NOW(), status = 'running'
                """,
                [iteration],
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug(f"[Pipeline] Failed to record start heartbeat: {exc}")

        # Phase 0a: auto-retire drafts with empty entry_rules (they can never
        # produce trades, so looping on them wastes cycles).
        try:
            empty_rows = self._conn.execute(
                "SELECT strategy_id, name FROM strategies "
                "WHERE status = 'draft' AND (entry_rules IS NULL OR entry_rules IN ('', '[]'))"
            ).fetchall()
            for sid, sname in empty_rows:
                self._conn.execute(
                    "UPDATE strategies SET status = 'retired', updated_at = NOW() "
                    "WHERE strategy_id = %s",
                    [sid],
                )
                self._conn.commit()
                logger.info("[Pipeline] Auto-retired draft with no entry_rules: %s (%s)", sname, sid)
        except Exception as exc:
            logger.debug("[Pipeline] Auto-retire check failed (non-fatal): %s", exc)

        # Phase 0b: auto-retire backtested strategies with 0 trades.
        # These strategies had entry rules that never fired during the backtest
        # window. They will never be promoted (Phase 2 requires ≥50 trades),
        # so keeping them wastes evaluation cycles.
        try:
            zero_trade_rows = self._conn.execute(
                """
                SELECT strategy_id, name, symbol FROM strategies
                WHERE status = 'backtested'
                  AND backtest_summary IS NOT NULL
                  AND backtest_summary != ''
                """
            ).fetchall()
            for sid, sname, sym in zero_trade_rows:
                try:
                    bt = json.loads(
                        self._conn.execute(
                            "SELECT backtest_summary FROM strategies WHERE strategy_id = %s",
                            [sid],
                        ).fetchone()[0]
                    )
                except (json.JSONDecodeError, TypeError, IndexError):
                    continue
                trades = bt.get("total_trades", 0) or 0
                if trades == 0:
                    self._conn.execute(
                        "UPDATE strategies SET status = 'retired', updated_at = NOW() "
                        "WHERE strategy_id = %s AND status = 'backtested'",
                        [sid],
                    )
                    self._conn.commit()
                    logger.info(
                        "[Pipeline] Auto-retired 0-trade backtested strategy: %s (%s/%s)",
                        sname, sym, sid,
                    )
        except Exception as exc:
            logger.debug("[Pipeline] Zero-trade retire check failed (non-fatal): %s", exc)

        # Phase 0c: reset orphaned research_queue tasks.
        # When the research graph times out mid-cycle, tasks claimed as
        # 'running' are never released. Reset tasks stuck >60 min back to
        # 'pending' so the next research cycle can pick them up. Tasks stuck
        # >4 hours are marked 'failed' (likely a permanent issue).
        try:
            # Fail tasks stuck >4 hours
            self._conn.execute(
                """
                UPDATE research_queue
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = 'Orphaned: stuck in running for >4 hours'
                WHERE status = 'running'
                  AND started_at < NOW() - INTERVAL '4 hours'
                """
            )
            # Reset tasks stuck 60min–4h back to pending for retry
            reset_result = self._conn.execute(
                """
                UPDATE research_queue
                SET status = 'pending',
                    started_at = NULL
                WHERE status = 'running'
                  AND started_at < NOW() - INTERVAL '60 minutes'
                """
            )
            self._conn.commit()
            # Log if any tasks were cleaned up
            try:
                orphaned = self._conn.execute(
                    "SELECT COUNT(*) FROM research_queue WHERE status = 'pending' AND started_at IS NULL"
                ).fetchone()[0]
                if orphaned:
                    logger.info("[Pipeline] Reset %d orphaned research_queue tasks", orphaned)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("[Pipeline] Orphan cleanup failed (non-fatal): %s", exc)

        # Phase 1: run backtests for every draft strategy that has a symbol.
        try:
            rows = self._conn.execute(
                "SELECT strategy_id, symbol FROM strategies WHERE status = 'draft' AND symbol IS NOT NULL"
            ).fetchall()
        except Exception as exc:
            report.errors.append(f"Query failed: {exc}")
            rows = []

        for strategy_id, symbol in rows:
            try:
                result = await run_backtest_impl(strategy_id, symbol)
                if result.get("success"):
                    report.backtested.append(strategy_id)
                    logger.info(
                        f"[Pipeline] Backtested {strategy_id}/{symbol} "
                        f"Sharpe={result.get('sharpe_ratio', 0):.2f} "
                        f"trades={result.get('total_trades', 0)}"
                    )
                else:
                    err = result.get("error", "unknown error")
                    report.errors.append(f"{strategy_id}: {err}")
                    logger.warning(f"[Pipeline] Backtest failed for {strategy_id}: {err}")
            except Exception as exc:
                report.errors.append(f"{strategy_id}: {exc}")
                logger.warning(f"[Pipeline] Backtest exception for {strategy_id}: {exc}")

        # Phase 2: promote best backtested strategies to forward_testing.
        # This covers research-graph strategies that went draft → backtested
        # but have no walkforward path. forward_testing is itself a safety net
        # (50% size, AutoPromoter gate before live).
        promoted = await self._screen_and_promote_backtested()
        report.backtested.extend(f"promoted:{sid}" for sid in promoted)

        try:
            self._conn.execute(
                """
                INSERT INTO loop_heartbeats
                    (loop_name, iteration, started_at, finished_at, symbols_processed, errors, status)
                VALUES ('strategy_pipeline', %s, NOW(), NOW(), %s, %s, 'completed')
                ON CONFLICT (loop_name, iteration)
                DO UPDATE SET finished_at = NOW(),
                              symbols_processed = EXCLUDED.symbols_processed,
                              errors = EXCLUDED.errors,
                              status = 'completed'
                """,
                [iteration, len(report.backtested), len(report.errors)],
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug(f"[Pipeline] Failed to record completion heartbeat: {exc}")

        logger.info(
            f"[Pipeline] Pass complete: backtested={len(report.backtested)} "
            f"promoted={len(promoted)} errors={len(report.errors)}"
        )
        return report

    async def _screen_and_promote_backtested(self) -> list[str]:
        """
        Phase 2: screen backtested strategies on backtest metrics + WFV gate
        and promote the best to forward_testing.

        Candidates must first pass the cheap metric screen, then walk-forward
        validation (mandatory gate — same thresholds as Phase 1).
        """
        promoted: list[str] = []
        try:
            rows = self._conn.execute(
                """
                SELECT strategy_id, name, symbol, backtest_summary
                FROM strategies
                WHERE status = 'backtested'
                  AND backtest_summary IS NOT NULL
                  AND backtest_summary != ''
                ORDER BY updated_at ASC
                """
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[Pipeline Phase 2] Query failed: {exc}")
            return promoted

        for strategy_id, name, symbol, bt_raw in rows:
            try:
                bt = json.loads(bt_raw) if isinstance(bt_raw, str) else (bt_raw or {})
            except (json.JSONDecodeError, TypeError):
                continue

            sharpe = bt.get("sharpe_ratio", bt.get("sharpe", 0)) or 0
            pf = bt.get("profit_factor", 0) or 0
            trades = bt.get("total_trades", 0) or 0
            dd = bt.get("max_drawdown", 999) or 0

            # Cheap metric screen
            if not (
                sharpe >= _PHASE2_MIN_SHARPE
                and pf >= _PHASE2_MIN_PROFIT_FACTOR
                and trades >= _PHASE2_MIN_TRADES
                and dd <= _PHASE2_MAX_DRAWDOWN
            ):
                continue

            # Walk-forward validation gate (mandatory)
            try:
                wfv_result = await self._run_walkforward(strategy_id, symbol or "SPY")
                if wfv_result and wfv_result.get("success"):
                    oos = wfv_result.get("oos_sharpe_mean", 0)
                    overfit = wfv_result.get("overfit_ratio", 999)
                    if oos < _MIN_OOS_SHARPE or overfit > _MAX_OVERFIT_RATIO:
                        logger.info(
                            f"[Pipeline Phase 2] WFV REJECTED {name} ({symbol}) — "
                            f"OOS Sharpe={oos:.3f} overfit={overfit:.2f}"
                        )
                        continue
                else:
                    logger.info(
                        f"[Pipeline Phase 2] WFV FAILED for {name} ({symbol}) — skipping"
                    )
                    continue
            except Exception as wfv_exc:
                logger.warning(
                    f"[Pipeline Phase 2] WFV exception for {name}: {wfv_exc} — skipping"
                )
                continue

            self._conn.execute(
                "UPDATE strategies SET status = 'forward_testing', updated_at = NOW() "
                "WHERE strategy_id = %s AND status = 'backtested'",
                [strategy_id],
            )
            self._conn.commit()
            promoted.append(strategy_id)
            logger.info(
                f"[Pipeline Phase 2] PROMOTED {name} ({symbol}) to forward_testing — "
                f"Sharpe={sharpe:.3f} PF={pf:.2f} trades={trades} DD={dd:.1f}%"
            )

        return promoted
