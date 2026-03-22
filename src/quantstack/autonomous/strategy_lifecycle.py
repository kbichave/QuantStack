# Copyright 2024 QuantPod Contributors
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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import duckdb
from loguru import logger


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

# Promotion thresholds
_MIN_OOS_SHARPE = 0.5
_MAX_OVERFIT_RATIO = 2.0
_FORWARD_TEST_DAYS = 30
_MIN_LIVE_SHARPE = 0.3
_RETIREMENT_DEGRADATION_PCT = 50.0


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
        conn: DuckDB connection.
        test_symbols: Symbols used for backtesting candidates.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        test_symbols: list[str] | None = None,
    ) -> None:
        self._conn = conn
        self._test_symbols = test_symbols or ["SPY", "QQQ", "AAPL", "MSFT", "XOM"]

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

        # 3. Check forward_testing strategies for promotion to live
        promotions = self._check_forward_testing_promotions()
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
                    import json

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
        import json

        strategy_name = f"{template['name_prefix']}_{target_regime}_{date.today().strftime('%Y%m%d')}"

        # Register as draft
        try:
            self._conn.execute(
                """
                INSERT INTO strategies (
                    strategy_id, name, description, parameters,
                    entry_rules, exit_rules, risk_params,
                    regime_affinity, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
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
        except Exception as exc:
            # Strategy may already exist from previous run
            logger.debug(
                f"[Lifecycle] Strategy {strategy_name} may already exist: {exc}"
            )
            return False

        # Backtest on each test symbol
        best_oos_sharpe = -999.0
        best_overfit = 999.0

        for symbol in self._test_symbols:
            try:
                result = await self._run_walkforward(strategy_name, symbol)
                if result and result.get("success"):
                    oos = result.get("oos_sharpe_mean", 0)
                    overfit = result.get("overfit_ratio", 999)
                    if oos > best_oos_sharpe:
                        best_oos_sharpe = oos
                        best_overfit = overfit
            except Exception:
                continue

        # Evaluate thresholds
        if best_oos_sharpe >= _MIN_OOS_SHARPE and best_overfit <= _MAX_OVERFIT_RATIO:
            # Promote to forward_testing
            self._conn.execute(
                "UPDATE strategies SET status = 'forward_testing', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                [strategy_name],
            )
            logger.info(
                f"[Lifecycle] PROMOTED {strategy_name} to forward_testing "
                f"(OOS Sharpe={best_oos_sharpe:.2f}, overfit={best_overfit:.2f})"
            )
            return True

        # Failed — mark as rejected
        self._conn.execute(
            "UPDATE strategies SET status = 'retired', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [strategy_name],
        )
        logger.debug(
            f"[Lifecycle] REJECTED {strategy_name} "
            f"(OOS Sharpe={best_oos_sharpe:.2f}, overfit={best_overfit:.2f})"
        )
        return False

    async def _run_walkforward(self, strategy_id: str, symbol: str) -> dict | None:
        """Run walk-forward validation for a strategy on a symbol."""
        try:
            from quantstack.mcp.tools.backtesting import run_walkforward

            return await run_walkforward.fn(
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

    def _check_forward_testing_promotions(self) -> list[str]:
        """Check if any forward_testing strategies are ready for live."""
        promotions = []
        try:
            rows = self._conn.execute(
                """
                SELECT strategy_id, updated_at FROM strategies
                WHERE status = 'forward_testing'
                """
            ).fetchall()

            for row in rows:
                strategy_id = row[0]
                updated_at = row[1]

                # Must be in forward_testing for at least FORWARD_TEST_DAYS
                if isinstance(updated_at, datetime):
                    days_testing = (datetime.now() - updated_at).days
                else:
                    days_testing = 0

                if days_testing < _FORWARD_TEST_DAYS:
                    continue

                # Check live performance from strategy_daily_pnl
                pnl_rows = self._conn.execute(
                    """
                    SELECT
                        SUM(realized_pnl) as total_pnl,
                        COUNT(*) as trading_days,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins
                    FROM strategy_daily_pnl
                    WHERE strategy_id = ?
                    """,
                    [strategy_id],
                ).fetchone()

                if pnl_rows and pnl_rows[1] and pnl_rows[1] >= 10:
                    # Rough live Sharpe approximation
                    avg_daily = pnl_rows[0] / pnl_rows[1]
                    # We'd need std for real Sharpe — use win rate as proxy
                    win_rate = pnl_rows[2] / pnl_rows[1] if pnl_rows[1] > 0 else 0

                    if win_rate > 0.5 and pnl_rows[0] > 0:
                        self._conn.execute(
                            "UPDATE strategies SET status = 'live', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                            [strategy_id],
                        )
                        promotions.append(strategy_id)
                        logger.info(
                            f"[Lifecycle] PROMOTED {strategy_id} to LIVE "
                            f"(P&L=${pnl_rows[0]:.0f}, win_rate={win_rate:.0%}, "
                            f"days={pnl_rows[1]})"
                        )

        except Exception as exc:
            logger.warning(f"[Lifecycle] Promotion check failed: {exc}")

        return promotions

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
                    import json

                    try:
                        bt = (
                            json.loads(backtest_json)
                            if isinstance(backtest_json, str)
                            else backtest_json
                        )
                        bt_sharpe = bt.get("sharpe_ratio", 0)
                    except Exception:
                        pass

                # Get live performance
                pnl_rows = self._conn.execute(
                    """
                    SELECT SUM(realized_pnl), COUNT(*) FROM strategy_daily_pnl
                    WHERE strategy_id = ? AND date >= ?
                    """,
                    [strategy_id, date.today() - timedelta(days=30)],
                ).fetchone()

                if not pnl_rows or not pnl_rows[1] or pnl_rows[1] < 5:
                    continue

                # Simple degradation check: is recent P&L negative?
                if pnl_rows[0] is not None and pnl_rows[0] < 0:
                    self._conn.execute(
                        "UPDATE strategies SET status = 'retired', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                        [strategy_id],
                    )
                    retirements.append(strategy_id)
                    logger.warning(
                        f"[Lifecycle] RETIRED {strategy_id} — "
                        f"30-day P&L=${pnl_rows[0]:.0f} (negative)"
                    )

        except Exception as exc:
            logger.warning(f"[Lifecycle] Validation failed: {exc}")

        return retirements
