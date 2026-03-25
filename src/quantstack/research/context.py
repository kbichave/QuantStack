# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Research context loader — gives each pod deep knowledge of what's happened.

At HRT, a PhD researcher walks into the office knowing what experiments ran
last week, which strategies are making money, which features matter, and
what the current market regime is. This module provides equivalent context
to the LLM research pods.

Every context loading function queries PostgreSQL and returns structured dicts
that get injected into the pod's system prompt.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from loguru import logger

from quantstack.db import PgConnection

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.execution.strategy_breaker import StrategyBreaker
from quantstack.signal_engine.collectors.regime import _rule_based_regime


class ResearchContext:
    """
    Loads research context from the database for pod system prompts.

    Each method returns a dict that can be JSON-serialized into the prompt.
    Methods are safe — they return empty/default on any error.
    """

    def __init__(self, conn: PgConnection) -> None:
        self._conn = conn

    def get_experiment_history(self, days: int = 30, limit: int = 20) -> list[dict]:
        """Last N experiments with results."""
        try:
            rows = self._conn.execute(
                """
                SELECT experiment_id, symbol, model_type, feature_tiers,
                       test_auc, cv_auc_mean, top_features, verdict,
                       failure_analysis, hypothesis_id, created_at
                FROM ml_experiments
                WHERE created_at >= ?
                ORDER BY created_at DESC LIMIT ?
                """,
                [date.today() - timedelta(days=days), limit],
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "model_type": r[2],
                    "feature_tiers": _safe_json(r[3]),
                    "test_auc": r[4],
                    "cv_auc_mean": r[5],
                    "top_features": _safe_json(r[6]),
                    "verdict": r[7],
                    "failure_analysis": r[8],
                    "hypothesis_id": r[9],
                    "date": str(r[10]),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug(f"[ResearchContext] experiment_history failed: {exc}")
            return []

    def get_strategy_pnl_summary(self, days: int = 30) -> list[dict]:
        """Per-strategy P&L over recent period."""
        try:
            rows = self._conn.execute(
                """
                SELECT strategy_id,
                       SUM(realized_pnl) as total_pnl,
                       SUM(num_trades) as total_trades,
                       SUM(win_count) as wins,
                       SUM(loss_count) as losses,
                       COUNT(*) as trading_days
                FROM strategy_daily_pnl
                WHERE date >= ?
                GROUP BY strategy_id
                ORDER BY total_pnl DESC
                """,
                [date.today() - timedelta(days=days)],
            ).fetchall()
            return [
                {
                    "strategy_id": r[0],
                    "total_pnl": round(r[1], 2),
                    "total_trades": r[2],
                    "wins": r[3],
                    "losses": r[4],
                    "win_rate": (
                        round(r[3] / (r[3] + r[4]), 3) if (r[3] + r[4]) > 0 else 0
                    ),
                    "trading_days": r[5],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug(f"[ResearchContext] strategy_pnl failed: {exc}")
            return []

    def get_active_investigations(self) -> list[dict]:
        """Current alpha research program investigations."""
        try:
            rows = self._conn.execute(
                """
                SELECT investigation_id, thesis, status, priority,
                       experiments_run, best_oos_sharpe, last_result_summary,
                       next_steps, target_regimes, target_symbols
                FROM alpha_research_program
                WHERE status = 'active'
                ORDER BY priority ASC, updated_at DESC
                """
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "thesis": r[1],
                    "status": r[2],
                    "priority": r[3],
                    "experiments_run": r[4],
                    "best_oos_sharpe": r[5],
                    "last_result": r[6],
                    "next_steps": r[7],
                    "target_regimes": _safe_json(r[8]),
                    "target_symbols": _safe_json(r[9]),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug(f"[ResearchContext] active_investigations failed: {exc}")
            return []

    def get_dead_ends(self) -> list[dict]:
        """Abandoned investigations — don't repeat these."""
        try:
            rows = self._conn.execute(
                """
                SELECT investigation_id, thesis, dead_end_reason
                FROM alpha_research_program
                WHERE status = 'abandoned'
                ORDER BY updated_at DESC LIMIT 20
                """
            ).fetchall()
            return [{"id": r[0], "thesis": r[1], "reason": r[2]} for r in rows]
        except Exception:
            return []

    def get_breakthrough_features(self) -> list[dict]:
        """Features that appear in 3+ winning strategies."""
        try:
            rows = self._conn.execute(
                """
                SELECT feature_name, occurrence_count, avg_shap_importance,
                       winning_strategies, regimes_effective
                FROM breakthrough_features
                WHERE occurrence_count >= 3
                ORDER BY avg_shap_importance DESC LIMIT 15
                """
            ).fetchall()
            return [
                {
                    "feature": r[0],
                    "occurrences": r[1],
                    "avg_importance": round(r[2], 4) if r[2] else 0,
                    "strategies": _safe_json(r[3]),
                    "regimes": _safe_json(r[4]),
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_regime_summary(self) -> dict:
        """Current and recent regime information."""
        try:
            store = DataStore(read_only=True)
            df = store.load_ohlcv("SPY", Timeframe.D1)
            if df is not None and len(df) >= 60:
                regime = _rule_based_regime(df)
                return {
                    "spy_trend": regime.get("trend_regime", "unknown"),
                    "spy_vol": regime.get("volatility_regime", "normal"),
                    "spy_confidence": regime.get("confidence", 0),
                }
        except Exception:
            pass
        return {"spy_trend": "unknown", "spy_vol": "normal", "spy_confidence": 0}

    def get_strategy_breaker_states(self) -> list[dict]:
        """Which strategies are tripped or scaled."""
        try:
            breaker = StrategyBreaker()
            states = breaker.get_all_states()
            return [
                {
                    "strategy_id": sid,
                    "status": s.status,
                    "consecutive_losses": s.consecutive_losses,
                    "drawdown_pct": round(s.drawdown_pct, 2),
                    "reason": s.reason,
                }
                for sid, s in states.items()
                if s.status != "ACTIVE"
            ]
        except Exception:
            return []

    def get_live_strategies(self) -> list[dict]:
        """All live and forward_testing strategies."""
        try:
            rows = self._conn.execute(
                """
                SELECT strategy_id, name, status, regime_affinity,
                       backtest_summary, walkforward_summary
                FROM strategies
                WHERE status IN ('live', 'forward_testing')
                ORDER BY status, name
                """
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "status": r[2],
                    "regime_affinity": _safe_json(r[3]),
                    "backtest": _safe_json(r[4]),
                    "walkforward": _safe_json(r[5]),
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_equity_summary(self, days: int = 30) -> dict:
        """Portfolio performance summary."""
        try:
            rows = self._conn.execute(
                """
                SELECT daily_return_pct, drawdown_pct, total_equity
                FROM daily_equity
                WHERE date >= ?
                ORDER BY date
                """,
                [date.today() - timedelta(days=days)],
            ).fetchall()
            if not rows:
                return {}

            returns = [r[0] for r in rows]
            n = len(returns)
            avg = sum(returns) / n if n > 0 else 0
            std = (sum((r - avg) ** 2 for r in returns) / max(n - 1, 1)) ** 0.5
            sharpe = (avg / std * 252**0.5) if std > 0 else 0
            max_dd = min(r[1] for r in rows)

            return {
                "trading_days": n,
                "current_equity": round(rows[-1][2], 0),
                "avg_daily_return": round(avg, 4),
                "sharpe_30d": round(sharpe, 2),
                "max_drawdown_30d": round(max_dd, 2),
                "total_return_30d": round(sum(returns), 2),
            }
        except Exception:
            return {}

    def get_model_status(self) -> list[dict]:
        """Status of all trained ML models."""
        try:
            rows = self._conn.execute(
                """
                SELECT symbol, model_type, test_auc, cv_auc_mean,
                       created_at, top_features
                FROM ml_experiments
                WHERE verdict = 'champion'
                ORDER BY symbol, created_at DESC
                """
            ).fetchall()
            # Deduplicate by symbol (latest champion only)
            seen = set()
            results = []
            for r in rows:
                if r[0] not in seen:
                    seen.add(r[0])
                    age_days = (
                        (date.today() - r[4].date()).days
                        if hasattr(r[4], "date")
                        else 0
                    )
                    results.append(
                        {
                            "symbol": r[0],
                            "model_type": r[1],
                            "test_auc": r[2],
                            "cv_auc_mean": r[3],
                            "age_days": age_days,
                            "stale": age_days > 30,
                            "top_features": _safe_json(r[5]),
                        }
                    )
            return results
        except Exception:
            return []

    def build_alpha_researcher_context(self) -> dict[str, Any]:
        """Build full context for the Alpha Researcher pod."""
        return {
            "experiments": self.get_experiment_history(days=30),
            "strategy_pnl": self.get_strategy_pnl_summary(days=30),
            "active_investigations": self.get_active_investigations(),
            "dead_ends": self.get_dead_ends(),
            "breakthrough_features": self.get_breakthrough_features(),
            "regime": self.get_regime_summary(),
            "breaker_states": self.get_strategy_breaker_states(),
            "live_strategies": self.get_live_strategies(),
            "equity_summary": self.get_equity_summary(),
        }

    def build_ml_scientist_context(self) -> dict[str, Any]:
        """Build full context for the ML Scientist pod."""
        return {
            "experiments": self.get_experiment_history(days=60),
            "model_status": self.get_model_status(),
            "breakthrough_features": self.get_breakthrough_features(),
            "equity_summary": self.get_equity_summary(),
        }

    def build_execution_researcher_context(self) -> dict[str, Any]:
        """Build full context for the Execution Researcher pod."""
        return {
            "strategy_pnl": self.get_strategy_pnl_summary(days=30),
            "equity_summary": self.get_equity_summary(),
            "live_strategies": self.get_live_strategies(),
            "breaker_states": self.get_strategy_breaker_states(),
        }


def _safe_json(val: Any) -> Any:
    """Parse JSON string or return as-is if already parsed."""
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val
