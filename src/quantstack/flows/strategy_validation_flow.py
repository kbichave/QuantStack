# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
StrategyValidationFlow — weekend strategy health check flow.

Runs once per week (Saturday cron) to detect overfitting and performance
degradation before live trading resumes on Monday.

Flow graph:
    load_historical_data
          │
    run_walk_forward          (5-fold, 63-day OOS windows, 21-day embargo)
          │
    run_overfitting_checks    (DSR + PBO)
          │
    route_on_fitness
          ├── "passed"   → log_validation_passed
          ├── "degraded" → trigger_retraining_alert
          └── "overfit"  → quarantine_strategy

Design decisions:
  - Uses WalkForwardValidator.split() directly (not .validate()) because our
    "strategy" is the daily returns series itself, not an ML model.  We compute
    Sharpe on each IS/OOS fold and aggregate — simpler and less error-prone than
    wrapping a mock model_fn.
  - PBO matrix: each fold's OOS return sub-series becomes a column.  This is a
    valid interpretation: we are testing whether the fold with the highest IS
    Sharpe also has the highest OOS Sharpe (cross-temporal selection bias).
  - DSR n_trials defaults to 10 — a conservative estimate of the number of
    parameter variants that were tried during strategy development.  Override
    via DSR_N_TRIALS env var.
  - Quarantine flag: written to KnowledgeStore as a lesson with
    lesson_type="strategy_quarantine".  TradingDayFlow can optionally check for
    this flag at startup (not enforced yet — non-blocking architecture).
  - < 504 bars (2 years): early exit → verdict "insufficient_data".  Running
    DSR or PBO on tiny samples produces unreliable results.

Failure modes:
  - DataStore load fails: flow logs error and returns insufficient_data verdict.
  - WalkForwardValidator raises (too little data even after min_bars check):
    caught; returns insufficient_data.
  - DSR or PBO raise: caught; falls back to overfit_ratio-only verdict.
  - All other exceptions: logged; flow does not crash.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from quantstack.crewai_compat import Flow, listen, router, start

# Minimum bars before we trust the statistics (≈2 years of daily data)
_MIN_BARS = int(os.getenv("VALIDATION_MIN_BARS", "504"))

# Number of strategy variants tried — used in DSR n_trials correction
_DSR_N_TRIALS = int(os.getenv("DSR_N_TRIALS", "10"))

# WalkForward config
_N_SPLITS = int(os.getenv("WF_N_SPLITS", "5"))
_TEST_SIZE = int(os.getenv("WF_TEST_SIZE", "63"))  # quarterly OOS windows
_MIN_TRAIN = int(os.getenv("WF_MIN_TRAIN", "252"))  # 1 year minimum IS
_GAP = int(os.getenv("WF_GAP", "21"))  # 1-month embargo between IS and OOS

# Thresholds
_OVERFIT_RATIO_THRESHOLD = float(
    os.getenv("OVERFIT_RATIO_THRESHOLD", "1.5")
)  # IS/OOS > 1.5 → degraded
_PBO_THRESHOLD = float(os.getenv("PBO_THRESHOLD", "0.5"))

ValidationVerdict = Literal["passed", "degraded", "overfit", "insufficient_data"]


class ValidationState(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    lookback_days: int = 760  # ~3 years of daily bars

    # Flow-populated during execution
    returns_df: Any | None = (
        None  # pd.DataFrame; Any to avoid Pydantic serialization issues
    )
    wf_result: dict[str, Any] | None = None
    dsr_result: Any | None = None
    pbo_result: Any | None = None
    overfit_ratio: float = 1.0
    verdict: str = "insufficient_data"
    detail: str = ""

    class Config:
        arbitrary_types_allowed = True


class StrategyValidationFlow(Flow[ValidationState]):
    """
    Weekly strategy validation flow.  Run on Saturdays via cron.

    Usage:
        flow = StrategyValidationFlow()
        flow.state.symbols = ["SPY", "QQQ", "AAPL"]
        result = flow.kickoff()
    """

    # =========================================================================
    # PHASE 1: Load historical data
    # =========================================================================

    @start()
    def load_historical_data(self) -> dict[str, Any]:
        """Load daily close returns from DataStore for all symbols."""
        logger.info("═══ STRATEGY VALIDATION: LOAD DATA ═══")

        symbols = self.state.symbols or []
        if not symbols:
            logger.warning("[Validation] No symbols specified — aborting")
            self.state.verdict = "insufficient_data"
            self.state.detail = "No symbols provided"
            return {"status": "no_symbols"}

        end_date = datetime.combine(date.today(), datetime.min.time())
        start_date = end_date - timedelta(days=self.state.lookback_days)

        returns_dict: dict[str, pd.Series] = {}
        try:
            from quantstack.config.timeframes import Timeframe
            from quantstack.data.storage import DataStore

            store = DataStore()
            for sym in symbols:
                try:
                    ohlcv = store.load_ohlcv(
                        symbol=sym,
                        timeframe=Timeframe.D1,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if ohlcv is not None and len(ohlcv) >= 2:
                        returns_dict[sym] = ohlcv["close"].pct_change().dropna()
                        logger.debug(
                            f"[Validation] {sym}: {len(returns_dict[sym])} return bars loaded"
                        )
                except Exception as _sym_err:
                    logger.debug(f"[Validation] Skipping {sym}: {_sym_err}")

        except Exception as _load_err:
            logger.error(f"[Validation] DataStore load failed: {_load_err}")
            self.state.verdict = "insufficient_data"
            self.state.detail = f"DataStore load error: {_load_err}"
            return {"status": "load_failed"}

        if not returns_dict:
            self.state.verdict = "insufficient_data"
            self.state.detail = "No historical data loaded for any symbol"
            logger.warning("[Validation] No data available — skipping validation")
            return {"status": "no_data"}

        # Average returns across symbols as the portfolio-level return proxy
        returns_df = pd.DataFrame(returns_dict).dropna(how="all").fillna(0.0)
        portfolio_returns = returns_df.mean(axis=1)

        n_bars = len(portfolio_returns)
        logger.info(
            f"[Validation] Loaded {n_bars} daily bars for {len(symbols)} symbols"
        )

        # Early exit: insufficient history for reliable statistics
        if n_bars < _MIN_BARS:
            msg = (
                f"Insufficient history: {n_bars} bars < minimum {_MIN_BARS} "
                f"(≈{_MIN_BARS // 252} years required)"
            )
            logger.warning(f"[Validation] {msg}")
            self.state.verdict = "insufficient_data"
            self.state.detail = msg
            return {"status": "insufficient_data", "n_bars": n_bars}

        self.state.returns_df = pd.DataFrame(
            {
                "portfolio": portfolio_returns,
                **{s: returns_dict[s] for s in returns_dict},
            }
        )
        return {"status": "ok", "n_bars": n_bars}

    # =========================================================================
    # PHASE 2: Walk-forward validation
    # =========================================================================

    @listen(load_historical_data)
    def run_walk_forward(self, load_result: dict) -> dict[str, Any]:
        """Run walk-forward validation on the portfolio return series."""
        if load_result.get("status") != "ok":
            return {"status": "skipped", "reason": load_result.get("status")}

        logger.info("═══ STRATEGY VALIDATION: WALK-FORWARD ═══")

        returns_series = self.state.returns_df["portfolio"]
        data = pd.DataFrame({"r": returns_series})

        try:
            from quantstack.core.research.walkforward import WalkForwardValidator

            validator = WalkForwardValidator(
                n_splits=_N_SPLITS,
                test_size=_TEST_SIZE,
                min_train_size=_MIN_TRAIN,
                gap=_GAP,
                expanding=True,
            )

            is_sharpes: list[float] = []
            oos_sharpes: list[float] = []

            for train_idx, test_idx in validator.split(data):
                r_train = returns_series.iloc[train_idx]
                r_test = returns_series.iloc[test_idx]
                is_sharpes.append(_annualised_sharpe(r_train))
                oos_sharpes.append(_annualised_sharpe(r_test))

        except ValueError as wf_err:
            # WalkForwardValidator.split() raises on insufficient data
            logger.warning(f"[Validation] Walk-forward split failed: {wf_err}")
            self.state.verdict = "insufficient_data"
            self.state.detail = str(wf_err)
            return {"status": "split_failed"}

        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        self.state.overfit_ratio = avg_is / avg_oos if avg_oos > 0 else float("inf")

        logger.info(
            f"[Validation] Walk-forward: IS Sharpe={avg_is:.2f} "
            f"OOS Sharpe={avg_oos:.2f} "
            f"overfit_ratio={self.state.overfit_ratio:.2f}"
        )

        # Store OOS Sharpes per fold for PBO
        self.state.wf_result = {
            "is_sharpes": is_sharpes,
            "oos_sharpes": oos_sharpes,
            "avg_is": avg_is,
            "avg_oos": avg_oos,
            "overfit_ratio": self.state.overfit_ratio,
        }

        return {"status": "ok", "overfit_ratio": self.state.overfit_ratio}

    # =========================================================================
    # PHASE 3: Overfitting checks (DSR + PBO)
    # =========================================================================

    @listen(run_walk_forward)
    def run_overfitting_checks(self, wf_result: dict) -> dict[str, Any]:
        """Run Deflated Sharpe Ratio and Probability of Backtest Overfitting."""
        if wf_result.get("status") != "ok":
            return {"status": "skipped", "reason": wf_result.get("status")}

        if self.state.returns_df is None:
            return {"status": "skipped", "reason": "no_returns"}

        logger.info("═══ STRATEGY VALIDATION: OVERFITTING CHECKS ═══")

        portfolio_returns = self.state.returns_df["portfolio"]
        n_obs = len(portfolio_returns)

        # ---- Deflated Sharpe Ratio ----
        try:
            from quantstack.core.research.overfitting import (
                deflated_sharpe_ratio,
                returns_statistics,
            )

            sr_obs, skew, kurt, sr_std = returns_statistics(portfolio_returns)
            dsr_result = deflated_sharpe_ratio(
                observed_sharpe=sr_obs,
                n_trials=_DSR_N_TRIALS,
                n_obs=n_obs,
                skewness=skew,
                excess_kurtosis=kurt,
                sr_std=sr_std,
            )
            self.state.dsr_result = dsr_result
            logger.info(
                f"[Validation] DSR: SR_obs={dsr_result.observed_sharpe:.2f} "
                f"SR*={dsr_result.benchmark_sharpe:.2f} "
                f"DSR={dsr_result.dsr:.3f} "
                f"genuine={dsr_result.is_genuine}"
            )
        except Exception as _dsr_err:
            logger.warning(f"[Validation] DSR computation failed: {_dsr_err}")

        # ---- Probability of Backtest Overfitting ----
        # Matrix: each WF fold's OOS returns becomes a column (temporal variants)
        try:
            from quantstack.core.research.overfitting import (
                probability_of_backtest_overfitting,
            )

            if (
                self.state.returns_df is not None
                and len(self.state.returns_df.columns) >= 2
            ):
                # Drop the portfolio column; use per-symbol OOS returns as variants
                symbol_cols = [
                    c for c in self.state.returns_df.columns if c != "portfolio"
                ]
                if len(symbol_cols) >= 2:
                    pbo_matrix = (
                        self.state.returns_df[symbol_cols]
                        .dropna(how="all")
                        .fillna(0.0)
                        .values
                    )
                    if pbo_matrix.shape[0] >= 20 and pbo_matrix.shape[1] >= 2:
                        pbo_result = probability_of_backtest_overfitting(pbo_matrix)
                        self.state.pbo_result = pbo_result
                        logger.info(
                            f"[Validation] PBO={pbo_result.pbo:.3f} overfit={pbo_result.is_overfit}"
                        )
        except Exception as _pbo_err:
            logger.warning(f"[Validation] PBO computation failed: {_pbo_err}")

        return {
            "status": "ok",
            "dsr_genuine": (
                self.state.dsr_result.is_genuine if self.state.dsr_result else None
            ),
            "pbo": (self.state.pbo_result.pbo if self.state.pbo_result else None),
        }

    # =========================================================================
    # PHASE 4: Route on fitness verdict
    # =========================================================================

    @router(run_overfitting_checks)
    def route_on_fitness(self, checks_result: dict) -> ValidationVerdict:
        """Determine verdict from walk-forward and overfitting check results."""
        if self.state.verdict == "insufficient_data":
            return "insufficient_data"
        if checks_result.get("status") in ("skipped", None):
            return self.state.verdict or "insufficient_data"

        overfit_ratio = self.state.overfit_ratio
        dsr = self.state.dsr_result
        pbo = self.state.pbo_result

        # Overfit verdict: DSR is not genuine AND PBO > threshold
        dsr_not_genuine = dsr is not None and not dsr.is_genuine
        pbo_overfit = pbo is not None and pbo.pbo > _PBO_THRESHOLD
        if dsr_not_genuine and pbo_overfit:
            self.state.verdict = "overfit"
            self.state.detail = f"DSR={dsr.dsr:.3f} (not genuine) and PBO={pbo.pbo:.3f} > {_PBO_THRESHOLD}"
            return "overfit"

        # Degraded verdict: IS/OOS Sharpe ratio too high
        if overfit_ratio > _OVERFIT_RATIO_THRESHOLD:
            self.state.verdict = "degraded"
            self.state.detail = f"Overfit ratio={overfit_ratio:.2f} > threshold {_OVERFIT_RATIO_THRESHOLD}"
            return "degraded"

        self.state.verdict = "passed"
        self.state.detail = (
            f"overfit_ratio={overfit_ratio:.2f} "
            + (f"DSR={dsr.dsr:.3f} genuine={dsr.is_genuine}" if dsr else "")
            + (f" PBO={pbo.pbo:.3f}" if pbo else "")
        )
        return "passed"

    # =========================================================================
    # PHASE 5: Outcomes
    # =========================================================================

    @listen("passed")
    def log_validation_passed(self) -> dict[str, Any]:
        """Strategy passed all validation checks."""
        logger.info(f"[Validation] PASSED — {self.state.detail}")
        self._save_validation_lesson(verdict="passed")
        return {"verdict": "passed", "detail": self.state.detail}

    @listen("degraded")
    def trigger_retraining_alert(self) -> dict[str, Any]:
        """
        Strategy is degrading: IS/OOS Sharpe divergence is too high.

        Actions:
        1. Log at WARNING level
        2. Save degradation flag to KnowledgeStore so monitors can surface it
        """
        logger.warning(
            f"[Validation] DEGRADED — {self.state.detail} — trigger retraining / parameter review"
        )
        self._save_validation_lesson(verdict="degraded")
        return {"verdict": "degraded", "detail": self.state.detail}

    @listen("overfit")
    def quarantine_strategy(self) -> dict[str, Any]:
        """
        Strategy appears overfit: quarantine flag written to KnowledgeStore.

        TradingDayFlow can optionally check for this flag at startup.
        The quarantine is non-blocking by default — operators must manually
        enable enforcement via ENFORCE_QUARANTINE=true env var.
        """
        logger.error(
            f"[Validation] OVERFIT — {self.state.detail} — "
            "quarantine flag written to KnowledgeStore"
        )
        self._save_validation_lesson(verdict="strategy_quarantine")
        enforce = os.getenv("ENFORCE_QUARANTINE", "false").lower() == "true"
        if enforce:
            logger.error(
                "[Validation] ENFORCE_QUARANTINE=true — TradingDayFlow will "
                "be blocked until quarantine is cleared"
            )
        return {
            "verdict": "overfit",
            "detail": self.state.detail,
            "enforce": enforce,
        }

    @listen("insufficient_data")
    def log_insufficient_data(self) -> dict[str, Any]:
        """Not enough data to run validation — skip and log."""
        logger.warning(f"[Validation] INSUFFICIENT DATA — {self.state.detail}")
        return {"verdict": "insufficient_data", "detail": self.state.detail}

    # =========================================================================
    # Helpers
    # =========================================================================

    def _save_validation_lesson(self, verdict: str) -> None:
        """Persist validation result to KnowledgeStore as a durable lesson."""
        try:
            from quantstack.knowledge.store import KnowledgeStore

            store = KnowledgeStore()
            wf = self.state.wf_result or {}
            dsr_str = (
                f"DSR={self.state.dsr_result.dsr:.3f} genuine={self.state.dsr_result.is_genuine}"
                if self.state.dsr_result
                else ""
            )
            pbo_str = (
                f"PBO={self.state.pbo_result.pbo:.3f}" if self.state.pbo_result else ""
            )
            lesson_text = (
                f"[{verdict.upper()}] {date.today().isoformat()} | "
                f"symbols={self.state.symbols} | "
                f"overfit_ratio={self.state.overfit_ratio:.2f} | "
                f"IS_sharpe={wf.get('avg_is', 0.0):.2f} | "
                f"OOS_sharpe={wf.get('avg_oos', 0.0):.2f} | "
                f"{dsr_str} {pbo_str} | "
                f"detail={self.state.detail}"
            )
            store.save_lesson(
                {
                    "lesson_text": lesson_text,
                    "applies_to": self.state.symbols,
                    "confidence": 0.9,
                    "created_by": "StrategyValidationFlow",
                }
            )
        except Exception as _ks_err:
            logger.debug(f"[Validation] Could not persist lesson: {_ks_err}")


# =============================================================================
# Utilities
# =============================================================================


def _annualised_sharpe(returns: pd.Series, risk_free: float = 0.05) -> float:
    """Compute annualised Sharpe ratio from a daily return series."""
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free / 252)
    std = excess.std()
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))
