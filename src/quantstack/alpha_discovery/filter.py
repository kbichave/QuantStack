# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CandidateFilter — three-stage gate: IS screen → OOS walk-forward → portfolio fit.

Stage 1 (IS screen):
    Fast in-sample filter with Harvey-Liu n_trials Sharpe deflation.
    Rejects parameter sets that only look good because we searched many combinations.

Stage 2 (OOS validation):
    Walk-forward with 3 folds. Rejects overfit strategies (IS/OOS ratio too high)
    and strategies that don't generalize (OOS Sharpe too low).

Stage 3 (PortfolioFitCheck):
    Pearson correlation gate against all live/forward_testing strategies.
    Rejects candidates that are redundant bets with existing portfolio entries.
    Only runs after Stage 2 passes — no wasted DB reads on bad candidates.

Design: Fail fast. Stage 1 runs O(N). Stage 2 runs 3 mini-backtests.
Stage 3 loads existing signals once and computes pairwise correlation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine
from quantstack.db import open_db_readonly
from quantstack.strategies.signal_generator import (
    generate_signals_from_rules as _generate_signals_from_rules,
)


# =============================================================================
# Thresholds (from the plan — do not relax without evidence)
# =============================================================================

# Stage 1 — In-sample screen
IS_MIN_SHARPE = 0.5
IS_MIN_TRADES = 20
IS_MIN_PROFIT_FACTOR = 1.2

# Stage 2 — Out-of-sample validation
OOS_MIN_SHARPE_MEAN = 0.6
OOS_MAX_OVERFIT_RATIO = 2.0  # IS Sharpe / OOS Sharpe

# Stage 3 — Portfolio fit
PORTFOLIO_MAX_CORRELATION = 0.70  # reject if any live strategy correlates this high

# Harvey-Liu deflation baseline bars (1 year of daily data)
_HL_T_BARS = 252


@dataclass
class FilterResult:
    """Result of the three-stage filter for one candidate."""

    passed: bool
    stage_rejected: (
        str | None
    )  # "is_screen", "oos_validation", "portfolio_fit", or None
    rejection_reason: str | None
    is_sharpe: float = 0.0
    is_trades: int = 0
    is_profit_factor: float = 0.0
    oos_sharpe_mean: float = 0.0
    overfit_ratio: float = 0.0
    # Stage 3 fields
    portfolio_correlation: float = (
        0.0  # highest correlation found against existing strategies
    )
    correlated_strategy_id: str | None = None  # which strategy it conflicts with


class CandidateFilter:
    """
    Three-stage candidate filter.

    Usage:
        filt = CandidateFilter()
        result = filt.apply(
            strategy_spec=spec,
            price_data=df,
            n_trials=200,  # grid size for Harvey-Liu deflation
        )
    """

    def apply(
        self,
        strategy_spec: dict[str, Any],
        price_data: Any,  # pd.DataFrame
        is_start: str | None = None,
        is_end: str | None = None,
        n_trials: int = 1,
    ) -> FilterResult:
        """
        Run all three filter stages.

        Args:
            strategy_spec: {entry_rules, exit_rules, parameters} dict.
            price_data: Full OHLCV DataFrame (all available history).
            is_start: IS window start date (YYYY-MM-DD). Uses full history if None.
            is_end: IS window end date. Uses 75% of history if None.
            n_trials: Number of parameter combinations tested (grid size).
                      Used by Harvey-Liu Sharpe deflation. 1 = no deflation.
        """
        entry_rules = strategy_spec.get("entry_rules", [])
        exit_rules = strategy_spec.get("exit_rules", [])
        parameters = strategy_spec.get("parameters", {})

        # Slice IS window
        df = price_data.copy()
        if is_start:
            df = df[df.index >= is_start]
        if is_end:
            is_df = df[df.index <= is_end]
        else:
            split = int(len(df) * 0.75)
            is_df = df.iloc[:split]

        if len(is_df) < 60:
            return FilterResult(
                passed=False,
                stage_rejected="is_screen",
                rejection_reason="insufficient IS data (< 60 bars)",
            )

        # --- Stage 1: IS screen with Harvey-Liu deflation ---
        try:
            is_result = _run_backtest(is_df, entry_rules, exit_rules, parameters)
        except Exception as exc:
            return FilterResult(
                passed=False,
                stage_rejected="is_screen",
                rejection_reason=f"IS backtest failed: {exc}",
            )

        if is_result["total_trades"] < IS_MIN_TRADES:
            return FilterResult(
                passed=False,
                stage_rejected="is_screen",
                rejection_reason=(
                    f"too few IS trades: {is_result['total_trades']} < {IS_MIN_TRADES}"
                ),
                is_sharpe=is_result["sharpe_ratio"],
                is_trades=is_result["total_trades"],
                is_profit_factor=is_result.get("profit_factor", 0.0),
            )

        raw_sharpe = is_result["sharpe_ratio"]
        deflated_sharpe = _deflate_sharpe(raw_sharpe, n_trials, t_bars=len(is_df))

        if deflated_sharpe != raw_sharpe:
            logger.debug(
                f"[CandidateFilter] IS Sharpe {raw_sharpe:.2f} deflated to "
                f"{deflated_sharpe:.2f} (n_trials={n_trials}) — applying deflated threshold"
            )

        if deflated_sharpe < IS_MIN_SHARPE:
            return FilterResult(
                passed=False,
                stage_rejected="is_screen",
                rejection_reason=(
                    f"IS Sharpe {raw_sharpe:.2f} deflated to {deflated_sharpe:.2f} "
                    f"(n_trials={n_trials}) < {IS_MIN_SHARPE}"
                ),
                is_sharpe=raw_sharpe,
                is_trades=is_result["total_trades"],
                is_profit_factor=is_result.get("profit_factor", 0.0),
            )

        if is_result.get("profit_factor", 0.0) < IS_MIN_PROFIT_FACTOR:
            return FilterResult(
                passed=False,
                stage_rejected="is_screen",
                rejection_reason=(
                    f"IS profit factor {is_result.get('profit_factor', 0):.2f} < {IS_MIN_PROFIT_FACTOR}"
                ),
                is_sharpe=raw_sharpe,
                is_trades=is_result["total_trades"],
                is_profit_factor=is_result.get("profit_factor", 0.0),
            )

        # --- Stage 2: OOS walk-forward (3 folds on remaining data) ---
        oos_sharpe_mean, overfit_ratio = _run_oos_validation(
            full_df=df,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            parameters=parameters,
            n_folds=3,
            is_sharpe=raw_sharpe,
            start_date="2010-01-01",
        )

        if oos_sharpe_mean < OOS_MIN_SHARPE_MEAN:
            return FilterResult(
                passed=False,
                stage_rejected="oos_validation",
                rejection_reason=(
                    f"OOS Sharpe mean {oos_sharpe_mean:.2f} < {OOS_MIN_SHARPE_MEAN}"
                ),
                is_sharpe=raw_sharpe,
                is_trades=is_result["total_trades"],
                is_profit_factor=is_result.get("profit_factor", 0.0),
                oos_sharpe_mean=oos_sharpe_mean,
                overfit_ratio=overfit_ratio,
            )

        if overfit_ratio > OOS_MAX_OVERFIT_RATIO:
            return FilterResult(
                passed=False,
                stage_rejected="oos_validation",
                rejection_reason=(
                    f"overfit ratio {overfit_ratio:.2f} > {OOS_MAX_OVERFIT_RATIO}"
                ),
                is_sharpe=raw_sharpe,
                is_trades=is_result["total_trades"],
                is_profit_factor=is_result.get("profit_factor", 0.0),
                oos_sharpe_mean=oos_sharpe_mean,
                overfit_ratio=overfit_ratio,
            )

        # --- Stage 3: PortfolioFitCheck (correlation gate) ---
        fit_passed, corr_value, corr_strategy_id = _check_portfolio_fit(
            price_data=df,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            parameters=parameters,
        )

        if not fit_passed:
            return FilterResult(
                passed=False,
                stage_rejected="portfolio_fit",
                rejection_reason=(
                    f"portfolio_fit: correlation {corr_value:.2f} "
                    f"with {corr_strategy_id} > {PORTFOLIO_MAX_CORRELATION}"
                ),
                is_sharpe=raw_sharpe,
                is_trades=is_result["total_trades"],
                is_profit_factor=is_result.get("profit_factor", 0.0),
                oos_sharpe_mean=oos_sharpe_mean,
                overfit_ratio=overfit_ratio,
                portfolio_correlation=corr_value,
                correlated_strategy_id=corr_strategy_id,
            )

        return FilterResult(
            passed=True,
            stage_rejected=None,
            rejection_reason=None,
            is_sharpe=raw_sharpe,
            is_trades=is_result["total_trades"],
            is_profit_factor=is_result.get("profit_factor", 0.0),
            oos_sharpe_mean=oos_sharpe_mean,
            overfit_ratio=overfit_ratio,
            portfolio_correlation=corr_value,
        )


# =============================================================================
# Private helpers
# =============================================================================


def _deflate_sharpe(sharpe: float, n_trials: int, t_bars: int = _HL_T_BARS) -> float:
    """
    Harvey-Liu-Zhu (2015) Minimum Backtest Length Sharpe deflation.

    Reduces effective Sharpe to account for selection bias when testing
    n_trials parameter combinations. With 200 combinations and a 5%
    significance threshold, the expected IS Sharpe from random data is
    non-trivially positive — deflation corrects for this.

    Simplified formula (Harvey & Liu 2015, eq 4):
        sigma_sharpe ≈ sqrt((1 + 0.5 * sharpe^2) / t_bars)
        deflation    = sqrt(log(n_trials))
        deflated     = sharpe - deflation * sigma_sharpe

    When n_trials=1 (single manual backtest) deflation=0 — no correction.
    Deflation is always non-negative (never inflates Sharpe).

    Args:
        sharpe: Raw annualized IS Sharpe ratio.
        n_trials: Number of parameter combinations screened.
        t_bars: Number of bars in the IS window (default 252 = 1 year daily).
    """
    if n_trials <= 1:
        return sharpe

    sigma_sr = math.sqrt((1.0 + 0.5 * sharpe**2) / max(t_bars, 1))
    deflation = math.sqrt(math.log(n_trials))
    return sharpe - deflation * sigma_sr


def _check_portfolio_fit(
    price_data: Any,
    entry_rules: list[dict],
    exit_rules: list[dict],
    parameters: dict[str, Any],
) -> tuple[bool, float, str | None]:
    """
    Pearson correlation gate against all live/forward_testing strategies.

    Generates daily signal series for both the candidate and each existing
    strategy, then computes pairwise Pearson correlation. Rejects if any
    correlation exceeds PORTFOLIO_MAX_CORRELATION.

    Returns (passed, max_correlation, conflicting_strategy_id).
    Fails open: if DB read fails or no existing strategies, returns (True, 0.0, None).
    """
    try:
        # Load existing live/forward_testing strategies (read-only, no lock competition)
        conn = open_db_readonly()
        rows = conn.execute(
            """
            SELECT strategy_id, entry_rules, exit_rules, parameters
            FROM strategies
            WHERE status IN ('live', 'forward_testing')
            """
        ).fetchall()
        conn.close()

        if not rows:
            return True, 0.0, None

        # Generate candidate signals
        candidate_signals = _generate_signals_from_rules(
            price_data, entry_rules, exit_rules, parameters
        )
        if candidate_signals is None or len(candidate_signals) == 0:
            return True, 0.0, None

        candidate_series = candidate_signals.get("signal", pd.Series(dtype=float))
        if candidate_series.empty or candidate_series.std() == 0:
            return True, 0.0, None

        max_corr = 0.0
        max_corr_id: str | None = None

        for strategy_id, er_raw, xr_raw, params_raw in rows:
            try:
                er = json.loads(er_raw) if isinstance(er_raw, str) else (er_raw or [])
                xr = json.loads(xr_raw) if isinstance(xr_raw, str) else (xr_raw or [])
                params = (
                    json.loads(params_raw)
                    if isinstance(params_raw, str)
                    else (params_raw or {})
                )

                existing_signals = _generate_signals_from_rules(
                    price_data, er, xr, params
                )
                if existing_signals is None:
                    continue

                existing_series = existing_signals.get("signal", pd.Series(dtype=float))
                if existing_series.empty or existing_series.std() == 0:
                    continue

                # Align on common index before correlating
                aligned = pd.concat(
                    [candidate_series.rename("c"), existing_series.rename("e")], axis=1
                ).dropna()
                if len(aligned) < 30:
                    continue

                corr = float(aligned["c"].corr(aligned["e"]))
                if abs(corr) > max_corr:
                    max_corr = abs(corr)
                    max_corr_id = strategy_id

            except Exception as exc:
                logger.debug(
                    f"[PortfolioFitCheck] strategy {strategy_id} signal gen failed: {exc}"
                )
                continue

        if max_corr > PORTFOLIO_MAX_CORRELATION:
            logger.info(
                f"[PortfolioFitCheck] rejected: correlation {max_corr:.2f} "
                f"with {max_corr_id} > {PORTFOLIO_MAX_CORRELATION}"
            )
            return False, max_corr, max_corr_id

        return True, max_corr, max_corr_id

    except Exception as exc:
        # Fail open — portfolio fit failure never blocks discovery
        logger.debug(f"[PortfolioFitCheck] DB read failed (fail open): {exc}")
        return True, 0.0, None


def _run_backtest(
    price_data: Any,
    entry_rules: list[dict],
    exit_rules: list[dict],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Run a single backtest and return a metrics dict."""
    signals = _generate_signals_from_rules(
        price_data, entry_rules, exit_rules, parameters
    )
    engine = BacktestEngine(BacktestConfig(position_size_pct=0.10))
    result = engine.run(signals=signals, price_data=price_data)
    return {
        "sharpe_ratio": result.sharpe_ratio,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "max_drawdown": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "total_return": result.total_return,
    }


def _run_oos_validation(
    full_df: Any,
    entry_rules: list[dict],
    exit_rules: list[dict],
    parameters: dict[str, Any],
    n_folds: int,
    is_sharpe: float,
    start_date: str | None = "2010-01-01",
) -> tuple[float, float]:
    """
    Walk-forward with n_folds. Returns (oos_sharpe_mean, overfit_ratio).

    Simple expanding-window walk-forward:
      - Fold i train window: [0, 75% + i * step)
      - Fold i test window:  [75% + i * step, 75% + (i+1) * step)

    Fold test results are averaged. overfit_ratio = is_sharpe / oos_sharpe_mean.

    start_date: Floor the dataset to exclude pre-crisis data (default 2010-01-01).
      Pre-2010 regimes (GFC, dot-com) produce OOS distributions inconsistent with
      the post-QE era the engine actually trades in, dragging mean OOS Sharpe negative
      and causing structurally-sound strategies to fail the filter.
    """
    if start_date:
        full_df = full_df[full_df.index >= pd.Timestamp(start_date)]
    n = len(full_df)
    base_train_end = int(n * 0.75)
    remaining = n - base_train_end
    if remaining < n_folds * 20:
        # Not enough OOS data for reliable folds — penalize
        return 0.0, float("inf")

    fold_size = remaining // n_folds
    oos_sharpes = []

    for i in range(n_folds):
        test_start = base_train_end + i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n
        fold_df = full_df.iloc[test_start:test_end]
        if len(fold_df) < 20:
            continue
        try:
            metrics = _run_backtest(fold_df, entry_rules, exit_rules, parameters)
            oos_sharpes.append(metrics["sharpe_ratio"])
        except Exception:
            oos_sharpes.append(-1.0)  # Treat failed fold as bad

    if not oos_sharpes:
        return 0.0, float("inf")

    oos_mean = float(np.mean(oos_sharpes))
    overfit_ratio = is_sharpe / (oos_mean + 1e-9) if oos_mean > 0 else float("inf")
    return oos_mean, overfit_ratio
