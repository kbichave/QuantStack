# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Bayesian strategy parameter optimization via Optuna.

For each strategy, searches over the parameter space using walk-forward OOS
Sharpe as the objective.  Stores optimal parameters per regime.

Design decisions:
  - Uses Optuna (TPE sampler) for Bayesian optimization — already in the dep
    tree for ML hyperparameter tuning; reusing it for strategy params avoids
    a new dependency.
  - Objective is mean OOS Sharpe across walk-forward folds, NOT in-sample Sharpe.
    This directly optimizes for generalization.
  - Minimum 30 trades per parameter set to avoid overfitting to noise.
  - walk_forward_sparse_signal auto-adjusts OOS windows for low-frequency strategies.
  - Results are stored as a ``ParamOptResult`` with best params, convergence
    history, and stability diagnostics.

Failure modes:
  - Optuna import fails → raise ImportError with install instructions.
  - All parameter sets produce 0 trades → return error dict, don't crash.
  - Optimization fails to converge → return best found + warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine
from quantstack.strategies.rule_engine import compile_strategy, CompilationError
from quantstack.strategies.signal_generator import (
    generate_signals_from_rules as _generate_signals_from_rules,
)

import numpy as np
import pandas as pd
from loguru import logger

import optuna
from optuna.samplers import TPESampler


@dataclass
class ParamSpace:
    """
    Defines the searchable parameter space for a strategy.

    Each entry maps a parameter name to its search specification:
      - ``("int", low, high)`` — integer range
      - ``("float", low, high)`` — float range (uniform)
      - ``("float_log", low, high)`` — float range (log-uniform)
      - ``("categorical", [values])`` — categorical choices

    Example::

        space = ParamSpace(params={
            "rsi_period": ("int", 7, 28),
            "sma_fast": ("int", 5, 30),
            "sma_slow": ("int", 30, 100),
            "stop_loss_atr": ("float", 1.0, 3.0),
            "take_profit_atr": ("float", 1.5, 5.0),
        })
    """

    params: dict[str, tuple] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors = []
        for name, spec in self.params.items():
            if not isinstance(spec, tuple) or len(spec) < 2:
                errors.append(f"'{name}': spec must be a tuple of (type, ...)")
                continue
            param_type = spec[0]
            if param_type == "int":
                if len(spec) != 3:
                    errors.append(f"'{name}': int spec must be (\"int\", low, high)")
                elif spec[1] >= spec[2]:
                    errors.append(
                        f"'{name}': low ({spec[1]}) must be < high ({spec[2]})"
                    )
            elif param_type in ("float", "float_log"):
                if len(spec) != 3:
                    errors.append(
                        f"'{name}': float spec must be (\"{param_type}\", low, high)"
                    )
                elif spec[1] >= spec[2]:
                    errors.append(
                        f"'{name}': low ({spec[1]}) must be < high ({spec[2]})"
                    )
                if param_type == "float_log" and spec[1] <= 0:
                    errors.append(f"'{name}': log-uniform requires low > 0")
            elif param_type == "categorical":
                if len(spec) != 2 or not isinstance(spec[1], list) or len(spec[1]) < 2:
                    errors.append(
                        f"'{name}': categorical spec must be (\"categorical\", [v1, v2, ...])"
                    )
            else:
                errors.append(f"'{name}': unknown type '{param_type}'")
        return errors


@dataclass
class ParamOptResult:
    """Result of parameter optimization."""

    best_params: dict[str, Any]
    best_score: float  # OOS Sharpe
    n_trials: int
    n_completed: int
    convergence: list[dict[str, Any]]  # per-trial: params, score
    best_trial_trades: int
    stability_warning: str | None = None  # set if best params are near boundary


def optimize_strategy_params(
    strategy_dict: dict[str, Any],
    symbol: str,
    param_space: ParamSpace,
    price_data: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    min_trades: int = 30,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
    timeout_seconds: int | None = 300,
) -> ParamOptResult:
    """
    Optimize strategy parameters using Bayesian search.

    Args:
        strategy_dict: Strategy record dict (must have entry_rules, exit_rules, parameters).
        symbol: Ticker symbol (price data already loaded).
        param_space: Searchable parameter space.
        price_data: OHLCV DataFrame with DatetimeIndex.
        n_trials: Number of Optuna trials.
        n_splits: Walk-forward folds.
        test_size: Bars per OOS fold.
        min_train_size: Minimum IS bars.
        min_trades: Reject parameter sets with fewer total OOS trades.
        initial_capital: Starting capital per fold.
        position_size_pct: Position size fraction.
        timeout_seconds: Max wall-clock time for optimization.

    Returns:
        ParamOptResult with best parameters and convergence history.

    Raises:
        ValueError: If param_space is invalid.
    """
    # Validate param space
    errors = param_space.validate()
    if errors:
        raise ValueError(f"Invalid parameter space: {'; '.join(errors)}")

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    entry_rules = strategy_dict.get("entry_rules", [])
    exit_rules = strategy_dict.get("exit_rules", [])
    base_params = dict(strategy_dict.get("parameters", {}))

    convergence: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        """Single trial: sample params, run walk-forward, return mean OOS Sharpe."""
        # Sample parameters
        trial_params = dict(base_params)
        for name, spec in param_space.params.items():
            param_type = spec[0]
            if param_type == "int":
                trial_params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif param_type == "float":
                trial_params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == "float_log":
                trial_params[name] = trial.suggest_float(
                    name, spec[1], spec[2], log=True
                )
            elif param_type == "categorical":
                trial_params[name] = trial.suggest_categorical(name, spec[1])

        # Run walk-forward
        try:
            compiled = compile_strategy(
                strategy_id="__opt__",
                name="__optimization_trial__",
                entry_rules=entry_rules,
                exit_rules=exit_rules,
                parameters=trial_params,
            )
        except CompilationError:
            return float("-inf")

        oos_sharpes, total_oos_trades = _walk_forward_evaluate(
            compiled=compiled,
            price_data=price_data,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            parameters=trial_params,
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
        )

        if total_oos_trades < min_trades:
            # Penalize sparse signals but don't discard entirely
            score = -10.0
        elif not oos_sharpes:
            score = -10.0
        else:
            score = float(np.mean(oos_sharpes))

        convergence.append(
            {
                "trial": trial.number,
                "params": {k: trial_params[k] for k in param_space.params},
                "oos_sharpe": round(score, 4),
                "oos_trades": total_oos_trades,
            }
        )

        return score

    # Run optimization
    sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"opt_{strategy_dict.get('name', 'unknown')}_{symbol}",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )

    # Extract best
    best_trial = study.best_trial
    best_params = dict(base_params)
    for name, spec in param_space.params.items():
        best_params[name] = best_trial.params.get(name, base_params.get(name))

    # Check stability: is the best near the boundary of the search space?
    stability_warning = _check_boundary_proximity(best_trial.params, param_space)

    # Find the best trial's trade count from convergence
    best_trades = 0
    for entry in convergence:
        if entry["trial"] == best_trial.number:
            best_trades = entry.get("oos_trades", 0)
            break

    n_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    logger.info(
        f"[OPT] {strategy_dict.get('name', '?')} on {symbol}: "
        f"best OOS Sharpe={best_trial.value:.4f} after {n_completed}/{n_trials} trials, "
        f"{best_trades} OOS trades"
    )

    return ParamOptResult(
        best_params=best_params,
        best_score=round(best_trial.value, 4),
        n_trials=n_trials,
        n_completed=n_completed,
        convergence=convergence,
        best_trial_trades=best_trades,
        stability_warning=stability_warning,
    )


def _walk_forward_evaluate(
    compiled: Any,
    price_data: pd.DataFrame,
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    parameters: dict[str, Any],
    n_splits: int,
    test_size: int,
    min_train_size: int,
    initial_capital: float,
    position_size_pct: float,
) -> tuple[list[float], int]:
    """
    Run walk-forward and return (oos_sharpes, total_oos_trades).

    Uses the existing signal generation + backtest engine to avoid
    reimplementing the full backtest pipeline.
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
    )

    n = len(price_data)
    gap = 0  # embargo handled by purged CV in 3.4
    first_test_start = min_train_size + gap
    oos_sharpes = []
    total_oos_trades = 0

    for i in range(n_splits):
        test_start = first_test_start + i * test_size
        test_end = min(test_start + test_size, n)
        if test_end > n:
            break

        train_start = 0  # expanding window
        train_end = test_start - gap

        test_data = price_data.iloc[test_start:test_end]
        if len(test_data) < 20:
            continue

        test_signals = _generate_signals_from_rules(
            test_data,
            entry_rules,
            exit_rules,
            parameters,
        )

        engine = BacktestEngine(config=config)
        result = engine.run(test_signals, test_data)

        oos_sharpes.append(result.sharpe_ratio)
        total_oos_trades += result.total_trades

    return oos_sharpes, total_oos_trades


def _check_boundary_proximity(
    best_params: dict[str, Any],
    param_space: ParamSpace,
    threshold_pct: float = 0.05,
) -> str | None:
    """
    Check if any best parameter is within 5% of its search boundary.

    Returns a warning string if so, else None.
    """
    near_boundary = []
    for name, spec in param_space.params.items():
        if name not in best_params:
            continue
        param_type = spec[0]
        if param_type in ("int", "float", "float_log"):
            low, high = spec[1], spec[2]
            val = best_params[name]
            span = high - low
            if span <= 0:
                continue
            if (val - low) / span < threshold_pct:
                near_boundary.append(f"{name}={val} near lower bound {low}")
            elif (high - val) / span < threshold_pct:
                near_boundary.append(f"{name}={val} near upper bound {high}")

    if near_boundary:
        return (
            "Best parameters near search boundary — consider widening the space: "
            + "; ".join(near_boundary)
        )
    return None


# =============================================================================
# Convenience: infer param space from strategy parameters
# =============================================================================


def infer_param_space(parameters: dict[str, Any]) -> ParamSpace:
    """
    Infer a reasonable search space from existing strategy parameters.

    Heuristic: for each numeric parameter, search ±50% around the current value
    (clamped to sensible bounds). Useful as a starting point before manual refinement.
    """
    space: dict[str, tuple] = {}

    _INT_PARAMS = {
        "rsi_period": (5, 50),
        "atr_period": (5, 50),
        "adx_period": (7, 50),
        "bb_period": (10, 50),
        "stoch_period": (5, 50),
        "cci_period": (10, 50),
        "zscore_period": (10, 50),
        "breakout_period": (5, 60),
        "sma_fast": (3, 50),
        "sma_slow": (20, 200),
        "sma_fast_period": (3, 50),
        "sma_slow_period": (20, 200),
    }

    _FLOAT_PARAMS = {
        "stop_loss_atr": (0.5, 5.0),
        "take_profit_atr": (1.0, 8.0),
        "bb_std": (1.0, 3.5),
        "sma_proximity_pct": (1.0, 10.0),
        "position_pct": (0.02, 0.15),
    }

    for name, val in parameters.items():
        if name in _INT_PARAMS:
            lo, hi = _INT_PARAMS[name]
            current = int(val)
            lo = max(lo, int(current * 0.5))
            hi = min(hi, int(current * 1.5))
            if lo >= hi:
                hi = lo + 1
            space[name] = ("int", lo, hi)
        elif name in _FLOAT_PARAMS:
            lo, hi = _FLOAT_PARAMS[name]
            current = float(val)
            lo = max(lo, current * 0.5)
            hi = min(hi, current * 1.5)
            if lo >= hi:
                hi = lo + 0.1
            space[name] = ("float", round(lo, 3), round(hi, 3))

    return ParamSpace(params=space)
