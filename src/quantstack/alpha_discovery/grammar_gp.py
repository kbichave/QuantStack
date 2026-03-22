# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Grammar-Guided Genetic Programming engine for alpha template discovery.

Unlike grid search (which permutes known templates x parameter grids),
GP discovers novel rule COMBINATIONS through crossover and mutation:
  - Crossover: combine entry rules from two parents (e.g., RSI entry + Bollinger exit)
  - Mutation: swap an indicator, change a threshold, add/remove a rule
  - Selection: tournament selection based on OOS Sharpe (not IS — prevents overfitting)

The grammar constrains the search space to only produce valid strategy JSON
that ``_generate_signals_from_rules()`` can evaluate. Every individual is a
complete strategy spec: ``{entry_rules, exit_rules, parameters}``.

Integration:
    Called by ``AlphaDiscoveryEngine._process_symbol()`` after grid search.
    Grid survivors seed the initial population so GP starts from structurally
    interesting candidates rather than pure random.

Design reference: AlphaCFG (arxiv 2601.22119), HARLA (Frontiers CS 2025).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.strategies.signal_generator import generate_signals_from_rules


# =============================================================================
# Grammar — defines the valid search space for strategy rules
# =============================================================================

GRAMMAR: dict[str, Any] = {
    "indicators": [
        "rsi",
        "sma_fast",
        "sma_slow",
        "adx",
        "atr",
        "stoch_k",
        "cci",
        "bb_pct",
        "zscore",
        "regime",
        "atr_percentile",
        "fund_pe_ratio",
        "fund_roe",
        "yield_curve_10y2y",
        "earn_days_to",
        "flow_insider_net_90d",
    ],
    "conditions": ["above", "below", "crosses_above", "crosses_below", "between"],
    "rule_types": ["plain", "prerequisite", "confirmation"],
    "exit_types": ["time_stop", "take_profit", "stop_loss"],
    "regime_values": ["trending_up", "trending_down", "ranging"],
    "value_ranges": {
        "rsi": (10.0, 90.0),
        "adx": (15.0, 50.0),
        "stoch_k": (10.0, 90.0),
        "cci": (-200.0, 200.0),
        "bb_pct": (0.0, 1.0),
        "zscore": (-3.0, 3.0),
        "atr_percentile": (10.0, 90.0),
        "fund_pe_ratio": (5.0, 50.0),
        "fund_roe": (0.0, 0.5),
        "yield_curve_10y2y": (-1.0, 3.0),
        "earn_days_to": (1.0, 30.0),
        "flow_insider_net_90d": (-1e6, 1e6),
    },
    "exit_params": {
        "time_stop_days": (3, 20),
        "take_profit_atr": (1.0, 5.0),
        "stop_loss_atr": (0.5, 3.0),
    },
}

# Indicators that compare against a column name rather than a numeric value
_COLUMN_REF_INDICATORS: dict[str, str] = {"sma_fast": "sma_slow"}

# Indicators where the condition must compare against a regime string
_REGIME_INDICATORS: set[str] = {"regime"}

# Special indicators with fixed value — crossover/mutation preserve the value
_SPECIAL_INDICATORS: dict[str, dict[str, Any]] = {
    "sma_crossover": {"conditions": ["crosses_above", "crosses_below"], "value": 0},
    "breakout": {"conditions": ["above", "below"], "value": 0},
}

# All valid numeric conditions (subset of GRAMMAR["conditions"] excluding "between")
_NUMERIC_CONDITIONS = ["above", "below", "crosses_above", "crosses_below"]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GPConfig:
    """Hyperparameters for the GP evolution loop."""

    population_size: int = 50
    n_generations: int = 20
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    tournament_size: int = 5
    elite_count: int = 5
    # Rule count bounds per individual
    min_entry_rules: int = 2
    max_entry_rules: int = 4
    min_exit_rules: int = 2
    max_exit_rules: int = 3
    # OOS split for fitness evaluation
    oos_fraction: float = 0.3
    # Wall-clock budget — GP aborts if exceeded (seconds)
    wall_clock_budget_seconds: float = 180.0
    # Early stopping: halt if best fitness stagnates for this many generations
    stagnation_limit: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size > 100:
            self.population_size = 100
        if self.elite_count >= self.population_size:
            self.elite_count = max(1, self.population_size // 5)


# =============================================================================
# GrammarGP — the evolution engine
# =============================================================================


class GrammarGP:
    """
    Grammar-guided GP for alpha template discovery.

    Evolves a population of strategy individuals using crossover and mutation
    operators constrained by ``GRAMMAR``. Fitness is OOS Sharpe ratio
    (walk-forward 70/30 split) to prevent overfitting.

    Produces valid strategy specs (entry_rules + exit_rules + parameters)
    that ``_generate_signals_from_rules()`` can evaluate directly.
    """

    def __init__(self, config: GPConfig | None = None) -> None:
        self._cfg = config or GPConfig()
        self._rng = random.Random(self._cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        seed_population: list[dict[str, Any]],
        price_data: pd.DataFrame,
        n_prior_trials: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Evolve a population of strategies over ``n_generations``.

        Args:
            seed_population: Strategy specs from grid search that had
                sufficient trades (IS_MIN_TRADES). Used to seed the GP
                so evolution starts from structurally interesting candidates
                rather than pure random.
            price_data: Full OHLCV DataFrame (must have >= 252 rows).
            n_prior_trials: Candidates already evaluated by grid search.
                Added to GP eval count for logging; does not affect GP
                internals (fitness is OOS Sharpe, not deflated).

        Returns:
            (survivors, n_evaluations) where survivors are strategy spec
            dicts sorted by descending OOS Sharpe, each with keys
            ``entry_rules``, ``exit_rules``, ``parameters``.
        """
        cfg = self._cfg
        rng = self._rng
        start_time = time.monotonic()
        n_evals = 0

        if len(price_data) < 100:
            logger.warning("[GrammarGP] insufficient price data for GP evolution")
            return [], 0

        # Build initial population: seeds first, random fill for the rest
        population = self._init_population(seed_population)

        # Evaluate initial population
        fitnesses = []
        for ind in population:
            fitness = self._evaluate_fitness(ind, price_data)
            fitnesses.append(fitness)
            n_evals += 1

        best_ever = max(fitnesses) if fitnesses else -1.0
        stagnation_count = 0

        logger.info(
            f"[GrammarGP] gen=0 best_fitness={best_ever:.3f} "
            f"pop={len(population)} seeds={len(seed_population)}"
        )

        # Main evolutionary loop
        for gen in range(1, cfg.n_generations + 1):
            # Wall-clock budget check
            elapsed = time.monotonic() - start_time
            if elapsed >= cfg.wall_clock_budget_seconds:
                logger.info(
                    f"[GrammarGP] wall-clock budget exhausted at gen={gen} "
                    f"({elapsed:.1f}s)"
                )
                break

            next_pop: list[dict] = []

            # Elitism: carry forward top individuals unchanged
            elite_indices = np.argsort(fitnesses)[-cfg.elite_count :]
            for idx in elite_indices:
                next_pop.append(copy.deepcopy(population[idx]))

            # Fill remaining slots with crossover + mutation
            while len(next_pop) < cfg.population_size:
                if rng.random() < cfg.crossover_rate:
                    parent_a = self._tournament_select(population, fitnesses)
                    parent_b = self._tournament_select(population, fitnesses)
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = copy.deepcopy(
                        self._tournament_select(population, fitnesses)
                    )

                if rng.random() < cfg.mutation_rate:
                    child = self._mutate(child)

                next_pop.append(child)

            population = next_pop

            # Evaluate new population
            fitnesses = []
            for ind in population:
                fitness = self._evaluate_fitness(ind, price_data)
                fitnesses.append(fitness)
                n_evals += 1

            gen_best = max(fitnesses) if fitnesses else -1.0
            if gen_best > best_ever:
                best_ever = gen_best
                stagnation_count = 0
            else:
                stagnation_count += 1

            if gen % 5 == 0 or gen == cfg.n_generations:
                logger.info(
                    f"[GrammarGP] gen={gen} best_fitness={gen_best:.3f} "
                    f"best_ever={best_ever:.3f} evals={n_evals}"
                )

            if stagnation_count >= cfg.stagnation_limit:
                logger.info(
                    f"[GrammarGP] early stop — stagnation for "
                    f"{stagnation_count} generations at gen={gen}"
                )
                break

        # Return survivors with positive OOS Sharpe, sorted descending
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        survivors = []
        for ind, fitness in ranked:
            if fitness <= 0:
                break
            survivors.append(copy.deepcopy(ind))

        elapsed = time.monotonic() - start_time
        logger.info(
            f"[GrammarGP] done — survivors={len(survivors)} "
            f"total_evals={n_evals} elapsed={elapsed:.1f}s "
            f"best_oos_sharpe={best_ever:.3f}"
        )
        return survivors, n_evals

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _init_population(self, seeds: list[dict]) -> list[dict]:
        """Build initial population from seeds + random individuals."""
        population: list[dict] = []

        # Seed from grid search survivors (structurally interesting)
        for seed in seeds[: self._cfg.population_size]:
            population.append(copy.deepcopy(seed))

        # Fill remaining with random grammar individuals
        while len(population) < self._cfg.population_size:
            population.append(self._random_individual())

        return population[: self._cfg.population_size]

    def _random_individual(self) -> dict[str, Any]:
        """Generate a random strategy from the grammar."""
        rng = self._rng
        n_entry = rng.randint(self._cfg.min_entry_rules, self._cfg.max_entry_rules)
        n_exit = rng.randint(self._cfg.min_exit_rules, self._cfg.max_exit_rules)

        entry_rules = [self._random_entry_rule() for _ in range(n_entry)]
        # Ensure at least one prerequisite so the structured AND gate fires
        if not any(r.get("type") == "prerequisite" for r in entry_rules):
            entry_rules[0]["type"] = "prerequisite"

        exit_rules = self._random_exit_rules(n_exit)

        parameters: dict[str, Any] = {
            "sma_fast": rng.choice([5, 8, 10, 13, 20]),
            "sma_slow": rng.choice([20, 30, 50, 100, 200]),
            "rsi_period": rng.choice([7, 10, 14, 21]),
            "bb_period": rng.choice([15, 20, 30]),
            "bb_std": round(rng.uniform(1.5, 2.5), 1),
        }

        return {
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "parameters": parameters,
        }

    def _random_entry_rule(self) -> dict[str, Any]:
        """Generate a single random entry rule from the grammar."""
        rng = self._rng
        indicator = rng.choice(GRAMMAR["indicators"])
        rule_type = rng.choice(GRAMMAR["rule_types"])

        # Regime rules use string equality
        if indicator in _REGIME_INDICATORS:
            return {
                "indicator": indicator,
                "condition": rng.choice(["above", "below"]),
                "value": rng.choice(GRAMMAR["regime_values"]),
                "type": rule_type,
            }

        # Column-reference indicators (e.g., sma_fast > sma_slow)
        if indicator in _COLUMN_REF_INDICATORS:
            return {
                "indicator": indicator,
                "condition": rng.choice(_NUMERIC_CONDITIONS),
                "value": _COLUMN_REF_INDICATORS[indicator],
                "type": rule_type,
            }

        # Special indicators with fixed value (sma_crossover, breakout)
        if indicator in _SPECIAL_INDICATORS:
            spec = _SPECIAL_INDICATORS[indicator]
            return {
                "indicator": indicator,
                "condition": rng.choice(spec["conditions"]),
                "value": spec["value"],
                "type": rule_type,
            }

        # Numeric indicators — sample from valid range
        if indicator in GRAMMAR["value_ranges"]:
            lo, hi = GRAMMAR["value_ranges"][indicator]
            value = round(rng.uniform(lo, hi), 2)
        else:
            value = round(rng.uniform(0, 100), 2)

        return {
            "indicator": indicator,
            "condition": rng.choice(_NUMERIC_CONDITIONS),
            "value": value,
            "type": rule_type,
        }

    def _random_exit_rules(self, n: int) -> list[dict[str, Any]]:
        """Generate exit rules — always includes a stop loss."""
        rng = self._rng
        ep = GRAMMAR["exit_params"]
        rules: list[dict[str, Any]] = []

        # Mandatory stop loss
        sl_lo, sl_hi = ep["stop_loss_atr"]
        rules.append(
            {
                "type": "stop_loss",
                "atr_multiple": round(rng.uniform(sl_lo, sl_hi), 1),
            }
        )

        # Fill remaining with take_profit and/or time_stop
        for _ in range(n - 1):
            exit_type = rng.choice(["take_profit", "time_stop"])
            if exit_type == "take_profit":
                tp_lo, tp_hi = ep["take_profit_atr"]
                rules.append(
                    {
                        "type": "take_profit",
                        "atr_multiple": round(rng.uniform(tp_lo, tp_hi), 1),
                    }
                )
            else:
                ts_lo, ts_hi = ep["time_stop_days"]
                rules.append(
                    {
                        "type": "time_stop",
                        "days": rng.randint(ts_lo, ts_hi),
                    }
                )

        return _dedup_structural_exits(rules)

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _crossover(self, parent_a: dict, parent_b: dict) -> dict:
        """
        Uniform crossover on rules: each entry rule drawn from either parent.

        Exit rules from a randomly chosen parent. Parameters blended
        (average of numeric values, random pick for non-numeric).
        """
        rng = self._rng
        a_entry = parent_a.get("entry_rules", [])
        b_entry = parent_b.get("entry_rules", [])

        # Pool all entry rules, pick a random subset
        pool = copy.deepcopy(a_entry) + copy.deepcopy(b_entry)
        if not pool:
            pool = [self._random_entry_rule()]

        n_rules = rng.randint(
            self._cfg.min_entry_rules,
            min(self._cfg.max_entry_rules, len(pool)),
        )
        child_entry = rng.sample(pool, n_rules)

        # Ensure at least one prerequisite
        if not any(r.get("type") == "prerequisite" for r in child_entry):
            child_entry[0]["type"] = "prerequisite"

        # Exit rules from one parent
        donor = parent_a if rng.random() < 0.5 else parent_b
        child_exit = copy.deepcopy(donor.get("exit_rules", []))
        if not child_exit:
            child_exit = self._random_exit_rules(self._cfg.min_exit_rules)
        child_exit = _dedup_structural_exits(child_exit)

        # Blend numeric parameters
        child_params = _blend_parameters(
            parent_a.get("parameters", {}),
            parent_b.get("parameters", {}),
        )

        return {
            "entry_rules": child_entry,
            "exit_rules": child_exit,
            "parameters": child_params,
        }

    def _mutate(self, individual: dict) -> dict:
        """
        Apply one random mutation:
          0 — swap an entry rule's indicator
          1 — perturb a numeric threshold by +/- 20%
          2 — add a new entry rule (if below max)
          3 — remove an entry rule (if above min)
          4 — change an entry rule's type (plain/prerequisite/confirmation)
          5 — perturb an exit parameter
        """
        rng = self._rng
        ind = copy.deepcopy(individual)
        entry = ind.get("entry_rules", [])
        exits = ind.get("exit_rules", [])

        mutation = rng.randint(0, 5)

        if mutation == 0 and entry:
            rule = rng.choice(entry)
            new_indicator = rng.choice(GRAMMAR["indicators"])
            rule["indicator"] = new_indicator
            # Reset value to valid range for the new indicator
            if new_indicator in _REGIME_INDICATORS:
                rule["value"] = rng.choice(GRAMMAR["regime_values"])
            elif new_indicator in _COLUMN_REF_INDICATORS:
                rule["value"] = _COLUMN_REF_INDICATORS[new_indicator]
            elif new_indicator in _SPECIAL_INDICATORS:
                rule["value"] = _SPECIAL_INDICATORS[new_indicator]["value"]
                conds = _SPECIAL_INDICATORS[new_indicator]["conditions"]
                if rule.get("condition") not in conds:
                    rule["condition"] = rng.choice(conds)
            elif new_indicator in GRAMMAR["value_ranges"]:
                lo, hi = GRAMMAR["value_ranges"][new_indicator]
                rule["value"] = round(rng.uniform(lo, hi), 2)

        elif mutation == 1 and entry:
            rule = rng.choice(entry)
            if isinstance(rule.get("value"), (int, float)):
                factor = rng.uniform(0.8, 1.2)
                rule["value"] = round(rule["value"] * factor, 2)
                indicator = rule.get("indicator", "")
                if indicator in GRAMMAR["value_ranges"]:
                    lo, hi = GRAMMAR["value_ranges"][indicator]
                    rule["value"] = max(lo, min(hi, rule["value"]))

        elif mutation == 2 and len(entry) < self._cfg.max_entry_rules:
            entry.append(self._random_entry_rule())

        elif mutation == 3 and len(entry) > self._cfg.min_entry_rules:
            # Never remove the last prerequisite
            deletable = [
                j
                for j, r in enumerate(entry)
                if r.get("type") != "prerequisite"
                or sum(1 for r2 in entry if r2.get("type") == "prerequisite") > 1
            ]
            if deletable:
                entry.pop(rng.choice(deletable))

        elif mutation == 4 and entry:
            rule = rng.choice(entry)
            rule["type"] = rng.choice(GRAMMAR["rule_types"])

        elif mutation == 5 and exits:
            rule = rng.choice(exits)
            ep = GRAMMAR["exit_params"]
            if "atr_multiple" in rule:
                factor = rng.uniform(0.8, 1.2)
                rule["atr_multiple"] = round(rule["atr_multiple"] * factor, 1)
                if rule.get("type") == "stop_loss":
                    lo, hi = ep["stop_loss_atr"]
                    rule["atr_multiple"] = max(lo, min(hi, rule["atr_multiple"]))
                elif rule.get("type") == "take_profit":
                    lo, hi = ep["take_profit_atr"]
                    rule["atr_multiple"] = max(lo, min(hi, rule["atr_multiple"]))
            elif "days" in rule:
                lo, hi = ep["time_stop_days"]
                rule["days"] = rng.randint(lo, hi)

        ind["entry_rules"] = entry
        ind["exit_rules"] = exits
        return ind

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(
        self, population: list[dict], fitnesses: list[float]
    ) -> dict:
        """Tournament selection: pick ``tournament_size`` individuals, return fittest."""
        indices = self._rng.sample(
            range(len(population)),
            min(self._cfg.tournament_size, len(population)),
        )
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    # ------------------------------------------------------------------
    # Fitness (OOS Sharpe via walk-forward split)
    # ------------------------------------------------------------------

    def _evaluate_fitness(self, individual: dict, price_data: pd.DataFrame) -> float:
        """
        Backtest the individual and return OOS Sharpe ratio.

        Uses a single 70/30 train/test split. Signals are generated on the
        full history (indicators need lookback warmup from the training
        window), then only the OOS portion is evaluated. This prevents
        overfitting: the training window warms up indicator state, but
        fitness is measured exclusively on unseen data.

        Returns -1.0 on any failure so broken individuals are selected
        against but not dropped (they may carry useful partial rules
        for crossover).
        """
        try:
            from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine

            entry_rules = individual.get("entry_rules", [])
            exit_rules = individual.get("exit_rules", [])
            parameters = individual.get("parameters", {})

            split = int(len(price_data) * (1.0 - self._cfg.oos_fraction))
            oos_data = price_data.iloc[split:]

            if len(oos_data) < 30:
                return -1.0

            # Generate signals on full history (warmup), slice to OOS
            signals = generate_signals_from_rules(
                price_data, entry_rules, exit_rules, parameters
            )
            if signals is None or signals.empty:
                return -1.0

            oos_signals = signals.iloc[split:]
            oos_price = price_data.iloc[split:]

            engine = BacktestEngine(BacktestConfig(position_size_pct=0.10))
            result = engine.run(signals=oos_signals, price_data=oos_price)

            # Penalize strategies with very few trades — likely curve-fitted
            if result.total_trades < 5:
                return -1.0

            return float(result.sharpe_ratio)

        except Exception as exc:
            logger.debug(f"[GrammarGP] fitness eval failed: {exc}")
            return -1.0


# =============================================================================
# Helpers
# =============================================================================


def _blend_parameters(
    params_a: dict[str, Any], params_b: dict[str, Any]
) -> dict[str, Any]:
    """
    Blend two parameter dicts: average numeric values, random pick for non-numeric.

    Non-overlapping keys are included from both parents.
    """
    merged: dict[str, Any] = {}
    all_keys = set(params_a) | set(params_b)

    for key in all_keys:
        val_a = params_a.get(key)
        val_b = params_b.get(key)

        if val_a is None:
            merged[key] = val_b
        elif val_b is None:
            merged[key] = val_a
        elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            avg = (val_a + val_b) / 2
            if isinstance(val_a, int) and isinstance(val_b, int):
                merged[key] = int(round(avg))
            else:
                merged[key] = round(avg, 2)
        else:
            merged[key] = random.choice([val_a, val_b])

    return merged


def _dedup_structural_exits(exits: list[dict]) -> list[dict]:
    """Keep at most one of each structural exit type; preserve indicator exits."""
    structural_types = {"stop_loss", "take_profit", "time_stop"}
    seen: dict[str, dict] = {}
    indicator_exits: list[dict] = []
    for rule in exits:
        rtype = rule.get("type", "")
        if rtype in structural_types:
            seen[rtype] = rule
        else:
            indicator_exits.append(rule)
    return list(seen.values()) + indicator_exits


# =============================================================================
# Standalone smoke test
# =============================================================================

if __name__ == "__main__":
    import sys

    log_level = "DEBUG" if "--debug" in sys.argv else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Synthetic OHLCV for smoke testing (no network, no DB)
    n_bars = 500
    dates = pd.bdate_range("2022-01-01", periods=n_bars)
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    volume = np.random.randint(500_000, 5_000_000, size=n_bars)

    synthetic_data = pd.DataFrame(
        {
            "open": close + np.random.randn(n_bars) * 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    cfg = GPConfig(population_size=20, n_generations=5, elite_count=3)
    gp = GrammarGP(config=cfg)

    logger.info("Running GP on synthetic data ...")
    survivors, total_evals = gp.evolve(
        seed_population=[],
        price_data=synthetic_data,
        n_prior_trials=0,
    )

    print(f"\nSurvivors: {len(survivors)}  |  Total evaluations: {total_evals}")
    for i, s in enumerate(survivors[:5]):
        print(
            f"  #{i + 1}  "
            f"entry_rules={len(s['entry_rules'])}  "
            f"exit_rules={len(s['exit_rules'])}"
        )
