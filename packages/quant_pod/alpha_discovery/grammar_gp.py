# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Grammar-guided genetic programming for alpha discovery.

Extends the AlphaDiscoveryEngine with evolutionary search over rule syntax
trees. GP operators (crossover, mutation) recombine entry/exit rules from
different templates — discovering strategy structures that fixed templates
and parameter grids cannot reach.

Grammar constraints ensure every generated individual is a valid strategy
spec consumable by ``_generate_signals_from_rules`` and ``BacktestEngine``.

Design reference: AlphaCFG (arxiv 2601.22119), HARLA (Frontiers CS 2025).
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GPConfig:
    """Tunable knobs for the GP evolution loop."""

    population_size: int = 30
    generations: int = 10
    mutation_rate: float = 0.3
    tournament_size: int = 3
    elite_count: int = 2
    max_entry_rules: int = 4
    max_exit_rules: int = 4
    wall_clock_budget_seconds: float = 120.0
    stagnation_limit: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size > 100:
            self.population_size = 100
        if self.elite_count >= self.population_size:
            self.elite_count = max(1, self.population_size // 5)


# =============================================================================
# Individual (genome)
# =============================================================================


@dataclass
class Individual:
    """A single strategy genome: entry rules + exit rules + parameters."""

    entry_rules: list[dict[str, Any]]
    exit_rules: list[dict[str, Any]]
    parameters: dict[str, Any]
    fitness: float = float("-inf")
    is_trades: int = 0
    generation: int = 0
    lineage: str = ""


# =============================================================================
# Grammar — constrains GP operations to valid rule combinations
# =============================================================================

# Indicators producing numeric series — used for entry and indicator-based exits.
# Each maps to: valid conditions, numeric value range, optional parameter key + range.
NUMERIC_INDICATORS: dict[str, dict[str, Any]] = {
    "rsi": {
        "conditions": ["above", "below", "crosses_above", "crosses_below"],
        "value_range": (10.0, 90.0),
        "param_key": "rsi_period",
        "param_range": (7, 28),
    },
    "adx": {
        "conditions": ["above", "below"],
        "value_range": (15.0, 50.0),
        "param_key": "adx_period",
        "param_range": (10, 21),
    },
    "bb_pct": {
        "conditions": ["above", "below", "crosses_above", "crosses_below"],
        "value_range": (0.0, 1.0),
        "param_key": "bb_period",
        "param_range": (10, 30),
    },
    "cci": {
        "conditions": ["above", "below", "crosses_above", "crosses_below"],
        "value_range": (-200.0, 200.0),
        "param_key": "cci_period",
        "param_range": (10, 30),
    },
    "stoch_k": {
        "conditions": ["above", "below", "crosses_above", "crosses_below"],
        "value_range": (10.0, 90.0),
        "param_key": "stoch_period",
        "param_range": (7, 21),
    },
    "stoch_d": {
        "conditions": ["above", "below", "crosses_above", "crosses_below"],
        "value_range": (10.0, 90.0),
    },
    "zscore": {
        "conditions": ["above", "below", "crosses_above", "crosses_below"],
        "value_range": (-3.0, 3.0),
        "param_key": "zscore_period",
        "param_range": (10, 40),
    },
    "atr_percentile": {
        "conditions": ["above", "below"],
        "value_range": (10.0, 90.0),
    },
    "price_vs_sma200": {
        "conditions": ["above", "below"],
        "value_range": (-20.0, 20.0),
    },
}

# Special indicators with restricted condition sets — no numeric value needed.
SPECIAL_INDICATORS: dict[str, dict[str, Any]] = {
    "sma_crossover": {
        "conditions": ["crosses_above", "crosses_below"],
        "value": 0,
    },
    "breakout": {
        "conditions": ["above", "below"],
        "value": 0,
    },
}

# Structural exit types with ATR / bars ranges.
EXIT_STRUCTURES: dict[str, dict[str, Any]] = {
    "stop_loss": {"atr_multiple_range": (0.5, 4.0)},
    "take_profit": {"atr_multiple_range": (1.0, 6.0)},
    "time_stop": {"bars_range": (5, 40)},
}

# Rule type assignments for GP-generated entry rules.
_ENTRY_RULE_TYPES = ("prerequisite", "confirmation")


class RuleGrammar:
    """Production tables + validation for GP rule dicts."""

    def random_entry_rule(self, rng: random.Random) -> dict[str, Any]:
        """Generate a random valid entry rule from the grammar."""
        rule_type = rng.choice(_ENTRY_RULE_TYPES)

        # 70% numeric, 30% special — numeric has more variety
        if rng.random() < 0.7:
            indicator = rng.choice(list(NUMERIC_INDICATORS))
            spec = NUMERIC_INDICATORS[indicator]
            lo, hi = spec["value_range"]
            return {
                "indicator": indicator,
                "condition": rng.choice(spec["conditions"]),
                "value": round(rng.uniform(lo, hi), 2),
                "type": rule_type,
            }

        indicator = rng.choice(list(SPECIAL_INDICATORS))
        spec = SPECIAL_INDICATORS[indicator]
        return {
            "indicator": indicator,
            "condition": rng.choice(spec["conditions"]),
            "value": spec["value"],
            "type": rule_type,
        }

    def random_indicator_exit_rule(self, rng: random.Random) -> dict[str, Any]:
        """Generate a random indicator-based exit rule (not structural)."""
        indicator = rng.choice(list(NUMERIC_INDICATORS))
        spec = NUMERIC_INDICATORS[indicator]
        lo, hi = spec["value_range"]
        return {
            "indicator": indicator,
            "condition": rng.choice(spec["conditions"]),
            "value": round(rng.uniform(lo, hi), 2),
        }

    def random_structural_exit(self, rng: random.Random) -> dict[str, Any]:
        """Generate a random structural exit (stop_loss, take_profit, or time_stop)."""
        exit_type = rng.choice(list(EXIT_STRUCTURES))
        spec = EXIT_STRUCTURES[exit_type]
        if "atr_multiple_range" in spec:
            lo, hi = spec["atr_multiple_range"]
            return {"type": exit_type, "atr_multiple": round(rng.uniform(lo, hi), 2)}
        lo, hi = spec["bars_range"]
        return {"type": exit_type, "days": rng.randint(lo, hi)}

    def random_individual(self, rng: random.Random) -> Individual:
        """Generate a fully random valid individual from the grammar."""
        n_entry = rng.randint(1, 3)
        entry_rules = [self.random_entry_rule(rng) for _ in range(n_entry)]
        # Ensure at least one prerequisite
        if not any(r.get("type") == "prerequisite" for r in entry_rules):
            entry_rules[0]["type"] = "prerequisite"

        # 1-2 structural exits + 0-1 indicator exits
        exit_rules = [self.random_structural_exit(rng)]
        if rng.random() < 0.5:
            exit_rules.append(self.random_structural_exit(rng))
        if rng.random() < 0.4:
            exit_rules.append(self.random_indicator_exit_rule(rng))

        # Deduplicate structural exit types (keep last of each type)
        exit_rules = _dedup_structural_exits(exit_rules)

        parameters = self._random_parameters(rng, entry_rules)
        return Individual(
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            parameters=parameters,
            lineage="random",
        )

    def validate_individual(self, ind: Individual) -> bool:
        """Check that an individual satisfies grammar invariants."""
        if not ind.entry_rules:
            return False
        if not ind.exit_rules:
            return False

        # Must have at least one prerequisite entry rule
        if not any(r.get("type") == "prerequisite" for r in ind.entry_rules):
            return False

        # Must have at least one structural exit
        structural_types = {"stop_loss", "take_profit", "time_stop"}
        if not any(r.get("type") in structural_types for r in ind.exit_rules):
            return False

        # Validate each entry rule
        for rule in ind.entry_rules:
            indicator = rule.get("indicator", "")
            if indicator in NUMERIC_INDICATORS:
                if rule.get("condition") not in NUMERIC_INDICATORS[indicator]["conditions"]:
                    return False
            elif indicator in SPECIAL_INDICATORS:
                if rule.get("condition") not in SPECIAL_INDICATORS[indicator]["conditions"]:
                    return False
            else:
                return False

        # Validate each exit rule
        for rule in ind.exit_rules:
            rule_type = rule.get("type", "")
            if rule_type in structural_types:
                continue  # structural exits are valid by construction
            indicator = rule.get("indicator", "")
            if indicator not in NUMERIC_INDICATORS and indicator not in SPECIAL_INDICATORS:
                return False

        return True

    def clamp_value(self, indicator: str, value: float) -> float:
        """Clamp a numeric value to the indicator's valid range."""
        if indicator in NUMERIC_INDICATORS:
            lo, hi = NUMERIC_INDICATORS[indicator]["value_range"]
            return max(lo, min(hi, round(value, 2)))
        return value

    def _random_parameters(
        self, rng: random.Random, entry_rules: list[dict]
    ) -> dict[str, Any]:
        """Build a parameter dict that covers indicators used in the rules."""
        params: dict[str, Any] = {
            "sma_fast_period": rng.choice([10, 20]),
            "sma_slow_period": rng.choice([50, 100, 200]),
        }
        for rule in entry_rules:
            indicator = rule.get("indicator", "")
            if indicator in NUMERIC_INDICATORS:
                spec = NUMERIC_INDICATORS[indicator]
                if "param_key" in spec:
                    lo, hi = spec["param_range"]
                    params[spec["param_key"]] = rng.randint(lo, hi)
        return params


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
# GrammarGP — the evolution engine
# =============================================================================


class GrammarGP:
    """
    Grammar-guided genetic programming for strategy discovery.

    Evolves a population of strategy individuals using crossover and mutation
    operators constrained by ``RuleGrammar``. Returns IS-passing candidates
    for the caller to validate through full OOS + portfolio-fit filters.
    """

    def __init__(self, config: GPConfig | None = None) -> None:
        self._config = config or GPConfig()
        self._grammar = RuleGrammar()

    def evolve(
        self,
        seed_population: list[dict[str, Any]],
        price_data: Any,
        n_prior_trials: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Run GP evolution and return IS-passing strategy specs.

        Args:
            seed_population: Strategy specs from grid search that had
                sufficient trades (IS_MIN_TRADES). Used to seed the GP.
            price_data: Full OHLCV DataFrame (all available history).
            n_prior_trials: Candidates already evaluated by grid search.
                Added to GP eval count for Harvey-Liu deflation.

        Returns:
            Tuple of (list of IS-passing strategy specs, total evaluations).
        """
        cfg = self._config
        rng = random.Random(cfg.seed)
        start_time = time.monotonic()
        total_evals = 0

        # Build initial population
        population = self._build_initial_population(seed_population, rng)

        # IS data slice (75% split — same as CandidateFilter)
        is_split = int(len(price_data) * 0.75)
        is_data = price_data.iloc[:is_split]

        if len(is_data) < 60:
            logger.warning("[GrammarGP] insufficient IS data for GP evolution")
            return [], 0

        # Evaluate initial population
        for ind in population:
            self._evaluate_fitness(ind, is_data, n_prior_trials + total_evals)
            total_evals += 1

        best_fitness = max(ind.fitness for ind in population)
        stagnation_count = 0

        logger.info(
            f"[GrammarGP] gen=0 pop={len(population)} "
            f"best_fitness={best_fitness:.3f} evals={total_evals}"
        )

        for gen in range(1, cfg.generations + 1):
            # Wall-clock budget check
            elapsed = time.monotonic() - start_time
            if elapsed >= cfg.wall_clock_budget_seconds:
                logger.info(
                    f"[GrammarGP] wall-clock budget exhausted at gen={gen} "
                    f"({elapsed:.1f}s)"
                )
                break

            new_population: list[Individual] = []

            # Elitism — carry top individuals unchanged
            ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
            for elite in ranked[: cfg.elite_count]:
                elite_copy = _clone_individual(elite)
                elite_copy.generation = gen
                new_population.append(elite_copy)

            # Fill rest via crossover + mutation
            while len(new_population) < cfg.population_size:
                parent_a = self._select_parent(population, rng)
                parent_b = self._select_parent(population, rng)
                child = self._crossover(parent_a, parent_b, rng)
                child = self._mutate(child, rng)
                child.generation = gen

                if not self._grammar.validate_individual(child):
                    # Invalid offspring — replace with random
                    child = self._grammar.random_individual(rng)
                    child.generation = gen
                    child.lineage = "random_repair"

                self._evaluate_fitness(child, is_data, n_prior_trials + total_evals)
                total_evals += 1
                new_population.append(child)

            population = new_population

            gen_best = max(ind.fitness for ind in population)
            if gen_best <= best_fitness:
                stagnation_count += 1
            else:
                stagnation_count = 0
                best_fitness = gen_best

            logger.debug(
                f"[GrammarGP] gen={gen} best={gen_best:.3f} "
                f"stagnation={stagnation_count} evals={total_evals}"
            )

            if stagnation_count >= cfg.stagnation_limit:
                logger.info(
                    f"[GrammarGP] early stop — stagnation for "
                    f"{stagnation_count} generations"
                )
                break

        # Collect IS-passing survivors
        from quant_pod.alpha_discovery.filter import IS_MIN_SHARPE, IS_MIN_TRADES

        survivors = [
            _individual_to_spec(ind)
            for ind in population
            if ind.fitness >= IS_MIN_SHARPE and ind.is_trades >= IS_MIN_TRADES
        ]

        elapsed = time.monotonic() - start_time
        logger.info(
            f"[GrammarGP] done — survivors={len(survivors)} "
            f"total_evals={total_evals} elapsed={elapsed:.1f}s"
        )

        return survivors, total_evals

    # -------------------------------------------------------------------------
    # Population seeding
    # -------------------------------------------------------------------------

    def _build_initial_population(
        self,
        seed_specs: list[dict[str, Any]],
        rng: random.Random,
    ) -> list[Individual]:
        """Build initial population from seeds + random grammar individuals."""
        cfg = self._config
        population: list[Individual] = []

        # Seed from grid search IS survivors (top by IS Sharpe proxy — those
        # with the most trades are structurally interesting)
        for spec in seed_specs[: cfg.population_size * 2 // 3]:
            ind = _spec_to_individual(spec)
            ind.lineage = "seed"
            if self._grammar.validate_individual(ind):
                population.append(ind)
            if len(population) >= cfg.population_size * 2 // 3:
                break

        # Fill remaining with random grammar individuals
        while len(population) < cfg.population_size:
            ind = self._grammar.random_individual(rng)
            population.append(ind)

        return population[: cfg.population_size]

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def _select_parent(
        self, population: list[Individual], rng: random.Random
    ) -> Individual:
        """Tournament selection."""
        candidates = rng.sample(
            population, min(self._config.tournament_size, len(population))
        )
        return max(candidates, key=lambda x: x.fitness)

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def _crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: random.Random,
    ) -> Individual:
        """Recombine two parents into a child."""
        if rng.random() < 0.5:
            return self._crossover_rule_level(parent_a, parent_b, rng)
        return self._crossover_entry_exit_swap(parent_a, parent_b, rng)

    def _crossover_rule_level(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: random.Random,
    ) -> Individual:
        """Prerequisites from A, confirmations from B. Mixed exits."""
        prereqs_a = [
            r for r in parent_a.entry_rules if r.get("type") == "prerequisite"
        ]
        confirms_b = [
            r for r in parent_b.entry_rules if r.get("type") == "confirmation"
        ]

        entry_rules = copy.deepcopy(prereqs_a) + copy.deepcopy(confirms_b)
        if not entry_rules:
            entry_rules = copy.deepcopy(parent_a.entry_rules)

        # Exits: structural from A, indicator-based from B
        structural_types = {"stop_loss", "take_profit", "time_stop"}
        struct_exits_a = [r for r in parent_a.exit_rules if r.get("type") in structural_types]
        indicator_exits_b = [r for r in parent_b.exit_rules if r.get("type") not in structural_types]

        exit_rules = copy.deepcopy(struct_exits_a) + copy.deepcopy(indicator_exits_b)
        if not any(r.get("type") in structural_types for r in exit_rules):
            exit_rules.insert(0, copy.deepcopy(parent_a.exit_rules[0]))

        exit_rules = _dedup_structural_exits(exit_rules)

        # Merge parameters — A takes precedence on conflicts
        params = {**parent_b.parameters, **parent_a.parameters}

        child = Individual(
            entry_rules=entry_rules[: self._config.max_entry_rules],
            exit_rules=exit_rules[: self._config.max_exit_rules],
            parameters=params,
            lineage="crossover_rule",
        )
        return child

    def _crossover_entry_exit_swap(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: random.Random,
    ) -> Individual:
        """All entry rules from A, all exit rules from B."""
        entry_rules = copy.deepcopy(parent_a.entry_rules)
        exit_rules = copy.deepcopy(parent_b.exit_rules)

        # Ensure structural exit exists
        structural_types = {"stop_loss", "take_profit", "time_stop"}
        if not any(r.get("type") in structural_types for r in exit_rules):
            exit_rules.insert(0, self._grammar.random_structural_exit(rng))

        params = {**parent_b.parameters, **parent_a.parameters}

        return Individual(
            entry_rules=entry_rules[: self._config.max_entry_rules],
            exit_rules=_dedup_structural_exits(exit_rules)[: self._config.max_exit_rules],
            parameters=params,
            lineage="crossover_swap",
        )

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def _mutate(self, ind: Individual, rng: random.Random) -> Individual:
        """Apply mutation operators with probability mutation_rate per rule."""
        ind = _clone_individual(ind)

        for i, rule in enumerate(ind.entry_rules):
            if rng.random() < self._config.mutation_rate:
                operator = rng.choice([
                    self._mutate_threshold,
                    self._mutate_indicator_swap,
                    self._mutate_condition_flip,
                ])
                ind.entry_rules[i] = operator(rule, rng)

        # Possibly insert or delete a rule
        if rng.random() < self._config.mutation_rate:
            action = rng.choice(["insert", "delete"])
            if action == "insert" and len(ind.entry_rules) < self._config.max_entry_rules:
                new_rule = self._grammar.random_entry_rule(rng)
                new_rule["type"] = "confirmation"  # additions are always confirmations
                ind.entry_rules.append(new_rule)
            elif action == "delete" and len(ind.entry_rules) > 1:
                # Never delete the last prerequisite
                deletable = [
                    j for j, r in enumerate(ind.entry_rules)
                    if r.get("type") != "prerequisite"
                    or sum(1 for r2 in ind.entry_rules if r2.get("type") == "prerequisite") > 1
                ]
                if deletable:
                    ind.entry_rules.pop(rng.choice(deletable))

        # Mutate structural exit parameters
        structural_types = {"stop_loss", "take_profit", "time_stop"}
        for i, rule in enumerate(ind.exit_rules):
            if rule.get("type") in structural_types and rng.random() < self._config.mutation_rate:
                ind.exit_rules[i] = self._mutate_structural_exit(rule, rng)

        return ind

    def _mutate_threshold(
        self, rule: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Perturb a numeric threshold by +/- 5-25% of the valid range."""
        rule = dict(rule)
        indicator = rule.get("indicator", "")
        if indicator not in NUMERIC_INDICATORS:
            return rule

        spec = NUMERIC_INDICATORS[indicator]
        lo, hi = spec["value_range"]
        range_span = hi - lo
        perturbation = rng.uniform(0.05, 0.25) * range_span * rng.choice([-1, 1])
        old_val = float(rule.get("value", (lo + hi) / 2))
        rule["value"] = self._grammar.clamp_value(indicator, old_val + perturbation)
        return rule

    def _mutate_indicator_swap(
        self, rule: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Replace indicator with another from the same category."""
        rule = dict(rule)
        old_indicator = rule.get("indicator", "")

        if old_indicator in NUMERIC_INDICATORS:
            candidates = [k for k in NUMERIC_INDICATORS if k != old_indicator]
            if not candidates:
                return rule
            new_indicator = rng.choice(candidates)
            spec = NUMERIC_INDICATORS[new_indicator]
            lo, hi = spec["value_range"]
            rule["indicator"] = new_indicator
            rule["value"] = round((lo + hi) / 2, 2)  # midpoint
            # Ensure condition is valid for new indicator
            if rule.get("condition") not in spec["conditions"]:
                rule["condition"] = rng.choice(spec["conditions"])
        elif old_indicator in SPECIAL_INDICATORS:
            candidates = [k for k in SPECIAL_INDICATORS if k != old_indicator]
            if not candidates:
                return rule
            new_indicator = rng.choice(candidates)
            spec = SPECIAL_INDICATORS[new_indicator]
            rule["indicator"] = new_indicator
            rule["value"] = spec["value"]
            if rule.get("condition") not in spec["conditions"]:
                rule["condition"] = rng.choice(spec["conditions"])

        return rule

    def _mutate_condition_flip(
        self, rule: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Flip the condition within the indicator's valid set."""
        rule = dict(rule)
        indicator = rule.get("indicator", "")

        if indicator in NUMERIC_INDICATORS:
            conditions = NUMERIC_INDICATORS[indicator]["conditions"]
        elif indicator in SPECIAL_INDICATORS:
            conditions = SPECIAL_INDICATORS[indicator]["conditions"]
        else:
            return rule

        other_conditions = [c for c in conditions if c != rule.get("condition")]
        if other_conditions:
            rule["condition"] = rng.choice(other_conditions)
        return rule

    def _mutate_structural_exit(
        self, rule: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Perturb a structural exit's ATR multiple or bar count."""
        rule = dict(rule)
        exit_type = rule.get("type", "")
        if exit_type not in EXIT_STRUCTURES:
            return rule

        spec = EXIT_STRUCTURES[exit_type]
        if "atr_multiple_range" in spec:
            lo, hi = spec["atr_multiple_range"]
            old_val = float(rule.get("atr_multiple", (lo + hi) / 2))
            perturbation = rng.uniform(0.1, 0.5) * rng.choice([-1, 1])
            rule["atr_multiple"] = round(max(lo, min(hi, old_val + perturbation)), 2)
        elif "bars_range" in spec:
            lo, hi = spec["bars_range"]
            old_val = int(rule.get("days", (lo + hi) // 2))
            perturbation = rng.randint(1, 5) * rng.choice([-1, 1])
            rule["days"] = max(lo, min(hi, old_val + perturbation))

        return rule

    # -------------------------------------------------------------------------
    # Fitness evaluation
    # -------------------------------------------------------------------------

    def _evaluate_fitness(
        self, ind: Individual, is_data: Any, n_trials_so_far: int
    ) -> None:
        """Run IS backtest and set deflated Sharpe as fitness."""
        try:
            from quant_pod.alpha_discovery.filter import (
                IS_MIN_PROFIT_FACTOR,
                IS_MIN_TRADES,
                _deflate_sharpe,
                _run_backtest,
            )

            metrics = _run_backtest(
                is_data, ind.entry_rules, ind.exit_rules, ind.parameters
            )
            ind.is_trades = metrics["total_trades"]

            if ind.is_trades < IS_MIN_TRADES:
                ind.fitness = float("-inf")
                return

            raw_sharpe = metrics["sharpe_ratio"]
            # Use n_trials_so_far + 1 (this evaluation) for deflation
            deflated = _deflate_sharpe(raw_sharpe, max(n_trials_so_far, 1), t_bars=len(is_data))

            profit_factor = metrics.get("profit_factor", 0.0)
            if profit_factor < IS_MIN_PROFIT_FACTOR:
                ind.fitness = float("-inf")
                return

            ind.fitness = deflated

        except Exception as exc:
            logger.debug(f"[GrammarGP] fitness eval failed: {exc}")
            ind.fitness = float("-inf")


# =============================================================================
# Helpers
# =============================================================================


def _clone_individual(ind: Individual) -> Individual:
    """Deep-copy an individual."""
    return Individual(
        entry_rules=copy.deepcopy(ind.entry_rules),
        exit_rules=copy.deepcopy(ind.exit_rules),
        parameters=copy.deepcopy(ind.parameters),
        fitness=ind.fitness,
        is_trades=ind.is_trades,
        generation=ind.generation,
        lineage=ind.lineage,
    )


def _spec_to_individual(spec: dict[str, Any]) -> Individual:
    """Convert a strategy spec dict to an Individual."""
    return Individual(
        entry_rules=copy.deepcopy(spec.get("entry_rules", [])),
        exit_rules=copy.deepcopy(spec.get("exit_rules", [])),
        parameters=copy.deepcopy(spec.get("parameters", {})),
    )


def _individual_to_spec(ind: Individual) -> dict[str, Any]:
    """Convert an Individual back to a strategy spec dict."""
    return {
        "entry_rules": copy.deepcopy(ind.entry_rules),
        "exit_rules": copy.deepcopy(ind.exit_rules),
        "parameters": copy.deepcopy(ind.parameters),
    }
