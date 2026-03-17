# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for grammar-guided genetic programming alpha discovery."""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from quant_pod.alpha_discovery.grammar_gp import (
    EXIT_STRUCTURES,
    GPConfig,
    GrammarGP,
    Individual,
    NUMERIC_INDICATORS,
    RuleGrammar,
    SPECIAL_INDICATORS,
    _clone_individual,
    _dedup_structural_exits,
    _individual_to_spec,
    _spec_to_individual,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def grammar() -> RuleGrammar:
    return RuleGrammar()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def sample_individual(grammar: RuleGrammar, rng: random.Random) -> Individual:
    return grammar.random_individual(rng)


@pytest.fixture
def synthetic_price_data():
    """Generate a synthetic OHLCV DataFrame (random walk, 500 bars)."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10.0)  # keep positive
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100_000, 1_000_000, size=n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


# =============================================================================
# Grammar Tests
# =============================================================================


class TestRuleGrammar:
    """Tests for RuleGrammar production tables and validation."""

    def test_random_entry_rule_valid_structure(self, grammar: RuleGrammar, rng: random.Random):
        """Every random entry rule has required keys and valid values."""
        for _ in range(100):
            rule = grammar.random_entry_rule(rng)
            assert "indicator" in rule
            assert "condition" in rule
            assert "value" in rule
            assert rule["type"] in ("prerequisite", "confirmation")

            indicator = rule["indicator"]
            assert indicator in NUMERIC_INDICATORS or indicator in SPECIAL_INDICATORS

            if indicator in NUMERIC_INDICATORS:
                lo, hi = NUMERIC_INDICATORS[indicator]["value_range"]
                assert lo <= rule["value"] <= hi
                assert rule["condition"] in NUMERIC_INDICATORS[indicator]["conditions"]
            else:
                assert rule["condition"] in SPECIAL_INDICATORS[indicator]["conditions"]

    def test_random_indicator_exit_valid(self, grammar: RuleGrammar, rng: random.Random):
        for _ in range(50):
            rule = grammar.random_indicator_exit_rule(rng)
            assert rule["indicator"] in NUMERIC_INDICATORS
            lo, hi = NUMERIC_INDICATORS[rule["indicator"]]["value_range"]
            assert lo <= rule["value"] <= hi

    def test_random_structural_exit_valid(self, grammar: RuleGrammar, rng: random.Random):
        for _ in range(50):
            rule = grammar.random_structural_exit(rng)
            assert rule["type"] in EXIT_STRUCTURES
            if rule["type"] in ("stop_loss", "take_profit"):
                lo, hi = EXIT_STRUCTURES[rule["type"]]["atr_multiple_range"]
                assert lo <= rule["atr_multiple"] <= hi
            elif rule["type"] == "time_stop":
                lo, hi = EXIT_STRUCTURES["time_stop"]["bars_range"]
                assert lo <= rule["days"] <= hi

    def test_random_individual_passes_validation(self, grammar: RuleGrammar, rng: random.Random):
        """Every random individual is valid by construction."""
        for _ in range(50):
            ind = grammar.random_individual(rng)
            assert grammar.validate_individual(ind), f"Invalid individual: {ind}"

    def test_random_individual_has_prerequisite(self, grammar: RuleGrammar, rng: random.Random):
        for _ in range(50):
            ind = grammar.random_individual(rng)
            prereqs = [r for r in ind.entry_rules if r.get("type") == "prerequisite"]
            assert len(prereqs) >= 1

    def test_random_individual_has_structural_exit(self, grammar: RuleGrammar, rng: random.Random):
        for _ in range(50):
            ind = grammar.random_individual(rng)
            structural = [r for r in ind.exit_rules if r.get("type") in EXIT_STRUCTURES]
            assert len(structural) >= 1

    def test_validate_rejects_empty_entry(self, grammar: RuleGrammar):
        ind = Individual(entry_rules=[], exit_rules=[{"type": "stop_loss", "atr_multiple": 2.0}], parameters={})
        assert not grammar.validate_individual(ind)

    def test_validate_rejects_empty_exit(self, grammar: RuleGrammar):
        ind = Individual(
            entry_rules=[{"indicator": "rsi", "condition": "below", "value": 30, "type": "prerequisite"}],
            exit_rules=[],
            parameters={},
        )
        assert not grammar.validate_individual(ind)

    def test_validate_rejects_no_prerequisite(self, grammar: RuleGrammar):
        ind = Individual(
            entry_rules=[{"indicator": "rsi", "condition": "below", "value": 30, "type": "confirmation"}],
            exit_rules=[{"type": "stop_loss", "atr_multiple": 2.0}],
            parameters={},
        )
        assert not grammar.validate_individual(ind)

    def test_validate_rejects_no_structural_exit(self, grammar: RuleGrammar):
        ind = Individual(
            entry_rules=[{"indicator": "rsi", "condition": "below", "value": 30, "type": "prerequisite"}],
            exit_rules=[{"indicator": "rsi", "condition": "above", "value": 70}],
            parameters={},
        )
        assert not grammar.validate_individual(ind)

    def test_validate_rejects_unknown_indicator(self, grammar: RuleGrammar):
        ind = Individual(
            entry_rules=[{"indicator": "magic_indicator", "condition": "above", "value": 50, "type": "prerequisite"}],
            exit_rules=[{"type": "stop_loss", "atr_multiple": 2.0}],
            parameters={},
        )
        assert not grammar.validate_individual(ind)

    def test_clamp_value_within_range(self, grammar: RuleGrammar):
        assert grammar.clamp_value("rsi", 50.0) == 50.0
        assert grammar.clamp_value("rsi", -10.0) == 10.0
        assert grammar.clamp_value("rsi", 100.0) == 90.0
        assert grammar.clamp_value("bb_pct", 1.5) == 1.0
        assert grammar.clamp_value("bb_pct", -0.5) == 0.0


# =============================================================================
# Deduplication
# =============================================================================


class TestDedup:
    def test_dedup_keeps_one_per_type(self):
        exits = [
            {"type": "stop_loss", "atr_multiple": 1.5},
            {"type": "stop_loss", "atr_multiple": 2.0},
            {"type": "take_profit", "atr_multiple": 3.0},
            {"indicator": "rsi", "condition": "above", "value": 70},
        ]
        result = _dedup_structural_exits(exits)
        sl_count = sum(1 for r in result if r.get("type") == "stop_loss")
        tp_count = sum(1 for r in result if r.get("type") == "take_profit")
        ind_count = sum(1 for r in result if "indicator" in r)
        assert sl_count == 1
        assert tp_count == 1
        assert ind_count == 1
        # Last stop_loss wins
        sl = [r for r in result if r.get("type") == "stop_loss"][0]
        assert sl["atr_multiple"] == 2.0


# =============================================================================
# Crossover Tests
# =============================================================================


class TestCrossover:
    def test_crossover_produces_valid_individual(self, grammar: RuleGrammar, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        for _ in range(50):
            parent_a = grammar.random_individual(rng)
            parent_b = grammar.random_individual(rng)
            child = gp._crossover(parent_a, parent_b, rng)
            assert grammar.validate_individual(child), (
                f"Invalid crossover child: entry={child.entry_rules}, exit={child.exit_rules}"
            )

    def test_rule_level_crossover_mixes_parents(self, grammar: RuleGrammar, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        parent_a = Individual(
            entry_rules=[
                {"indicator": "rsi", "condition": "below", "value": 30, "type": "prerequisite"},
            ],
            exit_rules=[{"type": "stop_loss", "atr_multiple": 2.0}],
            parameters={"rsi_period": 14},
        )
        parent_b = Individual(
            entry_rules=[
                {"indicator": "adx", "condition": "above", "value": 25, "type": "prerequisite"},
                {"indicator": "cci", "condition": "below", "value": -100, "type": "confirmation"},
            ],
            exit_rules=[
                {"type": "take_profit", "atr_multiple": 3.0},
                {"indicator": "bb_pct", "condition": "above", "value": 0.9},
            ],
            parameters={"adx_period": 14},
        )
        child = gp._crossover_rule_level(parent_a, parent_b, rng)
        # Should have RSI prerequisite from A
        prereqs = [r for r in child.entry_rules if r.get("type") == "prerequisite"]
        assert any(r["indicator"] == "rsi" for r in prereqs)

    def test_entry_exit_swap_crossover(self, grammar: RuleGrammar, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        parent_a = Individual(
            entry_rules=[{"indicator": "stoch_k", "condition": "below", "value": 20, "type": "prerequisite"}],
            exit_rules=[{"type": "stop_loss", "atr_multiple": 1.5}],
            parameters={},
        )
        parent_b = Individual(
            entry_rules=[{"indicator": "cci", "condition": "above", "value": 100, "type": "prerequisite"}],
            exit_rules=[{"type": "take_profit", "atr_multiple": 4.0}],
            parameters={},
        )
        child = gp._crossover_entry_exit_swap(parent_a, parent_b, rng)
        # Entry from A, exit from B
        assert child.entry_rules[0]["indicator"] == "stoch_k"
        assert any(r.get("type") == "take_profit" for r in child.exit_rules)


# =============================================================================
# Mutation Tests
# =============================================================================


class TestMutation:
    def test_mutation_produces_valid_individual(self, grammar: RuleGrammar, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42, mutation_rate=1.0))  # 100% mutation rate
        for _ in range(100):
            ind = grammar.random_individual(rng)
            mutated = gp._mutate(ind, rng)
            assert grammar.validate_individual(mutated), (
                f"Invalid mutant: entry={mutated.entry_rules}, exit={mutated.exit_rules}"
            )

    def test_threshold_perturbation_stays_in_bounds(self, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        rule = {"indicator": "rsi", "condition": "below", "value": 30.0, "type": "prerequisite"}
        for _ in range(100):
            mutated = gp._mutate_threshold(rule, rng)
            lo, hi = NUMERIC_INDICATORS["rsi"]["value_range"]
            assert lo <= mutated["value"] <= hi

    def test_indicator_swap_changes_indicator(self, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        rule = {"indicator": "rsi", "condition": "below", "value": 30.0, "type": "prerequisite"}
        changed = False
        for _ in range(20):
            mutated = gp._mutate_indicator_swap(rule, rng)
            if mutated["indicator"] != "rsi":
                changed = True
                break
        assert changed, "Indicator swap never changed the indicator"

    def test_condition_flip_stays_valid(self, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        for indicator, spec in NUMERIC_INDICATORS.items():
            rule = {"indicator": indicator, "condition": spec["conditions"][0], "value": 50, "type": "prerequisite"}
            for _ in range(10):
                mutated = gp._mutate_condition_flip(rule, rng)
                assert mutated["condition"] in spec["conditions"]

    def test_structural_exit_mutation_stays_in_bounds(self, rng: random.Random):
        gp = GrammarGP(GPConfig(seed=42))
        for exit_type, spec in EXIT_STRUCTURES.items():
            if "atr_multiple_range" in spec:
                rule = {"type": exit_type, "atr_multiple": 2.0}
                for _ in range(50):
                    mutated = gp._mutate_structural_exit(rule, rng)
                    lo, hi = spec["atr_multiple_range"]
                    assert lo <= mutated["atr_multiple"] <= hi
            elif "bars_range" in spec:
                rule = {"type": exit_type, "days": 10}
                for _ in range(50):
                    mutated = gp._mutate_structural_exit(rule, rng)
                    lo, hi = spec["bars_range"]
                    assert lo <= mutated["days"] <= hi

    def test_mutation_never_deletes_last_prerequisite(self, grammar: RuleGrammar, rng: random.Random):
        """Even at 100% mutation rate, the last prerequisite is never deleted."""
        gp = GrammarGP(GPConfig(seed=42, mutation_rate=1.0))
        for _ in range(100):
            ind = Individual(
                entry_rules=[{"indicator": "rsi", "condition": "below", "value": 30, "type": "prerequisite"}],
                exit_rules=[{"type": "stop_loss", "atr_multiple": 2.0}],
                parameters={"rsi_period": 14},
            )
            mutated = gp._mutate(ind, rng)
            prereqs = [r for r in mutated.entry_rules if r.get("type") == "prerequisite"]
            assert len(prereqs) >= 1


# =============================================================================
# Helper Tests
# =============================================================================


class TestHelpers:
    def test_clone_is_deep(self, sample_individual: Individual):
        clone = _clone_individual(sample_individual)
        assert clone.entry_rules == sample_individual.entry_rules
        assert clone.entry_rules is not sample_individual.entry_rules
        clone.entry_rules[0]["value"] = -999
        assert sample_individual.entry_rules[0]["value"] != -999

    def test_spec_roundtrip(self, sample_individual: Individual):
        spec = _individual_to_spec(sample_individual)
        restored = _spec_to_individual(spec)
        assert restored.entry_rules == sample_individual.entry_rules
        assert restored.exit_rules == sample_individual.exit_rules
        assert restored.parameters == sample_individual.parameters

    def test_gpconfig_caps_population(self):
        cfg = GPConfig(population_size=500)
        assert cfg.population_size == 100

    def test_gpconfig_caps_elite(self):
        cfg = GPConfig(population_size=5, elite_count=10)
        assert cfg.elite_count < cfg.population_size


# =============================================================================
# Evolution Integration Tests (with mocked backtest)
# =============================================================================


class TestEvolution:
    """Integration tests using a mock backtest to avoid heavy dependencies."""

    def _mock_run_backtest(self, price_data, entry_rules, exit_rules, parameters):
        """Return plausible metrics for any strategy."""
        rng = random.Random(hash(str(entry_rules) + str(exit_rules)))
        sharpe = rng.uniform(-0.5, 2.5)
        trades = rng.randint(5, 50)
        return {
            "sharpe_ratio": sharpe,
            "total_trades": trades,
            "win_rate": rng.uniform(0.3, 0.7),
            "max_drawdown": rng.uniform(0.05, 0.30),
            "profit_factor": rng.uniform(0.8, 2.5),
            "total_return": rng.uniform(-0.1, 0.5),
        }

    @patch("quant_pod.alpha_discovery.filter._run_backtest")
    def test_evolve_returns_specs_and_evals(self, mock_bt, synthetic_price_data):
        mock_bt.side_effect = self._mock_run_backtest

        gp = GrammarGP(GPConfig(population_size=10, generations=3, seed=42))
        seeds = []  # empty seeds — will fill with random

        survivors, total_evals = gp.evolve(seeds, synthetic_price_data, n_prior_trials=100)

        assert isinstance(survivors, list)
        assert isinstance(total_evals, int)
        assert total_evals > 0
        # Every survivor should be a valid spec dict
        for spec in survivors:
            assert "entry_rules" in spec
            assert "exit_rules" in spec
            assert "parameters" in spec

    @patch("quant_pod.alpha_discovery.filter._run_backtest")
    def test_evolve_with_seeds(self, mock_bt, synthetic_price_data):
        mock_bt.side_effect = self._mock_run_backtest

        seed_specs = [
            {
                "entry_rules": [{"indicator": "rsi", "condition": "crosses_below", "value": 30, "type": "prerequisite"}],
                "exit_rules": [{"type": "stop_loss", "atr_multiple": 2.0}],
                "parameters": {"rsi_period": 14, "sma_fast_period": 10, "sma_slow_period": 50},
            },
            {
                "entry_rules": [{"indicator": "adx", "condition": "above", "value": 25, "type": "prerequisite"}],
                "exit_rules": [{"type": "stop_loss", "atr_multiple": 1.5}, {"type": "take_profit", "atr_multiple": 3.0}],
                "parameters": {"adx_period": 14, "sma_fast_period": 20, "sma_slow_period": 100},
            },
        ]

        gp = GrammarGP(GPConfig(population_size=10, generations=3, seed=42))
        survivors, total_evals = gp.evolve(seed_specs, synthetic_price_data, n_prior_trials=200)

        assert total_evals > 0

    @patch("quant_pod.alpha_discovery.filter._run_backtest")
    def test_evolve_respects_wall_clock_budget(self, mock_bt, synthetic_price_data):
        """With a tiny budget, GP should terminate early."""
        mock_bt.side_effect = self._mock_run_backtest

        gp = GrammarGP(GPConfig(
            population_size=10,
            generations=100,  # many generations
            wall_clock_budget_seconds=0.001,  # tiny budget
            seed=42,
        ))
        survivors, total_evals = gp.evolve([], synthetic_price_data, n_prior_trials=0)

        # Should have completed at most gen 0 + 1 generation
        assert total_evals <= 30  # pop_size + one generation max

    @patch("quant_pod.alpha_discovery.filter._run_backtest")
    def test_evolve_stagnation_early_stop(self, mock_bt, synthetic_price_data):
        """If fitness never improves, GP should stop early."""
        # Return constant metrics so fitness never changes
        mock_bt.return_value = {
            "sharpe_ratio": 1.0,
            "total_trades": 25,
            "win_rate": 0.55,
            "max_drawdown": 0.10,
            "profit_factor": 1.5,
            "total_return": 0.15,
        }

        gp = GrammarGP(GPConfig(
            population_size=10,
            generations=20,
            stagnation_limit=3,
            seed=42,
        ))
        survivors, total_evals = gp.evolve([], synthetic_price_data, n_prior_trials=0)

        # With stagnation_limit=3 and 20 possible generations, should stop early
        # Max evals = pop(10) + ~4 gens * ~8 offspring = ~42
        assert total_evals < 20 * 10

    @patch("quant_pod.alpha_discovery.filter._run_backtest")
    def test_evolve_handles_backtest_exceptions(self, mock_bt, synthetic_price_data):
        """GP should survive individual backtest failures."""
        call_count = 0

        def flaky_backtest(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise RuntimeError("simulated backtest failure")
            return self._mock_run_backtest(*args, **kwargs)

        mock_bt.side_effect = flaky_backtest

        gp = GrammarGP(GPConfig(population_size=10, generations=3, seed=42))
        survivors, total_evals = gp.evolve([], synthetic_price_data, n_prior_trials=0)

        # Should complete without raising
        assert total_evals > 0

    @patch("quant_pod.alpha_discovery.filter._run_backtest")
    def test_n_trials_accounting(self, mock_bt, synthetic_price_data):
        """total_evaluations should match actual backtest calls."""
        mock_bt.side_effect = self._mock_run_backtest

        gp = GrammarGP(GPConfig(population_size=10, generations=2, seed=42))
        _, total_evals = gp.evolve([], synthetic_price_data, n_prior_trials=500)

        # total_evals should equal number of backtest calls
        assert total_evals == mock_bt.call_count
