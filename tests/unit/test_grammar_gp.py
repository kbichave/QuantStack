# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for grammar-guided genetic programming alpha discovery."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from quantstack.alpha_discovery.grammar_gp import (
    GRAMMAR,
    GPConfig,
    GrammarGP,
    _dedup_structural_exits,
)


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
# Config Tests
# =============================================================================


class TestGPConfig:
    def test_gpconfig_caps_population(self):
        cfg = GPConfig(population_size=500)
        assert cfg.population_size == 100

    def test_gpconfig_caps_elite(self):
        cfg = GPConfig(population_size=5, elite_count=10)
        assert cfg.elite_count < cfg.population_size


# =============================================================================
# Grammar Structure
# =============================================================================


class TestGrammar:
    def test_grammar_has_indicators(self):
        assert "indicators" in GRAMMAR
        assert len(GRAMMAR["indicators"]) > 0

    def test_grammar_has_conditions(self):
        assert "conditions" in GRAMMAR
        assert "above" in GRAMMAR["conditions"]
        assert "below" in GRAMMAR["conditions"]

    def test_grammar_has_exit_types(self):
        assert "exit_types" in GRAMMAR
        assert "stop_loss" in GRAMMAR["exit_types"]

    def test_grammar_has_regime_values(self):
        assert "regime_values" in GRAMMAR
        assert "trending_up" in GRAMMAR["regime_values"]


# =============================================================================
# GrammarGP Integration
# =============================================================================


class TestGrammarGPInit:
    def test_init_with_defaults(self):
        gp = GrammarGP(GPConfig(seed=42))
        assert gp._cfg.seed == 42

    def test_random_individual_has_entry_and_exit_rules(self):
        gp = GrammarGP(GPConfig(seed=42))
        ind = gp._random_individual()
        assert "entry_rules" in ind
        assert "exit_rules" in ind
        assert len(ind["entry_rules"]) > 0
        assert len(ind["exit_rules"]) > 0

    def test_random_individual_has_stop_loss(self):
        gp = GrammarGP(GPConfig(seed=42))
        for _ in range(20):
            ind = gp._random_individual()
            sl = [r for r in ind["exit_rules"] if r.get("type") == "stop_loss"]
            assert len(sl) >= 1, f"Missing stop loss: {ind['exit_rules']}"
