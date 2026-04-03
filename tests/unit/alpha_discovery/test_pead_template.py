"""Tests for PEAD template registration and rules (Section 08)."""

from __future__ import annotations

from quantstack.alpha_discovery.search_space import (
    MAX_COMBINATIONS_PER_TEMPLATE,
    PEAD_SPACE,
    ParameterGrid,
    TEMPLATE_REGISTRY,
    get_templates_for_regime,
)
from quantstack.alpha_discovery.engine import _get_rules_for_template


def test_auto_pead_registered_in_template_registry():
    """'auto_pead' appears in TEMPLATE_REGISTRY."""
    assert "auto_pead" in TEMPLATE_REGISTRY


def test_pead_in_all_regime_types():
    """PEAD appears for all regime types (regime-agnostic)."""
    for regime in ["trending_up", "trending_down", "ranging", "unknown"]:
        templates = get_templates_for_regime(regime)
        names = [t[0] for t in templates]
        assert "auto_pead" in names, f"auto_pead missing for regime={regime}"


def test_pead_parameter_grid_bounded():
    """Parameter grid <= 200 combinations."""
    grid = ParameterGrid(PEAD_SPACE)
    assert grid.total_combinations <= MAX_COMBINATIONS_PER_TEMPLATE
    # 4 x 5 = 20
    assert grid.total_combinations == 20


def test_pead_rules_resolve_correctly():
    """_get_rules_for_template('auto_pead') returns valid entry/exit rules."""
    entry_rules, exit_rules = _get_rules_for_template("auto_pead")

    assert len(entry_rules) > 0
    assert entry_rules[0]["indicator"] == "sue"
    assert entry_rules[0]["_param_value"] == "sue_threshold"

    assert len(exit_rules) > 0
    assert any(r.get("type") == "holding_period" for r in exit_rules)
    assert any(r.get("type") == "stop_loss" for r in exit_rules)
