"""Tests for LangGraph state schemas."""
import operator
from typing import get_type_hints

import pytest


def test_research_state_has_all_required_fields():
    from quantstack.graphs.state import ResearchState

    hints = get_type_hints(ResearchState, include_extras=True)
    required = {
        "cycle_number", "regime", "context_summary", "selected_domain",
        "selected_symbols", "hypothesis", "validation_result", "backtest_id",
        "ml_experiment_id", "registered_strategy_id", "errors", "decisions",
    }
    assert required.issubset(hints.keys()), f"Missing fields: {required - hints.keys()}"


def test_trading_state_has_all_required_fields():
    from quantstack.graphs.state import TradingState

    hints = get_type_hints(TradingState, include_extras=True)
    required = {
        "cycle_number", "regime", "portfolio_context", "daily_plan",
        "position_reviews", "exit_orders", "entry_candidates", "risk_verdicts",
        "fund_manager_decisions", "options_analysis", "entry_orders",
        "reflection", "errors", "decisions",
    }
    assert required.issubset(hints.keys()), f"Missing fields: {required - hints.keys()}"


def test_supervisor_state_has_all_required_fields():
    from quantstack.graphs.state import SupervisorState

    hints = get_type_hints(SupervisorState, include_extras=True)
    required = {
        "cycle_number", "health_status", "diagnosed_issues",
        "recovery_actions", "strategy_lifecycle_actions",
        "scheduled_task_results", "errors",
    }
    assert required.issubset(hints.keys()), f"Missing fields: {required - hints.keys()}"


def test_append_only_fields_use_operator_add_reducer():
    from quantstack.graphs.state import ResearchState, TradingState, SupervisorState

    for schema in (ResearchState, TradingState, SupervisorState):
        hints = get_type_hints(schema, include_extras=True)
        errors_meta = hints["errors"].__metadata__
        assert operator.add in errors_meta, f"{schema.__name__}.errors missing operator.add reducer"

    for schema in (ResearchState, TradingState):
        hints = get_type_hints(schema, include_extras=True)
        decisions_meta = hints["decisions"].__metadata__
        assert operator.add in decisions_meta, f"{schema.__name__}.decisions missing operator.add reducer"


def test_node_returning_empty_list_for_append_field_does_not_error():
    existing = ["previous error"]
    update = []
    result = operator.add(existing, update)
    assert result == ["previous error"]


def test_node_omitting_append_field_from_return_does_not_error():
    update = {}
    assert "errors" not in update


def test_node_returning_none_for_append_field_raises():
    existing = ["previous error"]
    with pytest.raises(TypeError):
        operator.add(existing, None)
