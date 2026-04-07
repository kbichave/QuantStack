"""Static audit: every key returned by every node exists in its state schema.

This test prevents ghost fields from sneaking back in after the Pydantic
migration (section-03). If a node starts returning a new key, the curated
map below must be updated AND the state class must declare the field.
"""

from quantstack.graphs.state import (
    ResearchState,
    SupervisorState,
    SymbolValidationState,
    TradingState,
)

# ---------------------------------------------------------------------------
# Curated maps: node_name → set of keys that node returns.
#
# These maps are the source of truth. If a node changes its return dict,
# update the map here AND add the field to the state class if needed.
# ---------------------------------------------------------------------------

TRADING_NODE_KEYS: dict[str, set[str]] = {
    "market_intel": {"market_context", "decisions", "errors"},
    "earnings_analysis": {"earnings_analysis", "decisions", "errors"},
    "data_refresh": {"data_refresh_summary", "errors"},
    "safety_check": {"decisions", "errors"},
    "plan_day": {"daily_plan", "earnings_symbols", "decisions", "errors"},
    "position_review": {"position_reviews", "decisions", "errors"},
    "execute_exits": {"exit_orders", "decisions", "errors"},
    "entry_scan": {"entry_candidates", "decisions", "errors"},
    "merge_parallel": set(),
    "merge_pre_execution": set(),
    "risk_sizing": {
        "alpha_signals",
        "alpha_signal_candidates",
        "vol_state",
        "decisions",
        "errors",
    },
    "portfolio_construction": {
        "portfolio_target_weights",
        "risk_verdicts",
        "last_covariance",
        "decisions",
        "errors",
    },
    "portfolio_review": {"fund_manager_decisions", "decisions", "errors"},
    "analyze_options": {"options_analysis", "decisions", "errors"},
    "execute_entries": {"entry_orders", "decisions", "errors"},
    "reflect": {
        "reflection",
        "trade_quality_scores",
        "attribution_contexts",
        "decisions",
        "errors",
    },
}

RESEARCH_NODE_KEYS: dict[str, set[str]] = {
    "context_load": {
        "context_summary",
        "regime",
        "regime_detail",
        "decisions",
        "hypothesis_attempts",
        "hypothesis_confidence",
        "hypothesis_critique",
        "queued_task_ids",
        "errors",
    },
    "domain_selection": {"selected_domain", "selected_symbols", "decisions", "errors"},
    "hypothesis_generation": {"hypothesis", "hypothesis_attempts", "decisions", "errors"},
    "signal_validation": {"validation_result", "decisions", "errors"},
    "backtest_validation": {"backtest_id", "decisions", "errors"},
    "ml_experiment": {"ml_experiment_id", "decisions", "errors"},
    "strategy_registration": {"registered_strategy_id", "decisions", "errors"},
    "knowledge_update": {"decisions", "errors"},
    "hypothesis_critique": {"hypothesis_confidence", "hypothesis_critique", "decisions", "errors"},
    "filter_results": {"validation_result", "decisions"},
}

SUPERVISOR_NODE_KEYS: dict[str, set[str]] = {
    "health_check": {"health_status", "errors"},
    "diagnose_issues": {"diagnosed_issues", "errors"},
    "execute_recovery": {"recovery_actions", "errors"},
    "strategy_pipeline": {"strategy_pipeline_report", "errors"},
    "strategy_lifecycle": {"strategy_lifecycle_actions", "errors"},
    "scheduled_tasks": {"scheduled_task_results"},
    "eod_data_sync": {"eod_refresh_summary", "errors"},
}

SYMBOL_VALIDATION_NODE_KEYS: dict[str, set[str]] = {
    "validate_symbol": {"validation_results", "errors"},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check_node_keys(node_map: dict[str, set[str]], state_class: type, label: str):
    """Assert every key in the node map exists in the state class annotations."""
    state_fields = set(state_class.__annotations__.keys())
    undeclared: dict[str, set[str]] = {}
    for node, keys in node_map.items():
        ghost = keys - state_fields
        if ghost:
            undeclared[node] = ghost
    assert not undeclared, (
        f"Ghost fields in {label}: "
        + ", ".join(f"{n}: {sorted(g)}" for n, g in undeclared.items())
    )


def test_state_key_audit_trading():
    """Every key returned by trading nodes exists in TradingState."""
    _check_node_keys(TRADING_NODE_KEYS, TradingState, "TradingState")


def test_state_key_audit_research():
    """Every key returned by research nodes exists in ResearchState."""
    _check_node_keys(RESEARCH_NODE_KEYS, ResearchState, "ResearchState")


def test_state_key_audit_supervisor():
    """Every key returned by supervisor nodes exists in SupervisorState."""
    _check_node_keys(SUPERVISOR_NODE_KEYS, SupervisorState, "SupervisorState")


def test_state_key_audit_symbol_validation():
    """Every key returned by validation workers exists in SymbolValidationState."""
    _check_node_keys(
        SYMBOL_VALIDATION_NODE_KEYS, SymbolValidationState, "SymbolValidationState"
    )


def test_alpha_signals_ghost_resolved():
    """TradingState declares alpha_signals and alpha_signal_candidates."""
    fields = set(TradingState.__annotations__.keys())
    assert "alpha_signals" in fields, "alpha_signals missing from TradingState"
    assert "alpha_signal_candidates" in fields, "alpha_signal_candidates missing from TradingState"
