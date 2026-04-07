"""Tests for node output models (section-04).

Verifies output models reject unknown fields, safe_default() returns valid
instances, and field ownership is enforced across graphs.
"""

import pytest
from pydantic import ValidationError

from quantstack.graphs.state import (
    ResearchState,
    SupervisorState,
    SymbolValidationState,
    TradingState,
)
from quantstack.graphs.trading.models import (
    DataRefreshOutput,
    EarningsAnalysisOutput,
    EntryScanOutput,
    ExecuteEntriesOutput,
    ExecuteExitsOutput,
    MarketIntelOutput,
    MergeParallelOutput,
    MergePreExecutionOutput,
    OptionsAnalysisOutput,
    PlanDayOutput,
    PortfolioConstructionOutput,
    PortfolioReviewOutput,
    PositionReviewOutput,
    ReflectionOutput,
    RiskSizingOutput,
    SafetyCheckOutput,
)
from quantstack.graphs.research.models import (
    BacktestValidationOutput,
    ContextLoadOutput,
    DomainSelectionOutput,
    FilterResultsOutput,
    HypothesisCritiqueOutput,
    HypothesisGenerationOutput,
    KnowledgeUpdateOutput,
    MlExperimentOutput,
    SignalValidationOutput,
    StrategyRegistrationOutput,
    ValidateSymbolOutput,
)
from quantstack.graphs.supervisor.models import (
    DiagnoseIssuesOutput,
    EodDataSyncOutput,
    ExecuteRecoveryOutput,
    HealthCheckOutput,
    ScheduledTasksOutput,
    StrategyLifecycleOutput,
    StrategyPipelineOutput,
)

# All trading output model classes
TRADING_MODELS = [
    MarketIntelOutput, EarningsAnalysisOutput, DataRefreshOutput,
    SafetyCheckOutput, PlanDayOutput, PositionReviewOutput,
    ExecuteExitsOutput, EntryScanOutput, MergeParallelOutput,
    MergePreExecutionOutput, RiskSizingOutput, PortfolioConstructionOutput,
    PortfolioReviewOutput, OptionsAnalysisOutput, ExecuteEntriesOutput,
    ReflectionOutput,
]

RESEARCH_MODELS = [
    ContextLoadOutput, DomainSelectionOutput, HypothesisGenerationOutput,
    HypothesisCritiqueOutput, SignalValidationOutput, BacktestValidationOutput,
    MlExperimentOutput, StrategyRegistrationOutput, KnowledgeUpdateOutput,
    FilterResultsOutput,
]

SUPERVISOR_MODELS = [
    HealthCheckOutput, DiagnoseIssuesOutput, ExecuteRecoveryOutput,
    StrategyPipelineOutput, StrategyLifecycleOutput, ScheduledTasksOutput,
    EodDataSyncOutput,
]


# ---------------------------------------------------------------------------
# Trading Graph Tests
# ---------------------------------------------------------------------------


def test_data_refresh_output_accepts_valid():
    out = DataRefreshOutput(data_refresh_summary={"status": "ok"})
    assert out.data_refresh_summary == {"status": "ok"}


def test_data_refresh_output_rejects_unknown_field():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DataRefreshOutput(data_refresh_summary={}, daily_plan="nope")


def test_plan_day_safe_default():
    d = PlanDayOutput.safe_default()
    assert isinstance(d, PlanDayOutput)
    assert d.daily_plan != ""
    assert d.earnings_symbols == []
    assert d.errors == []


def test_every_trading_safe_default_keys_in_state():
    state_fields = set(TradingState.model_fields.keys())
    for cls in TRADING_MODELS:
        d = cls.safe_default()
        for key in d.model_dump().keys():
            assert key in state_fields, f"{cls.__name__}.safe_default() has key {key!r} not in TradingState"


def test_entry_scan_output_rejects_wrong_type():
    with pytest.raises(ValidationError):
        EntryScanOutput(entry_candidates="not_a_list")


def test_risk_sizing_output_accepts_alpha_signals():
    out = RiskSizingOutput(alpha_signals=[0.5, 0.3])
    assert out.alpha_signals == [0.5, 0.3]


def test_blocking_node_safe_default_has_errors():
    for cls in [DataRefreshOutput, SafetyCheckOutput, PositionReviewOutput,
                ExecuteExitsOutput, RiskSizingOutput]:
        d = cls.safe_default()
        assert len(d.errors) > 0, f"{cls.__name__}.safe_default() should have errors (blocking)"


def test_non_blocking_safe_default_has_no_errors():
    for cls in [PlanDayOutput, EntryScanOutput, MarketIntelOutput,
                PortfolioConstructionOutput, ExecuteEntriesOutput,
                ReflectionOutput, PortfolioReviewOutput, OptionsAnalysisOutput,
                EarningsAnalysisOutput]:
        d = cls.safe_default()
        assert d.errors == [], f"{cls.__name__}.safe_default() should have empty errors (non-blocking)"


# ---------------------------------------------------------------------------
# Research Graph Tests
# ---------------------------------------------------------------------------


def test_context_load_output_accepts_valid():
    out = ContextLoadOutput(context_summary="test", regime="ranging")
    assert out.context_summary == "test"


def test_context_load_output_rejects_unknown():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ContextLoadOutput(nonexistent="bad")


def test_hypothesis_generation_safe_default():
    d = HypothesisGenerationOutput.safe_default()
    assert d.hypothesis == ""
    assert d.hypothesis_attempts == 1


def test_every_research_safe_default_keys_in_state():
    state_fields = set(ResearchState.model_fields.keys())
    for cls in RESEARCH_MODELS:
        d = cls.safe_default()
        for key in d.model_dump().keys():
            assert key in state_fields, f"{cls.__name__}.safe_default() has key {key!r} not in ResearchState"


def test_validate_symbol_output_keys_in_symbol_state():
    state_fields = set(SymbolValidationState.model_fields.keys())
    d = ValidateSymbolOutput.safe_default()
    for key in d.model_dump().keys():
        assert key in state_fields, f"ValidateSymbolOutput key {key!r} not in SymbolValidationState"


def test_filter_results_output_fields():
    out = FilterResultsOutput(validation_result={"passed": True})
    assert out.validation_result["passed"] is True


# ---------------------------------------------------------------------------
# Supervisor Graph Tests
# ---------------------------------------------------------------------------


def test_health_check_output_accepts_valid():
    out = HealthCheckOutput(health_status={"overall": "healthy"})
    assert out.health_status["overall"] == "healthy"


def test_scheduled_tasks_output_accepts_list():
    out = ScheduledTasksOutput(scheduled_task_results=[{"task": "attribution", "ok": True}])
    assert len(out.scheduled_task_results) == 1


def test_every_supervisor_safe_default_keys_in_state():
    state_fields = set(SupervisorState.model_fields.keys())
    for cls in SUPERVISOR_MODELS:
        d = cls.safe_default()
        for key in d.model_dump().keys():
            assert key in state_fields, f"{cls.__name__}.safe_default() has key {key!r} not in SupervisorState"


def test_eod_data_sync_safe_default():
    d = EodDataSyncOutput.safe_default()
    assert d.eod_refresh_summary.get("skipped") is True


# ---------------------------------------------------------------------------
# Cross-Cutting Tests
# ---------------------------------------------------------------------------


def test_all_output_models_use_extra_forbid():
    all_models = TRADING_MODELS + RESEARCH_MODELS + SUPERVISOR_MODELS + [ValidateSymbolOutput]
    for cls in all_models:
        config = cls.model_config
        assert config.get("extra") == "forbid", f"{cls.__name__} missing extra='forbid'"


def test_non_accumulating_fields_unique_ownership():
    """Non-accumulating fields should appear in at most one output model per graph."""
    # Fields legitimately shared across nodes (accumulators or multi-writer by design)
    shared_fields = {
        "errors", "decisions",
        # hypothesis lifecycle: context_load resets, generation/critique update
        "hypothesis_attempts", "hypothesis_confidence", "hypothesis_critique",
        # validation pipeline: signal_validation produces, filter_results refines
        "validation_result",
    }

    for models, label in [
        (TRADING_MODELS, "trading"),
        (RESEARCH_MODELS, "research"),
        (SUPERVISOR_MODELS, "supervisor"),
    ]:
        field_owners: dict[str, list[str]] = {}
        for cls in models:
            for field_name in cls.model_fields:
                if field_name in shared_fields:
                    continue
                field_owners.setdefault(field_name, []).append(cls.__name__)

        for field, owners in field_owners.items():
            assert len(owners) == 1, (
                f"{label} field {field!r} written by multiple models: {owners}"
            )
