# Section 04: Node Output Models

## Purpose

Every node in the three LangGraph graphs (Trading, Research, Supervisor) currently returns a plain `dict[str, Any]`. There is no enforcement of which state fields a node is allowed to write. A typo in a return key (e.g., `{"daly_plan": "..."}`) is silently merged into state, leaving the real field stale. A node that accidentally writes to a field owned by another node causes subtle cross-contamination.

This section creates a typed Pydantic output model for every node in all three graphs. Each model defines the exact subset of parent graph state fields that the node writes. The node function returns an instance of its output model, which Pydantic validates before the LangGraph state merge. Combined with the `extra="forbid"` policy on the parent state (section-03), this creates a two-layer defense: the output model catches writes to fields the node shouldn't touch, and the parent state catches typos that slip through.

Each output model also includes a `safe_default()` class method that returns a typed, neutral response. This is consumed by the circuit breaker (section-07) and error blocking (section-05) when a node fails or is skipped.

## Dependencies

- **section-03-pydantic-state-migration** must be complete. The output models reference fields defined in the Pydantic `TradingState`, `ResearchState`, `SupervisorState`, and `SymbolValidationState` models. The parent state's `extra="forbid"` is what makes these output models effective as a safety layer.

## Blocked By This Section

- **section-05-error-blocking**: uses `safe_default()` to continue pipeline after non-blocking node failures.
- **section-06-race-condition-fix**: the new `resolve_symbol_conflicts` node needs its own output model.
- **section-07-circuit-breaker**: the `@circuit_breaker` decorator calls `OutputModel.safe_default()` when the breaker is open.

---

## Tests (Write First)

Tests live in `tests/unit/test_node_output_models.py`.

### Trading Graph Output Model Tests

```python
# Test: DataRefreshOutput only allows fields that data_refresh writes to state
#   Construct DataRefreshOutput with valid fields → accepts.
#   Construct with a field not in its schema (e.g., "daily_plan") → ValidationError.

# Test: PlanDayOutput.safe_default() returns valid neutral response
#   Call PlanDayOutput.safe_default() → verify it returns a valid PlanDayOutput instance.
#   Verify daily_plan is a non-empty fallback string, earnings_symbols is [].

# Test: Every trading node output model's safe_default() passes parent state validation
#   For each output model class in trading/models.py:
#     call safe_default(), convert to dict, verify all keys exist in TradingState fields.

# Test: DataRefreshOutput rejects fields not in its schema
#   Construct DataRefreshOutput(data_refresh_summary={...}, daily_plan="x") → ValidationError.

# Test: Output models enforce correct types
#   Construct EntryScanOutput(entry_candidates="not_a_list") → ValidationError.

# Test: RiskSizingOutput alpha_signals field accepts list of floats
#   Construct with alpha_signals=[0.5, 0.3] → accepts.

# Test: Blocking node safe_default includes error flag
#   DataRefreshOutput.safe_default() → verify "errors" field is populated with a descriptive
#   string so the execution gate (section-05) can detect the failure.

# Test: Non-blocking node safe_default has empty errors
#   PlanDayOutput.safe_default() → verify "errors" field is [].
```

### Research Graph Output Model Tests

```python
# Test: ContextLoadOutput accepts valid fields (context_summary, regime, regime_detail, etc.)
# Test: ContextLoadOutput rejects unknown fields (extra="forbid")
# Test: HypothesisGenerationOutput.safe_default() returns empty hypothesis string, incremented attempts
# Test: Every research node output model's safe_default() keys are valid ResearchState fields
# Test: ValidateSymbolOutput writes only to validation_results (Annotated list)
# Test: FilterResultsOutput writes only to validation_result and decisions
```

### Supervisor Graph Output Model Tests

```python
# Test: HealthCheckOutput accepts health_status dict and optional errors list
# Test: ScheduledTasksOutput accepts scheduled_task_results list
# Test: Every supervisor node output model's safe_default() keys are valid SupervisorState fields
# Test: EodDataSyncOutput.safe_default() returns skipped summary
```

### Cross-Cutting Tests

```python
# Test: No output model allows writing to a field that belongs exclusively to another node
#   For each graph, build a map of {field: set_of_models_that_write_it}.
#   Verify shared fields are only the accumulating ones (errors, decisions).
#   Non-accumulating fields must appear in exactly one output model.

# Test: All output models use extra="forbid"
#   Introspect model_config for every output model class → verify extra == "forbid".
```

---

## Implementation Details

### File Locations

| File | Status | Contents |
|------|--------|----------|
| `src/quantstack/graphs/trading/models.py` | NEW | Output models for all trading graph nodes |
| `src/quantstack/graphs/research/models.py` | NEW | Output models for all research graph nodes |
| `src/quantstack/graphs/supervisor/models.py` | NEW | Output models for all supervisor graph nodes |
| `src/quantstack/graphs/trading/nodes.py` | MODIFY | Return typed output model instances instead of plain dicts |
| `src/quantstack/graphs/research/nodes.py` | MODIFY | Return typed output model instances instead of plain dicts |
| `src/quantstack/graphs/supervisor/nodes.py` | MODIFY | Return typed output model instances instead of plain dicts |
| `tests/unit/test_node_output_models.py` | NEW | Tests above |

### Design Pattern

Every output model follows the same structure:

```python
from pydantic import BaseModel, ConfigDict

class DataRefreshOutput(BaseModel):
    """Output model for the data_refresh node."""
    model_config = ConfigDict(extra="forbid")

    data_refresh_summary: dict
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> "DataRefreshOutput":
        """Neutral fallback when the node is circuit-broken or fails."""
        return cls(
            data_refresh_summary={"skipped": True, "reason": "node_unavailable"},
            errors=["data_refresh: node unavailable (circuit breaker or failure)"],
        )
```

Key conventions:
- `model_config = ConfigDict(extra="forbid")` on every model -- rejects writes to fields the node shouldn't touch.
- `errors: list[str] = []` and `decisions: list[dict] = []` are the only fields shared across multiple models (they use `operator.add` reducers in the parent state).
- `safe_default()` is a `@classmethod` returning a valid instance. For blocking nodes, the `errors` list is populated so the execution gate detects the failure. For non-blocking nodes, `errors` is empty.
- The model is a strict subset of the parent graph state. Only fields the node actually writes appear in the model.

### Trading Graph Models (`src/quantstack/graphs/trading/models.py`)

The following output models are needed, one per node. The field lists are derived from the actual return statements in `src/quantstack/graphs/trading/nodes.py`.

**MarketIntelOutput** -- `make_market_intel`
- `market_context: dict`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `market_context={}`, empty errors (non-blocking)

**EarningsAnalysisOutput** -- `make_earnings_analysis`
- `earnings_analysis: dict`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `earnings_analysis={}`, empty errors (non-blocking)

**DataRefreshOutput** -- `make_data_refresh`
- `data_refresh_summary: dict`
- `errors: list[str] = []`
- safe_default: `data_refresh_summary={"skipped": True, "reason": "node_unavailable"}`, errors populated (blocking)

**SafetyCheckOutput** -- `make_safety_check`
- `decisions: list[dict]`
- `errors: list[str] = []`
- safe_default: `decisions=[{"node": "safety_check", "halted": True, "error": "node_unavailable"}]`, errors populated (blocking)

**PlanDayOutput** -- `make_daily_plan`
- `daily_plan: str`
- `earnings_symbols: list[str] = []`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `daily_plan="Plan unavailable — using neutral bias"`, empty earnings/errors (non-blocking)

**PositionReviewOutput** -- `make_position_review`
- `position_reviews: list[dict]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `position_reviews=[]`, errors populated (blocking -- failure to review positions = unknown exposure)

**ExecuteExitsOutput** -- `make_execute_exits`
- `exit_orders: list[dict]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `exit_orders=[]`, errors populated (blocking -- failure to close exposure is dangerous)

**EntryScanOutput** -- `make_entry_scan`
- `entry_candidates: list[dict]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `entry_candidates=[]`, empty errors (non-blocking -- missed opportunity only)

**MergeParallelOutput** -- `merge_parallel`
- No fields (returns `{}`). This is a no-op join node. Model is an empty BaseModel.
- safe_default: empty instance

**MergePreExecutionOutput** -- `merge_pre_execution`
- Same as MergeParallelOutput.

**RiskSizingOutput** -- `make_risk_sizing`
- `alpha_signals: list`
- `alpha_signal_candidates: list`
- `vol_state: str = "normal"`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `alpha_signals=[]`, `alpha_signal_candidates=[]`, errors populated (blocking)
- Note: `risk_sizing` writes `alpha_signals` and `alpha_signal_candidates` which are NOT declared in the current `TradingState` TypedDict. The state key audit (section-02) should have surfaced these as ghost fields. They must be added to the Pydantic TradingState in section-03 before this model is valid.

**PortfolioConstructionOutput** -- `make_portfolio_construction`
- `portfolio_target_weights: dict`
- `risk_verdicts: list[dict]`
- `last_covariance: list`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `portfolio_target_weights={}`, `risk_verdicts=[]`, `last_covariance=[]`, empty errors (non-blocking)

**PortfolioReviewOutput** -- `make_portfolio_review`
- `fund_manager_decisions: list[dict]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `fund_manager_decisions=[]`, empty errors (non-blocking)

**OptionsAnalysisOutput** -- `make_options_analysis`
- `options_analysis: list[dict]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `options_analysis=[]`, empty errors (non-blocking)

**ExecuteEntriesOutput** -- `make_execute_entries`
- `entry_orders: list[dict]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `entry_orders=[]`, empty errors (non-blocking)

**ReflectionOutput** -- `make_reflection`
- `reflection: str`
- `trade_quality_scores: list[dict] = []`
- `attribution_contexts: dict = {}`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `reflection="Reflection skipped"`, empty scores/contexts (non-blocking)

### Research Graph Models (`src/quantstack/graphs/research/models.py`)

**ContextLoadOutput** -- `make_context_load`
- `context_summary: str`
- `regime: str`
- `regime_detail: dict`
- `hypothesis_attempts: int`
- `hypothesis_confidence: float`
- `hypothesis_critique: str`
- `queued_task_ids: list[str] = []`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `context_summary="Context unavailable"`, `regime="unknown"`, `regime_detail={}`, zeros for hypothesis fields

**DomainSelectionOutput** -- `make_domain_selection`
- `selected_domain: str`
- `selected_symbols: list[str]`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `selected_domain="swing"`, `selected_symbols=[]`

**HypothesisGenerationOutput** -- `make_hypothesis_generation`
- `hypothesis: str`
- `hypothesis_attempts: int`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `hypothesis=""`, `hypothesis_attempts` = current + 1 (note: safe_default cannot access state, so use a reasonable default like 1)

**HypothesisCritiqueOutput** -- `make_hypothesis_critique`
- `hypothesis_confidence: float`
- `hypothesis_critique: str`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `hypothesis_confidence=0.0`, `hypothesis_critique="Critique unavailable"`

**SignalValidationOutput** -- `make_signal_validation`
- `validation_result: dict`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `validation_result={"passed": False, "reason": "node_unavailable"}`

**BacktestValidationOutput** -- `make_backtest_validation`
- `backtest_id: str`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `backtest_id=""`

**MlExperimentOutput** -- `make_ml_experiment`
- `ml_experiment_id: str`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `ml_experiment_id=""`

**StrategyRegistrationOutput** -- `make_strategy_registration`
- `registered_strategy_id: str`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: `registered_strategy_id=""`

**KnowledgeUpdateOutput** -- `make_knowledge_update`
- `decisions: list[dict] = []`
- `errors: list[str] = []`
- safe_default: empty lists

**ValidateSymbolOutput** -- `make_validate_symbol` (uses `SymbolValidationState`)
- `validation_results: list[dict]`
- safe_default: `validation_results=[]`

**FilterResultsOutput** -- `make_filter_results`
- `validation_result: dict`
- `decisions: list[dict] = []`
- safe_default: `validation_result={"passed": False, "reason": "filter unavailable"}`

### Supervisor Graph Models (`src/quantstack/graphs/supervisor/models.py`)

**HealthCheckOutput** -- `make_health_check`
- `health_status: dict`
- `errors: list[str] = []`
- safe_default: `health_status={"overall": "unknown", "error": "node_unavailable"}`

**DiagnoseIssuesOutput** -- `make_diagnose_issues`
- `diagnosed_issues: list[dict]`
- `errors: list[str] = []`
- safe_default: `diagnosed_issues=[]`

**ExecuteRecoveryOutput** -- `make_execute_recovery`
- `recovery_actions: list[dict]`
- `errors: list[str] = []`
- safe_default: `recovery_actions=[]`

**StrategyPipelineOutput** -- `make_strategy_pipeline`
- `strategy_pipeline_report: dict`
- `errors: list[str] = []`
- safe_default: `strategy_pipeline_report={"skipped": True, "reason": "node_unavailable"}`

**StrategyLifecycleOutput** -- `make_strategy_lifecycle`
- `strategy_lifecycle_actions: list[dict]`
- `errors: list[str] = []`
- safe_default: `strategy_lifecycle_actions=[]`

**ScheduledTasksOutput** -- `make_scheduled_tasks`
- `scheduled_task_results: list[dict]`
- `errors: list[str] = []`
- safe_default: `scheduled_task_results=[]`

**EodDataSyncOutput** -- `make_eod_data_sync`
- `eod_refresh_summary: dict`
- `errors: list[str] = []`
- safe_default: `eod_refresh_summary={"skipped": True, "reason": "node_unavailable"}`

---

## Node Function Migration

After creating the output models, modify each node function to return an instance of its output model instead of a plain dict. The change per node is mechanical:

**Before:**
```python
return {
    "market_context": parsed,
    "decisions": [{"node": "market_intel", "mode": mode}],
}
```

**After:**
```python
return MarketIntelOutput(
    market_context=parsed,
    decisions=[{"node": "market_intel", "mode": mode}],
)
```

LangGraph accepts Pydantic model instances as node return values and extracts their fields for state merging the same way it does with dicts. No graph wiring changes are needed.

Error paths also return typed output models:

**Before:**
```python
except Exception as exc:
    return {
        "market_context": {},
        "errors": [f"market_intel: {exc}"],
    }
```

**After:**
```python
except Exception as exc:
    return MarketIntelOutput(
        market_context={},
        errors=[f"market_intel: {exc}"],
    )
```

This is a pure refactor: same behavior, better structure. Existing tests that construct state dicts directly do not need changes (the state merge happens at the LangGraph level, not in test fixtures).

---

## Ghost Fields Discovered During Audit

The `risk_sizing` node returns `alpha_signals` and `alpha_signal_candidates`, but these fields do not exist in the current `TradingState` TypedDict. Under the current TypedDict regime this is silently accepted. Under Pydantic with `extra="forbid"`, these returns will raise `ValidationError`.

**Resolution**: These fields must be added to `TradingState` in section-03 (Pydantic state migration). The output model `RiskSizingOutput` declares them, and the parent state must accept them. This is a concrete example of why the state key audit (section-02) must complete before this section.

---

## Tradeoffs

**Cost: Boilerplate.** Each node gets a model class with field declarations and a `safe_default()` method. Across 30+ nodes, this is significant new code. The payoff is compile-time-equivalent field safety that prevents an entire class of silent-merge bugs.

**Cost: Migration effort.** Every return statement in every node function changes. This is mechanical but touches many lines. The risk of introducing bugs during migration is mitigated by the fact that Pydantic will immediately reject any return that doesn't match the model schema -- bugs surface as loud `ValidationError` exceptions, not silent state corruption.

**Why not use TypedDict for output models instead of Pydantic?** TypedDict provides no runtime validation. A `TypedDict` output model would document intent but wouldn't catch the bugs we're trying to prevent. The whole point is runtime enforcement.

**Why one model per node instead of shared models?** Some nodes write overlapping fields (e.g., both `position_review` and `entry_scan` write to list fields). Shared models would either be too permissive (allowing writes to fields the node doesn't touch) or require complex inheritance hierarchies. One model per node is simple, explicit, and easy to audit.
