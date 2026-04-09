# Section 07: Research Graph Integration

## Objective

Expose causal discovery, treatment effect estimation, and counterfactual analysis as LLM-facing tools for the research graph agents. Add a causal hypothesis template to the research queue.

## Dependencies

- Section 05 (Causal Signal Collector) -- tools wrap the causal infrastructure.
- Section 06 (Counterfactual Analysis) -- `run_counterfactual` tool.

## Files to Create

### `src/quantstack/tools/langchain/causal_tools.py`
- **Tool `discover_causal_graph`**:
  - Input: `features` (list of feature column names), `returns_horizon` (int, default 30), `symbol` (optional, for symbol-specific analysis).
  - Fetches feature matrix from DataStore, runs `CausalGraphBuilder.build_graph()`.
  - Validates against domain priors.
  - Stores result in `causal_graphs` table.
  - Returns: human-readable summary of discovered edges, domain agreement score, and the graph adjacency list.

- **Tool `estimate_treatment_effect`**:
  - Input: `treatment` (str), `outcome` (str, default "forward_return"), `outcome_horizon` (int), `confounders` (list[str], optional), `method` (str, "dml" or "psm").
  - Runs `TreatmentEffectEstimator.estimate_dml()` or `estimate_psm()`.
  - Runs `CausalRefuter.run_all()` for robustness checks.
  - If all refutations pass, registers factor in `CausalFactorLibrary`.
  - Returns: ATE, CI, p-value, refutation results, factor status.

- **Tool `run_counterfactual_analysis`**:
  - Input: `trade_id` (int).
  - Runs `run_counterfactual()` from Section 06.
  - Returns: actual return, counterfactual return, causal alpha, p-value, control weights.

- **Tool `list_causal_factors`**:
  - Input: `status` (optional filter: "active", "validated", "all").
  - Returns: list of causal factors with their ATEs, confidence intervals, regime stability, and status.

### `src/quantstack/tools/langchain/causal_models.py`
- Pydantic input/output models for each tool (following the pattern in `tools/models.py`):
  - `DiscoverCausalGraphInput`, `DiscoverCausalGraphOutput`
  - `EstimateTreatmentEffectInput`, `EstimateTreatmentEffectOutput`
  - `RunCounterfactualInput`, `RunCounterfactualOutput`
  - `ListCausalFactorsInput`, `ListCausalFactorsOutput`

## Files to Modify

### `src/quantstack/tools/registry.py`
- Import and register the four new causal tools in `TOOL_REGISTRY`:
  ```python
  from quantstack.tools.langchain.causal_tools import (
      discover_causal_graph,
      estimate_treatment_effect,
      run_counterfactual_analysis,
      list_causal_factors,
  )
  ```
- Add to the appropriate registry section (Research tools).

### `src/quantstack/graphs/research/config/agents.yaml`
- Add causal tools to the `quant_researcher` agent's tool list:
  ```yaml
  - discover_causal_graph
  - estimate_treatment_effect
  - run_counterfactual_analysis
  - list_causal_factors
  ```

### Research Queue Hypothesis Template
- Define the causal hypothesis template format for research queue entries (in the research graph node that generates hypotheses, or in a config):
  ```
  hypothesis: "{treatment} causes {outcome} because {mechanism}"
  test: "DML with confounders [{confounders}], refutation via placebo + subset"
  accept_if: "ATE > 0, p < 0.05, all refutations pass, regime_stability > 0.7"
  ```

## Implementation Details

1. **Tool Design Pattern**: Follow existing patterns in `tools/langchain/ml_tools.py`:
   - Use `@tool` decorator from langchain.
   - Pydantic input model with `model_config = {"json_schema_extra": {...}}` for examples.
   - Return structured dict that the LLM can parse.
   - Handle all errors gracefully with informative messages (no raw tracebacks).

2. **discover_causal_graph Workflow**:
   - Fetch feature data for the specified features + forward returns.
   - Build the causal graph via `CausalGraphBuilder`.
   - Validate against domain priors (hardcoded known causal paths).
   - Store in DB, return summary.

3. **estimate_treatment_effect Workflow**:
   - Fetch data, run DML or PSM estimation.
   - Automatically run all three refutations.
   - If refutations pass AND p < 0.05: register in CausalFactorLibrary as `validated`.
   - If refutations fail: still return results but mark as `unvalidated` with explanation.

4. **Agent YAML Binding**: Tools are bound by string name in the YAML config. The registry resolves names to tool objects at graph build time. No code changes needed beyond registry + YAML.

5. **Hypothesis-Driven Research**: The research graph should use `discover_causal_graph` first to identify candidate edges, then `estimate_treatment_effect` for each promising edge. This creates a systematic pipeline: discovery -> estimation -> validation -> activation.

## Test Requirements

- **Tool invocation**: Call each tool with valid inputs and verify structured output format.
- **Error handling**: Call with invalid inputs (unknown treatment variable, bad trade_id). Verify graceful error messages.
- **Registry integration**: Verify all four tools appear in `TOOL_REGISTRY` and can be resolved by name.
- **End-to-end**: Discover graph -> estimate effect -> verify factor registered in library.

## Acceptance Criteria

- [ ] Four LLM-facing tools created with proper `@tool` decorators and Pydantic I/O models
- [ ] All tools registered in `TOOL_REGISTRY` and resolvable by string name
- [ ] `quant_researcher` agent YAML updated with causal tools
- [ ] Tools return structured, LLM-parseable output
- [ ] Error handling is graceful (no raw exceptions to LLM)
- [ ] End-to-end flow works: discover -> estimate -> refute -> register factor
- [ ] Tests pass: `uv run pytest tests/unit/tools/test_causal_tools.py`
