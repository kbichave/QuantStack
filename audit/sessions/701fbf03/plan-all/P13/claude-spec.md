# P13 Spec: Causal Alpha Discovery

## Deliverables

### D1: Causal Discovery Engine
- PC algorithm via DoWhy for structure discovery
- Input: feature matrix + forward returns
- Output: DAG (adjacency list) stored in causal_graphs table
- Validation against domain knowledge (known causal paths)
- Batch job: run monthly overnight

### D2: Treatment Effect Estimation
- Double ML (EconML LinearDML + CausalForestDML) for ATE + CATE
- Propensity score matching as fallback method
- Priority treatments: earnings revision, insider buy, short interest, analyst upgrade, volume surge
- Confounders: sector, market_cap, momentum, volatility, regime

### D3: Robustness Checks
- 3 refutation tests per causal estimate: placebo treatment, random common cause, subset validation
- Rejection criteria: placebo p < 0.05 OR effect change > 30%
- Unvalidated factors stored but not used in signal

### D4: Causal Factor Library
- Schema: factor_name, treatment, outcome_horizon, ATE/CI/p-value, refutation results, regime stability
- Status lifecycle: discovered → validated → active → retired
- Signal collector: compute CATE for active factors, output causal_signal

### D5: Counterfactual Analysis
- Synthetic control for post-trade attribution
- causal_alpha = actual_return - counterfactual_return
- Integration with trade journal and research prioritization

### D6: Research Graph Integration
- Agent tools: discover_causal_graph, estimate_treatment_effect, run_counterfactual
- Hypothesis template for causal research queue entries

## Dependencies
- P01 (IC Tracking): IC infrastructure for causal signal evaluation
- P03 (ML Pipeline): model registry for CATE model storage
