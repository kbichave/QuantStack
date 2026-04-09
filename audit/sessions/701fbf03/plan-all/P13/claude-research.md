# P13 Research: Causal Alpha Discovery

## Codebase Research

### What Exists
- **IC Attribution**: `src/quantstack/learning/ic_attribution.py` — Spearman rank correlation between signals and returns
- **ML Pipeline (P03)**: model registry, walk-forward validation, feature importance
- **Feature importance**: `src/quantstack/ml/feature_importance.py` — SHAP/permutation importance (correlation-based, not causal)
- **Signal collectors**: 27+ collectors providing features that could be tested for causal relationships
- **Research graph**: hypothesis generation and validation pipeline

### What's Needed (Gaps)
1. **Causal discovery engine**: No DAG learning infrastructure — need DoWhy + PC algorithm
2. **Treatment effect estimation**: No DML/propensity matching — need EconML integration
3. **Robustness checks**: No causal refutation tests — need placebo, subset, random common cause
4. **Causal factor library**: No storage for validated causal factors
5. **Counterfactual analysis**: No synthetic control methods for post-trade attribution
6. **Research graph tools**: No causal hypothesis tools for agents

## Domain Research

### Causal Inference in Finance
- Key challenge: observational data only (can't randomize stock prices)
- DoWhy framework handles this via do-calculus and instrumental variables
- Double ML (EconML) is the current gold standard for treatment effects in observational data
- Confounders in finance: sector, market cap, momentum, volatility — all must be controlled for

### Practical Implementation Notes
- PC algorithm for structure discovery requires N >> number of variables (need 500+ observations per edge)
- DML training on rolling windows (252 days) provides sufficient data for 5-10 treatment variables
- Refutation tests catch spurious causal claims — essential quality gate
- Causal factors that survive regime changes are genuinely valuable (causation ≠ correlation)

### Key Packages
- `dowhy>=0.11`: Microsoft's causal inference framework — structure discovery + effect estimation
- `econml>=0.15`: Microsoft's economic ML — DML, causal forests, instrumental variables
- `causalml>=0.15`: Uber's uplift modeling — treatment effect estimation
- Note: `pgmpy` for graphical models is optional — DoWhy handles most needs
