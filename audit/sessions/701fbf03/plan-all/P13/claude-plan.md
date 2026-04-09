# P13 Implementation Plan: Causal Alpha Discovery

## 1. Background

P01 (IC tracking) and P03 (ML pipeline) provide the statistical infrastructure for evaluating signals. P13 moves beyond correlation to causation — using DoWhy, EconML, and CausalML to discover causal factors, estimate treatment effects, and generate counterfactual analysis. Causal signals are more robust across regime changes because causation persists when correlation breaks down.

## 2. Anti-Goals

- **Do NOT replace existing correlation-based signals** — causal signals supplement, they don't replace. IC tracking decides the weight.
- **Do NOT require full causal graph for every signal** — start with targeted causal questions (insider buys → returns, earnings revisions → returns)
- **Do NOT build custom causal inference** — use DoWhy/EconML, well-tested libraries
- **Do NOT trust causal claims without robustness checks** — every causal estimate must pass refutation tests (placebo, subset, random common cause)
- **Do NOT deploy causal signals without regime stability validation** — must survive at least 2 regime transitions

## 3. Causal Discovery Engine

### 3.1 Causal Graph Builder

New `src/quantstack/core/causal/discovery.py`:
- Input: feature matrix (earnings revisions, insider trades, momentum, volume, etc.) + forward returns
- Method: PC algorithm (constraint-based) via DoWhy for initial structure discovery
- Output: DAG representing causal relationships between features and returns
- Validation: compare discovered DAG against domain knowledge (known causal paths)
- Storage: serialize DAG to `causal_graphs` table (JSONB adjacency list)

### 3.2 Feature-Return Causal Pairs

Priority causal hypotheses to test:
1. Earnings revision → 30-day return (mechanism: information incorporation)
2. Insider buy → 60-day return (mechanism: private information)
3. Short interest change → 20-day return (mechanism: supply/demand)
4. Analyst upgrade → 10-day return (mechanism: attention/flow)
5. Volume surge → 5-day return (mechanism: institutional activity)

## 4. Treatment Effect Estimation

### 4.1 Double Machine Learning (DML)

New `src/quantstack/core/causal/treatment_effects.py`:
- Use EconML's `LinearDML` and `CausalForestDML`
- Treatment: binary (insider_buy=1/0) or continuous (earnings_revision_pct)
- Outcome: forward return (5d, 10d, 30d)
- Confounders: sector, market_cap, momentum, volatility, regime
- Output: Average Treatment Effect (ATE) + Conditional ATE (CATE) per symbol

### 4.2 Propensity Score Matching

Fallback method when DML assumptions are questionable:
- Match treated units (e.g., stocks with insider buys) to control units with similar covariates
- Caliper matching with 0.05 tolerance on propensity score
- Verify covariate balance post-matching (standardized mean differences < 0.1)

### 4.3 Robustness Checks

Every causal estimate undergoes 3 refutation tests (DoWhy built-in):
1. **Placebo treatment**: randomize treatment assignment — effect should vanish
2. **Random common cause**: add random confounder — effect should persist
3. **Subset validation**: estimate on random 80% subset — effect should be stable

If any refutation fails (p < 0.05 for placebo, effect changes > 30%), mark factor as "unvalidated".

## 5. Causal Factor Library

### 5.1 Schema

`causal_factors` table:
- `factor_name`, `treatment_variable`, `outcome_horizon_days`
- `ate`, `ate_ci_lower`, `ate_ci_upper`, `ate_p_value`
- `refutation_placebo_p`, `refutation_subset_stability`
- `regime_stability_score` (survives how many regime transitions)
- `status`: `discovered`, `validated`, `active`, `retired`

### 5.2 Signal Collector

New `src/quantstack/signal_engine/collectors/causal.py`:
- For each active causal factor, compute CATE for current symbol
- Weight by ATE magnitude × regime stability × refutation confidence
- Output: `causal_signal` field in SymbolBrief
- Initial synthesis weight: 0.05, adjusted by IC like all other collectors

## 6. Counterfactual Analysis

### 6.1 Synthetic Control

New `src/quantstack/core/causal/counterfactual.py`:
- After trade exit, construct synthetic control (weighted average of non-traded similar stocks)
- Compare actual return vs synthetic control return
- Attribution: how much of the return was due to the trade vs market movement

### 6.2 Integration with Trade Journal

Extend trade outcome logging:
- `counterfactual_return`: what would have happened without the trade
- `causal_alpha`: actual_return - counterfactual_return
- Use for research prioritization (P10): strategies with high causal_alpha get more research time

## 7. Research Graph Integration

### 7.1 Causal Hypothesis Generator

Extend `quant_researcher` agent tools:
- `discover_causal_graph(features, returns)` → DAG
- `estimate_treatment_effect(treatment, outcome, confounders)` → ATE + CATE
- `run_counterfactual(trade_id)` → counterfactual analysis

### 7.2 Hypothesis Template

Research queue entries for causal hypotheses:
```
hypothesis: "{treatment} causes {outcome} because {mechanism}"
test: "DML with confounders [{confounders}], refutation via placebo + subset"
accept_if: "ATE > 0, p < 0.05, all refutations pass, regime_stability > 0.7"
```

## 8. Testing

- Causal discovery: synthetic dataset with known DAG → verify recovery
- DML: known treatment effect in synthetic data → verify ATE estimate within CI
- Refutation: intentionally non-causal correlation → verify placebo catches it
- Collector: mock causal factors → verify signal computation
- Counterfactual: known synthetic control → verify attribution accuracy
