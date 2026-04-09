# P13: Causal Alpha Discovery

**Objective:** Move beyond correlation-based alpha to causal inference — discover WHY prices move, not just what correlates with moves. Enables more robust, regime-stable signals.

**Scope:** New: core/causal/, research/

**Depends on:** P01 (IC tracking), P03 (ML pipeline)

**Effort estimate:** 2 weeks

---

## What Changes

### 13.1 Causal Discovery
- Use `DoWhy` + `CausalML` to discover causal relationships between features and returns
- Distinguish: "earnings revisions cause price moves" vs "momentum correlates with returns"
- Causal signals are more robust across regimes (causation doesn't break when correlation does)

### 13.2 Treatment Effect Estimation
- Estimate: "What is the causal effect of an insider buy on 30-day returns?"
- Use propensity score matching or double ML (DML) for unbiased estimates
- Build causal factor library: factors with proven causal links to returns

### 13.3 Counterfactual Analysis
- "What would have happened if we DIDN'T enter this trade?"
- Use synthetic control methods for post-trade counterfactual
- Enables genuine performance attribution (not just benchmark comparison)

### 13.4 Integration with Research Graph
- Add `causal_researcher` agent or extend `quant_researcher`
- Hypothesis template: "X causes Y because [mechanism], testable via [natural experiment]"
- Validation: causal effect must survive regime change (not just statistical significance)

## Key Packages
- `dowhy` (Microsoft) — causal inference framework
- `causalml` (Uber) — uplift modeling, treatment effects
- `econml` (Microsoft) — double ML, causal forests
- `pgmpy` — probabilistic graphical models for causal structure

## Acceptance Criteria

1. Causal graph discovered between features and returns
2. At least 5 causal factors validated with treatment effect > 0
3. Causal signals survive regime change better than correlation-based signals
4. Research graph generates causal hypotheses
