# Integration Notes — Opus Review Feedback

## Integrating

1. **Migrate StrategyBreaker + ICAttributionTracker to PostgreSQL** — Correct. JSON persistence violates the "DB writes use db_conn()" hard rule and risks state loss on container restart. Safety-critical components must use PostgreSQL. Integrating.

2. **Replace tiered IC factor with continuous sigmoid** — Good catch on boundary oscillation. `ic_factor = sigmoid(50 * (ic - 0.02))` gives smooth S-curve. Integrating.

3. **Make LLM failure classification async** — Correct. Trade close hook should never block on LLM. Queue for background classification. Integrating.

4. **Use filtered probabilities for regime transitions** — Critical fix. `model.predict_proba(X)` gives time-varying state uncertainty vs static `transmat_`. Integrating.

5. **Add cold-start/bootstrap logic for each feedback loop** — Fair point. Each item needs explicit "insufficient data → default to X" behavior. Integrating.

6. **Add total sizing factor floor** — Good. Compound of 5-6 multiplicative factors can produce micro-orders. Add minimum tradeable threshold. Integrating.

7. **Add kill-switch config flags per feedback loop** — Pragmatic. Each adjustment should be independently toggleable. Integrating.

8. **Add rollback paths per section** — Correct for autonomous system. Each section needs a one-line revert. Integrating.

9. **Concrete OutcomeTracker decay halflife** — Fair. Specify 20-trade exponential halflife (balances responsiveness vs stability). Integrating.

10. **Wire 5 survivorship bias in IC measurement** — Important nuance. Backfilling only traded symbols skews IC. Use the nightly `run_ic_computation()` (which computes cross-sectional IC across all symbols with signals) as the primary IC source. ICAttributionTracker becomes supplementary per-trade feedback, not the primary IC source. Integrating.

## NOT Integrating

1. **Champion/challenger shadow period 60 days** — The review suggests 60 days to match research. I'm keeping 30 days as the minimum for promotion eligibility but noting that 60 days is recommended for statistical confidence. The 30-day minimum is already a compromise between the spec's 21 days and research's 60. With a small universe and infrequent model retraining, waiting 60 days creates an unreasonably long feedback loop. The 60-day retirement deadline still applies.

2. **Reorder Model Versioning to later** — The review suggests Week 1-2 time on model versioning would be better on signal intelligence. Disagree: model versioning is a prerequisite for safe model updates, and building it early means every subsequent model operation (including drift-triggered retraining) is properly versioned from the start. Building it after drift detection means the first auto-retrain has no versioning safety net.

3. **Signal correlation lookback 126+ days** — The review notes 63 days gives wide confidence intervals. True, but 126 days means the correlation matrix takes 6 months to populate and is less responsive to correlation regime changes. Keep 63 days but add the softer penalty suggestion (scale penalty linearly with correlation above 0.5 instead of hard 0.7 threshold). This is a partial integration.
