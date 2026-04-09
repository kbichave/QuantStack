# P13 Self-Interview: Causal Alpha Discovery

## Q1: How does causal discovery differ from the existing feature importance analysis?
**A:** Feature importance (SHAP, permutation) answers "which features are correlated with the model's predictions" — it's still correlation-based. Causal discovery answers "which features CAUSE price movements" — using do-calculus to identify genuine causal links. The practical difference: a causal factor should maintain predictive power across regime changes (because the causal mechanism persists), while a correlated feature may lose predictive power when the regime shifts.

## Q2: What's the minimum data requirement for reliable causal discovery?
**A:** For the PC algorithm (structure discovery): N should be >500 per variable for reliable edge detection. With 252 trading days/year, we have ~2 years of daily data for reliable discovery on 5-10 variables. For DML treatment effects: 252 observations per rolling window is sufficient for single-treatment estimation with 5-10 confounders.

## Q3: How do you handle the problem of unobserved confounders in financial data?
**A:** Three approaches: (a) sensitivity analysis — test how strong an unobserved confounder would need to be to invalidate the causal estimate; (b) instrumental variables — when available (e.g., analyst coverage changes as instrument for information flow); (c) conservative interpretation — treat estimates as upper bounds on causal effect. DoWhy's refutation tests specifically test robustness to unobserved confounders.

## Q4: How does the causal signal collector compute signals in real-time?
**A:** It doesn't compute causal graphs in real-time. Causal discovery and treatment effect estimation run as batch jobs (overnight). The causal factor library stores validated factors with their CATE models. At signal collection time, the collector looks up active causal factors for the symbol, runs the pre-trained CATE model on current features, and returns the conditional treatment effect as the signal value.

## Q5: What's the expected IC range for causal signals?
**A:** Similar to or slightly better than existing signals: IC 0.03-0.08. The value isn't necessarily higher IC — it's more STABLE IC. Causal signals should maintain their IC across regime changes while correlation-based signals degrade. The metric to track is IC stability (variance across regimes), not just IC level.

## Q6: How do counterfactual analysis results feed back into the system?
**A:** Two feedback paths: (a) `causal_alpha` (actual_return - counterfactual_return) feeds into research prioritization (P10) — strategies with high causal alpha get more research time; (b) counterfactual attribution feeds into the learning loop — if a trade's return was mostly market movement (low causal alpha), the strategy doesn't get credit for it.

## Q7: What happens when the causal discovery engine finds no significant causal links?
**A:** This is a valid and informative result. It means the features tested don't have detectable causal effects on returns at the specified horizon. Action: (a) try different time horizons (5d vs 30d vs 60d), (b) try sector-specific analysis (insider buys may cause returns in small-cap but not mega-cap), (c) accept that some features are only correlated, not causal — they still have IC value, just less regime-stable.

## Q8: How do you version and track causal graphs as they evolve?
**A:** Store each causal graph snapshot with date in `causal_graphs` table (JSONB adjacency list + metadata). Compare monthly: which edges appeared, disappeared, or changed strength. This provides a measure of causal stability — edges that persist across many snapshots are more trustworthy. Edges that appear and disappear are likely spurious.
