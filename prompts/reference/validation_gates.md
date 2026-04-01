# Validation Gates (Step D)

Referenced by `research_shared.md` after Step C. Run gates in sequence before promoting any strategy.

**You are proving claims, not executing a checklist. Each gate states what must be true.
Use the right method for each question — Python imports or direct computation.
`run_backtest` is one option; `run_backtest_mtf`, `run_backtest_options`, `run_monte_carlo`,
`compute_information_coefficient`, `run_walkforward_mtf`, `finrl_evaluate_model`, and
`run_combinatorial_cv` are others. See `prompts/reference/python_toolkit.md` for all
available functions and their import paths.**

---

**Gate 0 — Register:** `register_strategy(...)` with composite rules from Step C.
MUST include `economic_mechanism`. Without it → draft-only, cannot promote.

---

**Gate 1 — Signal Validity** *(before spending compute on backtests)*

If ANY answer is "no" or "unknown", log as failed hypothesis and stop:
- **Does the signal predict returns?** IC between signal and forward returns at your intended horizon must be positive. IC < 0.02 = noise.
- **Is alpha decay consistent with holding period?** Half-life of signal must exceed holding_period_days. If peak IC is at a different horizon than designed, adjust the holding period to match.
- **Are features stationary?** Raw price levels create spurious correlations. Use returns, rolling z-scores, or spreads-from-mean. Non-stationary features → transform, don't proceed.
- **Do you have enough expected trades?** N >= (1.96/target_SR)² × 252/hold_days. Below minimum → exploratory only.

---

**Gate 1 — Per-Domain Thresholds**

| Domain | IC min | Alpha half-life |
|--------|--------|----------------|
| `equity_swing` | > 0.03, IC_IR > 0.5 | > holding period |
| `equity_investment` | positive | > 40 trading days |
| `options_swing` | VRP IC positive OOS | vol model beats GARCH (OOS R^2 > 0.05) |
| `options_weekly` | (skip — trade count gates) | N/A |

---

**Gate 2 — In-Sample Performance**

Prove IS performance using whatever backtest tools fit the strategy type:
- IS Sharpe meets domain threshold (table below)
- Trade count meets statistical minimum from Gate 1
- P&L attribution: which rules/signals contribute? Which are noise? Drop noise rules.
- Multi-timeframe consistency: does the edge hold across timeframes if applicable?

| Domain | IS Sharpe | Min Trades | Holding Period |
|--------|-----------|-----------|---------------|
| `equity_swing` | > 0.8 | >= 100 | 3-10 days |
| `equity_investment` | > 0.5 | >= 100 | 20-120 days |
| `options_swing` | > 0.6 | >= 60 | avg DTE at exit > 2 |
| `options_weekly` | > 0.4 | >= 200 | avg hold < 5 days |

---

**Gate 3 — Out-of-Sample Consistency** *(mandatory before promotion)*

If ANY triggers a red flag, log as negative result and stop.
Thresholds come from `params["kill_thresholds"][instrument_type]`:
- **OOS Sharpe** meets domain threshold (table below) across folds and across 3+ symbols
- **IS/OOS ratio < `max_is_oos_ratio`** (default 2.5) — more than this degradation = fragility, not alpha
- **PBO < `max_pbo`** (default 0.50) — use combinatorial cross-validation. Above threshold = more likely overfit than real. DELETE.
- **Deflated Sharpe > 0** — account for N hypotheses tested this cycle. DSR <= 0 = selection bias explains the Sharpe. DELETE.
- **No data leakage** — any feature using future information produces spectacular backtests and zero live performance. If detected, INVESTIGATE then DELETE.

| Domain | OOS Sharpe | Win Rate | PBO | Additional |
|--------|-----------|----------|-----|-----------|
| `equity_swing` | > 0.7 | — | < 0.40 | consistent 3+ symbols |
| `equity_investment` | > 0.5 | > 55% | < 0.40 | beats SPY alpha-adjusted |
| `options_swing` | > 0.6 | > 55% (premium selling) | < 0.40 | EV > 0 for directional |
| `options_weekly` | > 0.4 | > 60% (credit) | < 0.45 | — |

---

**Gate 4 — Robustness** *(mandatory for promotion)*

- **Cost sensitivity:** Sharpe still meets threshold at 2x assumed slippage. If not, the strategy has no real edge after execution costs.
- **Stress test:** Max drawdown within domain limits during worst 5% of historical periods.
- **Regime stability:** Does it hold across the regimes it claims to target?
- *(Options only)* Greeks under worst-case: delta/gamma/vega/theta at entry AND at underlying +/-2 ATR, VIX +50%.

| Domain | Sharpe at 2x slip | Max Drawdown | Additional |
|--------|-------------------|-------------|-----------|
| `equity_swing` | > 0.5 | < 15% | — |
| `equity_investment` | > 0.5 | < 20% | positive idiosyncratic alpha (not just factor beta) |
| `options_swing` | > 0.5 (2x bid-ask) | single-trade < 3% equity | survives VIX +50%; Greeks within limits at +/-2 ATR |
| `options_weekly` | > 0.3 (2x bid-ask) | single-trade < 2% equity | gamma-aware: daily delta rebalance cost modeled |

---

**Gate 5 — ML/RL Lift** *(strongly preferred; not a hard block)*

Use whatever ML and RL tools best answer these questions:
- Does a supervised model improve on the rule-based signal? Search available tools — classification, regression, ensemble, stacking, RL sizing/execution agents are all options.
- Log SHAP/feature importance to `breakthrough_features`. One variable at a time (Rule 9).
- Champion vs challenger: if a model already exists for this symbol, beat it or retire it.
- RL agents for execution timing or position sizing if sufficient trade history exists.

**CausalFilter (MANDATORY for ML-backed strategies before walk-forward):**

Validate that features Granger-cause returns before spending compute on full validation:

```python
from quantstack.core.validation.causal_filter import CausalFilter

causal = CausalFilter(max_lag=5, significance_level=0.05)
X_filtered = causal.fit_transform(features_df, forward_returns)
result = causal.get_result()

drop_rate = len(result.dropped_features) / (len(result.surviving_features) + len(result.dropped_features))

if drop_rate > 0.30:
    # >30% of features are non-causal — hypothesis needs rework
    pass
```

- drop_rate > 0.30: re-evaluate the feature set before proceeding. Log surviving vs dropped features.
- Lagged-price-only top SHAP features after filtering: autocorrelation artefact — discard.
- **Do not register an ML model directly as a strategy.** Convert top SHAP features into auditable entry rules, then proceed through Gates 2-4.
- Skipped for pure rule-based strategies (RSI/SMA-only with no ML component).

---

**Gate 6 — Update**

- Update strategy status in DB
- Write findings to memory files (strategy_registry, ml_model_registry, ml_experiment_log, workshop_lessons)

---

## Incremental Promotion Tiers

A strategy does NOT need to pass all gates before moving forward:

| Status | Criteria |
|--------|----------|
| `draft` | Gate 0 (register) + Gate 1 (signal validity) pass. Gates 2-5 pending. |
| `forward_testing` | Gates 0-4 pass (4 of 6). Missing gate must be documented. Forward testing results replace the missing gate after 20 live trades. |
| `live` | All gates pass (6 of 6). No exceptions. |

Rationale: a strategy failing one robustness gate (e.g., Gate 4 at 2x slippage just misses) but passing everything else has unknown live-trading value. Forward testing is cheaper than the token cost of re-running validation. Let live data resolve marginal gates.

**Pre-Promotion Checklist** — answer all before calling `promote_strategy` or setting `status="forward_testing"`:

- [ ] Total trades > 50 in backtest
- [ ] Walk-forward OOS Sharpe > 0 in majority of folds
- [ ] No parameter was tuned to a single data point
- [ ] Strategy logic is explainable in one sentence
- [ ] Entry/exit rules use different indicators (not the same twice)
- [ ] Risk params include stop loss (no open-ended risk)
- [ ] If ML-backed: features passed CausalFilter (Granger causality at p<0.05 after Bonferroni)
- [ ] Document any gates that did NOT pass and why forward testing is the resolution path

---

**On failure at any gate:** Diagnose the specific failure mode before moving on.
"OOS Sharpe low" is not a diagnosis. "OOS Sharpe low because signal decays by day 8 but holding period is 20 days" is. Log the specific root cause.

**Short-history symbols** (< 504 bars, e.g., recent IPOs): the validation tools auto-adjust parameters when data is insufficient — proceed with their suggested params and document the wider confidence intervals as a limitation.
