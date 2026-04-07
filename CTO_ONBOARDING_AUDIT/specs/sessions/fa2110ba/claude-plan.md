# Implementation Plan — Phase 7: Feedback Loops & Learning

---

## Overview

QuantStack is an autonomous trading system built on three LangGraph StateGraphs (Research, Trading, Supervisor) that research strategies, execute trades, and monitor health. The system has a critical gap: **six fully-implemented learning modules with zero consumers**. Losses are recorded in multiple tables but never read downstream — position sizing the next morning is identical, as if the loss never happened.

This plan wires those ghost modules into the live system, adds failure classification, connects IC degradation to signal weights, implements concept drift detection beyond PSI, builds model versioning with champion/challenger, and adds regime transition detection. The result: every loss improves the next decision.

**Scope:** 12 items (7.1–7.12), organized into 3 parallel implementation streams.

---

## Architecture Context

**Key systems touched:**

- **Signal Engine** (`src/quantstack/signal_engine/`): 14+ collectors → `RuleBasedSynthesizer` → `SymbolBrief`. Static regime-conditional weights. Additive conviction adjustments.
- **Trading Graph** (`src/quantstack/graphs/trading/nodes.py`, 1185 lines): `risk_sizing` (lines 452-591), `execute_entries` (687-730), `daily_plan` (229-284).
- **Learning modules** (`src/quantstack/learning/`): OutcomeTracker, SkillTracker (421 lines), ICAttributionTracker (420 lines), ExpectancyEngine (98 lines), drift_detector (312 lines).
- **StrategyBreaker** (`src/quantstack/execution/strategy_breaker.py`, 553 lines): ACTIVE → SCALED (0.5) → TRIPPED (0.0) state machine.
- **EventBus** (`src/quantstack/coordination/event_bus.py`): PostgreSQL-backed, poll-based, per-consumer cursors. Already has `IC_DECAY`, `DEGRADATION_DETECTED`, `MODEL_DEGRADATION`, `REGIME_CHANGE`.
- **Trade hooks** (`src/quantstack/hooks/trade_hooks.py`): `on_trade_close()`, `on_trade_fill()`, `on_daily_close()`.
- **Supervisor batch** (`src/quantstack/graphs/supervisor/nodes.py`): Nightly scheduled tasks including `run_ic_computation()`, `run_signal_scoring()`, `run_ic_retirement_sweep()`.

**IC tracking state:** The `signal_ic` table is populated nightly with cross-sectional rank IC per strategy (5/10/21-day horizons). However, per-collector IC (`ICAttributionTracker`) and per-agent IC (`SkillTracker.record_ic()`) are dead code with zero data flow. Wiring these is a prerequisite for several items.

---

## Stream A: Core Wiring (Sequential: 7.1 → 7.2 → 7.6)

### Section 1: Ghost Module API Audit

Before wiring any module, audit and fix each API's math and thresholds. The decision is to fix obvious issues during integration rather than wire broken formulas.

**OutcomeTracker** — The current affinity update formula `new_affinity = clip(current + 0.05 * tanh(pnl_pct / 5.0), 0.1, 1.0)` is too slow to adapt: a -2% loss produces `tanh(-0.4) ≈ -0.38`, so the step is `0.05 * -0.38 = -0.019`. Starting from affinity 1.0, it takes ~47 consecutive losses of this magnitude to reach the floor of 0.1. This is not a meaningful feedback signal.

**Fix:** Increase the step multiplier to 0.15 and use a steeper tanh divisor of 2.0. This makes a -2% loss produce a step of ~0.11, reaching meaningful reduction in ~8 losses. Add a recency-weighted exponential decay with a **20-trade halflife** — each outcome's contribution decays by `0.5^(trades_since / 20)`. This means the last 20 trades contribute ~50% of the affinity signal, preventing stale outcomes from anchoring the value while smoothing single-trade noise. The floor of 0.1 and ceiling of 1.0 should remain — no regime should reach zero affinity.

**Cold-start:** With < 5 outcomes for a regime, do not adjust affinity — default to 1.0 (full allocation). The learning signal is too noisy with fewer observations.

**SkillTracker** — The `get_confidence_adjustment()` formula is reasonable (clamps 0.5–1.5) but has a flaw: the ICIR adjustment uses `icir * 0.2` which is unbounded before the outer clamp. With ICIR of 3.0 (high but possible), the adjustment would be 0.6, well above the max 0.3. The outer `max(-0.2, min(0.3, ...))` catches this, but the intent is unclear. Simplify to `min(0.3, icir * 0.15)` for clarity.

**ICAttributionTracker** — The API is clean. Verify `scipy` is in project dependencies for Spearman correlation. The `get_weights()` method normalizes by IC > 0, which is correct. Add a minimum observation count check (currently defaults to 20, which is good).

**ExpectancyEngine** — This module duplicates `core/kelly_sizing.py` functionality. Rather than wire it, leave it dormant. The Kelly sizing in `regime_kelly_fraction()` already incorporates IC, which is more principled. Mark ExpectancyEngine as deprecated with a TODO for removal.

**StrategyBreaker** — The state machine (ACTIVE → SCALED → TRIPPED) and thresholds (5% DD, 3 consecutive losses) are reasonable. The 24h cooldown on TRIPPED is appropriate for swing-trading timeframes. The `force_trip()` and `force_scale()` methods from degradation_enforcer are good escalation paths. **Critical fix: migrate persistence from `~/.quantstack/strategy_breakers.json` to PostgreSQL.** JSON file persistence violates the "DB writes use `db_conn()` context managers" hard rule and risks state loss on Docker container restart — a TRIPPED strategy would silently resume trading. Add a `strategy_breaker_states` table and convert the save/load methods to use `db_conn()`.

**ICAttributionTracker** also needs the same migration — its `~/.quantstack/ic_attribution.json` persistence has the same container restart risk. Add an `ic_attribution_data` table.

**TradeEvaluator** — The 6-dimension LLM scoring (execution_quality, thesis_accuracy, risk_management, timing_quality, sizing_quality, overall_score) is well-structured. The read pattern needs definition: the daily planner should query aggregated patterns, not individual scores. Add a summary function that computes rolling averages per dimension and identifies consistent weaknesses.

### Section 2: Wire 6 Ghost Module Readpoints (7.1)

Six specific integration points, each a small change in an existing file.

**Wire 1: `get_regime_strategies()`** — In `tools/langchain/meta_tools.py`, replace the stub that returns `{"error": "Tool pending implementation"}` with a real implementation. Query the `strategies` table, read the `regime_affinity` JSONB column, filter by the requested regime, and return strategies sorted by affinity score descending. Include strategy status (ACTIVE/SCALED/TRIPPED from StrategyBreaker) in the response. This gives LLM agents the ability to ask "which strategies work in this regime?" and get a real answer.

**Wire 2: StrategyBreaker in `risk_sizing`** — In `trading/nodes.py` within the `risk_sizing` node (around line 550, after the Kelly fraction computation), add a call to `strategy_breaker.get_scale_factor(strategy_id)`. Multiply the computed `alpha_signal` amount by this factor. When a strategy is SCALED (0.5), positions are halved. When TRIPPED (0.0), the signal is zeroed out and won't generate an order. This is the key circuit breaker — a strategy that's losing stops getting capital.

**Defensive bounds check:** If `get_scale_factor()` raises an exception or returns a value outside [0.0, 1.0], default to 1.0 (fail-open — the risk gate is downstream) and log an error. This prevents a corrupted breaker state from halting all trading or amplifying positions.

**Wire 3: StrategyBreaker in `execute_entries`** — In `trading/nodes.py` within `execute_entries` (around line 695), before placing each order, check `strategy_breaker.get_scale_factor(strategy_id)`. If TRIPPED (0.0), skip the order entirely and log the skip with the breaker reason. If SCALED, the sizing reduction already happened in Wire 2, so just log the scaled status.

**Wire 4: SkillTracker in trade hooks** — In `trade_hooks.py::on_trade_close()`, after the existing ReflectionManager call, call `skill_tracker.update_agent_skill(agent_name, prediction_correct, signal_pnl)`. The `agent_name` comes from the trade's `debate_verdict` or execution context. The `prediction_correct` boolean is true if the trade was profitable. This starts populating per-agent win rates.

**Wire 5: ICAttribution in signal engine** — In `signal_engine/engine.py`, after `run()` completes synthesis, iterate over each collector's contribution and call `ic_attribution.record(symbol, collector_name, signal_value, forward_return=None)`. The `forward_return` is unknown at synthesis time — it gets backfilled when the trade closes (new hook in `on_trade_close()`). This creates the signal→outcome pairs that ICAttributionTracker needs to compute per-collector IC.

**Survivorship bias mitigation:** Per-trade IC backfill only covers traded symbols, not the full signal universe. To avoid bias, use the nightly `run_ic_computation()` (which computes cross-sectional IC across all symbols with signals in the `signals` table) as the **primary IC source** for weight adjustments (Section 5). ICAttributionTracker becomes a **supplementary per-trade feedback** channel that adds granularity but doesn't drive weights alone. The nightly cross-sectional IC is unbiased because it covers all signaled symbols regardless of whether a trade was taken.

**Wire 6: Trade quality in daily_plan** — In `trading/nodes.py` within the `daily_plan` node, add a DB query to `trade_quality_scores` that computes rolling 30-trade averages per dimension. Identify the weakest dimension (lowest average score). Include this in the daily plan prompt as context: "Recent trade quality analysis shows {weakness_dimension} is consistently low ({avg_score}/10). Focus on improving {specific_guidance}."

### Section 3: Failure Mode Taxonomy (7.2)

Currently, all losses > 1% are routed to the research queue as generic `bug_fix` tasks. This makes every loss look identical to the research system.

**Failure mode enum:** Add to a new file `src/quantstack/learning/failure_taxonomy.py`:

```python
class FailureMode(str, Enum):
    REGIME_MISMATCH = "regime_mismatch"
    FACTOR_CROWDING = "factor_crowding"
    DATA_STALE = "data_stale"
    TIMING_ERROR = "timing_error"
    THESIS_WRONG = "thesis_wrong"
    BLACK_SWAN = "black_swan"
    UNCLASSIFIED = "unclassified"
```

**Hybrid classification approach:**

Rule-based classifier runs first with deterministic checks:
- `REGIME_MISMATCH`: regime at entry differs from regime at exit (already stored in `strategy_outcomes`)
- `DATA_STALE`: any key data source was stale at entry time (check data freshness timestamps)
- `BLACK_SWAN`: loss magnitude > 3 standard deviations from strategy's historical loss distribution
- `TIMING_ERROR`: entry was within 2 bars of a key level (support/resistance from daily plan)

If rule-based produces `UNCLASSIFIED`, queue an **asynchronous** LLM classification task. The trade close hook must never block on an LLM call — fire-and-forget the classification request. Use haiku tier (cost-efficient). If the LLM result arrives before the nightly loss aggregation batch, great. If not, the loss aggregation treats it as `UNCLASSIFIED` and the LLM result backfills later. The trade context (entry rationale, market conditions, signal values) is passed for classification into FACTOR_CROWDING or THESIS_WRONG.

**Schema change:** Add `failure_mode TEXT` column to `strategy_outcomes` table.

**Research queue enhancement:** Replace the hardcoded `task_type='bug_fix'` in `trade_hooks.py` with the classified failure mode. Change priority computation from the current binary (5 or 7) to: `priority = min(9, int(cumulative_loss_30d * recency_weight * 10))` where `recency_weight` decays at 0.95^days_ago. This ensures persistent failure patterns get higher priority than isolated losses.

### Section 4: Loss Aggregation in Supervisor (7.6)

New supervisor batch node: `run_loss_aggregation()`.

**Schedule:** Daily at 16:30 ET (after market close, before overnight research).

**Logic:**
1. Query `strategy_outcomes` for losses in the trailing 30 days, joined with `failure_mode`
2. Group by failure mode, then by strategy, then by symbol
3. Compute cumulative P&L impact per group
4. Rank groups by absolute P&L impact
5. For top 3 failure patterns: auto-generate targeted research tasks with the failure mode as task type, the affected strategies/symbols as context, and priority based on cumulative impact
6. Store the daily aggregation snapshot in a new `loss_aggregation` table for trend analysis

**New DB table: `loss_aggregation`** — Columns: date, failure_mode, strategy_id, symbol, trade_count, cumulative_pnl, avg_loss_pct, rank.

---

## Stream B: Signal Intelligence (7.3, 7.7, 7.8, 7.9)

### Section 5: IC Degradation → Weight Adjustment (7.3)

The signal engine uses static regime-conditional weights. Even when a collector's IC drops to zero, it still gets its static weight in synthesis. The decision is to keep static weights as priors and multiply by IC-derived factors.

**Weight adjustment formula:**

```
effective_weight(collector, regime) = static_weight(collector, regime) * ic_factor(collector)
```

Where `ic_factor` is a **continuous sigmoid function** of the collector's rolling 21-day IC:

```
ic_factor(ic) = 1 / (1 + exp(-50 * (ic - 0.02)))
```

This gives a smooth S-curve centered at IC=0.02: full weight above ~0.04, near-zero below ~0.00, and a smooth transition in between. This avoids the boundary oscillation problem that discrete tiers would cause — a collector hovering around IC=0.02 gets a factor of ~0.5 instead of flipping between 0.5 and 0.8 on successive days.

Additionally, if IC_IR (mean IC / std IC) < 0.1, apply a further penalty of 0.7× to account for inconsistency.

**Floor check:** After applying IC factors to all collectors, verify that total effective weight > 0.1. If all collectors have been driven to near-zero (data quality issue), fall back to equal static weights and publish a `SIGNAL_DEGRADATION` alert. This prevents division-by-zero or NaN conviction from a pathological IC collapse.

**Implementation location:** Modify `signal_engine/synthesis.py` to accept an optional `ic_adjustments: dict[str, float]` parameter. When provided, multiply each collector's static weight by its IC factor before normalizing. The IC factors are computed by a new helper that reads from ICAttributionTracker (wired in Section 2, Wire 5).

**EventBus integration:** When a collector's IC drops below 0.02 (from a previously healthy level), publish a `SIGNAL_DEGRADATION` event with payload `{collector, current_ic, previous_ic, regime}`. Add `SIGNAL_DEGRADATION` to the `EventType` enum. The research graph polls for these events and queues an investigation task.

**Rebalancing frequency:** Compute IC factors daily (after nightly IC computation), but only update synthesis weights weekly to avoid excessive churn. Store the weekly weight snapshot for audit.

### Section 6: Signal Correlation Tracking (7.7)

22 collectors run independently with static weights, but some are highly correlated (e.g., technical RSI and ML direction often > 0.7 correlated). This means effective independent signal count may be 10-12, not 22, inflating conviction.

**New supervisor batch node: `run_signal_correlation()`**

**Schedule:** Weekly (Friday after market close).

**Logic:**
1. Collect the last 63 trading days of per-symbol signal values from each collector (from the signals table or ICAttributionTracker records)
2. Compute pairwise Spearman correlation matrix across all collectors
3. For correlated pairs, apply a **continuous penalty** to the weaker signal (lower IC): `correlation_penalty = max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)`. This gives no penalty below 0.5 correlation, linearly increasing penalty from 0.5 to 0.7+, bottoming at 0.2× weight. Avoids the cliff problem of a hard 0.7 threshold with 63-day sample sizes (standard error ~0.13).
4. Compute effective independent signal count using eigenvalue decomposition: count eigenvalues > 0.1 of the correlation matrix
5. Store the correlation matrix in `signal_correlation_matrix` table (date, collector_a, collector_b, correlation, action_taken)
6. Log a summary: "Effective independent signals: {count} out of {total}. Penalized: {list}"

**Integration with Section 5:** The correlation penalties feed into the same IC adjustment mechanism. The final effective weight becomes: `static_weight * ic_factor * correlation_penalty`. This stacks naturally — a collector with low IC AND high correlation with a stronger signal gets heavily downweighted.

### Section 7: Conflicting Signal Resolution (7.8)

When collectors disagree strongly (technical bullish, ML bearish, sentiment neutral), the weighted average produces a middling conviction that doesn't reflect the actual uncertainty.

**Detection rule:** After computing per-collector vote scores in synthesis, check: `if max(votes) - min(votes) > 0.5: flag as conflicting`.

**Response:** Cap conviction at 0.3 (configurable). This prevents the system from trading on weak, conflicting signals. The cap applies after the multiplicative conviction adjustment (Section 8) but before the final clip.

**EventBus:** Publish `SIGNAL_CONFLICT` event with payload `{symbol, conflicting_collectors, max_signal, min_signal, spread}`. Add `SIGNAL_CONFLICT` to `EventType` enum.

**Logging:** Record conflict events in a structured format for pattern analysis. Over time, certain collector pairs may consistently conflict — this informs whether one should be removed or whether the conflict itself is a useful signal (e.g., "technical/ML disagreement often precedes reversals").

### Section 8: Conviction Calibration — Multiplicative (7.9)

The current conviction logic uses additive adjustments (+0.10, -0.15, etc.), which create non-proportional effects depending on base conviction. Converting to multiplicative ensures each factor scales proportionally.

**Six multiplicative factors (1:1 conversion from existing additive rules):**

1. **ADX strength factor:** `1.0 + 0.15 * min(1.0, (ADX - 15) / 35)`. When ADX = 15 (weak trend), factor = 1.0. When ADX = 50 (strong trend), factor = 1.15. Smooth ramp instead of binary threshold.

2. **Regime stability factor:** `0.85 + 0.20 * hmm_stability`. When stability = 0.0, factor = 0.85 (penalty). When stability = 1.0, factor = 1.05 (slight boost). When stability > 0.8 (previous threshold), factor > 1.01.

3. **Timeframe agreement factor:** `0.80` if weekly trend contradicts daily, `1.0` otherwise. Preserves the current -0.15 effect at base conviction ~0.75 (0.75 * 0.80 = 0.60, vs 0.75 - 0.15 = 0.60).

4. **Regime source agreement factor:** `0.85` if HMM and rule-based disagree, `1.0` otherwise. Preserves the -0.10 effect.

5. **ML confirmation factor:** `1.10` if ML confirms rule-based regime, `1.0` otherwise. Slight boost for cross-validation.

6. **Data quality factor:** `0.75` if technical or regime collector failed, `1.0` otherwise. Significant penalty for missing data.

**Final formula:** `adjusted = base * f1 * f2 * f3 * f4 * f5 * f6`, clipped to [0.05, 0.95].

**Implementation:** Modify the conviction adjustment section of `signal_engine/synthesis.py`. Replace the additive block with multiplicative computation. Log both the individual factors and the final adjusted value for debugging and calibration.

**Calibration:** Store factor inputs and conviction outcomes alongside trade results. Quarterly, compute which factors actually improve conviction accuracy (correlation between factor-adjusted conviction and trade outcome) and tune coefficients.

---

## Stream C: Autonomous Learning (7.4, 7.5, 7.10, 7.11, 7.12)

### Section 9: Agent Decision Quality Tracking (7.5)

LLM agents (trade_debater, exit_evaluator, etc.) make recommendations that lead to trade outcomes, but there's no feedback loop — the same prompt produces the same quality regardless of past performance.

**Per-agent tracking:** Leverage the existing SkillTracker (wired in Section 2, Wire 4). After wiring, every trade close populates `agent_skills` with prediction accuracy and signal P&L. The rolling 30-trade win rate is already computed by SkillTracker's internals.

**Alert threshold:** When any agent's rolling win rate drops below 40% (where random would be ~50%):
1. Publish `AGENT_DEGRADATION` event via EventBus (new event type) with payload `{agent_id, win_rate, trade_count, recent_losses}`
2. Queue a research task of type `agent_prompt_investigation` with the degraded agent's details

**Daily plan integration:** In the daily plan prompt, include per-agent confidence from `SkillTracker.get_confidence_adjustment()`. This lets the planning agent know which execution agents are currently reliable and which are struggling. Format: "Agent confidence: trade_debater=1.2 (reliable), exit_evaluator=0.7 (degraded, under investigation)."

**New EventBus type:** `AGENT_DEGRADATION`.

### Section 10: Live vs. Backtest Sharpe Demotion (7.4)

Strategies can paper trade forever with poor live performance because the current circuit breakers only trigger on absolute drawdown (5%) or consecutive losses (3). A strategy with live Sharpe 0.2 (backtest was 1.5) can trade indefinitely if it avoids 3 losses in a row.

**Live Sharpe computation:** Add a function to the strategy lifecycle that computes rolling 21-day Sharpe from realized daily returns. Source: `strategy_outcomes` table filtered by strategy_id, grouped by date.

**Demotion gate:** If live Sharpe < 50% of backtest Sharpe for 21+ consecutive trading days:
1. Auto-demote strategy status to `forward_testing`
2. Apply 0.25× position sizing multiplier (75% reduction) via StrategyBreaker's `force_scale()` method
3. Publish `STRATEGY_DEMOTED` event via EventBus
4. Queue research task for degradation investigation

**Backtest Sharpe source:** The strategy's backtest Sharpe must be stored at registration time. Add a `backtest_sharpe` column to the `strategies` table if not already present.

**Where to run:** New supervisor batch check, daily after market close. Query each active strategy, compute live Sharpe, compare to stored backtest Sharpe.

### Section 11: Concept Drift Detection (7.10)

The existing drift detector (`learning/drift_detector.py`, 312 lines) only monitors feature distributions via PSI. It misses label drift (target distribution change) and interaction drift (feature-target relationship change) — the two most dangerous forms for trading models.

**Extend the existing `DriftDetector` class with three new detection layers:**

**Layer 1: IC-based concept drift** (daily, after IC computation)
- For each feature used by the ML model, compute rolling Spearman correlation between that feature and realized 5-day forward returns
- Compare to baseline IC (stored from training period)
- Alert if IC drops > 2 standard deviations from baseline over 5 trading days
- This catches changes in feature predictive power that PSI (distribution-only) misses

**Layer 2: Label drift** (weekly)
- Compute KS test on rolling 63-day return distribution vs training-period return distribution
- Alert if p < 0.01 (distribution has shifted significantly)
- This catches regime changes where the target variable itself behaves differently

**Layer 3: Interaction drift** (monthly)
- Train a simple classifier (logistic regression) to distinguish "recent 63-day data" from "training data" using (feature, target) pairs
- If classifier AUC > 0.60, the joint distribution has shifted — flag for investigation
- This catches the insidious case where features and targets look individually stable but their relationship has changed

**Auto-retrain decision tree:**
- Feature drift detected + IC still healthy (> 0.01) → log warning, no action (benign covariate shift)
- IC degradation + gradual (declining over 60+ days) → auto-retrain with recent 252-day data window
- IC degradation + abrupt (step change) → publish `MODEL_DEGRADATION` event, queue manual investigation
- Retraining cooldown: maximum once per 20 trading days to avoid overfitting to noise

**New supervisor batch nodes:**
- `run_drift_detection()`: daily, after IC computation. Runs Layer 1 and existing PSI.
- `run_adversarial_validation()`: monthly (1st trading day). Runs Layer 3.
- Layer 2 (KS test) is lightweight enough to include in the daily run.

### Section 12: Model Versioning + Champion/Challenger (7.11)

No model versioning exists today. Models are trained ad-hoc with no way to compare versions, run shadow evaluations, or safely roll back.

**New DB table: `model_registry`**

```python
@dataclass
class ModelVersion:
    model_id: str           # UUID
    strategy_id: str        # Which strategy this model serves
    version: int            # Auto-incrementing per strategy
    train_date: date
    train_data_range: str   # "2025-01-01 to 2026-03-15"
    features_hash: str      # SHA256 of sorted feature list
    hyperparams: dict       # JSON: n_estimators, learning_rate, etc.
    backtest_sharpe: float
    backtest_ic: float
    backtest_max_dd: float
    model_path: str         # ~/.quantstack/models/{strategy_id}/v{version}/model.pkl
    status: str             # champion | challenger | retired
    promoted_at: datetime | None
    retired_at: datetime | None
    shadow_start: date | None
    shadow_ic: float | None
    shadow_sharpe: float | None
    created_at: datetime
```

**File storage:** `~/.quantstack/models/{strategy_id}/v{version}/model.pkl` (or joblib). Include a `metadata.json` alongside with training config for reproducibility.

**Champion/challenger workflow:**
1. When `ml/trainer.py` produces a new model, register it as `challenger` with shadow_start = today
2. During signal collection, run both champion and challenger inference. Champion drives real signals; challenger predictions are logged to a `model_shadow_predictions` table.
3. After 30 trading days of shadow data, compare:
   - Challenger IC > champion IC by at least 0.005
   - Challenger Sharpe > champion Sharpe by at least 0.15
   - Challenger max drawdown <= 1.1× champion max drawdown
4. If all criteria met: promote challenger to champion, demote old champion to retired. Publish `MODEL_TRAINED` event.
5. If criteria not met after 60 days: retire the challenger.

**Integration with ML collector:** Modify `signal_engine/collectors/ml_signal.py` to load the current champion model from the registry (instead of whatever model file exists on disk). When a challenger exists, also run its inference and log predictions.

### Section 13: Regime Transition Detection (7.12)

The HMM regime model identifies 4 states but doesn't expose transition probabilities. Most losses occur during transitions when the model is most uncertain — the system trades with full conviction during the most dangerous period.

**Expose filtered transition probabilities:** In `signal_engine/collectors/regime.py`, after fitting the HMM model, use the **filtered state probabilities** from `model.predict_proba(X)` — not the static `transmat_` matrix. The static matrix gives the same transition probability regardless of recent data, which defeats the purpose. The filtered probabilities reflect actual uncertainty given the observation sequence. Add to the collector output:
- `transition_probability: float` — `1.0 - max(predict_proba(X)[-1])`, the probability of NOT being in the most likely state at time t
- `state_probabilities: dict` — per-state filtered probabilities at current time
- `most_likely_next_regime: str` — the state with second-highest filtered probability

**Degraded mode:** If HMM fails to fit (insufficient data, convergence failure), `transition_probability` defaults to 0.0 (no transition adjustment). The `risk_sizing` node must handle a missing or None transition probability gracefully — default to 1.0 sizing factor.

**Sizing response (moderate approach):**
- P(transition) < 0.10: no adjustment
- P(transition) 0.10–0.30: multiply sizing by 0.75 (25% reduction)
- P(transition) 0.30–0.50: multiply sizing by 0.50 (50% reduction)
- P(transition) > 0.50: multiply sizing by 0.25 (75% reduction, but don't block)

This is applied in the `risk_sizing` node alongside the StrategyBreaker scale factor: `final_size = kelly_size * breaker_factor * transition_factor`.

**Minimum tradeable size floor:** After all multiplicative adjustments (Kelly × breaker × transition × Sharpe demotion), if the resulting position value is below a minimum tradeable threshold ($100 or 1 share), skip the trade entirely. The compound of 4-5 factors can produce micro-orders (e.g., SCALED 0.5 × transition 0.5 × Sharpe demotion 0.25 = 0.0625× normal size). Micro-orders cost more in commissions than they're worth. Log the skip for visibility.

**Vol-conditioned sub-regimes:** Extend the regime output with a volatility dimension using 20-day realized volatility percentile:
- Low vol: < 30th percentile of trailing 252-day vol distribution
- Normal vol: 30th–70th percentile
- High vol: > 70th percentile

This creates combined regimes like `trending_up_low_vol`, `ranging_high_vol`, etc. The synthesis weight profiles (currently keyed by 4 regimes) expand to 12 regime-vol combinations. Start by extending the existing 4 profiles with vol-aware adjustments rather than defining 12 independent profiles: for each base regime, apply a vol factor (low_vol → boost trend signals, high_vol → boost mean-reversion signals).

**EventBus:** Publish `REGIME_CHANGE` event (already exists) when transition is detected. Include the transition probability and the previous/new regime in the payload.

---

## Cross-Cutting Concerns

### New EventBus Types

Add three new types to `EventType` enum in `coordination/event_bus.py`:

```python
SIGNAL_DEGRADATION = "signal_degradation"
SIGNAL_CONFLICT = "signal_conflict"
AGENT_DEGRADATION = "agent_degradation"
```

### New DB Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `model_registry` | Model versioning (Section 12) | model_id, strategy_id, version, status, metrics |
| `model_shadow_predictions` | Challenger predictions (Section 12) | model_id, symbol, date, prediction, realized_return |
| `signal_correlation_matrix` | Weekly correlations (Section 6) | date, collector_a, collector_b, correlation |
| `loss_aggregation` | Daily failure aggregation (Section 4) | date, failure_mode, strategy_id, cumulative_pnl |
| `strategy_breaker_states` | StrategyBreaker persistence (migrated from JSON) | strategy_id, state, scale_factor, consecutive_losses, drawdown_pct, updated_at |
| `ic_attribution_data` | ICAttributionTracker persistence (migrated from JSON) | collector, symbol, signal_value, forward_return, recorded_at |

### Schema Changes to Existing Tables

| Table | Change |
|-------|--------|
| `strategy_outcomes` | Add `failure_mode TEXT` column |
| `strategies` | Add `backtest_sharpe FLOAT` column (if not exists) |

### Supervisor Batch Schedule

| Time | Node | Frequency |
|------|------|-----------|
| 16:30 ET | `run_loss_aggregation()` | Daily |
| After IC computation | `run_drift_detection()` | Daily |
| Friday close | `run_signal_correlation()` | Weekly |
| 1st trading day | `run_adversarial_validation()` | Monthly |
| After market close | `run_sharpe_demotion_check()` | Daily |

### Risk Gate Interaction

**The risk gate (`execution/risk_gate.py`) is LAW — never bypassed.** All sizing reductions from Phase 7 (StrategyBreaker, transition probability, Sharpe demotion) are applied BEFORE the risk gate. The risk gate remains the final authority. If Phase 7 adjustments reduce sizing to near-zero but the risk gate would have allowed it, that's correct — the learning system is more conservative than the safety system in this case.

### Kill-Switch Config Flags

Each feedback loop must be independently toggleable via environment variables (defaulting to OFF for safe rollout):

| Flag | Controls | Default |
|------|----------|---------|
| `FEEDBACK_IC_WEIGHT_ADJUSTMENT` | Section 5: IC-based weight adjustment | `false` |
| `FEEDBACK_CORRELATION_PENALTY` | Section 6: Signal correlation penalties | `false` |
| `FEEDBACK_CONVICTION_MULTIPLICATIVE` | Section 8: Multiplicative conviction | `false` |
| `FEEDBACK_TRANSITION_SIZING` | Section 13: Regime transition sizing | `false` |
| `FEEDBACK_SHARPE_DEMOTION` | Section 10: Live vs backtest demotion | `false` |
| `FEEDBACK_DRIFT_DETECTION` | Section 11: Concept drift detection | `false` |

When a flag is `false`, the corresponding adjustment defaults to 1.0 (no effect). The wiring (Sections 1-4) and data collection (agent tracking, model versioning) are always active — only the sizing/weight adjustments are toggleable. This allows data to accumulate while safely enabling each feedback loop one at a time.

### Cold-Start / Bootstrap Behavior

Every feedback mechanism must define behavior when insufficient data exists:

| Section | Cold-Start Condition | Default Behavior |
|---------|---------------------|-----------------|
| 1 (OutcomeTracker) | < 5 outcomes per regime | Affinity = 1.0 (no adjustment) |
| 5 (IC weights) | < 21 days of IC data per collector | ic_factor = 1.0 (full static weight) |
| 6 (Correlation) | < 63 days of signal data | No correlation penalty |
| 8 (Conviction) | N/A (uses current data) | Factors default to 1.0 if input missing |
| 9 (Agent quality) | < 30 trades per agent | No alert, confidence = 1.0 |
| 10 (Sharpe demotion) | < 21 trading days of live returns | No demotion check |
| 11 (Drift) | < 63 days of feature data | Skip drift check |
| 12 (Champion/challenger) | No champion model registered | Use existing model loading path |
| 13 (Transition) | HMM not fit / < 120 bars | transition_probability = 0.0 (no sizing adjustment) |

### Rollback Paths

Each section has a one-line rollback:

| Section | Rollback |
|---------|----------|
| 1 (API audit) | Revert file changes (git) |
| 2 (Readpoint wiring) | Revert file changes; ghost modules return to disconnected state |
| 3 (Failure taxonomy) | Set all new losses to `UNCLASSIFIED`; research queue reverts to `bug_fix` |
| 4 (Loss aggregation) | Disable supervisor batch node; aggregation table goes stale |
| 5 (IC weights) | Set `FEEDBACK_IC_WEIGHT_ADJUSTMENT=false` |
| 6 (Correlation) | Set `FEEDBACK_CORRELATION_PENALTY=false` |
| 7 (Conflict resolution) | Remove conflict detection check in synthesis |
| 8 (Conviction) | Set `FEEDBACK_CONVICTION_MULTIPLICATIVE=false`; reverts to additive |
| 9 (Agent quality) | Disable agent quality check; SkillTracker data keeps accumulating |
| 10 (Sharpe demotion) | Set `FEEDBACK_SHARPE_DEMOTION=false` |
| 11 (Drift detection) | Set `FEEDBACK_DRIFT_DETECTION=false` |
| 12 (Model versioning) | ML collector falls back to disk-based model loading |
| 13 (Transition) | Set `FEEDBACK_TRANSITION_SIZING=false` |

### Database Migration Strategy

All schema changes (4 new tables, 2 column additions, 2 persistence migrations) should use the existing pattern in `db.py` — `CREATE TABLE IF NOT EXISTS` with `ON CONFLICT DO NOTHING` for initial data. For the `strategy_outcomes.failure_mode` column addition, use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS failure_mode TEXT DEFAULT 'unclassified'`. Existing rows get the default.

---

## Implementation Order

### Week 1: Stream A (Core Wiring)
1. Section 1: Ghost module API audit (0.5 days)
2. Section 2: Wire 6 readpoints (2 days)
3. Section 3: Failure mode taxonomy (2 days)

### Week 1-2: Stream C Start (Independent)
4. Section 9: Agent decision quality tracking (2 days, parallel with Stream A)
5. Section 12: Model versioning (2 days, parallel)

### Week 2: Stream A Completion + Stream B Start
6. Section 4: Loss aggregation (1 day)
7. Section 7: Conflicting signal resolution (1 day, independent)
8. Section 8: Conviction calibration multiplicative (1.5 days)

### Week 3: Stream B + Stream C
9. Section 5: IC degradation → weight adjustment (2 days)
10. Section 6: Signal correlation tracking (1.5 days)
11. Section 10: Live vs backtest Sharpe demotion (1 day)

### Week 4: Advanced Items
12. Section 11: Concept drift detection (2 days)
13. Section 13: Regime transition detection (3 days)

### Week 5: Integration Testing + Validation
14. Run the full validation plan (see spec) across all 12 items
15. Verify EventBus events are published and consumed correctly
16. Verify supervisor batch nodes run on schedule
17. End-to-end test: simulate a losing trade → verify it flows through taxonomy → aggregation → research queue → investigation task
