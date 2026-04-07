# Research Findings — Phase 7: Feedback Loops & Learning

---

## Part A: Codebase Analysis

### 1. Ghost Modules — Current State

#### 1.1 OutcomeTracker (`src/quantstack/learning/outcome_tracker.py`)
**Status:** SINK — writes active, reads stubbed

- **Write path:** `hooks/trade_hooks.py::_on_trade_fill()` calls `record_entry()` / `record_exit()`
- **Learning logic:** `apply_learning()` computes `outcome_weight = tanh(pnl_pct / 5.0)`, updates `regime_affinity` as `clip(current + 0.05 * outcome_weight, 0.1, 1.0)`
- **Stub:** `get_regime_strategies()` in `tools/langchain/meta_tools.py` returns `{"error": "Tool pending implementation"}`
- **Gap:** No consumer reads `regime_affinity` to route strategy allocation

#### 1.2 SkillTracker (`src/quantstack/learning/skill_tracker.py` — 421 lines)
**Status:** GHOST — read-only API exposure (only `/skills` endpoint)

- **API:** `update_agent_skill()`, `record_ic()`, `get_confidence_adjustment()` (returns 0.5–1.5), `needs_retraining()` (IC < 0.01 or win_rate < 52%), `ic_summary()`
- **Write paths:** Can be called from reflection hooks; IC stored in `agent_ic_observations` table
- **Gaps:** No agent selector reads confidence adjustments, no retraining pipeline acts on `needs_retraining()`, no learning loop throttles low-ICIR agents
- **Confidence adjustment formula:**
  ```
  adjustment = 1.0
  += max(-0.2, min(0.3, prediction_accuracy - 0.5))  # if count >= 5
  += max(-0.2, min(0.2, signal_win_rate - 0.5))       # if count >= 2
  += max(-0.2, min(0.3, icir * 0.2))                  # if IC obs >= 10
  -= 0.15 if trend == "DECAYING"
  clamp(0.5, 1.5)
  ```

#### 1.3 ICAttributionTracker (`src/quantstack/learning/ic_attribution.py` — 420 lines)
**Status:** GHOST — zero consumers

- **API:** `record(symbol, collector, signal_value, forward_return, timestamp)`, `get_collector_ic()` (Spearman), `get_report()` → ICAttributionReport, `get_weights()` → normalized by IC > 0
- **Persistence:** JSON at `~/.quantstack/ic_attribution.json`
- **Status classification:** strong (IC > 0.05), weak (0 < IC <= 0.05), degraded (IC <= 0), insufficient
- **Gaps:** No signal engine reads suggested weights, no recording pipeline, no degradation alerts

#### 1.4 ExpectancyEngine (`src/quantstack/learning/expectancy_engine.py` — 98 lines)
**Status:** ORPHAN — bypassed by `core/kelly_sizing.py`

- **API:** `calculate_expectancy()`, `get_kelly_fraction()`, `get_trade_quality_score()`
- **Gap:** Graph nodes use `core/kelly_sizing.py::regime_kelly_fraction()` instead

#### 1.5 StrategyBreaker (`src/quantstack/execution/strategy_breaker.py` — 553 lines)
**Status:** 1 consumer (research context only)

- **State machine:** ACTIVE (1.0) → SCALED (0.5) → TRIPPED (0.0, 24h cooldown)
- **Thresholds:** Trip at 5% drawdown or 3 consecutive losses; scale at 3% or 2 losses
- **API:** `record_trade()`, `get_scale_factor()`, `reset()`, `force_trip()`, `force_scale()`, `get_all_states()`
- **Persistence:** `~/.quantstack/strategy_breakers.json`
- **Gaps:** No sizing node reads `get_scale_factor()`, no execution node checks breaker status

#### 1.6 TradeEvaluator (`src/quantstack/performance/trade_evaluator.py` — 59 lines)
**Status:** SINK — writes to `trade_quality_scores`, nobody reads

- **API:** `create_trade_evaluator()` → LLM-as-judge (openevals), scores 6 dimensions
- **Write path:** `graphs/trading/nodes.py::execute_entries()` calls evaluator post-exit
- **Gap:** No reflection pipeline reads scores

---

### 2. Signal Engine (`src/quantstack/signal_engine/`)

#### 2.1 Collector Architecture
14+ collectors returning `dict[str, Any]`: technical (RSI, MACD, Bollinger, ADX, trend), regime (HMM + rule-based), sentiment, ML (LightGBM/XGBoost), flow, volume, fundamentals, options, earnings.

#### 2.2 Synthesis Weights (Regime-Conditional)
Located in `src/quantstack/signal_engine/synthesis.py`. Static profiles per regime:
```python
"trending_up": {"trend": 0.35, "macd": 0.20, "ml": 0.15, "sentiment": 0.10, "rsi": 0.10, "flow": 0.05, "bb": 0.05}
"ranging": {"rsi": 0.25, "bb": 0.25, "ml": 0.15, "macd": 0.10, "sentiment": 0.10, "flow": 0.10, "trend": 0.05}
```

#### 2.3 Conviction Adjustment (ADDITIVE — this is the problem)
```
base = abs(weighted_score)
if ADX > 25: conviction += 0.10
if HMM_stability > 0.8: conviction += 0.05
if weekly_trend contradicts daily: conviction -= 0.15
if HMM/rule-based disagree: conviction -= 0.10
if ML confirms rule-based: conviction += 0.05
if technical or regime fails: conviction -= 0.20
clipped to [0.05, 0.95]
```
**No correlation/conflict detection between collectors. No dynamic ML weight adjustment based on IC.**

---

### 3. Trade Hooks (`src/quantstack/hooks/trade_hooks.py`)

#### Loss Flow (Lines 118–144)
```python
if realized_pnl_pct < -1.0:
    INSERT INTO research_queue (task_type='bug_fix', priority=7 if pnl < -3% else 5, context_json={...})
```
**All losses are generic "bug_fix." No failure mode taxonomy.**

#### Hooks Registered
- `on_trade_close()`: ReflectionManager, research queue (losses > 1%), PromptTuner, ReflexionMemory (losses), CreditAssigner
- `on_daily_close()`: ReflectionManager.daily_reflection()
- `on_trade_fill()`: OutcomeTracker.record_entry() / .record_exit()

---

### 4. Strategy Lifecycle
No explicit promotion/demotion pipeline found. Approximation via:
- `OutcomeTracker.apply_learning()` adjusts `regime_affinity`
- `StrategyBreaker` trips on drawdown/consecutive losses
- `get_regime_strategies()` tool is stubbed

---

### 5. Regime Detection (`src/quantstack/signal_engine/collectors/regime.py`)

- **HMM States:** trending_up / trending_down / ranging / unknown
- **Rule-based fallback:** WeeklyRegimeClassifier using EMA alignment + momentum
- **HMM output includes:** hmm_probabilities, hmm_stability, hmm_expected_duration
- **Transition probabilities NOT exposed** — model is fit but transition matrices not surfaced
- **Minimum bars:** 120 for HMM, 60 for rule-based

---

### 6. Trading Graph Nodes (`src/quantstack/graphs/trading/nodes.py` — 1,185 lines)

#### risk_sizing (Lines 452–591)
1. Query `signal_ic` table for IC values per strategy (21-day horizon)
2. Compute EWMA volatility from 63-day closes
3. Fetch regime state + confidence
4. Call `regime_kelly_fraction(regime, vol_state, ic)`
5. Output `alpha_signals` with conviction
**Missing:** StrategyBreaker.get_scale_factor() not applied

#### execute_entries (Lines 687–730)
Loop over orders, call broker API. **Missing:** No StrategyBreaker check.

#### daily_plan (Lines 229–284)
LLM-generated trading plan from regime + portfolio context. **Missing:** No trade quality pattern surfacing.

---

### 7. ML Pipeline

#### Training (`src/quantstack/ml/trainer.py` — 476 lines)
LightGBM/XGBoost/CatBoost, 500 estimators, 5-fold CV, balanced class weights.

#### Drift Detection (`src/quantstack/learning/drift_detector.py` — 312 lines)
PSI-only on features: rsi_14, atr_pct, adx_14, bb_pct, volume_ratio, regime_confidence.
Thresholds: <0.10 NONE, 0.10–0.25 WARNING, >=0.25 CRITICAL.
Baselines in `~/.quantstack/drift_baselines/`. Pure numpy, ~1ms.
**No label drift, no interaction drift, no IC-based drift.**

#### Model Registry/Versioning
**Not found.** Models trained ad-hoc, no versioning system.

---

### 8. Testing Setup

#### Framework
`tests/unit/conftest.py` (177 lines): mock_settings, patch_get_settings, run_async fixture, OHLCV generators (uptrend/downtrend/V-shape/W-shape/flat/impulse).

#### Organization
186 test files under `tests/unit/`, including subdirs for signal_engine, ml, execution, graphs.
Class-based tests with `pytest`. Async tests via `run_async` fixture.

---

### 9. Database Schema (Relevant Tables)

| Table | Key Columns |
|-------|-------------|
| `strategy_outcomes` | strategy_id, symbol, regime_at_entry, entry/exit_price, realized_pnl_pct |
| `strategies` | strategy_id, name, regime_affinity (JSONB), status |
| `agent_skills` | agent_id, prediction_count, correct_predictions, signal metrics |
| `agent_ic_observations` | agent_id, ic_value, recorded_at |
| `trade_quality_scores` | trade_id, 6 quality dimensions, cycle_number |
| `signal_ic` | strategy_id, mean_rank_ic_21d, horizon_days, date |
| `research_queue` | task_type, priority, context_json, source |

---

## Part B: Best Practices Research (2024–2026)

### 1. IC Degradation Monitoring & Signal Weight Adjustment

#### Degradation Detection Thresholds

| Metric | Alert | Critical |
|--------|-------|----------|
| Rolling IC mean | < 0.02 (from >0.05) | < 0.0 or negative |
| IC t-statistic | < 1.5 | < 1.0 |
| IC hit rate | < 52% | < 50% |
| IC_IR (mean/std) | < 0.3 | < 0.1 |
| Cumulative IC slope | Negative over 6mo | Negative over 3mo |

**IC_IR matters more than raw IC** — it captures consistency. IC=0.03 with IC_IR=0.5 beats IC=0.06 with IC_IR=0.15.

#### Dynamic Weight Approaches
1. **IC-weighted:** `weight_i(t) = rolling_IC_i(t) / sum(IC_j(t))` for IC > threshold
2. **IC_IR-weighted (recommended):** penalizes unstable signals
3. **Bayesian shrinkage:** shrink toward cross-sectional average IC
4. **Exponentially-weighted IC:** halflife ~63 days for regime responsiveness

#### Synthesis Best Practices
- Orthogonalize signals before combining (PCA or residualization)
- Weight by IC_IR, not raw IC
- Apply decay penalties for declining IC slope
- Regime-condition weights
- Rebalance weights monthly (not daily — turnover costs)
- Minimum IC threshold for inclusion: IC > 0.01 over trailing 252 days

**Sources:** Wikipedia IC article; Grinold & Kahn, "Active Portfolio Management"

---

### 2. HMM Regime Transition Detection

#### Position Sizing During Transitions

| P(adverse regime) | Action |
|-------------------|--------|
| < 0.10 | Normal sizing |
| 0.10–0.25 | Reduce 25-50% |
| 0.25–0.50 | Minimum size or paper-only |
| > 0.50 | Close all directional positions |

#### Vol-Conditioned Sub-Regimes
Recommended: 3–4 states via BIC selection. Features: log returns + 20-day realized vol.
- 2 states: low-vol / high-vol
- 3 states: trending-up / ranging / trending-down
- 4 states: trend-up-low-vol / trend-up-high-vol / trend-down-low-vol / trend-down-high-vol

#### Alternatives to HMM
- **Markov-Switching Regression (statsmodels):** supports exogenous variables, switching coefficients. More flexible than HMM.
- **Change-point detection (BOCPD):** online, no pre-specified states, but doesn't identify regime type
- **Clustering (GMM):** flexible features, no temporal dynamics

**Recommendation:** 3-state Markov-Switching model with switching_variance=True, retrained monthly on 2-year rolling window.

**Source:** QuantStart HMM article; statsmodels MarkovRegression docs

---

### 3. Champion/Challenger Model Deployment

#### Versioning
MLflow registry pattern: auto-incrementing versions, `@champion`/`@challenger` aliases, metadata tags (sharpe, IC, features hash, training date range).

#### Shadow Mode
Deploy challenger alongside champion — same inputs, champion drives real trades, challenger predictions logged. Compare after statistically meaningful sample (60+ trading days).

#### Promotion Criteria

| Metric | Requirement |
|--------|------------|
| IC improvement | > 0.005 over champion |
| Sharpe improvement | > 0.15 over champion |
| Max drawdown | <= 1.1x champion |
| Shadow period | >= 60 trading days |
| Statistical significance | Paired t-test p < 0.10 |
| Regime coverage | >= 2 different regime states |

#### Safe Rollout
Backtest → Paper (60d) → Canary (10% capital, 20d) → Gradual ramp (25→50→75→100% over 4 weeks) → Full deployment. Automated rollback on 2x worst historical drawdown.

**Sources:** MLflow docs; Martin Fowler CD4ML; Google Cloud MLOps

---

### 4. Concept Drift Detection Beyond PSI

#### Drift Taxonomy (Lu et al. 2020)
- **Virtual drift (covariate shift):** p(X) changes, p(y|X) unchanged
- **Real drift:** p(y|X) changes, p(X) unchanged
- **Total drift:** both change

#### Method Comparison

| Method | Detects | Best For |
|--------|---------|----------|
| PSI | Feature distribution shift | Covariate drift in tabular features |
| KL Divergence | Distribution divergence | Comparing distributions with same support |
| Wasserstein Distance | Distribution shift with geometry | When magnitude matters |
| KS Test | Max CDF difference | Univariate with small samples |
| ADWIN | Mean shift in streams | Streaming abrupt/gradual mean drift |
| DDM | Error rate increase | Supervised with fast labels |
| Adversarial Validation | Joint (X,Y) shift | Complex multivariate interaction drift |

#### Recommended 5-Layer Detection for QuantStack

| Layer | Method | Frequency | Action |
|-------|--------|-----------|--------|
| L1 | PSI per feature | Daily | Warn if > 0.1, alert if > 0.25 |
| L2 | KS test on model output | Daily | Alert if p < 0.01 |
| L3 | Rolling Spearman IC per signal | Weekly | Reduce weight if IC < 0.01, remove if IC < 0 for 4 weeks |
| L4 | Adversarial validation on (features, target) | Monthly | Manual investigation if AUC > 0.60 |
| L5 | Wasserstein on return distribution | Weekly | Regime model update if > 2 sigma |

#### Auto-Retrain Decision Tree
- Feature drift + healthy IC → log warning, no action (benign covariate shift)
- Feature drift + IC degradation + gradual → auto-retrain with recent data
- Feature drift + IC degradation + abrupt → **manual investigation** (structural break)
- No feature drift + IC degradation → interaction drift likely → adversarial validation
- Retraining cooldown: max once per 20 trading days

**Sources:** Lu et al. arXiv:2004.05785; Pan et al. (Uber) arXiv:2004.03045
