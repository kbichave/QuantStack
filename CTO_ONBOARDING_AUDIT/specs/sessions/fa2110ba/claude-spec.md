# Complete Specification — Phase 7: Feedback Loops & Learning

---

## 1. Mission

Wire the 6 ghost learning modules into the live system, implement failure mode taxonomy, connect IC degradation to signal weight adjustment, add live-vs-backtest Sharpe demotion, track agent decision quality, detect signal correlation/conflict, switch conviction to multiplicative, add concept drift detection, build model versioning, and detect regime transitions. After this phase: every loss makes the next decision better.

---

## 2. Current System State (from codebase research)

### 2.1 Ghost Modules

| Module | File | Lines | Status | Write Path | Read Path |
|--------|------|-------|--------|------------|-----------|
| OutcomeTracker | `learning/outcome_tracker.py` | ~active | SINK | `trade_hooks._on_trade_fill()` | `get_regime_strategies()` returns stub |
| SkillTracker | `learning/skill_tracker.py` | 421 | GHOST | Can be called, never is | Only `/skills` API endpoint |
| ICAttributionTracker | `learning/ic_attribution.py` | 420 | GHOST | JSON persistence | Zero consumers |
| ExpectancyEngine | `learning/expectancy_engine.py` | 98 | ORPHAN | N/A | Bypassed by `core/kelly_sizing.py` |
| StrategyBreaker | `execution/strategy_breaker.py` | 553 | 1 consumer (context) | `record_trade()` works | `get_scale_factor()` never called |
| TradeEvaluator | `performance/trade_evaluator.py` | 59 | SINK | Writes to `trade_quality_scores` | Nobody reads |

### 2.2 IC Tracking State

- `signal_ic` table: **populated nightly** by `run_ic_computation()` in supervisor. Computes cross-sectional rank IC + ICIR for 5/10/21-day horizons. Consumers: `auto_promoter.py`, `ic_retirement.py`, `trading/nodes.py`.
- `ICAttributionTracker`: dead code — zero callers, no pipeline feeds signal/return pairs.
- `SkillTracker.record_ic()`: dead code — zero callers, `agent_ic_observations` table unpopulated.
- **Upstream dependency:** `signals` table must have data (≥5 symbols, ≥21 days) for IC computation.

### 2.3 Signal Engine

- 14+ collectors (technical, regime, sentiment, ML, flow, volume, fundamentals, options, earnings)
- **Synthesis:** Static regime-conditional weight profiles in `synthesis.py`
- **Conviction:** ADDITIVE adjustments (ADX > 25 → +0.10, stability → +0.05, etc.), clipped [0.05, 0.95]
- **Missing:** No correlation detection, no conflict detection, no dynamic IC-based weighting

### 2.4 Trade Hooks

- Loss > 1% → `research_queue` with generic `bug_fix` task type, priority 5 or 7
- No failure mode taxonomy — all losses treated identically
- Hooks: `on_trade_close()`, `on_daily_close()`, `on_trade_fill()`

### 2.5 EventBus

- Full PostgreSQL-backed EventBus at `coordination/event_bus.py`
- Poll-based, per-consumer cursors, 7-day TTL
- Existing types: `IC_DECAY`, `DEGRADATION_DETECTED`, `MODEL_DEGRADATION`, `REGIME_CHANGE`, `STRATEGY_PROMOTED/RETIRED/DEMOTED`, etc.
- **Decision:** Use as-is, add new event types for Phase 7

### 2.6 Regime Detection

- HMM model with 4 states: trending_up/trending_down/ranging/unknown
- Outputs: hmm_probabilities, hmm_stability, hmm_expected_duration
- **Transition probabilities NOT exposed** — model fit but transition matrix not surfaced
- Rule-based fallback for insufficient data (< 120 bars)

### 2.7 Trading Graph Nodes

- `risk_sizing` (lines 452-591): queries `signal_ic`, computes EWMA vol, calls `regime_kelly_fraction()`. **Missing:** StrategyBreaker scale factor.
- `execute_entries` (lines 687-730): places orders. **Missing:** StrategyBreaker check.
- `daily_plan` (lines 229-284): LLM trading plan. **Missing:** trade quality pattern surfacing.

### 2.8 ML Pipeline

- Training: LightGBM/XGBoost/CatBoost, 5-fold CV
- Drift: PSI-only on 6 features. No label drift, no interaction drift, no IC-based drift.
- **No model versioning system.**

### 2.9 Testing

- pytest, 186 test files under `tests/unit/`
- conftest.py: mock_settings, run_async, OHLCV generators
- Class-based tests, async via `run_async` fixture

---

## 3. Implementation Decisions (from interview)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sequencing | Parallel streams | Group independent items, maximize throughput |
| Ghost module APIs | Audit and fix before wiring | Review math/thresholds, fix obvious issues during integration |
| Weight model | Static as priors, IC adjusts | Regime profiles as baseline × IC-derived factors. Prevents zero-weight on temporary dips. |
| Regime transition response | Moderate: 50% sizing reduction | Allow entries at reduced size, don't block. Autonomous system needs to keep trading. |
| Failure classifier | Hybrid: rules + LLM | Rule-based for obvious (regime mismatch), LLM for ambiguous cases |
| Model registry | Custom lightweight | DB table + file storage. No MLflow dependency. |
| Scheduling | Add to supervisor batch | Single orchestration point for all scheduled analysis |
| Agent alerts | EventBus alert + auto-research task | Immediate alert, then queued prompt investigation |
| Conviction calibration | Convert all 6 to multiplicative | 1:1 mapping, each additive rule → multiplicative factor |
| EventBus extension | Add event types only | Existing pattern is solid, just extend |

---

## 4. Items (Enriched)

### 4.1 Wire 6 Ghost Module Readpoints (7.1)

**Effort:** 2-3 days | **Severity:** CRITICAL | **Dependencies:** None

**Pre-work: Audit each module's API before wiring:**
- OutcomeTracker: affinity formula `tanh(pnl/5.0) * 0.05` is too slow — consider larger step size or exponential decay
- SkillTracker: `get_confidence_adjustment()` formula is reasonable (0.5-1.5 range), verify edge cases
- ICAttributionTracker: Spearman IC computation needs scipy — verify it's in dependencies
- ExpectancyEngine: evaluate whether to wire this or leave `core/kelly_sizing.py` as primary
- StrategyBreaker: state machine thresholds (5% DD, 3 consecutive losses) seem reasonable, verify persistence
- TradeEvaluator: 6-dimension scoring is good, need to define read patterns

**6 wiring changes:**

| # | Wire | Integration Point | Change |
|---|------|-------------------|--------|
| 1 | `get_regime_strategies()` | `meta_tools.py` → DB | Replace stub with query returning affinity-weighted allocations from `strategies.regime_affinity` |
| 2 | `risk_sizing` → StrategyBreaker | `trading/nodes.py:452-591` | After Kelly fraction, multiply by `strategy_breaker.get_scale_factor(strategy_id)` |
| 3 | `execute_entries` → StrategyBreaker | `trading/nodes.py:687-730` | Check breaker status before placing orders; skip TRIPPED strategies |
| 4 | Trade hooks → SkillTracker | `trade_hooks.py::on_trade_close()` | Call `skill_tracker.update_agent_skill(agent_name, outcome)` on every close |
| 5 | Signal engine → ICAttribution | `signal_engine/engine.py` after synthesis | Call `ic_attribution.record(symbol, collector, signal_value, forward_return)` |
| 6 | `daily_plan` → trade quality scores | `trading/nodes.py:229-284` | Query `trade_quality_scores` and surface top patterns to daily planning prompt |

### 4.2 Failure Mode Taxonomy (7.2)

**Effort:** 2 days | **Severity:** CRITICAL | **Dependencies:** 7.1

**Enum:**
```
REGIME_MISMATCH   — entry regime != exit regime
FACTOR_CROWDING   — high factor overlap at entry (correlated signals)
DATA_STALE        — key data source stale at entry time
TIMING_ERROR      — entry within 2 bars of stop/resistance
THESIS_WRONG      — thesis held but market moved against for fundamental reasons
BLACK_SWAN        — loss > 3 std deviations from expected
```

**Classification approach:** Hybrid rules + LLM
- Rule-based: regime comparison, data freshness check, loss magnitude
- LLM: for ambiguous cases, analyze trade context and classify
- Store classification in `strategy_outcomes` table (new column: `failure_mode`)

**Research queue enhancement:** Replace generic `bug_fix` with `failure_mode`-specific task types. Priority = `f(cumulative_loss_30d * recency_weight)`.

### 4.3 IC Degradation → Weight Adjustment (7.3)

**Effort:** 2 days | **Severity:** CRITICAL | **Dependencies:** IC tracking (partially available via `signal_ic`)

**Approach:** Static weights as priors × IC-derived adjustment factors
- Daily: read per-collector IC from ICAttributionTracker (once wired in 7.1 wire #5)
- Rolling 21-day IC < 0.02: halve collector weight (multiply static weight by 0.5)
- Rolling 21-day IC < 0.0: zero out collector weight
- IC_IR (mean/std) < 0.1: additional penalty factor of 0.7
- Publish `SIGNAL_DEGRADATION` event via EventBus
- Research graph picks up investigation task

**New EventBus type:** `SIGNAL_DEGRADATION`

### 4.4 Live vs. Backtest Sharpe Demotion (7.4)

**Effort:** 1 day | **Severity:** HIGH | **Dependencies:** Phase 2 item 2.6 (walk-forward)

- Add live Sharpe computation to strategy lifecycle (rolling 21-day from realized returns)
- Compare to backtest Sharpe stored in strategy metadata
- Gate: live Sharpe < 50% of backtest for 21+ days → auto-demote to `forward_testing`
- Demoted strategies: position sizes × 0.25 (75% reduction)
- Publish `STRATEGY_DEMOTED` event via EventBus
- Queue research task for strategy degradation investigation

### 4.5 Agent Decision Quality Tracking (7.5)

**Effort:** 3 days | **Severity:** HIGH | **Dependencies:** None (independent stream)

- Track per-agent recommendation → outcome (existing SkillTracker has the API)
- Compute per-agent win rate over rolling 30 trades
- Alert when win rate < 40%: publish `AGENT_DEGRADATION` EventBus event + queue research task
- Surface agent quality in daily plan prompt context
- Wire `SkillTracker.update_agent_skill()` in trade hooks (overlaps with 7.1 wire #4)

**New EventBus type:** `AGENT_DEGRADATION`

### 4.6 Loss Aggregation in Supervisor (7.6)

**Effort:** 1 day | **Severity:** HIGH | **Dependencies:** 7.2

- New supervisor batch node: `run_loss_aggregation()`, scheduled 16:30 ET daily
- Aggregate losses by failure mode, strategy, symbol over trailing 30 days
- Identify top 3 failure patterns by cumulative P&L impact
- Auto-generate targeted research tasks (not generic `bug_fix`)
- Store daily aggregation results for trend analysis

### 4.7 Signal Correlation Tracking (7.7)

**Effort:** 2 days | **Severity:** CRITICAL | **Dependencies:** ICAttributionTracker wired (7.1)

- Weekly supervisor batch node: `run_signal_correlation()`
- Compute pairwise Spearman correlation matrix across all collectors
- If `corr(A, B) > 0.7`: halve weight of the collector with lower IC
- Compute effective independent signal count = eigenvalues > 0.1
- Store correlation matrix in DB for trend analysis
- Log correlation report

### 4.8 Conflicting Signal Resolution (7.8)

**Effort:** 1 day | **Severity:** HIGH | **Dependencies:** None

- In signal synthesis: detect when `max_signal - min_signal > 0.5`
- When conflicting: cap conviction at 0.3 (or configurable threshold)
- Publish `SIGNAL_CONFLICT` event via EventBus
- Log conflict details (which collectors, what signals)

**New EventBus type:** `SIGNAL_CONFLICT`

### 4.9 Conviction Calibration — Multiplicative (7.9)

**Effort:** 2 days | **Severity:** HIGH | **Dependencies:** IC tracking for calibration data

**Convert 6 additive rules to multiplicative factors:**

| Current Additive | Multiplicative Factor |
|------------------|----------------------|
| ADX > 25: +0.10 | `adx_factor = 1.0 + 0.15 * min(1.0, (ADX - 15) / 35)` |
| HMM stability > 0.8: +0.05 | `stability_factor = 0.85 + 0.20 * hmm_stability` |
| Weekly contradicts daily: -0.15 | `timeframe_factor = 0.80 if contradicting else 1.0` |
| HMM/rule disagree: -0.10 | `regime_agreement_factor = 0.85 if disagree else 1.0` |
| ML confirms rule-based: +0.05 | `ml_confirmation_factor = 1.10 if confirms else 1.0` |
| Technical/regime fail: -0.20 | `data_quality_factor = 0.75 if failure else 1.0` |

`adjusted_conviction = base * adx_factor * stability_factor * timeframe_factor * regime_agreement_factor * ml_confirmation_factor * data_quality_factor`

Clip to [0.05, 0.95]. Calibrate factor coefficients quarterly from realized performance.

### 4.10 Concept Drift Detection (7.10)

**Effort:** 2 days | **Severity:** HIGH | **Dependencies:** IC tracking

**Extend existing PSI-only drift detector (`learning/drift_detector.py`) with 3 new layers:**

1. **IC-based drift:** Rolling Spearman IC per feature vs target. Alert if IC drops > 2 std deviations from baseline over 5 trading days.
2. **Label drift:** KS test on rolling return distribution vs training-period distribution. Alert if p < 0.01.
3. **Interaction drift:** Monthly adversarial validation — classifier distinguishing recent vs training data on (feature, target) pairs. Alert if AUC > 0.60.

**Auto-retrain decision tree:**
- Gradual IC degradation (60+ days) → auto-retrain with recent data window
- Abrupt drift → manual investigation (publish `MODEL_DEGRADATION` event, queue research)
- Retraining cooldown: max once per 20 trading days

### 4.11 Model Versioning + A/B (7.11)

**Effort:** 2 days | **Severity:** HIGH | **Dependencies:** None (independent stream)

**Custom lightweight registry:**
- New DB table: `model_registry` (model_id, version, strategy_id, train_date, features_hash, hyperparams_json, backtest_sharpe, backtest_ic, model_path, status: champion/challenger/retired, promoted_at, retired_at)
- File storage: `~/.quantstack/models/{strategy_id}/v{version}/model.pkl`
- Champion/challenger framework:
  - New model → status=challenger, runs in shadow mode (predictions logged, not executed)
  - Shadow for 21+ days (spec) or 60+ days (best practice) — use 30 days as compromise
  - Promote if IC > champion IC by 0.005 AND no drawdown regression > 10%
  - Publish `MODEL_TRAINED` event on new version, `STRATEGY_PROMOTED` on promotion

### 4.12 Regime Transition Detection (7.12)

**Effort:** 3 days | **Severity:** HIGH | **Dependencies:** None

**Changes to regime collector (`signal_engine/collectors/regime.py`):**

1. **Expose transition probability matrix** from fitted HMM model
2. **Compute P(transition)** = 1 - P(staying in current state)
3. **Transition response (moderate):**
   - P(transition) 0.10-0.30: reduce sizing by 25%
   - P(transition) 0.30-0.50: reduce sizing by 50%
   - P(transition) > 0.50: reduce sizing by 75% (don't block)
4. **Vol-conditioned sub-regimes:** Add volatility dimension to regime output
   - trending_up_low_vol, trending_up_high_vol, etc.
   - Use 20-day realized vol percentile: low (<30th), normal (30-70th), high (>70th)
5. **Publish `REGIME_CHANGE` event** (already exists in EventBus) on detected transition

---

## 5. Parallel Implementation Streams

Based on dependency analysis, 3 parallel streams:

### Stream A: Core Wiring (7.1 → 7.2 → 7.6)
- Wire ghost modules, add failure taxonomy, implement loss aggregation
- Strict sequential — each depends on prior

### Stream B: Signal Intelligence (7.3, 7.7, 7.8, 7.9)
- IC degradation, signal correlation, conflict resolution, conviction calibration
- 7.3 and 7.8 can run in parallel; 7.7 depends on 7.1 wire #5; 7.9 independent

### Stream C: Autonomous Learning (7.5, 7.10, 7.11, 7.12)
- Agent quality, drift detection, model versioning, regime transitions
- All independent of each other; 7.5 partially overlaps with Stream A (wire #4)

### Stream Dependencies on Phase 2
- 7.4 depends on Phase 2 item 2.6 (walk-forward). Can proceed once available.
- 7.3, 7.7, 7.9, 7.10 partially depend on IC data flowing — mitigated by wiring ICAttributionTracker in 7.1.

---

## 6. New EventBus Types Required

| Event Type | Publisher | Consumer |
|-----------|-----------|----------|
| `SIGNAL_DEGRADATION` | IC degradation monitor (7.3) | Research graph |
| `SIGNAL_CONFLICT` | Signal synthesis (7.8) | Dashboard, research |
| `AGENT_DEGRADATION` | Agent quality tracker (7.5) | Research graph, dashboard |

Existing types already cover: `STRATEGY_DEMOTED`, `MODEL_TRAINED`, `REGIME_CHANGE`, `IC_DECAY`, `MODEL_DEGRADATION`.

---

## 7. New Supervisor Batch Nodes

| Node | Schedule | Purpose |
|------|----------|---------|
| `run_loss_aggregation()` | Daily 16:30 ET | Aggregate losses by failure mode (7.6) |
| `run_signal_correlation()` | Weekly (Friday close) | Pairwise collector correlation matrix (7.7) |
| `run_drift_detection()` | Daily (after IC computation) | IC-based + label drift checks (7.10) |
| `run_adversarial_validation()` | Monthly (1st trading day) | Interaction drift detection (7.10) |

---

## 8. New DB Tables

| Table | Purpose |
|-------|---------|
| `model_registry` | Model versioning metadata (7.11) |
| `signal_correlation_matrix` | Weekly pairwise correlation storage (7.7) |
| `loss_aggregation` | Daily failure mode aggregation (7.6) |

---

## 9. Validation Plan

1. **Readpoints (7.1):** Query `regime_affinity` → verify non-stub data. Execute trade with TRIPPED strategy → verify blocked/scaled.
2. **Taxonomy (7.2):** Close trade at loss in wrong regime → verify classified as `REGIME_MISMATCH`. Test LLM classification for ambiguous case.
3. **IC degradation (7.3):** Inject declining IC series for one collector → verify weight halved after 21 days. Verify `SIGNAL_DEGRADATION` event published.
4. **Sharpe demotion (7.4):** Run strategy with live Sharpe 0.3 vs backtest 1.5 for 21 days → verify demotion + 75% size reduction.
5. **Agent quality (7.5):** Simulate 30 trades → verify win rate computed per agent. Set agent to <40% → verify alert fires + research task queued.
6. **Loss aggregation (7.6):** Run supervisor batch with test loss data → verify top 3 patterns identified.
7. **Signal correlation (7.7):** Inject correlated collectors → verify weaker one's weight halved. Verify effective signal count reported.
8. **Signal conflict (7.8):** Inject opposing signals (spread > 0.5) → verify conviction capped at 0.3. Verify `SIGNAL_CONFLICT` event.
9. **Conviction (7.9):** Compare old additive vs new multiplicative on same inputs → verify proportional behavior.
10. **Drift (7.10):** Inject feature-target correlation shift → verify detection within 5 days. Inject label shift → verify KS alert.
11. **Model versioning (7.11):** Train model → verify version stored. Run shadow → verify promotion after criteria met.
12. **Regime transition (7.12):** Inject high transition probability → verify sizing reduction. Verify `REGIME_CHANGE` event.
