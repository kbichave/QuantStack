# P00 Research: Wire Learning Modules

**Date:** 2026-04-07
**Source:** Codebase exploration (43 file reads across learning/, execution/, hooks/, signal_engine/, graphs/)

---

## Critical Finding: Phase Spec is Outdated

The P00 phase spec describes 6 wires as "ghost modules with zero callers." Codebase analysis reveals **4 of 6 wires are already operational**. The CTO audit implementation (169 findings) wired most of these. The remaining work is activating flag-gated feedback loops and closing 2 integration gaps.

## Wire Status Matrix

| Wire | Producer | Consumer | Data Collection | Feedback Active | Remaining Work |
|------|----------|----------|-----------------|-----------------|----------------|
| 1 | OutcomeTracker | daily_plan (LLM context) | ACTIVE | YES | None — fully wired |
| 2 | OutcomeTracker | risk_sizing (Kelly scaling) | ACTIVE | FLAG-GATED | Complete multiplication logic in `compute_alpha_signals()` |
| 3a | StrategyBreaker | execute_entries | ACTIVE | YES | None — fully wired |
| 3b | StrategyBreaker | on_trade_close hook | ACTIVE | YES | None — fully wired |
| 4 | SkillTracker | conviction scaling | ACTIVE | FLAG-GATED | Wire consumption into signal synthesis |
| 5a | ICAttribution | on_trade_close hook | ACTIVE | YES | None — data collection wired |
| 5b | ICAttribution | synthesis.py weights | ACTIVE | FLAG-GATED | Integrate `get_weights_for_regime()` into `_get_weights()` |
| 6 | TradeEvaluator | daily_plan (LLM context) | PARTIAL | YES (consumption) | Wire `create_trade_evaluator()` call in `on_trade_close` |

## What's Already Working

### Wire 1: OutcomeTracker → Strategy Selection (COMPLETE)
- `daily_plan` node (nodes.py:287-314) queries `strategies.regime_affinity`, ranks by score, injects into LLM prompt
- `get_regime_strategies()` tool (meta_tools.py:29-64) filters strategies by regime affinity with breaker status
- `OutcomeTracker.apply_learning()` uses Bayesian momentum (step=0.15, halflife=20 trades, min 5 outcomes)

### Wire 3: StrategyBreaker → Order Execution (COMPLETE)
- `execute_entries` node (nodes.py:1163-1195) checks `get_scale_factor()` — TRIPPED=reject, SCALED=0.5x size
- `on_trade_close` hook (trade_hooks.py:226-236) calls `breaker.record_trade()` on every close
- States: ACTIVE(1.0) → SCALED(0.5) at 3% DD or 2 consecutive losses → TRIPPED(0.0) at 5% DD or 3 losses

### Wire 6: TradeEvaluator → Agent Prompts (MOSTLY COMPLETE)
- `daily_plan` node (nodes.py:264-285) queries last 20 `trade_quality_scores`, computes avg, injects into prompt
- **Gap:** `create_trade_evaluator()` never called in `on_trade_close` — scores only exist if scored externally

## What Needs Work

### Wire 2: Regime Affinity → Position Sizing (FLAG-GATED, INCOMPLETE)
- **Location:** `risk_sizing()` node (nodes.py:619-634)
- **Current:** Code reads affinity when `FEEDBACK_REGIME_AFFINITY_SIZING=true`, but multiplication into `compute_alpha_signals()` not applied
- **Fix:** Multiply `signal_value *= max(regime_affinity[sid], 0.1)` before Kelly sizing

### Wire 4: SkillTracker → Conviction Scaling (FLAG-GATED, NOT WIRED)
- **Location:** No consumer code exists yet
- **Current:** `SkillTracker.get_confidence_adjustment(agent_id)` returns [0.5, 1.5] multiplier based on win rate + IC
- **Fix:** Call in signal synthesis or agent output processing, multiply conviction by adjustment
- **Additional gap:** `SkillTracker.record_ic()` is never called — skill adjustment based only on win rate, not IC

### Wire 5b: IC-Driven Signal Weights (FLAG-GATED, NOT INTEGRATED)
- **Location:** `synthesis.py:_get_weights()` (lines 138-170)
- **Current:** Always returns static `_WEIGHT_PROFILES`. `FEEDBACK_IC_DRIVEN_WEIGHTS` flag exists but no code path calls `ic_tracker.get_weights_for_regime()`
- **Fix:** Add IC weight lookup in `_get_weights()` when flag enabled, fallback to static profiles

### Wire 6 Gap: TradeEvaluator Not Called on Close
- **Location:** `trade_hooks.py:on_trade_close()`
- **Current:** All other learning modules called, but TradeEvaluator scoring missing
- **Fix:** Add `create_trade_evaluator()` call after outcome recording (dependency: `openevals`)

### OutcomeTracker Auto-Trigger
- `apply_learning()` is not called automatically — requires manual/script invocation
- Should trigger on daily market close or per-cycle

## Feature Flags

**Location:** `src/quantstack/config/feedback_flags.py`

| Flag | Wire | Default | Purpose |
|------|------|---------|---------|
| `FEEDBACK_REGIME_AFFINITY_SIZING` | 2 | false | Scale position by regime affinity |
| `FEEDBACK_SKILL_CONFIDENCE` | 4 | false | Adjust conviction by agent skill |
| `FEEDBACK_IC_DRIVEN_WEIGHTS` | 5b | false | Replace static weights with IC-derived |

## Database Tables (All Created)

1. `strategy_outcomes` — trade entry/exit per strategy+regime
2. `strategy_breaker_states` — per-strategy circuit breaker state (PK: strategy_id)
3. `ic_attribution_data` — per-collector signal→return observations
4. `agent_skills` — per-agent prediction accuracy and IC
5. `agent_ic_observations` — IC time series per agent
6. `trade_quality_scores` — LLM-as-judge evaluations (6 dimensions)
7. `strategies.regime_affinity` — JSON column on strategies table

## Test Coverage

**File:** `tests/unit/test_learning_wiring.py` (781 lines)
- TestWire1-6 classes cover all wiring points
- TestFeatureFlags validates flag toggling
- TestP05* classes cover adaptive synthesis extensions
- Additional tests in `test_circuit_breaker.py`, `test_skill_tracker_ic.py`

## Revised Scope for P00 Plan

The original spec estimated 2-3 days for 6 wires. With 4 already operational, the actual work is:

1. **Complete Wire 2** — Apply affinity multiplication in `compute_alpha_signals()` (~2 hours)
2. **Wire Wire 4 consumer** — Add conviction scaling call in signal synthesis (~4 hours)
3. **Integrate Wire 5b** — Add IC weight lookup in `_get_weights()` (~4 hours)
4. **Wire TradeEvaluator scoring** — Add to `on_trade_close` hook (~2 hours)
5. **Auto-trigger apply_learning()** — Hook into daily close cycle (~2 hours)
6. **Activation testing** — Enable flags one at a time, verify Sharpe/drawdown (~4 hours)

**Revised estimate:** 1-1.5 days (down from 2-3 days)

## Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| IC data insufficient for regime-conditioned weights | Medium | Require 60+ obs per regime, fallback to static |
| Activating all flags simultaneously causes instability | High | Enable one flag at a time with monitoring |
| SkillTracker.record_ic() never called | Medium | Wire IC recording from collector-level evaluation |
| OutcomeTracker needs 5+ outcomes before updating | Low | Expected behavior, no fix needed |
