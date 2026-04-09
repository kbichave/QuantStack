# P00 Spec: Wire Learning Modules

**Date:** 2026-04-07
**Source:** Phase spec + codebase research + stakeholder interview

---

## Objective

Close the 5 ghost learning module feedback loops so the system learns from every trade. 4 of 6 wires are already operational (Wires 1, 3a, 3b, 6 consumption). This phase completes the remaining integration work and activates the flag-gated loops.

## Scope

### Already Working (Verify Only)
- **Wire 1:** OutcomeTracker → daily_plan (affinity context in LLM prompt) — `nodes.py:287-314`
- **Wire 3a:** StrategyBreaker → execute_entries (TRIPPED=reject, SCALED=0.5x) — `nodes.py:1163-1195`
- **Wire 3b:** StrategyBreaker ← on_trade_close (record_trade called) — `trade_hooks.py:226-236`
- **Wire 5a:** ICAttribution ← on_trade_close (per-collector recording) — `trade_hooks.py:238-260`
- **Wire 6 consumer:** TradeEvaluator → daily_plan (quality scores in LLM prompt) — `nodes.py:264-285`

### New Work

#### 1. Complete Wire 2: Regime Affinity → Position Sizing
- **Location:** `risk_sizing()` in `nodes.py:619-634`
- **Change:** Apply `signal_value *= max(regime_affinity[sid], 0.1)` in `compute_alpha_signals()`
- **Activation:** Default flag to `true` (was `false`)

#### 2. Wire Wire 4: SkillTracker → Conviction Scaling (Both Layers)
- **Layer 1 — Agent executor:** In agent output processing, multiply conviction by `get_confidence_adjustment(agent_id)`. Range [0.5, 1.5].
- **Layer 2 — Signal synthesis:** In `synthesis.py`, apply per-agent confidence as additional weight factor during signal aggregation.
- **Activation:** Default flag to `true`

#### 3. Integrate Wire 5b: IC-Driven Signal Weights
- **Location:** `_get_weights()` in `synthesis.py:138-170`
- **Change:** When `FEEDBACK_IC_DRIVEN_WEIGHTS=true`, call `ic_tracker.get_weights_for_regime(regime)`. If result is None (insufficient data), fall back to static `_WEIGHT_PROFILES`.
- **Activation:** Default flag to `true`

#### 4. Wire TradeEvaluator Scoring in on_trade_close
- **Location:** `trade_hooks.py:on_trade_close()`
- **Change:** After outcome recording, call trade evaluator. Try `openevals` LLM-as-judge first; fall back to heuristic scorer.
- **Heuristic fallback:** `score_trade_heuristic()` — derives 6 dimension scores from realized P&L, hold duration, slippage estimate, and position size vs target.

#### 5. Auto-Trigger OutcomeTracker.apply_learning()
- **Location:** `trade_hooks.py:on_trade_close()`
- **Change:** Call `outcome_tracker.apply_learning(strategy_id)` after `record_exit()`. Updates regime_affinity in real-time.

#### 6. Default Feature Flags to True
- **Location:** `config/feedback_flags.py`
- **Change:** Flip defaults for `FEEDBACK_REGIME_AFFINITY_SIZING`, `FEEDBACK_SKILL_CONFIDENCE`, `FEEDBACK_IC_DRIVEN_WEIGHTS` from `false` to `true`.
- **Rationale:** System is paper-only. Risk bounded. Flags remain as kill switches.

## Files Modified

| File | Changes |
|------|---------|
| `src/quantstack/graphs/trading/nodes.py` | Wire 2: apply affinity multiplication in risk_sizing |
| `src/quantstack/graphs/agent_executor.py` | Wire 4 Layer 1: conviction scaling at agent output |
| `src/quantstack/signal_engine/synthesis.py` | Wire 4 Layer 2: agent confidence in aggregation; Wire 5b: IC weight lookup |
| `src/quantstack/hooks/trade_hooks.py` | Wire 6 producer: add evaluator call; Wire 5 auto-trigger: add apply_learning() |
| `src/quantstack/performance/trade_evaluator.py` | Add `score_trade_heuristic()` fallback |
| `src/quantstack/config/feedback_flags.py` | Flip 3 flag defaults to true |

## Files Created

| File | Purpose |
|------|---------|
| None | All modules and tables already exist |

## Tests

| Test | Verifies |
|------|----------|
| `test_wire2_affinity_multiplied` | Position size scaled by regime affinity when flag on |
| `test_wire4_agent_executor_confidence` | Agent conviction adjusted by SkillTracker |
| `test_wire4_synthesis_confidence` | Synthesis applies per-agent confidence factor |
| `test_wire5b_ic_weights_integrated` | IC weights replace static profiles when data sufficient |
| `test_wire5b_ic_weights_fallback` | Static profiles used when IC data insufficient |
| `test_trade_evaluator_heuristic` | Heuristic scorer produces valid TradeQualityScore |
| `test_trade_evaluator_called_on_close` | on_trade_close invokes evaluator |
| `test_apply_learning_auto_triggered` | apply_learning called after record_exit |
| `test_flags_default_true` | All 3 feedback flags default to true |

## Acceptance Criteria

1. After trade close, `strategies.regime_affinity` updates within the same session (not manual)
2. Position sizes differ from base Kelly when regime affinity < 1.0
3. Agent conviction scores vary based on SkillTracker history (not always 1.0)
4. Signal weights differ from static `_WEIGHT_PROFILES` after 60+ IC observations per regime
5. Every trade close produces a `trade_quality_scores` row (LLM or heuristic)
6. All 3 feedback flags default to `true`

## Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| IC data insufficient for all regimes | Medium | Fallback to static profiles per-regime; don't require all regimes |
| Double conviction dampening too aggressive | Medium | Floor at 0.5 on each layer → minimum 0.25x total (still allows trading) |
| Heuristic evaluator too simplistic | Low | Acceptable for data accumulation; LLM evaluator upgrades scores once available |
| apply_learning on every close adds latency | Low | Non-blocking try/except wrapper, ~50ms DB write |

## Non-Goals

- Changing StrategyBreaker thresholds (already tuned)
- Adding new learning modules
- P05 adaptive synthesis features (separate phase)
- Changing the signal collector set
