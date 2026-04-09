# P00: Wire Learning Modules

**Objective:** Connect the 5 ghost learning modules to their consumers, closing the feedback loops that make the system learn from every trade.

**Scope:** learning/, execution/, trading/nodes.py, hooks/trade_hooks.py

**Depends on:** None

**Enables:** P05 (Adaptive Synthesis), P10 (Meta-Learning)

**Effort estimate:** 2-3 days

---

## What Changes

The CTO audit (DO-1) discovered 5 fully-implemented learning modules with zero callers. The system records losses but behavior never changes. This phase wires the 6 missing connections.

### Wire 1: OutcomeTracker → Strategy Selection
- **From:** `learning/outcome_tracker.py` (already writes `regime_affinity`)
- **To:** `meta_tools.py:get_regime_strategies()` (currently stubbed)
- **Fix:** Replace stub with DB query that returns affinity-weighted allocations
- **Impact:** Daily planner now knows which strategies work in which regimes

### Wire 2: OutcomeTracker → Position Sizing
- **From:** `learning/outcome_tracker.py:get_affinity()`
- **To:** `trading/nodes.py` risk_sizing node
- **Fix:** Multiply Kelly fraction by regime affinity before sizing
- **Impact:** Position sizes shrink for strategies in hostile regimes

### Wire 3: StrategyBreaker → Order Execution
- **From:** `execution/strategy_breaker.py:get_scale_factor()` (ACTIVE→SCALED→TRIPPED)
- **To:** `trading/nodes.py` execute_entries node
- **Fix:** Check scale factor pre-execution. SCALED=0.5x, TRIPPED=0.0x
- **Also:** Wire `strategy_breaker.record_trade()` in trade_hooks.py
- **Impact:** Losing strategies auto-scale down, then halt

### Wire 4: SkillTracker → Agent Confidence
- **From:** `learning/skill_tracker.py:get_confidence_adjustment()`
- **To:** `graphs/agent_executor.py` (post-agent output processing)
- **Fix:** Call `update_agent_skill()` on every trade close with agent_name + outcome
- **Also:** Use `get_confidence_adjustment()` to scale agent conviction outputs
- **Impact:** Agents that make bad calls get their conviction discounted

### Wire 5: ICAttribution → Signal Weights
- **From:** `learning/ic_attribution.py:get_weights()`
- **To:** `signal_engine/synthesis.py` (currently uses static `_WEIGHT_PROFILES`)
- **Fix:** Call `ic_attribution.record()` after each signal → return comparison. Use `get_weights()` in synthesis if available, fall back to static profiles.
- **Impact:** Signal weights adapt based on realized predictive power

### Wire 6: TradeEvaluator → Agent Prompts
- **From:** `performance/trade_evaluator.py` (already scores on 6 dimensions, written to DB)
- **To:** `trading/nodes.py` daily_planner context loading
- **Fix:** Query `trade_quality_scores` for last 20 trades, inject summary into daily_planner prompt
- **Impact:** Trading decisions informed by what worked and what didn't

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/quantstack/tools/langchain/meta_tools.py` | Implement `get_regime_strategies()` — DB query |
| `src/quantstack/graphs/trading/nodes.py` | risk_sizing: multiply by affinity. execute_entries: check breaker |
| `src/quantstack/hooks/trade_hooks.py` | Add `strategy_breaker.record_trade()` + `skill_tracker.update_agent_skill()` |
| `src/quantstack/signal_engine/synthesis.py` | Add ICAttribution weight override path |
| `src/quantstack/graphs/agent_executor.py` | Add confidence adjustment from SkillTracker |
| `src/quantstack/graphs/trading/nodes.py` | daily_planner: inject trade_quality_scores summary |

## Tests

| Test | What It Verifies |
|------|-----------------|
| `test_outcome_tracker_wiring` | `get_regime_strategies()` returns non-stub data after trade close |
| `test_strategy_breaker_blocks` | TRIPPED strategy → order rejected |
| `test_skill_tracker_updates` | Agent skill score changes after win/loss |
| `test_ic_attribution_weights` | Signal weights change after IC recording |
| `test_trade_evaluator_readable` | Daily planner prompt includes quality scores |

## Acceptance Criteria

1. After a losing trade closes, `strategy_breaker.state` transitions from ACTIVE → SCALED or TRIPPED
2. `get_regime_strategies()` returns real data (not stub error)
3. Signal weights in synthesis differ from static `_WEIGHT_PROFILES` after 10+ IC recordings
4. Agent confidence adjustments are applied (visible in LangFuse traces)
5. Daily planner prompt includes last 20 trade quality summaries

## Risk

| Risk | Severity | Mitigation |
|------|----------|-----------|
| IC Attribution weights oscillate wildly | Medium | Use EWMA smoothing (decay=0.95) on weight updates |
| StrategyBreaker false-trips on normal variance | Medium | Require 5+ trades before enabling breaker (min sample) |
| Circular dependency in synthesis | Low | ICAttribution is additive override, falls back to static |

## References

- See `../gaps/tier1-2-critical-gaps.md` — Ghost Learning Modules (Tier 1)
- CTO Audit: DO-1 (Ghost Component Registry)
- CTO Audit: Loops 1-5 (The Five Missing Loops)
