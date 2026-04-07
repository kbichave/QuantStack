# 06 — Feedback Loops: Make The System Learn From Its Mistakes

**Priority:** P3 — What Makes a System into a Company
**Timeline:** Week 5-7
**Gate:** Losses drive research priorities. IC decay adjusts weights. Live Sharpe triggers demotion.

---

## Why This Section Matters

This is the single biggest gap all three audits identified. QuantStack has **five fully-implemented learning modules with zero consumers**. When a trade loses 2%, the loss is recorded in 3 different tables. No downstream system reads those tables. Position sizing the next morning is identical — as if the loss never happened.

A system that doesn't learn from losses will repeat them. A system that learns from losses compounds intelligence alongside capital. This is the difference between a trading system and a trading company.

---

## The Ghost Component Registry

Five modules exist, are implemented, have write paths — but their read paths are either stubbed or never called:

| Component | File | Lines | Write Called? | Read Called? | Verdict |
|-----------|------|-------|--------------|-------------|---------|
| `OutcomeTracker` | `learning/outcome_tracker.py` | ~200 | Yes — on fill hook | **NO** — `get_regime_strategies()` returns stub | **SINK** (data goes in, never comes out) |
| `SkillTracker` | `learning/skill_tracker.py` | ~250 | **NEVER** | **NEVER** | **GHOST** (fully implemented, zero callers) |
| `ICAttribution` | `learning/ic_attribution.py` | ~200 | **NEVER** | **NEVER** | **GHOST** |
| `ExpectancyEngine` | `learning/expectancy_engine.py` | ~200 | **NEVER** | **NEVER** | **ORPHAN** (sizing uses `core/kelly_sizing.py` instead) |
| `StrategyBreaker` | `execution/strategy_breaker.py` | ~200 | **NEVER** | **NEVER** | **GHOST** |
| `TradeEvaluator` | `performance/trade_evaluator.py` | ~150 | Yes — from reflection node | **NEVER** — scores written to DB, nobody reads | **SINK** |

---

## What Actually Happens When a Trade Loses -2%

```
1. Trade closes (execution_monitor detects price crossed stop_price)
   → on_trade_close() hook fires
   → Writes to reflection_journal .................. LOGGING ONLY
   → Inserts into research_queue if loss > 1% ....... QUEUED, not prioritized by failure mode
   → Credit assignment computes "worst step" ........ NEVER READ by any policy

2. Fill hook fires → OutcomeTracker.apply_learning()
   → regime_affinity updated: 0.50 → 0.49 ......... WRITTEN to DB
   → get_regime_strategies() tool would read it ..... STUBBED
   → SkillTracker.update_agent_skill() .............. NOT CALLED
   → ICAttribution.record() ......................... NOT CALLED

3. Reflection node runs (30 min later)
   → TradeEvaluator scores on 6 dimensions ......... WRITTEN to trade_quality_scores table
   → Nobody queries trade_quality_scores ............ DEAD DATA

4. Next morning, daily_planner runs
   → Does NOT check regime_affinity ................. Would get stub anyway
   → Does NOT check StrategyBreaker ................. Breaker state never updated
   → Position sized via Kelly ........................ AS IF LOSS NEVER HAPPENED

TIME FROM LOSS TO SYSTEM ADAPTATION: NEVER
```

---

## The Five Broken Loops

### Loop 1: Trade Outcome → Research Priority (INCOMPLETE)

**Status:** One-directional path exists (`trade_hooks.py:118-144`). Loss > 1% → `research_queue` with `bug_fix` task type. This is real and credit-worthy.

**What's Missing:**

| Gap | Impact |
|-----|--------|
| No failure mode taxonomy | All losses are "bug_fix" — no distinction between regime mismatch, factor crowding, data staleness |
| No aggregation over time | Can't say "3 regime mismatches this week — maybe our regime detection needs work" |
| No graduated response | Binary: loss > 1% → research queue. Smaller systematic losses ignored |
| No auto-weight adjustment | Research task queued, but signal weights unchanged |

**Fix (3 days):**

| Step | Action |
|------|--------|
| 1 | Add failure mode taxonomy enum: `REGIME_MISMATCH`, `FACTOR_CROWDING`, `DATA_STALE`, `TIMING_ERROR`, `THESIS_WRONG`, `BLACK_SWAN` |
| 2 | Classify each loss in `trade_hooks.py` using regime-at-entry vs. regime-at-exit, factor overlap, data freshness |
| 3 | Aggregate failure modes daily in supervisor (16:30 ET) |
| 4 | Research queue priority = `f(cumulative_loss_30d * recency_weight)` |
| 5 | Top failure mode by cumulative P&L impact → highest priority research task |

### Loop 2: Realized Execution Cost → Cost Model Calibration (BROKEN)

**Status:** Pre-trade TCA forecasts cost. Post-trade TCA stores realized cost. No connection between them.

```
Current:  forecast 15 bps → trade executes with 35 bps → stored → next trade still forecasts 15 bps
Required: forecast 15 bps → trade executes with 35 bps → EWMA update → next trade forecasts 17 bps
```

**Fix (covered in Section 03, item 3.5):** EWMA recalibration after every fill.

### Loop 3: Signal IC Degradation → Weight Adjustment (BROKEN)

**Status:** IC is not tracked (QS-S1). Even if it were, no mechanism adjusts synthesis weights based on IC.

```
Current:  technical IC drops 0.05 → 0.01 → nobody notices → still gets 25% weight → conviction inflated
Required: IC_tracker detects drop → synthesis halves technical weight → conviction drops → smaller positions
```

**Fix (2 days):**

| Step | Action |
|------|--------|
| 1 | Daily IC per collector (from Section 02, item 2.1) |
| 2 | If rolling 21-day IC < 0.02: halve collector weight in synthesis |
| 3 | Publish `SIGNAL_DEGRADATION` event to EventBus |
| 4 | Research graph picks up investigation task |

### Loop 4: Strategy Performance → Strategy Demotion (PARTIALLY BROKEN)

**Status:** `strategy_promoter` handles forward_testing → live promotion. `strategy_breaker` handles 5% drawdown or 3 consecutive losses → circuit break. But no slow-bleed detection.

```
Current:  Strategy live 30 days, Sharpe 0.2 (backtest was 1.5) → keeps trading until 5% drawdown
Required: Live Sharpe < 50% of backtest for 21 days → auto-demote to forward_testing, reduce size 75%
```

**Fix (1 day):**

| Step | Action |
|------|--------|
| 1 | Add live-vs-backtest Sharpe comparison to `strategy_promoter` |
| 2 | Gate: live Sharpe < 50% of backtest for 21+ days → auto-demote |
| 3 | Queue research task: "investigate strategy degradation for {strategy_id}" |

### Loop 5: Agent Decision Quality → Prompt Improvement (BROKEN)

**Status:** `trade_debater` recommends ENTER. Position loses 3%. `trade_reflector` logs "thesis was wrong." Debater's next call uses the exact same prompt. No learning.

```
Current:  Debater recommends → loss → reflector logs → debater unchanged → same mistakes
Required: Track recommendation → outcome per agent → identify weak spots → improve prompt
```

**Fix (3 days for tracking, meta-optimization deferred to Section 09):**

| Step | Action |
|------|--------|
| 1 | Track per-agent decision quality: recommendation → outcome |
| 2 | Compute per-agent win rate over rolling 30 trades |
| 3 | Alert when win rate drops below 40% (baseline should be >50%) |
| 4 | Manual prompt improvement until meta-optimization layer built |

---

## The 6 Readpoints That Must Be Wired (2-3 Days Total)

These are the specific code changes that connect the ghost modules to the live system:

| # | Missing Wire | From → To | Fix |
|---|-------------|-----------|-----|
| 1 | Implement `get_regime_strategies()` | `meta_tools.py` → reads `strategies.regime_affinity` | Replace stub with DB query returning affinity-weighted allocations |
| 2 | `risk_sizing` checks regime_affinity | `trading/nodes.py` → `outcome_tracker.get_affinity()` | Multiply Kelly fraction by affinity before sizing |
| 3 | `execute_entries` checks strategy_breaker | `trading/nodes.py` → `strategy_breaker.get_scale_factor()` | Reduce/block orders for SCALED/TRIPPED strategies |
| 4 | Trade hooks populate SkillTracker | `trade_hooks.py` → `skill_tracker.update_agent_skill()` | Call on every trade close with agent_name + outcome |
| 5 | Signal engine records IC attribution | `signal_engine/engine.py` → `ic_attribution.record()` | After synthesis, record each collector's contribution vs. forward return |
| 6 | `daily_planner` queries quality scores | `trading/nodes.py` → `SELECT * FROM trade_quality_scores` | Surface patterns: "exit_evaluator gives HOLD on positions that then lose >3%" |

---

## Summary: Feedback Loops Delivery

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 6.1 | Wire 6 readpoints (ghost → live) | 2-3 days | Losses change system behavior |
| 6.2 | Failure mode taxonomy | 2 days | Research targets root causes, not symptoms |
| 6.3 | IC degradation → weight adjustment | 2 days | Stale signals automatically downweighted |
| 6.4 | Live vs. backtest Sharpe demotion | 1 day | Slow-bleed strategies caught |
| 6.5 | Agent decision quality tracking | 3 days | Identify weakest agents |
| 6.6 | Loss aggregation in supervisor | 1 day | Pattern detection across losses |

**Total estimated effort: 11-14 engineering days.**
**After this section:** the system learns. Every loss makes the next decision better. This is what separates a demo from a fund.
