# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06T00:00:00Z

---

## Phase 7 Implementation Plan — Critical Review

### 1. Completeness

**All 12 spec items are covered.** The plan maps spec items 7.1-7.12 to Sections 1-13 (with Section 1 as a pre-audit not in the original spec). The three-stream parallel structure (A: Core Wiring, B: Signal Intelligence, C: Autonomous Learning) is a reasonable decomposition.

**One notable gap:** The spec calls for a 5-week timeline (Weeks 6-10, 23-26 days). The plan schedules 5 weeks but the effort estimates sum to roughly 20-22 days of individual work. With parallelism across three streams, calendar time may be shorter, but the plan does not state who is doing what. If this is single-engineer work, parallelism is fictional. Clarify whether the three streams are sequential for one engineer or parallel across multiple.

**Understated item:** Wire 5 (IC Attribution in signal engine) has a critical gap. The plan says `forward_return=None` at synthesis time, backfilled on trade close. But if the system trades a symbol infrequently (e.g., once every 30 days), the backfill lag means ICAttributionTracker accumulates records without outcomes for extended periods. The plan does not address what happens to records that never get backfilled (e.g., a signal was computed for AAPL but no trade was taken). Per-collector IC computed from only traded symbols introduces survivorship bias -- you only measure IC where you actually entered, not across the full signal universe.

### 2. Technical Soundness

**OutcomeTracker fix (Section 1) is directionally correct but undertested.** The plan changes the step multiplier from 0.05 to 0.15 and tanh divisor from 5.0 to 2.0. A -2% loss now steps -0.11 instead of -0.019. But the flip side is concerning: a +2% win also steps +0.11 upward. With daily noise in P&L, affinity will oscillate rapidly. The plan mentions "recency-weighted exponential decay" but does not specify the decay halflife or the formula. This needs to be concrete -- different halflives produce radically different behavior. A 10-trade halflife versus a 50-trade halflife is the difference between a jittery allocator and a sluggish one.

**IC degradation thresholds (Section 5) have a cliff problem.** The tiered IC factor (1.0 / 0.8 / 0.5 / 0.25 / 0.0) creates discrete jumps at boundaries. A collector oscillating around IC=0.02 will flip between 0.5 and 0.8 weight factor on successive days. The plan says "update synthesis weights weekly to avoid excessive churn," but the IC factors are computed daily. If a collector crosses the 0.02 boundary on Monday and crosses back on Tuesday, the weekly snapshot timing determines which side it lands on -- effectively random. Use a continuous function (e.g., sigmoid or linear interpolation) instead of tiers, or add hysteresis (only change tier after N consecutive days on the new side).

**Signal correlation matrix (Section 6) has a sample size problem.** 63 trading days of per-symbol signal values across 22 collectors produces a 22x22 correlation matrix. Spearman correlation between two collectors using 63 observations has a standard error of roughly 0.13. A measured correlation of 0.7 has a 95% CI of approximately [0.55, 0.81]. The plan applies a hard 0.5x penalty at the 0.7 threshold, but a collector at measured correlation 0.68 gets no penalty while one at 0.72 gets a 50% haircut. This is not statistically distinguishable. Either increase the lookback window (126+ days) or use a softer penalty that scales with the measured correlation rather than a threshold.

**Conviction calibration (Section 8) is well-designed** but has a subtle issue with factor interaction. Six multiplicative factors can compound aggressively: worst case `0.85 * 0.80 * 0.85 * 0.75 = 0.434`, which would cut a base conviction of 0.6 to 0.26. That may be correct behavior (everything is misaligned), but the plan should document the expected range of the product of all factors and verify it produces sensible results at the extremes. The [0.05, 0.95] clip is necessary but could mask pathological combinations.

**Regime transition detection (Section 13) says "P(leaving current state) = 1 - P(staying)."** This is only correct for the steady-state transition matrix, not the filtered probability at time t. After fitting an HMM, the transition probability at a specific time step depends on the observation sequence, not just the static transition matrix. The plan should use the filtered state probabilities from the forward algorithm (which hmmlearn provides via `predict_proba()`), not the model's `transmat_` parameter. Using the static matrix means every observation gets the same transition probability regardless of recent data, which defeats the purpose.

### 3. Architecture

**StrategyBreaker persistence on JSON files is a concern.** The plan wires StrategyBreaker into two critical trading graph nodes (`risk_sizing` and `execute_entries`). But StrategyBreaker persists to `~/.quantstack/strategy_breakers.json`. When running in Docker containers (as the architecture describes), this file lives on ephemeral container filesystem unless a volume is mounted. A container restart resets all breaker states to ACTIVE, meaning a TRIPPED strategy resumes trading. This needs to either (a) migrate to PostgreSQL persistence (consistent with the "DB writes use `db_conn()` context managers" hard rule) or (b) explicitly document the volume mount requirement. Given that this is a safety-critical component, PostgreSQL is strongly preferred.

**ICAttributionTracker has the same problem** -- it persists to `~/.quantstack/ic_attribution.json`. Same risk of data loss on restart.

**The plan adds 4 new DB tables and 2 schema changes but does not provide a migration strategy.** No mention of Alembic, manual migration scripts, or how to handle existing data. The `strategy_outcomes` table already has data -- adding `failure_mode TEXT` needs a default for existing rows.

**The multiplying factors chain in `risk_sizing` is getting long:** `kelly_size * breaker_factor * transition_factor` (from Section 13). Combined with Wire 2's integration of breaker_factor and Section 10's 0.25x multiplier from Sharpe demotion, the final position size is now the product of 5-6 multiplicative adjustments. A strategy that is SCALED (0.5) in a regime transition (0.5) with Sharpe demotion (0.25) gets 0.5 * 0.5 * 0.25 = 0.0625 of normal sizing. The plan should document the interaction and confirm the resulting position sizes are non-trivially tradeable.

### 4. Risk Assessment

**Highest risk: Wire 2 and Wire 3 (StrategyBreaker in live trading path).** A bug in `get_scale_factor()` returning 0.0 for all strategies halts all trading with no error signal. A bug returning values > 1.0 amplifies positions. Add a defensive check: if `get_scale_factor()` raises or returns a value outside [0.0, 1.0], default to 1.0 (fail-open for sizing, since the risk gate is downstream) and log an error.

**Second highest risk: Section 13 (Regime Transition Detection).** Modifying the regime collector output format changes the signal engine's data contract. If the transition probability field is missing (e.g., HMM fails to fit), the `risk_sizing` node must not crash. Address degraded mode.

**Medium risk: Section 3 (Failure Mode Taxonomy) with LLM classification fallback.** Adding an LLM call to the trade close hook blocks the hook. Make the LLM classification asynchronous.

**Low-probability high-impact: Section 5 IC weight adjustment reducing all collectors to zero.** Add a floor: if total effective weight after IC adjustments < some minimum, fall back to equal weights and publish an alert.

### 5. Dependencies and Ordering

**One hidden dependency:** Wire 5 must be live ~21 days before Section 5 produces meaningful weight adjustments. The plan gives only ~10 trading days. Add bootstrap period logic.

**Another hidden dependency:** Section 12 (Model Versioning) in Week 1-2, before Section 11 (Concept Drift) in Week 4 which triggers retraining. Model registry will sit unused for 2-3 weeks.

### 6. Specific Suggestions

1. **Add bootstrap/cold-start logic for every feedback loop.** Define "insufficient data" behavior for each item.
2. **Migrate StrategyBreaker and ICAttributionTracker from JSON to PostgreSQL.**
3. **Replace tiered IC factor with continuous function** (e.g., sigmoid centered at IC=0.02).
4. **Make LLM failure classification asynchronous.** Never block trade hook on LLM call.
5. **Add "total sizing factor" sanity check** -- skip trade if result below minimum tradeable size.
6. **Use filtered probabilities for regime transitions**, not static transition matrix.
7. **Add kill-switch config flags** for each feedback loop (IC weights, correlation penalties, conviction calibration).
8. **Change champion/challenger shadow period to 60 days** to match research recommendations.
9. **Flag recursive quality signal risk** in Wire 6 (LLM evaluating LLM decisions).
10. **Add rollback paths** for each section.

### Summary Assessment

The plan is thorough, well-researched, and demonstrates genuine understanding of the codebase. Main concerns:
- Boundary effects in thresholds that will cause oscillation
- JSON file persistence for safety-critical state
- Missing cold-start/degraded-mode behavior
- Multiplicative sizing factor chain compounding to near-zero
- HMM transition probability implementation using static matrix
- No rollback paths
