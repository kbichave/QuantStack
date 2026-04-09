# P05 Self-Interview: Adaptive Signal Synthesis

## Q1: IC Weight Batch Job — DB load and caching strategy

**Q:** The spec says IC weights should update weekly from a rolling 63-day window, but currently a fresh `ICAttributionTracker` is instantiated on every synthesis call and loads all observations. What is the expected call volume for `synthesize()`, and have you measured the current DB load? Weekly batch job destination — dedicated DB table or materialized view?

**A:** Synthesis runs once per symbol per trading cycle. With ~50 symbols in universe, that's ~50 calls per cycle (roughly every 15 minutes during market hours = ~200 calls/day). The DB load from instantiating ICAttributionTracker on each call is wasteful but not catastrophic at this scale. Still, it should be fixed.

Plan: Dedicated `precomputed_ic_weights` table with columns `(regime, collector, weight, computed_at)`. Weekly batch job populates this. Synthesis reads from this table via a single query — no ICAttributionTracker instantiation. TTL: if `computed_at` > 7 days old, fall back to static. No materialized view — explicit table is simpler and the scheduler already runs batch jobs overnight.

## Q2: Cold-start for new collectors / rare regimes

**Q:** Fallback triggers with <60 days of data. What about new collectors or rare regimes where 60 days may never accumulate?

**A:** Per-collector-per-regime threshold is correct. A new collector in a rare regime (e.g., `trending_down_high_vol` may only appear 10-15 days per quarter) should use static weights for that regime indefinitely — which is fine. The threshold should be per-collector-per-regime. If a specific regime-collector pair has <60 days of data, that pair uses static weight. Other pairs with sufficient data use IC weights. This is already what `get_weights_for_regime()` does — it returns None for insufficient data, and synthesis falls back to static.

No special bootstrapping needed. Static weights are a reasonable default. The system improves gradually as data accumulates.

## Q3: Position sizing during transitions — enforcement point

**Q:** Where should position size reduction during transitions be enforced? Via SymbolBrief or risk gate?

**A:** Via `SymbolBrief`. Add `transition_zone: bool` field. The consuming node in `graphs/trading/nodes.py` already multiplies signal_value by affinity and skill adjustments — it should also apply a 0.5× scalar when `transition_zone=True`. This keeps the contract clean: the signal engine communicates the condition, the execution path applies the sizing.

The risk gate should NOT independently query regime state for transitions. The risk gate's job is hard limits (max position size, exposure caps). Transition dampening is a signal quality judgment, not a risk limit — it belongs in the signal path.

## Q4: Conviction calibration — join key and lineage

**Q:** What is the join key for mapping signals to outcomes? Does factor-level lineage exist?

**A:** The join key is `(symbol, signal_date, strategy_id='synthesis_v1')` in the `signals` table. The `metadata` JSONB column already stores `{votes: {collector: score}, weights: {collector: weight}}`. However, it does NOT currently store conviction factor values (ADX, stability, etc.).

**Prerequisite:** Add conviction factors to the `metadata` JSONB: `{votes: {...}, weights: {...}, conviction_factors: {adx: 1.15, stability: 0.92, ...}}`. This is a small change in synthesis.py where the INSERT happens (line ~325). Then the calibration job can join `signals.metadata.conviction_factors` with `closed_trades.realized_pnl_pct` to regress factor values against outcomes.

Realized returns are in `closed_trades` table (columns: symbol, entry_date, exit_date, realized_pnl_pct).

## Q5: A/B evaluation metric and promotion

**Q:** What metric evaluates ensemble methods? Who promotes the winner?

**A:** Primary metric: IC (signal score vs 5-day forward return) per method. Secondary: hit rate (% of signals with correct direction). Statistical test: paired t-test on daily IC differences, promote at p < 0.05.

Promotion is automated: the weekly comparison job checks if any non-default method has p < 0.05 improvement over weighted average. If yes, it updates an `ensemble_config` table row. Synthesis reads this on next cycle. If no clear winner after 60 days, keep weighted average (the prior).

This feeds into P10 meta-learning as a concrete example of "system improves itself via data."

## Q6: Order of operations for weight adjustments

**Q:** What is the order of IC weights → IC gate → correlation penalty? Can weights sum to near-zero?

**A:** Current order (in synthesis.py lines 530-579):
1. Get static weights from `_get_weights(regime)` 
2. Replace with IC-driven weights if available (`ic_driven_weights_enabled`)
3. IC gate: zero out collectors below 0.02 IC threshold, renormalize
4. Correlation penalty: multiply by penalty factor, renormalize

Each step renormalizes to sum=1.0, so the final weight vector always sums to 1.0. The invariant is maintained. The only degenerate case is if ALL collectors are gated (IC < 0.02 for all) — then the gate returns empty dict and synthesis falls back to static weights.

The `ic_weights.py` `check_weight_floor()` function provides an additional safety net: if total effective weight < 0.1, it falls back to static weights entirely. This is currently in the codebase but not wired into synthesis. It should be.

## Q7: Sub-regime granularity for IC weights

**Q:** Does `get_weights_for_regime()` key on sub-regime strings or only base regimes?

**A:** Currently only base regimes (4: trending_up, trending_down, ranging, unknown). The regime parameter passed is `trend_regime`, not `sub_regime`.

This is a gap. Two options:
- **Option A**: Compute IC weights at sub-regime level. Risk: insufficient data for rare sub-regimes.
- **Option B**: Keep IC weights at base regime level, use sub-regime profiles for the static fallback. IC weights override the general shape; sub-regime profiles provide the detail when IC data is insufficient.

**Decision**: Option B. Sub-regimes are rare enough that IC data at that granularity will be sparse for years. IC weights at the base regime level capture the main signal (which collectors work in which regime). Sub-regime profiles provide differentiation within regime when IC data isn't available for the specific sub-regime.

## Q8: P10 interface contract — what to persist

**Q:** What data does P05 persist for P10 meta-learning to observe?

**A:** P10 needs to see:
1. **Weight decisions**: Which weights were used for each synthesis (IC-driven vs static fallback), for each regime → already in `signals.metadata.weights`
2. **IC trajectories**: Per-collector rolling IC over time → already in `signal_ic` table
3. **Conviction factors**: Factor values that produced each conviction → needs to be added to `signals.metadata` (see Q4)
4. **Ensemble method assignment and outcome**: Which method was used per symbol → needs new tracking (A/B results table)
5. **Transition zone decisions**: When transitions were detected and sizing was reduced → needs new column or log

The `precomputed_ic_weights` table (with history) + `signals.metadata` enrichment + `ensemble_ab_results` table give P10 everything it needs. No additional schema required beyond what this phase creates.
